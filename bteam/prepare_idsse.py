"""B팀 학습 데이터 만들기: IDSSE (분데스리가 7경기, 22명 + 공 좌표) -> 장면 단위 npz.

한 장면 = 2 fps 한 프레임, 한 팀 시점. 경기 x 하프 x 팀(Home/Away) 마다 파일 하나:
  ball     (T, 2)      공 위치 (m)
  own      (T, 10, 2)  우리 팀 필드 선수 10명, 포지션(자리) 순서  <- 정답
  opp      (T, 10, 2)  상대 팀 필드 선수 10명, 상대 포지션 순서
  own_mean (10, 2)     우리 포지션 10개의 평균 위치 (= 쿼리 초기값, 비교 기준 1)
  opp_mean (10, 2)     상대 포지션 평균 위치
  phase    (T,)        1 = 우리 공 (공격), 0 = 상대 공 (수비)
  t        (T,)        하프 시작부터 초
좌표: 경기장 중앙 (0, 0), m 단위, 우리 팀이 +x 방향으로 공격하도록 뒤집음 (하프마다).
없는 선수(퇴장, 교체 직후 누락)는 NaN.

포지션 = Bialkowski 2014 자리 배정 (analysis/role_distributions.py, 경기·하프·팀마다 따로 EM).
자리 번호는 평균 위치를 뒤에서 앞으로(x), 같은 줄 안에서는 왼쪽에서 오른쪽(y)으로 정렬해 붙인다
-> 다른 팀이어도 0~3번은 대체로 수비 라인, 마지막 번호들은 공격수.

usage (deep07):
  pip install floodlight
  python bteam/prepare_idsse.py --out ~/data/bteam            # 7경기 전부 (처음엔 약 2.4 GB 다운로드)
  python bteam/prepare_idsse.py --matches J03WMX --out ...    # 한 경기만
  python bteam/prepare_idsse.py --idsse-dir PATH ...          # 이미 받아둔 IDSSE 폴더 재사용 (floodlight DATA_DIR 아래 폴더 이름)
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'analysis'))
import role_distributions as R  # noqa: E402

MATCHES = ['J03WMX', 'J03WN1', 'J03WPY', 'J03WOH', 'J03WQQ', 'J03WOY', 'J03WR9']
FPS_OUT = 2.0
LINE_GAP = 6.0          # m: roles whose mean x differs by less than this are treated as one line


# ------------------------------------------------------------------ helpers (plain numpy, testable)
def canonical_order(mu):
    """role order: deepest line first, inside a line left (-y) to right (+y)"""
    idx = list(np.argsort(mu[:, 0]))
    lines, cur = [], [idx[0]]
    for i in idx[1:]:
        if mu[i, 0] - mu[cur[-1], 0] < LINE_GAP:
            cur.append(i)
        else:
            lines.append(cur); cur = [i]
    lines.append(cur)
    return [i for ln in lines for i in sorted(ln, key=lambda j: mu[j, 1])]


def team_roles(P, restarts=3, fit_stride=2, min_on=8):
    """P: (T, N, 2) outfield positions of one team in its own attack-+x frame (NaN = off pitch).
    returns ordered (T, 10, 2) positions in role order and the role means (10, 2) of this half.
    Every on-pitch player gets a role (no gate) because the data is full-pitch."""
    T = len(P)
    on = np.isfinite(P[..., 0])
    frames = [(t, P[t, on[t]]) for t in range(0, T, fit_stride) if on[t].sum() >= min_on]
    best = R.best_fit(frames, restarts=restarts, max_iter=40)
    mu = best['mu']
    out = np.full((T, R.K, 2), np.nan)
    s_prev = None
    for t in range(T):
        idx = np.where(on[t])[0]
        if len(idx) < 3:
            s_prev = None; continue
        s, ri, ci, _ = R.match_frame(P[t, idx], mu, gate=1e3, s_prev=s_prev)
        s_prev = s
        out[t, ci] = P[t, idx[ri]]
    order = canonical_order(mu)
    out = out[:, order]
    mean_abs = np.nanmean(out, axis=0)           # absolute mean place of each role over the half
    return out, mean_abs


def process_segment(home, away, ball, poss, alive, gk_home, gk_away, fps_in, restarts=3):
    """one half. home/away: (T, N*2) floodlight XY arrays in metres, pitch centre (0,0).
    poss: 1 home / 2 away, alive: 1 / 0. gk_*: column (xID) of the goalkeeper(s).
    returns {'Home': dict, 'Away': dict} of arrays as described in the module docstring."""
    step = max(1, int(round(fps_in / FPS_OUT)))
    sl = slice(0, None, step)
    H = home[sl].reshape(len(home[sl]), -1, 2); A = away[sl].reshape(len(away[sl]), -1, 2)
    b = ball[sl]; ps = np.asarray(poss, float)[sl]; al = np.asarray(alive, float)[sl]
    t = np.arange(len(b)) / (fps_in / step)

    def split(X, gk):
        gkx = np.nanmean(X[:, gk, 0]) if len(gk) else np.nan
        field = [i for i in range(X.shape[1]) if i not in set(gk)]
        F = X[:, field]
        if not np.isfinite(gkx):                     # no keeper info: own half = side of the team centre
            gkx = np.nanmean(F[..., 0])
        fl = 1.0 if gkx < 0 else -1.0                # keeper on the left -> attack +x
        return F, fl

    FH, flH = split(H, gk_home); FA, flA = split(A, gk_away)
    if flH == flA:                                   # both on the same side: trust the home keeper
        flA = -flH
    ordH, meanH = team_roles(FH * flH, restarts)
    ordA, meanA = team_roles(FA * flA, restarts)
    keep = (al > 0) & np.isfinite(b[:, 0]) & np.isin(ps, (1, 2))
    res = {}
    for team, own, own_mean, fl, opp, opp_mean, ofl, code in (
            ('Home', ordH, meanH, flH, ordA, meanA, flA, 1), ('Away', ordA, meanA, flA, ordH, meanH, flH, 2)):
        k = fl * ofl                                  # opponent frame -> our frame
        res[team] = {'ball': (b * fl)[keep].astype(np.float32),
                     'own': own[keep].astype(np.float32),
                     'opp': (opp * k)[keep].astype(np.float32),
                     'own_mean': own_mean.astype(np.float32),
                     'opp_mean': (opp_mean * k).astype(np.float32),
                     'phase': (ps[keep] == code).astype(np.int8),
                     't': t[keep].astype(np.float32)}
    return res


# ------------------------------------------------------------------ IDSSE adapter (floodlight)
def keeper_columns(sheet):
    df = getattr(sheet, 'teamsheet', None)
    if df is None or 'position' not in df.columns:
        return []
    if 'xID' not in df.columns:
        sheet.add_xIDs(); df = sheet.teamsheet
    return [int(r['xID']) for _, r in df.iterrows() if str(r['position']).upper() == 'TW']


def run_match(mid, out_dir, dir_name, restarts):
    from floodlight.io.datasets import IDSSEDataset
    ds = IDSSEDataset(dataset_dir_name=dir_name, match_id=mid)   # NB: first positional arg is the folder name
    _, xy, poss, ballst, sheets, _ = ds.get(match_id=mid, events=False)
    gk = {team: keeper_columns(sheets[team]) for team in ('Home', 'Away')}
    for seg in xy:
        fps_in = xy[seg]['Home'].framerate or 25
        t0 = time.time()
        res = process_segment(xy[seg]['Home'].xy, xy[seg]['Away'].xy, xy[seg]['Ball'].xy,
                              poss[seg].code, ballst[seg].code, gk['Home'], gk['Away'], fps_in, restarts)
        for team, d in res.items():
            path = os.path.join(out_dir, f'{mid}_{seg}_{team}.npz')
            np.savez_compressed(path, match=mid, seg=seg, team=team, **d)
            n_full = int(np.isfinite(d['own'][..., 0]).all(1).sum())
            print(f'{path}: {len(d["ball"])} scenes ({n_full} with all 10 own players), '
                  f'attack {100 * d["phase"].mean():.0f}%  [{time.time() - t0:.0f}s]', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--matches', nargs='+', default=MATCHES)
    ap.add_argument('--idsse-dir', default='idsse_dataset', help='folder name under floodlight DATA_DIR')
    ap.add_argument('--restarts', type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for mid in a.matches:
        run_match(mid, a.out, a.idsse_dir, a.restarts)
