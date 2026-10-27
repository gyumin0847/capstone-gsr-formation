"""Full-pitch -> simulated broadcast test on the IDSSE dataset (7 Bundesliga matches, all 22 players + ball).

Team plan: "IDSSE 풀피치 데이터셋 -> 가로폭 줄여서 non-풀피치로 변환 후 원본과 비교".
For each match and half (2 fps, outfield players only):
  1. roles from ALL players  (= the Bialkowski paper's condition)       -> mu_full
  2. hide players outside a camera window that follows the ball (W m of pitch length)
     roles from the VISIBLE players only (= our broadcast condition)    -> mu_vis
  A. role centre error  |mu_vis - mu_full| after 1:1 matching (m)
  B. role agreement     visible player gets the same role as with full data (%)
  C. off-screen error   hidden players: role mean + team shift (ours) vs team centre / last seen (m),
                        split by how long the player has been off-screen
  D. attack/defence     our proxy (team shift x > +5 m attack, < -5 m defence) vs real possession

usage (deep07, needs internet for the first download, ~2.4 GB for all matches):
  pip install floodlight
  python analysis/idsse_eval.py J03WMX            # one match
  python analysis/idsse_eval.py all --width 35     # all 7 matches
  python analysis/idsse_eval.py --selftest         # synthetic check, no download
"""
import json, sys, argparse
import numpy as np
from scipy.optimize import linear_sum_assignment
import role_distributions as R

MATCHES = ['J03WMX', 'J03WN1', 'J03WPY', 'J03WOH', 'J03WQQ', 'J03WOY', 'J03WR9']


# ---------------------------------------------------------------- core (plain numpy)
def camera_centre(ball_x, fps, half_len, width, tau=2.0):
    """x centre of a camera that follows the ball smoothly (exponential smoothing, tau seconds)"""
    a = 1.0 / (1.0 + tau * fps)
    c = np.zeros(len(ball_x)); prev = 0.0
    for i, b in enumerate(ball_x):
        if np.isfinite(b):
            prev = prev + a * (b - prev)
        c[i] = prev
    lim = half_len - width / 2
    return np.clip(c, -lim, lim)


def evaluate_team(P, cam, width, fps, poss_own=None, restarts=3):
    """P: (T, N, 2) outfield positions of one team, attacking +x (NaN = not on pitch).
    cam: (T,) camera x centre in the same coordinates. poss_own: (T,) 1 own ball, 0 opponent, NaN unknown."""
    T, N, _ = P.shape
    on = np.isfinite(P[..., 0])
    vis = on & (np.abs(P[..., 0] - cam[:, None]) <= width / 2)
    full_frames = [(t, P[t, on[t]]) for t in range(T) if on[t].sum() >= 8]
    vis_frames = [(t, P[t, vis[t]]) for t in range(T) if vis[t].sum() >= 3]
    best = lambda fr: min((R.fit(fr, init=i, seed=s, max_iter=40) for i, s in
                           [('kmeans', 0)] + [('tangled', k) for k in range(1, restarts)]), key=lambda r: r['nll'])
    mu_full, mu_vis = best(full_frames)['mu'], best(vis_frames)['mu']
    errA, col = R.match_error(mu_vis, mu_full)          # col[k_vis] = matching full-role index
    out = {'frames': T, 'visible_mean': float(vis[on.any(1)].sum(1).mean()), 'on_mean': float(on.sum(1).mean()),
           'A_role_centre_error_m': round(errA, 2)}

    agree = tot = 0
    rows = []                                            # off-screen errors
    last_seen = np.full((N, 2), np.nan); last_t = np.full(N, -1); last_shift = np.full((N, 2), np.nan)
    proxy = []
    prev_s = None
    for t in range(T):
        idx_on = np.where(on[t])[0]; idx_vis = np.where(vis[t])[0]
        if len(idx_on) < 8:
            continue
        s_full, ri_f, ci_f, _ = R.match_frame(P[t, idx_on], mu_full)
        role_full = {idx_on[i]: k for i, k in zip(ri_f, ci_f)}
        if len(idx_vis) < 3:
            prev_s = None
            continue
        s, ri, ci, _ = R.match_frame(P[t, idx_vis], mu_vis, s_prev=prev_s); prev_s = s
        for i, k in zip(ri, ci):
            p = idx_vis[i]
            if p in role_full:
                tot += 1; agree += int(col[k] == role_full[p])
        # off-screen players this frame
        hid = [p for p in idx_on if p not in set(idx_vis.tolist())]
        if hid:
            miss = [k for k in range(len(mu_vis)) if k not in set(ci.tolist())]
            if miss:
                pred = mu_vis[miss] + s
                C = np.linalg.norm(pred[:, None] - P[t, hid][None], axis=2)
                r_, c_ = linear_sum_assignment(C)
                ours = {hid[c]: C[r, c] for r, c in zip(r_, c_)}
                centre = P[t, idx_vis].mean(0)
                for p in hid:
                    gone = (t - last_t[p]) / fps if last_t[p] >= 0 else np.inf
                    rec = {'gone_s': float(gone), 'team_centre': float(np.hypot(*(centre - P[t, p])))}
                    if p in ours: rec['role_shift'] = float(ours[p])
                    if last_t[p] >= 0:
                        rec['last_seen'] = float(np.hypot(*(last_seen[p] - P[t, p])))
                        rec['last_seen_plus_shift'] = float(np.hypot(*(last_seen[p] + (s - last_shift[p]) - P[t, p])))
                    rows.append(rec)
        for i in idx_vis:
            last_seen[i] = P[t, i]; last_t[i] = t; last_shift[i] = s
        if poss_own is not None and np.isfinite(poss_own[t]):
            guess = 1 if s[0] > 5 else (0 if s[0] < -5 else -1)
            proxy.append((guess, int(poss_own[t])))
    out['B_role_agreement_pct'] = round(100 * agree / max(tot, 1), 1)
    buckets = [('<=3s', 0, 3), ('3-10s', 3, 10), ('>10s', 10, np.inf)]
    out['C_offscreen_median_m'] = {}
    for name, lo, hi in buckets + [('all', -1, np.inf)]:
        xs = [r for r in rows if lo < r['gone_s'] <= hi or (name == 'all')]
        if not xs:
            continue
        out['C_offscreen_median_m'][name] = {m: round(float(np.median([r[m] for r in xs if m in r])), 2)
                                              for m in ('role_shift', 'team_centre', 'last_seen', 'last_seen_plus_shift')
                                              if any(m in r for r in xs)} | {'n': len(xs)}
    if proxy:
        pr = np.array(proxy); dec = pr[:, 0] >= 0
        out['D_attack_defence'] = {'coverage_pct': round(100 * dec.mean(), 1),
                                   'accuracy_when_decided_pct': round(100 * (pr[dec, 0] == pr[dec, 1]).mean(), 1) if dec.any() else None}
    out['mu_full'] = np.round(mu_full, 2).tolist(); out['mu_vis'] = np.round(mu_vis, 2).tolist()
    return out


# ---------------------------------------------------------------- IDSSE adapter (floodlight)
def run_match(mid, width, restarts):
    from floodlight.io.datasets import IDSSEDataset
    ds = IDSSEDataset(mid)
    _, xy, poss, ballst, sheets, pitch = ds.get(mid, events=False)
    xmid = np.mean(pitch.xlim); ymid = np.mean(pitch.ylim); half_len = (pitch.xlim[1] - pitch.xlim[0]) / 2
    res = {}
    for seg in xy:
        fps_in = xy[seg]['Home'].framerate or 25
        step = max(1, int(round(fps_in / 2))); fps = fps_in / step
        ball = xy[seg]['Ball'].xy[::step] - [xmid, ymid]
        pos = poss[seg].code[::step].astype(float); alive = ballst[seg].code[::step].astype(float)
        for team, code_own in (('Home', 1), ('Away', 2)):
            A = xy[seg][team].xy[::step]
            P = A.reshape(len(A), -1, 2) - [xmid, ymid]
            sh = sheets[team]
            df = getattr(sh, 'teamsheet', None)       # DFL position code 'TW' = goalkeeper
            gk = [int(r['xID']) for _, r in df.iterrows() if str(r.get('position', '')).upper() == 'TW'] \
                if df is not None and 'xID' in df.columns else []
            keep = [i for i in range(P.shape[1]) if i not in gk]
            if not gk:                                    # fallback: player deepest on average is the keeper
                mx = np.nanmean(P[..., 0], 0); side = np.sign(np.nanmean(mx))
                keep = [i for i in range(P.shape[1]) if i != int(np.nanargmax(mx * side))]
            P = P[:, keep]
            fl = -np.sign(np.nanmean(P[..., 0]))          # defend left goal -> attack +x
            P = P * fl; b = ball * fl
            cam = camera_centre(b[:, 0], fps, half_len, width)
            own = np.where(alive > 0, (pos == code_own).astype(float), np.nan)
            print(f'{mid} {seg} {team}: {len(P)} frames at {fps:.1f} fps', flush=True)
            res[f'{seg}/{team}'] = evaluate_team(P, cam, width, fps, own, restarts)
    return res


# ---------------------------------------------------------------- synthetic self-test
def selftest():
    rng = np.random.default_rng(0)
    T = np.array([[x, y] for _, x, y in R.FORMS['4-4-2']], float); T -= T.mean(0)
    n = 1200; fps = 2.0
    centre = np.c_[np.cumsum(rng.normal(0, 1.2, n)), np.cumsum(rng.normal(0, 0.6, n))]
    centre[:, 0] = 25 * np.sin(np.linspace(0, 12, n)) + 0.3 * centre[:, 0] / (1 + np.abs(centre[:, 0]).max() / 20)
    centre[:, 1] = 6 * np.sin(np.linspace(0, 7, n))
    P = centre[:, None] + T[None] + rng.normal(0, 2.5, (n, 10, 2))
    ball = centre[:, 0] + rng.normal(0, 8, n)
    cam = camera_centre(ball, fps, 52.5, 35)
    own = (centre[:, 0] + rng.normal(0, 4, n) > 0).astype(float)
    out = evaluate_team(P, cam, 35, fps, own, restarts=2)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith('mu_')}, indent=1, ensure_ascii=False))
    assert out['A_role_centre_error_m'] < 6 and out['B_role_agreement_pct'] > 60, 'selftest failed'
    print('selftest ok')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('match', nargs='?', default='J03WMX')
    ap.add_argument('--width', type=float, default=35.0, help='camera window along the pitch length (m)')
    ap.add_argument('--restarts', type=int, default=3)
    ap.add_argument('--out', default='idsse_eval.json')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit()
    ids = MATCHES if a.match == 'all' else [a.match]
    allres = {mid: run_match(mid, a.width, a.restarts) for mid in ids}
    json.dump(allres, open(a.out, 'w'), indent=1, ensure_ascii=False)
    for mid, r in allres.items():
        for k, v in r.items():
            print(mid, k, {kk: vv for kk, vv in v.items() if not kk.startswith('mu_')})
    print('written:', a.out)
