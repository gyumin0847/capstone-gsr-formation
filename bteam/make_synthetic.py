"""가짜 경기로 B팀 파이프라인 시험 (IDSSE 없이). IDSSE와 같은 원본 형태(양 팀 GK 포함 좌표, 공, 볼 소유,
볼 인/아웃, 하프마다 공격 방향 바뀜, 교체 1명)를 만들어 prepare_idsse.process_segment 에 그대로 넣는다.

수비 팀은 가까운 공격수 쪽으로 붙는(마크) 규칙이 있어서, 공만 보는 모델보다 상대까지 보는 모델이
나아야 정상이다 -> run.py 결과에서 tfA/knn/mlp < ballshift < mean 이면 파이프라인이 제대로 도는 것.

usage: python bteam/make_synthetic.py --out /tmp/bteam_syn --matches 4 --frames 1200
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'analysis'))
import role_distributions as R  # noqa: E402
from prepare_idsse import process_segment  # noqa: E402

FORMS = ['4-4-2', '4-3-3', '4-2-3-1', '3-5-2']


def template(name):
    T = np.array([[x, y] for _, x, y in R.FORMS[name]], float)
    return T - T.mean(0)


def simulate_half(rng, T, form_home, form_away, home_dir):
    """returns floodlight-like arrays: home/away (T, 12*2) [col 0 = GK, 1..10 outfield, 11 = substitute]"""
    # ball: smoothed random walk; possession: Markov switches
    b = np.zeros((T, 2)); v = np.zeros(2)
    for t in range(1, T):
        v = 0.85 * v + rng.normal(0, [2.2, 1.6]); b[t] = np.clip(b[t - 1] + v, [-50, -32], [50, 32])
        v = np.where(np.abs(b[t]) >= [50, 32], -v, v)
    poss = np.ones(T); cur = 1
    for t in range(T):
        if rng.random() < 0.03:
            cur = 3 - cur
        poss[t] = cur
    alive = (rng.random(T) > 0.05).astype(float)
    tpl = {1: template(form_home), 2: template(form_away)}
    fl = {1: home_dir, 2: -home_dir}
    noise = {c: np.zeros((10, 2)) for c in (1, 2)}
    out = {c: np.full((T, 12, 2), np.nan) for c in (1, 2)}
    for t in range(T):
        pos = {}
        for c in (1, 2):                                       # base shape in the team's own frame
            att = poss[t] == c
            bl = b[t] * fl[c]
            sc = np.array([1.15, 1.1]) if att else np.array([0.8, 0.85])
            s = np.array([0.55 * bl[0] + (6 if att else -6), 0.35 * bl[1]])
            noise[c] = 0.9 * noise[c] + rng.normal(0, 0.9, (10, 2))
            pos[c] = (tpl[c] * sc + s + noise[c]) * fl[c]          # global frame
        a = int(poss[t]); d = 3 - a                              # defenders step towards the nearest attacker
        A = pos[a]
        for k in range(10):
            j = np.argmin(np.hypot(*(A - pos[d][k]).T))
            pos[d][k] += 0.35 * (A[j] - pos[d][k])
        for c in (1, 2):
            P = np.clip(pos[c], [-52, -33], [52, 33])
            out[c][t, 1:11] = P
            out[c][t, 0] = [(-48 + 0.08 * b[t, 0] * fl[c]) * fl[c], 0.1 * b[t, 1]]
    for c in (1, 2):                                           # substitution: outfield col 5 -> col 11
        h = T // 2
        out[c][h:, 11] = out[c][h:, 5]; out[c][h:, 5] = np.nan
    return out[1].reshape(T, -1), out[2].reshape(T, -1), b, poss, alive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--matches', type=int, default=4)
    ap.add_argument('--frames', type=int, default=1200); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for m in range(a.matches):
        mid = f'SYN{m:02d}'
        fh, fa = rng.choice(FORMS, 2, replace=False)
        for si, seg in enumerate(('firstHalf', 'secondHalf')):
            home, away, b, poss, alive = simulate_half(rng, a.frames, fh, fa, 1.0 if si == 0 else -1.0)
            res = process_segment(home, away, b, poss, alive, [0], [0], fps_in=2.0, restarts=2)
            for team, d in res.items():
                np.savez_compressed(os.path.join(a.out, f'{mid}_{seg}_{team}.npz'), match=mid, seg=seg, team=team, **d)
        print(f'{mid}: home {fh} vs away {fa}', flush=True)


if __name__ == '__main__':
    main()
