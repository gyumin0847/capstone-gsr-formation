"""Off-screen imputation hold-out test (usage: python analysis/impute_eval.py manutd_viewer.json 0|1).
Fit roles on one half of the first half (train), then on test frames where >= 8 players are visible,
hide everyone outside a narrower simulated camera window (15/20/25 m of pitch length) and predict the
hidden players' positions. Methods: team_mean (centre of the kept players), role_naive (role mean + mean
shift), role_median (role mean + median shift, = role_distributions.impute), role_cont (+ previous-frame
shift), oracle_shift (role mean + shift from ALL players: the floor set by how much a role moves).
Truth roles of hidden players come from matching the full frame. Roles are fitted on minutes 0-22.5 and tested on 22.5-45, then the other way round."""
import json, sys, numpy as np

import role_distributions as R

d = json.load(open(sys.argv[1]))
t = sys.argv[2]; fl = 1. if t == '0' else -1.
rng = np.random.default_rng(0)
fr_all = R.team_frames(d, t, fl, min_vis=1)
fr_fit = [(f, P) for f, P in fr_all if len(P) >= 3]
rows = []
for split in (0, 1):
    tr = [(f, P) for f, P in fr_fit if (f <= 2700) == (split == 0)]
    te = [(f, P) for f, P in fr_all if (f <= 2700) != (split == 0)]
    fit = min((R.fit(tr, init=i, seed=s, max_iter=30) for i, s in (('kmeans', 0), ('tangled', 1))), key=lambda r: r['nll'])
    mu = fit['mu']
    print('split', split, 'fit iters', fit['iters'], flush=True)
    prev_full = (None, None)
    prev_sim = {W: (None, None) for W in (15, 20, 25)}
    for f, P in te:
        sp = prev_full[1] if prev_full[0] is not None and f - prev_full[0] <= 2 else None
        s_full, ri, ci, _ = R.match_frame(P, mu, s_prev=sp)
        prev_full = (f, s_full)
        if len(P) < 8:
            for W in prev_sim: prev_sim[W] = (None, None)
            continue
        role_of = dict(zip(ri.tolist(), ci.tolist()))
        c = np.median(P[:, 0]) + rng.uniform(-8, 8)
        for W in (15, 20, 25):
            keep = np.abs(P[:, 0] - c) <= W / 2
            hid = [i for i in np.where(~keep)[0] if i in role_of]
            if keep.sum() < 3 or len(hid) < 2:
                prev_sim[W] = (None, None); continue
            K_ = P[keep]
            pf, ps = prev_sim[W]
            s_cont = R.match_frame(K_, mu, s_prev=ps if pf is not None and f - pf <= 2 else None)[0]
            s_med = R.match_frame(K_, mu)[0]
            prev_sim[W] = (f, s_cont)
            s_naive = K_.mean(0) - mu.mean(0)
            for i in hid:
                k = role_of[i]; p = P[i]
                rows.append({'W': W, 'kept': int(keep.sum()),
                             'team_mean': float(np.hypot(*(K_.mean(0) - p))),
                             'role_naive': float(np.hypot(*(mu[k] + s_naive - p))),
                             'role_median': float(np.hypot(*(mu[k] + s_med - p))),
                             'role_cont': float(np.hypot(*(mu[k] + s_cont - p))),
                             'oracle_shift': float(np.hypot(*(mu[k] + s_full - p)))})

M = ('team_mean', 'role_naive', 'role_median', 'role_cont', 'oracle_shift')
for W in (15, 20, 25):
    x = [r for r in rows if r['W'] == W]
    print(f'team {t} W={W}m n={len(x)} ' + ' '.join(f"{m} {np.median([r[m] for r in x]):.1f}" for m in M))
x = rows
print(f'team {t} ALL n={len(x)} ' + ' '.join(f"{m} med {np.median([r[m] for r in x]):.1f} mean {np.mean([r[m] for r in x]):.1f}" for m in M))
