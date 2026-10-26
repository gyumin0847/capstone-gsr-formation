"""Re-do the per-frame role assignment with the best EM role distributions.

pipeline/manutd_v2.py writes a viewer json whose per-frame assignment `a` and team shift `sh` come from one
run of the role finder. Hard-assignment EM on broadcast data has several local optima, so this script fits
the roles with restarts (role_distributions.best_fit, optionally warm-started from the pipeline's own slots)
and rewrites `a` and `sh`. build.py / phase.py / make_tool.py then run unchanged on the output.

usage: python analysis/reassign.py manutd_viewer.json manutd_viewer_em.json [restarts]
Team "1" is rotated 180 deg so both teams attack to the right (same convention as build.py).
"""
import json, sys
import numpy as np
import role_distributions as R

src, dst = sys.argv[1], sys.argv[2]
restarts = int(sys.argv[3]) if len(sys.argv) > 3 else 4
d = json.load(open(src))
FLIP = {'0': 1.0, '1': -1.0}
mus = {}
for t, fl in FLIP.items():
    fr = R.team_frames(d, t, fl)
    # warm start from the pipeline's slots (mean of relative positions per assigned slot)
    rel, role = [], []
    for f in d['frames']:
        x = f.get(t)
        if not x or x.get('sh') is None:
            continue
        for p, a in zip(x['p'], x['a']):
            if a >= 0:
                rel.append([(p[0] - x['sh'][0]) * fl, (p[1] - x['sh'][1]) * fl]); role.append(a)
    warm = None
    if rel:
        rel, role = np.array(rel), np.array(role)
        if all((role == k).sum() >= 5 for k in range(R.K)):
            warm = (np.array([rel[role == k].mean(0) for k in range(R.K)]),
                    np.array([np.cov(rel[role == k].T) + R.REG * np.eye(2) for k in range(R.K)]))
    best = R.best_fit(fr, restarts=restarts, warm=warm, max_iter=40)
    mus[t] = best['mu']
    print(f'team {t}: {len(fr)} frames, best cost {best["nll"]:.0f}')

out = {k: v for k, v in d.items() if k != 'frames'}
out['frames'] = []
for f in d['frames']:
    g = {'f': f['f']}
    for t, fl in FLIP.items():
        x = f.get(t)
        if not x:
            g[t] = None
            continue
        P = np.asarray(x['p'], float)
        a, sh = [-1] * len(P), None
        if len(P) >= 3:
            s, ri, ci, _ = R.match_frame(P * fl, mus[t])
            for i, k in zip(ri, ci):
                a[int(i)] = int(k)
            sh = [round(float(s[0] * fl), 1), round(float(s[1] * fl), 1)]
        g[t] = {'p': x['p'], 'sh': sh, 'a': a}
    out['frames'].append(g)
json.dump(out, open(dst, 'w'), separators=(',', ':'))
print('written:', dst)
