"""Measure the 64-game phase-2 model the way the paper reports it (Le et al. 2017, p.5-6, Fig. 4):
whole sequence from the first frame, all 10 outfield players generated, deviation by position group.
Read-only, no training.   python check_paper.py  -> outputs/paper_check/report.md
"""
import glob, os, time
import numpy as np
import torch
from src import load, scenes as S, roles, model as M
from src.features import make_batch
from src.model import rollout

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(2)
out = 'outputs/paper_check'; os.makedirs(out, exist_ok=True)
t0 = time.time()
ck = torch.load('outputs/wc_all64/model_phase2.pt', map_location='cpu', weights_only=False)
net = M.RoleLSTMs(hidden=ck['state']['wh.0'].shape[1]); net.load_state_dict(ck['state']); net.to(dev).eval()
TD, TA = np.asarray(ck['template_def']), np.asarray(ck['template_att'])
tmpl = torch.tensor(TD, device=dev)


def group(x, y):
    if x > 12: return 'forward'
    if abs(y) >= 12: return 'full-back' if x < 0 else 'winger'
    if x < -9: return 'centre-back'
    return 'central midfield'


G = [group(*TD[r]) for r in range(10)]
sc = []
for g in ('10513', '3816', '3817'):
    trk = load.load_game(g, cache=f'outputs/cache/{g}_10fps.npz')
    sc += S.cut_scenes(trk, load.load_meta(g)[2], 'events')[0]
roles.apply_roles(sc, TD, TA)

E, V, T, SL = [], [], [], []      # per frame x role: error, visible, time in scene, scene length
with torch.no_grad():
    order = sorted(range(len(sc)), key=lambda i: sc[i]['L'])
    for c in range(0, len(order), 32):
        idx = order[c:c + 32]
        b = make_batch(sc, [(i, 0, sc[i]['L']) for i in idx], dev)
        L = b['D'].shape[1]
        ur = torch.zeros(len(idx), L, dtype=torch.bool, device=dev); ur[:, 0] = True
        e = (rollout(net, b, tmpl, ur, True) - b['D'][:, 1:]).norm(dim=-1)
        valid = b['valid'][:, 1:]
        for j, i in enumerate(idx):
            n = sc[i]['L'] - 1
            E.append(e[j, :n].cpu().numpy()); V.append(b['vis'][j, 1:n + 1].cpu().numpy())
            T.append(np.repeat(np.arange(1, n + 1)[:, None], 10, 1)); SL.append(np.full((n, 10), n))
E, V, T, SL = (np.concatenate(x) for x in (E, V, T, SL))
lens = np.array([(s['L'] - 1) / 10 for s in sc])

f = lambda x: f'{x:.2f}'
L = ['# 64-game phase-2 model measured like the paper (whole sequence, all 10 generated, test games)', '',
     f'Test scenes {len(sc)}; scene length mean {lens.mean():.1f} s, median {np.median(lens):.1f} s '
     f'(paper: mean about 17 s). Role groups from the template: ' +
     ', '.join(f'D{r + 1} ({TD[r][0]:.0f},{TD[r][1]:.0f}) {G[r]}' for r in range(10)), '',
     '| frames | mean | median | share < 3 m |', '|---|---|---|---|',
     f'| VISIBLE only (our main number) | {f(E[V].mean())} | {f(np.median(E[V]))} | {(E[V] < 3).mean():.0%} |',
     f'| all frames (incl. ESTIMATED) | {f(E.mean())} | {f(np.median(E))} | {(E < 3).mean():.0%} |',
     f'| VISIBLE, scenes >= 17 s only | {f(E[V & (SL >= 170)].mean())} | {f(np.median(E[V & (SL >= 170)]))} | {(E[V & (SL >= 170)] < 3).mean():.0%} |',
     '', '## By position group (VISIBLE frames)', '', '| group | roles | mean | median | share < 3 m |', '|---|---|---|---|---|']
for gname in ('centre-back', 'full-back', 'central midfield', 'winger', 'forward'):
    cols = [r for r in range(10) if G[r] == gname]
    if not cols: continue
    x = E[:, cols][V[:, cols]]
    L.append(f'| {gname} | {", ".join(f"D{r + 1}" for r in cols)} | {f(x.mean())} | {f(np.median(x))} | {(x < 3).mean():.0%} |')
L += ['', f'Runtime {(time.time() - t0) / 60:.1f} min, {dev}.']
open(f'{out}/report.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
