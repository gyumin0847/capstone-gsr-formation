"""Phase 2 of the paper: start from the phase-1 model and train with all 10 defenders on their own predictions.
   Same games, same split and same template as run_multi.py.
   python run_phase2.py                 # uses outputs/wc_events/model_both.pt
"""
import argparse, copy, glob, json, os, random, time
from multiprocessing import Pool
import numpy as np
import torch
from src import load, scenes as S, roles, model as M, train as T, evaluate as E

ap = argparse.ArgumentParser()
ap.add_argument('--src', default='outputs/wc_events', help='folder with the phase-1 model')
ap.add_argument('--mask', default='both')
ap.add_argument('--out', default='outputs/wc_events_phase2')
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--n-val', type=int, default=3)
ap.add_argument('--n-test', type=int, default=3)
ap.add_argument('--raw', default=load.RAW)
ap.add_argument('--epochs', type=int, default=5, help='epochs per curriculum stage')
ap.add_argument('--ks', default='5,10,20,50')
ap.add_argument('--lr', type=float, default=5e-4)
ap.add_argument('--bs', type=int, default=256)
ap.add_argument('--workers', type=int, default=2)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); os.makedirs('outputs/cache', exist_ok=True)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
logf = open(f'{a.out}/run_log.txt', 'w')


def log(*x):
    s = ' '.join(str(v) for v in x)
    print(s, flush=True); logf.write(s + '\n'); logf.flush()


def prep(g):
    trk = load.load_game(g, a.raw, cache=f'outputs/cache/{g}_10fps.npz')
    _, _, ev = load.load_meta(g, a.raw)
    return g, S.cut_scenes(trk, ev)[0], trk['info']['names']


t0 = time.time()
games = sorted({os.path.basename(p)[:-5] for p in glob.glob(f'{a.raw}/Event Data/*/*.json')}
               & {os.path.basename(p).split('.')[0] for p in glob.glob(f'{a.raw}/Tracking Data/*.jsonl.bz2')})
if a.limit:
    games = games[:a.limit]
with Pool(a.workers) as pool:
    res = pool.map(prep, games)
scenes = {g: sc for g, sc, _ in res}
names = {g: nm for g, _, nm in res}
games = sorted(scenes)
rng = random.Random(a.seed); order = games[:]; rng.shuffle(order)
g_test, g_val, g_train = sorted(order[:a.n_test]), sorted(order[a.n_test:a.n_test + a.n_val]), sorted(order[a.n_test + a.n_val:])
sc_tr = [s for g in g_train for s in scenes[g]]
sc_va = [s for g in g_val for s in scenes[g]]
sc_te = [s for g in g_test for s in scenes[g]]
log(f'== phase 2  {len(games)} games  train {len(g_train)} / val {g_val} / test {g_test}  device {a.device}')
TD, TA, rhist = roles.assign_roles(sc_tr, sc_tr + sc_va + sc_te)

dev = torch.device(a.device)
tmpl = torch.tensor(TD, device=dev)
ck = torch.load(f'{a.src}/model_{a.mask}.pt', map_location='cpu', weights_only=False)
same = np.allclose(ck['template_def'], TD, atol=1e-4)
log(f'phase-1 model {a.src}/model_{a.mask}.pt, template matches this split: {same}')
if not same and not a.limit:
    raise SystemExit('template differs from the phase-1 run: games or split changed')
net = M.RoleLSTMs(hidden=ck['args']['hidden']).to(dev)
net.load_state_dict(ck['state'])
p1 = copy.deepcopy(net)

# phase-1 model, evaluated two ways
R = {}
R['phase1_single'] = dict(h=E.horizon_eval(p1, sc_te, tmpl, dev, joint=False), scene=E.scene_eval(p1, sc_te, tmpl, dev, joint=False))
R['phase1_joint'] = dict(h=E.horizon_eval(p1, sc_te, tmpl, dev, joint=True), scene=E.scene_eval(p1, sc_te, tmpl, dev, joint=True))
log('phase-1 model on test, 5 s strict: single', round(R['phase1_single']['h'][50]['strict']['lstm'], 2),
    'joint', round(R['phase1_joint']['h'][50]['strict']['lstm'], 2))

# phase-2 training
ks = tuple(int(k) for k in a.ks.split(','))
t1 = time.time()
hist = T.train(net, sc_tr, sc_va, tmpl, dev, ks=ks, epochs=a.epochs, lr=a.lr, bs=a.bs, seed=a.seed,
               mask=a.mask, joint=True, keep_best=True, log=log)
train_min = round((time.time() - t1) / 60, 1)
torch.save(dict(state=net.state_dict(), template_def=TD, template_att=TA, args=vars(a), mask=a.mask, phase=2), f'{a.out}/model_phase2.pt')
T.plot_curve(hist, f'{a.out}/learning_curve_phase2.png')
R['phase2_joint'] = dict(h=E.horizon_eval(net, sc_te, tmpl, dev, joint=True), scene=E.scene_eval(net, sc_te, tmpl, dev, joint=True))

# figures: same pick rule as run_multi, ghost with all 10 defenders generated
cand = [i for i, s in enumerate(sc_te) if s['L'] >= 50]
pick = sorted(cand, key=lambda i: -sc_te[i]['vis'].mean())[:3] or sorted(range(len(sc_te)), key=lambda i: -sc_te[i]['L'])[:3]
figs = []
for n, i in enumerate(pick):
    s = sc_te[i]
    gh = E.ghost(net, sc_te, i, tmpl, dev, joint=True)
    mm, ss = divmod(int(s['time']), 60)
    p = f'{a.out}/ghost_joint_test_scene_{n + 1}.png'
    E.plot_scene(s, gh, p, f'PHASE 2, all 10 generated. game {s["game"]}: {s["defend_name"]} defending, half {s["period"]} '
                           f'+{mm:02d}:{ss:02d}, {(s["L"] - 1) / 10:.1f} s, VISIBLE {s["vis"].mean():.0%} '
                           f'(ghost error at end {np.linalg.norm(gh[-1] - s["D"][-1], axis=-1).mean():.1f} m)')
    figs.append(os.path.basename(p))

total = round((time.time() - t0) / 60, 1)
json.dump(dict(results=R, history=hist, test_games=[f'{g} {names[g]["home"]}-{names[g]["away"]}' for g in g_test],
               train_minutes=train_min, minutes=total, args=vars(a)), open(f'{a.out}/metrics.json', 'w'), indent=1)

f2 = lambda x: '-' if x is None else f'{x:.2f}'
pc = lambda x: '-' if x is None else f'{x:.0%}'
lab = {'phase1_single': 'phase 1 model, team-mates real', 'phase1_joint': 'phase 1 model, all 10 generated',
       'phase2_joint': 'phase 2 model, all 10 generated'}
b0 = R['phase2_joint']['h']
best = [h for h in hist if 'best_epoch' in h][0]
tg = ', '.join('%s %s-%s' % (g, names[g]['home'], names[g]['away']) for g in g_test)
L = ['# Phase 2 (all 10 defenders generated) on World Cup test games', '',
     f'Command: `python run_phase2.py --epochs {a.epochs}` ({total} min, training {train_min} min, {a.device}). '
     f'Started from {a.src}/model_{a.mask}.pt; kept epoch {best["best_epoch"]} (val K=50 joint {best["best_val_k50"]:.3f} m^2). '
     f'Data: PFF FC 2022 World Cup dataset.', '',
     f'Test games: {tg}.', '',
     '## Error after rolling out from a real start (m, start and end VISIBLE)', '',
     '| model / evaluation | 1 s | 2 s | 5 s | within 3 m at 5 s |', '|---|---|---|---|---|']
for k, r in R.items():
    L.append(f'| {lab[k]} | ' + ' | '.join(f2(r['h'][h]['strict']['lstm']) for h in (10, 20, 50)) +
             f' | {pc(r["h"][50]["within_3m_strict"]["lstm"])} |')
for bk, bl in (('still', 'stand still'), ('const_vel', 'constant velocity')):
    L.append(f'| baseline: {bl} | ' + ' | '.join(f2(b0[h]['strict'][bk]) for h in (10, 20, 50)) +
             f' | {pc(b0[50]["within_3m_strict"][bk])} |')
L += ['', '## Whole scene from its first frame (paper style, m, VISIBLE frames)', '',
      '| model / evaluation | mean over scene | first 10 s | within 3 m |', '|---|---|---|---|']
for k, r in R.items():
    s = r['scene']
    L.append(f'| {lab[k]} | {s["lstm"]:.2f} | {s["lstm_10s"]:.2f} | {s["lstm_within_3m"]:.0%} |')
s = R['phase2_joint']['scene']
L += [f'| baseline: stand still | {s["still"]:.2f} | {s["still_10s"]:.2f} | - |', '',
      f'Frames counted: {s["frames"]}.', '',
      'Files: learning_curve_phase2.png, ' + ', '.join(figs) + ', metrics.json, model_phase2.pt, run_log.txt']
open(f'{a.out}/summary.md', 'w').write('\n'.join(L) + '\n')
log('\n'.join(L))
log(f'== done in {total} min')
