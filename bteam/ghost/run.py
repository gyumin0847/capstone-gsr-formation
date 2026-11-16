"""One command for the whole pipeline:  python run.py --game 10515"""
import argparse, csv, json, os, random, time
import numpy as np
import torch
from src import load, scenes as S, roles, features, model as M, train as T, evaluate as E

ap = argparse.ArgumentParser()
ap.add_argument('--game', default='10515')
ap.add_argument('--raw', default=load.RAW)
ap.add_argument('--out', default=None)
ap.add_argument('--epochs', type=int, default=30, help='epochs per curriculum stage')
ap.add_argument('--ks', default='1,5,10,20,50')
ap.add_argument('--hidden', type=int, default=128)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
a = ap.parse_args()
out = a.out or f'outputs/game_{a.game}'
os.makedirs(out, exist_ok=True)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
logf = open(f'{out}/run_log.txt', 'w')


def log(*x):
    s = ' '.join(str(v) for v in x)
    print(s, flush=True); logf.write(s + '\n'); logf.flush()


t_start = time.time()
log(f'== game {a.game}  device {a.device}  epochs/stage {a.epochs}  K {a.ks}')

# 1. load and 10 fps
trk = load.load_game(a.game, a.raw, cache=f'{out}/cache_10fps.npz')
info = trk['info']
log('[1] load', json.dumps(info['stats']))
_, _, events = load.load_meta(a.game, a.raw)

# 2. scenes
scenes, rows, rep = S.cut_scenes(trk, events)
log('[2] scenes', json.dumps(rep, ensure_ascii=False, indent=1))
with open(f'{out}/scenes.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# 3. roles
TD, TA, rhist = roles.assign_roles(scenes)
roles.plot_template(TD, TA, f'{out}/role_template.png', rhist['defence_mean_centre'], rhist['attack_mean_centre'])
log('[3] template mean distance to slot per iteration (m)', json.dumps({k: [round(x, 2) for x in v] for k, v in rhist.items()}))

# 4. inputs / targets / masks
sc_tr = [s for s in scenes if s['period'] == 1]
sc_va = [s for s in scenes if s['period'] == 2]
log(f'[4] input dim {features.N_IN} ->', '; '.join(f'{n} {d}' for n, d in features.PARTS))
log(f'    train scenes (1st half) {len(sc_tr)}, val scenes (2nd half) {len(sc_va)}')

# 5. training
dev = torch.device(a.device)
tmpl = torch.tensor(TD, device=dev)
net = M.RoleLSTMs(hidden=a.hidden).to(dev)
log(f'[5] 10 role LSTMs, 2 layers x {a.hidden}, params {sum(p.numel() for p in net.parameters()):,}')
ks = tuple(int(k) for k in a.ks.split(','))
hist = T.train(net, sc_tr, sc_va, tmpl, dev, ks=ks, epochs=a.epochs, lr=a.lr, seed=a.seed, log=log)
torch.save(dict(state=net.state_dict(), template_def=TD, template_att=TA, args=vars(a)), f'{out}/model.pt')
T.plot_curve(hist, f'{out}/learning_curve.png')

# 6. evaluation
hz = E.horizon_eval(net, sc_va, tmpl, dev)
vs = E.vis_stats(scenes, sc_tr, sc_va)
order = sorted([i for i, s in enumerate(sc_va) if s['L'] >= 50], key=lambda i: -sc_va[i]['vis'].mean())[:3]
if len(order) < 3:
    order = sorted(range(len(sc_va)), key=lambda i: -sc_va[i]['L'])[:3]
figs = []
for n, i in enumerate(order):
    s = sc_va[i]
    gh = E.ghost(net, sc_va, i, tmpl, dev)
    mm, ss = divmod(int(s['time']), 60)
    p = f'{out}/ghost_val_scene_{n + 1}.png'
    E.plot_scene(s, gh, p, f'{info["names"][s["side"]]} defending, half 2 +{mm:02d}:{ss:02d}, {(s["L"] - 1) / 10:.1f} s, '
                           f'ends: {s["why"]}, defenders VISIBLE {s["vis"].mean():.0%}  (ghost error at end '
                           f'{np.linalg.norm(gh[-1] - s["D"][-1], axis=-1).mean():.1f} m)')
    figs.append(os.path.basename(p))

metrics = dict(game=a.game, scenes=rep, template=dict(defence=TD.tolist(), attack=TA.tolist(), **rhist),
               input_dim=features.N_IN, input_parts=features.PARTS, history=hist, horizon=hz, visibility=vs,
               figures=figs, minutes=round((time.time() - t_start) / 60, 1))
json.dump(metrics, open(f'{out}/metrics.json', 'w'), indent=1, ensure_ascii=False)

# summary tables
f2 = lambda x: '-' if x is None else f'{x:.2f}'
L = [f'# Game {a.game} ({info["names"]["home"]} vs {info["names"]["away"]}) ghosting pipeline', '',
     f'Command: `python run.py --game {a.game} --epochs {a.epochs}` ({metrics["minutes"]} min, {a.device})', '',
     '## Scenes', '', '| item | value |', '|---|---|',
     f'| candidate scenes | {rep["candidate_total"]} |', f'| excluded | {rep["excluded_total"]} {rep["excluded"]} |',
     f'| kept | {rep["kept"]} (half 1: {rep["kept_by_half"][1]}, half 2: {rep["kept_by_half"][2]}) |',
     f'| kept by defending team | {rep["kept_by_defending_team"]} |',
     f'| length mean / median | {rep["mean_length_s"]:.1f} s / {rep["median_length_s"]:.1f} s |',
     f'| total | {rep["total_minutes"]:.1f} min |', f'| end reasons (kept) | {rep["end_reason_of_kept"]} |',
     f'| ball gaps interpolated | {rep["ball_gap_interpolated_scenes"]} scenes |', '',
     f'Input dim {features.N_IN} (paper 399).', '',
     '## Rollout error on half 2 (m, VISIBLE end points only)', '',
     '| horizon | start points | visible end points | LSTM | stand still | constant velocity |', '|---|---|---|---|---|---|']
for h, r in hz.items():
    if r:
        L.append(f'| {h / 10:.0f} s | {r["start_points"]} | {r["n_end"]} | {r["end"]["lstm"]:.2f} | '
                 f'{r["end"]["still"]:.2f} | {r["end"]["const_vel"]:.2f} |')
L += ['', '## Per role', '', '| role | template x,y | VISIBLE share all | VISIBLE share 10 s before shots | '
      'targets train | targets val | ' + ' | '.join(f'LSTM / still / const-vel {h / 10:.0f} s' for h in hz) + ' |',
      '|' + '---|' * (6 + len(hz))]
for r in range(10):
    sv = vs['visible_share_shot_10s']
    cells = [f'D{r + 1}', f'{TD[r, 0]:.0f}, {TD[r, 1]:.0f}', f'{vs["visible_share_all_scenes"][r]:.0%}',
             '-' if sv is None else f'{sv[r]:.0%}', str(vs['targets_train'][r]), str(vs['targets_val'][r])]
    for h, rr in hz.items():
        cells.append('-' if not rr else ' / '.join(f2(rr['per_role_end'][k][r]) for k in ('lstm', 'still', 'const_vel')))
    L.append('| ' + ' | '.join(cells) + ' |')
L += ['', f'VISIBLE share overall: all scenes {vs["visible_share_all_overall"]:.0%} ({vs["frames_all"]} frames); '
      f'10 s before shots {"-" if vs["visible_share_shot_10s_overall"] is None else format(vs["visible_share_shot_10s_overall"], ".0%")} '
      f'({vs["shots_in_kept_scenes"]} shots, {vs["frames_shot_10s"]} frames).',
      f'Targets after mask: train {sum(vs["targets_train"])} / {vs["target_slots_train"]}, '
      f'val {sum(vs["targets_val"])} / {vs["target_slots_val"]}.', '',
      'Files: role_template.png, learning_curve.png, ' + ', '.join(figs) + ', scenes.csv, metrics.json, model.pt, run_log.txt']
open(f'{out}/summary.md', 'w').write('\n'.join(L) + '\n')
log('\n'.join(L))
log(f'== done in {metrics["minutes"]} min')
