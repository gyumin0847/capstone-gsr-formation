"""Many World Cup games, split by game (train / val / test), two loss masks compared on val, picked one checked on test.
   python run_multi.py                # all games that have an event file
"""
import argparse, collections, csv, glob, json, os, random, time
from multiprocessing import Pool
import numpy as np
import torch
from src import load, scenes as S, roles, features, model as M, train as T, evaluate as E

ap = argparse.ArgumentParser()
ap.add_argument('--games', default='events', help="'events' = every game with an event file, 'all' = every tracking file, or comma list")
ap.add_argument('--limit', type=int, default=0, help='use only the first N games (smoke test)')
ap.add_argument('--n-val', type=int, default=3)
ap.add_argument('--n-test', type=int, default=3)
ap.add_argument('--val-games', default='', help='fixed validation games (comma list); default: random')
ap.add_argument('--test-games', default='', help='fixed test games (comma list); default: random')
ap.add_argument('--rule', default='events', choices=['events', 'tracking'], help='scene rule for training games')
ap.add_argument('--eval-rule', default='events', choices=['events', 'tracking'], help='scene rule for fixed val/test games')
ap.add_argument('--keep-best', action='store_true', help='keep the weights with the best val loss at K=50')
ap.add_argument('--phase2-epochs', type=int, default=0, help='>0: continue the picked model with phase 2 (all 10 generated)')
ap.add_argument('--phase2-ks', default='5,10,20,50')
ap.add_argument('--phase2-lr', type=float, default=5e-4)
ap.add_argument('--raw', default=load.RAW)
ap.add_argument('--out', default='outputs/wc_events')
ap.add_argument('--epochs', type=int, default=10, help='epochs per curriculum stage')
ap.add_argument('--ks', default='1,5,10,20,50')
ap.add_argument('--masks', default='next,both')
ap.add_argument('--hidden', type=int, default=128)
ap.add_argument('--bs', type=int, default=256)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--workers', type=int, default=4)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); os.makedirs('outputs/cache', exist_ok=True)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
logf = open(f'{a.out}/run_log.txt', 'w')


def log(*x):
    s = ' '.join(str(v) for v in x)
    print(s, flush=True); logf.write(s + '\n'); logf.flush()


def prep(job):
    """Load one game and cut its scenes (runs in a worker)."""
    g, rule = job
    try:
        trk = load.load_game(g, a.raw, cache=f'outputs/cache/{g}_10fps.npz')
        _, _, ev = load.load_meta(g, a.raw)
        if rule == 'events' and not ev:
            rule = 'tracking'
        sc, rows, rep = S.cut_scenes(trk, ev, rule)
        rep['rule'] = rule
        return g, sc, rows, rep, trk['info']['names'], None
    except Exception as e:      # report and skip a broken game
        return g, [], [], None, None, repr(e)


t0 = time.time()
tracking_ids = {os.path.basename(p).split('.')[0] for p in glob.glob(f'{a.raw}/Tracking Data/*.jsonl.bz2')}
if a.games == 'events':
    games = sorted({os.path.basename(p)[:-5] for p in glob.glob(f'{a.raw}/Event Data/*/*.json')} & tracking_ids)
elif a.games == 'all':
    games = sorted(tracking_ids)
else:
    games = a.games.split(',')
if a.limit:
    games = games[:a.limit]
fixed_val = [g for g in a.val_games.split(',') if g]
fixed_test = [g for g in a.test_games.split(',') if g]
games = sorted(set(games) | set(fixed_val) | set(fixed_test))
eval_set = set(fixed_val) | set(fixed_test)
log(f'== {len(games)} games, device {a.device}, epochs/stage {a.epochs}, K {a.ks}, masks {a.masks}, batch {a.bs}, '
    f'rule train {a.rule} / fixed val+test {a.eval_rule}, keep_best {a.keep_best}, phase2 epochs {a.phase2_epochs}')

# 1-2. load and cut, per game
with Pool(a.workers) as pool:
    res = pool.map(prep, [(g, a.eval_rule if g in eval_set else a.rule) for g in games])
scenes, rows, per_game, names = {}, [], {}, {}
for g, sc, rw, rep, nm, err in res:
    if err:
        log(f'   game {g} skipped: {err}'); continue
    scenes[g], names[g] = sc, nm
    rows += rw
    per_game[g] = dict(rule=rep['rule'], candidates=rep['candidate_total'], kept=rep['kept'], minutes=round(rep['total_minutes'], 1),
                       excluded=rep['excluded'], gk_side_matches_metadata=rep['goal_side'], events=rep['events'])
    log(f'   {g} [{rep["rule"]}] {nm["home"]} vs {nm["away"]}: {rep["candidate_total"]} candidates -> {rep["kept"]} scenes, '
        f'{rep["total_minutes"]:.1f} min, excluded {rep["excluded"]}')
with open(f'{a.out}/scenes.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
games = sorted(scenes)
log(f'[1-2] loaded {len(games)} games in {(time.time() - t0) / 60:.1f} min')

# split by game
if fixed_test and fixed_val:
    g_test, g_val = sorted(fixed_test), sorted(fixed_val)
    g_train = sorted(g for g in games if g not in eval_set)
else:
    rng = random.Random(a.seed); order = games[:]; rng.shuffle(order)
    g_test, g_val, g_train = sorted(order[:a.n_test]), sorted(order[a.n_test:a.n_test + a.n_val]), sorted(order[a.n_test + a.n_val:])
sc_tr = [s for g in g_train for s in scenes[g]]
sc_va = [s for g in g_val for s in scenes[g]]
sc_te = [s for g in g_test for s in scenes[g]]
mins = lambda sc: sum((s['L'] - 1) * 0.1 for s in sc) / 60
split = {k: dict(games=[f'{g} {names[g]["home"]}-{names[g]["away"]}' for g in gg], scenes=len(sc), minutes=round(mins(sc), 1))
         for k, gg, sc in (('train', g_train, sc_tr), ('val', g_val, sc_va), ('test', g_test, sc_te))}
log('[split]', json.dumps(split, ensure_ascii=False))
allsc = sc_tr + sc_va + sc_te
excl = collections.Counter()
for g in games:
    excl.update(per_game[g]['excluded'])

# 3. roles: fitted on train games only, applied everywhere
TD, TA, rhist = roles.assign_roles(sc_tr, allsc)
roles.plot_template(TD, TA, f'{a.out}/role_template.png', rhist['defence_mean_centre'], rhist['attack_mean_centre'])
log('[3] centred template, mean distance to slot per iteration (m)',
    json.dumps({k: [round(x, 2) for x in v] for k, v in rhist.items() if k.endswith('iter')}))
log(f'[4] input dim {features.N_IN}')

# 5. train one model per mask
dev = torch.device(a.device)
tmpl = torch.tensor(TD, device=dev)
ks = tuple(int(k) for k in a.ks.split(','))
runs = {}
for mask in a.masks.split(','):
    torch.manual_seed(a.seed)
    net = M.RoleLSTMs(hidden=a.hidden).to(dev)
    log(f'[5] mask={mask}: 10 role LSTMs 2x{a.hidden}, params {sum(p.numel() for p in net.parameters()):,}')
    t1 = time.time()
    hist = T.train(net, sc_tr, sc_va, tmpl, dev, ks=ks, epochs=a.epochs, lr=a.lr, bs=a.bs, seed=a.seed, mask=mask,
                   keep_best=a.keep_best, log=log)
    torch.save(dict(state=net.state_dict(), template_def=TD, template_att=TA, args=vars(a), mask=mask), f'{a.out}/model_{mask}.pt')
    T.plot_curve(hist, f'{a.out}/learning_curve_{mask}.png')
    hv = E.horizon_eval(net, sc_va, tmpl, dev)
    score = float(np.mean([hv[h]['strict']['lstm'] for h in hv if hv[h]]))
    runs[mask] = dict(net=net, hist=hist, val=hv, val_score=score, minutes=round((time.time() - t1) / 60, 1))
    log(f'    val strict LSTM error 1/2/5 s: ' + ' / '.join(f'{hv[h]["strict"]["lstm"]:.2f}' for h in hv) + f'  -> mean {score:.2f} m')

# 6. pick on val, report on test
best = min(runs, key=lambda m: runs[m]['val_score'])
log(f'[6] picked mask "{best}" on validation games')
for m in runs:
    runs[m]['test'] = E.horizon_eval(runs[m]['net'], sc_te, tmpl, dev)
vs = E.vis_stats(allsc, sc_tr, sc_va + sc_te)
net = runs[best]['net']
cand = [i for i, s in enumerate(sc_te) if s['L'] >= 50]
pick = sorted(cand, key=lambda i: -sc_te[i]['vis'].mean())[:3] or sorted(range(len(sc_te)), key=lambda i: -sc_te[i]['L'])[:3]
figs = []
for n, i in enumerate(pick):
    s = sc_te[i]
    gh = E.ghost(net, sc_te, i, tmpl, dev)
    mm, ss = divmod(int(s['time']), 60)
    p = f'{a.out}/ghost_test_scene_{n + 1}.png'
    E.plot_scene(s, gh, p, f'game {s["game"]}: {s["defend_name"]} defending, half {s["period"]} +{mm:02d}:{ss:02d}, '
                           f'{(s["L"] - 1) / 10:.1f} s, ends: {s["why"]}, VISIBLE {s["vis"].mean():.0%}  '
                           f'(ghost error at end {np.linalg.norm(gh[-1] - s["D"][-1], axis=-1).mean():.1f} m)')
    figs.append(os.path.basename(p))

# 7. phase 2: continue the picked model with all 10 defenders generated
P2 = None
if a.phase2_epochs > 0:
    import copy
    p1 = runs[best]['net']
    net2 = copy.deepcopy(p1)
    log(f'[7] phase 2 from mask "{best}" model, epochs/stage {a.phase2_epochs}, K {a.phase2_ks}')
    t2 = time.time()
    h2 = T.train(net2, sc_tr, sc_va, tmpl, dev, ks=tuple(int(k) for k in a.phase2_ks.split(',')), epochs=a.phase2_epochs,
                 lr=a.phase2_lr, bs=a.bs, seed=a.seed, mask=best, joint=True, keep_best=True, log=log)
    torch.save(dict(state=net2.state_dict(), template_def=TD, template_att=TA, args=vars(a), mask=best, phase=2),
               f'{a.out}/model_phase2.pt')
    T.plot_curve(h2, f'{a.out}/learning_curve_phase2.png')
    P2 = dict(minutes=round((time.time() - t2) / 60, 1), history=h2, results={
        'phase1_single': dict(h=runs[best]['test'], scene=E.scene_eval(p1, sc_te, tmpl, dev, joint=False)),
        'phase1_joint': dict(h=E.horizon_eval(p1, sc_te, tmpl, dev, joint=True), scene=E.scene_eval(p1, sc_te, tmpl, dev, joint=True)),
        'phase2_joint': dict(h=E.horizon_eval(net2, sc_te, tmpl, dev, joint=True), scene=E.scene_eval(net2, sc_te, tmpl, dev, joint=True))})

total_min = round((time.time() - t0) / 60, 1)
metrics = dict(split=split, per_game=per_game, excluded_total=dict(excl), template=dict(defence=TD.tolist(), attack=TA.tolist(), **rhist),
               input_dim=features.N_IN, picked_mask=best, visibility=vs, figures=figs, minutes=total_min,
               runs={m: {k: v for k, v in r.items() if k != 'net'} for m, r in runs.items()}, phase2=P2)
json.dump(metrics, open(f'{a.out}/metrics.json', 'w'), indent=1, ensure_ascii=False)

# summary
f2 = lambda x: '-' if x is None else f'{x:.2f}'
L = [f'# World Cup {len(games)} games: ghosting LSTM', '',
     f'Command: `python run_multi.py --games {a.games} --epochs {a.epochs}` ({total_min} min, {a.device}). '
     f'Data: PFF FC 2022 World Cup dataset.', '',
     '## Data split (by game)', '', '| split | games | scenes | minutes |', '|---|---|---|---|']
for k in ('train', 'val', 'test'):
    L.append(f'| {k} | {len(split[k]["games"])} | {split[k]["scenes"]} | {split[k]["minutes"]} |')
L += ['', f'Val games: {", ".join(split["val"]["games"])}. Test games: {", ".join(split["test"]["games"])}.',
      f'Excluded scenes over all games: {dict(excl)}.', '',
      '## Mask choice (validation games, strict: start and end VISIBLE)', '',
      '| mask | training time (min) | val LSTM 1 s | 2 s | 5 s | mean |', '|---|---|---|---|---|---|']
for m, r in runs.items():
    L.append(f'| {m}{" (picked)" if m == best else ""} | {r["minutes"]} | ' +
             ' | '.join(f2(r['val'][h]['strict']['lstm']) for h in r['val']) + f' | {r["val_score"]:.2f} |')
L += ['', '`next` = loss where the defender is VISIBLE at t+1 (spec). `both` = VISIBLE at t and t+1.', '',
      '## Test games (m)', '', '| horizon | eval | points | LSTM next | LSTM both | stand still | constant velocity |',
      '|---|---|---|---|---|---|---|']
for h in (10, 20, 50):
    for ev in ('strict', 'end'):
        r0 = runs[best]['test'][h]
        if not r0:
            continue
        L.append(f'| {h / 10:.0f} s | {ev} | {r0["n_" + ev]} | ' +
                 ' | '.join(f2(runs[m]['test'][h][ev]['lstm']) if m in runs else '-' for m in ('next', 'both')) +
                 f' | {f2(r0[ev]["still"])} | {f2(r0[ev]["const_vel"])} |')
L += ['', '`strict` = start and end VISIBLE, `end` = end VISIBLE only.', '',
      f'## Per role (test, picked mask "{best}", strict)', '',
      '| role | slot (centred) | VISIBLE share | 10 s before shots | targets train | ' +
      ' | '.join(f'LSTM / still / const-vel {h / 10:.0f} s' for h in (10, 20, 50)) + ' |', '|' + '---|' * 8]
for r in range(10):
    sv = vs['visible_share_shot_10s']
    cells = [f'D{r + 1}', f'{TD[r, 0]:.0f}, {TD[r, 1]:.0f}', f'{vs["visible_share_all_scenes"][r]:.0%}',
             '-' if sv is None else f'{sv[r]:.0%}', str(vs['targets_train'][r])]
    for h in (10, 20, 50):
        rr = runs[best]['test'][h]
        cells.append('-' if not rr else ' / '.join(f2(rr['per_role_strict'][k][r]) for k in ('lstm', 'still', 'const_vel')))
    L.append('| ' + ' | '.join(cells) + ' |')
L += ['', f'VISIBLE share: all scenes {vs["visible_share_all_overall"]:.0%}; 10 s before shots '
      f'{"-" if vs["visible_share_shot_10s_overall"] is None else format(vs["visible_share_shot_10s_overall"], ".0%")} '
      f'({vs["shots_in_kept_scenes"]} shots).', '']
if P2:
    pc = lambda x: '-' if x is None else f'{x:.0%}'
    lab = {'phase1_single': 'phase 1 model, team-mates real', 'phase1_joint': 'phase 1 model, all 10 generated',
           'phase2_joint': 'phase 2 model, all 10 generated'}
    R2, b0 = P2['results'], P2['results']['phase2_joint']['h']
    bst = [h for h in P2['history'] if 'best_epoch' in h][0]
    L += [f'## Phase 2 on test games (training {P2["minutes"]} min, kept epoch {bst["best_epoch"]})', '',
          '| model / evaluation | 1 s | 2 s | 5 s | within 3 m at 5 s | whole scene mean | whole scene first 10 s |',
          '|---|---|---|---|---|---|---|']
    for k, r in R2.items():
        L.append(f'| {lab[k]} | ' + ' | '.join(f2(r['h'][h]['strict']['lstm']) for h in (10, 20, 50)) +
                 f' | {pc(r["h"][50]["within_3m_strict"]["lstm"])} | {r["scene"]["lstm"]:.2f} | {r["scene"]["lstm_10s"]:.2f} |')
    sc0 = R2['phase2_joint']['scene']
    for bk, bl in (('still', 'stand still'), ('const_vel', 'constant velocity')):
        L.append(f'| baseline: {bl} | ' + ' | '.join(f2(b0[h]['strict'][bk]) for h in (10, 20, 50)) +
                 f' | {pc(b0[50]["within_3m_strict"][bk])} | ' +
                 (f'{sc0["still"]:.2f} | {sc0["still_10s"]:.2f} |' if bk == 'still' else '- | - |'))
    L.append('')
L += [f'Scene rule: training games {a.rule}, fixed val/test games {a.eval_rule}.',
      'Files: role_template.png, learning_curve_*.png, ' + ', '.join(figs) + ', scenes.csv, metrics.json, model_*.pt, run_log.txt']
open(f'{a.out}/summary.md', 'w').write('\n'.join(L) + '\n')
log('\n'.join(L))
log(f'== done in {total_min} min')
