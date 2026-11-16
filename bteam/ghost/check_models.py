"""Read-only checks of the trained models (no training). Compares the 23-game and 64-game runs on the same
test scenes, and on 3 games both runs trained on (generalisation gap).
   python check_models.py            -> outputs/model_check/report.md, report.json
"""
import argparse, glob, json, os, time
import numpy as np
import torch
from src import load, scenes as S, roles, model as M
from src.features import make_batch
from src.model import rollout

ap = argparse.ArgumentParser()
ap.add_argument('--raw', default=load.RAW)
ap.add_argument('--test', default='10513,3816,3817')
ap.add_argument('--evalset', default='10502,10512,10516,10513,3816,3817')
ap.add_argument('--n-train-games', type=int, default=3)
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
ap.add_argument('--boot', type=int, default=2000)
ap.add_argument('--out', default='outputs/model_check')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.set_num_threads(2)
dev = torch.device(a.device)
t0 = time.time()

MODELS = {  # name: (path, joint)
    '23g_p1': ('outputs/wc_events/model_both.pt', False),
    '23g_p2': ('outputs/wc_events_phase2/model_phase2.pt', True),
    '64g_p1': ('outputs/wc_all64/model_both.pt', False),
    '64g_p2': ('outputs/wc_all64/model_phase2.pt', True),
}


def load_model(p):
    ck = torch.load(p, map_location='cpu', weights_only=False)
    net = M.RoleLSTMs(hidden=ck['state']['wh.0'].shape[1]); net.load_state_dict(ck['state']); net.to(dev).eval()
    return net, np.asarray(ck['template_def']), np.asarray(ck['template_att'])


def scenes_of(games):
    out = []
    for g in games:
        trk = load.load_game(g, a.raw, cache=f'outputs/cache/{g}_10fps.npz')
        out += S.cut_scenes(trk, load.load_meta(g, a.raw)[2], 'events')[0]
    return out


@torch.no_grad()
def horizon(net, sc, tmpl, joint, hs=(5, 10, 15, 20, 30, 40, 50), every=10, bs=256):
    """Per start point: errors of the model and baselines h steps later, strict (start, start-1, end VISIBLE).
    Returns {h: dict(scene, lstm, still, cv, team, ball_d)} of flat arrays (one entry per defender-start)."""
    res = {}
    for h in hs:
        smp = [(i, s0) for i, s in enumerate(sc) for s0 in range(1, s['L'] - h, every)]
        acc = {k: [] for k in ('scene', 'lstm', 'still', 'cv', 'team', 'att', 'ball_d', 'shot')}
        for c in range(0, len(smp), bs):
            ch = smp[c:c + bs]
            b = make_batch(sc, [(i, 0, s0 + h + 1) for i, s0 in ch], dev)
            L = b['D'].shape[1]
            s0 = torch.tensor([s for _, s in ch], device=dev); k = torch.arange(len(ch), device=dev)
            pred = rollout(net, b, tmpl, torch.arange(L, device=dev)[None] <= s0[:, None], joint)
            D = b['D']; true, p0, pm1 = D[k, s0 + h], D[k, s0], D[k, s0 - 1]
            # team baseline: move with the other 9 real defenders (fair only for phase 1, which sees them)
            tot0, tot1 = D[k, s0].sum(1, keepdim=True), D[k, s0 + h].sum(1, keepdim=True)
            team = p0 + ((tot1 - true) - (tot0 - p0)) / 9
            # attacker baseline: move with the attacking team's centre (uses only what phase 2 sees)
            att = p0 + (b['A'][k, s0 + h].mean(1) - b['A'][k, s0].mean(1))[:, None]
            m = b['vis'][k, s0 + h] & b['vis'][k, s0] & b['vis'][k, s0 - 1]
            e = lambda x: (x - true).norm(dim=-1)[m].cpu().numpy()
            acc['lstm'].append(e(pred[k, s0 + h - 1])); acc['still'].append(e(p0))
            acc['cv'].append(e(p0 + h * (p0 - pm1))); acc['team'].append(e(team)); acc['att'].append(e(att))
            acc['ball_d'].append((true - b['ball'][k, s0 + h][:, None]).norm(dim=-1)[m].cpu().numpy())
            sid = torch.tensor([i for i, _ in ch], device=dev)[:, None].expand(-1, 10)
            acc['scene'].append(sid[m].cpu().numpy())
            near = np.array([any(0 <= sh - (st + h) <= 100 for sh in sc[i]['shots']) for i, st in ch])
            acc['shot'].append(np.repeat(near[:, None], 10, 1)[m.cpu().numpy()])
        res[h] = {k: np.concatenate(v) for k, v in acc.items()}
    return res


@torch.no_grad()
def whole(net, sc, tmpl, joint, bs=32):
    """Paper-style whole scene from the first frame. Per scene: mean error over VISIBLE frames, and per frame
    errors with time-in-scene, plus ghost / real speed for a plausibility check."""
    order = sorted(range(len(sc)), key=lambda i: sc[i]['L'])
    per_scene = np.zeros(len(sc)); per_scene_att = np.zeros(len(sc)); tt, ee = [], []
    spd_g, spd_r = [], []
    for c in range(0, len(order), bs):
        idx = order[c:c + bs]
        b = make_batch(sc, [(i, 0, sc[i]['L']) for i in idx], dev)
        L = b['D'].shape[1]
        ur = torch.zeros(len(idx), L, dtype=torch.bool, device=dev); ur[:, 0] = True
        pred = rollout(net, b, tmpl, ur, joint); true = b['D'][:, 1:]
        m = b['vis'][:, 1:] & b['valid'][:, 1:, None]
        e = (pred - true).norm(dim=-1)
        attb = b['D'][:, :1] + (b['A'][:, 1:].mean(2) - b['A'][:, :1].mean(2))[:, :, None]
        ea = (attb - true).norm(dim=-1)
        full = torch.cat([b['D'][:, :1], pred], 1)
        v = (full[:, 1:] - full[:, :-1]).norm(dim=-1) / 0.1
        vr = (b['D'][:, 1:] - b['D'][:, :-1]).norm(dim=-1) / 0.1
        mv = m & b['vis'][:, :-1]
        for j, i in enumerate(idx):
            per_scene[i] = float(e[j][m[j]].mean()) if m[j].any() else np.nan
            per_scene_att[i] = float(ea[j][m[j]].mean()) if m[j].any() else np.nan
        tgrid = torch.arange(1, L, device=dev)[None, :, None].expand_as(e)
        tt.append(tgrid[m].cpu().numpy()); ee.append(e[m].cpu().numpy())
        vm = b['valid'][:, 1:, None].expand_as(v)
        spd_g.append(v[vm].cpu().numpy()); spd_r.append(vr[mv].cpu().numpy())
    return dict(per_scene=per_scene, per_scene_att=per_scene_att, t=np.concatenate(tt), e=np.concatenate(ee),
                spd_g=np.concatenate(spd_g), spd_r=np.concatenate(spd_r))


def scene_mean(scene_ids, x, n):
    s = np.bincount(scene_ids, x, n); c = np.bincount(scene_ids, None, n)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan), c


def boot_diff(xa, xb, w, B, rng):
    """Paired bootstrap over scenes of the weighted mean difference xb - xa. Returns (diff, lo, hi)."""
    ok = ~np.isnan(xa) & ~np.isnan(xb) & (w > 0)
    xa, xb, w = xa[ok], xb[ok], w[ok]
    d = lambda ix: np.sum((xb[ix] - xa[ix]) * w[ix]) / np.sum(w[ix])
    full = d(np.arange(len(xa)))
    bs = [d(rng.integers(0, len(xa), len(xa))) for _ in range(B)]
    return float(full), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ---------------------------------------------------------------- data
test_games = a.test.split(',')
evalset = set(a.evalset.split(','))
event_games = sorted({os.path.basename(p)[:-5] for p in glob.glob(f'{a.raw}/Event Data/*/*.json')})
train_games = [g for g in event_games if g not in evalset][:a.n_train_games]   # in both training sets
SC = {'test': scenes_of(test_games), 'train': scenes_of(train_games)}
game_of = {k: np.array([s['game'] for s in v]) for k, v in SC.items()}
print(f'scenes: test {len(SC["test"])} ({test_games}), train-check {len(SC["train"])} ({train_games})', flush=True)

R = {}
for name, (p, joint) in MODELS.items():
    net, TD, TA = load_model(p)
    tmpl = torch.tensor(TD, device=dev)
    R[name] = {}
    for split, sc in SC.items():
        roles.apply_roles(sc, TD, TA)
        R[name][split] = dict(h=horizon(net, sc, tmpl, joint), w=whole(net, sc, tmpl, joint))
    print(f'{name} done {(time.time() - t0) / 60:.1f} min', flush=True)

# ---------------------------------------------------------------- report
rng = np.random.default_rng(0)
J, L = {}, ['# Model check: 23-game vs 64-game runs (read-only, no training)', '',
            f'Test games {test_games} ({len(SC["test"])} scenes, event rule). Train-check games {train_games} '
            f'({len(SC["train"])} scenes): in the training set of both runs. Errors in m, strict (start, start-1, end VISIBLE).', '']
f2 = lambda x: f'{x:.2f}'

# 1. horizon curve, test
L += ['## 1. Error vs look-ahead (test)', '',
      '| model | ' + ' | '.join(f'{h / 10:g} s' for h in R['64g_p1']['test']['h']) + ' |', '|---|' + '---|' * 7]
for name in MODELS:
    H = R[name]['test']['h']
    L.append(f'| {name} | ' + ' | '.join(f2(H[h]['lstm'].mean()) for h in H) + ' |')
H = R['64g_p1']['test']['h']
for k, lab in (('cv', 'constant velocity'), ('still', 'stand still'), ('team', 'move with real team-mates (phase-1 fair)'),
               ('att', 'move with attackers centre (phase-2 fair)')):
    L.append(f'| {lab} | ' + ' | '.join(f2(H[h][k].mean()) for h in H) + ' |')
J['horizon_test'] = {n: {h: float(R[n]['test']['h'][h]['lstm'].mean()) for h in H} for n in MODELS}
J['horizon_test_baselines'] = {k: {h: float(H[h][k].mean()) for h in H} for k in ('cv', 'still', 'team', 'att')}

# 2. is 64 better than 23? paired bootstrap over test scenes
n = len(SC['test'])
L += ['', '## 2. 64 games minus 23 games, paired by test scene (95% bootstrap interval; negative = 64 better)', '',
      '| comparison | metric | 23 games | 64 games | difference [95% CI] |', '|---|---|---|---|---|']
J['boot'] = {}
for lab, x, y in (('phase 1', '23g_p1', '64g_p1'), ('phase 2 (all 10 generated)', '23g_p2', '64g_p2')):
    hx, hy = R[x]['test']['h'][50], R[y]['test']['h'][50]
    mx, cx = scene_mean(hx['scene'], hx['lstm'], n); my, _ = scene_mean(hy['scene'], hy['lstm'], n)
    d = boot_diff(mx, my, cx, a.boot, rng)
    L.append(f'| {lab} | 5 s error | {f2(hx["lstm"].mean())} | {f2(hy["lstm"].mean())} | {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}] |')
    wx, wy = R[x]['test']['w'], R[y]['test']['w']
    nfr = np.array([(s['vis'][1:]).sum() for s in SC['test']], float)
    d2 = boot_diff(wx['per_scene'], wy['per_scene'], nfr, a.boot, rng)
    L.append(f'| {lab} | whole scene | {f2(np.nansum(wx["per_scene"] * nfr) / nfr.sum())} | '
             f'{f2(np.nansum(wy["per_scene"] * nfr) / nfr.sum())} | {d2[0]:+.2f} [{d2[1]:+.2f}, {d2[2]:+.2f}] |')
    J['boot'][lab] = dict(h5=d, whole=d2)

# 3. per test game
L += ['', '## 3. Per test game (5 s error / whole scene)', '', '| game | scenes | ' + ' | '.join(MODELS) + ' |',
      '|---|---|' + '---|' * len(MODELS)]
J['per_game'] = {}
for g in test_games:
    gi = np.where(game_of['test'] == g)[0]; row = []
    for name in MODELS:
        h = R[name]['test']['h'][50]; sel = np.isin(h['scene'], gi)
        w = R[name]['test']['w']['per_scene'][gi]
        nfr = np.array([SC['test'][i]['vis'][1:].sum() for i in gi], float)
        row.append(f'{f2(h["lstm"][sel].mean())} / {f2(np.nansum(w * nfr) / nfr.sum())}')
        J['per_game'].setdefault(g, {})[name] = [float(h['lstm'][sel].mean()), float(np.nansum(w * nfr) / nfr.sum())]
    L.append(f'| {g} | {len(gi)} | ' + ' | '.join(row) + ' |')

# 4. generalisation gap
L += ['', '## 4. Seen games (train) vs unseen games (test): the gap shows how much more data could still help', '',
      '| model | 5 s train | 5 s test | gap | whole scene train | whole scene test | gap |', '|---|---|---|---|---|---|---|']
J['gap'] = {}
for name in MODELS:
    tr, te = R[name]['train'], R[name]['test']
    a5, b5 = tr['h'][50]['lstm'].mean(), te['h'][50]['lstm'].mean()
    aw, bw = tr['w']['e'].mean(), te['w']['e'].mean()
    L.append(f'| {name} | {f2(a5)} | {f2(b5)} | {b5 - a5:+.2f} | {f2(aw)} | {f2(bw)} | {bw - aw:+.2f} |')
    J['gap'][name] = [float(a5), float(b5), float(aw), float(bw)]

# 5. where does it go wrong (64g models, test)
L += ['', '## 5. Where the 64-game models miss (test)', '']
h = R['64g_p1']['test']['h'][50]; h2 = R['64g_p2']['test']['h'][50]
L += ['5 s error by distance of the defender to the ball at the end point:', '',
      '| distance to ball | share | phase 1 | phase 2 | team-mates baseline | attackers baseline |', '|---|---|---|---|---|---|']
for lo, hi in ((0, 10), (10, 20), (20, 35), (35, 200)):
    s = (h['ball_d'] >= lo) & (h['ball_d'] < hi); s2 = (h2['ball_d'] >= lo) & (h2['ball_d'] < hi)
    L.append(f'| {lo}-{hi} m | {s.mean():.0%} | {f2(h["lstm"][s].mean())} | {f2(h2["lstm"][s2].mean())} | '
             f'{f2(h["team"][s].mean())} | {f2(h["att"][s].mean())} |')
L += ['', f'5 s error in the 10 s before a shot: phase 1 {f2(h["lstm"][h["shot"]].mean())} (n={int(h["shot"].sum())}), '
      f'phase 2 {f2(h2["lstm"][h2["shot"]].mean())}; other times phase 1 {f2(h["lstm"][~h["shot"]].mean())}, '
      f'phase 2 {f2(h2["lstm"][~h2["shot"]].mean())}.', '',
      f'Share of 5 s errors above 8 m: phase 1 {(h["lstm"] > 8).mean():.0%}, phase 2 {(h2["lstm"] > 8).mean():.0%}; '
      f'median phase 1 {f2(np.median(h["lstm"]))}, phase 2 {f2(np.median(h2["lstm"]))}.', '']
w = R['64g_p2']['test']['w']
L += ['Whole scene (phase 2, all 10 generated) by time since the scene started:', '',
      '| time in scene | share of frames | error | attackers-centre baseline |', '|---|---|---|---|']
wa = R['64g_p2']['test']['w']
for lo, hi in ((0, 50), (50, 100), (100, 200), (200, 400), (400, 10 ** 6)):
    s = (w['t'] > lo) & (w['t'] <= hi)
    L.append(f'| {lo / 10:g}-{hi / 10:g} s | {s.mean():.0%} | {f2(w["e"][s].mean())} | - |')
nfr = np.array([s['vis'][1:].sum() for s in SC['test']], float)
L += ['', f'Whole scene, attackers-centre baseline (everyone keeps his start offset and moves with the attackers): '
      f'{f2(np.nansum(wa["per_scene_att"] * nfr) / nfr.sum())} m vs phase 2 {f2(np.nansum(wa["per_scene"] * nfr) / nfr.sum())} m.', '']
J['by_ball'] = None

# 6. plausibility of ghost motion
L += ['## 6. Is the ghost movement human-like? (speed, m/s, test, whole scene)', '',
      '| | mean | 95th pct | 99th pct | max | share > 8 m/s |', '|---|---|---|---|---|---|']
r = R['64g_p2']['test']['w']['spd_r']
L.append(f'| real defenders (VISIBLE) | {f2(r.mean())} | {f2(np.percentile(r, 95))} | {f2(np.percentile(r, 99))} | {f2(r.max())} | {(r > 8).mean():.2%} |')
for name in ('23g_p2', '64g_p2', '64g_p1'):
    g = R[name]['test']['w']['spd_g']
    L.append(f'| ghost {name} | {f2(g.mean())} | {f2(np.percentile(g, 95))} | {f2(np.percentile(g, 99))} | {f2(g.max())} | {(g > 8).mean():.2%} |')

L += ['', f'Runtime {(time.time() - t0) / 60:.1f} min, {a.device}.']
open(f'{a.out}/report.md', 'w').write('\n'.join(L) + '\n')
json.dump(J, open(f'{a.out}/report.json', 'w'), indent=1, default=str)
print('\n'.join(L))
