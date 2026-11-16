"""Scene cutting: paper rule with the event file (E) vs tracking-only rule (T), on the games that have both.
Read-only on the raw data; CPU only.   python compare_rules.py
"""
import argparse, collections, glob, json, os
from multiprocessing import Pool
import numpy as np
from src import load, scenes as S

ap = argparse.ArgumentParser()
ap.add_argument('--raw', default=load.RAW)
ap.add_argument('--hold', type=float, default=1.0, help='T rule: one defending on-ball event this long (s) ends the scene')
ap.add_argument('--workers', type=int, default=2)
ap.add_argument('--out', default='outputs/rule_check')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def intervals(rows, idx, kept_only):
    out = collections.defaultdict(list)
    for r in rows:
        if kept_only and r['excluded']:
            continue
        out[r['defend']].append((idx[r['start_frame']], idx[r['end_frame']]))
    return out


def iou(x, y):
    inter = max(0, min(x[1], y[1]) - max(x[0], y[0]))
    union = max(x[1], y[1]) - min(x[0], y[0])
    return inter / union if union > 0 else float(x == y)


def one(g):
    trk = load.load_game(g, a.raw, cache=f'outputs/cache/{g}_10fps.npz')
    ev = load.load_meta(g, a.raw)[2]
    scE, rowsE, _ = S.cut_scenes(trk, ev, 'events')
    scT, rowsT, _ = S.cut_scenes(trk, ev, 'tracking', a.hold)
    n = len(trk['frame']); idx = {int(f): i for i, f in enumerate(trk['frame'])}
    res = dict(game=g, kept_min_E=sum((s['L'] - 1) / 600 for s in scE), kept_min_T=sum((s['L'] - 1) / 600 for s in scT),
               scenes_E=len(rowsE), scenes_T=len(rowsT), kept_E=len(scE), kept_T=len(scT),
               end_T=collections.Counter(s['why'] for s in scT))
    for kind, kept in (('all', False), ('kept', True)):
        IE, IT = intervals(rowsE, idx, kept), intervals(rowsT, idx, kept)
        c = collections.Counter()
        for side in set(IE) | set(IT):
            mE, mT = np.zeros(n, bool), np.zeros(n, bool)
            for s, e in IE[side]: mE[s:e + 1] = True
            for s, e in IT[side]: mT[s:e + 1] = True
            c['both'] += int((mE & mT).sum()); c['E'] += int(mE.sum()); c['T'] += int(mT.sum()); c['either'] += int((mE | mT).sum())
            startsT = {s for s, _ in IT[side]}
            for x in IE[side]:
                best = max((iou(x, y) for y in IT[side]), default=0.0)
                c['n'] += 1; c['iou80'] += best >= 0.8; c['iou50'] += best >= 0.5; c['same_start'] += x[0] in startsT
                if x[0] in startsT:
                    y = [y for y in IT[side] if y[0] == x[0]][0]
                    c['end_diff_sum'] += abs(y[1] - x[1]); c['end_same'] += y[1] == x[1]
        res[kind] = dict(c)
    # what follows a foul / offside that was given? (is it an OUT in the tracking events?)
    L, _ = S.event_list(ev, trk['info']['ge'])
    nxt = collections.Counter()
    for k, e in enumerate(L):
        f = [x for x in e['fouls'] if x.get('foulType') == 'I']
        if not f:
            continue
        kind = 'offside' if any(x.get('finalOffenseType') == 'O' for x in f) else 'foul'
        later = [x for x in L[k + 1:] if x['g'].get('gameEventType') not in ('SUB', 'ON', 'OFF')]
        t = later[0]['g'] if later else {}
        nxt[f'{kind} -> {t.get("gameEventType")}{"/" + str(t.get("outType")) if t.get("gameEventType") == "OUT" else ""}'] += 1
    res['after_foul'] = nxt
    return res


games = sorted({os.path.basename(p)[:-5] for p in glob.glob(f'{a.raw}/Event Data/*/*.json')}
               & {os.path.basename(p).split('.')[0] for p in glob.glob(f'{a.raw}/Tracking Data/*.jsonl.bz2')})
with Pool(a.workers) as pool:
    R = pool.map(one, games)

tot = {k: collections.Counter() for k in ('all', 'kept')}
end_T, after = collections.Counter(), collections.Counter()
for r in R:
    for k in tot: tot[k].update(r[k])
    end_T.update(r['end_T']); after.update(r['after_foul'])
pct = lambda x, y: f'{100 * x / max(y, 1):.0f}%'
L = [f'# Scene cutting: event-file rule (E) vs tracking-only rule (T), {len(R)} games', '',
     f'T rule: start = attacking team on the ball; end = OUT / END, or defending team has two on-ball events in a row, '
     f'or one on-ball event of at least {a.hold:.1f} s.', '',
     '| | E rule | T rule |', '|---|---|---|',
     f'| candidate scenes | {sum(r["scenes_E"] for r in R)} | {sum(r["scenes_T"] for r in R)} |',
     f'| kept scenes (after the same filters) | {sum(r["kept_E"] for r in R)} | {sum(r["kept_T"] for r in R)} |',
     f'| kept minutes | {sum(r["kept_min_E"] for r in R):.0f} | {sum(r["kept_min_T"] for r in R):.0f} |', '',
     '## Agreement', '', '| | all candidates | kept scenes |', '|---|---|---|']
for lab, f in (('defence frames in both / in either (Jaccard)', lambda c: pct(c['both'], c['either'])),
               ('E frames also in T', lambda c: pct(c['both'], c['E'])),
               ('T frames also in E', lambda c: pct(c['both'], c['T'])),
               ('E scenes with a T scene of IoU >= 0.8', lambda c: pct(c['iou80'], c['n'])),
               ('E scenes with a T scene of IoU >= 0.5', lambda c: pct(c['iou50'], c['n'])),
               ('E scenes with exactly the same start', lambda c: pct(c['same_start'], c['n'])),
               ('... of those, same end', lambda c: pct(c['end_same'], c['same_start'])),
               ('... of those, mean end difference', lambda c: f'{c["end_diff_sum"] / max(c["same_start"], 1) / 10:.2f} s')):
    L.append(f'| {lab} | {f(tot["all"])} | {f(tot["kept"])} |')
L += ['', f'T rule end reasons (kept): {dict(end_T)}', '',
      '## What comes right after a given foul / offside (event file)', '']
L += [f'- {k}: {v}' for k, v in after.most_common()]
L += ['', '## Per game (kept minutes E / T)', '']
L += [f'- {r["game"]}: {r["kept_min_E"]:.1f} / {r["kept_min_T"]:.1f}' for r in R]
open(f'{a.out}/report.md', 'w').write('\n'.join(L) + '\n')
json.dump(R, open(f'{a.out}/per_game.json', 'w'), indent=1, default=dict)
print('\n'.join(L))
