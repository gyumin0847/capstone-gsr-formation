"""Load the npz scenes written by prepare_idsse.py and build model inputs.

All coordinates stay in metres here; models divide by SCALE themselves.
A scene is used when the ball is known and all 10 of our outfield players are on the pitch
(the target). Missing opponents (red card) are kept as NaN and masked / filled per model.
"""
import glob, os
import numpy as np
import situations as SIT

SCALE = 52.5            # half pitch length: x and y divided by this (doc: "52.5로 나눠 -1~1로")


def load(data_dir, matches=None):
    """returns a dict of stacked arrays over all files (optionally only some matches)"""
    files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    if not files:
        raise SystemExit(f'no npz files in {data_dir} (run prepare_idsse.py or make_synthetic.py first)')
    cols = {k: [] for k in ('ball', 'own', 'opp', 'own_mean', 'opp_mean', 'phase', 't', 'match', 'group', 'since_change')}
    for gi, f in enumerate(files):
        d = np.load(f)
        mid = str(d['match'])
        if matches is not None and mid not in matches:
            continue
        ok = np.isfinite(d['own'][..., 0]).all(1) & np.isfinite(d['ball'][:, 0])
        sc = SIT.since_change(d['phase'], d['t'])          # before filtering: needs the full time line
        n = int(ok.sum())
        if n == 0:
            continue
        cols['ball'].append(d['ball'][ok]); cols['own'].append(d['own'][ok]); cols['opp'].append(d['opp'][ok])
        cols['own_mean'].append(np.repeat(d['own_mean'][None], n, 0))
        cols['opp_mean'].append(np.repeat(d['opp_mean'][None], n, 0))
        cols['phase'].append(d['phase'][ok]); cols['t'].append(d['t'][ok])
        cols['match'].append(np.array([mid] * n)); cols['group'].append(np.full(n, gi))
        cols['since_change'].append(sc[ok])
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out['opp_filled'] = fill_opp(out)
    return SIT.label(out)


def fill_opp(D):
    """fixed-size opponent vector for kNN / MLP: a missing opponent is put at his role mean + team shift"""
    opp = D['opp'].copy()
    miss = ~np.isfinite(opp[..., 0])
    if miss.any():
        shift = np.nanmedian(opp - D['opp_mean'], axis=1)               # (N, 2)
        fill = D['opp_mean'] + shift[:, None]
        opp[miss] = fill[miss]
    return opp


def subset(D, mask):
    return {k: v[mask] for k, v in D.items()}


def split(D, test, val):
    """match-level split: scenes of one match never appear in two parts"""
    m = D['match']
    te = np.isin(m, test); va = np.isin(m, val); tr = ~(te | va)
    return subset(D, tr), subset(D, va), subset(D, te)


def flat_features(D):
    """ball (2) + phase (1) + opponents in their role order (20) + our role means (20) = 43, in SCALE units"""
    n = len(D['ball'])
    return np.concatenate([D['ball'] / SCALE, D['phase'][:, None].astype(np.float32),
                           D['opp_filled'].reshape(n, -1) / SCALE,
                           D['own_mean'].reshape(n, -1) / SCALE], 1).astype(np.float32)


def errors(pred, D):
    """Euclidean error (m) per scene and role: (N, 10)"""
    return np.hypot(*(pred - D['own']).transpose(2, 0, 1))


def summary(err, D):
    ph = D['phase'].astype(bool)
    return {'all_m': round(float(err.mean()), 2),
            'attack_m': round(float(err[ph].mean()), 2) if ph.any() else None,
            'defence_m': round(float(err[~ph].mean()), 2) if (~ph).any() else None,
            'per_role_m': [round(float(v), 2) for v in err.mean(0)],
            'median_m': round(float(np.median(err)), 2), 'scenes': int(len(err))}
