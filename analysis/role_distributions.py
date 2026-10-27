"""Bialkowski et al. (ICDM 2014) role distributions on broadcast (partially visible) players.

Each role k is a 2D Gaussian N(mu_k, Sigma_k) in team-relative coordinates.
Matching cost: 'euclid' (default, same as pipeline/manutd_v2.py: distance with a 15 m gate),
'shared' (one pooled isotropic Gaussian), 'full' (each role's own Sigma_k, as in the paper; with only
4-8 visible players a wide role absorbs far-away players, so it is unstable on broadcast data).
Hard-assignment EM has several local optima on broadcast data -> use best_fit() (restarts, optional warm start).
EM with hard assignment (the paper's minimum-entropy formulation):
  E-step  per frame: estimate the team shift s from the visible players, then assign
          visible players to roles with the Hungarian algorithm on cost -log N(p - s; mu_k, Sigma_k)
          (rectangular: k visible players <= 10 roles).
  M-step  re-estimate mu_k, Sigma_k from the assigned relative positions.
Broadcast change vs. the paper: the paper subtracts the mean of all 10 players; with only
some players visible that mean is biased, so s is the median of (p - mu_assigned), refined
a few times per frame.

Input : viewer json from pipeline/manutd_v2.py  {frames: [{f, "0": {p: [[x,y],...]}, "1": ...}]}
Usage : python analysis/role_distributions.py manutd_viewer.json roles.json
"""
import json, sys
import numpy as np
from scipy.optimize import linear_sum_assignment

K = 10
REG = 0.5          # m^2 added to each covariance (keeps tiny roles from collapsing)


def team_frames(d, team, flip=1.0, min_vis=3):
    """list of (frame_id, visible points[k,2]) for one team; flip=-1 rotates the pitch 180 deg"""
    out = []
    for fr in d['frames']:
        x = fr.get(str(team))
        if not x or len(x['p']) < min_vis:
            continue
        out.append((fr['f'], np.asarray(x['p'], float)[:, :2] * flip))
    return out


def _cost(R, mu, icov, logdet, gate=None):
    """negative log-likelihood (up to a constant) of relative points R[k,2] under each role;
    icov=None -> plain Euclidean distance with a hard gate (pipeline/manutd_v2.py behaviour)"""
    if icov is None:
        C = np.linalg.norm(R[:, None, :] - mu[None], axis=2)
        return np.where(C > gate, 1e6, C)
    D = R[:, None, :] - mu[None]                     # k x K x 2
    m = np.einsum('ikx,kxy,iky->ik', D, icov, D)
    return 0.5 * (m + logdet[None])


def _match_from(P, mu, icov, logdet, gate, inner, s):
    for _ in range(inner):
        C = _cost(P - s, mu, icov, logdet, gate)
        ri, ci = linear_sum_assignment(C)
        if icov is None:
            keep = C[ri, ci] < 1e5; ri, ci = (ri[keep], ci[keep]) if keep.sum() >= 2 else (ri, ci)
        s = np.median(P[ri] - mu[ci], axis=0)
    C = _cost(P - s, mu, icov, logdet, gate)
    ri, ci = linear_sum_assignment(C)
    # gate: a player > gate metres from the role it was forced onto is treated as noise
    # (wrong team, projection error at the image edge) and left out of the M-step
    ok = np.hypot(*(P[ri] - s - mu[ci]).T) <= gate
    cost = np.minimum(C[ri, ci], gate).sum() + gate * (~ok).sum()
    return s, ri[ok], ci[ok], cost


SMOOTH = 0.5   # cost per metre the team shift jumps from the previous frame (0.5 s earlier at 2 fps)


def match_frame(P, mu, icov=None, logdet=None, gate=15.0, inner=3, s_prev=None):
    """one frame: team shift s (median of p - mu over matched pairs) and the gated player->role pairs.
    A partial view is often ambiguous (4 defenders look like a midfield line shifted back), so when the
    previous frame's shift is known we also start from it and prefer the answer that does not jump.
    returns s, player idx, role idx, cost"""
    best = _match_from(P, mu, icov, logdet, gate, inner, P.mean(0) - mu.mean(0))
    if s_prev is None:
        return best
    cand = _match_from(P, mu, icov, logdet, gate, inner, np.asarray(s_prev, float))
    score = lambda r: r[3] + SMOOTH * float(np.hypot(*(r[0] - s_prev)))
    return min((best, cand), key=score)


def impute(P, mu, gate=15.0, s_prev=None):
    """off-screen players: every role nobody visible was matched to is placed at mu_k + s.
    returns {role: (x, y)} for the missing roles, the shift s and the visible pairs"""
    s, ri, ci, _ = match_frame(np.asarray(P, float), mu, gate=gate, s_prev=s_prev)
    miss = [k for k in range(len(mu)) if k not in set(ci.tolist())]
    return {k: mu[k] + s for k in miss}, s, (ri, ci)


def impute_track(last_pos, last_shift, last_role, gone_s, s_now, mu, tau=5.0):
    """off-screen player whose identity is kept by the tracker (e.g. CAMELTrack):
    blend 'where he was, moved with the team' and 'his role's usual place', trusting the first
    less the longer he has been off-screen (weight exp(-gone/tau))."""
    seen = np.asarray(last_pos, float) + (np.asarray(s_now, float) - np.asarray(last_shift, float))
    role = np.asarray(mu[last_role], float) + np.asarray(s_now, float)
    w = float(np.exp(-gone_s / tau))
    return w * seen + (1 - w) * role


def e_step(frames, mu, cov, inner=3, cost='euclid', gate=15.0, smooth=False):
    if cost == 'euclid':
        icov = logdet = None
    elif cost == 'shared':
        # one pooled isotropic covariance for all roles: cost = squared distance / (2 s^2).
        # With only some players visible, per-role full covariances let a wide role "absorb"
        # far-away players (rich get richer); the shared cost keeps the 1:1 matching balanced.
        s2 = float(np.mean([np.trace(c) / 2 for c in cov]))
        icov = np.repeat(np.eye(2)[None] / s2, K, axis=0); logdet = np.zeros(K)
    elif cost == 'full':
        icov = np.linalg.inv(cov); logdet = np.linalg.slogdet(cov)[1]
    rel, role, shifts, nll = [], [], [], 0.0
    pf, ps = None, None
    for f, P in frames:
        sp = ps if (smooth and pf is not None and f - pf <= 2) else None
        s, ri, ci, c = match_frame(P, mu, icov, logdet, gate, inner, s_prev=sp)
        pf, ps = f, s
        rel.append(P[ri] - s); role.append(ci); shifts.append(s); nll += c
    return np.concatenate(rel), np.concatenate(role), np.array(shifts), nll


def m_step(rel, role, mu_old, cov_old):
    mu, cov = mu_old.copy(), cov_old.copy()
    for k in range(K):
        X = rel[role == k]
        if len(X) >= 5:
            mu[k] = X.mean(0)
            cov[k] = np.cov(X.T) + REG * np.eye(2)
    cov = np.where(np.isfinite(cov), cov, cov_old)
    c = mu.mean(0)                                   # gauge: roles centred on the team centre
    return mu - c, cov


def init_roles(frames, mode='tangled', seed=0):
    """'tangled': all roles start near the team centre with the global spread (paper Fig. 'initial roles')
       'kmeans' : k-means on mean-subtracted points (faster, safer)"""
    if isinstance(mode, tuple):                      # warm start from given (mu, cov)
        mu, cov = (np.array(a, float) for a in mode)
        return mu - mu.mean(0), cov
    rng = np.random.default_rng(seed)
    R = np.concatenate([P - P.mean(0) for _, P in frames])
    if mode == 'kmeans':
        mu = R[rng.choice(len(R), K, replace=False)]
        for _ in range(30):
            lab = np.argmin(((R[:, None] - mu[None]) ** 2).sum(-1), 1)
            mu = np.array([R[lab == k].mean(0) if (lab == k).any() else mu[k] for k in range(K)])
        cov = np.array([np.cov(R[lab == k].T) + REG * np.eye(2) if (lab == k).sum() >= 5 else np.cov(R.T) * 0.2 for k in range(K)])
    else:
        g = np.cov(R.T)
        mu = rng.multivariate_normal(np.zeros(2), g * 0.15, size=K)
        cov = np.repeat(g[None] * 0.6 + REG * np.eye(2), K, axis=0)
    return mu - mu.mean(0), cov


def entropy(cov):
    """sum of Gaussian entropies (the paper minimises this)"""
    return float(sum(np.log(2 * np.pi * np.e) + 0.5 * np.linalg.slogdet(c)[1] for c in cov))


def fit(frames, init='tangled', seed=0, max_iter=60, tol=0.01, keep_history=False, cost='euclid', gate=15.0, smooth=False):
    mu, cov = init_roles(frames, init, seed)
    hist = [(mu.copy(), cov.copy(), entropy(cov))]
    for it in range(1, max_iter + 1):
        rel, role, shifts, nll = e_step(frames, mu, cov, cost=cost, gate=gate, smooth=smooth)
        mu_new, cov = m_step(rel, role, mu, cov)
        diff = float(np.abs(mu_new - mu).max()); mu = mu_new
        if keep_history:
            hist.append((mu.copy(), cov.copy(), entropy(cov)))
        if diff < tol:
            break
    return {'mu': mu, 'cov': cov, 'iters': it, 'entropy': entropy(cov), 'nll': nll,
            'history': hist, 'role': role, 'rel': rel, 'shifts': shifts}


def best_fit(frames, restarts=3, warm=None, **kw):
    """hard-assignment EM has local minima: keep the run with the lowest final cost.
    warm=(mu, cov) adds a run started from a previous solution (e.g. the pipeline's slots)"""
    runs = [fit(frames, init=('kmeans' if s == 0 else 'tangled'), seed=s, **kw) for s in range(restarts)]
    if warm is not None:
        runs.append(fit(frames, init=warm, **kw))
    return min(runs, key=lambda r: r['nll'])


B4 = [('LB', -15, -24), ('LCB', -18, -8), ('RCB', -18, 8), ('RB', -15, 24)]
FORMS = {   # x: towards the opponent goal, y: left(-) to right(+), metres around the team centre
    '4-2-3-1': B4 + [('LDM', -6, -7), ('RDM', -6, 7), ('LAM', 6, -20), ('CAM', 8, 0), ('RAM', 6, 20), ('ST', 18, 0)],
    '4-4-2':   B4 + [('LM', -2, -22), ('LCM', -4, -7), ('RCM', -4, 7), ('RM', -2, 22), ('LST', 14, -6), ('RST', 14, 6)],
    '4-3-3':   B4 + [('LCM', -4, -10), ('CM', -6, 0), ('RCM', -4, 10), ('LW', 12, -20), ('ST', 16, 0), ('RW', 12, 20)],
    '4-1-4-1': B4 + [('DM', -8, 0), ('LM', 2, -22), ('LCM', 0, -8), ('RCM', 0, 8), ('RM', 2, 22), ('ST', 16, 0)],
    '3-5-2':   [('LCB', -18, -14), ('CB', -20, 0), ('RCB', -18, 14), ('LWB', -4, -26), ('LCM', -4, -9), ('CM', -8, 0),
                ('RCM', -4, 9), ('RWB', -4, 26), ('LST', 14, -6), ('RST', 14, 6)],
}


def name_roles(mu, form, method='icp'):
    """match the 10 roles to a formation template; returns a name per role.
    'icp'   : fit per-axis scale + offset of the template and the Hungarian matching alternately (default)
    'zscore': per-axis z-score both, one Hungarian (what analysis/build.py does)"""
    tp = FORMS[form]
    T = np.array([[x, y] for _, x, y in tp], float); T -= T.mean(0)
    M = np.asarray(mu, float); M = M - M.mean(0)
    if method == 'zscore':
        nz = lambda A: A / A.std(0)
        _, col = linear_sum_assignment(np.linalg.norm(nz(M)[:, None] - nz(T)[None], axis=2))
        return [tp[c][0] for c in col]
    sc, tr = M.std(0) / T.std(0), np.zeros(2)
    for _ in range(15):
        r, col = linear_sum_assignment(np.linalg.norm(M[:, None] - (T * sc + tr)[None], axis=2))
        for a in range(2):
            A = np.c_[T[col, a], np.ones(len(col))]
            (sc[a], tr[a]), *_ = np.linalg.lstsq(A, M[r, a], rcond=None)
        sc = np.clip(sc, 0.2, 3.0)
    return [tp[c][0] for c in col]


def name_confidence(mu, form, noise=1.5, n=200, seed=0):
    """how often each role keeps its name when every role centre is jittered by ~noise metres
    (about the spread between equally good EM solutions on broadcast data). returns names, share 0..1"""
    rng = np.random.default_rng(seed)
    base = name_roles(mu, form)
    hits = np.zeros(len(mu))
    for _ in range(n):
        nm = name_roles(np.asarray(mu) + rng.normal(0, noise, np.shape(mu)), form)
        hits += [a == b for a, b in zip(nm, base)]
    return base, hits / n


def match_error(mu_a, mu_b):
    """mean distance (m) between two role sets after optimal 1:1 matching (both are centred)"""
    C = np.linalg.norm(mu_a[:, None] - mu_b[None], axis=2)
    r, c = linear_sum_assignment(C)
    return float(C[r, c].mean()), c


if __name__ == '__main__':
    d = json.load(open(sys.argv[1]))
    out = {}
    for t, fl in (('0', 1.0), ('1', -1.0)):
        fr = team_frames(d, t, fl)
        r = best_fit(fr)
        out[t] = {'mu': np.round(r['mu'], 2).tolist(), 'cov': np.round(r['cov'], 3).tolist(),
                  'iters': r['iters'], 'entropy': round(r['entropy'], 3), 'frames': len(fr)}
        print(f"team {t}: {len(fr)} frames, {r['iters']} iterations, entropy {r['entropy']:.2f}")
    json.dump(out, open(sys.argv[2], 'w'), indent=1)
