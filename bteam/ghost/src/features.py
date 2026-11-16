"""Batches of scene windows and the per-role input vector (built on the fly so rollout can swap in predictions)."""
import numpy as np
import torch
from .load import DT

GOAL = (-52.5, 0.0)     # defended goal centre after normalisation
IDX9 = [[j for j in range(10) if j != r] for r in range(10)]
PARTS = [('own position + velocity', 4), ('9 team-mates position + velocity (role order)', 36),
         ('11 attackers position + velocity (template order)', 44), ('2 goalkeepers position', 4),
         ('ball position + velocity', 4), ('to own slot: distance, cos, sin', 3), ('to goal centre: distance, cos, sin', 3),
         ('to ball: distance, cos, sin', 3), ('3 nearest attackers relative position', 6)]
N_IN = sum(n for _, n in PARTS)     # 107


def make_batch(scenes, wins, device):
    """wins: list of (scene index, start frame, length). Left-aligned and zero-padded."""
    B, Lm = len(wins), max(n for _, _, n in wins)
    z = lambda *s: np.zeros((B, Lm) + s, np.float32)
    D, Dv, A, Av, GK, ball, bv = z(10, 2), z(10, 2), z(11, 2), z(11, 2), z(4), z(2), z(2)
    vis = np.zeros((B, Lm, 10), bool); valid = np.zeros((B, Lm), bool)
    prev0 = np.zeros((B, 10, 2), np.float32)
    for k, (i, a, n) in enumerate(wins):
        s, sl = scenes[i], slice(a, a + n)
        D[k, :n], Dv[k, :n], A[k, :n], Av[k, :n] = s['D'][sl], s['Dv'][sl], s['A'][sl], s['Av'][sl]
        GK[k, :n] = np.concatenate([s['Dgk'][sl], s['Agk'][sl]], -1)
        ball[k, :n], bv[k, :n] = s['ball'][sl], s['ballv'][sl]
        vis[k, :n], valid[k, :n] = s['vis'][sl], True
        prev0[k] = s['D'][a - 1] if a > 0 else s['D'][0]
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(device)
    return dict(D=t(D), A=t(A), ball=t(ball), vis=t(vis), valid=t(valid), Dprev0=t(prev0),
                Dpv=t(np.concatenate([D / 50, Dv / 10], -1)),
                Apv=t(np.concatenate([A / 50, Av / 10], -1).reshape(B, Lm, 44)),
                GK=t(GK / 50), Bpv=t(np.concatenate([ball / 50, bv / 10], -1)))


def take(b, idx):
    return {k: v[idx] for k, v in b.items()}


def _polar(vec):
    d = vec.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    return torch.cat([d / 50, vec / d], -1)


def feats(b, t, own, prev, tmpl, mates=None):
    """Input for all 10 roles at time t. own/prev: [B,10,2] own positions at t and t-1
    (real or predicted). mates=None: team-mates from the real data (phase 1).
    mates=(pos, prev) [B,10,2]: team-mates from these, i.e. their predictions (phase 2).
    Attackers and the ball always come from the real data. Returns [B,10,N_IN]."""
    B = own.shape[0]
    idx9 = torch.as_tensor(IDX9, device=own.device)
    rep = lambda x: x[:, None].expand(-1, 10, -1)
    if mates is None:
        mpos, mpv = b['D'][:, t], b['Dpv'][:, t]
    else:
        mpos = mates[0]
        mpv = torch.cat([mpos / 50, (mpos - mates[1]) / DT / 10], -1)
    mates_f = mpv[:, idx9].reshape(B, 10, 36)
    rel = b['A'][:, t, None] - own[:, :, None]                       # [B,10,11,2]
    k = rel.norm(dim=-1).topk(3, dim=-1, largest=False).indices
    near = rel.gather(2, k[..., None].expand(-1, -1, -1, 2)).reshape(B, 10, 6)
    goal = torch.tensor(GOAL, device=own.device)
    # own slot on the pitch: centred template slot placed at the real centre of the 9 team-mates
    # (the role's own real position is left out so it cannot leak into a rollout)
    slot = tmpl + mpos[:, idx9].mean(2) - tmpl[idx9].mean(1)
    x = torch.cat([own / 50, (own - prev) / DT / 10, mates_f, rep(b['Apv'][:, t]), rep(b['GK'][:, t]),
                   rep(b['Bpv'][:, t]), _polar(slot - own), _polar(goal - own),
                   _polar(b['ball'][:, t, None] - own), near / 50], -1)
    return x
