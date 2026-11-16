"""10 role LSTMs (separate weights, run together with einsum), rollout and masked loss."""
import torch
import torch.nn as nn
from .features import feats, N_IN


class RoleLSTMs(nn.Module):
    """One 2-layer LSTM per role. Weights have a leading role axis so the 10 models run in one call."""

    def __init__(self, n_roles=10, n_in=N_IN, hidden=128, layers=2):
        super().__init__()
        self.R, self.H, self.nl = n_roles, hidden, layers
        k = hidden ** -0.5
        u = lambda *s: nn.Parameter(torch.empty(*s).uniform_(-k, k))
        self.wi = nn.ParameterList([u(n_roles, n_in if l == 0 else hidden, 4 * hidden) for l in range(layers)])
        self.wh = nn.ParameterList([u(n_roles, hidden, 4 * hidden) for _ in range(layers)])
        self.b = nn.ParameterList([u(n_roles, 4 * hidden) for _ in range(layers)])
        self.wo = nn.Parameter(torch.empty(n_roles, hidden, 2).uniform_(-0.1 * k, 0.1 * k))
        self.bo = nn.Parameter(torch.zeros(n_roles, 2))

    def init_state(self, B, device):
        z = torch.zeros(self.nl, B, self.R, self.H, device=device)
        return z, z.clone()

    def step(self, x, state):
        h, c = state
        hs, cs, inp = [], [], x
        for l in range(self.nl):
            g = torch.einsum('brf,rfg->brg', inp, self.wi[l]) + torch.einsum('brh,rhg->brg', h[l], self.wh[l]) + self.b[l]
            i, f, gg, o = g.chunk(4, -1)
            cl = torch.sigmoid(f + 1.0) * c[l] + torch.sigmoid(i) * torch.tanh(gg)
            hl = torch.sigmoid(o) * torch.tanh(cl)
            hs.append(hl); cs.append(cl); inp = hl
        d = torch.einsum('brh,rho->bro', inp, self.wo) + self.bo       # predicted (dx, dy) in m for the next 0.1 s
        return d, (torch.stack(hs), torch.stack(cs))


def rollout(model, b, tmpl, use_real, joint=False):
    """use_real[B,L]: at t, feed the role's real position if True, else its own previous prediction.
    joint=False (phase 1): team-mates are real. joint=True (phase 2): team-mates are the other roles'
    predictions too, so all 10 defenders are generated. Attackers and the ball are always real.
    Returns predicted positions for t+1: [B,L-1,10,2]."""
    D = b['D']
    B, L = D.shape[:2]
    state = model.init_state(B, D.device)
    out, p_hat, used_prev = [], None, None
    for t in range(L - 1):
        if t == 0:
            own, prev = D[:, 0], b['Dprev0']
        else:
            r = use_real[:, t].view(B, 1, 1)
            own = torch.where(r, D[:, t], p_hat)
            prev = torch.where(r, D[:, t - 1], used_prev)
        d, state = model.step(feats(b, t, own, prev, tmpl, (own, prev) if joint else None), state)
        p_hat, used_prev = own + d, own
        out.append(p_hat)
    return torch.stack(out, 1)


def masked_loss(pred, b, mode='next'):
    """pred - target with target = real p(t+1) - fed p(t); with K=1 this is exactly the real (dx, dy).
    mode 'next': the defender is VISIBLE at t+1.  mode 'both': VISIBLE at t and t+1."""
    err = ((pred - b['D'][:, 1:]) ** 2).sum(-1)
    m = b['vis'][:, 1:] & b['valid'][:, 1:, None]
    if mode == 'both':
        m = m & b['vis'][:, :-1]
    mask = m.float()
    return (err * mask).sum() / mask.sum().clamp(min=1), mask.sum()
