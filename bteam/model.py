"""교수님 구조 (DETR 디코더식) + 비교 기준 4번 MLP, 같은 학습 루프.

PositionDecoder (참고 구현 — 성준 담당 부분의 출발점):
  memory 토큰  : 공 1개 + 상대 필드 선수 10명 + 국면(공격/수비) 1개   [+ 우리 팀 정보, variant]
                 좌표 (x, y) -> 작은 MLP -> d차원 벡터 + 종류 임베딩 (교수님이 말한 projection)
  쿼리 10개    : 우리 포지션 10개. 각 쿼리는 그 포지션의 평균 위치에서 시작 (MLP(평균 위치) + 국면)
  블록         : self-attention(쿼리끼리) -> cross-attention(쿼리 -> 공·상대) -> FFN, num_layers 번
  출력         : 포지션마다 '평균 위치에서 얼마나 움직일지' (dx, dy)
  위치 인코딩  : 없음. 선수 순서에 의미가 없으므로 (상대 순서를 섞어도 결과가 같다)

variant (우리 팀 입력, B팀 문서 '우리 팀 입력 변형'):
  A  없음 (교수님 초안)
  B  우리 팀 중심 1개 토큰 (팀이 지금 어디 있는지만)
  C  우리 선수 토큰 10개 중 예측 대상 1명을 가림 -> 나머지 9명을 보고 가린 1명의 자리를 맞힘
     (학습: 장면마다 무작위 1명 가림, 손실은 그 1명만. 평가: 10번 돌며 한 명씩 가림)
"""
import copy, time
import numpy as np
import torch
import torch.nn as nn
from data import SCALE, flat_features

TOK_BALL, TOK_OPP, TOK_OWN, TOK_CENTRE = range(4)


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.GELU()]; d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class PositionDecoder(nn.Module):
    def __init__(self, variant='A', d=128, heads=4, layers=3, dropout=0.1):
        super().__init__()
        self.variant = variant
        self.coord = mlp(2, d, d)                 # one (x, y) -> token
        self.kind = nn.Embedding(4, d)
        self.phase = nn.Embedding(2, d)
        self.query = mlp(2, d, d)                 # role mean place -> query
        layer = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, layers)
        self.head = mlp(d, d, 2)

    def forward(self, b):
        ball, opp, mean, ph = b['ball'], b['opp'], b['own_mean'], b['phase']
        B = len(ball)
        opp_missing = torch.isnan(opp[..., 0])
        tok = [self.coord(ball)[:, None] + self.kind.weight[TOK_BALL],
               self.coord(torch.nan_to_num(opp)) + self.kind.weight[TOK_OPP],
               self.phase(ph)[:, None]]
        pad = [torch.zeros(B, 1, dtype=torch.bool, device=ball.device), opp_missing,
               torch.zeros(B, 1, dtype=torch.bool, device=ball.device)]
        if self.variant == 'B':
            tok.append(self.coord(b['own'].mean(1))[:, None] + self.kind.weight[TOK_CENTRE]); pad.append(pad[0])
        elif self.variant == 'C':
            tok.append(self.coord(b['own']) + self.kind.weight[TOK_OWN]); pad.append(b['hide'])
        mem = torch.cat(tok, 1); pad = torch.cat(pad, 1)
        q = self.query(mean) + self.phase(ph)[:, None]
        h = self.dec(q, mem, memory_key_padding_mask=pad)
        return mean + self.head(h)                 # (B, 10, 2) in SCALE units


class FlatMLP(nn.Module):
    """비교 기준 4: attention 없이 [공, 국면, 상대 20개 좌표, 우리 평균 자리 20개]를 이어붙여 바로 출력"""
    variant = 'A'

    def __init__(self, d=256, layers=3, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(43, d), nn.GELU(), nn.Dropout(dropout),
                                 *[m for _ in range(layers - 2) for m in (nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))],
                                 nn.Linear(d, 20))

    def forward(self, b):
        return b['own_mean'] + self.net(b['flat']).view(-1, 10, 2)


# ------------------------------------------------------------------ data -> tensors
def to_tensors(D, device):
    t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=device)
    return {'ball': t(D['ball'] / SCALE), 'opp': t(D['opp'] / SCALE), 'own': t(D['own'] / SCALE),
            'own_mean': t(D['own_mean'] / SCALE), 'phase': t(D['phase'], torch.long), 'flat': t(flat_features(D))}


def batch(T, idx, variant, gen=None, hide_role=None):
    b = {k: v[idx] for k, v in T.items()}
    if variant == 'C':
        n = len(idx); dev = b['ball'].device
        r = (torch.randint(0, 10, (n,), generator=gen, device='cpu').to(dev) if hide_role is None
             else torch.full((n,), hide_role, device=dev))
        b['hide'] = torch.nn.functional.one_hot(r, 10).bool()
    return b


def loss_of(model, b):
    pred = model(b)
    err = (pred - b['own']).abs().sum(-1)            # L1 per role, SCALE units
    if model.variant == 'C':
        return (err * b['hide']).sum() / b['hide'].sum()
    return err.mean()


@torch.no_grad()
def predict(model, T, bs=4096):
    """(N, 10, 2) metres. variant C: role k predicted with role k hidden"""
    model.eval()
    n = len(T['ball']); out = []
    for s in range(0, n, bs):
        idx = torch.arange(s, min(s + bs, n), device=T['ball'].device)
        if model.variant == 'C':
            p = torch.stack([model(batch(T, idx, 'C', hide_role=k))[:, k] for k in range(10)], 1)
        else:
            p = model(batch(T, idx, model.variant))
        out.append(p.cpu())
    return torch.cat(out).numpy() * SCALE


def val_error_m(model, T):
    pred = predict(model, T)
    own = T['own'].cpu().numpy() * SCALE
    return float(np.hypot(*(pred - own).transpose(2, 0, 1)).mean())


def train(model, Ttr, Tva, epochs=30, bs=256, lr=1e-3, wd=1e-4, patience=6, seed=0, log=print):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    dev = Ttr['ball'].device
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(Ttr['ball']); best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0.0
        perm = torch.randperm(n, generator=gen).to(dev)
        for s in range(0, n, bs):
            b = batch(Ttr, perm[s:s + bs], model.variant, gen)
            loss = loss_of(model, b)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss) * len(perm[s:s + bs])
        sched.step()
        v = val_error_m(model, Tva)
        log(f'  epoch {ep + 1:3d}  train L1 {tot / n * SCALE:6.2f} m  val error {v:5.2f} m  [{time.time() - t0:.0f}s]')
        if v < best - 1e-3:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                log('  early stop'); break
    model.load_state_dict(best_state)
    return model, best
