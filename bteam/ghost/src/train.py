"""Curriculum training: K = 1 -> 5 -> 10 -> 20 -> 50 rollout steps inside 5 s windows."""
import time
import numpy as np
import torch
from .features import make_batch, take
from .model import rollout, masked_loss

W = 51   # 51 frames = 50 predictions = 5 s


def windows(scenes, W=W, stride=10):
    out = []
    for i, s in enumerate(scenes):
        if s['L'] <= W:
            out.append((i, 0, s['L'])); continue
        st = list(range(0, s['L'] - W + 1, stride))
        if st[-1] != s['L'] - W:
            st.append(s['L'] - W)
        out += [(i, a, W) for a in st]
    return out


def use_real_k(b, K):
    B, L = b['D'].shape[:2]
    return (torch.arange(L, device=b['D'].device) % K == 0)[None].expand(B, -1)


@torch.no_grad()
def eval_loss(model, b, tmpl, K, bs=512, mask='next', joint=False):
    model.eval()
    tot = n = 0.0
    for c in range(0, b['D'].shape[0], bs):
        bb = take(b, slice(c, c + bs))
        loss, m = masked_loss(rollout(model, bb, tmpl, use_real_k(bb, K), joint), bb, mask)
        tot += loss.item() * m.item(); n += m.item()
    return tot / max(n, 1)


def train(model, sc_tr, sc_va, tmpl, device, ks=(1, 5, 10, 20, 50), epochs=30, lr=1e-3, bs=64, seed=0,
          mask='next', joint=False, keep_best=False, log=print):
    """joint=True trains phase 2 (all 10 defenders on their predictions). keep_best=True restores the
    weights with the lowest validation loss at K=50 at the end (guards against late overfitting)."""
    g = torch.Generator().manual_seed(seed)
    btr = make_batch(sc_tr, windows(sc_tr), device)
    bva = make_batch(sc_va, windows(sc_va, stride=25), device)
    n = btr['D'].shape[0]
    _, m_tr = masked_loss(torch.zeros_like(btr['D'][:, 1:]), btr, mask)
    log(f'mask {mask}: train windows {n}, val windows {bva["D"].shape[0]}, '
        f'train target slots kept {int(m_tr)} of {int(btr["valid"][:, 1:].sum()) * 10}')
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist, ep_all = [], 0
    best = (eval_loss(model, bva, tmpl, 50, mask=mask, joint=joint), 0,
            {k: v.detach().clone() for k, v in model.state_dict().items()}) if keep_best else None
    if keep_best:
        log(f'val(K=50{", joint" if joint else ""}) before training {best[0]:.4f} m^2')
    for K in ks:
        for ep in range(epochs):
            model.train(); t0 = time.time()
            tot = cnt = 0.0
            for idx in torch.randperm(n, generator=g).split(bs):
                bb = take(btr, idx.to(device))
                loss, m = masked_loss(rollout(model, bb, tmpl, use_real_k(bb, K), joint), bb, mask)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item() * m.item(); cnt += m.item()
            ep_all += 1
            h = dict(epoch=ep_all, K=K, train=tot / max(cnt, 1),
                     val=eval_loss(model, bva, tmpl, K, mask=mask, joint=joint),
                     val_k50=eval_loss(model, bva, tmpl, 50, mask=mask, joint=joint), sec=round(time.time() - t0, 1))
            hist.append(h)
            if keep_best and h['val_k50'] < best[0]:
                best = (h['val_k50'], ep_all, {k: v.detach().clone() for k, v in model.state_dict().items()})
            if ep == 0 or ep == epochs - 1 or ep % 5 == 4:
                log(f'K={K:2d} epoch {ep_all:3d}  train {h["train"]:.4f}  val {h["val"]:.4f}  '
                    f'val(K=50) {h["val_k50"]:.4f}  m^2  ({h["sec"]} s)')
    if keep_best:
        model.load_state_dict(best[2])
        log(f'kept weights from epoch {best[1]} (val K=50 {best[0]:.4f} m^2; 0 = before training)')
        hist.append(dict(best_epoch=best[1], best_val_k50=best[0]))
    return hist


def plot_curve(hist, path):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    hist = [h for h in hist if 'epoch' in h]
    e = [h['epoch'] for h in hist]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(e, [h['train'] for h in hist], label='train (current K)', c='#1f5fbf')
    ax.plot(e, [h['val'] for h in hist], label='val (current K)', c='#c0392b')
    ax.plot(e, [h['val_k50'] for h in hist], label='val (K=50, full 5 s rollout)', c='#c0392b', ls='--', alpha=0.6)
    for i in range(1, len(hist)):
        if hist[i]['K'] != hist[i - 1]['K']:
            ax.axvline(hist[i]['epoch'] - 0.5, c='0.7', lw=0.8)
    for K in dict.fromkeys(h['K'] for h in hist):
        xs = [h['epoch'] for h in hist if h['K'] == K]
        ax.text(np.mean(xs), 1.0, f'K={K}', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=8)
    ax.set_yscale('log'); ax.set_xlabel('epoch'); ax.set_ylabel('masked MSE (m$^2$, VISIBLE only)')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
