"""Rollout error at 1/2/5 s vs two baselines, visibility stats, and pitch plots."""
import numpy as np
import torch
from .features import make_batch
from .model import rollout
from .scenes import SHOT_WINDOW


@torch.no_grad()
def horizon_eval(model, scenes, tmpl, device, hs=(10, 20, 50), every=10, bs=128, joint=False):
    """Start points every 1 s. Up to the start the role is fed its real positions (LSTM warm-up),
    then it rolls out h steps on its own predictions (joint=False: team-mates real; joint=True: all 10
    defenders predicted). Error at the end point, counted only where that defender is VISIBLE."""
    model.eval()
    res = {}
    for h in hs:
        smp = [(i, s0) for i, sc in enumerate(scenes) for s0 in range(1, sc['L'] - h, every)]
        if not smp:
            res[h] = None; continue
        E = {k: [] for k in ('lstm', 'still', 'const_vel')}
        V, V0 = [], []
        for c in range(0, len(smp), bs):
            ch = smp[c:c + bs]
            b = make_batch(scenes, [(i, 0, s0 + h + 1) for i, s0 in ch], device)
            L = b['D'].shape[1]
            s0 = torch.tensor([s for _, s in ch], device=device)
            pred = rollout(model, b, tmpl, torch.arange(L, device=device)[None] <= s0[:, None], joint)
            k = torch.arange(len(ch), device=device)
            true, p0, pm1 = b['D'][k, s0 + h], b['D'][k, s0], b['D'][k, s0 - 1]
            E['lstm'].append((pred[k, s0 + h - 1] - true).norm(dim=-1).cpu())
            E['still'].append((p0 - true).norm(dim=-1).cpu())
            E['const_vel'].append((p0 + h * (p0 - pm1) - true).norm(dim=-1).cpu())
            V.append(b['vis'][k, s0 + h].cpu()); V0.append((b['vis'][k, s0] & b['vis'][k, s0 - 1]).cpu())
        V = torch.cat(V).numpy()
        VS = V & torch.cat(V0).numpy()      # strict: start (and the frame before it) and end all VISIBLE
        E = {k: torch.cat(v).numpy() for k, v in E.items()}
        mean = lambda e, M: float(e[M].mean()) if M.any() else None
        res[h] = dict(start_points=len(smp), n_end=int(V.sum()), n_strict=int(VS.sum()),
                      end={k: mean(e, V) for k, e in E.items()}, strict={k: mean(e, VS) for k, e in E.items()},
                      within_3m_strict={k: mean((e < 3).astype(float), VS) for k, e in E.items()},
                      per_role_end={k: [mean(e[:, r], V[:, r]) for r in range(10)] for k, e in E.items()},
                      per_role_strict={k: [mean(e[:, r], VS[:, r]) for r in range(10)] for k, e in E.items()})
    return res


@torch.no_grad()
def ghost(model, scenes, i, tmpl, device, joint=False):
    """Full-scene ghost: real start, then roles on their own predictions
    (joint=False: team-mates real; joint=True: all 10 defenders generated)."""
    model.eval()
    b = make_batch(scenes, [(i, 0, scenes[i]['L'])], device)
    ur = torch.zeros(1, b['D'].shape[1], dtype=torch.bool, device=device); ur[:, 0] = True
    pred = rollout(model, b, tmpl, ur, joint)[0].cpu().numpy()
    return np.concatenate([scenes[i]['D'][:1], pred], 0)


@torch.no_grad()
def scene_eval(model, scenes, tmpl, device, joint=True, bs=32):
    """Whole-scene ghost like the paper: real first frame, then rolled out to the end of the scene.
    Mean distance to the real defender over all later frames where he is VISIBLE; also the first 10 s only.
    Baseline: everyone stays where he was at the first frame."""
    model.eval()
    order = sorted(range(len(scenes)), key=lambda i: scenes[i]['L'])
    acc = {k: [0.0, 0] for k in ('lstm', 'still', 'lstm_10s', 'still_10s', 'lstm_within_3m')}
    for c in range(0, len(order), bs):
        idx = order[c:c + bs]
        b = make_batch(scenes, [(i, 0, scenes[i]['L']) for i in idx], device)
        L = b['D'].shape[1]
        ur = torch.zeros(len(idx), L, dtype=torch.bool, device=device); ur[:, 0] = True
        pred = rollout(model, b, tmpl, ur, joint)
        true = b['D'][:, 1:]
        m = b['vis'][:, 1:] & b['valid'][:, 1:, None]
        m10 = m & (torch.arange(1, L, device=device) <= 100)[None, :, None]
        e, s = (pred - true).norm(dim=-1), (b['D'][:, :1] - true).norm(dim=-1)
        for k, x, mm in (('lstm', e, m), ('still', s, m), ('lstm_10s', e, m10), ('still_10s', s, m10),
                         ('lstm_within_3m', (e < 3).float(), m)):
            acc[k][0] += float(x[mm].sum()); acc[k][1] += int(mm.sum())
    out = {k: v[0] / max(v[1], 1) for k, v in acc.items()}
    out['frames'] = acc['lstm'][1]
    return out


def vis_stats(scenes, sc_tr, sc_va):
    allv = np.concatenate([s['vis'] for s in scenes])
    shot_rows, n_shots = [], 0
    for s in scenes:
        fr = set()
        for k in s['shots']:
            n_shots += 1
            fr.update(range(max(0, k - SHOT_WINDOW), k + 1))
        if fr:
            shot_rows.append(s['vis'][sorted(fr)])
    shotv = np.concatenate(shot_rows) if shot_rows else np.zeros((0, 10), bool)
    tgt = lambda sc: np.sum([s['vis'][1:].sum(0) for s in sc], 0).astype(int).tolist()
    slots = lambda sc: int(sum((s['L'] - 1) * 10 for s in sc))
    return dict(visible_share_all_scenes=allv.mean(0).tolist(), visible_share_all_overall=float(allv.mean()),
                frames_all=int(len(allv)), shots_in_kept_scenes=n_shots, frames_shot_10s=int(len(shotv)),
                visible_share_shot_10s=shotv.mean(0).tolist() if len(shotv) else None,
                visible_share_shot_10s_overall=float(shotv.mean()) if len(shotv) else None,
                targets_train=tgt(sc_tr), targets_val=tgt(sc_va),
                target_slots_train=slots(sc_tr), target_slots_val=slots(sc_va))


def pitch(ax):
    from matplotlib.patches import Rectangle, Circle
    kw = dict(fill=False, ec='0.55', lw=1)
    ax.add_patch(Rectangle((-52.5, -34), 105, 68, **kw))
    ax.plot([0, 0], [-34, 34], c='0.55', lw=1)
    ax.add_patch(Circle((0, 0), 9.15, **kw))
    for sx in (-1, 1):
        ax.add_patch(Rectangle((sx * 52.5 - (16.5 if sx > 0 else 0), -20.16), 16.5, 40.32, **kw))
        ax.add_patch(Rectangle((sx * 52.5 - (5.5 if sx > 0 else 0), -9.16), 5.5, 18.32, **kw))
        ax.plot([sx * 52.5] * 2, [-3.66, 3.66], c='k', lw=3)
    ax.set_xlim(-56, 56); ax.set_ylim(-37, 37); ax.set_aspect('equal'); ax.axis('off')


def _runs(mask):
    out, s = [], 0
    for t in range(1, len(mask) + 1):
        if t == len(mask) or mask[t] != mask[s]:
            out.append((s, t, bool(mask[s]))); s = t
    return out


def plot_scene(sc, gh, path, title):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 7.4))
    pitch(ax)
    for j in range(sc['A'].shape[1]):
        ax.plot(sc['A'][:, j, 0], sc['A'][:, j, 1], c='0.75', lw=1)
        ax.scatter(*sc['A'][-1, j], c='0.6', s=18)
    ax.plot(sc['ball'][:, 0], sc['ball'][:, 1], c='#e69f00', lw=1, alpha=0.8)
    ax.scatter(*sc['ball'][0], c='#e69f00', s=25, marker='o')
    for r in range(10):
        p, v = sc['D'][:, r], sc['vis'][:, r]
        seg_vis = np.r_[v[1:], v[-1:]]           # segment t->t+1 styled by visibility at t+1
        for a, b, isv in _runs(seg_vis):
            e = min(b + 1, len(p))
            ax.plot(p[a:e, 0], p[a:e, 1], c='#1f5fbf', lw=1.8, ls='-' if isv else ':')
        ax.scatter(*p[0], c='#1f5fbf', s=22, marker='s')
        ax.plot(gh[:, r, 0], gh[:, r, 1], c='#c0392b', lw=1.4, alpha=0.9)
        ax.scatter(*gh[-1, r], c='#c0392b', s=30, marker='x')
        ax.text(*p[0] + np.array([0.8, 0.8]), f'D{r + 1}', fontsize=7, color='#1f5fbf')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], c='#1f5fbf', lw=1.8, label='real defender, VISIBLE'),
                       Line2D([], [], c='#1f5fbf', lw=1.8, ls=':', label='real defender, ESTIMATED'),
                       Line2D([], [], c='#c0392b', lw=1.4, label='ghost (LSTM rollout from scene start)'),
                       Line2D([], [], c='0.75', lw=1, label='attackers'),
                       Line2D([], [], c='#e69f00', lw=1, label='ball')],
              loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=8, frameon=False)
    ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
