"""Role templates (iterative Hungarian) and per-scene role order."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .load import DT


def _match(m, T):
    C = ((m[:, None] - T[None]) ** 2).sum(-1)
    r, c = linear_sum_assignment(C)
    o = np.empty(len(T), int)
    o[c] = r            # o[role] = player column
    return o


def _scene_means(arrs):
    """Per-scene mean position of each player, minus the team centre of that scene
    (so the template keeps the formation shape instead of averaging where on the pitch the block was)."""
    out = []
    for a in arrs:
        m = a.mean(0)
        out.append(m - m.mean(0))
    return out


def fit_template(arrs, iters=10):
    """Init from the first scene's centred mean positions, then 10 rounds of assign-and-average."""
    means = _scene_means(arrs)
    T = means[0].copy()
    hist = []
    for _ in range(iters):
        orders = [_match(m, T) for m in means]
        hist.append(float(np.mean([np.sqrt(((m[o] - T) ** 2).sum(-1)).mean() for m, o in zip(means, orders)])))
        T = np.mean([m[o] for m, o in zip(means, orders)], 0)
    return T.astype(np.float32), [_match(m, T) for m in means], hist


def _vel(p):
    v = np.zeros_like(p)
    v[1:] = (p[1:] - p[:-1]) / DT
    return v


def apply_roles(scenes, TD, TA):
    """Order players by given (centred) templates, e.g. the ones saved with a model."""
    for s in scenes:
        d = _match(_scene_means([s['D_all']])[0], TD)
        a = _match(_scene_means([s['A_all']])[0], TA)
        s['D'], s['vis'] = s['D_all'][:, d], s['D_vis_all'][:, d]
        s['D_shirt'] = [s['D_shirt_all'][k] for k in d]
        s['A'], s['A_vis'] = s['A_all'][:, a], s['A_vis_all'][:, a]
        s['Dv'], s['Av'], s['ballv'] = _vel(s['D']), _vel(s['A']), _vel(s['ball'])


def assign_roles(fit_scenes, all_scenes=None, iters=10):
    """Templates are centred (team centre = 0). They are fitted on fit_scenes and applied to all_scenes."""
    all_scenes = fit_scenes if all_scenes is None else all_scenes
    TD, _, hD = fit_template([s['D_all'] for s in fit_scenes], iters)
    TA, _, hA = fit_template([s['A_all'] for s in fit_scenes], iters)
    cD = np.mean([s['D_all'].mean((0, 1)) for s in fit_scenes], 0)
    cA = np.mean([s['A_all'].mean((0, 1)) for s in fit_scenes], 0)
    apply_roles(all_scenes, TD, TA)
    return TD, TA, dict(defence_mean_dist_per_iter=hD, attack_mean_dist_per_iter=hA,
                        defence_mean_centre=cD.tolist(), attack_mean_centre=cA.tolist())


def plot_template(TD, TA, path, cD=(0, 0), cA=(0, 0)):
    """Centred templates drawn around the average team centre."""
    TD, TA = TD + np.asarray(cD), TA + np.asarray(cA)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .evaluate import pitch
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, T, c, name, lab in ((axes[0], TD, '#1f5fbf', 'Defence template (10 roles, own goal at x<0)', 'D'),
                                (axes[1], TA, '#c0392b', 'Attack template (11 slots, attacking x<0 goal)', 'A')):
        pitch(ax)
        ax.scatter(T[:, 0], T[:, 1], s=320, c=c, zorder=3)
        for i, (x, y) in enumerate(T):
            ax.text(x, y, f'{lab}{i + 1}', ha='center', va='center', color='white', fontsize=8, weight='bold', zorder=4)
        ax.set_title(name, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
