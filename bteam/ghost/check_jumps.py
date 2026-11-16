"""Read-only check of the video scenes: where do real defenders move faster than a human can,
and does it happen when they switch between VISIBLE and ESTIMATED?"""
import numpy as np
from src import load, scenes as S

SCENES = {'10513': 170861, '3816': 103977, '3817': 42466}
for g, f0 in SCENES.items():
    trk = load.load_game(g, cache=f'outputs/cache/{g}_10fps.npz')
    sc = [s for s in S.cut_scenes(trk, load.load_meta(g)[2])[0] if s['start_frame'] == f0][0]
    D, V = sc['D_all'], sc['D_vis_all']            # column order (role order does not matter here)
    sp = np.zeros(V.shape); sp[1:] = np.linalg.norm(np.diff(D, axis=0), axis=-1) / 0.1
    print(f'== game {g} {sc["defend_name"]} defending, {(sc["L"] - 1) / 10:.1f} s, VISIBLE {V.mean():.0%}')
    for j in range(D.shape[1]):
        v = V[:, j]
        sw = np.where(v[1:] != v[:-1])[0] + 1
        fast = np.where(sp[:, j] > 9)[0]
        if len(sw) == 0 and len(fast) == 0:
            continue
        near = [int(t) for t in fast if len(sw) and np.min(np.abs(sw - t)) <= 5]
        print(f'  shirt {sc["D_shirt_all"][j]:2d}: visible {v.mean():.0%}, switches at '
              f'{[round(t / 10, 1) for t in sw]} s, max speed {sp[:, j].max():.1f} m/s, '
              f'>9 m/s at {[round(t / 10, 1) for t in fast]} s, of which within 0.5 s of a switch: {len(near)}')
        for t in fast[:6]:
            print(f'      t={t / 10:4.1f}s  {"VIS" if v[t - 1] else "EST"}->{"VIS" if v[t] else "EST"}  '
                  f'{sp[t, j]:5.1f} m/s  ({D[t - 1, j, 0]:6.1f},{D[t - 1, j, 1]:6.1f}) -> ({D[t, j, 0]:6.1f},{D[t, j, 1]:6.1f})')
