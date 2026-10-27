"""Final game-state JSON (team plan: "GSR + 보간 적용된 최종 json").

For every frame and team: all 10 outfield roles, each either OBSERVED (a detected player matched to the role)
or IMPUTED (off-screen: role mean + team shift, or, when the tracker kept the player's ID, a blend of where he
was last seen and his role's usual place). Downstream code (heatmaps, pass maps, LLM prompts) must look at
`observed` and leave imputed rows out where a real position is needed.

Inputs (pick one):
  --viewer manutd_viewer.json                    pitch points per frame (pipeline/manutd_v2.py), no IDs
  --gsr interpolate_SEQ.txt --calib CALIB.json --teams teams.json
                                                 GSR tracker output (frame,id,x,y,w,h,...,role at col 7),
                                                 SoccerNet camera calibration and our team split -> keeps IDs
usage:
  python analysis/export_game_state.py --viewer manutd_viewer.json --form 4-2-3-1 4-4-2 --out game_state.json
"""
import argparse, json, collections
import numpy as np
import role_distributions as R

FLIP = {'0': 1.0, '1': -1.0}            # team 1 rotated so both attack +x (build.py convention)
HOLD_S = 5.0                             # keep the team shift this long when the team leaves the screen
ROLE_ERR_M = 8.3                         # median off-screen error of role mean + shift on Man Utd hold-out (impute_eval.py)


# ------------------------------------------------------------------ inputs -> {f: {team: [(track_id|None, x, y, extra)]}}
def from_viewer(path):
    d = json.load(open(path)); out = collections.OrderedDict()
    for fr in d['frames']:
        out[fr['f']] = {t: [(None, p[0], p[1], {}) for p in (fr.get(t) or {}).get('p', [])] for t in FLIP}
    return out


def from_gsr(txt, calib, teams):
    """same projection as pipeline/manutd_v2.py: bottom-centre of the box through the inverse homography"""
    team = {int(k): str(v) for k, v in json.load(open(teams)).items()}
    cal = json.load(open(calib))
    if isinstance(cal, dict):
        for k in ('predictions', 'annotations', 'calibration'):
            if k in cal:
                cal = cal[k]; break

    def Hinv(i):
        try:
            e = cal[i]; e = e[0] if isinstance(e, list) else e
            h = e.get('homography') or e.get('H')
            return None if h is None else np.linalg.inv(np.array(h, float).reshape(3, 3))
        except Exception:
            return None
    out = collections.OrderedDict(); cache = {}
    for ln in open(txt):
        p = ln.strip().split(',')
        if len(p) < 8 or p[7].strip().lower() != 'player':
            continue
        f, tid = int(p[0]), int(p[1])
        if tid not in team:
            continue
        if f not in cache:
            cache[f] = Hinv(f - 1)
        H = cache[f]
        if H is None:
            continue
        x, y, w, h = (float(p[i]) for i in (2, 3, 4, 5))
        pts = {}
        for name, u in (('bottom_left', x), ('bottom_middle', x + w / 2), ('bottom_right', x + w)):
            v = H @ np.array([u, y + h, 1.0]); pts[name] = [round(float(v[0] / v[2]), 2), round(float(v[1] / v[2]), 2)]
        X, Y = pts['bottom_middle']
        if abs(X) > 55 or abs(Y) > 36:
            continue
        extra = {'bbox_image': [round(x, 1), round(y, 1), round(w, 1), round(h, 1)], 'bbox_pitch': pts}
        if len(p) > 8 and p[8].strip().isdigit():
            extra['jersey'] = int(p[8])
        out.setdefault(f, {t: [] for t in FLIP})[team[tid]].append((tid, X, Y, extra))
    return collections.OrderedDict(sorted(out.items()))


# ------------------------------------------------------------------ export
def export(frames, forms, restarts=3, tau=5.0, fps=2.0):
    meta = {'coords': 'metres, pitch centre = (0,0); x = towards the right goal of the broadcast; '
                      'team "1" positions are NOT rotated in the output (same axes as the input)',
            'fps': fps, 'teams': {}}
    mus = {}
    for t, fl in FLIP.items():
        fr = [(f, np.array([[x, y] for _, x, y, _ in v[t]]) * fl) for f, v in frames.items() if len(v[t]) >= 3]
        best = R.best_fit(fr, restarts=restarts, max_iter=40)
        mus[t] = best['mu']
        names, conf = R.name_confidence(best['mu'], forms[t])
        meta['teams'][t] = {'formation': forms[t], 'frames_used': len(fr),
                            'roles': [{'slot': k, 'position': names[k], 'name_confidence': round(float(conf[k]), 2),
                                       'mean_rel': [round(float(v), 2) for v in best['mu'][k] * fl]} for k in range(R.K)]}
    out_frames = []
    state = {t: {'s_prev': None, 'f_prev': None, 'seen': {}} for t in FLIP}   # seen[track] = (f, pos_fl, s_fl, role)
    for f, v in frames.items():
        rec = {'f': f, 'teams': {}}
        for t, fl in FLIP.items():
            st = state[t]; dets = v[t]; mu = mus[t]
            rows = []
            recent = st['f_prev'] is not None and (f - st['f_prev']) / fps <= HOLD_S
            if len(dets) < 3 and not recent:
                rec['teams'][t] = None; continue
            P = np.array([[x, y] for _, x, y, _ in dets]).reshape(-1, 2) * fl
            if len(dets) >= 3:
                sp = st['s_prev'] if st['f_prev'] is not None and f - st['f_prev'] <= 2 else None
                s, ri, ci, _ = R.match_frame(P, mu, s_prev=sp)
                st['s_prev'], st['f_prev'] = s, f
            else:
                # 0-2 players of this team on screen: keep the last team shift for up to HOLD_S seconds
                s = st['s_prev']; ri, ci = np.array([], int), np.array([], int)
                if len(dets):
                    C = np.linalg.norm(P[:, None] - (mu + s)[None], axis=2)
                    from scipy.optimize import linear_sum_assignment
                    ri, ci = linear_sum_assignment(C); ok = C[ri, ci] <= 15; ri, ci = ri[ok], ci[ok]
            taken = set()
            for i, k in zip(ri, ci):
                tid, x, y, extra = dets[i]; taken.add(int(k))
                rows.append(dict({'slot': int(k), 'position': meta['teams'][t]['roles'][k]['position'], 'x': round(x, 2),
                                  'y': round(y, 2), 'observed': True, 'track_id': tid, 'source': 'detection'}, **extra))
                if tid is not None:
                    st['seen'][tid] = (f, P[i], s, int(k))
            visible_ids = {d[0] for d in dets}
            for k in range(R.K):
                if k in taken:
                    continue
                est, src, gone, tid = mu[k] + s, 'role_mean+team_shift', None, None
                # a tracked player who last held this role and is now off-screen
                cand = [(ff, tt) for tt, (ff, _, _, kk) in st['seen'].items() if kk == k and tt not in visible_ids]
                if cand:
                    ff, tid = max(cand)
                    gone = (f - ff) / fps
                    _, lp, ls, _ = st['seen'][tid]
                    est = R.impute_track(lp, ls, k, gone, s, mu, tau); src = f'blend(last_seen,role) tau={tau:g}s'
                xy = est * fl
                rows.append({'slot': k, 'position': meta['teams'][t]['roles'][k]['position'], 'x': round(float(xy[0]), 2),
                             'y': round(float(xy[1]), 2), 'observed': False, 'track_id': tid, 'source': src,
                             'off_screen_s': None if gone is None else round(gone, 1),
                             'expected_error_m': ROLE_ERR_M if gone is None else None})
            rec['teams'][t] = {'team_shift': [round(float(c), 2) for c in s * fl], 'shift_from': 'this_frame' if len(dets) >= 3 else 'held',
                              'players': sorted(rows, key=lambda r: r['slot'])}
        rec['ball'] = None                       # filled when a ball detector/Astra possession is connected
        out_frames.append(rec)
    return {'meta': meta, 'frames': out_frames}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--viewer'); ap.add_argument('--gsr'); ap.add_argument('--calib'); ap.add_argument('--teams')
    ap.add_argument('--form', nargs=2, default=['4-2-3-1', '4-4-2'], help='formation of team 0 and team 1')
    ap.add_argument('--fps', type=float, default=2.0); ap.add_argument('--tau', type=float, default=5.0)
    ap.add_argument('--restarts', type=int, default=3)
    ap.add_argument('--out', default='game_state.json')
    a = ap.parse_args()
    frames = from_viewer(a.viewer) if a.viewer else from_gsr(a.gsr, a.calib, a.teams)
    res = export(frames, {'0': a.form[0], '1': a.form[1]}, a.restarts, a.tau, a.fps)
    json.dump(res, open(a.out, 'w'), ensure_ascii=False, separators=(',', ':'))
    n_obs = sum(p['observed'] for fr in res['frames'] for tm in fr['teams'].values() if tm for p in tm['players'])
    n_all = sum(len(tm['players']) for fr in res['frames'] for tm in fr['teams'].values() if tm)
    print(f"written {a.out}: {len(res['frames'])} frames, {n_obs}/{n_all} player rows observed ({100 * n_obs / n_all:.0f}%)")
