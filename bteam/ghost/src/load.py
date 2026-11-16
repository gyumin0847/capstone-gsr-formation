"""Load one PFF World Cup game into 10 fps arrays. The raw files are only read."""
import bz2, glob, json, os
import numpy as np

RAW = os.environ.get('PFF_RAW', '/lab/users/gyumin/data/pff_raw')
STEP = 3            # 29.97 fps -> every 3rd frame, about 10 fps
DT = 0.1
SIDES = ('home', 'away')


def _first(x):
    return x[0] if isinstance(x, list) else x


def load_meta(game, raw=RAW):
    meta = _first(json.load(open(f'{raw}/Metadata/{game}.json')))
    roster = json.load(open(f'{raw}/Rosters/{game}.json'))
    paths = glob.glob(f'{raw}/Event Data/*/{game}.json')
    events = json.load(open(paths[0])) if paths else None
    return meta, roster, events


def roster_info(meta, roster):
    home = str(meta['homeTeam']['id'])
    shirts = {s: [] for s in SIDES}
    gk = {s: [] for s in SIDES}
    for r in roster:
        s = 'home' if str(r['team']['id']) == home else 'away'
        n = int(r['shirtNumber'])
        if n not in shirts[s]:
            shirts[s].append(n)
        if r.get('positionGroupType') == 'GK' and n not in gk[s]:
            gk[s].append(n)
    return {s: sorted(v) for s, v in shirts.items()}, gk


def _jersey(q):
    try:
        return int(q.get('jerseyNum'))
    except (TypeError, ValueError):
        return None


def _xy(d):
    if not isinstance(d, dict) or d.get('x') is None or d.get('y') is None:
        return None
    return float(d['x']), float(d['y'])


def load_game(game, raw=RAW, cache=None):
    """Returns arrays on a 10 fps grid per half. Positions are the smoothed values,
    visibility is the raw value. Grid frames missing from the file are marked ok=False."""
    if cache and os.path.exists(cache):
        z = np.load(cache)
        out = {k: z[k] for k in z.files}
        out['info'] = json.loads(str(out['info']))
        return out
    meta, roster, _ = load_meta(game, raw)
    shirts, gk = roster_info(meta, roster)
    col = {s: {n: i for i, n in enumerate(shirts[s])} for s in SIDES}
    st = dict(lines=0, duplicate_lines=0, jersey_not_in_roster=0, duplicate_jersey_in_frame=0, no_smoothed_xy=0)
    seen, ge = set(), {}
    fr, per, tt, ball = [], [], [], []
    pos = {s: [] for s in SIDES}
    vis = {s: [] for s in SIDES}
    with bz2.open(f'{raw}/Tracking Data/{game}.jsonl.bz2', 'rt') as f:
        for line in f:
            r = json.loads(line)
            st['lines'] += 1
            g, gid = r.get('game_event'), r.get('game_event_id')
            if g and gid is not None:
                ge.setdefault(str(int(gid)), {k: g.get(k) for k in
                                              ('game_event_type', 'start_frame', 'end_frame', 'home_ball', 'team_id')})
            fn = int(r['frameNum'])
            if fn in seen:
                st['duplicate_lines'] += 1
                continue
            seen.add(fn)
            fr.append(fn); per.append(int(r['period'])); tt.append(float(r['periodElapsedTime']))
            for s in SIDES:
                p = np.full((len(shirts[s]), 2), np.nan, np.float32)
                v = np.zeros(len(shirts[s]), bool)
                sm = {_jersey(q): _xy(q) for q in (r.get(f'{s}PlayersSmoothed') or [])}
                for q in r.get(f'{s}Players') or []:
                    n = _jersey(q)
                    if n not in col[s]:
                        st['jersey_not_in_roster'] += 1
                        continue
                    i = col[s][n]
                    if not np.isnan(p[i, 0]):
                        st['duplicate_jersey_in_frame'] += 1
                        continue
                    xy = sm.get(n)
                    if xy is None:
                        st['no_smoothed_xy'] += 1
                        continue
                    p[i] = xy
                    v[i] = q.get('visibility') == 'VISIBLE'
                pos[s].append(p); vis[s].append(v)
            b = _xy(r.get('ballsSmoothed'))
            ball.append(b if b else (np.nan, np.nan))

    fr = np.array(fr); o = np.argsort(fr, kind='stable'); fr = fr[o]
    per = np.array(per)[o]; tt = np.array(tt)[o]
    P = {s: np.stack(pos[s])[o] for s in SIDES}
    V = {s: np.stack(vis[s])[o] for s in SIDES}
    B = np.array(ball, np.float32)[o]
    fps = float(meta['fps'])
    g_fr, g_per, g_t, g_i = [], [], [], []
    for p in sorted(set(per.tolist())):
        m = np.where(per == p)[0]
        grid = np.arange(fr[m[0]], fr[m[-1]] + 1, STEP)
        j = np.clip(np.searchsorted(fr, grid), 0, len(fr) - 1)
        hit = fr[j] == grid
        g_fr.append(grid); g_per.append(np.full(len(grid), p))
        g_t.append(tt[m[0]] + (grid - fr[m[0]]) / fps); g_i.append(np.where(hit, j, -1))
    gi = np.concatenate(g_i); ok = gi >= 0; j = np.maximum(gi, 0)
    out = dict(frame=np.concatenate(g_fr), period=np.concatenate(g_per), time=np.concatenate(g_t), ok=ok,
               ball=np.where(ok[:, None], B[j], np.nan).astype(np.float32))
    for s in SIDES:
        out[f'pos_{s}'] = np.where(ok[:, None, None], P[s][j], np.nan).astype(np.float32)
        out[f'vis_{s}'] = V[s][j] & ok[:, None]
    st.update(unique_frames=int(len(fr)), grid_frames_10fps=int(len(gi)), grid_frames_missing=int((~ok).sum()))
    info = dict(game=str(game), fps=fps, shirts=shirts, gk=gk, stats=st, ge=ge,
                team_id={'home': str(meta['homeTeam']['id']), 'away': str(meta['awayTeam']['id'])},
                names={'home': meta['homeTeam']['name'], 'away': meta['awayTeam']['name']},
                home_start_left=meta.get('homeTeamStartLeft'))
    out['info'] = info
    if cache:
        np.savez_compressed(cache, **{k: v for k, v in out.items() if k != 'info'}, info=np.array(json.dumps(info)))
    return out
