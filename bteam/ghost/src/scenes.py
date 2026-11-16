"""Cut defence scenes with the paper's rule and normalise direction (defending team guards x<0)."""
import collections
import numpy as np
from .load import SIDES

KICKOFF = {'FIRSTKICKOFF', 'SECONDKICKOFF', 'THIRDKICKOFF', 'THIRD KICKOFF', 'FOURTHKICKOFF'}
MIN_LEN = 10        # frames at 10 fps = 1 s
MAX_BALL_GAP = 5    # frames = 0.5 s
SHOT_WINDOW = 100   # frames = 10 s


def _team(g):
    try:
        return str(int(float(g.get('teamId'))))
    except (TypeError, ValueError):
        return None


def event_list(events, ge):
    """Merge event-file rows per gameEventId and attach tracking start/end frames."""
    by = {}
    for x in events:
        i = int(x['gameEventId'])
        e = by.setdefault(i, dict(id=i, g=x['gameEvents'], fouls=[], po=[]))
        f = x.get('fouls')
        for ff in (f if isinstance(f, list) else [f]):
            if isinstance(ff, dict) and any(v is not None for v in ff.values()):
                e['fouls'].append(ff)
        if x.get('possessionEvents'):
            e['po'].append(x['possessionEvents'])
    out, miss, nonev = [], 0, 0
    for i, e in by.items():
        if e['g'].get('initialNonEvent'):
            nonev += 1
            continue
        t = ge.get(str(i))
        if not t or t.get('start_frame') is None:
            miss += 1
            continue
        e['start'] = int(t['start_frame'])
        e['end'] = int(t['end_frame']) if t.get('end_frame') is not None else e['start']
        e['team'] = _team(e['g'])
        out.append(e)
    out.sort(key=lambda e: (e['start'], e['id']))
    return out, dict(event_ids=len(by), without_tracking_frame=miss, disallowed=nonev)


def segment(L, def_team):
    """Paper rule. Start: attacking team gets the ball (OTB or kick-off).
    End: goal, ball out, whistle / foul / offside, or the defending team has two touches in a row."""
    out, cur, last_def = [], None, False

    def close(end, why):
        nonlocal cur, last_def
        cur.update(end=end, why=why)
        out.append(cur)
        cur, last_def = None, False

    for e in L:
        typ = e['g'].get('gameEventType')
        if typ in KICKOFF or typ == 'OTB':
            if e['team'] is None:
                continue
            att = e['team'] != def_team
            if cur is None:
                if att:
                    cur, last_def = dict(start=e['start']), False
                continue
            if any(p.get('shotOutcomeType') == 'G' for p in e['po']):
                close(e['end'], 'goal'); continue
            fouls = [f for f in e['fouls'] if f.get('foulType') == 'I']
            if fouls:
                off = any(f.get('finalOffenseType') == 'O' for f in fouls)
                close(e['end'], 'offside' if off else 'foul'); continue
            if att:
                last_def = False
                continue
            # defending team on the ball: second touch in a row ends the scene at the regain moment
            if last_def or (e['g'].get('touches') or 0) >= 2:
                close(e['start'], 'regain_2_touches'); continue
            last_def = True
        elif typ == 'OUT' and cur is not None:
            ot = e['g'].get('outType')
            close(e['start'], 'goal' if ot in ('H', 'A') else {'T': 'ball_out', 'W': 'whistle'}.get(ot, 'out_other'))
        elif typ == 'END' and cur is not None:
            close(e['start'], 'end_of_half')
    return out


def tracking_event_list(ge):
    """Game events embedded in the tracking file (available for all 64 games)."""
    out = []
    for i, t in ge.items():
        if t.get('start_frame') is None:
            continue
        try:
            team = str(int(float(t.get('team_id'))))
        except (TypeError, ValueError):
            team = None
        s = int(t['start_frame'])
        out.append(dict(id=int(i), typ=t.get('game_event_type'), team=team, start=s,
                        end=int(t['end_frame']) if t.get('end_frame') is not None else s))
    out.sort(key=lambda e: (e['start'], e['id']))
    return out


def segment_tracking(L, def_team, hold_frames=30):
    """Tracking-only version of the paper rule. Start: attacking team gets the ball.
    End: OUT (ball out, goal or whistle; the type is not in the tracking file), END, or the defending
    team keeps the ball: two of its on-ball events in a row, or one on-ball event lasting >= hold_frames
    (stands in for 'two touches', which needs the event file)."""
    out, cur, last_def = [], None, False

    def close(end, why):
        nonlocal cur, last_def
        cur.update(end=end, why=why)
        out.append(cur)
        cur, last_def = None, False

    for e in L:
        typ = e['typ']
        if typ in KICKOFF or typ == 'OTB':
            if e['team'] is None:
                continue
            att = e['team'] != def_team
            if cur is None:
                if att:
                    cur, last_def = dict(start=e['start']), False
                continue
            if att:
                last_def = False
                continue
            if last_def:
                close(e['start'], 'regain_2_events'); continue
            if e['end'] - e['start'] >= hold_frames:
                close(e['start'], 'regain_long_possession'); continue
            last_def = True
        elif typ == 'OUT' and cur is not None:
            close(e['start'], 'stoppage')
        elif typ == 'END' and cur is not None:
            close(e['start'], 'end_of_half')
    return out


def goal_sides(trk):
    """True if that side defends the x<0 goal in that half, from the goalkeepers' mean x."""
    info, res = trk['info'], {}
    for s in SIDES:
        cols = [i for i, n in enumerate(info['shirts'][s]) if n in info['gk'][s]]
        for p in sorted(set(trk['period'].tolist())):
            m = trk['period'] == p
            x = trk[f'pos_{s}'][m][:, cols, 0]
            res[(s, int(p))] = (bool(np.nanmean(x) < 0), float(np.nanmean(x)))
    return res


def _longest_nan(x):
    best = cur = 0
    for v in np.isnan(x):
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _interp(b):
    t = np.arange(len(b)); ok = ~np.isnan(b[:, 0])
    return np.stack([np.interp(t, t[ok], b[ok, k]) for k in range(2)], 1).astype(np.float32)


def _nearest(F, f):
    i = int(np.searchsorted(F, f))
    cand = [k for k in (i - 1, i) if 0 <= k < len(F)]
    return min(cand, key=lambda k: abs(int(F[k]) - f))


def cut_scenes(trk, events, rule='events', hold_s=1.0):
    """rule='events': paper rule with the event file. rule='tracking': only the tracking file
    (events may be None; then there are no shot marks)."""
    info, F, per = trk['info'], trk['frame'], trk['period']
    L, ev_rep = event_list(events, info['ge']) if events else ([], {})
    if rule == 'events':
        seg = lambda team: segment(L, team)
    else:
        LT = tracking_event_list(info['ge'])
        ev_rep = dict(ev_rep, tracking_events=len(LT))
        seg = lambda team: segment_tracking(LT, team, int(round(hold_s * info['fps'])))
    sides = goal_sides(trk)
    shots = [(_nearest(F, e['start']), e['team']) for e in L
             if any(p.get('possessionEventType') == 'SH' for p in e['po'])]
    scenes, rows, drop, raw = [], [], collections.Counter(), collections.Counter()
    n_interp = 0
    for side in SIDES:
        opp = 'away' if side == 'home' else 'home'
        for sc in seg(info['team_id'][side]):
            raw[side] += 1
            a, b = _nearest(F, sc['start']), _nearest(F, sc['end'])
            row = dict(game=info['game'], defend=info['names'][side], period=int(per[a]), start_frame=int(F[a]),
                       end_frame=int(F[b]), length_s=round((b - a) * 0.1, 1), end_reason=sc['why'], excluded='')
            rows.append(row)
            why, sl = None, slice(a, b + 1)
            if per[a] != per[b]:
                why = 'crosses_half'
            elif per[a] > 4:
                why = 'penalty_shootout'
            elif b - a < MIN_LEN:
                why = 'shorter_than_1s'
            elif not trk['ok'][sl].all():
                why = 'tracking_frames_missing'
            if why is None:
                Pd, Pa = trk[f'pos_{side}'][sl], trk[f'pos_{opp}'][sl]
                pd, pa = ~np.isnan(Pd[..., 0]), ~np.isnan(Pa[..., 0])
                if (pd.sum(1) != 11).any() or (pa.sum(1) != 11).any():
                    why = 'not_11_v_11'
                elif not ((pd == pd[0]).all() and (pa == pa[0]).all()):
                    why = 'player_changed_in_scene'
            if why is None:
                gd = [c for c in np.where(pd[0])[0] if info['shirts'][side][c] in info['gk'][side]]
                ga = [c for c in np.where(pa[0])[0] if info['shirts'][opp][c] in info['gk'][opp]]
                if len(gd) != 1 or len(ga) != 1:
                    why = 'goalkeeper_unclear'
            if why is None:
                ball = trk['ball'][sl]
                gap = _longest_nan(ball[:, 0])
                if gap > MAX_BALL_GAP:
                    why = 'ball_missing_over_0.5s'
                elif gap:
                    ball = _interp(ball); n_interp += 1
            if why:
                drop[why] += 1; row['excluded'] = why
                continue
            s = 1.0 if sides[(side, int(per[a]))][0] else -1.0     # rotate 180 deg so the defenders guard x<0
            outd = [c for c in np.where(pd[0])[0] if c != gd[0]]
            atk = list(np.where(pa[0])[0])
            scenes.append(dict(
                game=info['game'], defend_name=info['names'][side], side=side, opp=opp, period=int(per[a]), a=a, b=b, L=b - a + 1, why=sc['why'],
                start_frame=int(F[a]), time=float(trk['time'][a]),
                D_all=Pd[:, outd] * s, D_vis_all=trk[f'vis_{side}'][sl][:, outd],
                D_shirt_all=[info['shirts'][side][c] for c in outd], Dgk=Pd[:, gd[0]] * s,
                A_all=Pa[:, atk] * s, A_vis_all=trk[f'vis_{opp}'][sl][:, atk],
                A_shirt_all=[info['shirts'][opp][c] for c in atk], Agk=Pa[:, ga[0]] * s,
                ball=(ball * s).astype(np.float32),
                shots=[k - a for k, t in shots if t == info['team_id'][opp] and a <= k <= b]))
    lens = np.array([(s['L'] - 1) * 0.1 for s in scenes])
    rep = dict(events=ev_rep, candidate_scenes=dict(raw), candidate_total=int(sum(raw.values())),
               excluded=dict(drop), excluded_total=int(sum(drop.values())), kept=len(scenes),
               kept_by_half={p: sum(s['period'] == p for s in scenes) for p in (1, 2, 3, 4)},
               kept_by_defending_team={info['names'][sd]: sum(s['side'] == sd for s in scenes) for sd in SIDES},
               mean_length_s=float(lens.mean()) if len(lens) else 0.0,
               median_length_s=float(np.median(lens)) if len(lens) else 0.0,
               total_minutes=float(lens.sum() / 60), ball_gap_interpolated_scenes=n_interp,
               end_reason_of_kept=dict(collections.Counter(s['why'] for s in scenes)),
               goal_side={f'{info["names"][k[0]]} half {k[1]}': dict(defends_left=v[0], gk_mean_x=round(v[1], 1))
                          for k, v in sides.items()},
               metadata_home_start_left=info['home_start_left'])
    return scenes, rows, rep
