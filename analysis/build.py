import json, sys, numpy as np
rng = np.random.default_rng(0)

def cline(C, band, k):
    """line height from central players only (wide players excluded), as analysts measure line height"""
    idx = band == k; cy = C[:, 1].mean()
    sel = idx & (np.abs(C[:, 1] - cy) <= 12)
    if sel.sum() == 0: sel = idx
    return float(C[sel, 0].mean())
d = json.load(open(sys.argv[1]))
FLIP = {'0': 1.0, '1': -1.0}           # rotate Leicester 180 deg so both attack to the right
ZX = ['수비', '미드', '공격']; ZY = ['왼쪽', '중앙', '오른쪽']
out = {'teams': {}}
for t in '01':
    o = '1' if t == '0' else '0'; fl = FLIP[t]
    absP, rel, role, fidx = [], [], [], []
    for fr in d['frames']:
        x = fr[t]
        if not x: continue
        for p, a in zip(x['p'], x['a']):
            absP.append([p[0]*fl, p[1]*fl])
            if a >= 0 and x['sh'] is not None:
                rel.append([(p[0]-x['sh'][0])*fl, (p[1]-x['sh'][1])*fl]); role.append(a); fidx.append(fr['f'])
    absP = np.array(absP); rel = np.array(rel); role = np.array(role); fidx = np.array(fidx)
    M = np.array([rel[role == j].mean(0) for j in range(10)])
    cx, cy = M.mean(0)
    # position names: match the 10 slots to the team's published starting formation
    from scipy.optimize import linear_sum_assignment
    KO = {'LB': '왼쪽 풀백', 'LCB': '왼쪽 센터백', 'RCB': '오른쪽 센터백', 'RB': '오른쪽 풀백',
          'LDM': '왼쪽 수비형 미드', 'RDM': '오른쪽 수비형 미드', 'LAM': '왼쪽 윙', 'CAM': '공격형 미드',
          'RAM': '오른쪽 윙', 'ST': '스트라이커', 'LM': '왼쪽 미드', 'LCM': '왼쪽 중앙 미드',
          'RCM': '오른쪽 중앙 미드', 'RM': '오른쪽 미드', 'LST': '왼쪽 스트라이커', 'RST': '오른쪽 스트라이커'}
    B4 = [('LB', -15, -24, 0), ('LCB', -18, -8, 0), ('RCB', -18, 8, 0), ('RB', -15, 24, 0)]
    FORM = {'0': ('4-2-3-1', B4 + [('LDM', -6, -7, 1), ('RDM', -6, 7, 1), ('LAM', 6, -20, 2), ('CAM', 8, 0, 2), ('RAM', 6, 20, 2), ('ST', 18, 0, 2)]),
            '1': ('4-4-2', B4 + [('LM', -2, -22, 1), ('LCM', -4, -7, 1), ('RCM', -4, 7, 1), ('RM', -2, 22, 1), ('LST', 14, -6, 2), ('RST', 14, 6, 2)])}
    form, tp = FORM[t]
    nz = lambda A: (A - A.mean(0)) / A.std(0)
    P = np.array([[x, y] for _, x, y, _ in tp], float)
    cost = np.linalg.norm(nz(M)[:, None] - nz(P)[None], axis=2)
    _, col = linear_sum_assignment(cost)
    names = [KO[tp[col[j]][0]] + ' (' + tp[col[j]][0] + ')' for j in range(10)]
    band = np.array([tp[col[j]][3] for j in range(10)])
    stats = []
    for j in range(10):
        R = rel[role == j]
        stats.append({'n': int(len(R)), 'sx': round(float(R[:, 0].std()), 2), 'sy': round(float(R[:, 1].std()), 2),
                      'md': round(float(np.hypot(*(R - M[j]).T).mean()), 2)})
    per = []
    for a, b in ((0, 15), (15, 30), (30, 46)):
        mins = (fidx - 1) / 2 / 60
        cc = np.array([rel[(mins >= a) & (mins < b) & (role == j)].mean(0) if ((mins >= a) & (mins < b) & (role == j)).sum() > 20 else M[j] for j in range(10)])
        bl = [cline(cc, band, k) for k in range(3)]
        per.append({'depth': round(float(cc[:, 0].max() - cc[:, 0].min()), 2), 'width': round(float(cc[:, 1].max() - cc[:, 1].min()), 2),
                    'lines': [round(float(v), 2) for v in bl]})
    bl = [cline(M, band, k) for k in range(3)]
    # snapshots: moments where one slot is clearly away from its usual place
    snaps, last = [], -1e9
    cand = []
    frames = {fr['f']: fr for fr in d['frames']}
    for fr in d['frames']:
        x = fr[t]
        if not x or x['sh'] is None or sum(1 for a in x['a'] if a >= 0) < 5: continue
        best = None
        for p, a in zip(x['p'], x['a']):
            if a < 0: continue
            r = np.array([(p[0]-x['sh'][0])*fl, (p[1]-x['sh'][1])*fl]); dv = r - M[a]; dd = float(np.hypot(*dv))
            if 9 <= dd <= 22 and (best is None or dd > best[0]): best = (dd, a, dv)
        if best: cand.append((best[0], fr['f'], best[1], best[2]))
    for dd, f, a, dv in sorted(cand, reverse=True):
        if any(abs(f - s['f']) < 180 for s in snaps): continue
        x = frames[f][t]; y = frames[f][o]
        sh = [x['sh'][0]*fl, x['sh'][1]*fl]
        snaps.append({'f': f, 'slot': int(a), 'd': round(dd, 1), 'dx': round(float(dv[0]), 1), 'dy': round(float(dv[1]), 1), 'sh': sh,
                      'p': [[round(q[0]*fl, 1), round(q[1]*fl, 1), int(aa)] for q, aa in zip(x['p'], x['a'])],
                      'opp': [[round(q[0]*fl, 1), round(q[1]*fl, 1)] for q in (y['p'] if y else [])]})
        if len(snaps) == 8: break
    snaps.sort(key=lambda s: s['f'])
    ia = rng.choice(len(absP), min(7000, len(absP)), replace=False)
    ir = rng.choice(len(rel), min(9000, len(rel)), replace=False)
    out['teams'][t] = {'formation': form, 'M': np.round(M, 2).tolist(), 'names': names, 'band': band.tolist(),
                       'abs': np.round(absP[ia], 1).tolist(), 'rel': np.round(rel[ir], 1).tolist(), 'role': role[ir].tolist(),
                       'nAbs': int(len(absP)), 'nRel': int(len(rel)),
                       'nFrames': int(sum(1 for fr in d['frames'] if fr[t])),
                       'stats': stats, 'per': per, 'lines': [round(v, 2) for v in bl],
                       'depth': round(float(M[:, 0].max() - M[:, 0].min()), 2), 'width': round(float(M[:, 1].max() - M[:, 1].min()), 2),
                       'compact': round(float(np.hypot(M[:, 0]-cx, M[:, 1]-cy).mean()), 2),
                       'snaps': snaps}
    T = out['teams'][t]
    print(t, 'depth', T['depth'], 'width', T['width'], 'compact', T['compact'], 'lines', T['lines'])
    for j in range(10): print('   ', names[j], np.round(M[j], 1), stats[j])
    print('   per', [(p['depth'], p['width'], p['lines']) for p in per])
    print('   snaps', [(s['f'], (s['f']-1)//2//60, names[s['slot']], s['d'], s['dx'], s['dy'], len(s['p'])) for s in snaps])
json.dump(out, open(sys.argv[2], 'w'), separators=(',', ':'), ensure_ascii=False)
