import json, sys, numpy as np
d = json.load(open(sys.argv[1])); S = json.load(open(sys.argv[2]))
FLIP = {'0': 1.0, '1': -1.0}; TH = 5.0

def cline(C, band, k):
    """line height from central players only (wide players excluded), as analysts measure line height"""
    idx = band == k; cy = C[:, 1].mean()
    sel = idx & (np.abs(C[:, 1] - cy) <= 12)
    if sel.sum() == 0: sel = idx
    return float(C[sel, 0].mean())
for t in '01':
    fl = FLIP[t]; T = S['teams'][t]; M = np.array(T['M']); band = np.array(T['band'])
    buck = {'att': [], 'def': []}; nf = {'att': 0, 'def': 0, 'mid': 0}
    for fr in d['frames']:
        x = fr[t]
        if not x or x['sh'] is None: continue
        cx = x['sh'][0] * fl + M[:, 0].mean()
        ph = 'att' if cx > TH else ('def' if cx < -TH else 'mid'); nf[ph] += 1
        if ph == 'mid': continue
        for p, a in zip(x['p'], x['a']):
            if a >= 0: buck[ph].append(((p[0]-x['sh'][0])*fl, (p[1]-x['sh'][1])*fl, a))
    res = {}
    for ph in ('att', 'def'):
        A = np.array(buck[ph])
        C = np.array([A[A[:, 2] == j, :2].mean(0) if (A[:, 2] == j).sum() > 30 else M[j] for j in range(10)])
        cnt = [int((A[:, 2] == j).sum()) for j in range(10)]
        lines = [cline(C, band, k) for k in range(3)]
        f = int(np.argmax(C[:, 0])); iso = min(np.hypot(*(C[j]-C[f])) for j in range(10) if j != f)
        cx, cy = C.mean(0)
        res[ph] = {'frames': nf[ph], 'M': np.round(C, 2).tolist(), 'cnt': cnt,
                   'depth': round(float(C[:, 0].max()-C[:, 0].min()), 2), 'width': round(float(C[:, 1].max()-C[:, 1].min()), 2),
                   'gaps': [round(lines[1]-lines[0], 2), round(lines[2]-lines[1], 2)], 'lines': [round(v, 2) for v in lines],
                   'iso': round(float(iso), 2), 'compact': round(float(np.hypot(C[:, 0]-cx, C[:, 1]-cy).mean()), 2)}
    T['phase'] = res; T['phaseMid'] = nf['mid']
    print(t, nf, {ph: {k: v for k, v in r.items() if k not in ('M', 'cnt')} for ph, r in res.items()})
    print('   cnt att', res['att']['cnt'], ' def', res['def']['cnt'])
json.dump(S, open(sys.argv[2], 'w'), separators=(',', ':'), ensure_ascii=False)
