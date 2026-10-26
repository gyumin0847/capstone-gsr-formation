"""stats.json -> self-contained feedback tool HTML.
usage: python analysis/make_tool.py stats.json web/template.html formation_tool.html [alt_stats.json ...]

alt_stats: stats.json files built from other, equally good role solutions (see reassign.py / README).
Their slots are aligned to the main solution (Hungarian on slot centres) and embedded, so the page can
show for every feedback row how many of the solutions agree ("확실" vs "경계")."""
import json, sys
import numpy as np
from scipy.optimize import linear_sum_assignment

stats, tpl, out = sys.argv[1:4]
main = json.load(open(stats, encoding='utf-8'))
alt = {t: [] for t in main['teams']}
for path in sys.argv[4:]:
    A = json.load(open(path, encoding='utf-8'))['teams']
    for t, T in main['teams'].items():
        M0, M1 = np.array(T['M']), np.array(A[t]['M'])
        if np.allclose(M0, M1, atol=0.05):          # this file did not change this team
            continue
        _, col = linear_sum_assignment(np.linalg.norm(M0[:, None] - M1[None], axis=2))
        pick = lambda X: [X[c] for c in col]
        alt[t].append({'M': pick(A[t]['M']), 'att': pick(A[t]['phase']['att']['M']), 'def': pick(A[t]['phase']['def']['M'])})
enc = lambda o: json.dumps(o, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
html = open(tpl, encoding='utf-8').read().replace('__DATA__', enc(main)).replace('__ALT__', enc(alt))
open(out, 'w', encoding='utf-8').write(html)
print('written:', out, '| alternative solutions per team:', {t: len(v) for t, v in alt.items()})
