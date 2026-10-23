import json, sys, numpy as np
from itertools import permutations
from scipy.optimize import linear_sum_assignment
S = json.load(open(sys.argv[1]))['teams']
B4 = [('LB', -15, -24, 'D'), ('LCB', -18, -8, 'D'), ('RCB', -18, 8, 'D'), ('RB', -15, 24, 'D')]
T = {
 '4-2-3-1': B4 + [('LDM', -6, -7, 'M'), ('RDM', -6, 7, 'M'), ('LAM', 6, -20, 'A'), ('CAM', 8, 0, 'A'), ('RAM', 6, 20, 'A'), ('ST', 18, 0, 'F')],
 '4-4-2':   B4 + [('LM', -2, -22, 'M'), ('LCM', -4, -7, 'M'), ('RCM', -4, 7, 'M'), ('RM', -2, 22, 'M'), ('LST', 14, -6, 'F'), ('RST', 14, 6, 'F')],
 '4-3-3':   B4 + [('LCM', -2, -12, 'M'), ('DM', -7, 0, 'M'), ('RCM', -2, 12, 'M'), ('LW', 14, -20, 'F'), ('ST', 17, 0, 'F'), ('RW', 14, 20, 'F')],
 '4-1-4-1': B4 + [('DM', -8, 0, 'M'), ('LM', 2, -22, 'M'), ('LCM', 0, -8, 'M'), ('RCM', 0, 8, 'M'), ('RM', 2, 22, 'M'), ('ST', 16, 0, 'F')],
 '3-5-2':   [('LCB', -18, -14, 'D'), ('CB', -20, 0, 'D'), ('RCB', -18, 14, 'D'), ('LWB', -4, -26, 'M'), ('LCM', -4, -9, 'M'), ('DM', -9, 0, 'M'), ('RCM', -4, 9, 'M'), ('RWB', -4, 26, 'M'), ('LST', 14, -6, 'F'), ('RST', 14, 6, 'F')],
 '3-4-3':   [('LCB', -18, -14, 'D'), ('CB', -20, 0, 'D'), ('RCB', -18, 14, 'D'), ('LM', -4, -24, 'M'), ('LCM', -5, -7, 'M'), ('RCM', -5, 7, 'M'), ('RM', -4, 24, 'M'), ('LW', 13, -18, 'F'), ('ST', 17, 0, 'F'), ('RW', 13, 18, 'F')],
 '5-3-2':   [('LWB', -12, -28, 'D'), ('LCB', -18, -12, 'D'), ('CB', -20, 0, 'D'), ('RCB', -18, 12, 'D'), ('RWB', -12, 28, 'D'), ('LCM', -4, -12, 'M'), ('CM', -6, 0, 'M'), ('RCM', -4, 12, 'M'), ('LST', 14, -6, 'F'), ('RST', 14, 6, 'F')],
}
def norm(A):
    A = A - A.mean(0); return A / A.std(0)   # per-axis scale: our shape is compressed differently in depth and width
TRUE = {'0': '4-2-3-1', '1': '4-4-2'}
NAME = {'0': '맨유', '1': '레스터'}
for t in '01':
    M = np.array(S[t]['M']); ours = S[t]['names']; band = S[t]['band']
    res = []
    for k, tp in T.items():
        P = np.array([[x, y] for _, x, y, _ in tp], float)
        C = np.linalg.norm(norm(M)[:, None] - norm(P)[None], axis=2)
        r, c = linear_sum_assignment(C)
        res.append((C[r, c].mean(), k, c))
    res.sort()
    print(f"\n== {NAME[t]} (실제 {TRUE[t]}) ==")
    for i, (d, k, _) in enumerate(res): print(f"  {i+1}. {k:8s} 거리 {d:.3f}" + ('  <- 실제' if k == TRUE[t] else ''))
    d, k, c = [x for x in res if x[1] == TRUE[t]][0]
    tp = T[TRUE[t]]
    print(f"  실제 포메이션({TRUE[t]})에 짝지은 결과:")
    agree = 0
    for j in np.argsort([-M[j, 0] for j in range(10)]):
        pos, _, _, grp = tp[c[j]]
        ob = ['D', 'M', 'F'][band[j]]
        ok = (ob == grp) or (ob == 'F' and grp in ('A', 'F')) or (ob == 'M' and grp == 'A')
        agree += ok
        print(f"    우리 '{ours[j]:8s}' ({M[j,0]:6.1f},{M[j,1]:6.1f}) -> 실제 {pos:4s} [{grp}] {'O' if ok else 'X'}")
    print(f"  수비/미드/공격 구분 일치: {agree}/10")
