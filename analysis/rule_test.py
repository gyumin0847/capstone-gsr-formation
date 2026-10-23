import json, sys, numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, '')
exec(open(sys.argv[2]).read().split("def norm")[0])            # reuse templates T, B4 from check_lineup.py
def norm(A): A = A - A.mean(0); return A / A.std(0)
S = json.load(open(sys.argv[1]))['teams']
TRUE = {'0': '4-2-3-1', '1': '4-4-2'}
def truth(M, k):
    tp = T[k]; P = np.array([[x, y] for _, x, y, _ in tp], float)
    C = np.linalg.norm(norm(M)[:, None] - norm(P)[None], axis=2); r, c = linear_sum_assignment(C)
    g = {'D': 0, 'M': 1, 'A': 2, 'F': 2}
    return [g[tp[c[j]][3]] for j in range(10)]
def rank_band(M):
    rk = np.argsort(np.argsort(M[:, 0])); return list(np.where(rk < 4, 0, np.where(rk < 7, 1, 2)))
for t in '01':
    M = np.array(S[t]['M']); gt = truth(M, TRUE[t])
    Md = np.array(S[t]['phase']['def']['M'])
    now = rank_band(M); a = rank_band(Md)
    # B: by construction equals gt on overall shape; test robustness on def-phase shape
    b_def = truth(Md, TRUE[t])
    acc = lambda x: sum(int(u == v) for u, v in zip(x, gt))
    print(t, 'now', acc(now), '/10 | A(def-phase rank)', acc(a), '/10 | B on def-phase shape', acc(b_def), '/10')
