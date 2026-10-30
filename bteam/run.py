"""B팀 v0 실험 한 번에 돌리기: 비교 기준 1~4 + Transformer (우리 팀 입력 A/B/C) -> 결과 표 + 미니맵 그림.

usage:
  # deep07: GPU 1장만 (연구실 규칙: 단일 GPU 작업 최대 2개)
  CUDA_VISIBLE_DEVICES=0 python bteam/run.py --data ~/data/bteam --out ~/results/bteam_v0
  python bteam/run.py --data ... --models mean ballshift knn          # 학습 없는 것만 빠르게
  python bteam/run.py --data ... --cv                                 # 7경기를 한 번씩 시험 경기로 (오래 걸림)
기본 나누기: 시험 J03WR9, 검증 J03WOY, 나머지 5경기 학습 (경기 단위).
출력: results.json, results.md (전체·상황별·팀 형태 표 + 합격 기준 판정), situations.png (상황별 막대),
      minimap_*.png (실제 vs 권장), 모델 가중치 *.pt
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as DT  # noqa: E402
import baselines as BL  # noqa: E402
import situations as SIT  # noqa: E402

ALL_MODELS = ['mean', 'ballshift', 'knn', 'mlp', 'tfA', 'tfB', 'tfC']
LABEL = {'mean': '1 포지션 평균', 'ballshift': '2 평균+공 이동', 'knn': '3 kNN', 'mlp': '4 MLP',
         'tfA': 'Transformer A (우리 팀 없음)', 'tfB': 'Transformer B (우리 팀 중심)',
         'tfC': 'Transformer C (나머지 9명)'}


def run_split(D, test, val, models, a, out_dir, log):
    tr, va, te = DT.split(D, test, val)
    log(f'scenes  train {len(tr["ball"])}  val {len(va["ball"])}  test {len(te["ball"])}  (test {test}, val {val})')
    res, preds = {}, {}
    for name in models:
        t0 = time.time()
        if name in ('mean', 'ballshift', 'knn'):
            m = {'mean': BL.Mean, 'ballshift': BL.BallShift, 'knn': BL.KNN}[name]().fit(tr)
            p = m.predict(te)
        else:
            import torch
            import model as M
            dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
            Ttr, Tva, Tte = (M.to_tensors(x, dev) for x in (tr, va, te))
            net = M.FlatMLP() if name == 'mlp' else M.PositionDecoder(name[-1], d=a.d, layers=a.layers)
            log(f'{name}: training on {dev}, {sum(p.numel() for p in net.parameters()) / 1e3:.0f}k parameters')
            net, best = M.train(net, Ttr, Tva, epochs=a.epochs, bs=a.bs, lr=a.lr, log=log)
            p = M.predict(net, Tte)
            torch.save(net.state_dict(), os.path.join(out_dir, f'{name}_{test[0]}.pt'))
        err = DT.errors(p, te)
        res[name] = DT.summary(err, te) | {'seconds': round(time.time() - t0, 1)}
        res[name]['shape_m'] = SIT.shape_errors(p, te['own'])
        res[name]['cells_m'] = by_group(err, te['sit_cell'], 12)
        res[name]['sides_m'] = by_group(err, te['sit_side'], 2)
        preds[name] = p
        log(f'{LABEL[name]:32s} test error {res[name]["all_m"]:5.2f} m  (attack {res[name]["attack_m"]}, '
            f'defence {res[name]["defence_m"]})  [{res[name]["seconds"]:.0f}s]')
    return res, preds, te


def by_group(err, g, n):
    return [round(float(err[g == k].mean()), 2) if (g == k).any() else None for k in range(n)]


def situation_table(res_by_split, counts):
    names = list(next(iter(res_by_split.values())).keys())
    avg = lambda n, key, k: np.mean([r[n][key][k] for r in res_by_split.values() if r[n][key][k] is not None] or [np.nan])
    out = ['## 상황별 평균 오차 (m)', '', '국면 x 공 구역 12칸. 장면 수는 시험 경기 기준.', '',
           '| 상황 | 장면 수 | ' + ' | '.join(LABEL[n] for n in names) + ' |', '|---|---|' + '---|' * len(names)]
    for c in range(12):
        out.append(f'| {SIT.cell_name(c)} | {counts["cell"][c]} | ' + ' | '.join(f'{avg(n, "cells_m", c):.2f}' for n in names) + ' |')
    out += ['', '측면 (공 위치 기준)', '', '| 측면 | 장면 수 | ' + ' | '.join(LABEL[n] for n in names) + ' |',
            '|---|---|' + '---|' * len(names)]
    for k in range(2):
        out.append(f'| {SIT.SIDES[k]} | {counts["side"][k]} | ' + ' | '.join(f'{avg(n, "sides_m", k):.2f}' for n in names) + ' |')
    out += ['', '## 팀 형태 오차 (m): 수비 라인 높이 / 팀 폭 / 팀 길이', '',
            '| 모델 | 라인 높이 | 폭 | 길이 |', '|---|---|---|---|']
    for n in names:
        f = lambda k: np.mean([r[n]['shape_m'][k] for r in res_by_split.values()])
        out.append(f'| {LABEL[n]} | {f("line_height"):.2f} | {f("width"):.2f} | {f("length"):.2f} |')
    return '\n'.join(out) + '\n'


def criteria(res_by_split):
    """합격 기준 (실험 전에 정해 둔 것): 모델이 '상황을 읽는다'고 말하려면"""
    names = list(next(iter(res_by_split.values())).keys())
    tfs = [n for n in ('tfA', 'tfB', 'tfC') if n in names]
    base = [n for n in ('mean', 'ballshift', 'knn', 'mlp') if n in names]
    if not tfs or not base:
        return '## 합격 기준\n\nTransformer와 비교 기준이 모두 있어야 판정합니다.\n'
    m = lambda n: np.mean([r[n]['all_m'] for r in res_by_split.values()])
    tf = 'tfA' if 'tfA' in tfs else tfs[0]
    best_base = min(base, key=m)
    lines = ['## 합격 기준 (Transformer A = 교수님 원안 기준)', '']
    ok1 = m(tf) < m(best_base)
    lines.append(f'1. 전체 오차가 가장 좋은 비교 기준({LABEL[best_base]} {m(best_base):.2f} m)보다 작다: '
                 f'{m(tf):.2f} m -> {"통과" if ok1 else "미달"}')
    wins = sum(r[tf]['all_m'] < min(r[b]['all_m'] for b in base) for r in res_by_split.values())
    lines.append(f'2. 시험 경기별로도 이긴다: {wins}/{len(res_by_split)} 경기 '
                 f'-> {"통과" if wins >= max(1, int(np.ceil(0.8 * len(res_by_split)))) else "미달"} (기준 80% 이상, --cv 로 7경기)')
    ref = 'ballshift' if 'ballshift' in names else best_base
    cells = [c for c in range(12) if all(r[tf]['cells_m'][c] is not None for r in res_by_split.values())]
    cw = sum(np.mean([r[tf]['cells_m'][c] for r in res_by_split.values()]) <
             np.mean([r[ref]['cells_m'][c] for r in res_by_split.values()]) for c in cells)
    lines.append(f'3. 상황 12칸 중 대부분에서 {LABEL[ref]}보다 작다: {cw}/{len(cells)} 칸 '
                 f'-> {"통과" if cw >= int(np.ceil(0.8 * len(cells))) else "미달"} (기준 80% 이상)')
    sh = lambda n: np.mean([sum(r[n]['shape_m'][k] for k in ('line_height', 'width', 'length')) for r in res_by_split.values()])
    ok4 = sh(tf) < min(sh(b) for b in base)
    lines.append(f'4. 팀 형태(라인 높이+폭+길이) 오차도 비교 기준보다 작다: {sh(tf):.2f} vs {min(sh(b) for b in base):.2f} '
                 f'-> {"통과" if ok4 else "미달"}')
    return '\n'.join(lines) + '\n'


def situation_figure(res_by_split, path, names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    en = {'공격': 'attack', '수비': 'defence', '공격 전환': 'won ball', '수비 전환': 'lost ball',
          '우리 진영': 'own third', '중앙': 'middle', '상대 진영': 'opp. third'}
    lab = [f'{en[SIT.PHASES[c // 3]]}\n{en[SIT.ZONES[c % 3]]}' for c in range(12)]
    names = [n for n in names if n != 'mean'] or names    # the role-mean bars dwarf the rest (see results.md)
    fig, ax = plt.subplots(figsize=(13, 3.8))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        v = [np.mean([r[n]['cells_m'][c] for r in res_by_split.values() if r[n]['cells_m'][c] is not None] or [np.nan])
             for c in range(12)]
        ax.bar(np.arange(12) + (i - (len(names) - 1) / 2) * w, v, w, label=n)
    ax.set_xticks(range(12)); ax.set_xticklabels(lab, fontsize=8); ax.set_ylabel('mean error (m)')
    ax.legend(fontsize=8, ncol=len(names), loc='upper center', bbox_to_anchor=(0.5, -0.22), frameon=False)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('error by situation (phase x ball zone), role-mean baseline left out - lower is better', fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def table(res_by_split):
    names = list(next(iter(res_by_split.values())).keys())
    lines = ['| 모델 | 전체 (m) | 공격 (m) | 수비 (m) | 중앙값 (m) |', '|---|---|---|---|---|']
    for n in names:
        rows = [r[n] for r in res_by_split.values()]
        f = lambda k: f'{np.mean([r[k] for r in rows]):.2f}'
        lines.append(f'| {LABEL[n]} | {f("all_m")} | {f("attack_m")} | {f("defence_m")} | {f("median_m")} |')
    per = ['', '포지션별 평균 오차 (m), 0번 = 가장 뒤 라인의 -y 쪽 … 9번 = 가장 앞 라인의 +y 쪽', '',
           '| 모델 | ' + ' | '.join(str(k) for k in range(10)) + ' |', '|---|' + '---|' * 10]
    for n in names:
        v = np.mean([r[n]['per_role_m'] for r in res_by_split.values()], 0)
        per.append(f'| {LABEL[n]} | ' + ' | '.join(f'{x:.1f}' for x in v) + ' |')
    return '\n'.join(lines + per) + '\n'


def minimap(te, preds, path, best, n=4, seed=0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    rng = np.random.default_rng(seed)
    ph = te['phase'].astype(bool)
    pick = list(rng.choice(np.where(ph)[0], n // 2, replace=False)) + list(rng.choice(np.where(~ph)[0], n - n // 2, replace=False))
    fig, axs = plt.subplots(1, n, figsize=(4.6 * n, 3.4))
    for ax, i in zip(np.atleast_1d(axs), pick):
        ax.add_patch(Rectangle((-52.5, -34), 105, 68, fill=False, lw=1, color='#888'))
        ax.plot([0, 0], [-34, 34], color='#888', lw=0.8); ax.add_patch(Circle((0, 0), 9.15, fill=False, color='#888', lw=0.8))
        for sx in (-1, 1):
            ax.add_patch(Rectangle((sx * 52.5 - (16.5 if sx > 0 else 0), -20.16), 16.5, 40.32, fill=False, color='#888', lw=0.8))
        o = te['opp_filled'][i]; own = te['own'][i]; p = preds[best][i]; m0 = preds['mean'][i] if 'mean' in preds else None
        ax.scatter(o[:, 0], o[:, 1], marker='x', c='#c0392b', s=28, label='opponent')
        ax.scatter(own[:, 0], own[:, 1], c='#1f5fa8', s=26, label='ours (actual)')
        ax.scatter(p[:, 0], p[:, 1], facecolors='none', edgecolors='#1f5fa8', s=70, lw=1.4, label=f'suggested ({best})')
        for k in range(10):
            ax.plot([own[k, 0], p[k, 0]], [own[k, 1], p[k, 1]], color='#1f5fa8', lw=0.7, alpha=0.6)
        if m0 is not None:
            ax.scatter(m0[:, 0], m0[:, 1], marker='+', c='#999', s=30, label='role mean')
        ax.scatter(*te['ball'][i], c='k', s=20, zorder=5, label='ball')
        e = np.hypot(*(p - own).T).mean()
        ax.set_title(f'{"attack" if ph[i] else "defence"} · {te["match"][i]} t={te["t"][i]:.0f}s · mean gap {e:.1f} m', fontsize=9)
        ax.set_xlim(-55, 55); ax.set_ylim(-36, 36); ax.set_aspect('equal'); ax.axis('off')
    np.atleast_1d(axs)[0].legend(loc='lower left', fontsize=7, framealpha=0.8)
    fig.suptitle('our team attacks to the right (+x) - dot = actual, ring = model suggestion', fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--out', default='results/bteam_v0')
    ap.add_argument('--models', nargs='+', default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument('--test', nargs='+', default=['J03WR9']); ap.add_argument('--val', nargs='+', default=['J03WOY'])
    ap.add_argument('--cv', action='store_true', help='each match once as test (val = the next match)')
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--layers', type=int, default=3); ap.add_argument('--device', default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, 'log.txt'), 'a')

    def log(s):
        print(s, flush=True); logf.write(s + '\n'); logf.flush()

    D = DT.load(a.data)
    ms = sorted(set(D['match'].tolist()))
    log(f'{len(D["ball"])} scenes from {len(ms)} matches: {ms}')
    splits = [([m], [ms[(i + 1) % len(ms)]]) for i, m in enumerate(ms)] if a.cv else [(a.test, a.val)]
    allres = {}
    for test, val in splits:
        res, preds, te = run_split(D, test, val, a.models, a, a.out, log)
        allres[test[0]] = res
        trained = [m for m in ('tfA', 'tfB', 'tfC', 'mlp', 'knn', 'ballshift') if m in preds]
        if trained:
            best = min(trained, key=lambda m: res[m]['all_m'])
            minimap(te, preds, os.path.join(a.out, f'minimap_{test[0]}.png'), best)
    json.dump(allres, open(os.path.join(a.out, 'results.json'), 'w'), indent=1, ensure_ascii=False)
    te_all = DT.subset(D, np.isin(D['match'], list(allres)))
    counts = {'cell': np.bincount(te_all['sit_cell'], minlength=12).tolist(),
              'side': np.bincount(te_all['sit_side'], minlength=2).tolist()}
    md = '# 전체\n\n' + table(allres) + '\n' + situation_table(allres, counts) + '\n' + criteria(allres)
    open(os.path.join(a.out, 'results.md'), 'w').write(md)
    situation_figure(allres, os.path.join(a.out, 'situations.png'), a.models)
    log('\n' + md)
    log(f'written: {a.out}/results.json, results.md, situations.png, minimap_*.png')


if __name__ == '__main__':
    main()
