"""학습된 Transformer로 경기 한 구간의 ghost 영상 만들기 (장면마다 계산 -> 화면에서만 부드럽게 이어 붙임).

우리 방식 그대로:
  1. 2 fps 장면마다 모델이 우리 10명의 위치(ghost)를 따로 계산한다 (앞뒤 장면을 보지 않음).
  2. 화면용으로만 ghost를 부드럽게 한다: 이전 ghost 쪽으로 alpha 만큼만 따라가고, 한 장면에 max-speed(m/s)보다
     빨리 못 움직이게 자른다 (지금 시점까지만 보는 방식이라 실시간에도 그대로 쓸 수 있음).
  3. ghost는 상대 공일 때만 그린다. 피드백(빨간색)은 '자리 잡은 수비'(공 뺏긴 지 4초 이후)에서만:
     실제 선수와 (부드럽게 하기 전) ghost 거리가 그 자리 기준(검증 경기에서 잰 상위 20% 거리)보다 큰 상태가 2초 이상 이어지면 빨간색.
  4. 장면 사이(0.5초)는 선형으로 채워서 fps 로 그린다. 데드볼로 끊긴 곳은 이어 붙이지 않는다.

모델 파일은 run.py 가 저장한 것 (<모델>_<시험 경기>.pt). 그 경기를 학습에 안 쓴 모델이므로 그 경기로 영상을 만든다.

usage (deep07, GPU 없어도 됨):
  python bteam/ghost_video.py --data ~/data/bteam --model ~/results/bteam_v0/tfC_J03WR9.pt \
      --seg firstHalf --team Home --start 600 --seconds 90 --out ~/results/ghost_J03WR9.mp4
  # --val 경기: 피드백 기준을 잴 경기 (기본: run.py --cv 와 같은 '다음 경기')
  # --alert-m 8 : 기준을 숫자 하나로 고정 (검증 경기 계산 생략)
"""
import argparse, glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as DT  # noqa: E402

GAP_S = 0.75            # scenes further apart than this (s) are not joined (dead ball in between)
ALERT_S = 2.0           # deviation must last this long before it is flagged
SETTLED = 1             # situations.label: sit_phase 1 = 수비 (상대 공, 뺏긴 지 4초 이후)


def load_model(path, variant, d, layers, device):
    import torch
    import model as M
    net = M.PositionDecoder(variant, d=d, layers=layers)
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return net.to(device).eval()


def predict(net, D, device):
    import model as M
    return M.predict(net, M.to_tensors(D, device))


def one_file(data_dir, mid, seg, team):
    """scenes of one match half and team, in time order (the 'group' index of data.load)"""
    files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    name = f'{mid}_{seg}_{team}.npz'
    gi = [os.path.basename(f) for f in files].index(name)
    D = DT.load(data_dir, matches=[mid])
    S = DT.subset(D, D['group'] == gi)
    return DT.subset(S, np.argsort(S['t'], kind='stable'))


def smooth(G, t, alpha, vmax):
    """causal: follow the new scene-wise ghost by alpha, never faster than vmax m/s; restart after a dead ball"""
    out = G.copy()
    for i in range(1, len(G)):
        dt = t[i] - t[i - 1]
        if dt > GAP_S:
            continue
        step = alpha * (G[i] - out[i - 1])
        n = np.linalg.norm(step, axis=-1, keepdims=True)
        out[i] = out[i - 1] + step * np.minimum(1.0, vmax * dt / np.maximum(n, 1e-9))
    return out


def alerts(gap, t, active, thr):
    """(N, 10) bool: gap > thr[role] continuously for ALERT_S seconds, only where active (settled defence)"""
    over = (gap > thr[None]) & active[:, None]
    flag = np.zeros_like(over); start = np.full(10, np.nan)
    for i in range(len(t)):
        if i > 0 and t[i] - t[i - 1] > GAP_S:
            start[:] = np.nan
        for k in range(10):
            if over[i, k]:
                if np.isnan(start[k]):
                    start[k] = t[i]
                flag[i, k] = t[i] - start[k] >= ALERT_S - 1e-6
            else:
                start[k] = np.nan
    return flag


def frames(t, sub):
    """(scene index, weight to the next scene) for every video frame; dead-ball gaps are cut, not filled"""
    out = []
    for i in range(len(t)):
        joined = i + 1 < len(t) and t[i + 1] - t[i] <= GAP_S
        out += [(i, k / sub) for k in range(sub)] if joined else [(i, 0.0)] * sub
    return out


def lerp(A, i, w):
    return A[i] if w == 0 else (1 - w) * A[i] + w * A[i + 1]


def render(S, ghost, flag, show, out, fps, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Rectangle, Circle

    fig, ax = plt.subplots(figsize=(10, 6.9))
    fig.subplots_adjust(0.01, 0.07, 0.99, 0.92)
    ax.add_patch(Rectangle((-52.5, -34), 105, 68, fill=False, lw=1.2, color='#888'))
    ax.plot([0, 0], [-34, 34], color='#888', lw=1); ax.add_patch(Circle((0, 0), 9.15, fill=False, color='#888', lw=1))
    for sx in (-1, 1):
        ax.add_patch(Rectangle((sx * 52.5 - (16.5 if sx > 0 else 0), -20.16), 16.5, 40.32, fill=False, color='#888', lw=1))
    ax.set_xlim(-55, 55); ax.set_ylim(-36, 36); ax.set_aspect('equal'); ax.axis('off')
    opp = ax.scatter([], [], marker='x', c='#c0392b', s=55, lw=2, zorder=3)
    own = ax.scatter([], [], c='#1f5fa8', s=60, zorder=4)
    gh = ax.scatter([], [], facecolors='none', edgecolors='#1f5fa8', s=150, lw=1.8, zorder=4)
    lines = [ax.plot([], [], lw=1.4, color='#1f5fa8', alpha=0.7, zorder=2)[0] for _ in range(10)]
    ball = ax.scatter([], [], c='k', s=40, zorder=6)
    head = fig.text(0.5, 0.955, '', ha='center', fontsize=12)
    foot = fig.text(0.5, 0.025, 'dot = actual   ring = ghost (scene by scene, smoothed)   x = opponent   '
                    'red = off the pro range for 2 s+ (settled defence)   we attack ->', ha='center', fontsize=9, color='#555')
    plan = frames(S['t'], max(1, int(round(fps * 0.5))))
    ph_name = {0: 'attack', 1: 'defence', 2: 'won ball', 3: 'lost ball'}

    def draw(j):
        i, w = plan[j]
        P = lerp(S['own'], i, w); O = lerp(S['opp_filled'], i, w); B = lerp(S['ball'], i, w); G = lerp(ghost, i, w)
        red = flag[i]
        own.set_offsets(P); opp.set_offsets(O); ball.set_offsets(B[None])
        own.set_color(np.where(red[:, None], [[0.75, 0.22, 0.17, 1]], [[0.12, 0.37, 0.66, 1]]))
        if show[i]:
            gh.set_offsets(G)
            for k, ln in enumerate(lines):
                ln.set_data([P[k, 0], G[k, 0]], [P[k, 1], G[k, 1]]); ln.set_color('#c0392b' if red[k] else '#1f5fa8')
            gap = np.hypot(*(P - G).T).mean()
            extra = f'mean gap {gap:.1f} m' + (f'  -  {int(red.sum())} flagged' if red.any() else '')
        else:
            gh.set_offsets(np.empty((0, 2)))
            for ln in lines:
                ln.set_data([], [])
            extra = 'our ball - no ghost'
        tt = S['t'][i] + w * 0.5
        head.set_text(f'{title}   {int(tt // 60):02d}:{tt % 60:04.1f}   {ph_name[int(S["sit_phase"][i])]}   {extra}')
        return [own, opp, ball, gh, head, *lines]

    anim = animation.FuncAnimation(fig, draw, frames=len(plan), blit=False)
    try:                                       # no system ffmpeg in the lab container: pip install imageio-ffmpeg
        import imageio_ffmpeg
        matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    if out.endswith('.gif') or not animation.writers.is_available('ffmpeg'):
        out = os.path.splitext(out)[0] + '.gif'
        anim.save(out, writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=2400, extra_args=['-pix_fmt', 'yuv420p']))
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--model', required=True, help='e.g. results/bteam_v0/tfC_J03WR9.pt')
    ap.add_argument('--match', default=None, help='default: the test match in the model file name')
    ap.add_argument('--variant', default=None, help='A/B/C, default: from the model file name (tfC_... -> C)')
    ap.add_argument('--seg', default='firstHalf'); ap.add_argument('--team', default='Home')
    ap.add_argument('--start', type=float, default=0.0, help='seconds from the start of the half')
    ap.add_argument('--seconds', type=float, default=90.0)
    ap.add_argument('--val', default=None, help='match used for the per-role thresholds (default: the next match, as run.py --cv)')
    ap.add_argument('--alert-m', type=float, default=None, help='fixed threshold in m instead of the validation-match 80th percentile')
    ap.add_argument('--alpha', type=float, default=0.5); ap.add_argument('--max-speed', type=float, default=10.0)
    ap.add_argument('--fps', type=int, default=10); ap.add_argument('--out', default='ghost.mp4')
    ap.add_argument('--d', type=int, default=128); ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()

    stem = os.path.splitext(os.path.basename(a.model))[0]            # tfC_J03WR9
    mid = a.match or stem.split('_', 1)[1]
    variant = a.variant or stem.split('_', 1)[0][-1]
    net = load_model(a.model, variant, a.d, a.layers, a.device)

    S = one_file(a.data, mid, a.seg, a.team)
    S = DT.subset(S, (S['t'] >= a.start) & (S['t'] < a.start + a.seconds))
    if len(S['t']) < 2:
        raise SystemExit('no scenes in that window (check --seg/--team/--start)')
    raw = predict(net, S, a.device)
    ghost = smooth(raw, S['t'], a.alpha, a.max_speed)
    show = S['phase'] == 0                                             # opponent ball
    settled = S['sit_phase'] == SETTLED

    if a.alert_m is not None:
        thr = np.full(10, a.alert_m)
    else:
        ms = sorted({os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(a.data, '*.npz'))})
        val = a.val or ms[(ms.index(mid) + 1) % len(ms)]
        V = DT.load(a.data, matches=[val])
        V = DT.subset(V, V['sit_phase'] == SETTLED)
        thr = np.percentile(DT.errors(predict(net, V, a.device), V), 80, axis=0)
        print(f'thresholds from {val} (settled defence, {len(V["t"])} scenes, 80th percentile per role): '
              + ' '.join(f'{x:.1f}' for x in thr))
    gap = np.hypot(*(S['own'] - raw).transpose(2, 0, 1))             # feedback uses the scene-wise ghost; smoothing is display only
    flag = alerts(gap, S['t'], settled, thr)

    out = render(S, ghost, flag, show, os.path.expanduser(a.out), a.fps, f'{mid} {a.seg} {a.team} (model {stem})')
    n_def = int(settled.sum())
    print(f'{len(S["t"])} scenes ({n_def} settled defence). written: {out}')
    if n_def:
        print('flagged share of settled-defence time per role: '
              + ' '.join(f'{k}:{100 * flag[settled, k].mean():.0f}%' for k in range(10)))


if __name__ == '__main__':
    main()
