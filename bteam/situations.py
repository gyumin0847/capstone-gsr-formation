"""상황 구분 기준 (규민 담당) — 모델에 들어가는 정보만으로 장면을 나눈다.

원칙: 우리 팀 위치는 모델이 맞혀야 할 '정답'이므로 상황을 나누는 데 쓰지 않는다.
      (우리 수비 라인 높이로 상황을 나누면 답을 보고 문제를 고르는 셈) -> 공, 볼 소유, 시간만 사용.
      우리 라인 높이·팀 폭·길이는 대신 '채점 항목'(shape_errors)으로 쓴다.

기준 1 국면 (볼 소유 + 직전 변화)
  공격          우리 공, 소유가 바뀐 지 4초 넘음
  수비          상대 공, 소유가 바뀐 지 4초 넘음
  공격 전환     우리가 공을 뺏은 지 4초 이내 (역습 시작)
  수비 전환     공을 뺏긴 지 4초 이내 (다시 압박 / 복귀)
  * 사이에 공이 경기장 밖에 나간 구간(데드볼)이 있으면 전환이 아니라 재개로 보고 공격/수비로 둔다.
기준 2 공 구역 (경기장 길이 3등분, 우리가 +x로 공격)
  우리 진영 x < -17.5 m / 중앙 / 상대 진영 x > 17.5 m
기준 3 측면 (경기장 폭 3등분)
  중앙 |y| <= 11.3 m / 측면 |y| > 11.3 m
  * 왼쪽·오른쪽은 전술적으로 대칭이고, 데이터의 y축 방향(어느 쪽이 왼쪽인지)에 따라 이름이 바뀌므로 합쳐서 본다.
기본 표는 국면 x 공 구역 = 12칸, 측면은 따로 본다.
"""
import numpy as np

PHASES = ['공격', '수비', '공격 전환', '수비 전환']
ZONES = ['우리 진영', '중앙', '상대 진영']
SIDES = ['중앙', '측면']
TRANSITION_S = 4.0
DEAD_GAP_S = 1.0            # a jump in t larger than this = ball was dead in between (prepare drops dead frames)
THIRD_X = 105 / 6           # 17.5 m
THIRD_Y = 68 / 6            # 11.33 m


def since_change(phase, t):
    """seconds since possession last changed in live play (inf if never / interrupted by a dead ball).
    phase, t: arrays of ONE half and one team, in time order, before any filtering"""
    out = np.full(len(phase), np.inf)
    last = None
    for i in range(len(phase)):
        if i > 0:
            if t[i] - t[i - 1] > DEAD_GAP_S:
                last = None                   # restart after a dead ball: not a transition
            elif phase[i] != phase[i - 1]:
                last = t[i]                   # ball won / lost in open play
        if last is not None:
            out[i] = t[i] - last
    return out


def label(D):
    """adds integer codes 'sit_phase' (0..3), 'sit_zone' (0..2), 'sit_side' (0 centre, 1 wing) and 'sit_cell' (0..11) to D"""
    trans = D['since_change'] <= TRANSITION_S
    own = D['phase'].astype(bool)
    ph = np.where(own, np.where(trans, 2, 0), np.where(trans, 3, 1))
    bx, by = D['ball'][:, 0], D['ball'][:, 1]
    zone = np.digitize(bx, [-THIRD_X, THIRD_X])
    side = (np.abs(by) > THIRD_Y).astype(int)
    D['sit_phase'], D['sit_zone'], D['sit_side'] = ph, zone, side
    D['sit_cell'] = ph * 3 + zone
    return D


def cell_name(c):
    return f'{PHASES[c // 3]} · {ZONES[c % 3]}'


# ------------------------------------------------------------------ team-shape scoring (채점 항목)
def shape(P):
    """P: (N, 10, 2) -> back-line height (mean x of the 4 deepest), width (y range), length (x range), centre x"""
    xs = np.sort(P[..., 0], axis=1)
    return {'line_height': xs[:, :4].mean(1),
            'width': P[..., 1].max(1) - P[..., 1].min(1),
            'length': P[..., 0].max(1) - P[..., 0].min(1),
            'centre_x': P[..., 0].mean(1)}


def shape_errors(pred, actual):
    """mean absolute error (m) of each team-shape number: does the model get the block height / compactness right?"""
    a, p = shape(actual), shape(pred)
    return {k: round(float(np.abs(p[k] - a[k]).mean()), 2) for k in a}
