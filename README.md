# 중계 영상 → 포지션별 전술 피드백 (캡스톤1 · 1팀 GSR)

중계 영상에서 뽑은 선수 좌표로 **팀별 자리 10개**를 찾고, 입력한 **포메이션에 짝지어 포지션 이름**을 붙인 뒤, **공격/수비 상황별 포지션 피드백**을 주는 코드.

```
사진(img1/*.jpg) → [SoccernetGSR] → 선수 박스·추적 ID
  → [pipeline] 경기장 좌표 변환 → 팀 나누기 → 자리 10개 → 프레임별 팀 이동/짝짓기 (viewer json)
  → [analysis] 포지션 이름 → 공격/수비 통계 → 피드백 HTML
```

> ⚠️ **이 저장소는 public. 코드만 올린다.**
> - SoccerNet NDA(연구·비상업, 재배포 금지): 사진·영상·보정 파일·결과 좌표(json/csv/npy)·결과 영상은 **절대 커밋 금지**. 각자 NDA 신청 후 직접 받기.
> - 토큰·비밀번호를 코드에 쓰지 말 것. HF 토큰은 `hf auth login` 프롬프트로만 입력.
> - `git add -f` 금지. 한 번 올라간 파일은 지워도 기록에 남는다. 커밋 전 `git status`로 확인.

## 데이터 받기
- Hugging Face `SoccerNet/SoccerNet_raw_HQ`, 브랜치 **`frames-720p-2fps`** (1초 2장 사진)
- 예시 경기: `england_epl/2016-2017/2016-09-24 - 14-30 Manchester United 4 - 1 Leicester`, 전반 → 사진 5,400장, 약 700 MB
- 카메라 보정: 같은 경기 폴더의 `1_field_calib_ccbv.json`
- GSR 코드: [yinmayoo185/SoccernetGSR](https://github.com/yinmayoo185/SoccernetGSR) (사진 폴더 `<set>/<seq>/img1/*.jpg` 입력)

## 폴더
| 폴더 | 파일 | 하는 일 |
|---|---|---|
| `pipeline/` | `run_manutd.sh` | 사진 받기 → 압축 풀기 → GSR 폴더 형식 연결 → GSR 추론 |
| | `genoa_formation.py` | 보정 파일로 화면 좌표 → 경기장 좌표(m) (`load_calib`, `get_H`) |
| | `team_hist.py` | 유니폼 색 히스토그램 + 2-means로 두 팀 나누기 → `*_teams.json` |
| | `role_assign.py` | **Bialkowski 2014** 방식 자리 10개 (팀 이동 빼기 + 헝가리안 반복) |
| | `compare_gt_pred.py` | GSR 정답 클립으로 자리 위치 검증 (맨유 평균 2.5 m) |
| | `manutd_formation.py` | 팀별 자리 그림 |
| | `manutd_v2.py` | 프레임별 팀 이동·짝짓기 → `manutd_viewer.json` + 중계 오버레이 영상 |
| `analysis/` | `build.py` | 자리 10개 ↔ 포메이션 템플릿 헝가리안 짝짓기(축별 z-score) → 포지션 이름, 라인 높이 → `stats.json` |
| | `phase.py` | 공격/수비 나누기(팀 중심 x > +5 m 공격, < −5 m 수비) → 상황별 위치·간격 |
| | `check_lineup.py`, `rule_test.py` | 실제 라인업과 포지션 이름 비교, 이름 규칙 비교 |
| | `make_tool.py` | `stats.json`을 템플릿에 넣어 피드백 HTML 생성 |
| | `reassign.py` | EM 역할 분포(여러 번 시작해 가장 잘 맞는 해)로 프레임별 역할 배정 `a`와 팀 이동 `sh`를 다시 계산 → build/phase/make_tool을 그대로 다시 돌리면 됨 |
| | `impute_eval.py` | 화면 밖 선수 추정 검증 (보이는 선수를 일부러 가리고 오차 측정) |
| | `role_distributions.py` | **Bialkowski 2014 원형 구현.** 역할 10개를 가우시안 분포(μ, Σ)로 두고, 보이는 선수만으로 팀 이동(중앙값)을 뺀 뒤 −log 확률 비용 헝가리안으로 매칭 → 분포 재계산(EM)을 수렴까지 반복. 포메이션 템플릿으로 이름 붙이기(`name_roles`) 포함 |
| `web/` | `template.html` | 포메이션 입력 → 자동 매핑 → 포지션 피드백 → 코칭 보드 |

## 실행 순서 (deep07 컨테이너, `(gsr)` 환경)
```bash
# 1) 데이터 + GSR 추론 (경로는 스크립트 맨 위 변수 수정)
bash pipeline/run_manutd.sh
# 2) 팀 나누기
python pipeline/team_hist.py <GSR 결과 txt> "<사진 경로>/%08d.jpg" manutd_teams.json
# 3) 자리 10개 + 프레임별 짝짓기 (START, N = 오버레이 영상 구간)
python pipeline/manutd_v2.py 1200 600
# 4) 포지션 이름 + 공격/수비 통계
python analysis/build.py manutd_viewer.json stats.json
python analysis/phase.py manutd_viewer.json stats.json
# 4-1) (권장) 역할 배정을 EM 최적 해로 다시 계산한 뒤 4)를 그 파일로 실행
python analysis/reassign.py manutd_viewer.json manutd_viewer_em.json
# 4-2) (선택) 역할 분포 μ, Σ만 따로 계산 (논문 원형, 타원 시각화용)
python analysis/role_distributions.py manutd_viewer.json roles.json
# 5) 피드백 HTML
python analysis/make_tool.py stats.json web/template.html formation_tool.html
```
- 포메이션은 `analysis/build.py`의 `FORM`에서 팀별로 입력 (예: 맨유 4-2-3-1, 레스터 4-4-2).
- 다른 경기: 사진·보정 파일 경로와 `FORM`만 바꾸면 됨.

## 피드백 기준 (현재 규칙 기반, 추후 경기 데이터로 기준 학습 예정)
| 포지션 | 상황 | 기준 |
|---|---|---|
| 풀백 | 수비 | 센터백 라인보다 5 m 넘게 앞 |
| 풀백 | 공격 | 미드 라인보다 6 m 넘게 뒤 |
| 센터백 | 공통 / 공격 | 두 센터백 간격 16 m 초과 / 중앙 미드와 9 m 초과 |
| 수비·중앙 미드 | 수비 / 공격 | 센터백과 8 m 초과 / 가장 가까운 공격수와 15 m 초과 |
| 윙·측면 미드 | 공격 / 수비 | 중앙에서 12 m 미만(좁음) / 미드 라인보다 6 m 넘게 앞 |
| 공격형 미드 | 공통 | 스트라이커와 12 m 초과 |
| 스트라이커 | 공통 | 가장 가까운 동료와 11 m 초과(고립) |

## 역할 분포 실험 결과 (맨유 vs 레스터 전반, `role_distributions.py`)
- 논문처럼 "엉킨 초기 분포 → 반복 → 분리된 10개 분포"가 중계 데이터에서도 나온다 (반복 40회, 마지막 이동량 0.1 m 이하).
- **결과가 시작값에 따라 달라질 수 있다.** 45분 데이터라도 시작값을 바꾸면 2~4 m 다른 해가 나온다. 여러 번 돌려 가장 잘 맞는 것을 고르고(`best_fit`), 좋은 해끼리는 포지션 이름이 10/10 같았다.
- **짧은 구간:** 짧은 구간만으로 처음부터 계산하면 30분이어도 45분 결과와 2 m 정도 다르고, 포지션 이름 일치율도 30~87%로 들쭉날쭉하다. 앞선 결과를 시작값으로 쓰면 5분 구간에서 이름 100%, 위치 차이 1.6~2.5 m였다. → 권장: 하프 전체로 한 번 계산하고, 그 결과를 시작값으로 5분 단위 변화를 추적.
- **크기 확인:** 모델이 그리는 팀 크기(맨유 앞뒤 표준편차 6.8 m, 좌우 10.2 m)는 한 팀 10명이 다 보인 장면(7.0 m, 10.3 m)과 거의 같다. 모델이 모양을 줄이는 것은 아니고, 중계 화면에 잡히는 장면 자체가 촘촘한 장면 위주라 팀이 작게 보인다 (센터백→스트라이커 10~18 m).
- **포지션 이름 확신도:** 역할 위치를 1.5 m씩 흔들어 200번 다시 이름을 붙여, 같은 이름이 나오는 비율을 잰다 (`name_confidence`, 도구 카드에 85% 미만이면 표시). 파이프라인 1회 결과로는 낮은 확신 칸이 맨유 6개·레스터 7개였고, EM 최적 해(`reassign.py`)로 바꾸면 2개·0개로 줄었다. 피드백도 같이 바뀐다 (바꿔 볼 점 맨유 6→4개, 레스터 8→2개). 즉 **피드백은 역할 계산이 안정적이어야 믿을 수 있다.**

## 화면 밖 선수 추정 실험 (`impute_eval.py`)
10명 중 8명 이상 보인 장면에서 화면 폭을 15~25 m로 일부러 좁혀 선수를 가리고, 가려진 선수 위치를 예측했다 (역할은 앞 22.5분으로 학습하고 뒤 22.5분에서 시험, 반대로도 한 번 더).

| 방법 | 맨유 오차 중앙값 | 레스터 |
|---|---|---|
| 보이는 선수들의 중심에 두기 | 15.5 m | 15.3 m |
| **역할 평균 + 팀 이동 (`impute`)** | **8.2~8.5 m** | **8.3 m** |
| 역할 간 상관까지 쓰는 조건부 가우시안 | 8.7 m | 9.4 m (더 나쁨) |
| (참고) 팀 이동을 정답으로 줬을 때 | 5.0 m | 5.1 m |

- 역할 기반 추정이 단순 중심보다 오차가 거의 절반이다. 논문(Off-Screen Imputation)이 보고한 전체 중앙값 9.7~11.6 m와 비슷한 수준이다. 단, 시간 정보 없이 한 장면만 쓴 실험이라 직접 비교는 어렵다.
- 남은 오차의 대부분은 **팀 이동을 일부 선수로 추정하는 데서** 나온다 (8.3 m → 정답 이동이면 5.0 m).

## 아직 안 된 것
- 화면 밖 선수 추정 (자리 평균 + 팀 이동으로 가능, 미구현)
- 공 위치 기반 공격/수비 판단
- 기준값 데이터 학습 (~100경기)

## 참고 논문
- SoccerNet-GSR: https://arxiv.org/abs/2404.11335
- Bialkowski et al. 2014, Large-Scale Analysis of Soccer Matches using Spatiotemporal Tracking Data (ICDM)
- Shaw & Glickman 2019, Dynamic analysis of team strategy in professional football
- Off-Screen Imputation: https://arxiv.org/abs/2607.11548
