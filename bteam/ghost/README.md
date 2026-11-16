# Ghost defence (PFF World Cup 2022)

Le, Carr, Yue, Lucey, "Data-Driven Ghosting using Deep Imitation Learning" (SSAC 2017) 방식을 따라
수비 10명의 고스트(프로라면 있었을 위치)를 예측하는 코드입니다. **코드만 있고 데이터·결과·모델·영상은 없습니다.**

## 데이터

- PFF FC 2022 World Cup dataset (재배포 금지라 레포에 넣지 않음)
- 경로는 환경변수 `PFF_RAW`로 지정 (기본값 `/lab/users/gyumin/data/pff_raw`), 또는 각 스크립트의 `--raw`
- 결과는 `outputs/`에 저장되고 git에는 올라가지 않음

## 실행

필요 패키지: `numpy`, `torch`, `scipy`, `matplotlib`, `imageio-ffmpeg`

```bash
# 64경기: 경기 단위로 학습 58 / 검증 3 / 시험 3, 1단계 50바퀴 + 2단계 20바퀴 + 시험 (GPU 1장 약 2.5시간)
python run_multi.py --games all --epochs 10

# 고스트 영상 (CPU)
python make_video.py --model outputs/wc_all64/model_phase2.pt
```

## 파일

| 파일 | 하는 일 |
|---|---|
| `run_multi.py` | 전체 실행: 경기 나누기 → 장면 자르기 → 1단계 → 2단계 → 시험 |
| `run_phase2.py` | 2단계(10명 동시 예측)만 따로 실행 |
| `run.py` | 1경기로 빠르게 돌려 보는 버전 |
| `src/load.py` | PFF 트래킹을 10 fps 배열로 읽기 |
| `src/scenes.py` | 수비 장면 자르기 (상대가 공을 잡은 순간 ~ 되찾을 때까지) |
| `src/roles.py` | 평균 포메이션 10자리, 헝가리안 1:1 배정 |
| `src/features.py` | 0.1초마다 모델 입력 만들기 (내 위치·속도, 동료, 상대, 공 등) |
| `src/model.py` | 자리별 LSTM 10개, 이어서 예측 `rollout`, 화면에 보인 선수만 채점 `masked_loss` |
| `src/train.py` | 5초 조각 자르기 `windows`, 256개씩 학습, 바퀴마다 검증, 최고 모델 저장 |
| `src/evaluate.py` | 시험 채점 |
| `make_video.py` | 실제 수비 vs 고스트 영상 |
| `check_*.py`, `compare_rules.py` | 결과 점검 (논문 방식 비교, 모델 비교, 장면 규칙 비교 등) |

## 학습 요약

- 수비 장면을 5.1초 조각으로 1초씩 밀며 자름 (5초보다 짧은 장면은 통째로 1조각)
- 256조각씩 묶어 평균 오차로 한 번 고치기, Adam lr 1e-3
- 1단계: 정답 위치 주는 간격 1·5·10·20·50스텝, 간격마다 10바퀴 (동료는 진짜 위치)
- 2단계: 1단계 최고 모델에서 이어서 간격 5·10·20·50스텝, 간격마다 5바퀴, lr 5e-4 (10명 동시, 동료도 예측값)
- 화면에 보인(VISIBLE) 순간만 정답으로 채점, 화면 밖 추정 위치는 채점 제외
- 바퀴마다 검증 3경기에서 처음 위치만 주고 5초 혼자 그린 오차로 최고 모델 선택

## 결과 (시험 3경기, 장면 623개)

장면 첫 위치만 주고 끝까지 10명 모두 생성했을 때 실제 수비수와 평균 거리 **4.07 m** (논문 약 4 m).

Data: PFF FC 2022 World Cup dataset.
