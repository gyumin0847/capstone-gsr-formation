# B팀: 상황별 권장 위치 예측 (v0)

공 위치와 상대 선수 위치(+ 공격/수비)를 보고, 우리 팀 필드 선수 10명이 어디에 서 있어야 하는지 예측합니다.
정답은 IDSSE(분데스리가 7경기) 프로 선수들이 실제로 선 위치입니다. 그래서 '이상적인 위치'가 아니라 '프로 평균 행동'을 배웁니다.

## 파일

| 파일 | 하는 일 | 담당 |
|---|---|---|
| `prepare_idsse.py` | IDSSE → 장면 단위 npz. 2 fps, GK 제외, 우리 팀이 +x로 공격하도록 뒤집기, Bialkowski 포지션 라벨, 볼 소유 → 공격/수비 | 규민 |
| `data.py` | npz 읽기, 경기 단위로 나누기, 오차 계산 | 규민 |
| `baselines.py` | 비교 기준 1 포지션 평균, 2 평균+공 이동(선형), 3 kNN | 규민 |
| `model.py` | 비교 기준 4 MLP, 교수님 구조(DETR 디코더식) 참고 구현, 우리 팀 입력 A/B/C | 성준 (출발점) |
| `run.py` | 전부 돌려서 결과 표(`results.md`)와 미니맵 그림 만들기 | 같이 |
| `make_synthetic.py` | IDSSE 없이 가짜 경기로 전 과정 시험 | — |

## 실행 (deep07)

```bash
cd ~/capstone-gsr-formation && git pull
pip install floodlight scikit-learn          # torch는 서버에 있는 것 사용

# 1) 데이터 만들기: 처음 한 번, 7경기 약 2.4 GB 다운로드 (이미 받은 사람이 있으면 그 폴더 재사용)
python bteam/prepare_idsse.py --out ~/data/bteam

# 2) 실험: GPU는 1장만 (연구실 규칙)
CUDA_VISIBLE_DEVICES=0 python bteam/run.py --data ~/data/bteam --out ~/results/bteam_v0
```

빠르게 확인만 할 때:

```bash
python bteam/make_synthetic.py --out /tmp/bteam_syn                 # 30초
python bteam/run.py --data /tmp/bteam_syn --test SYN03 --val SYN02 --epochs 10
```

## 설계 (B팀 문서 '설계안 v0'과 같음)

- 장면 = 2 fps 한 프레임, 한 팀 시점. 좌표는 m, 경기장 중앙 (0, 0). 모델 안에서는 52.5로 나눕니다.
- 포지션 번호 0~9: 평균 위치를 뒤 라인부터 앞 라인 순서로, 같은 라인 안에서는 왼쪽부터 오른쪽 순서로 매깁니다. 팀이 달라도 0~3번은 대체로 수비 라인입니다.
- 쿼리 초기값 = 그 팀·하프의 포지션 평균 위치입니다. 실사용 때는 사용자가 입력한 포메이션 템플릿으로 바뀝니다.
- 출력 = 평균 위치에서 얼마나 움직일지. 손실은 L1입니다.
- 나누기: 경기 단위로 나눕니다. 기본은 시험 J03WR9, 검증 J03WOY, 학습 나머지 5경기입니다. `--cv`를 주면 7경기를 한 번씩 시험 경기로 씁니다.
- 우리 팀 입력: A 없음 / B 팀 중심 / C 나머지 9명. C는 장면마다 1명을 가리고, 그 1명만 맞힙니다.

## 주의

- `own_mean`(쿼리 초기값, 비교 기준 1)은 그 하프 전체에서 계산한 평균입니다. 시험 경기 정보가 조금 들어가지만, 실사용에서 포메이션을 입력받는 것과 같은 조건이라 그대로 둡니다. 발표 때 이렇게 설명하면 됩니다.
- 퇴장 등으로 상대가 10명보다 적으면 그 선수 토큰은 가립니다. kNN과 MLP에서는 '그 포지션 평균 + 팀 이동량'으로 채웁니다.
- 데이터(IDSSE 원본, npz, 결과)는 레포에 올리지 않습니다 (`.gitignore`).
