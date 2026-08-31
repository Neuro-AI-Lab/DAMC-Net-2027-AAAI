# [2027 AAAI]DAMC-Net: Dual-Branch Attention-Guided Multi-Scale Convolutional Network for EEG-Based Imagined Speech Decoding

15명 피험자의 상상 발화(imagined speech) EEG로부터 5개 단어(`Hello`, `Helpme`, `Stop`, `Thankyou`, `Yes`)를 분류하는 subject-dependent 모델입니다.

데이터셋은 OSF BCI Competition track 3 사용

## 실행 방법

`main.py` 만 실행하면 됩니다.

```bash
python main.py
```

GPU와 데이터 위치는 환경변수로 지정합니다. `DATA_ROOT`의 기본값은 이 폴더의
상위 폴더입니다.

```bash
CUDA_VISIBLE_DEVICES=1 DATA_ROOT=/path/to/data python main.py
```

데이터는 아래 구조로 있어야 합니다 (`subject_id`는 1–15).

```
<DATA_ROOT>/
├── Training set/     Data_Sample1.mat ... Data_Sample15.mat
├── Validation set/   Data_Sample1.mat ... Data_Sample15.mat
└── Test set/         Data_Sample1.mat ... Data_Sample15.mat
```

실행 설정은 `main.py` 상단 상수에서 조정합니다 (`SUBJECT_IDS`, `SEEDS`,
`NUM_EPOCHS`, `BATCH_SIZE` 등). 결과는 `results/` 아래에 저장됩니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `main.py` | 실행 엔트리포인트. 설정 상수를 정의하고 subject × seed 전체를 순회하며 학습을 돌린 뒤, 결과를 집계해 요약 파일을 저장합니다. |
| `train_eval.py` | 한 번의 subject × seed 실행 전체를 담당합니다. 데이터 로딩과 전처리 호출, DataLoader 구성, 학습 루프, validation 기준 best 모델 선택, test 평가, learning curve와 혼동행렬 저장. |
| `model.py` | DAMC-Net 모델 정의 (`DualBranchSimAMNet`). 단독 실행하면 파라미터 수를 출력합니다. |
| `preprocessing.py` | `.mat` 파일 로딩, CAR, 저역통과 필터, z-score 정규화, 클래스별 층화 분할, seed 고정 함수. |

## 모델 구조

`model.py`의 `DualBranchSimAMNet`. 입력 `(B, 64, T)` → 로짓 `(B, 5)`,
학습 파라미터 55,749개 (`python model.py`로 확인 가능).

```
input (B, 64, T)
   │
   ├─ temporal branch
   │     1x1 conv (전극 혼합)
   │     → depthwise multi-scale conv, dilation 1/2/3/4
   │     → scale별 [SimAM → log-power] → 학습 가중합
   │     → (B, 64)
   │
   └─ spatial branch
         full multi-scale conv, dilation 1/2/3/4
         → scale별 [SimAM → log-power] → 학습 가중합
         → (B, 64)

concat (B, 128) → Linear → logits (B, 5)
```

### 두 개의 branch

두 branch 모두 kernel size 3, dilation 1/2/3/4의 conv 4개로 서로 다른 시간
스케일을 병렬로 훑지만, 전극을 다루는 방식이 다릅니다.

- **temporal branch**: 먼저 1x1 conv로 전극을 섞은 뒤, depthwise conv
  (`groups=64`)로 각 채널을 독립적으로 시간축 필터링합니다. 전극 혼합과 시간
  필터링이 분리된 구조입니다.
- **spatial branch**: 원본 입력에 full conv (`groups=1`)를 적용해 64개 전극을
  모두 섞으면서 동시에 시간축을 필터링합니다.

각 branch가 `(B, 64)` 요약을 내고, 둘을 concat한 `(B, 128)`을 단일 Linear로
분류합니다.

### SimAM

파라미터가 없는 attention 모듈(Yang et al., ICML 2021)의 시간축 변형입니다.
채널별로 시간축 평균·분산을 구해 각 시점의 중요도를 닫힌 형태로 계산하고
sigmoid 게이팅을 적용합니다. 학습 파라미터가 0개라 모델 크기를 늘리지 않습니다.

### Log-power 도메인 가중합

SimAM을 통과한 각 scale을 시간축 log-power `log(mean(x²))`로 요약해 `(B, 64)`를
얻고, 네 scale을 concat하지 않고 가중합으로 합칩니다. 가중치는
(scale, channel)별 파라미터에 scale 축 softmax를 적용해 얻습니다 (초기값 0 →
균등 평균에서 학습을 시작).
