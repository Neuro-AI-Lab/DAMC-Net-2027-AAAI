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
