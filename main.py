import os

# 사용할 GPU는 실행 환경에서 CUDA_VISIBLE_DEVICES로 덮어쓸 수 있다.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# 다른 GPU 작업과 CPU 코어 경합 방지 (작은 모델이라 CPU-bound → 스레드 제한)
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import csv

import numpy as np
import torch

from train_eval import run_one_subject_one_seed

torch.set_num_threads(8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 데이터셋 위치. 기본값은 이 파일의 상위 폴더이며 DATA_ROOT 환경변수로 바꿀 수 있다.
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.dirname(BASE_DIR))
train_dir = os.path.join(DATA_ROOT, "Training set")
val_dir   = os.path.join(DATA_ROOT, "Validation set")
test_dir  = os.path.join(DATA_ROOT, "Test set")
ROOT_SAVE_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(ROOT_SAVE_DIR, exist_ok=True)

FS = 256
SEEDS = [42, 206, 420, 777, 820, 2023, 2026, 3033, 2034, 3036]
SUBJECT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
N_TRAIN_PER_CLASS = 60
N_VAL_PER_CLASS   = 10
N_TEST_PER_CLASS  = 10
NUM_EPOCHS = 200
BATCH_SIZE = 64
NUM_WORKERS = 0
PIN_MEMORY = True
TOP_K_LIST = (10, 5)


def summarize_top_k(rows, save_dir, top_k):
    """모든 subject에 공통으로 존재하는 seed를 (subject 평균 정확도 ↓, 표준편차 ↑)
    기준으로 정렬해 상위 top_k개를 뽑고, 그 seed들만으로 subject별 평균을 낸 뒤
    전체 subject 평균/표준편차를 반환한다.

    Returns
    -------
    (overall_mean, overall_std, num_subjects) : (float, float, int)
    """
    acc = {}
    for r in rows:
        acc.setdefault(r["subject_id"], {})[r["seed"]] = r["test_acc"]

    subject_ids = sorted(acc)
    if not subject_ids:
        raise ValueError("요약할 결과가 없습니다.")

    common_seeds = set(acc[subject_ids[0]])
    for sid in subject_ids[1:]:
        common_seeds &= set(acc[sid])
    common_seeds = sorted(common_seeds)
    if not common_seeds:
        raise ValueError("모든 subject에 공통으로 존재하는 seed가 없습니다.")

    # seed x subject 정확도 행렬
    matrix = np.array(
        [[acc[sid][seed] for sid in subject_ids] for seed in common_seeds],
        dtype=np.float64,
    )

    seed_mean = matrix.mean(axis=1)
    seed_std = matrix.std(axis=1, ddof=1) if len(subject_ids) > 1 else np.zeros(len(common_seeds))
    ranking = sorted(range(len(common_seeds)), key=lambda i: (-seed_mean[i], seed_std[i]))

    with open(os.path.join(save_dir, "seed_ranking.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["seed", "mean_acc_across_subjects", "std_across_subjects"])
        for i in ranking:
            w.writerow([common_seeds[i], f"{seed_mean[i]:.6f}", f"{seed_std[i]:.6f}"])

    top_rows = ranking[:top_k]
    subject_mean = matrix[top_rows].mean(axis=0)
    overall_mean = float(subject_mean.mean())
    overall_std = float(subject_mean.std(ddof=1)) if len(subject_ids) > 1 else 0.0

    with open(
        os.path.join(save_dir, f"subject_summary_top{top_k}seeds.csv"),
        "w", newline="", encoding="utf-8-sig",
    ) as f:
        w = csv.writer(f)
        w.writerow(["subject", f"mean_acc_top{top_k}seeds"])
        for sid, m in zip(subject_ids, subject_mean):
            w.writerow([sid, f"{m:.6f}"])
        w.writerow([])
        w.writerow(["overall_mean", f"{overall_mean:.6f}"])
        w.writerow(["overall_std", f"{overall_std:.6f}"])

    return overall_mean, overall_std, len(subject_ids)


if __name__ == '__main__':
    print("Using device:", device)
    all_rows = []

    for subject_id in SUBJECT_IDS:
        subj_rows = []

        for seed in SEEDS:
            try:
                print("\n" + "=" * 100)
                print(f"[RUN] subject={subject_id:02d}, seed={seed}")
                print("=" * 100)

                row = run_one_subject_one_seed(
                    subject_id=subject_id,
                    seed=seed,
                    num_epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE,
                    root_save_dir=ROOT_SAVE_DIR,
                    train_dir=train_dir,
                    val_dir=val_dir,
                    test_dir=test_dir,
                    fs=FS,
                    num_workers=NUM_WORKERS,
                    pin_memory=PIN_MEMORY,
                    device=device,
                    n_train_per_class=N_TRAIN_PER_CLASS,
                    n_val_per_class=N_VAL_PER_CLASS,
                    n_test_per_class=N_TEST_PER_CLASS,
                )

                subj_rows.append(row)
                all_rows.append(row)

            except Exception as e:
                print(f"[Fail] subject={subject_id:02d}, seed={seed} -> {repr(e)}")

        if len(subj_rows) > 0:
            accs = np.array([r["test_acc"] for r in subj_rows], dtype=np.float64)
            mean_acc = float(accs.mean())

            subj_dir = os.path.join(ROOT_SAVE_DIR, f"subject_{subject_id:02d}")
            os.makedirs(subj_dir, exist_ok=True)

            with open(os.path.join(subj_dir, "SUBJECT_SUMMARY.txt"), "w", encoding="utf-8") as f:
                f.write(f"Sub {subject_id}:\n")
                f.write(f"Avg acc: {mean_acc:.4f}\n")
                for r in subj_rows:
                    f.write(f"seed {r['seed']}: {r['test_acc']:.4f}\n")

            print(f"[Subject {subject_id:02d}] mean_test_acc={mean_acc:.4f}")

    summary_path = os.path.join(ROOT_SAVE_DIR, "ALL_SUBJECTS_SUMMARY.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for subject_id in SUBJECT_IDS:
            subj_rows = [r for r in all_rows if r["subject_id"] == subject_id]
            if len(subj_rows) == 0:
                continue

            accs = np.array([r["test_acc"] for r in subj_rows], dtype=np.float64)
            f.write(f"Sub {subject_id}:\n")
            f.write(f"Avg acc: {float(accs.mean()):.4f}\n")
            for r in subj_rows:
                f.write(f"seed {r['seed']}: {r['test_acc']:.4f}\n")
            f.write("\n")

    print(f"\n전체 요약 저장 완료: {summary_path}")

    # ── top-k seed 기준 결과 정리: 폴더에서 한눈에 보이도록 점수를 파일명에 표기 ──
    for k in TOP_K_LIST:
        try:
            overall_mean, overall_std, n_sub = summarize_top_k(all_rows, ROOT_SAVE_DIR, top_k=k)
            marker_path = os.path.join(
                ROOT_SAVE_DIR, f"RESULT_top{k}_{overall_mean:.4f}_{overall_std:.4f}.txt"
            )
            with open(marker_path, "w", encoding="utf-8") as f:
                f.write(f"[top{k} seeds 기준] 전체 {n_sub}명\n")
                f.write(f"평균 정확도: {overall_mean:.4f}\n")
                f.write(f"표준편차: {overall_std:.4f}\n")

            print(f"[정리 완료 top{k}] 평균 {overall_mean:.4f} / 표준편차 {overall_std:.4f}")
            print(f"[마커 파일] {marker_path}")
        except Exception as e:
            print(f"[정리 실패 top{k}] {repr(e)}")
