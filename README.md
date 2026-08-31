# [2027 AAAI] DAMC-Net: Dual-Branch Attention-Guided Multi-Scale Convolutional Network for EEG-Based Imagined Speech Decoding

<div align="center">

**Seung Won Kim**<sup></sup>, **Dae Hyeon Kim**<sup></sup>, **Young-Seok Choi**<sup>*</sup>

<sup></sup>Department of Electronics and Communications Engineering, Kwangwoon University, Seoul, South Korea

[![Conference](https://img.shields.io/badge/AAAI-2027-b31b1b.svg)](https://aaai.org/conference/aaai/)
[![Status](https://img.shields.io/badge/status-under%20review-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

## 📢 News
* **[Aug. 2026]** 📄 Our paper **"DAMC-Net: Dual-Branch Attention-Guided Multi-Scale Convolutional Network for EEG-Based Imagined Speech Decoding"** has been **submitted to AAAI 2027**. The code will be officially released upon acceptance.

---

## 📝 Abstract

Imagined speech decoding from electroencephalography (EEG) offers a promising foundation for intuitive, silent speech interfaces. However, the discriminative neural patterns of imagined speech unfold across multiple temporal scales and are distributed over spatially distinct electrodes, making them difficult to capture with conventional single-scale architectures.

We propose **DAMC-Net**, a dual-branch attention-guided multi-scale convolutional network for EEG-based imagined speech decoding. The two branches are designed to be complementary: a **depthwise temporal branch** models channel-independent temporal dynamics after cross-channel mixing, whereas a **cross-channel branch** jointly captures spatial dependency across electrodes. Within each branch, parallel dilated convolutions extract temporal features at multiple resolutions while preserving the original temporal resolution. A **parameter-free attention mechanism** then recalibrates salient activations from local first- and second-order statistics, with the comparison axis matched to each branch, and the recalibrated multi-scale responses are summarized as **log-power representations**. These representations are adaptively integrated through feature-specific scale weights, and the complementary features from the two branches are fused for final classification.

On two public imagined-speech EEG datasets, DAMC-Net attains the highest mean accuracy among the competing baselines, and ablation studies confirm the contribution of each component—multi-scale convolution, parameter-free recalibration, and log-power summarization. These results demonstrate that attention-guided multi-scale representation learning is an effective approach to EEG-based imagined speech decoding.

---

## 📊 Experimental Results

Subject-dependent evaluation on Track 3 of the 2020 International BCI Competition (5 classes, 15 subjects × 10 seeds).
Standard deviation is the **across-subject** sample standard deviation (`ddof=1`) of per-subject seed means.

| Dataset | Params | Accuracy (Mean $\pm$ std) |
| :--- | ---: | :---: |
| **2020 International BCI Competition (Track 3)** | **89,029** | **0.8413 $\pm$ 0.0597** |

### Complexity

| Metric | Value |
| :--- | :---: |
| Parameters | 89,029 |
| MACs | 69.401 M |

> MACs are counted for batch 1 with input `(1, 64, 795)` over `Conv1d` and `Linear` only; one
> multiply-accumulate is one MAC. The parameter-free recalibration, reductions, `log`, `softmax`,
> and element-wise operations are excluded.

### Ablation studies

All configurations share the same data splits and training protocol (15 subjects × 10 seeds × 200 epochs).

#### Component ablation (MS-conv × PF-Attn. × LP-pooling)

| MS-conv | PF-Attn. | LP-pooling | Accuracy (Mean $\pm$ std) |
| :---: | :---: | :---: | :---: |
| $\times$ | $\times$ | $\times$ | 0.4881 $\pm$ 0.0538 |
| $\checkmark$ | $\times$ | $\times$ | 0.5001 $\pm$ 0.0538 |
| $\times$ | $\checkmark$ | $\times$ | 0.5023 $\pm$ 0.0513 |
| $\checkmark$ | $\checkmark$ | $\times$ | 0.5181 $\pm$ 0.0535 |
| $\times$ | $\times$ | $\checkmark$ | 0.7965 $\pm$ 0.0697 |
| $\checkmark$ | $\times$ | $\checkmark$ | <u>0.8326 $\pm$ 0.0673</u> |
| $\times$ | $\checkmark$ | $\checkmark$ | 0.8016 $\pm$ 0.0643 |
| **$\checkmark$** | **$\checkmark$** | **$\checkmark$** | **0.8413 $\pm$ 0.0597** |

> **Key Findings:**
> 1. **The log-power readout carries most of the accuracy.** Its main effect is **+0.316** (0.8180 with LP vs. 0.5022 without, averaged over the other two factors), and every configuration without it collapses to ~0.50 regardless of what else is enabled. Replacing the log-power readout with time-axis mean pooling is the single most damaging change to the model.
> 2. **Multi-scale convolution adds a further 0.036.** With the log-power readout in place, enabling the parallel dilated scales lifts accuracy from 0.7965 to **0.8326** (+0.0361), and to 0.8413 (+0.0397) when the recalibration is also on.
> 3. **The parameter-free recalibration is a small but cost-free gain.** It contributes **+0.0087** on top of MS-conv + LP-pooling (0.8326 → 0.8413) while adding **zero parameters and zero MACs**. Its benefit depends on the multi-scale scales being present: without MS-conv it is worth only +0.0051 (0.7965 → 0.8016), consistent with its role of recalibrating activations before the per-scale readout.
> 4. **Lowest variability at the best accuracy.** The full model also has the smallest across-subject standard deviation of all eight configurations (0.0597), i.e. the components do not trade stability for mean accuracy.

#### Branch contribution and recalibration axis

| Configuration | Params | Accuracy (Mean $\pm$ std) |
| :--- | ---: | :---: |
| *Single branch* | | |
| T-AMC only | 6,277 | 0.6924 $\pm$ 0.0870 |
| S-AMC only | 82,757 | 0.8251 $\pm$ 0.0668 |
| *Both branches; axis in T-AMC / S-AMC* | | |
| Temporal / temporal | 89,029 | 0.8360 $\pm$ 0.0620 |
| Channel / channel | 89,029 | 0.8343 $\pm$ 0.0647 |
| **Temporal / channel (Ours)** | **89,029** | **0.8413 $\pm$ 0.0597** |

> The two branches are complementary: neither alone suffices (0.6924 and 0.8251 vs. 0.8413), and
> adding T-AMC on top of S-AMC costs only 6,272 further parameters. The recalibration axis must be
> matched to each branch — recalibrating both branches along the same axis costs 0.005–0.007 accuracy
> at identical parameter count and MACs, so the gain comes from *where* the recalibration is applied,
> not from added capacity.

---

## 📦 Repository Structure

This repository contains the minimal, self-contained code required to reproduce training and evaluation on Track 3 of the 2020 International BCI Competition.

```
main.py             Full protocol runner (15 subjects x 10 seeds) and result aggregation
train_eval.py       One run (single subject, single seed): preprocessing, training, evaluation
preprocessing.py    .mat loading, signal preprocessing, seed-wise data splitting
model.py            Model definition
```

| File | Role |
|---|---|
| `model.py` | Model definition (`DualBranchSimAMNet`). Contains the parameter-free recalibration module, the multi-scale dilated branch with log-power-domain weighted-sum fusion, and the two branches: **T-AMC** (1x1 cross-channel mixing → depthwise multi-scale convolution, recalibrated along time) and **S-AMC** (full multi-scale convolution over all electrodes, recalibrated across channels). Input `(B, 64, 795)` → class logits `(B, 5)`. Running it standalone prints the output shape and the trainable-parameter count. |
| `preprocessing.py` | Loads the per-subject `.mat` files (`epo_train` / `epo_validation` / `epo_test`), concatenates them, and re-splits them per seed into 60/10/10 trials **per class**. Applies common average reference, a 60 Hz Butterworth low-pass filter (order 4), and channel-wise z-scoring whose statistics are fitted on the training split only. Also provides global seed fixing. |
| `train_eval.py` | One complete run: seeding → preprocessing → training (AdamW, polynomial LR decay, mixed precision, cross-entropy with label smoothing) → best-validation checkpoint restoration → test evaluation, with learning curves, confusion matrix, and classification report written to disk. |
| `main.py` | Runs the full protocol over 15 subjects × 10 seeds, aggregates per-subject and overall accuracy. |

### Data

Place `Data_Sample{1..15}.mat` (containing `epo_train` / `epo_validation` / `epo_test`) in the following directories under `DATA_ROOT`, which defaults to the parent directory of this folder:

```
<DATA_ROOT>/
├── Training set/     Data_Sample1.mat ... Data_Sample15.mat
├── Validation set/   Data_Sample1.mat ... Data_Sample15.mat
└── Test set/         Data_Sample1.mat ... Data_Sample15.mat
```

Each trial uses the full epoch (−0.5 s to 2.6 s at 256 Hz, `T = 795`) without cropping.

### Quick Start

`main.py` is the only entry point.

```bash
python main.py
```

The GPU and the data location are set through environment variables:

```bash
CUDA_VISIBLE_DEVICES=1 DATA_ROOT=/path/to/data python main.py
```

Results are written to `results/subject_XX/seed_YYYY/`, with `SUBJECT_SUMMARY.txt` per subject and
`ALL_SUBJECTS_SUMMARY.txt`, `seed_ranking.csv`, `subject_summary_top{k}seeds.csv`, and
`RESULT_top{k}_{mean}_{std}.txt` at the top level.

### Key Hyperparameters

Run settings are defined as constants at the top of `main.py`.

| Argument | Value | Description |
|---|---|---|
| `NUM_EPOCHS` | 200 | Training epochs |
| `BATCH_SIZE` | 64 | Mini-batch size |
| `FS` | 256 | Sampling rate (Hz) |
| `SUBJECT_IDS` | 1–15 | Subject-dependent evaluation |
| `SEEDS` | 10 seeds | Each seed defines a different 60/10/10 per-class split |
| `N_TRAIN/VAL/TEST_PER_CLASS` | 60 / 10 / 10 | Stratified per-class split sizes |
| `kernel_size`, `dilations` | 5, (1, 2, 3, 4) | Multi-scale dilated convolutions in both branches |
| recalibration axis | time (T-AMC) / channel (S-AMC) | Axis of the parameter-free recalibration in each branch |
| `simam_lambda` | 1e-3 | Stabilizing term of the parameter-free recalibration |

Optimizer: AdamW, learning rate 1e-3 → 1e-6 (polynomial decay, power 2), weight decay 1e-2,
cross-entropy with label smoothing 0.01, mixed precision on CUDA. The model with the best
validation accuracy is restored before test evaluation.

> **Note on the released code.** The `model.py` in this folder is an earlier configuration
> (`kernel_size=3`, both branches recalibrated along the time axis, 55,749 parameters).
> Reproducing the reported 0.8413 requires `kernel_size=5` and channel-axis recalibration in the
> S-AMC branch; the final version will be released with the paper.

---

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

The paper is currently under review at AAAI 2027. A citation entry will be added upon acceptance.
