# STG-NF on RetailS

A faithful, production-quality reproduction of **STG-NF** — *Normalizing Flows for
Human Pose Anomaly Detection* (Hirschorn & Avidan, ICCV 2023,
[arXiv:2211.10946](https://arxiv.org/abs/2211.10946), official code
[orhir/STG-NF](https://github.com/orhir/STG-NF)) — adapted to the **RetailS**
shoplifting pose dataset (PoseLift protocol).

The model, flow math, ST-GCN coupling, graph construction, windowing and
normalization are ported faithfully from the official implementation. New work is
limited to a RetailS dataset adapter and production infrastructure (YAML config,
automatic GPU selection, mixed precision, checkpoint/resume, early stopping,
TensorBoard, an extended metric suite, and tests). Every engineering decision is
documented under [`docs/`](docs/).

---

## 1. Architecture

STG-NF is a **normalizing flow** (Glow backbone) whose coupling network is a
**spatio-temporal graph convolution (ST-GCN)**. It learns the density of *normal*
skeleton motion; test sequences with low likelihood are flagged as anomalous.
Training uses **normal behaviour only** (RetailS train split), i.e. the
unsupervised setting (`R = 0`).

### Input
A person's pose track is sliced into windows `x ∈ R^{C×T×V}`:
- `V = 18` joints — COCO-17 + a synthesized **neck** (mean of shoulders), reordered
  to the OpenPose layout.
- `T = 24` frames (sliding window, stride 6 for train / 1 for test).
- `C = 2` `(x, y)` by default (`C = 3` with confidence if `model_confidence`).

### Flow
`L` levels × `K` flow steps (default `L=1`, `K=8`). Each **flow step** composes:
1. **ActNorm** — per-channel affine normalization (data-dependent init).
2. **Invertible permutation** — `permute` / `shuffle` / LU-parameterized 1×1 conv.
3. **Affine coupling** — split channels; an **ST-GCN over the skeleton graph**
   predicts `(shift, scale)`, `scale = sigmoid(s + 2) + ε`.

The base density is a diagonal Gaussian. The loss is the **negative
log-likelihood in bits/dim**:

```
nll = -( log p_Z(z) + log|det J| ) / (log 2 · C · T · V)
```

### Score & aggregation
- **Window score** = `−nll` (normality; higher = more normal).
- Assigned to the frame at the window **centre** (`start + T/2`).
- **Frame score** = minimum normality across people (most-anomalous person wins).
- Gaussian temporal smoothing per video; the reported **anomaly** score is
  `−normality` vs. the raw RetailS GT (`1 = abnormal`).

### End-to-end pipeline
```
RetailS pose JSON
      ↓  dataset adapter (person→frame tracks, filename-agnostic)
      ↓  preprocessing (COCO-18, small-gap + missing-joint interpolation)
      ↓  sliding windows (T=24)  → segment normalization
      ↓  skeleton graph (uniform adjacency, max_hop=8)
      ↓  STG-NF normalizing flow
      ↓  negative log-likelihood → anomaly score
      ↓  temporal aggregation (frame centre, min-over-people, smoothing)
      ↓  shoplifting event detection
```

---

## 2. Project structure
```
configs/         retails.yaml — single source of runtime configuration
docs/            repository_audit · paper_specification · retails_dataset_audit · assumptions
scripts/         audit_retails.py · train.py · evaluate.py · infer_clip.py
src/stgnf/
  data/          adapter · preprocessing · augment · dataset (RetailS)
  graph/         skeleton adjacency (COCO-18)
  models/stg_nf/ modules · stgcn · model (faithful flow port)
  training/      trainer (AMP, ckpt/resume, early stopping) · optim
  evaluation/    scoring · metrics (AUC-ROC/PR, EER, F1) · plots · run
  inference/     single-clip scoring + event detection
  utils/         config · device (auto GPU) · seed · logging
tests/           unit + integration + RetailS smoke tests
outputs/         checkpoints · tensorboard · evaluation artifacts
```

---

## 3. Setup

Python 3.10+ and PyTorch 2.x.

```bash
pip install -r requirements.txt   # or: pip install -e .
export PYTHONPATH=src
```

Point `dataset.root` in [`configs/retails.yaml`](configs/retails.yaml) at your
RetailS folder (defaults to the layout `RetailS_train`, `RetailS_test_realworld`,
`RetailS_test_staged`).

Device is auto-selected by **free GPU memory** (the larger GPU is not assumed to be
`cuda:0`); override with `--set device=cuda:1` or `--set device=cpu`.

---

## 4. Commands

**Audit the dataset** (structure, ranges, GT alignment → JSON):
```bash
python scripts/audit_retails.py --config configs/retails.yaml --out outputs/retails_audit.json
```

**Train** (auto GPU, mixed precision, early stopping; auto-evaluates at the end):
```bash
python scripts/train.py --config configs/retails.yaml
```
Handy overrides:
```bash
python scripts/train.py --config configs/retails.yaml --set training.epochs=8
python scripts/train.py --config configs/retails.yaml --resume outputs/checkpoints/last.pt
python scripts/train.py --config configs/retails.yaml --set training.early_stopping.enable=false
```

**Evaluate** a checkpoint (writes metrics.json, ROC/PR curves, per-frame CSV):
```bash
python scripts/evaluate.py --config configs/retails.yaml --checkpoint outputs/checkpoints/best.pt
```

**Infer** on a single clip and detect shoplifting events:
```bash
python scripts/infer_clip.py --checkpoint outputs/checkpoints/best.pt \
    --pose /path/to/clip.json --threshold 0.0 --min-len 3
```

**Tests**:
```bash
pytest -q
```

---

## 5. Latest results

Unsupervised STG-NF (`K=8, L=1, R=0`, `seg_len=24`), trained on the RetailS normal
train split, evaluated frame-level on the two RetailS test splits. Scores are the
threshold-free AUCs; precision/recall/F1 are at the F1-optimal operating point.

| Split            | AUC-ROC | AUC-PR | EER   | best F1 | Frames | Abnormal |
|------------------|:-------:|:------:|:-----:|:-------:|:------:|:--------:|
| `test_realworld` | 0.6856  | 0.4207 | 0.3439| 0.5810  | 4 796  | 1 502 (31.3%) |
| `test_staged`    | 0.8151  | 0.8030 | 0.2421| 0.7733  | 40 360 | 20 156 (49.9%) |

These are consistent with STG-NF's reported regime on shoplifting-style pose
benchmarks (PoseLift), where the benchmark difficulty — not the flow — sets the
ceiling. Full artifacts (ROC/PR curves, per-clip score plots, per-frame CSV) are
under [`outputs/evaluation/`](outputs/evaluation/).

---

## 6. Documentation
- [`docs/repository_audit.md`](docs/repository_audit.md) — what the official code does and what we reused/changed.
- [`docs/paper_specification.md`](docs/paper_specification.md) — the STG-NF architecture as the source of truth.
- [`docs/retails_dataset_audit.md`](docs/retails_dataset_audit.md) — RetailS structure verified against raw data.
- [`docs/assumptions.md`](docs/assumptions.md) — every engineering decision and assumption.

## 7. Notes & limitations
- RetailS test clips are mostly single-person, so multi-person aggregation rarely
  activates on test (it still matters on the crowded train split).
- The `test_realworld` split contains some frames with NaN keypoint values from the
  upstream pose extractor; the training split is clean.
- Absolute AUC is bounded by pose-extraction quality and benchmark difficulty.

## Citation
```bibtex
@InProceedings{Hirschorn_2023_ICCV,
  author    = {Hirschorn, Or and Avidan, Shai},
  title     = {Normalizing Flows for Human Pose Anomaly Detection},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023},
  pages     = {13545-13554}
}
```
