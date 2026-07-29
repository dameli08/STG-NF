# Repository Audit — Official STG-NF (`orhir/STG-NF`)

Source of truth: <https://github.com/orhir/STG-NF> (ICCV 2023, Hirschorn & Avidan,
"Normalizing Flows for Human Pose Anomaly Detection"). A snapshot of `main` was
inspected file-by-file. This document records what the official code does, so we
can reuse the parts that correctly implement the paper and only change what is
necessary to run on RetailS.

## 1. Repository layout

| File | Role |
|------|------|
| `train_eval.py` | Entry point: build datasets, model, `Trainer`, train then score. |
| `args.py` | Argparse config + `init_sub_args` path wiring. |
| `dataset.py` | `PoseSegDataset`, `gen_dataset`, `keypoints17_to_coco18`, conf filtering. |
| `utils/pose_utils.py` | Per-person parsing, sliding windows, continuity checks, UBnormal labels. |
| `utils/data_utils.py` | `normalize_pose`, affine pose augmentations (`trans_list`). |
| `utils/scoring_utils.py` | Frame-level score assembly, temporal smoothing, AUC. |
| `models/STG_NF/model_pose.py` | `STG_NF`, `FlowNet`, `FlowStep` (Glow-style flow). |
| `models/STG_NF/modules_pose.py` | ActNorm, InvertibleConv1x1, Split2d, Squeeze, gaussian utils. |
| `models/STG_NF/stgcn.py` | `st_gcn` spatio-temporal graph conv (coupling network). |
| `models/STG_NF/graph.py` | Skeleton adjacency (`openpose`/COCO18, `ntu-rgb+d`, `alphapose`). |
| `models/training.py` | `Trainer` (train/test loop, checkpoint, LR schedule). |
| `utils/optim_init.py` | Optimizer/scheduler factories. |
| `gen_data.py` | AlphaPose-based JSON generation for custom videos. |

## 2. Architecture as implemented (matches the paper)

STG-NF is a **normalizing flow** (Glow backbone) whose coupling network is an
**ST-GCN**, operating on skeleton windows of shape `(B, C, T, V)`:

- `C` = 2 `(x, y)` by default, or 3 `(x, y, conf)` when `--model_confidence`.
- `T` = `seg_len` (24 for ShanghaiTech, 16 for UBnormal).
- `V` = 18 joints (COCO17 + a synthesized neck).

Flow structure (`FlowNet`): for each of `L` levels, an optional `SqueezeLayer`
(only when the level index `> 1`, so `L=1` performs no squeeze) followed by `K`
`FlowStep`s. Each `FlowStep` is:

1. `ActNorm2d` (data-dependent init on first batch),
2. a permutation (`invconv` = invertible 1×1 conv with LU decomposition, or
   `shuffle`, or fixed `reverse`),
3. an **affine coupling** whose `shift`/`scale` come from an `st_gcn` block over
   the skeleton adjacency `A`; `scale = sigmoid(s + 2) + 1e-6`.

The base density is a diagonal Gaussian. Log-likelihood is accumulated through
the flow and the prior; the loss is the **negative log-likelihood in bits/dim**:
`nll = -objective / (log(2) · C · T · V)`.

Supervision via `R` (`prior_h_normal`/`prior_h_abnormal`): the prior mean is
shifted by `+R` for normal samples (label `1`) and `-R` for abnormal (label
`-1`). ShanghaiTech runs unsupervised (`R=0`, every training window labelled
normal). UBnormal supervised runs use `R=10` with an abnormal-train split.

At test time the model returns the **normality score** `-nll` (higher = more
normal).

## 3. Data pipeline as implemented

1. `gen_dataset` lists `*_tracked_person.json` per clip.
2. `single_pose_dict2np` reads `person_dict[str(person_id)][frame_id]` →
   `{keypoints (17×3), scores}` and stacks frames sorted by frame id. **Outer key
   is the person/track id; inner key is the frame id.**
3. `split_pose_to_segments` slides a `seg_len` window with `seg_stride`, keeping a
   window only if `is_seg_continuous` (≤ `missing_th=2` missing frames). Metadata
   per window: `[scene_id, clip_id, person_id, start_frame]`.
4. `keypoints17_to_coco18` appends a neck (mean of shoulders) and reorders to the
   OpenPose-18 order.
5. `normalize_pose` divides by `[W, H, 1]`, then per segment subtracts the mean of
   `(x, y)` over `(T, V)` and divides by the **y-coordinate std** over `(T, V)`.
6. Optional confidence filtering drops windows whose mean score `< train_seg_conf_th`.

## 4. Scoring / evaluation as implemented

- Each window's normality score is placed at frame `start_frame + seg_len//2` for
  its person.
- Per clip, frames get the **min** normality across people (most-anomalous person).
- Missing frames default to `+inf`, later clipped to the finite max.
- ShanghaiTech ground-truth masks are inverted (`1 - gt`) so `1 = normal`; scores
  are gaussian-smoothed; the metric is **AUC-ROC**.

## 5. Hyperparameters (defaults)

`K=8`, `L=1`, `R=3` (arg default; ShanghaiTech uses `0`, UBnormal `10`),
`hidden_dim=0`, `flow_permutation=permute`, `flow_coupling=affine`,
`LU_decomposed=True`, `actnorm_scale=1.0`, `edge_importance=False`,
`adj_strategy=uniform`, `max_hops=8`, `temporal_kernel=None` → `T//2+1`.
Optimizer **Adamax**, `lr=5e-4`, `weight_decay=5e-5`; scheduler `exp_decay`
(manual `lr·0.99^epoch`); `epochs=8`; `batch_size=256`; grad clip `100`;
`seg_len=24`, `seg_stride=6`; `num_transform=2`.

## 6. Issues / outdated engineering found

| # | Finding | Action in this reproduction |
|---|---------|-----------------------------|
| 1 | `np.int` / `np.float` used (removed in NumPy ≥ 1.24). | Replaced with `int`/`float`. |
| 2 | `plt.style.use('seaborn-ticks')` removed in Matplotlib ≥ 3.6. | Dropped; use default style. |
| 3 | Filename parsing hard-codes ShanghaiTech/UBnormal patterns. | Replaced by a RetailS adapter decoupled from filenames. |
| 4 | `torch.lu`/`lu_unpack` deprecated. | Use `torch.linalg.lu_factor` path. |
| 5 | No AMP, no resume, no structured logging, no config file. | Added AMP, checkpoint/resume, logging, YAML config. |
| 6 | Device hard-coded `cuda:0`. | Added free-memory-based auto-selection + CPU fallback. |
| 7 | Only AUC-ROC reported. | Added AUC-PR, EER, precision/recall/F1, ROC/PR curves. |
| 8 | `KeyboardInterrupt` uses blocking `input()`. | Removed from the training loop. |
| 9 | Continuity gaps are dropped, never interpolated. | Optional small-gap linear interpolation (documented). |

## 7. Reuse decision

The **model** (`FlowStep`, `FlowNet`, `STG_NF`, `st_gcn`, `Graph`, gaussian/flow
modules) and the **core preprocessing math** (COCO18 conversion, segment
normalization, windowing, frame-score assembly, smoothing) correctly implement
the paper and are **ported faithfully** into `src/stgnf/` with only the NumPy/API
modernizations above. New code is limited to: RetailS adapter, config/logging/
device/AMP/checkpointing infrastructure, and the extended evaluation suite.
