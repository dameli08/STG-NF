# Assumptions & Engineering Decisions

Every decision below is made to keep the reproduction **faithful to the STG-NF
paper** while running correctly and productively on RetailS. Items marked
*(paper)* follow the official code; *(engineering)* are choices we made where the
paper/repo are silent, with the rationale given.

## Model / methodology (unchanged from the paper)

1. **Flow architecture** *(paper)* — Glow backbone with ST-GCN affine coupling,
   ActNorm, and permutation layers, ported verbatim (only NumPy/Torch API
   modernizations). `K=8`, `L=1`, `hidden=0`, uniform adjacency, `max_hop=8`.
2. **COCO-17 → COCO-18** *(paper)* — neck = mean of shoulders, reordered to the
   OpenPose layout, so the RetailS skeleton matches the model's graph exactly.
3. **Segment normalization** *(paper)* — divide by `[W, H, 1]`, then per window
   subtract the `(x,y)` mean over `(T,V)` and divide by the y-coordinate std over
   `(T,V)`. This preserves motion/trajectory (unlike per-frame centering).
4. **NLL in bits/dim** *(paper)* — objective and score sign kept identical;
   normality score `= −nll`.
5. **Unsupervised setting for RetailS** *(paper-consistent)* — RetailS train is
   normal-only, so `R = 0` and every training window is labelled normal, exactly
   like STG-NF's ShanghaiTech recipe. No shoplifting-specific model changes.

## Dataset adaptation *(engineering, no methodology change)*

6. **Filename-agnostic adapter** — RetailS filenames differ from ShanghaiTech /
   UBnormal. The adapter keys clips by **file stem** and pairs pose↔GT by stem,
   instead of regex-parsing `scene_clip`. Metadata carries `(clip_stem,
   person_id, start_frame)`; scene/clip integers are not required.
7. **Missing per-frame `scores`** — RetailS `scores` is `null`; we derive a
   per-frame score as the **mean of the 17 joint confidences**. Used only for the
   optional confidence threshold and confidence-weighted NLL, matching how the
   paper consumes `scores`.
8. **Resolution** — `vid_res = [1080, 1920]` (portrait), configurable.
9. **Small-gap interpolation** — where a track has ≤ `max_missing` (default 2)
   consecutive missing frames inside a candidate window, coordinates are **linearly
   interpolated** and confidence set to the min of the neighbours. This is a
   superset of the paper's "tolerate ≤ 2 missing" rule (the paper keeps such
   windows but leaves the gap unfilled); it avoids feeding the flow a jump
   discontinuity. Windows with larger gaps are still dropped. Toggle:
   `preprocessing.interpolate_gaps`.

## Missing joints *(engineering)*

10. Joints with confidence `≤ 0` or non-finite coordinates are treated as missing
    and filled by linear temporal interpolation within the window (nearest-value
    hold at the window edges). If an entire joint is missing across the window it
    is set to the window's per-joint mean position. This keeps the graph
    fully populated, which the flow's fixed adjacency requires.

## Optimization / infrastructure *(engineering)*

11. **Optimizer/scheduler** — Adamax `lr=5e-4`, `wd=5e-5`, exponential decay
    `0.99^epoch`, grad-clip `100` — identical to the paper defaults.
12. **Mixed precision** — autocast wraps the forward/backward. The flow's
    log-determinant and Gaussian likelihood are computed in fp32 (via autocast's
    numerically-sensitive op handling and an fp32 loss cast) to avoid NLL
    instability; ActNorm data-dependent init runs in fp32 on the first batch.
13. **Device selection** — query both GPUs' **free memory** (torch, with an
    `nvidia-smi` cross-check when available) and pick the largest; never assume the
    big GPU is `cuda:0`. Supports manual override (`device: cuda:1`) and CPU
    fallback. On this workstation `cuda:0` = RTX 2080 Ti (11 GB) is auto-selected.
14. **Checkpointing / resume** — full state (model, optimizer, epoch, config,
    ActNorm-inited flag) is saved every epoch and as `best`; training can resume.
15. **Seeding** — a fixed seed (default 42) seeds Python/NumPy/Torch;
    `seed: 999` reproduces the paper's "record a random seed" behaviour.
16. **Early stopping** *(engineering; off by default in the paper)* — a fraction
    (`training.early_stopping.val_fraction`, default 0.1) of the *training clips*
    is held out — **split by clip stem** to avoid person/window leakage — as an
    unsupervised validation set. After each epoch the mean validation NLL of this
    normal-only set is measured; the best epoch is saved to `best.pt`, and training
    stops after `patience` epochs without at least `min_delta` improvement. The
    best checkpoint is restored before the final test evaluation. Disable via
    `training.early_stopping.enable=false` to reproduce the paper's fixed-epoch run.

## Evaluation *(engineering superset of the paper)*

17. **Metrics** — AUC-ROC (paper) **plus** AUC-PR, EER, and precision/recall/F1 at
    the EER-optimal and F1-optimal thresholds; ROC and PR curves and per-video
    score plots are exported. All computed on the **anomaly** score vs. the raw
    `1 = abnormal` GT (polarity-unambiguous).
18. **Frame aggregation** *(paper)* — window score → frame `start + seg_len//2`;
    per-frame value = min normality across people; gaussian temporal smoothing
    (`sigma=7`).
19. **Threshold selection** — no single operating point is required for AUC, but
    for the reported precision/recall/F1 we select the threshold maximizing F1 on
    the evaluated split and separately report the EER point. These are diagnostic;
    the headline metric remains threshold-free AUC.

## Update — correctness fixes & precision focus (2026-07-30)

The project is now judged on **precision on the real-world test split**. The
following changes were made; none alter the STG-NF model or its methodology.

20. **NaN/inf keypoint sanitization** *(correctness)* — the real-world split
    contains frames with non-finite keypoints from the pose extractor. Non-finite
    joints are marked missing (confidence 0) at parse time and then filled by the
    existing missing-joint interpolation, instead of poisoning windows with NaN.
21. **GPU selection bug fix** *(correctness)* — free memory is now read from
    `torch.cuda.mem_get_info(i)` (authoritative per torch index). The previous
    `nvidia-smi`-by-index cross-check was wrong because torch orders GPUs
    FASTEST_FIRST while `nvidia-smi` uses PCI-bus order; it could select the
    smaller GPU. `nvidia-smi` is now only a UUID-matched fallback.
22. **In-epoch checkpointing** *(robustness)* — in addition to the end-of-epoch
    save, a checkpoint is written every `training.ckpt_interval` steps (default
    500) so a crash/interrupt mid-epoch loses at most a few hundred steps. `best.pt`
    is still selected by lowest validation NLL (early stopping); mid-epoch saves go
    to `last.pt` and record the *current* epoch so a resume restarts it cleanly.
23. **Robust per-frame aggregation** *(precision)* — `evaluation.aggregation=mean`
    assigns each window's anomaly to **every frame it covers** and averages, a
    lower-variance estimator than the paper's centre-frame assignment; it usually
    improves AUC-PR and precision. `center` (paper) remains the default in the base
    config; `mean` is enabled in `configs/retails_precision.yaml`.
24. **Confidence-gated scoring** *(precision)* — `evaluation.min_confidence` drops
    test windows whose mean keypoint confidence is below the threshold, so noisy
    pose detections do not raise false anomalies. This trades a little recall for
    precision and should be **swept per split** after full training (start ~0.2).
25. **Precision-oriented metrics** *(reporting)* — in addition to AUC/EER/F1 we now
    report precision & recall at the EER threshold and the **best precision at a
    configurable minimum recall** (`evaluation.target_recall`, default 0.5), so the
    precision operating point is explicit.

### Optional levers documented but left off by default
- **Confidence-weighted flow** *(paper knob)* — `model.model_confidence=true` with
  `model.in_channels=3` feeds `(x,y,conf)` and weights the NLL by pose confidence;
  worth trying for the noisy real-world split (requires retraining).
- We deliberately did **not** add shoplifting-specific heuristics (e.g. hand-region
  or shelf-proximity features) to the model: they would break the general
  skeleton-anomaly methodology the paper defines. Precision is improved through
  cleaner data, better aggregation and confidence gating instead.

## Known limitations

- Test clips are mostly single-person, so the multi-person "min across people"
  aggregation rarely activates on RetailS test (it still matters on the crowded
  train split and generally).
- RetailS provides no camera/session metadata beyond filenames, so cross-camera
  generalization cannot be measured; official file-level splits are used as-is.
- Absolute AUC is bounded by pose-extraction quality and the benchmark difficulty,
  not by the flow; see `paper_specification.md` §6.
