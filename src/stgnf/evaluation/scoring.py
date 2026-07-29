"""Frame-level score assembly from per-window normality scores.

Follows the STG-NF protocol: each window's normality score (``-nll``) is placed
at the frame ``start + seg_len//2`` for its person; per frame the value is the
**minimum normality across people** (most anomalous person); scores are gaussian
smoothed per video. The returned *anomaly* score is ``-normality`` so it aligns
with the raw RetailS ground truth where ``1 = abnormal`` (polarity-unambiguous).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.ndimage import gaussian_filter1d

from stgnf.data.adapter import load_gt_mask


@dataclass
class SplitScores:
    gt: np.ndarray                              # (sum F,) 1 = abnormal
    anomaly: np.ndarray                         # (sum F,) higher = more anomalous
    per_clip: Dict[str, dict] = field(default_factory=dict)  # stem -> {gt, anomaly}


def assemble_frame_scores(
    window_norm: np.ndarray,
    metadata: List[list],
    gt_dir: str | Path,
    seg_len: int,
    smoothing_sigma: float = 7.0,
    frame_offset: str = "half",
) -> SplitScores:
    """Aggregate window scores to per-frame anomaly scores for one split."""
    offset = seg_len // 2 if frame_offset == "half" else 0
    meta_stem = np.array([m[0] for m in metadata])
    meta_pid = np.array([m[1] for m in metadata], dtype=np.int64)
    meta_start = np.array([m[2] for m in metadata], dtype=np.int64)

    per_clip: Dict[str, dict] = {}
    stems = sorted(set(meta_stem.tolist()))
    for stem in stems:
        gt = load_gt_mask(gt_dir, stem)                 # (F,) 1 = abnormal
        F = gt.shape[0]
        sel = np.where(meta_stem == stem)[0]
        pids = np.unique(meta_pid[sel])
        person_norm = {int(p): np.full(F, np.inf, dtype=np.float64) for p in pids}
        for i in sel:
            frame = int(meta_start[i]) + offset
            if 0 <= frame < F:
                person_norm[int(meta_pid[i])][frame] = float(window_norm[i])
        clip_norm = np.amin(np.stack(list(person_norm.values()), axis=0), axis=0)  # (F,)
        per_clip[stem] = {"gt": gt, "norm": clip_norm}

    # Replace +/- inf (frames with no window) by the finite extremes.
    all_norm = np.concatenate([c["norm"] for c in per_clip.values()])
    finite = all_norm[np.isfinite(all_norm)]
    hi = finite.max() if finite.size else 0.0
    lo = finite.min() if finite.size else 0.0
    for stem in per_clip:
        norm = per_clip[stem]["norm"].copy()
        norm[norm == np.inf] = hi
        norm[norm == -np.inf] = lo
        if smoothing_sigma and smoothing_sigma > 0:
            norm = gaussian_filter1d(norm, sigma=smoothing_sigma)
        anomaly = -norm
        per_clip[stem]["anomaly"] = anomaly.astype(np.float64)
        per_clip[stem].pop("norm", None)

    gt_all = np.concatenate([per_clip[s]["gt"] for s in stems]) if stems else np.empty(0)
    anomaly_all = np.concatenate([per_clip[s]["anomaly"] for s in stems]) if stems else np.empty(0)
    return SplitScores(gt=gt_all.astype(np.int64), anomaly=anomaly_all, per_clip=per_clip)
