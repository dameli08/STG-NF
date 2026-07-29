"""End-to-end evaluation of a prediction function over the configured test splits."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict

import numpy as np

from stgnf.data.adapter import resolve_split_dir
from stgnf.evaluation.metrics import compute_metrics
from stgnf.evaluation.scoring import assemble_frame_scores
from stgnf.evaluation.plots import plot_roc, plot_pr, plot_clip_scores
from stgnf.utils.logging import get_logger

log = get_logger("evaluation")


def evaluate_splits(
    predict_fn: Callable,
    datasets: Dict,
    cfg,
    out_dir: str | Path,
    make_plots: bool = True,
    max_clip_plots: int = 8,
) -> Dict[str, dict]:
    """Predict, assemble frame scores, compute metrics and export artifacts.

    Args:
        predict_fn: maps a DataLoader/dataset to per-window normality scores.
        datasets: mapping split_name -> RetailSPoseDataset (evaluation datasets).
    Returns:
        ``{split_name: metrics_dict}``.
    """
    out_dir = Path(out_dir)
    ds_cfg = cfg.dataset
    ev = cfg.evaluation
    results: Dict[str, dict] = {}

    for split_name, spec in ds_cfg.test_splits.items():
        if split_name not in datasets:
            continue
        dataset = datasets[split_name]
        gt_dir = resolve_split_dir(ds_cfg.root, spec["gt"])
        split_out = out_dir / split_name
        split_out.mkdir(parents=True, exist_ok=True)

        window_norm = predict_fn(dataset)
        window_norm = np.atleast_1d(np.asarray(window_norm, dtype=np.float64))
        if window_norm.shape[0] != len(dataset.metadata):
            log.warning("[%s] score/metadata length mismatch: %d vs %d",
                        split_name, window_norm.shape[0], len(dataset.metadata))

        scores = assemble_frame_scores(
            window_norm, dataset.metadata, gt_dir, seg_len=cfg.preprocessing.seg_len,
            smoothing_sigma=ev.smoothing_sigma, frame_offset=ev.frame_offset)
        metrics = compute_metrics(scores.gt, scores.anomaly)
        results[split_name] = metrics.to_dict()

        log.info("[%s] AUC-ROC %.4f | AUC-PR %.4f | EER %.4f | F1 %.4f (%d frames, %d abnormal)",
                 split_name, metrics.auc_roc, metrics.auc_pr, metrics.eer, metrics.best_f1,
                 metrics.num_frames, metrics.num_abnormal)

        with open(split_out / "metrics.json", "w") as fh:
            json.dump(metrics.to_dict(), fh, indent=2)
        np.savez_compressed(split_out / "frame_scores.npz",
                            gt=scores.gt, anomaly=scores.anomaly)
        _dump_predictions_csv(split_out / "frame_scores.csv", scores)

        if make_plots and metrics.num_frames > 0 and not np.isnan(metrics.auc_roc):
            plot_roc(scores.gt, scores.anomaly, metrics.auc_roc,
                     split_out / "roc.png", title=f"ROC — {split_name}")
            plot_pr(scores.gt, scores.anomaly, metrics.auc_pr,
                    split_out / "pr.png", title=f"PR — {split_name}")
            clip_dir = split_out / "clip_plots"
            clip_dir.mkdir(exist_ok=True)
            for stem in list(scores.per_clip.keys())[:max_clip_plots]:
                c = scores.per_clip[stem]
                plot_clip_scores(c["gt"], c["anomaly"], clip_dir / f"{stem}.png",
                                 title=f"{split_name} / {stem}")

    with open(out_dir / "summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
    return results


def _dump_predictions_csv(path, scores) -> None:
    import csv
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["clip", "frame", "gt", "anomaly"])
        for stem, c in scores.per_clip.items():
            for f, (g, a) in enumerate(zip(c["gt"], c["anomaly"])):
                writer.writerow([stem, f, int(g), float(a)])
