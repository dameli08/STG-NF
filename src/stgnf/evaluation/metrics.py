"""Anomaly-detection metrics: AUC-ROC, AUC-PR, EER, precision/recall/F1."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
)


@dataclass
class Metrics:
    auc_roc: float
    auc_pr: float
    eer: float
    eer_threshold: float
    best_f1: float
    best_f1_threshold: float
    precision_at_best_f1: float
    recall_at_best_f1: float
    num_frames: int
    num_abnormal: int
    # Precision-oriented operating points (added for the RetailS precision focus).
    precision_at_eer: float = float("nan")
    recall_at_eer: float = float("nan")
    target_recall: float = float("nan")
    precision_at_target_recall: float = float("nan")
    threshold_at_target_recall: float = float("nan")

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _clean(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).copy()
    if np.isinf(scores).any():
        finite = scores[np.isfinite(scores)]
        hi = finite.max() if finite.size else 0.0
        lo = finite.min() if finite.size else 0.0
        scores[scores == np.inf] = hi
        scores[scores == -np.inf] = lo
    return scores


def precision_recall_at_threshold(gt: np.ndarray, scores: np.ndarray, thr: float):
    """Precision/recall for the rule ``score >= thr`` => abnormal."""
    pred = scores >= thr
    tp = int(np.sum(pred & (gt == 1)))
    fp = int(np.sum(pred & (gt == 0)))
    fn = int(np.sum((~pred) & (gt == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def precision_at_target_recall(gt: np.ndarray, scores: np.ndarray, target_recall: float):
    """Highest precision achievable while keeping recall >= ``target_recall``.

    Returns ``(precision, recall, threshold)``. Useful when the evaluation cares
    about precision at an acceptable recall level.
    """
    precision, recall, thr = precision_recall_curve(gt, scores)
    # precision/recall have len == len(thr)+1; align by dropping the final point.
    p, r = precision[:-1], recall[:-1]
    feasible = r >= target_recall
    if not feasible.any():
        return float("nan"), float("nan"), float("nan")
    idx_candidates = np.where(feasible)[0]
    best = idx_candidates[np.argmax(p[idx_candidates])]
    return float(p[best]), float(r[best]), float(thr[best])


def compute_eer(gt: np.ndarray, scores: np.ndarray):
    """Equal Error Rate and its threshold (higher score => abnormal)."""
    fpr, tpr, thr = roc_curve(gt, scores)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thr[idx])


def compute_metrics(gt: np.ndarray, anomaly_scores: np.ndarray,
                    target_recall: float = 0.5) -> Metrics:
    """Compute the full metric suite. ``gt``: 1 = abnormal; higher score = abnormal."""
    gt = np.asarray(gt, dtype=np.int64)
    scores = _clean(anomaly_scores)
    if gt.size == 0 or len(np.unique(gt)) < 2:
        return Metrics(float("nan"), float("nan"), float("nan"), float("nan"),
                       float("nan"), float("nan"), float("nan"), float("nan"),
                       int(gt.size), int(gt.sum()))

    auc_roc = float(roc_auc_score(gt, scores))
    auc_pr = float(average_precision_score(gt, scores))
    eer, eer_thr = compute_eer(gt, scores)

    precision, recall, thr = precision_recall_curve(gt, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1[:-1])) if f1.size > 1 else 0
    best_f1 = float(f1[best_idx])
    best_thr = float(thr[best_idx]) if thr.size else float("nan")

    p_eer, r_eer = precision_recall_at_threshold(gt, scores, eer_thr)
    p_tr, r_tr, thr_tr = precision_at_target_recall(gt, scores, target_recall)

    return Metrics(
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        eer=eer,
        eer_threshold=eer_thr,
        best_f1=best_f1,
        best_f1_threshold=best_thr,
        precision_at_best_f1=float(precision[best_idx]),
        recall_at_best_f1=float(recall[best_idx]),
        num_frames=int(gt.size),
        num_abnormal=int(gt.sum()),
        precision_at_eer=float(p_eer),
        recall_at_eer=float(r_eer),
        target_recall=float(target_recall),
        precision_at_target_recall=float(p_tr),
        threshold_at_target_recall=float(thr_tr),
    )


def roc_points(gt: np.ndarray, scores: np.ndarray):
    fpr, tpr, _ = roc_curve(gt, _clean(scores))
    return fpr, tpr


def pr_points(gt: np.ndarray, scores: np.ndarray):
    precision, recall, _ = precision_recall_curve(gt, _clean(scores))
    return recall, precision
