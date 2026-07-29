"""Plot helpers (headless / Agg backend)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from stgnf.evaluation.metrics import roc_points, pr_points


def plot_roc(gt, scores, auc: float, out_path: str | Path, title: str = "ROC"):
    fpr, tpr = roc_points(gt, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pr(gt, scores, ap: float, out_path: str | Path, title: str = "Precision-Recall"):
    recall, precision = pr_points(gt, scores)
    baseline = float(np.mean(gt)) if len(gt) else 0.0
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=f"AP = {ap:.4f}")
    ax.axhline(baseline, ls="--", color="grey", linewidth=1, label=f"chance = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_clip_scores(gt: np.ndarray, anomaly: np.ndarray, out_path: str | Path,
                     title: str = "Anomaly score"):
    fig, ax = plt.subplots(figsize=(9, 3))
    x = np.arange(len(anomaly))
    ax.plot(x, anomaly, color="tab:blue", label="anomaly score")
    ax.fill_between(x, anomaly.min(), anomaly.max(), where=gt.astype(bool),
                    color="tab:red", alpha=0.2, label="GT abnormal")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
