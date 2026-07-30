import numpy as np

from stgnf.evaluation.metrics import (
    compute_metrics, compute_eer, precision_at_target_recall,
    precision_recall_at_threshold,
)
from stgnf.evaluation.scoring import assemble_frame_scores


def test_metrics_perfect_separation():
    gt = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.15, 0.9, 0.95, 0.8])  # higher = abnormal
    m = compute_metrics(gt, scores)
    assert m.auc_roc == 1.0
    assert m.auc_pr == 1.0
    assert m.eer == 0.0


def test_metrics_handle_single_class():
    gt = np.zeros(10, dtype=int)
    scores = np.random.rand(10)
    m = compute_metrics(gt, scores)
    assert np.isnan(m.auc_roc)


def test_eer_between_zero_and_one():
    gt = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0.2, 0.8, 0.4, 0.6, 0.3, 0.7])
    eer, thr = compute_eer(gt, scores)
    assert 0.0 <= eer <= 1.0


def test_assemble_frame_scores(tmp_path):
    # Build a tiny GT + metadata and confirm frame assignment & alignment.
    gt = np.zeros(60, dtype=np.uint8)
    gt[30:45] = 1
    np.save(tmp_path / "01_0000.npy", gt)
    seg_len = 24
    # windows for person 1 at strides of 1, normality high (normal) everywhere
    metadata = [["01_0000", 1, s] for s in range(0, 60 - seg_len + 1)]
    window_norm = np.ones(len(metadata))
    res = assemble_frame_scores(window_norm, metadata, tmp_path, seg_len=seg_len,
                                smoothing_sigma=0)
    assert res.gt.shape[0] == 60
    assert res.anomaly.shape[0] == 60
    assert set(np.unique(res.gt)).issubset({0, 1})


def test_mean_aggregation_covers_more_frames(tmp_path):
    gt = np.zeros(60, dtype=np.uint8)
    gt[30:45] = 1
    np.save(tmp_path / "01_0000.npy", gt)
    seg_len = 24
    metadata = [["01_0000", 1, s] for s in range(0, 60 - seg_len + 1)]
    # Make abnormal-region windows more anomalous (lower normality).
    window_norm = np.ones(len(metadata))
    for i, m in enumerate(metadata):
        if 20 <= m[2] <= 30:
            window_norm[i] = -5.0
    center = assemble_frame_scores(window_norm, metadata, tmp_path, seg_len=seg_len,
                                   smoothing_sigma=0, aggregation="center")
    mean = assemble_frame_scores(window_norm, metadata, tmp_path, seg_len=seg_len,
                                 smoothing_sigma=0, aggregation="mean")
    # Both detect the abnormal region; mean spreads the signal across covered frames.
    assert mean.anomaly[30:45].mean() > mean.anomaly[:20].mean()
    assert center.gt.shape == mean.gt.shape


def test_min_confidence_filters_windows(tmp_path):
    gt = np.zeros(60, dtype=np.uint8)
    gt[30:45] = 1
    np.save(tmp_path / "01_0000.npy", gt)
    seg_len = 24
    metadata = [["01_0000", 1, s] for s in range(0, 60 - seg_len + 1)]
    window_norm = np.full(len(metadata), -5.0)      # everything looks anomalous
    conf = np.full(len(metadata), 0.1)              # ...but all low confidence
    res = assemble_frame_scores(window_norm, metadata, tmp_path, seg_len=seg_len,
                                smoothing_sigma=0, window_conf=conf, min_confidence=0.5)
    # All windows dropped -> no covered frames -> anomaly falls back to the min.
    assert np.isfinite(res.anomaly).all()


def test_precision_at_target_recall_and_threshold():
    gt = np.array([0, 0, 1, 1, 1, 0])
    scores = np.array([0.1, 0.2, 0.9, 0.6, 0.55, 0.5])
    p, r, thr = precision_at_target_recall(gt, scores, target_recall=0.6)
    assert 0.0 <= p <= 1.0
    assert r >= 0.6 - 1e-9
    pr_p, pr_r = precision_recall_at_threshold(gt, scores, thr)
    assert 0.0 <= pr_p <= 1.0


def test_metrics_include_precision_fields():
    gt = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.15, 0.9, 0.95, 0.8])
    m = compute_metrics(gt, scores, target_recall=0.5).to_dict()
    for key in ("precision_at_eer", "recall_at_eer", "precision_at_target_recall",
                "threshold_at_target_recall", "target_recall"):
        assert key in m
