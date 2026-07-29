import numpy as np

from stgnf.evaluation.metrics import compute_metrics, compute_eer
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
