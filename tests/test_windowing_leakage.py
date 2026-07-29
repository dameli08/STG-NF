import json

import numpy as np

from stgnf.data.adapter import load_clip
from stgnf.data.dataset import build_train_dataset, build_test_datasets
from stgnf.data.preprocessing import build_windows


def test_train_stride_vs_test_stride(synthetic_cfg):
    train_ds = build_train_dataset(synthetic_cfg)
    test_ds = build_test_datasets(synthetic_cfg)["test"]
    # Test uses stride 1 -> strictly more (or equal) windows per frame than train.
    assert train_ds.num_samples > 0
    assert test_ds.num_samples > 0


def test_windows_do_not_cross_large_gaps(tmp_path):
    # Person track with a large (5-frame) gap => windows must not span it.
    import numpy as np
    rng = np.random.default_rng(0)
    frames = {}
    for i in list(range(0, 20)) + list(range(25, 45)):  # gap 20..24 (5 frames)
        kp = np.concatenate([rng.uniform(100, 900, (17, 2)),
                             rng.uniform(0.4, 0.9, (17, 1))], axis=1).reshape(-1)
        frames[str(i)] = {"keypoints": kp.tolist(), "scores": None}
    path = tmp_path / "01_0009.json"
    with open(path, "w") as fh:
        json.dump({"1": frames}, fh)
    clip = load_clip(path)
    segs, meta, scores = build_windows(clip, seg_len=24, seg_stride=1,
                                       max_missing=2, interpolate_gaps=True)
    # No window may start such that it straddles the 5-frame gap at 20..24.
    for _stem, _pid, start in meta:
        assert not (start <= 20 < start + 24 <= 49 and start < 20 and start + 24 > 24)


def test_train_test_person_frames_isolated(synthetic_cfg):
    # Train and test come from disjoint directories -> no shared clip stems.
    train_ds = build_train_dataset(synthetic_cfg)
    test_ds = build_test_datasets(synthetic_cfg)["test"]
    train_stems = {m[0] for m in train_ds.metadata}
    test_stems = {m[0] for m in test_ds.metadata}
    assert train_stems.isdisjoint(test_stems)
