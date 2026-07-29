"""Shared pytest fixtures: synthetic RetailS-format data for fast, offline tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stgnf.utils.config import Config


def _make_person(num_frames: int, start_frame: int = 0, seed: int = 0, gap_at=None) -> dict:
    rng = np.random.default_rng(seed)
    frames = {}
    base = rng.uniform(300, 700, size=(17, 2))
    for i in range(num_frames):
        fid = start_frame + i
        if gap_at is not None and gap_at[0] <= i < gap_at[1]:
            continue  # simulate a missing-frame gap
        motion = base + rng.normal(0, 3, size=(17, 2)) + i * 1.5
        conf = rng.uniform(0.3, 0.95, size=(17, 1))
        kp = np.concatenate([motion, conf], axis=1).reshape(-1)
        frames[str(fid)] = {"keypoints": kp.tolist(), "scores": None}
    return frames


@pytest.fixture
def synthetic_pose_json(tmp_path) -> Path:
    """A single-clip pose JSON with two people (outer=person, inner=frame)."""
    clip = {
        "1": _make_person(40, start_frame=0, seed=1),
        "2": _make_person(30, start_frame=5, seed=2, gap_at=(10, 12)),
    }
    path = tmp_path / "01_0001.json"
    with open(path, "w") as fh:
        json.dump(clip, fh)
    return path


@pytest.fixture
def synthetic_split(tmp_path) -> dict:
    """A small train + test split with GT masks. Returns a paths dict."""
    root = tmp_path / "RetailS"
    train_dir = root / "train"
    test_pose = root / "test" / "pose"
    test_gt = root / "test" / "gt"
    for d in (train_dir, test_pose, test_gt):
        d.mkdir(parents=True, exist_ok=True)

    for c in range(4):
        clip = {"1": _make_person(60, start_frame=0, seed=10 + c),
                "2": _make_person(50, start_frame=3, seed=20 + c)}
        with open(train_dir / f"1_{c:04d}_1.json", "w") as fh:
            json.dump(clip, fh)

    for c in range(3):
        n = 60
        clip = {"1": _make_person(n, start_frame=0, seed=100 + c)}
        stem = f"01_{c:04d}"
        with open(test_pose / f"{stem}.json", "w") as fh:
            json.dump(clip, fh)
        gt = np.zeros(n, dtype=np.uint8)
        gt[30:45] = 1  # abnormal window
        np.save(test_gt / f"{stem}.npy", gt)

    return {"root": str(root), "train": "train",
            "test_pose": "test/pose", "test_gt": "test/gt"}


@pytest.fixture
def synthetic_cfg(synthetic_split) -> Config:
    return Config({
        "seed": 42, "device": "cpu", "amp": False,
        "dataset": {
            "name": "RetailS", "root": synthetic_split["root"],
            "pose_train": synthetic_split["train"],
            "test_splits": {"test": {"pose": synthetic_split["test_pose"],
                                     "gt": synthetic_split["test_gt"]}},
            "vid_res": [1080, 1920], "num_keypoints": 17,
            "kp18_format": True, "headless": False,
        },
        "preprocessing": {
            "seg_len": 24, "seg_stride": 6, "train_seg_conf_th": 0.0,
            "max_missing": 2, "interpolate_gaps": True,
            "normalization": {"symm_range": False, "seg_norm": True},
        },
        "model": {
            "in_channels": 2, "model_confidence": False, "K": 2, "L": 1, "R": 0.0,
            "hidden_channels": 0, "actnorm_scale": 1.0, "flow_permutation": "permute",
            "flow_coupling": "affine", "LU_decomposed": True, "learn_top": False,
            "edge_importance": False, "temporal_kernel_size": None,
            "adj_strategy": "uniform", "max_hops": 8,
        },
        "training": {
            "epochs": 1, "batch_size": 32, "num_workers": 0, "optimizer": "adamax",
            "lr": 5.0e-4, "weight_decay": 5.0e-5, "scheduler": "exp_decay",
            "lr_decay": 0.99, "grad_clip": 100.0, "num_transform": 2, "log_interval": 5,
        },
        "evaluation": {"smoothing_sigma": 3, "frame_offset": "half"},
        "output": {"root": "outputs", "checkpoint_dir": "outputs/ckpt",
                   "tensorboard_dir": "outputs/tb", "eval_dir": "outputs/eval"},
    })
