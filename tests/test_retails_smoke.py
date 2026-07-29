"""Smoke test against the real RetailS dataset. Skipped automatically when the
dataset root from ``configs/retails.yaml`` is not present.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from stgnf.data.adapter import list_clip_paths, load_clip, load_gt_mask, resolve_split_dir
from stgnf.data.preprocessing import build_windows, normalize_segments
from stgnf.models.build import build_model
from stgnf.utils.config import load_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "retails.yaml"


def _cfg_or_skip():
    if not CONFIG.exists():
        pytest.skip("config not found")
    cfg = load_config(CONFIG)
    root = Path(cfg.dataset.root)
    if not root.exists():
        pytest.skip(f"RetailS root not found: {root}")
    return cfg


def test_retails_train_parses_and_windows():
    cfg = _cfg_or_skip()
    ds = cfg.dataset
    train_dir = resolve_split_dir(ds.root, ds.pose_train)
    paths = list_clip_paths(train_dir)[:5]
    assert paths, "no train pose files found"
    total_windows = 0
    for p in paths:
        clip = load_clip(p)
        # outer=person, inner=frame: keypoints must be 17x3.
        for t in clip.tracks.values():
            assert t.keypoints.shape[1:] == (17, 3)
        segs, meta, scores = build_windows(
            clip, seg_len=cfg.preprocessing.seg_len, seg_stride=cfg.preprocessing.seg_stride,
            kp18_format=ds.kp18_format)
        total_windows += segs.shape[0]
        if segs.shape[0]:
            assert segs.shape[1:] == (3, cfg.preprocessing.seg_len, 18)
    assert total_windows > 0


def test_retails_test_gt_is_frame_level():
    cfg = _cfg_or_skip()
    ds = cfg.dataset
    name, spec = next(iter(ds.test_splits.items()))
    pose_dir = resolve_split_dir(ds.root, spec["pose"])
    gt_dir = resolve_split_dir(ds.root, spec["gt"])
    p = list_clip_paths(pose_dir)[0]
    clip = load_clip(p)
    gt = load_gt_mask(gt_dir, clip.stem)
    max_frame = max(int(t.frame_ids.max()) for t in clip.tracks.values())
    assert gt.ndim == 1
    assert len(gt) > max_frame            # frame-level, not track-index
    assert set(np.unique(gt)).issubset({0, 1})


def test_retails_forward_pass():
    cfg = _cfg_or_skip()
    ds = cfg.dataset
    train_dir = resolve_split_dir(ds.root, ds.pose_train)
    clip = load_clip(list_clip_paths(train_dir)[0])
    segs, meta, scores = build_windows(
        clip, seg_len=cfg.preprocessing.seg_len, seg_stride=cfg.preprocessing.seg_stride)
    if segs.shape[0] == 0:
        pytest.skip("first clip produced no windows")
    norm = normalize_segments(segs[:8], vid_res=ds.vid_res)
    device = torch.device("cpu")
    model = build_model(cfg, (cfg.model.in_channels, cfg.preprocessing.seg_len, 18), device)
    model.train()
    x = torch.from_numpy(norm[:, :cfg.model.in_channels]).float()
    _z, nll = model(x, label=torch.ones(x.shape[0]))
    assert torch.isfinite(nll).all()
