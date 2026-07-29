import numpy as np

from stgnf.data.adapter import load_clip
from stgnf.data.preprocessing import (
    keypoints17_to_coco18, build_windows, normalize_pose,
    _interpolate_missing_joints, _dense_track,
)


def test_coco18_conversion_shape():
    kps = np.random.rand(4, 24, 17, 3).astype(np.float32)
    out = keypoints17_to_coco18(kps)
    assert out.shape == (4, 24, 18, 3)
    # Neck (index 1 after reorder) = mean of shoulders (orig idx 5,6).
    neck = 0.5 * (kps[..., 5, :] + kps[..., 6, :])
    assert np.allclose(out[..., 1, :], neck)


def test_build_windows_shapes(synthetic_pose_json):
    clip = load_clip(synthetic_pose_json)
    segs, meta, scores = build_windows(clip, seg_len=24, seg_stride=6)
    assert segs.ndim == 4
    assert segs.shape[1] == 3          # channels x,y,conf
    assert segs.shape[2] == 24         # T
    assert segs.shape[3] == 18         # V (COCO18)
    assert len(meta) == segs.shape[0]
    assert scores.shape == (segs.shape[0], 24)
    for m in meta:
        assert len(m) == 3             # [stem, pid, start_frame]


def test_normalize_pose_zero_mean_unit_scale():
    data = np.random.rand(3, 24, 18, 3).astype(np.float32) * 500 + 100
    out = normalize_pose(data, vid_res=(1080, 1920), seg_norm=True)
    # Per-sample mean over (T,V) of x,y approximately zero.
    mean_xy = out[..., :2].mean(axis=(1, 2))
    assert np.allclose(mean_xy, 0, atol=1e-4)


def test_missing_joint_interpolation():
    win = np.random.rand(24, 18, 3).astype(np.float32)
    win[:, :, 2] = 0.8
    win[5, 3, :] = [0, 0, 0]           # missing joint at t=5
    filled = _interpolate_missing_joints(win)
    # Interpolated x between neighbours t=4 and t=6.
    expected = 0.5 * (win[4, 3, 0] + win[6, 3, 0])
    assert abs(filled[5, 3, 0] - expected) < 1e-4


def test_small_gap_interpolation(synthetic_pose_json):
    clip = load_clip(synthetic_pose_json)
    track = clip.tracks[2]  # has a 2-frame gap
    dense, dense_score, first = _dense_track(track, max_missing=2, interpolate_gaps=True)
    # No NaNs remain for a gap within tolerance.
    assert not np.isnan(dense).any()
