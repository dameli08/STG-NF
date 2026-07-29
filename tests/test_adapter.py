import numpy as np

from stgnf.data.adapter import load_clip, load_gt_mask


def test_outer_person_inner_frame_nesting(synthetic_pose_json):
    clip = load_clip(synthetic_pose_json)
    # Two persons keyed by id, each with a temporally ordered track.
    assert set(clip.tracks.keys()) == {1, 2}
    t1 = clip.tracks[1]
    assert t1.keypoints.shape[1:] == (17, 3)
    # Frame ids ascending.
    assert (np.diff(t1.frame_ids) > 0).all()


def test_scores_derived_from_confidence(synthetic_pose_json):
    clip = load_clip(synthetic_pose_json)
    t = clip.tracks[1]
    # scores field was null -> derived as mean joint confidence in [0,1].
    assert t.scores.shape[0] == len(t)
    assert np.all(t.scores >= 0)
    assert np.all(t.scores <= 1.01)


def test_num_frames(synthetic_pose_json):
    clip = load_clip(synthetic_pose_json)
    assert clip.num_frames >= 40


def test_gt_roundtrip(tmp_path):
    import numpy as np
    m = np.array([0, 1, 1, 0], dtype=np.uint8)
    np.save(tmp_path / "c.npy", m)
    loaded = load_gt_mask(tmp_path, "c")
    assert loaded.tolist() == [0, 1, 1, 0]
