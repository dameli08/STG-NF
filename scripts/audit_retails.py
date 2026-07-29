#!/usr/bin/env python
"""Audit the RetailS dataset: structure, validity, coordinate/confidence ranges,
track statistics, GT alignment. Writes a JSON report.

Usage:
    python scripts/audit_retails.py --config configs/retails.yaml \
        --out outputs/retails_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stgnf.data.adapter import (  # noqa: E402
    list_clip_paths, load_clip, load_gt_mask, resolve_split_dir,
)
from stgnf.utils.config import load_config  # noqa: E402


def audit_pose_dir(pose_dir: Path, gt_dir: Path | None, limit: int | None):
    paths = list_clip_paths(pose_dir)
    if limit:
        paths = paths[:limit]
    invalid = 0
    persons_per_clip = []
    track_lens = []
    xs, ys, cs = [], [], []
    gt_mismatch = 0
    abnormal, total = 0, 0
    for p in paths:
        try:
            clip = load_clip(p)
        except Exception:
            invalid += 1
            continue
        persons_per_clip.append(len(clip.tracks))
        max_frame = -1
        for t in clip.tracks.values():
            track_lens.append(len(t))
            max_frame = max(max_frame, int(t.frame_ids.max()))
            kp = t.keypoints.reshape(-1, 3)
            xs.append(kp[:, 0]); ys.append(kp[:, 1]); cs.append(kp[:, 2])
        if gt_dir is not None:
            try:
                m = load_gt_mask(gt_dir, clip.stem)
                total += len(m); abnormal += int(m.sum())
                if len(m) <= max_frame:
                    gt_mismatch += 1
            except FileNotFoundError:
                gt_mismatch += 1
    xs = np.concatenate(xs) if xs else np.zeros(1)
    ys = np.concatenate(ys) if ys else np.zeros(1)
    cs = np.concatenate(cs) if cs else np.zeros(1)
    tl = np.array(track_lens) if track_lens else np.zeros(1)
    return {
        "num_files": len(paths),
        "invalid_json": invalid,
        "persons_per_clip": {
            "min": int(min(persons_per_clip)) if persons_per_clip else 0,
            "median": int(np.median(persons_per_clip)) if persons_per_clip else 0,
            "max": int(max(persons_per_clip)) if persons_per_clip else 0,
        },
        "track_len_frames": {
            "min": int(tl.min()), "median": int(np.median(tl)), "max": int(tl.max()),
            "frac_ge_24": float((tl >= 24).mean()), "frac_ge_16": float((tl >= 16).mean()),
        },
        "coord_x": {"min": float(xs.min()), "max": float(xs.max())},
        "coord_y": {"min": float(ys.min()), "max": float(ys.max())},
        "confidence": {"min": float(cs.min()), "max": float(cs.max()), "mean": float(cs.mean())},
        "gt_len_le_maxframe": gt_mismatch,
        "abnormal_frame_fraction": float(abnormal / total) if total else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retails.yaml")
    ap.add_argument("--out", default="outputs/retails_audit.json")
    ap.add_argument("--limit", type=int, default=None, help="limit files per split (debug)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = cfg.dataset
    report = {}
    report["train"] = audit_pose_dir(
        resolve_split_dir(ds.root, ds.pose_train), None, args.limit)
    for name, spec in ds.test_splits.items():
        report[name] = audit_pose_dir(
            resolve_split_dir(ds.root, spec["pose"]),
            resolve_split_dir(ds.root, spec["gt"]), args.limit)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nSaved audit to {args.out}")


if __name__ == "__main__":
    main()
