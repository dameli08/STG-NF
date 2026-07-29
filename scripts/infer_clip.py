#!/usr/bin/env python
"""Score a single RetailS pose JSON with a trained STG-NF model and print the
per-frame anomaly track plus detected shoplifting events.

Usage:
    python scripts/infer_clip.py --checkpoint outputs/checkpoints/last.pt \
        --pose /path/to/clip.json --threshold 0.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stgnf.inference.infer import PoseAnomalyInference  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pose", required=True, help="path to a RetailS pose JSON")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="event threshold on anomaly score (default: 95th percentile)")
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    engine = PoseAnomalyInference(args.checkpoint, device=args.device)
    anomaly, meta = engine.score_clip(args.pose)
    finite = anomaly[np.isfinite(anomaly)]
    thr = args.threshold if args.threshold is not None else (
        float(np.percentile(finite, 95)) if finite.size else 0.0)
    events = engine.detect_events(anomaly, threshold=thr, min_len=args.min_len)

    result = {
        "clip": Path(args.pose).stem,
        "num_frames": int(len(anomaly)),
        "num_windows": len(meta),
        "threshold": thr,
        "events": [vars(e) for e in events],
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({**result, "anomaly": anomaly.tolist()}, fh, indent=2)
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
