#!/usr/bin/env python
"""Evaluate a trained STG-NF checkpoint on the RetailS test splits.

Usage:
    python scripts/evaluate.py --config configs/retails.yaml \
        --checkpoint outputs/checkpoints/last.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stgnf.data.dataset import build_loaders  # noqa: E402
from stgnf.evaluation.run import evaluate_splits  # noqa: E402
from stgnf.models.build import build_model  # noqa: E402
from stgnf.training.trainer import Trainer  # noqa: E402
from stgnf.utils.config import load_config  # noqa: E402
from stgnf.utils.device import select_device  # noqa: E402
from stgnf.utils.logging import get_logger  # noqa: E402

log = get_logger("evaluate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retails.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None, help="override eval output dir")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    device = select_device(cfg.device)
    log.info("Device: %s", device)

    datasets, loaders = build_loaders(cfg, only_test=True)
    eval_datasets = {k: v for k, v in datasets.items() if k != "train"}
    any_ds = next(iter(eval_datasets.values()))
    pose_shape = (cfg.model.in_channels, any_ds.T, any_ds.V)

    model = build_model(cfg, pose_shape, device)
    trainer = Trainer(cfg, model, loaders, device,
                      ckpt_dir=cfg.output.checkpoint_dir, tb_dir=None)
    trainer.load_checkpoint(args.checkpoint, load_optimizer=False)

    out_dir = args.out or cfg.output.eval_dir
    results = evaluate_splits(
        predict_fn=lambda ds: trainer.predict(loaders[_split_name(datasets, ds)]),
        datasets=eval_datasets, cfg=cfg, out_dir=out_dir)
    log.info("Results: %s", results)


def _split_name(datasets, ds):
    for name, d in datasets.items():
        if d is ds:
            return name
    raise KeyError("dataset not found")


if __name__ == "__main__":
    main()
