#!/usr/bin/env python
"""Train STG-NF on RetailS, then evaluate on the configured test splits.

Usage:
    python scripts/train.py --config configs/retails.yaml
    python scripts/train.py --config configs/retails.yaml \
        --set training.epochs=2 --set device=cuda:0 --resume outputs/checkpoints/last.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stgnf.data.dataset import build_loaders  # noqa: E402
from stgnf.evaluation.run import evaluate_splits  # noqa: E402
from stgnf.models.build import build_model, count_parameters  # noqa: E402
from stgnf.training.trainer import Trainer  # noqa: E402
from stgnf.utils.config import load_config  # noqa: E402
from stgnf.utils.device import select_device  # noqa: E402
from stgnf.utils.logging import get_logger  # noqa: E402
from stgnf.utils.seed import set_seed  # noqa: E402

log = get_logger("train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retails.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="dotted config override, e.g. training.epochs=2")
    ap.add_argument("--resume", default=None, help="checkpoint path to resume from")
    ap.add_argument("--no-eval", action="store_true", help="skip evaluation after training")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    seed = set_seed(cfg.seed)
    log.info("Seed: %d", seed)
    device = select_device(cfg.device)
    log.info("Device: %s", device)

    datasets, loaders = build_loaders(cfg, only_test=False)
    train_ds = datasets["train"]
    log.info("Train windows: %d (C=%d, T=%d, V=%d)",
             train_ds.num_samples, train_ds.C, train_ds.T, train_ds.V)

    pose_shape = (cfg.model.in_channels, train_ds.T, train_ds.V)
    model = build_model(cfg, pose_shape, device)
    log.info("Model parameters: %.1fK", count_parameters(model) / 1e3)

    out = cfg.output
    trainer = Trainer(cfg, model, loaders, device,
                      ckpt_dir=out.checkpoint_dir, tb_dir=out.tensorboard_dir)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()

    if not args.no_eval:
        eval_datasets = {k: v for k, v in datasets.items() if k not in ("train", "val")}
        evaluate_splits(
            predict_fn=lambda ds: trainer.predict(loaders[_split_name(datasets, ds)]),
            datasets=eval_datasets, cfg=cfg, out_dir=out.eval_dir)


def _split_name(datasets, ds):
    for name, d in datasets.items():
        if d is ds:
            return name
    raise KeyError("dataset not found in datasets map")


if __name__ == "__main__":
    main()
