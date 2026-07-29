import numpy as np
import torch

from stgnf.data.dataset import build_loaders
from stgnf.models.build import build_model
from stgnf.training.trainer import Trainer


def test_training_step_and_checkpoint(synthetic_cfg, tmp_path):
    cfg = synthetic_cfg
    cfg.output.checkpoint_dir = str(tmp_path / "ckpt")
    datasets, loaders = build_loaders(cfg, only_test=False)
    train_ds = datasets["train"]
    assert train_ds.num_samples > 0

    pose_shape = (cfg.model.in_channels, train_ds.T, train_ds.V)
    device = torch.device("cpu")
    model = build_model(cfg, pose_shape, device)
    trainer = Trainer(cfg, model, loaders, device, ckpt_dir=cfg.output.checkpoint_dir)

    trainer.train()
    ckpt = tmp_path / "ckpt" / "last.pt"
    assert ckpt.exists()

    # Reload into a fresh model and confirm predictions are finite.
    model2 = build_model(cfg, pose_shape, device)
    trainer2 = Trainer(cfg, model2, loaders, device, ckpt_dir=cfg.output.checkpoint_dir)
    trainer2.load_checkpoint(ckpt)
    scores = trainer2.predict(loaders["test"])
    assert scores.shape[0] == len(datasets["test"].metadata)
    assert np.isfinite(scores).all()


def test_predict_length_matches_metadata(synthetic_cfg):
    cfg = synthetic_cfg
    datasets, loaders = build_loaders(cfg, only_test=True)
    ds = datasets["test"]
    pose_shape = (cfg.model.in_channels, ds.T, ds.V)
    device = torch.device("cpu")
    model = build_model(cfg, pose_shape, device)
    trainer = Trainer(cfg, model, loaders, device, ckpt_dir="/tmp/ckpt_test")
    # Warm up ActNorm's data-dependent init (required before eval-mode inference).
    model.train()
    warmup = next(iter(loaders["test"]))
    samp, score, label = trainer._prep_batch(warmup)
    model(samp, label=torch.ones(samp.shape[0]), score=score)
    scores = trainer.predict(loaders["test"])
    assert scores.shape[0] == len(ds.metadata)


def test_early_stopping_val_split_and_best_ckpt(synthetic_cfg, tmp_path):
    cfg = synthetic_cfg
    cfg.output.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.training.epochs = 4
    cfg.training.early_stopping = {
        "enable": True, "patience": 1, "min_delta": 1e9,  # force a stop quickly
        "val_fraction": 0.25, "seed": 0,
    }
    datasets, loaders = build_loaders(cfg, only_test=False)
    # A held-out validation loader must exist and not overlap train clips.
    assert "val" in loaders and loaders["val"] is not None
    train_stems = {m[0] for m in datasets["train"].metadata}
    val_stems = {m[0] for m in datasets["val"].metadata}
    assert train_stems.isdisjoint(val_stems)

    pose_shape = (cfg.model.in_channels, datasets["train"].T, datasets["train"].V)
    device = torch.device("cpu")
    model = build_model(cfg, pose_shape, device)
    trainer = Trainer(cfg, model, loaders, device, ckpt_dir=cfg.output.checkpoint_dir)
    assert trainer.es_enable is True
    trainer.train()
    # Best checkpoint is written and training stopped before all epochs (patience=1,
    # min_delta huge => second epoch never "improves").
    assert (tmp_path / "ckpt" / "best.pt").exists()

