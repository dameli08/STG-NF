import numpy as np
import torch

from stgnf.data.dataset import build_loaders
from stgnf.evaluation.run import evaluate_splits
from stgnf.models.build import build_model
from stgnf.training.trainer import Trainer


def test_end_to_end_synthetic(synthetic_cfg, tmp_path):
    cfg = synthetic_cfg
    cfg.output.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.output.eval_dir = str(tmp_path / "eval")

    datasets, loaders = build_loaders(cfg, only_test=False)
    train_ds = datasets["train"]
    pose_shape = (cfg.model.in_channels, train_ds.T, train_ds.V)
    device = torch.device("cpu")
    model = build_model(cfg, pose_shape, device)
    trainer = Trainer(cfg, model, loaders, device, ckpt_dir=cfg.output.checkpoint_dir)
    trainer.train()

    eval_datasets = {k: v for k, v in datasets.items() if k != "train"}

    def predict(ds):
        for name, d in datasets.items():
            if d is ds:
                return trainer.predict(loaders[name])
        raise KeyError

    results = evaluate_splits(predict, eval_datasets, cfg, out_dir=cfg.output.eval_dir)
    assert "test" in results
    m = results["test"]
    assert "auc_roc" in m and "auc_pr" in m and "eer" in m
    # Artifacts written.
    assert (tmp_path / "eval" / "test" / "metrics.json").exists()
    assert (tmp_path / "eval" / "test" / "frame_scores.csv").exists()


def test_inference_events(synthetic_cfg, tmp_path, synthetic_pose_json):
    cfg = synthetic_cfg
    cfg.output.checkpoint_dir = str(tmp_path / "ckpt")
    datasets, loaders = build_loaders(cfg, only_test=False)
    train_ds = datasets["train"]
    pose_shape = (cfg.model.in_channels, train_ds.T, train_ds.V)
    device = torch.device("cpu")
    model = build_model(cfg, pose_shape, device)
    trainer = Trainer(cfg, model, loaders, device, ckpt_dir=cfg.output.checkpoint_dir)
    trainer.train()
    trainer.save_checkpoint(0, 0.0)

    from stgnf.inference.infer import PoseAnomalyInference
    engine = PoseAnomalyInference(tmp_path / "ckpt" / "last.pt", device="cpu")
    anomaly, meta = engine.score_clip(synthetic_pose_json)
    assert anomaly.shape[0] >= 40
    finite = anomaly[np.isfinite(anomaly)]
    thr = float(np.percentile(finite, 90)) if finite.size else 0.0
    events = engine.detect_events(anomaly, threshold=thr, min_len=1)
    assert isinstance(events, list)
