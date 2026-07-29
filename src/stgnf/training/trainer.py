"""Training / inference loop for STG-NF.

Adds production concerns over the official loop: mixed precision (bf16 preferred,
fp16 + GradScaler fallback), gradient clipping, per-epoch + best checkpointing,
resume, structured logging and TensorBoard. The optimization math (NLL in
bits/dim, Adamax, exponential decay) is unchanged from the paper.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from stgnf.training.optim import build_optimizer, build_scheduler
from stgnf.utils.logging import get_logger

log = get_logger("training")


class Trainer:
    def __init__(self, cfg, model, loaders, device: torch.device, ckpt_dir: str | Path,
                 tb_dir: Optional[str | Path] = None):
        self.cfg = cfg
        self.model = model
        self.loaders = loaders
        self.device = device
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.in_channels = cfg.model.in_channels
        self.model_confidence = cfg.model.model_confidence
        self.grad_clip = cfg.training.grad_clip
        self.start_epoch = 0

        self.optimizer = build_optimizer(
            model, cfg.training.optimizer, cfg.training.lr, cfg.training.weight_decay)
        self.scheduler = build_scheduler(
            self.optimizer, cfg.training.scheduler, cfg.training.lr,
            cfg.training.lr_decay, cfg.training.epochs)

        self.use_amp = bool(cfg.amp) and device.type == "cuda"
        self.amp_dtype = torch.float32
        self.scaler = None
        if self.use_amp:
            if torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = torch.float16
                self.scaler = torch.cuda.amp.GradScaler()
        self.writer = SummaryWriter(str(tb_dir)) if tb_dir is not None else None

        # Early stopping (optional). Requires a 'val' loader (held-out normal
        # windows). Since training is unsupervised, the stopping criterion is the
        # mean validation NLL of normal data.
        self.val_loader = loaders.get("val")
        es = cfg.get_path("training.early_stopping", None) or {}
        self.es_enable = bool(es.get("enable", False)) and self.val_loader is not None
        self.es_patience = int(es.get("patience", 3))
        self.es_min_delta = float(es.get("min_delta", 0.0))
        self.best_val = float("inf")
        self.epochs_no_improve = 0

    # ------------------------------------------------------------------ utils
    def _prep_batch(self, batch):
        sample, _trans, score, label = batch
        sample = sample.to(self.device, non_blocking=True).float()
        score = score.to(self.device, non_blocking=True).float()
        label = label.to(self.device, non_blocking=True)
        samp = sample if self.model_confidence else sample[:, :self.in_channels]
        return samp, score.amin(dim=-1), label

    def _autocast(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        return _NullCtx()

    # --------------------------------------------------------------- training
    def train(self):
        epochs = self.cfg.training.epochs
        log_interval = self.cfg.training.log_interval
        self.model.train()
        # ActNorm data-dependent init on the very first batch runs in fp32.
        first_batch = True
        global_step = self.start_epoch * len(self.loaders["train"])
        for epoch in range(self.start_epoch, epochs):
            self.model.train()
            log.info("Epoch %d/%d", epoch + 1, epochs)
            pbar = tqdm(self.loaders["train"], desc=f"epoch {epoch + 1}")
            running = 0.0
            n = 0
            for itern, batch in enumerate(pbar):
                samp, score, label = self._prep_batch(batch)
                self.optimizer.zero_grad(set_to_none=True)
                ctx = _NullCtx() if first_batch else self._autocast()
                with ctx:
                    _z, nll = self.model(samp, label=label, score=score)
                    if self.model_confidence:
                        nll = nll * score
                    loss = torch.mean(nll.float())
                if not torch.isfinite(loss):
                    log.warning("Non-finite loss at step %d; skipping batch", global_step)
                    continue
                if self.scaler is not None and not first_batch:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                first_batch = False
                running += loss.item()
                n += 1
                global_step += 1
                if itern % log_interval == 0:
                    pbar.set_description(f"epoch {epoch + 1} | nll {loss.item():.4f}")
                    if self.writer:
                        self.writer.add_scalar("train/nll", loss.item(), global_step)
            avg = running / max(1, n)
            new_lr = self.scheduler.step(epoch)
            log.info("Epoch %d done | mean nll %.4f | next lr %.3e", epoch + 1, avg, new_lr)
            if self.writer:
                self.writer.add_scalar("train/epoch_nll", avg, epoch + 1)
                self.writer.add_scalar("train/lr", new_lr, epoch + 1)

            if self.val_loader is not None:
                val_nll = self._eval_nll(self.val_loader)
                improved = val_nll < (self.best_val - self.es_min_delta)
                log.info("Epoch %d | val nll %.4f | best %.4f%s", epoch + 1, val_nll,
                         min(self.best_val, val_nll), "  <-- improved" if improved else "")
                if self.writer:
                    self.writer.add_scalar("val/epoch_nll", val_nll, epoch + 1)
                if improved:
                    self.best_val = val_nll
                    self.epochs_no_improve = 0
                    self.save_checkpoint(epoch, val_nll, is_best=True)
                else:
                    self.epochs_no_improve += 1
                self.save_checkpoint(epoch, val_nll)
                if self.es_enable and self.epochs_no_improve >= self.es_patience:
                    log.info("Early stopping at epoch %d (no val improvement for %d epochs)",
                             epoch + 1, self.es_patience)
                    break
            else:
                self.save_checkpoint(epoch, avg)

        # Restore the best checkpoint (by val NLL) for the caller's final eval.
        best_path = self.ckpt_dir / "best.pt"
        if self.es_enable and best_path.exists():
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["state_dict"], strict=False)
            self.model.set_actnorm_init()
            log.info("Restored best checkpoint (val nll %.4f) for evaluation.",
                     ckpt.get("metric", float("nan")))
        if self.writer:
            self.writer.flush()

    @torch.no_grad()
    def _eval_nll(self, loader) -> float:
        """Mean NLL (bits/dim) over a loader of normal windows (lower is better)."""
        self.model.eval()
        total, count = 0.0, 0
        for batch in loader:
            samp, score, _label = self._prep_batch(batch)
            label = torch.ones(samp.shape[0], device=self.device)
            with self._autocast():
                _z, nll = self.model(samp, label=label, score=score)
                if self.model_confidence:
                    nll = nll * score
            nll = nll.float()
            total += nll.sum().item()
            count += nll.shape[0]
        return total / max(1, count)

    @torch.no_grad()
    def predict(self, loader) -> np.ndarray:
        """Return per-window **normality** scores (``-nll``) in loader order."""
        self.model.eval()
        out = []
        for batch in tqdm(loader, desc="predict"):
            samp, score, _label = self._prep_batch(batch)
            label = torch.ones(samp.shape[0], device=self.device)
            with self._autocast():
                _z, nll = self.model(samp, label=label, score=score)
                if self.model_confidence:
                    nll = nll * score
            out.append((-1.0 * nll.float()).cpu())
        if not out:
            return np.empty((0,), dtype=np.float32)
        return torch.cat(out, dim=0).numpy().squeeze().astype(np.float32)

    # ------------------------------------------------------------ checkpoints
    def save_checkpoint(self, epoch: int, metric: float, is_best: bool = False):
        state = {
            "epoch": epoch + 1,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.cfg.to_dict(),
            "metric": metric,
        }
        path = self.ckpt_dir / "last.pt"
        torch.save(state, path)
        if is_best:
            torch.save(state, self.ckpt_dir / "best.pt")

    def load_checkpoint(self, path: str | Path, load_optimizer: bool = True):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"], strict=False)
        self.model.set_actnorm_init()
        if load_optimizer and "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except (ValueError, KeyError):
                log.warning("Could not restore optimizer state; continuing fresh.")
        self.start_epoch = ckpt.get("epoch", 0)
        log.info("Loaded checkpoint '%s' (epoch %d)", path, self.start_epoch)
        return ckpt


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False
