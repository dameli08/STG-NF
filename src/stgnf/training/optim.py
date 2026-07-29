"""Optimizer and LR-scheduler factories (faithful to STG-NF defaults)."""
from __future__ import annotations

import torch.optim as optim


def build_optimizer(model, name: str, lr: float, weight_decay: float):
    name = name.lower()
    if name in ("adamax", "adamx"):
        return optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


class ExpDecay:
    """Manual exponential LR decay: ``lr * decay**epoch`` (paper 'exp_decay')."""

    def __init__(self, optimizer, base_lr: float, decay: float):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.decay = decay

    def step(self, epoch: int) -> float:
        new_lr = self.base_lr * (self.decay ** epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
        return new_lr


def build_scheduler(optimizer, name: str, base_lr: float, lr_decay: float, epochs: int):
    """Return a stepper with a ``step(epoch) -> lr`` interface."""
    name = (name or "exp_decay").lower()
    if name in ("exp_decay", "none"):
        return ExpDecay(optimizer, base_lr, lr_decay if name == "exp_decay" else 1.0)
    if name == "cosine":
        import torch.optim.lr_scheduler as sched
        cos = sched.CosineAnnealingLR(optimizer, T_max=epochs)

        class _CosWrap:
            def step(self, epoch: int) -> float:
                cos.step()
                return cos.get_last_lr()[0]

        return _CosWrap()
    raise ValueError(f"Unknown scheduler: {name}")
