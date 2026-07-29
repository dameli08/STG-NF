"""Reproducibility helpers."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> int:
    """Seed Python, NumPy and Torch.

    ``seed == 999`` reproduces the paper behaviour of recording a fresh random
    seed from ``torch.initial_seed()`` (and is returned so callers can log it).
    """
    if seed == 999:
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(0)
        random.seed(seed)
        return seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
