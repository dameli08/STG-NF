"""Model construction from configuration."""
from __future__ import annotations

from typing import Tuple

import torch

from stgnf.models.stg_nf.model import STG_NF


def build_model(cfg, pose_shape: Tuple[int, int, int], device: torch.device) -> STG_NF:
    """Instantiate :class:`STG_NF` from a config and an input ``(C, T, V)`` shape."""
    m = cfg.model
    model = STG_NF(
        pose_shape=pose_shape,
        hidden_channels=m.hidden_channels,
        K=m.K,
        L=m.L,
        R=m.R,
        actnorm_scale=m.actnorm_scale,
        flow_permutation=m.flow_permutation,
        flow_coupling=m.flow_coupling,
        LU_decomposed=m.LU_decomposed,
        learn_top=m.learn_top,
        edge_importance=m.edge_importance,
        temporal_kernel_size=m.temporal_kernel_size,
        strategy=m.adj_strategy,
        max_hops=m.max_hops,
        device=str(device),
    )
    return model.to(device)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
