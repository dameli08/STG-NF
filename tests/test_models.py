import numpy as np
import torch

from stgnf.models.stg_nf.model import STG_NF


def _build(in_channels=2, T=24, V=18, K=2, L=1):
    return STG_NF(
        pose_shape=(in_channels, T, V), hidden_channels=0, K=K, L=L, R=0.0,
        actnorm_scale=1.0, flow_permutation="permute", flow_coupling="affine",
        LU_decomposed=True, learn_top=False, edge_importance=False,
        temporal_kernel_size=None, strategy="uniform", max_hops=8, device="cpu")


def test_forward_returns_finite_nll():
    model = _build()
    model.train()
    x = torch.randn(8, 2, 24, 18)
    label = torch.ones(8)
    z, nll = model(x, label=label)
    assert nll.shape[0] == 8
    assert torch.isfinite(nll).all()
    assert z.shape[0] == 8


def test_nll_is_bits_per_dim_scaled():
    model = _build()
    model.train()
    x = torch.randn(4, 2, 24, 18)
    _z, nll = model(x, label=torch.ones(4))
    # Bits/dim of standard normal-ish input should be a small-ish positive number.
    assert nll.mean().item() > 0


def test_confidence_channel_three():
    model = _build(in_channels=3)
    model.train()
    x = torch.randn(4, 3, 24, 18)
    _z, nll = model(x, label=torch.ones(4))
    assert torch.isfinite(nll).all()


def test_set_actnorm_init_flag():
    model = _build()
    model.train()
    x = torch.randn(2, 2, 24, 18)
    model(x, label=torch.ones(2))         # triggers data-dependent init
    model.set_actnorm_init(False)
    from stgnf.models.stg_nf.modules import _ActNorm
    assert all(not m.inited for m in model.modules() if isinstance(m, _ActNorm))
    model.set_actnorm_init(True)
    assert all(m.inited for m in model.modules() if isinstance(m, _ActNorm))
