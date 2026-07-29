"""STG-NF flow model.

Faithful port of the official ``models/STG_NF/model_pose.py`` (based on
https://github.com/y0ast/Glow-PyTorch). Additions over the original:

* :meth:`STG_NF.set_actnorm_init` — the official checkpoint loader calls this
  method but the class never defined it; it is implemented here so that resuming
  from a checkpoint does not re-run data-dependent ActNorm initialization.
"""
import math

import torch
import torch.nn as nn

from stgnf.models.stg_nf.modules import (
    Conv2d,
    Conv2dZeros,
    ActNorm2d,
    _ActNorm,
    InvertibleConv1x1,
    Permute2d,
    SqueezeLayer,
    Split2d,
    gaussian_likelihood,
    gaussian_sample,
)
from stgnf.models.stg_nf.flow_utils import split_feature
from stgnf.graph.graph import Graph
from stgnf.models.stg_nf.stgcn import st_gcn


def get_stgcn(in_channels, hidden_channels, out_channels, temporal_kernel_size=9,
              spatial_kernel_size=2, first=False):
    kernel_size = (temporal_kernel_size, spatial_kernel_size)
    if hidden_channels == 0:
        block = nn.ModuleList((
            st_gcn(in_channels, out_channels, kernel_size, 1, residual=(not first)),
        ))
    else:
        block = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=(not first)),
            st_gcn(hidden_channels, out_channels, kernel_size, 1, residual=(not first)),
        ))
    return block


def get_block(in_channels, out_channels, hidden_channels):
    return nn.Sequential(
        Conv2d(in_channels, hidden_channels),
        nn.ReLU(inplace=False),
        Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1), stride=(1, 1)),
        nn.ReLU(inplace=False),
        Conv2dZeros(hidden_channels, out_channels),
    )


class FlowStep(nn.Module):
    def __init__(self, in_channels, hidden_channels, actnorm_scale, flow_permutation,
                 flow_coupling, LU_decomposed, A=None, temporal_kernel_size=4,
                 edge_importance_weighting=False, last=False, first=False,
                 strategy="uniform", max_hops=8, device="cuda:0"):
        super().__init__()
        self.device = device
        self.flow_coupling = flow_coupling
        if A is None:
            g = Graph(strategy=strategy, max_hop=max_hops)
            self.register_buffer("A", torch.from_numpy(g.A).float())

        self.actnorm = ActNorm2d(in_channels, actnorm_scale)

        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)
            self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channels, shuffle=True)
            self.flow_permutation = lambda z, logdet, rev: (self.shuffle(z, rev), logdet)
        else:
            self.reverse = Permute2d(in_channels, shuffle=False)
            self.flow_permutation = lambda z, logdet, rev: (self.reverse(z, rev), logdet)

        if flow_coupling == "additive":
            self.block = get_stgcn(in_channels // 2, in_channels // 2, hidden_channels,
                                   temporal_kernel_size=temporal_kernel_size,
                                   spatial_kernel_size=self.A.size(0), first=first)
        elif flow_coupling == "affine":
            self.block = get_stgcn(in_channels // 2, hidden_channels, in_channels,
                                   temporal_kernel_size=temporal_kernel_size,
                                   spatial_kernel_size=self.A.size(0), first=first)

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size())) for _ in self.block
            ])
        else:
            self.edge_importance = [1] * len(self.block)

    def forward(self, input, logdet=None, reverse=False, label=None):
        if not reverse:
            return self.normal_flow(input, logdet)
        return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)
        z, logdet = self.flow_permutation(z, logdet, False)
        z1, z2 = split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.block(z1)
        elif self.flow_coupling == "affine":
            if len(z1.shape) == 3:
                z1 = z1.unsqueeze(dim=1)
            if len(z2.shape) == 3:
                z2 = z2.unsqueeze(dim=1)
            h = z1.clone()
            for gcn, importance in zip(self.block, self.edge_importance):
                h, _ = gcn(h, self.A * importance)
            shift, scale = split_feature(h, "cross")
            if len(scale.shape) == 3:
                scale = scale.unsqueeze(dim=1)
            if len(shift.shape) == 3:
                shift = shift.unsqueeze(dim=1)
            scale = torch.sigmoid(scale + 2.0) + 1e-6
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)
        return z, logdet

    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0
        z1, z2 = split_feature(input, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.block(z1)
        elif self.flow_coupling == "affine":
            if len(z1.shape) == 3:
                z1 = z1.unsqueeze(dim=1)
            if len(z2.shape) == 3:
                z2 = z2.unsqueeze(dim=1)
            h = z1.clone()
            for gcn, importance in zip(self.block, self.edge_importance):
                h, _ = gcn(h, self.A * importance)
            shift, scale = split_feature(h, "cross")
            if len(scale.shape) == 3:
                scale = scale.unsqueeze(dim=1)
            if len(shift.shape) == 3:
                shift = shift.unsqueeze(dim=1)
            scale = torch.sigmoid(scale + 2.0) + 1e-6
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)
        z, logdet = self.flow_permutation(z, logdet, True)
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)
        return z, logdet


class FlowNet(nn.Module):
    def __init__(self, pose_shape, hidden_channels, K, L, actnorm_scale,
                 flow_permutation, flow_coupling, LU_decomposed, edge_importance=False,
                 temporal_kernel_size=None, strategy="uniform", max_hops=8, device="cuda:0"):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.K = K
        C, T, V = pose_shape
        for i in range(L):
            if i > 1:
                C, T, V = C * 2, T // 2, V
                self.layers.append(SqueezeLayer(factor=2))
                self.output_shapes.append([-1, C, T, V])
            if temporal_kernel_size is None:
                temporal_kernel_size = T // 2 + 1
            for k in range(K):
                last = k == K - 1
                first = k == 0
                self.layers.append(
                    FlowStep(
                        in_channels=C,
                        hidden_channels=hidden_channels,
                        actnorm_scale=actnorm_scale,
                        flow_permutation=flow_permutation,
                        flow_coupling=flow_coupling,
                        LU_decomposed=LU_decomposed,
                        temporal_kernel_size=temporal_kernel_size,
                        edge_importance_weighting=edge_importance,
                        last=last,
                        first=first,
                        strategy=strategy,
                        max_hops=max_hops,
                        device=device,
                    )
                )
                self.output_shapes.append([-1, C, T, V])

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            return self.decode(input, temperature)
        return self.encode(input, logdet)

    def encode(self, z, logdet=0.0):
        logdet = torch.zeros(z.shape[0], device=z.device)
        for layer, shape in zip(self.layers, self.output_shapes):
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def decode(self, z, temperature=None):
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z, logdet = layer(z, logdet=0, reverse=True, temperature=temperature)
            else:
                z, logdet = layer(z, logdet=0, reverse=True)
        return z


class STG_NF(nn.Module):
    def __init__(self, pose_shape, hidden_channels, K, L, actnorm_scale, flow_permutation,
                 flow_coupling, LU_decomposed, learn_top, R=0, edge_importance=False,
                 temporal_kernel_size=None, strategy="uniform", max_hops=8, device="cuda:0"):
        super().__init__()
        self.flow = FlowNet(
            pose_shape=pose_shape,
            hidden_channels=hidden_channels,
            K=K, L=L,
            actnorm_scale=actnorm_scale,
            flow_permutation=flow_permutation,
            flow_coupling=flow_coupling,
            LU_decomposed=LU_decomposed,
            edge_importance=edge_importance,
            temporal_kernel_size=temporal_kernel_size,
            strategy=strategy,
            max_hops=max_hops,
            device=device,
        )
        self.R = R
        self.learn_top = learn_top

        if learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top_fn = Conv2dZeros(C * 2, C * 2)

        c_out = self.flow.output_shapes[-1][1]
        t_out = self.flow.output_shapes[-1][2]
        v_out = self.flow.output_shapes[-1][3]
        self.register_buffer("prior_h", torch.zeros([1, c_out * 2, t_out, v_out]))
        self.register_buffer(
            "prior_h_normal",
            torch.concat(
                (torch.ones([c_out, t_out, v_out]) * self.R, torch.zeros([c_out, t_out, v_out])),
                dim=0,
            ),
        )
        self.register_buffer(
            "prior_h_abnormal",
            torch.concat(
                (torch.ones([c_out, t_out, v_out]) * self.R * -1, torch.zeros([c_out, t_out, v_out])),
                dim=0,
            ),
        )

    def set_actnorm_init(self, inited=True):
        """Mark all ActNorm layers as initialized (used after loading a checkpoint)."""
        for module in self.modules():
            if isinstance(module, _ActNorm):
                module.inited = inited

    def prior(self, data, label=None):
        if data is not None:
            if label is not None:
                h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
                h[label == 1] = self.prior_h_normal
                h[label == -1] = self.prior_h_abnormal
            else:
                h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
        else:
            h = self.prior_h_normal.repeat(32, 1, 1, 1)
        if self.learn_top:
            h = self.learn_top_fn(h)
        return split_feature(h, "split")

    def forward(self, x=None, z=None, temperature=None, reverse=False, label=None, score=1):
        if reverse:
            return self.reverse_flow(z, temperature)
        return self.normal_flow(x, label, score)

    def normal_flow(self, x, label, score):
        b, c, t, v = x.shape
        z, objective = self.flow(x, reverse=False)
        mean, logs = self.prior(x, label)
        objective = objective + gaussian_likelihood(mean, logs, z)
        nll = (-objective) / (math.log(2.0) * c * t * v)
        return z, nll

    def reverse_flow(self, z, temperature):
        with torch.no_grad():
            if z is None:
                mean, logs = self.prior(z)
                z = gaussian_sample(mean, logs, temperature)
            x = self.flow(z, temperature=temperature, reverse=True)
        return x
