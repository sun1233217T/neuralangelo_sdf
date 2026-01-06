'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from projects.sdf_GS_gengeration.model import Model as FullModel
from projects.sdf_GS_gengeration.utils.modules import NeuralGS


class MiniNeuralSDF(torch.nn.Module):
    def __init__(self, hidden_dim=512, dropout_prob=0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 3:
            raise ValueError("hidden_dim must be > 3 for skip concatenation.")
        self.normal_eps = 1e-3
        self.warm_up_end = 0
        self.active_levels = 1
        self.anneal_levels = 1
        self.growth_rate = 1.0
        self.cfg_sdf = SimpleNamespace(
            encoding=SimpleNamespace(coarse2fine=SimpleNamespace(enabled=False), levels=0),
            gradient=SimpleNamespace(mode="analytical"),
        )

        self.fc1 = weight_norm(nn.Linear(3, self.hidden_dim))
        self.fc2 = weight_norm(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.fc3 = weight_norm(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.fc4 = weight_norm(nn.Linear(self.hidden_dim, self.hidden_dim - 3))
        self.fc5 = weight_norm(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.fc6 = weight_norm(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.fc7 = weight_norm(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.fc8 = nn.Linear(self.hidden_dim, 1)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.tanh = nn.Tanh()

    def forward(self, points_3d, with_sdf=True, with_feat=True):
        x_input = points_3d
        x = self.dropout(self.prelu(self.fc1(x_input)))
        x = self.dropout(self.prelu(self.fc2(x)))
        x = self.dropout(self.prelu(self.fc3(x)))
        x = self.dropout(self.prelu(self.fc4(x)))
        x = torch.cat([x, x_input], dim=-1)
        x = self.dropout(self.prelu(self.fc5(x)))
        x = self.dropout(self.prelu(self.fc6(x)))
        x = self.dropout(self.prelu(self.fc7(x)))
        sdf = self.tanh(self.fc8(x)) if with_sdf else None
        feat = x if with_feat else None
        return sdf, feat

    def sdf(self, points_3d):
        return self.forward(points_3d, with_sdf=True, with_feat=False)[0]

    def compute_gradients(self, x, training=False, sdf=None):
        requires_grad = x.requires_grad
        with torch.enable_grad():
            x.requires_grad_(True)
            sdf_val = self.sdf(x)
            gradient = torch.autograd.grad(sdf_val.sum(), x, create_graph=training)[0]
            if training:
                hessian = torch.zeros_like(gradient)
            else:
                gradient = gradient.detach()
                hessian = None
        x.requires_grad_(requires_grad)
        return gradient, hessian

    def set_active_levels(self, current_iter=None):
        self.active_levels = 1
        self.anneal_levels = 1
        return None

    def set_normal_epsilon(self):
        return None


class Model(FullModel):
    def build_model(self, cfg_model, cfg_data):
        mini_cfg = getattr(cfg_model, "mini", None)
        sdf_hidden = getattr(mini_cfg, "sdf_hidden_dim", 512) if mini_cfg else 512
        sdf_dropout = getattr(mini_cfg, "sdf_dropout", 0.1) if mini_cfg else 0.1
        self.neural_sdf = MiniNeuralSDF(hidden_dim=sdf_hidden, dropout_prob=sdf_dropout)
        self.neural_gs = NeuralGS(cfg_model.object.rgb, feat_dim=self.neural_sdf.hidden_dim,
                                  appear_embed=cfg_model.appear_embed, output_dim=10)
        self.s_var = torch.nn.Parameter(torch.tensor(cfg_model.object.s_var.init_val, dtype=torch.float32))
