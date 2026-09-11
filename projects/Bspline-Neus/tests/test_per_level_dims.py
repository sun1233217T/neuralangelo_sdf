"""Smoke test for per-level color feature dims."""
import sys
sys.path.insert(0, ".")
from imaginaire.config import Config
from importlib import import_module
import torch

Model = import_module("projects.Bspline-Neus.model").Model
cfg = Config("projects/Bspline-Neus/configs/stage_search/A1_perlevel_dim_150k.yaml")
cfg.data.val.subset = None
model = Model(cfg.model, cfg.data).cuda()
rgb = model.neural_rgb
print("per_level_dims:", rgb._per_level_dims)
print("channels_per_level:", rgb.hier_field.channels_per_level)
print("L0 values shape:", rgb.hier_field.levels[0].values.shape)

# Test evaluate_per_level
pts = torch.randn(10, 3).cuda() * 0.5
feats = rgb.hier_field.evaluate_per_level(pts)
print("evaluate_per_level output shape:", feats.shape)
expected_dim = sum(rgb.hier_field.channels_per_level)
print("expected dim:", expected_dim, "OK" if feats.shape[-1] == expected_dim else "MISMATCH")

# Test forward
normals = torch.randn(10, 3).cuda()
rays = torch.randn(10, 3).cuda()
out = rgb.forward(pts, normals, rays, None, None)
print("RGB output shape:", out.shape)
print("ALL OK")
