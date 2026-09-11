"""Quick smoke test for per-level LR param groups."""
import sys
sys.path.insert(0, ".")
from imaginaire.config import Config
from importlib import import_module

Model = import_module("projects.Bspline-Neus.model").Model

cfg = Config("projects/Bspline-Neus/configs/dtu_scan24_thb_v3_color_mlp_v12_edge_pe_E22a_share_deg2_maxL5_50k.yaml")
cfg.model.object.bspline.sdf_level_lr_scale = [2.0, 1.5, 1.0, 0.7, 0.5]
cfg.model.object.bspline.color_feature_level_lr_scale = [0.5, 0.7, 1.0, 1.5, 2.0]
cfg.data.val.subset = None
model = Model(cfg.model, cfg.data)

base_lr = cfg.optim.params.lr
feature_lr = cfg.model.object.bspline.color_feature_lr

groups = model.get_param_groups(cfg.optim)
print(f"Total param groups: {len(groups)}")
print(f"base_lr={base_lr}, feature_lr={feature_lr}")
for i, g in enumerate(groups):
    n = sum(p.numel() for p in g["params"])
    print(f"  group {i:2d}: lr={g['lr']:.6f}  params={n:>12,}")

# Without per-level scales (backward compat).
cfg2 = Config("projects/Bspline-Neus/configs/dtu_scan24_thb_v3_color_mlp_v12_edge_pe_E22a_share_deg2_maxL5_50k.yaml")
cfg2.data.val.subset = None
model2 = Model(cfg2.model, cfg2.data)
groups2 = model2.get_param_groups(cfg2.optim)
print(f"\nWithout level scales: {len(groups2)} groups")
for i, g in enumerate(groups2):
    n = sum(p.numel() for p in g["params"])
    print(f"  group {i:2d}: lr={g['lr']:.6f}  params={n:>12,}")
