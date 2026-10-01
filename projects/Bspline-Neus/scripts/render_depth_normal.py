"""Render depth and normal maps from a trained checkpoint."""
import sys, os
sys.path.insert(0, ".")
import torch
import numpy as np
from importlib import import_module
from imaginaire.config import Config
import torchvision.utils as tvu

Dataset = import_module("projects.Bspline-Neus.data").Dataset
Model = import_module("projects.Bspline-Neus.model").Model

cfg_path = sys.argv[1]
ckpt_path = sys.argv[2]
out_dir = sys.argv[3]
indices = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 12, 24, 36, 48]

cfg = Config(cfg_path)
cfg.data.val.subset = None
ds = Dataset(cfg, is_inference=True)
model = Model(cfg.model, cfg.data).cuda()
model.progress = 1.0
ckpt = torch.load(ckpt_path, map_location=lambda s, l: s)
sd = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt["model"].items()}
model.load_state_dict(sd, strict=False)
model.eval()

os.makedirs(out_dir, exist_ok=True)
with torch.no_grad():
    for i in indices:
        data = ds[i]
        data = {k: (v[None].cuda() if torch.is_tensor(v) else v) for k, v in data.items()}
        out = model.inference(data)
        depth = out["depth_map"][0]  # (1,H,W)
        normal = out["normal_map"][0]  # (3,H,W), in [-1,1]
        opacity = out["opacity_map"][0]
        # depth: normalize within the opaque region for visibility
        d = depth.clone()
        valid = opacity[0] > 0.5
        if valid.any():
            lo, hi = d[0][valid].min(), d[0][valid].max()
            d[0][valid] = (d[0][valid] - lo) / (hi - lo + 1e-8)
        d = 1.0 - d  # near = bright
        tvu.save_image(d, f"{out_dir}/depth_{i:03d}.png")
        tvu.save_image(normal * 0.5 + 0.5, f"{out_dir}/normal_{i:03d}.png")
        print(f"saved depth/normal for view {i}")
print("Done")
