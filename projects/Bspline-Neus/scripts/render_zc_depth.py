"""Render zero-crossing depth (hard surface) vs expectation depth.

For each ray, marches the renderer's own stratified samples and locates the
first + -> - SDF sign change; depth is the linearly interpolated crossing
distance.  Rays without a crossing are left black.  This bypasses the
compositing expectation, which blurs depth over the weight distribution.

Usage:
    python projects/Bspline-Neus/scripts/render_zc_depth.py CONFIG CKPT OUTDIR [views]
"""
import sys, os
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
from importlib import import_module
from imaginaire.config import Config
import torchvision.utils as tvu

Dataset = import_module("projects.Bspline-Neus.data").Dataset
Model = import_module("projects.Bspline-Neus.model").Model

cfg_path, ckpt_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
indices = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0]

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
        pose, intr = data["pose"], data["intr"]
        H, W = data["image"].shape[2], data["image"].shape[3]
        zc = torch.zeros(H * W, device="cuda")
        valid = torch.zeros(H * W, dtype=torch.bool, device="cuda")
        exp_d = torch.zeros(H * W, device="cuda")
        off = 0
        for center, ray, _ in model.ray_generator(pose, intr, (H, W), full_image=True):
            ray_unit = F.normalize(ray, dim=-1)
            out = model.render_rays(center, ray_unit)
            n_obj = out["gradients"].shape[2]
            d = out["dists"][0, :, :n_obj, 0]  # (R,No)
            s = out["sdfs"][0, :, :n_obj]  # (R,No)
            # first + -> - crossing
            cross = (s[:, :-1] > 0) & (s[:, 1:] <= 0)
            has = cross.any(dim=1)
            idx = cross.float().argmax(dim=1)
            ar = torch.arange(d.shape[0], device="cuda")
            d0, d1 = d[ar, idx], d[ar, idx + 1]
            s0, s1 = s[ar, idx], s[ar, idx + 1]
            t = (s0 / (s0 - s1).clamp_min(1e-8)).clamp(0, 1)
            dc = (d0 + t * (d1 - d0)) / ray[0].norm(dim=-1)
            R = d.shape[0]
            zc[off:off + R] = torch.where(has, dc, torch.zeros_like(dc))
            valid[off:off + R] = has
            w = out["weights"][0, :, :n_obj, 0]  # (R,No)
            exp_d[off:off + R] = (d * w).sum(-1) / ray[0].norm(dim=-1)
            off += R
        zc = zc.reshape(H, W)
        valid = valid.reshape(H, W)
        exp_d = exp_d.reshape(H, W)
        # normalize both on the valid region of the zc depth for a fair scale
        lo, hi = zc[valid].min(), zc[valid].max()
        zc_n = torch.zeros_like(zc)
        zc_n[valid] = 1.0 - (zc[valid] - lo) / (hi - lo + 1e-8)  # near = bright
        exp_n = 1.0 - (exp_d - lo) / (hi - lo + 1e-8)
        exp_n = exp_n.clamp(0, 1)
        tvu.save_image(zc_n[None], f"{out_dir}/zc_depth_{i:03d}.png")
        tvu.save_image(exp_n[None], f"{out_dir}/exp_depth_{i:03d}.png")
        print(f"view {i}: zc coverage {valid.float().mean().item():.1%}")
print("Done")
