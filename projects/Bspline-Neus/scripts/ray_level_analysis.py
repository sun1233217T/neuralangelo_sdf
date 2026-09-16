"""Per-pixel ray analysis: does fine geometry (high image complexity) land
inside the finest refined level?

For each pixel of a validation view:
  1. march the camera ray through the renderer's own stratified samples and
     find the SDF zero-crossing point (sign change + linear interpolation);
  2. query the finest hierarchy level whose region covers that point;
  3. record per-pixel render error and GT high-frequency energy (Sobel).

Outputs a level overlay on the GT image, an error map, and statistics that
answer: do high-complexity pixels whose crossing missed L4 also carry high
residual error?

Usage:
    python projects/Bspline-Neus/scripts/ray_level_analysis.py \
        --config projects/Bspline-Neus/configs/stage_search/D1_anchor_r1_500k.yaml \
        --checkpoint logs/D1_anchor_r1_500k/epoch_20833_iteration_000500000_checkpoint.pt \
        --outdir logs/D1_anchor_r1_500k/ray_level --views 0 24
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as tvu

sys.path.insert(0, ".")
from importlib import import_module

from imaginaire.config import Config

Dataset = import_module("projects.Bspline-Neus.data").Dataset
Model = import_module("projects.Bspline-Neus.model").Model


def finest_level_at(points, hier):
    """Finest level whose region covers each point.  points (N,3) -> (N,) int,
    -1 = outside every refined region."""
    result = torch.zeros(points.shape[0], dtype=torch.long, device=points.device)
    found = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    for li in reversed(range(hier.num_levels)):
        level = hier.levels[li]
        lower = level._lower.detach()  # (3,)
        step = level.step.detach()
        cc = level.cell_count
        idx = ((points - lower) / step).floor().long()  # (N,3)
        inside = ((idx >= 0) & (idx < cc)).all(dim=-1)
        hit = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
        if inside.any():
            ii = idx[inside]
            hit[inside] = level.region[ii[:, 0], ii[:, 1], ii[:, 2]]
        assign = hit & ~found
        result[assign] = li
        found |= hit
    result[~found] = -1
    return result


def zero_crossing(center, ray_unit, dists, sdfs):
    """First + -> - SDF crossing along each ray; linear interpolation.
    dists (R,N,1), sdfs (R,N) -> points (R,3), valid (R,) bool."""
    d = dists[..., 0]  # (R,N)
    s = sdfs  # (R,N)
    # sign change where s[:-1] > 0 >= s[1:]
    cross = (s[:, :-1] > 0) & (s[:, 1:] <= 0)  # (R,N-1)
    has = cross.any(dim=1)
    idx = cross.float().argmax(dim=1)  # first crossing index
    R = d.shape[0]
    ar = torch.arange(R, device=d.device)
    d0, d1 = d[ar, idx], d[ar, idx + 1]
    s0, s1 = s[ar, idx], s[ar, idx + 1]
    t = (s0 / (s0 - s1).clamp_min(1e-8)).clamp(0, 1)
    dc = d0 + t * (d1 - d0)  # (R,)
    pts = center + ray_unit * dc[:, None]
    return pts, has


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--views", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    cfg = Config(args.config)
    cfg.data.val.subset = None
    ds = Dataset(cfg, is_inference=True)
    model = Model(cfg.model, cfg.data).cuda()
    model.progress = 1.0
    ckpt = torch.load(args.checkpoint, map_location=lambda s, l: s)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    hier = model.neural_sdf.hier_field

    os.makedirs(args.outdir, exist_ok=True)
    for vi in args.views:
        data = ds[vi]
        data = {k: (v[None].cuda() if torch.is_tensor(v) else v) for k, v in data.items()}
        pose, intr = data["pose"], data["intr"]
        gt = data["image"][0]  # (3,H,W)
        H, W = gt.shape[1], gt.shape[2]

        levels_all = torch.full((H * W,), -2, dtype=torch.long)  # -2 = not processed
        err_all = torch.zeros(H * W)
        gt_flat = gt.reshape(3, -1).T  # (H*W, 3)
        off = 0
        with torch.no_grad():
            for center, ray, _ in model.ray_generator(pose, intr, (H, W), full_image=True):
                ray_unit = F.normalize(ray, dim=-1)
                out = model.render_rays(center, ray_unit)
                n_obj = out["gradients"].shape[2]
                dists = out["dists"][:, :, :n_obj]
                sdfs = out["sdfs"][:, :, :n_obj]
                pts, has = zero_crossing(center[0], ray_unit[0], dists[0], sdfs[0])
                lev = torch.full((pts.shape[0],), -1, dtype=torch.long, device=pts.device)
                if has.any():
                    lev[has] = finest_level_at(pts[has], hier)
                R = pts.shape[0]
                levels_all[off:off + R] = lev.cpu()
                err_all[off:off + R] = (out["rgb"][0] - gt_flat[off:off + R].cuda()).abs().mean(dim=-1).cpu()
                off += R
        levels_map = levels_all.reshape(H, W)
        err_map = err_all.reshape(H, W)

        # GT high-frequency energy: Sobel on luminance.
        lum = gt.mean(dim=0, keepdim=True)[None]  # (1,1,H,W)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=gt.device)[None, None]
        ky = kx.transpose(-1, -2)
        gx = F.conv2d(lum, kx, padding=1)
        gy = F.conv2d(lum, ky, padding=1)
        sobel = (gx.pow(2) + gy.pow(2)).sqrt()[0, 0]  # (H,W)
        # local smoothing to ~tile scale
        complex_map = F.avg_pool2d(sobel[None, None], 16, stride=1, padding=8)[0, 0, :H, :W].cpu()

        # ---- stats ----
        fg = levels_map >= 0  # hit some refined level
        bg = levels_map == -1  # crossing but outside all refined regions
        nc = levels_map == -2 | ((levels_map < 0) & ~bg)
        print(f"\n=== view {vi} ===")
        print(f"pixels: {H*W}, with crossing: {(fg | bg).sum().item()}, "
              f"no crossing (background): {(levels_map == -1).sum().item()}")
        n_levels = hier.num_levels
        print(f"{'level':>6} {'pixels':>8} {'share':>7} {'mean_err':>9} {'mean_cplx':>10}")
        for li in range(n_levels):
            m = levels_map == li
            if m.any():
                print(f"{li:>6} {m.sum().item():>8} {m.float().mean().item():>7.2%} "
                      f"{err_map[m].mean().item():>9.4f} {complex_map[m].mean().item():>10.4f}")
        m = bg
        if m.any():
            print(f"{'none':>6} {m.sum().item():>8} {m.float().mean().item():>7.2%} "
                  f"{err_map[m].mean().item():>9.4f} {complex_map[m].mean().item():>10.4f}")
        # Key question: high-complexity pixels split by finest level.
        hi = complex_map > torch.quantile(complex_map[fg | bg], 0.8)
        print(f"\nhigh-complexity pixels (top 20% sobel): {hi.sum().item()}")
        for li in list(range(n_levels)) + [-1]:
            m = hi & (levels_map == li)
            if m.any():
                print(f"  L{li if li >= 0 else 'X'}: {m.sum().item():>8} pixels, "
                      f"mean_err={err_map[m].mean().item():.4f}")
        # correlation error vs complexity within each level
        for li in list(range(n_levels)) + [-1]:
            m = (levels_map == li) & (fg | bg)
            if m.sum() > 100:
                e, c = err_map[m], complex_map[m]
                ec = ((e - e.mean()) * (c - c.mean())).mean() / (e.std() * c.std() + 1e-8)
                print(f"  corr(err, complexity) @ L{li if li>=0 else 'X'}: {ec.item():.3f}")

        # ---- maps ----
        # level overlay: colorize level on top of GT
        colors = torch.tensor([[0, 0, 0], [0, 0.4, 1], [0, 0.9, 0.4], [1, 0.85, 0], [1, 0.3, 0], [1, 0, 0.8]])
        lv = levels_map.clone()
        lv[lv < 0] = 0
        overlay = gt.cpu() * 0.5 + colors[lv].permute(2, 0, 1) * 0.5
        overlay[:, levels_map < 0] = gt.cpu()[:, levels_map < 0]  # background: plain GT
        tvu.save_image(overlay.clamp(0, 1), f"{args.outdir}/view{vi:03d}_level_overlay.png")
        tvu.save_image(gt, f"{args.outdir}/view{vi:03d}_gt.png")
        tvu.save_image(err_map[None].clamp(0, 0.2) / 0.2, f"{args.outdir}/view{vi:03d}_err.png")
        cmax = complex_map.max()
        tvu.save_image((complex_map / cmax)[None], f"{args.outdir}/view{vi:03d}_complexity.png")
        np.savez(f"{args.outdir}/view{vi:03d}_maps.npz",
                 levels=levels_map.numpy(), err=err_map.numpy(),
                 complexity=complex_map.numpy())
        print(f"saved maps to {args.outdir}/view{vi:03d}_*.png")


if __name__ == "__main__":
    main()
