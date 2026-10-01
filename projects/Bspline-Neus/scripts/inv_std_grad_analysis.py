"""Per-level gradient direction on inv_std (the NeuS sharpness parameter).

Hypothesis: converged (L4) regions push inv_std up (sharper density), while
weakly-converged regions (L0/bubble webs) push it down (prefer diffuse), so
the global s settles at a compromise instead of the optimum.

For each validation ray we find the first zero crossing and the finest level
covering it, then compute d(render loss)/d(raw_sdf_inv_std) restricted to
each level's rays.  Positive gradient means the optimizer would DECREASE
inv_std (more diffuse); negative means INCREASE (sharper).

Usage:
    python projects/Bspline-Neus/scripts/inv_std_grad_analysis.py CONFIG CKPT [--views 0 1 2 3]
"""
import argparse
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from importlib import import_module

from imaginaire.config import Config

Dataset = import_module("projects.Bspline-Neus.data").Dataset
Model = import_module("projects.Bspline-Neus.model").Model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--views", type=int, nargs="+", default=[0, 1, 2, 3])
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
    p_invstd = getattr(model.neural_sdf, "raw_sdf_inv_std_levels", None)
    if p_invstd is None:
        p_invstd = model.neural_sdf.raw_sdf_inv_std
    assert isinstance(p_invstd, torch.Tensor) and p_invstd.requires_grad
    print(f"inv_std = {model.neural_sdf.inv_std().item():.3f}")

    n_levels = hier.num_levels
    grad_sum = torch.zeros(n_levels)
    pix_cnt = torch.zeros(n_levels)
    err_sum = torch.zeros(n_levels)

    for vi in args.views:
        data = ds[vi]
        data = {k: (v[None].cuda() if torch.is_tensor(v) else v) for k, v in data.items()}
        pose, intr = data["pose"], data["intr"]
        gt_flat = data["image"][0].reshape(3, -1).T  # (HW,3)
        H, W = data["image"].shape[2], data["image"].shape[3]
        off = 0
        for center, ray, _ in model.ray_generator(pose, intr, (H, W), full_image=True):
            ray_unit = F.normalize(ray, dim=-1)
            with torch.enable_grad():
                out = model.render_rays(center, ray_unit)
                rgb = out["rgb"][0]  # (R,3) differentiable
            n_obj = out["gradients"].shape[2]
            dists = out["dists"][0, :, :n_obj].detach()
            sdfs = out["sdfs"][0, :, :n_obj].detach()
            # first crossing per ray (detached)
            cross = (sdfs[..., :-1] > 0) & (sdfs[..., 1:] <= 0)
            has = cross.any(dim=1)
            idx_c = cross.float().argmax(dim=1)
            ar = torch.arange(dists.shape[0], device="cuda")
            d0, d1 = dists[ar, idx_c, 0], dists[ar, idx_c + 1, 0]
            s0, s1 = sdfs[ar, idx_c], sdfs[ar, idx_c + 1]
            t = (s0 / (s0 - s1).clamp_min(1e-8)).clamp(0, 1)
            pts = center[0] + ray_unit[0] * (d0 + t * (d1 - d0))[:, None]
            # finest level covering the crossing
            lev = torch.full((pts.shape[0],), -1, dtype=torch.long, device="cuda")
            found = torch.zeros_like(lev, dtype=torch.bool)
            for li in reversed(range(n_levels)):
                lv = hier.levels[li]
                cc = lv.cell_count
                idl = ((pts - lv._lower) / lv.step).floor().long()
                ok = ((idl >= 0) & (idl < cc)).all(dim=-1)
                hit = torch.zeros_like(ok)
                if ok.any():
                    ii = idl[ok]
                    hit[ok] = lv.region[ii[:, 0], ii[:, 1], ii[:, 2]]
                take = hit & ~found
                lev[take] = li
                found |= hit
            # per-ray color error, grouped by level
            R = rgb.shape[0]
            err = (rgb - gt_flat[off:off + R]).abs().mean(dim=-1)  # (R,)
            for li in range(n_levels):
                m = (lev == li) & has
                n = int(m.sum())
                if n == 0:
                    continue
                l = err[m].sum()
                g = torch.autograd.grad(l, p_invstd, retain_graph=True)[0]
                grad_sum[li] += g.item()
                pix_cnt[li] += n
                err_sum[li] += err[m].detach().sum().item()
            off += R
        print(f"view {vi} done", flush=True)

    print(f"\n=== d(render loss)/d(raw_inv_std) by finest level ===")
    print("(positive -> pushes inv_std DOWN (more diffuse); negative -> sharper)")
    print(f"{'level':>6} {'pixels':>9} {'grad_sum':>12} {'grad/pix':>11} {'mean_err':>9}")
    for li in range(n_levels):
        if pix_cnt[li] > 0:
            print(f"{li:>6} {int(pix_cnt[li]):>9} {grad_sum[li]:>12.4f} "
                  f"{grad_sum[li] / pix_cnt[li]:>11.6f} {err_sum[li] / pix_cnt[li]:>9.4f}")
    print(f"total : {grad_sum.sum().item():.4f} over {int(pix_cnt.sum())} pixels")


if __name__ == "__main__":
    main()
