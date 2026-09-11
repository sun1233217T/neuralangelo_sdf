"""Generate a series of N axis-aligned slices through the trained hierarchy.

Each slice shows the SDF field (with zero contour) and the finest-responsible-
level map side by side.

Usage:
    python projects/Bspline-Neus/scripts/slice_series.py \
        --config projects/Bspline-Neus/configs/dtu_scan24_E22b_optimized_500k.yaml \
        --checkpoint logs/E22b_optimized_500k/epoch_20833_iteration_000500000_checkpoint.pt \
        --axis 0 --num_slices 128 --outdir logs/E22b_optimized_500k/slices
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from importlib import import_module
from imaginaire.config import Config
from imaginaire.utils.gpu_affinity import set_affinity

Dataset = import_module("projects.Bspline-Neus.data").Dataset
Model = import_module("projects.Bspline-Neus.model").Model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--axis", type=int, default=0, choices=[0, 1, 2],
                   help="Slice axis: 0=x (left-to-right), 1=y, 2=z")
    p.add_argument("--num_slices", type=int, default=128)
    p.add_argument("--res", type=int, default=384, help="Slice resolution")
    p.add_argument("--outdir", required=True)
    p.add_argument("--local_rank", type=int, default=os.getenv("LOCAL_RANK", 0))
    return p.parse_args()


def load_model(cfg, ckpt_path):
    model = Model(cfg.model, cfg.data).cuda()
    model.progress = 1.0
    ckpt = torch.load(ckpt_path, map_location=lambda s, l: s)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


@torch.no_grad()
def eval_sdf_slice(model, axis, coord, res, bounds, chunk=65536):
    free = [d for d in range(3) if d != axis]
    u = torch.linspace(bounds[free[0], 0], bounds[free[0], 1], res)
    v = torch.linspace(bounds[free[1], 0], bounds[free[1], 1], res)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    pts = torch.zeros(res, res, 3)
    pts[..., axis] = coord
    pts[..., free[0]] = uu
    pts[..., free[1]] = vv
    pts = pts.reshape(-1, 3).cuda()
    out = []
    for i in range(0, pts.shape[0], chunk):
        out.append(model.neural_sdf.sdf(pts[i:i+chunk]).reshape(-1).cpu())
    return torch.cat(out).reshape(res, res), u, v


def region_membership(level, axis, coord, u, v):
    free = [d for d in range(3) if d != axis]
    region = level.region
    cc = level.cell_count
    lower = level._lower.detach().cpu()
    step = level.step.detach().cpu()
    iu = ((u - lower[free[0]]) / step[free[0]]).floor().long().clamp(0, cc - 1)
    iv = ((v - lower[free[1]]) / step[free[1]]).floor().long().clamp(0, cc - 1)
    ia = int(np.clip(int((coord - lower[axis]) / step[axis]), 0, cc - 1))
    idx = [None, None, None]
    idx[axis] = torch.full_like(iu, ia)
    idx[free[0]] = iu
    idx[free[1]] = iv
    I = [None, None, None]
    I[axis] = idx[axis][:, None].expand(len(u), len(v))
    I[free[0]] = iu[:, None].expand(len(u), len(v))
    I[free[1]] = iv[None, :].expand(len(u), len(v))
    return region[I[0].cuda(), I[1].cuda(), I[2].cuda()].cpu()


def main():
    args = parse_args()
    set_affinity(args.local_rank)
    cfg = Config(args.config)
    cfg.data.val.subset = None
    if getattr(cfg.data, "num_workers", 4) == 0:
        cfg.data.num_workers = 4
    ds = Dataset(cfg, is_inference=True)
    del ds

    model = load_model(cfg, args.checkpoint)
    hier = model.neural_sdf.hier_field
    bounds = model.neural_sdf.bounds.detach().cpu()
    n_levels = len(hier.levels)
    axis_names = "xyz"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Slice positions: evenly spaced along the axis within bounds
    lo, hi = float(bounds[args.axis, 0]), float(bounds[args.axis, 1])
    coords = np.linspace(lo + 0.01, hi - 0.01, args.num_slices)

    # Pre-compute a colormap for levels
    level_cmap = plt.cm.viridis

    print(f"Generating {args.num_slices} slices along {axis_names[args.axis]}-axis, "
          f"res={args.res}, levels={n_levels}")

    for si, coord in enumerate(coords):
        sdf, u, v = eval_sdf_slice(model, args.axis, coord, args.res, bounds)
        free = [d for d in range(3) if d != args.axis]
        extent = [float(u[0]), float(u[-1]), float(v[0]), float(v[-1])]

        # Finest responsible level per pixel
        finest_map = torch.zeros(args.res, args.res, dtype=torch.long)
        for l, level in enumerate(hier.levels):
            m = region_membership(level, args.axis, coord, u, v)
            finest_map[m] = l + 1  # 0 = outside

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # SDF slice
        im0 = axes[0].imshow(sdf.T.numpy(), origin="lower", extent=extent,
                             cmap="RdBu", vmin=-0.2, vmax=0.2)
        axes[0].contour(sdf.T.numpy(), levels=[0.0], origin="lower", extent=extent,
                        colors="k", linewidths=0.8)
        axes[0].set_title(f"SDF  {axis_names[args.axis]}={coord:.3f}")
        fig.colorbar(im0, ax=axes[0], fraction=0.046)

        # Level map
        im1 = axes[1].imshow(finest_map.T.numpy(), origin="lower", extent=extent,
                             cmap="viridis", vmin=0, vmax=n_levels)
        axes[1].contour(sdf.T.numpy(), levels=[0.0], origin="lower", extent=extent,
                        colors="r", linewidths=0.8)
        axes[1].set_title(f"Finest level  {axis_names[args.axis]}={coord:.3f}")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, ticks=range(n_levels + 1))

        fig.tight_layout()
        fig.savefig(outdir / f"slice_{si:04d}.png", dpi=100)
        plt.close(fig)

        if (si + 1) % 16 == 0:
            print(f"  {si+1}/{args.num_slices} slices done")

    print(f"Saved {args.num_slices} slices to {outdir}")


if __name__ == "__main__":
    main()
