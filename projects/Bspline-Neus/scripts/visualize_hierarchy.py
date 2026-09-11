"""Visualize hierarchical refinement regions vs the SDF zero level set.

For a trained checkpoint, renders axis-aligned slices through the object
(estimated as the centroid of the finest level's active cells) and plots:

  1. the SDF slice with its zero contour,
  2. the "responsible level" map (finest level whose region covers each cell),
  3. each level's region mask,
  4. a histogram of |SDF| at the finest level's active cell centers
     (does the refined band actually hug the surface?).

Run from repo root::

    python projects/Bspline-Neus/scripts/visualize_hierarchy.py \
        --config projects/Bspline-Neus/configs/dtu_scan24_E22b_optimized_500k.yaml \
        --checkpoint logs/E22b_optimized_500k/epoch_20833_iteration_000500000_checkpoint.pt \
        --outdir logs/E22b_optimized_500k/diag
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
    parser = argparse.ArgumentParser(description="Visualize hierarchy refinement regions")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", default=None, help="Defaults to <checkpoint_dir>/diag")
    parser.add_argument("--res", type=int, default=512, help="Slice resolution.")
    parser.add_argument("--local_rank", type=int, default=os.getenv("LOCAL_RANK", 0))
    return parser.parse_args()


def load_model(cfg, checkpoint_path):
    model = Model(cfg.model, cfg.data)
    model = model.cuda()
    model.progress = 1.0
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    state_dict = checkpoint["model"]
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


@torch.no_grad()
def eval_sdf_slice(model, axis, coord, res, bounds, chunk=65536):
    """Evaluate the SDF on a res x res grid of the plane axis=coord."""
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
        out.append(model.neural_sdf.sdf(pts[i : i + chunk]).reshape(-1).cpu())
    return torch.cat(out).reshape(res, res), u, v


def region_membership(level, axis, coord, u, v):
    """Boolean (res, res) mask: does this level's region cover each slice cell?"""
    free = [d for d in range(3) if d != axis]
    region = level.region  # (cc, cc, cc) bool, cuda
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
    # Per-pixel index triplets.  free[0] varies along u (rows), free[1] along
    # v (columns); the sliced axis is constant so either expansion works.
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
    dataset = Dataset(cfg, is_inference=True)
    del dataset

    model = load_model(cfg, args.checkpoint)
    hier = model.neural_sdf.hier_field
    bounds = model.neural_sdf.bounds.detach().cpu()

    outdir = Path(args.outdir) if args.outdir else Path(args.checkpoint).parent / "diag"
    outdir.mkdir(parents=True, exist_ok=True)

    # Object centroid from the finest level's active cell centers.
    finest = hier.levels[-1]
    act = torch.nonzero(finest.region, as_tuple=False).float()
    centers = finest._lower.detach().cpu() + (act.cpu() + 0.5) * finest.step.detach().cpu()
    centroid = centers.mean(dim=0)
    print(f"finest-level active cells: {act.shape[0]}, centroid: {centroid.tolist()}")

    # |SDF| at finest-level active cell centers: does the band hug the surface?
    with torch.no_grad():
        sdf_at_cells = []
        pts = centers.cuda()
        for i in range(0, pts.shape[0], 262144):
            sdf_at_cells.append(model.neural_sdf.sdf(pts[i : i + 262144]).reshape(-1).cpu())
        sdf_at_cells = torch.cat(sdf_at_cells)
    cell_world = float(finest.step.detach().cpu().max())
    abs_sdf_cells = sdf_at_cells.abs() / cell_world  # in units of finest cells
    print(f"|SDF| at finest active cell centers (units of finest cell size {cell_world:.5f}):")
    for q in (0.1, 0.5, 0.9, 0.99):
        print(f"  q{int(q*100):02d}: {abs_sdf_cells.quantile(q):.2f} cells")
    frac_far = (abs_sdf_cells > 4.0).float().mean().item()
    print(f"  fraction > 4 cells away from surface: {frac_far:.2%}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(abs_sdf_cells.numpy(), bins=200, range=(0, 20))
    ax.set_xlabel("|SDF| / finest cell size")
    ax.set_ylabel("active cell count")
    ax.set_title("Finest-level band tightness")
    fig.tight_layout()
    fig.savefig(outdir / "band_tightness_hist.png", dpi=120)
    plt.close(fig)

    axis_names = "xyz"
    for axis in range(3):
        coord = float(centroid[axis])
        sdf, u, v = eval_sdf_slice(model, axis, coord, args.res, bounds)
        free = [d for d in range(3) if d != axis]
        extent = [float(u[0]), float(u[-1]), float(v[0]), float(v[-1])]

        # Finest responsible level per pixel.
        membership = []
        for level in hier.levels:
            membership.append(region_membership(level, axis, coord, u, v))
        finest_map = torch.zeros(args.res, args.res, dtype=torch.long)
        for l, m in enumerate(membership):
            finest_map[m] = l + 1  # 0 = outside all levels

        n_levels = len(hier.levels)
        n_panels = 2 + n_levels
        n_cols = 4
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows))
        axes = axes.ravel()

        im = axes[0].imshow(sdf.T.numpy(), origin="lower", extent=extent, cmap="RdBu",
                            vmin=-0.2, vmax=0.2)
        axes[0].contour(sdf.T.numpy(), levels=[0.0], origin="lower", extent=extent,
                        colors="k", linewidths=1.0)
        axes[0].set_title(f"SDF slice {axis_names[axis]}={coord:.3f}")
        fig.colorbar(im, ax=axes[0], fraction=0.046)

        im = axes[1].imshow(finest_map.T.numpy(), origin="lower", extent=extent,
                            cmap="viridis", vmin=0, vmax=n_levels)
        axes[1].contour(sdf.T.numpy(), levels=[0.0], origin="lower", extent=extent,
                        colors="r", linewidths=1.0)
        axes[1].set_title("finest responsible level (0=none)")
        fig.colorbar(im, ax=axes[1], fraction=0.046, ticks=range(n_levels + 1))

        for l, (level, m) in enumerate(zip(hier.levels, membership)):
            ax = axes[2 + l]
            ax.imshow(m.T.numpy(), origin="lower", extent=extent, cmap="gray")
            ax.contour(sdf.T.numpy(), levels=[0.0], origin="lower", extent=extent,
                       colors="r", linewidths=1.0)
            ratio = level.region.float().mean().item()
            ax.set_title(f"L{l} region (cc={level.cell_count}, {ratio:.2%} active)")
        for ax in axes[n_panels:]:
            ax.axis("off")

        fig.tight_layout()
        out = outdir / f"hierarchy_slice_{axis_names[axis]}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
