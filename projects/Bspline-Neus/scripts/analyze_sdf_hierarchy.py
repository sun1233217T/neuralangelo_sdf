"""Analyze the actual SDF hierarchical field usage for a Bspline-Neus checkpoint.

Prints per-level active control points, region coverage, total degrees of freedom,
final inv_std, and optionally a coarse 3D occupancy/volume histogram.

Run from repo root, e.g.::

    python projects/Bspline-Neus/scripts/analyze_sdf_hierarchy.py \
        --config projects/Bspline-Neus/configs/dtu_scan24_thb_v3_color_mlp_v12_edge_pe.yaml \
        --checkpoint logs/.../epoch_*_iteration_000500000_checkpoint.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from importlib import import_module
from imaginaire.config import Config
from imaginaire.utils.gpu_affinity import set_affinity

Dataset = import_module("projects.Bspline-Neus.data").Dataset
Model = import_module("projects.Bspline-Neus.model").Model


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze SDF hierarchy usage")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--volume_bins", type=int, default=64,
                        help="Spatial bins for the per-level occupancy histogram.")
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


def level_stats(level):
    cc = level.cell_count
    total_cells = cc ** 3
    active_cells = int(level.region.sum().item())
    active_cps = level.num_active
    max_cps = level.grid_size ** 3
    return {
        "grid_size": level.grid_size,
        "cell_count": cc,
        "active_cells": active_cells,
        "region_ratio": active_cells / total_cells,
        "active_control_points": active_cps,
        "max_control_points": max_cps,
        "cp_density": active_cps / max_cps,
    }


def spatial_histogram(level, bins=64):
    """Histogram of active cells in normalized [0,1]^3 space."""
    region = level.region  # (cc, cc, cc) bool
    cc = region.shape[0]
    # Each cell's center in [0, 1].
    idx = torch.nonzero(region, as_tuple=False).float()  # (N, 3)
    if idx.numel() == 0:
        return np.zeros((bins,) * 3, dtype=np.int32)
    centers = (idx + 0.5) / cc
    bin_idx = (centers * bins).long().clamp(0, bins - 1)
    hist = torch.zeros(bins ** 3, dtype=torch.int32, device=region.device)
    flat = bin_idx[:, 0] * bins * bins + bin_idx[:, 1] * bins + bin_idx[:, 2]
    hist.index_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
    return hist.reshape(bins, bins, bins).cpu().numpy()


def main():
    args = parse_args()
    set_affinity(args.local_rank)

    cfg = Config(args.config)
    # Force val subset to None so dataset init is consistent with render_train_set.
    cfg.data.val.subset = None
    if getattr(cfg.data, "num_workers", 4) == 0:
        cfg.data.num_workers = 4
    dataset = Dataset(cfg, is_inference=True)
    del dataset

    model = load_model(cfg, args.checkpoint)
    sdf_wrapper = model.neural_sdf

    print("=" * 70)
    print("SDF hierarchy analysis")
    print(f"checkpoint: {args.checkpoint}")
    print(f"hierarchical enabled: {sdf_wrapper.hierarchical_enabled}")
    print(f"sdf spline degree: {sdf_wrapper.spline_degree}")
    print(f"bounds: {sdf_wrapper.bounds.detach().cpu().tolist()}")
    print(f"inv_std final: {sdf_wrapper.inv_std().item():.6f}")
    print("=" * 70)

    hier = sdf_wrapper.hier_field
    total_active = 0
    total_max = 0
    all_stats = []
    for i, level in enumerate(hier.levels):
        stats = level_stats(level)
        total_active += stats["active_control_points"]
        total_max += stats["max_control_points"]
        hist = spatial_histogram(level, bins=args.volume_bins)
        stats["nonempty_bins"] = int((hist > 0).sum())
        stats["max_bin_cells"] = int(hist.max())
        print(f"\nLevel {i}:")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k:24s}: {v:.4f}")
            else:
                print(f"  {k:24s}: {v}")
        all_stats.append({"level": i, **stats})

    print("\n" + "=" * 70)
    print(f"Total active control points: {total_active:,}")
    print(f"Total max control points (dense sum): {total_max:,}")
    print(f"Overall sparsity ratio: {total_active / total_max:.4%}")

    # Compare to a single dense 256^3 grid.
    dense_256 = 256 ** 3
    print(f"Dense 256^3 grid would be: {dense_256:,}")
    print(f"Active / dense 256^3: {total_active / dense_256:.2%}")

    # Finest-level equivalent cell count and world-space cell size.
    finest = hier.levels[-1]
    bounds = sdf_wrapper.bounds.detach().cpu()
    extent = (bounds[:, 1] - bounds[:, 0]).numpy()
    world_cell_size = extent / finest.cell_count
    print(f"\nFinest level cell count: {finest.cell_count}")
    print(f"World-space cell size: {world_cell_size}")
    print(f"Min cell size: {world_cell_size.min():.6f}")

    # Save JSON for later comparison.
    out_dir = Path(args.checkpoint).parent
    out_path = out_dir / "sdf_hierarchy_stats.json"
    with open(out_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "inv_std": sdf_wrapper.inv_std().item(),
            "total_active_control_points": total_active,
            "total_max_control_points": total_max,
            "sparsity_ratio": total_active / total_max,
            "dense_256_ratio": total_active / dense_256,
            "levels": all_stats,
        }, f, indent=2)
    print(f"\nSaved stats to {out_path}")


if __name__ == "__main__":
    main()
