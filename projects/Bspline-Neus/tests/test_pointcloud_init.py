"""Quick regression test for point-cloud SDF initialization."""
from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

modules = importlib.import_module("projects.Bspline-Neus.utils.modules")
BSplineSDFWrapper = modules.BSplineSDFWrapper


def _write_ply(path, points):
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def test_pointcloud_init():
    BOUNDS = [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]]
    with tempfile.TemporaryDirectory() as tmp:
        ply_path = Path(tmp) / "points.ply"
        # Unit sphere-ish point cloud centered at (1, 0, 0) in world coords.
        # After auto_bounds normalization, object should map to roughly [-2, 2]^3.
        n = 200
        theta = np.linspace(0, 2 * np.pi, n)
        phi = np.linspace(0, np.pi, n)
        pts = []
        for t in theta:
            for p in phi:
                x = 1.0 + 0.5 * np.sin(p) * np.cos(t)
                y = 0.5 * np.sin(p) * np.sin(t)
                z = 0.5 * np.cos(p)
                pts.append([x, y, z])
        pts = np.array(pts, dtype=np.float32)
        _write_ply(ply_path, pts)

        cfg = SimpleNamespace(
            grid_size=34,
            sdf_spline_degree=3,
            bounds=BOUNDS,
            sdf_init=0.1,
            sdf_inv_s_init=1.0,
            sdf_init_mode="pointcloud",
            sdf_init_ply_path=str(ply_path),
            sdf_init_ply_quantile_low=0.0,
            sdf_init_ply_quantile_high=1.0,
            sdf_init_ply_padding_ratio=0.0,
            sdf_init_ply_force_cube=True,
            sdf_init_ply_radius_percentile=0.95,
            sdf_init_sphere_center="auto",
        )
        wrapper = BSplineSDFWrapper(cfg)
        grid = wrapper.raw_sdf_grid.detach().cpu()
        print("grid min/max/mean:", grid.min().item(), grid.max().item(), grid.mean().item())
        # Center of the grid (object center) should be negative (inside).
        center_idx = grid.shape[0] // 2
        assert grid[center_idx, center_idx, center_idx] < 0.0, "center should be inside the point cloud sphere"
        # Corners should be positive (outside).
        assert grid[0, 0, 0] > 0.0, "corner should be outside"
        print("pointcloud init test passed")


if __name__ == "__main__":
    test_pointcloud_init()
