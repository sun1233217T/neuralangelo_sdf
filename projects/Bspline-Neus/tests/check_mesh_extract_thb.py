"""Temporary check: extract_mesh compatibility with the THB hierarchical field."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

modules = importlib.import_module("projects.Bspline-Neus.utils.modules")
mesh_mod = importlib.import_module("projects.neuralangelo.utils.mesh")

BOUNDS = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]


def main():
    cfg = SimpleNamespace(
        grid_size=32, sdf_spline_degree=2, bounds=BOUNDS,
        sdf_init=0.1, sdf_inv_s_init=1.0, sdf_init_mode="sphere",
        sdf_init_sphere_radius=0.5, sdf_init_sphere_center=[0.0, 0.0, 0.0],
        hierarchical=dict(enabled=True, base_grid_size=32, max_levels=3,
                          refine_iters=[10, 20], refine_sdf_band=0.4,
                          transfer_mode="thb"))
    w = modules.BSplineSDFWrapper(cfg).cuda()
    w.maybe_refine(10)
    w.maybe_refine(20)
    print("levels:", w.hier_field.num_levels, "mode:", w.hier_field.transfer_mode)
    sdf_func = lambda x: -w.sdf(x)  # noqa: E731
    mesh = mesh_mod.extract_mesh(
        sdf_func=sdf_func, bounds=[(-1.0, 1.0)] * 3, intv=2.0 / 64, block_res=32)
    import numpy as np
    r = np.linalg.norm(mesh.vertices, axis=1)
    print("vertices:", mesh.vertices.shape, "faces:", mesh.faces.shape)
    print(f"radius min/max: {r.min():.4f} {r.max():.4f} (init sphere radius 0.5)")


if __name__ == "__main__":
    main()
