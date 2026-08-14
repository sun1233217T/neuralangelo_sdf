"""Phase 3 smoke tests: hierarchical SH color wrapper.

Run from the repository root:

    python projects/Bspline-Neus/tests/test_thb_color.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

modules = importlib.import_module("projects.Bspline-Neus.utils.modules")

BOUNDS = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]

HIER = dict(
    enabled=True, color_enabled=True, base_grid_size=32, max_levels=3,
    refine_iters=[10, 20], refine_sdf_band=0.4, transfer_mode="hb",
)


def _cfg(hierarchical=None):
    return SimpleNamespace(
        grid_size=32,
        sdf_spline_degree=2,
        color_spline_degree=2,
        color_sh_degree=2,
        bounds=BOUNDS,
        sdf_init=0.1,
        sdf_inv_s_init=1.0,
        sdf_init_mode="sphere",
        sdf_init_sphere_radius=0.5,
        sdf_init_sphere_center=[0.0, 0.0, 0.0],
        color_init=[0.2, 0.5, 0.8],
        color_lr=0.04,
        color_sh_rest_lr_multiplier=0.025,
        hierarchical=hierarchical,
    )


def _rand_pts(n, gen):
    return (torch.rand(n, 3, generator=gen) * 2 - 1) * 0.9


def test_color_init_constant():
    print("[test 1] hierarchical color init renders constant color")
    gen = torch.Generator().manual_seed(0)
    rgb_wrapper = modules.BSplineRGBWrapper(_cfg(hierarchical=HIER))
    pts = _rand_pts(512, gen)
    dirs = torch.randn(512, 3, generator=gen)
    rgb = rgb_wrapper(pts, None, dirs, None, None)
    assert rgb.shape == (512, 3)
    target = torch.tensor([0.2, 0.5, 0.8])
    err = (rgb - target).abs().max().item()
    print(f"  max deviation from color_init = {err:.3e}")
    assert err < 1e-5, err
    print("  ok")


def test_color_matches_dense():
    print("[test 2] hierarchical color matches dense wrapper after jitters")
    gen = torch.Generator().manual_seed(1)
    hier = modules.BSplineRGBWrapper(_cfg(hierarchical=HIER))
    dense = modules.BSplineRGBWrapper(_cfg())
    # Copy identical random SH coefficients into both backends.
    G = 32
    sh = hier.color_sh_basis_dim
    coeffs = torch.randn(G, G, G, 3, sh, generator=gen) * 0.1
    dense.raw_color_grid.data = coeffs.clone()
    hier.hier_field.levels[0].values.data = coeffs.reshape(-1, 3 * sh).clone()
    pts = _rand_pts(2048, gen)
    dirs = torch.randn(2048, 3, generator=gen)
    rgb_h = hier(pts, None, dirs, None, None)
    rgb_d = dense(pts, None, dirs, None, None)
    err = (rgb_h - rgb_d).abs().max().item()
    print(f"  hier vs dense max err = {err:.3e}")
    assert err < 1e-5, err
    print("  ok")


def test_color_aligned_refinement():
    print("[test 3] color refines with SDF marks and preserves the field")
    gen = torch.Generator().manual_seed(2)
    sdf = modules.BSplineSDFWrapper(_cfg(hierarchical=HIER))
    rgb = modules.BSplineRGBWrapper(_cfg(hierarchical=HIER))
    sh = rgb.color_sh_basis_dim
    rgb.hier_field.levels[0].values.data = torch.randn(
        rgb.hier_field.levels[0].values.shape, generator=gen
    ) * 0.1

    pts = _rand_pts(4096, gen)
    dirs = torch.randn(4096, 3, generator=gen)
    before = rgb(pts, None, dirs, None, None).detach().clone()

    sdf_info = sdf.maybe_refine(10)
    rgb_info = rgb.maybe_refine(10, sdf_wrapper=sdf)
    assert sdf_info["refined"] and rgb_info["refined"]
    assert rgb.hier_field.num_levels == 2
    # Regions stay aligned with the SDF hierarchy.
    assert torch.equal(
        rgb.hier_field.levels[1].region, sdf.hier_field.levels[1].region
    )
    after = rgb(pts, None, dirs, None, None).detach()
    err = (before - after).abs().max().item()
    print(f"  rgb refine info = {rgb_info}, field max err = {err:.3e}")
    assert err < 1e-4, err
    # SH hooks must still be active on the new parameters.
    assert len(rgb._sh_hook_handles) == rgb.hier_field.num_levels
    print("  ok")


def test_color_gradient_flow():
    print("[test 4] gradients flow into hierarchical color values")
    gen = torch.Generator().manual_seed(3)
    rgb = modules.BSplineRGBWrapper(_cfg(hierarchical=HIER))
    pts = _rand_pts(256, gen)
    dirs = torch.randn(256, 3, generator=gen)
    out = rgb(pts, None, dirs, None, None)
    loss = ((out - 0.5) ** 2).mean()
    loss.backward()
    grad = rgb.hier_field.levels[0].values.grad
    assert grad is not None and grad.abs().sum() > 0
    print(f"  grad norm = {grad.norm().item():.4e}")
    print("  ok")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_color_init_constant()
    test_color_matches_dense()
    test_color_aligned_refinement()
    test_color_gradient_flow()
    print("ALL COLOR WRAPPER TESTS PASSED")
