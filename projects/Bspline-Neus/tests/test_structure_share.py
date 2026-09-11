"""Test SDF/color hierarchical structure sharing.

Run from the repository root::

    python projects/Bspline-Neus/tests/test_structure_share.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

modules = importlib.import_module("projects.Bspline-Neus.utils.modules")
BSplineSDFWrapper = modules.BSplineSDFWrapper
BSplineRGBWrapper = modules.BSplineRGBWrapper

BOUNDS = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]

HIER = dict(
    enabled=True, color_enabled=True, base_grid_size=32, max_levels=3,
    refine_iters=[10, 20], refine_sdf_band=0.4, transfer_mode="hb",
    share_structure=True,
)


def _cfg():
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
        hierarchical=HIER,
    )


def test_structure_is_shared():
    print("[test 1] color hierarchy references the SDF owner's structures")
    sdf = BSplineSDFWrapper(_cfg())
    rgb = BSplineRGBWrapper(_cfg(), sdf_wrapper=sdf)
    assert rgb.hier_field.structure_owner is sdf.hier_field
    for lv_sdf, lv_rgb in zip(sdf.hier_field.levels, rgb.hier_field.levels):
        assert lv_rgb._structure_ref is lv_sdf._structure_ref
    print("  ok")


def test_shared_refine_preserves_color():
    print("[test 2] shared refinement preserves the color field")
    sdf = BSplineSDFWrapper(_cfg())
    rgb = BSplineRGBWrapper(_cfg(), sdf_wrapper=sdf)

    gen = torch.Generator().manual_seed(0)
    pts = (torch.rand(4096, 3, generator=gen) * 2 - 1) * 0.9
    dirs = torch.randn(4096, 3, generator=gen)
    before = rgb(pts, None, dirs, None, None).detach().clone()

    info1 = sdf.maybe_refine(10)
    rgb_info1 = rgb.maybe_refine(10, sdf_wrapper=sdf, sdf_refine_info=info1)
    assert rgb_info1["refined"]

    info2 = sdf.maybe_refine(20)
    rgb_info2 = rgb.maybe_refine(20, sdf_wrapper=sdf, sdf_refine_info=info2)
    assert rgb_info2["refined"]

    after = rgb(pts, None, dirs, None, None).detach()
    err = (before - after).abs().max().item()
    print(f"  field max err = {err:.3e}")
    assert err < 1e-4, err
    print("  ok")


def test_shared_state_dict_no_duplicate_structures():
    print("[test 3] shared color state dict does not duplicate structure buffers")
    sdf = BSplineSDFWrapper(_cfg())
    rgb = BSplineRGBWrapper(_cfg(), sdf_wrapper=sdf)
    rgb_sd = rgb.state_dict()
    rgb_struct = [k for k in rgb_sd if "_structures" in k]
    print(f"  rgb _structures keys: {len(rgb_struct)}")
    assert len(rgb_struct) == 0
    print("  ok")


def test_shared_checkpoint_roundtrip():
    print("[test 4] shared structure checkpoint save/load roundtrip")
    sdf = BSplineSDFWrapper(_cfg())
    rgb = BSplineRGBWrapper(_cfg(), sdf_wrapper=sdf)

    info1 = sdf.maybe_refine(10)
    rgb.maybe_refine(10, sdf_wrapper=sdf, sdf_refine_info=info1)

    prefix = "hier_field."
    sdf_sd = sdf.state_dict()
    rgb_sd = rgb.state_dict()

    sdf2 = BSplineSDFWrapper(_cfg())
    rgb2 = BSplineRGBWrapper(_cfg(), sdf_wrapper=sdf2)
    sdf2.hier_field.restore_levels_from_state_dict(sdf_sd, prefix)
    rgb2.hier_field.restore_levels_from_state_dict(rgb_sd, prefix)
    sdf2.load_state_dict(sdf_sd)
    rgb2.load_state_dict(rgb_sd)

    gen = torch.Generator().manual_seed(1)
    pts = (torch.rand(4096, 3, generator=gen) * 2 - 1) * 0.9
    dirs = torch.randn(4096, 3, generator=gen)
    s1, c1 = sdf.sdf(pts).detach(), rgb(pts, None, dirs, None, None).detach()
    s2, c2 = sdf2.sdf(pts).detach(), rgb2(pts, None, dirs, None, None).detach()
    print(f"  sdf err = {(s1 - s2).abs().max().item():.3e}, "
          f"rgb err = {(c1 - c2).abs().max().item():.3e}")
    assert torch.equal(s1, s2)
    assert torch.equal(c1, c2)
    print("  ok")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_structure_is_shared()
    test_shared_refine_preserves_color()
    test_shared_state_dict_no_duplicate_structures()
    test_shared_checkpoint_roundtrip()
    print("ALL STRUCTURE-SHARE TESTS PASSED")
