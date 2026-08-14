"""Phase 2 smoke tests: hierarchical SDF wrapper + eikonal gradient flow.

Run from the repository root:

    python projects/Bspline-Neus/tests/test_thb_wrapper.py

Checks:
1. Hierarchical wrapper initializes (level 0 matches the sphere-init shape)
   and evaluates like the dense wrapper.
2. Eikonal backprop: (||grad|| - 1)^2 populates parameter gradients for both
   the dense and the hierarchical backend (regression test for the
   unconditional-detach bug).
3. maybe_refine follows the schedule, adds a level, and preserves the field.
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

BOUNDS = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]


def _cfg(hierarchical=None):
    return SimpleNamespace(
        grid_size=32,
        sdf_spline_degree=2,
        bounds=BOUNDS,
        sdf_init=0.1,
        sdf_inv_s_init=1.0,
        sdf_init_mode="sphere",
        sdf_init_sphere_radius=0.5,
        sdf_init_sphere_center=[0.0, 0.0, 0.0],
        hierarchical=hierarchical,
    )


def _rand_pts(n, gen):
    return (torch.rand(n, 3, generator=gen) * 2 - 1) * 0.9


def test_hierarchical_init_and_eval():
    print("[test 1] hierarchical wrapper init + evaluation")
    gen = torch.Generator().manual_seed(0)
    hier_cfg = dict(
        enabled=True, base_grid_size=32, max_levels=3,
        refine_iters=[10, 20], refine_sdf_band=0.4, transfer_mode="hb",
    )
    wrapper = BSplineSDFWrapper(_cfg(hierarchical=hier_cfg))
    assert wrapper.hier_field.num_levels == 1
    pts = _rand_pts(1024, gen)
    sdf, feat = wrapper(pts)
    assert sdf.shape == (1024, 1)
    # Compare with the dense wrapper at the same resolution and init.
    dense = BSplineSDFWrapper(_cfg())
    sdf_dense, _ = dense(pts)
    err = (sdf - sdf_dense).abs().max().item()
    print(f"  hier vs dense init max err = {err:.3e}")
    assert err < 1e-5, err
    print("  ok")


def test_eikonal_gradient_flow():
    print("[test 2] eikonal loss backpropagates into the control values")
    gen = torch.Generator().manual_seed(1)
    pts = _rand_pts(256, gen)
    for name, wrapper in (
        ("dense", BSplineSDFWrapper(_cfg())),
        ("hierarchical", BSplineSDFWrapper(_cfg(hierarchical=dict(
            enabled=True, base_grid_size=32, max_levels=2,
            refine_iters=[], refine_sdf_band=0.4, transfer_mode="hb")))),
    ):
        gradients, hessians = wrapper.compute_gradients(pts, training=True)
        loss = ((gradients.norm(dim=-1) - 1.0) ** 2).mean()
        loss.backward()
        grads = [p.grad for p in wrapper.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), \
            f"eikonal produced no parameter gradient for {name}"
        print(f"  {name}: parameter gradient flows, hessian shape {tuple(hessians.shape)}")
    print("  ok")


def test_maybe_refine_schedule_and_preservation():
    print("[test 3] maybe_refine schedule + field preservation")
    gen = torch.Generator().manual_seed(2)
    wrapper = BSplineSDFWrapper(_cfg(hierarchical=dict(
        enabled=True, base_grid_size=32, max_levels=3,
        refine_iters=[10, 20, 30], refine_sdf_band=0.4, transfer_mode="hb")))
    pts = _rand_pts(2048, gen)

    assert wrapper.maybe_refine(5) is None  # not scheduled
    before = wrapper.sdf(pts).detach().clone()
    info = wrapper.maybe_refine(10)
    assert info is not None and info["refined"], info
    assert wrapper.hier_field.num_levels == 2
    after = wrapper.sdf(pts).detach()
    err = (before - after).abs().max().item()
    print(f"  refine info = {info}, field max err = {err:.3e}")
    assert err < 1e-4, err
    info2 = wrapper.maybe_refine(20)
    assert info2 is not None and info2["refined"] and wrapper.hier_field.num_levels == 3
    info3 = wrapper.maybe_refine(30)
    assert info3 is not None and not info3["refined"]  # max_levels reached
    print("  ok")


def test_wrapper_state_dict_roundtrip():
    print("[test 4] state_dict contains structure buffers")
    wrapper = BSplineSDFWrapper(_cfg(hierarchical=dict(
        enabled=True, base_grid_size=32, max_levels=3,
        refine_iters=[10], refine_sdf_band=0.4, transfer_mode="hb")))
    wrapper.maybe_refine(10)
    sd = wrapper.state_dict()
    keys = list(sd.keys())
    assert any("region" in k for k in keys) and any("index_grid" in k for k in keys)
    assert any(k.endswith("values") for k in keys)
    print(f"  {len(keys)} state entries, e.g. {keys[:4]}")
    print("  ok")


def test_wrapper_thb_mode_end_to_end():
    print("[test 5] THB mode through the wrapper: refine preserves sdf + eikonal grads")
    gen = torch.Generator().manual_seed(3)
    wrapper = BSplineSDFWrapper(_cfg(hierarchical=dict(
        enabled=True, base_grid_size=32, max_levels=3,
        refine_iters=[10, 20], refine_sdf_band=0.4, transfer_mode="thb")))
    assert wrapper.hier_field.transfer_mode == "thb"
    pts = _rand_pts(2048, gen)

    before = wrapper.sdf(pts).detach().clone()
    info = wrapper.maybe_refine(10)
    assert info is not None and info["refined"], info
    mid = wrapper.sdf(pts).detach()
    err1 = (before - mid).abs().max().item()
    info2 = wrapper.maybe_refine(20)
    assert info2 is not None and info2["refined"], info2
    after = wrapper.sdf(pts).detach()
    err2 = (before - after).abs().max().item()
    print(f"  refine x2 field max err = {err1:.3e}, {err2:.3e}")
    assert err1 < 1e-4 and err2 < 1e-4, (err1, err2)

    # Eikonal backprop through the truncated evaluation.
    gradients, _ = wrapper.compute_gradients(pts, training=True)
    loss = ((gradients.norm(dim=-1) - 1.0) ** 2).mean()
    loss.backward()
    for idx, lv in enumerate(wrapper.hier_field.levels):
        assert lv.values.grad is not None and lv.values.grad.abs().sum() > 0, idx
    print("  eikonal gradient flows into every level")
    print("  ok")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_hierarchical_init_and_eval()
    test_eikonal_gradient_flow()
    test_maybe_refine_schedule_and_preservation()
    test_wrapper_state_dict_roundtrip()
    test_wrapper_thb_mode_end_to_end()
    print("ALL WRAPPER TESTS PASSED")
