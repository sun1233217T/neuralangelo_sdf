"""Phase 4 tests for THB (truncated hierarchical B-spline) evaluation.

Run from the repository root:

    python projects/Bspline-Neus/tests/test_thb_truncation.py

Test 1: single-level ``evaluate_thb`` matches dense ``BSplineField``.
Test 2: THB refinement preserves the field — with ``transfer_mode="thb"``
        (full prolongation) and truncated evaluation, marking arbitrary
        cells and refining must leave ``evaluate`` unchanged everywhere,
        over several rounds and for degree 2 and 3.
Test 3: gradients flow through ``evaluate_thb`` into every level's values,
        and ``evaluate_gradient`` is preserved under THB refinement.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

field_mod = importlib.import_module("projects.Bspline-Neus.bspline_field.field")
hier_mod = importlib.import_module("projects.Bspline-Neus.bspline_field.hierarchical")

BSplineField = field_mod.BSplineField
HierarchicalBSplineField = hier_mod.HierarchicalBSplineField

BOUNDS = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]


def _random_points(n, gen):
    # Keep clear of the exact boundary where both implementations clamp.
    return (torch.rand(n, 3, generator=gen) * 2.0 - 1.0) * 0.98


def _box_mark_fn(cc, round_idx):
    lo = cc // 4 + round_idx
    hi = 3 * cc // 4

    def mark_fn(level_index, corner_points):
        m = torch.zeros(cc, cc, cc, dtype=torch.bool)
        m[lo:hi, lo:hi, lo:hi] = True
        return m

    return mark_fn


def test_single_level_thb_matches_dense():
    print("[test 1] single-level evaluate_thb matches dense BSplineField")
    gen = torch.Generator().manual_seed(11)
    for degree in (2, 3):
        G = 16
        coeffs = torch.randn(G, G, G, generator=gen)
        dense = BSplineField(G, coeffs, bounds=BOUNDS, spline_degree=degree, trainable=False)
        hier = HierarchicalBSplineField(
            G, degree, BOUNDS, channels=1, max_levels=1, transfer_mode="thb"
        )
        hier.levels[0].values.data = coeffs.reshape(-1, 1).clone()
        pts = _random_points(4096, gen)
        err = (dense.evaluate(pts) - hier.evaluate_thb(pts).squeeze(-1)).abs().max().item()
        print(f"  degree={degree}: max abs err = {err:.3e}")
        assert err < 1e-5, err
    print("  ok")


def test_thb_refinement_preserves_field():
    print("[test 2] THB refinement preserves the field (degree 2 and 3)")
    gen = torch.Generator().manual_seed(12)
    for degree in (2, 3):
        G = 16
        coeffs = torch.randn(G, G, G, generator=gen)
        hier = HierarchicalBSplineField(
            G, degree, BOUNDS, channels=1, max_levels=4, transfer_mode="thb"
        )
        hier.levels[0].values.data = coeffs.reshape(-1, 1).clone()

        pts = _random_points(8192, gen)
        before = hier.evaluate(pts).detach().clone()

        for round_idx in range(3):
            cc = hier.levels[-1].cell_count
            info = hier.refine(_box_mark_fn(cc, round_idx))
            after = hier.evaluate(pts).detach()
            err = (before - after).abs().max().item()
            print(f"  degree={degree} round {round_idx}: info={info}, max abs err = {err:.3e}")
            assert info["refined"], info
            assert err < 1e-4, err
    print("  ok")


def test_thb_gradient_flow_and_consistency():
    print("[test 3] autograd through evaluate_thb + gradient preserved under refine")
    gen = torch.Generator().manual_seed(13)
    degree, G = 2, 16
    coeffs = torch.randn(G, G, G, generator=gen)
    hier = HierarchicalBSplineField(
        G, degree, BOUNDS, channels=1, max_levels=4, transfer_mode="thb"
    )
    hier.levels[0].values.data = coeffs.reshape(-1, 1).clone()

    pts = _random_points(2048, gen)
    grad_before = hier.evaluate_gradient(pts).detach().clone()

    cc = hier.levels[-1].cell_count
    hier.refine(_box_mark_fn(cc, 0))

    # Values of every level receive gradients through the truncated evaluation.
    for lv in hier.levels:
        lv.values.grad = None
    out = hier.evaluate(pts)
    out.square().sum().backward()
    for idx, lv in enumerate(hier.levels):
        assert lv.values.grad is not None, f"level {idx} got no gradient"
        gnorm = lv.values.grad.abs().max().item()
        print(f"  level {idx}: |grad| max = {gnorm:.3e}")
        assert gnorm > 0, f"level {idx} gradient is zero"

    # Field preserved -> spatial gradient preserved (autograd path intact).
    grad_after = hier.evaluate_gradient(pts).detach()
    gerr = (grad_before - grad_after).abs().max().item()
    print(f"  gradient max abs err after refine = {gerr:.3e}")
    assert gerr < 1e-4, gerr
    print("  ok")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_single_level_thb_matches_dense()
    test_thb_refinement_preserves_field()
    test_thb_gradient_flow_and_consistency()
    print("ALL THB TRUNCATION TESTS PASSED")
