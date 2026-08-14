"""Phase 0 consistency tests for the hierarchical B-spline machinery.

Run from the repository root:

    python projects/Bspline-Neus/tests/test_thb_consistency.py

Test 1: dyadic prolongation is exact — a random dense field prolongated to
        the next level evaluates identically (up to float error).
Test 2: sparse hierarchical evaluation matches the dense ``BSplineField``
        when the whole region is active (single level).
Test 3: HB refinement preserves the field — marking arbitrary cells and
        refining must leave ``evaluate`` unchanged everywhere.
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
prolongate_dense = hier_mod.prolongate_dense
subdivision_mask_1d = hier_mod.subdivision_mask_1d

BOUNDS = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]


def _random_points(n, gen):
    # Keep clear of the exact boundary where both implementations clamp.
    return (torch.rand(n, 3, generator=gen) * 2.0 - 1.0) * 0.98


def test_subdivision_mask():
    print("[test 1a] subdivision mask sanity")
    for degree in (1, 2, 3):
        mask = subdivision_mask_1d(degree)
        assert mask.shape[0] == degree + 2
        assert torch.allclose(mask.sum(), torch.tensor(2.0)), mask.sum()
    print("  ok")


def test_prolongation_exact():
    print("[test 1b] prolongation exactness (degree 2 and 3)")
    gen = torch.Generator().manual_seed(0)
    for degree in (2, 3):
        G = 16
        coeffs = torch.randn(G, G, G, generator=gen)
        coarse = BSplineField(G, coeffs, bounds=BOUNDS, spline_degree=degree, trainable=False)
        fine_coeffs = prolongate_dense(coeffs.unsqueeze(-1), degree).squeeze(-1)
        G_f = 2 * G - degree
        assert fine_coeffs.shape == (G_f, G_f, G_f)
        fine = BSplineField(G_f, fine_coeffs, bounds=BOUNDS, spline_degree=degree, trainable=False)
        pts = _random_points(4096, gen)
        err = (coarse.evaluate(pts) - fine.evaluate(pts)).abs().max().item()
        print(f"  degree={degree}: max abs err = {err:.3e}")
        assert err < 1e-4, err
    print("  ok")


def test_sparse_matches_dense():
    print("[test 2] single-level hierarchical field matches dense BSplineField")
    gen = torch.Generator().manual_seed(1)
    G, degree = 16, 2
    coeffs = torch.randn(G, G, G, generator=gen)
    dense = BSplineField(G, coeffs, bounds=BOUNDS, spline_degree=degree, trainable=False)

    hier = HierarchicalBSplineField(G, degree, BOUNDS, channels=1, max_levels=1)
    hier.levels[0].values.data = coeffs.reshape(-1, 1).clone()

    pts = _random_points(4096, gen)
    dense_val = dense.evaluate(pts)
    hier_val = hier.evaluate(pts).squeeze(-1)
    err = (dense_val - hier_val).abs().max().item()
    print(f"  value max abs err = {err:.3e}")
    assert err < 1e-5, err

    dense_grad = dense.evaluate_gradient(pts)
    hier_grad = hier.evaluate_gradient(pts).squeeze(1)
    gerr = (dense_grad - hier_grad).abs().max().item()
    print(f"  gradient max abs err = {gerr:.3e}")
    assert gerr < 1e-4, gerr
    print("  ok")


def test_hb_refinement_preserves_field():
    print("[test 3] HB refinement preserves the field")
    gen = torch.Generator().manual_seed(2)
    G, degree = 16, 2
    coeffs = torch.randn(G, G, G, generator=gen)
    hier = HierarchicalBSplineField(G, degree, BOUNDS, channels=1, max_levels=4)
    hier.levels[0].values.data = coeffs.reshape(-1, 1).clone()

    pts = _random_points(8192, gen)
    before = hier.evaluate(pts).detach().clone()

    # Mark a blob of cells (a centered box) at each refinement round.
    for round_idx in range(3):
        level = hier.levels[-1]
        cc = level.cell_count

        def mark_fn(level_index, corner_points, cc=cc, round_idx=round_idx):
            coords = torch.arange(cc)
            lo = cc // 4 + round_idx
            hi = 3 * cc // 4
            m = torch.zeros(cc, cc, cc, dtype=torch.bool)
            m[lo:hi, lo:hi, lo:hi] = True
            return m

        info = hier.refine(mark_fn)
        after = hier.evaluate(pts).detach()
        err = (before - after).abs().max().item()
        print(f"  round {round_idx}: info={info}, max abs err = {err:.3e}")
        assert info["refined"], info
        assert err < 1e-4, err
    print("  ok")


def test_multi_level_region_consistency():
    """Points in refined areas get contributions from the right levels."""
    print("[test 3b] region bookkeeping after refinement")
    gen = torch.Generator().manual_seed(3)
    hier = HierarchicalBSplineField(16, 2, BOUNDS, channels=1, max_levels=3)
    hier.levels[0].values.data = torch.randn(16, 16, 16, generator=gen).reshape(-1, 1)

    cc = hier.levels[0].cell_count

    def mark_half(level_index, corner_points, cc=cc):
        m = torch.zeros(cc, cc, cc, dtype=torch.bool)
        m[: cc // 2] = True  # refine the x-lower half
        return m

    info = hier.refine(mark_half)
    assert info["refined"]
    coarse, fine = hier.levels[0], hier.levels[1]
    # Coarse region must have lost the marked cells; fine region holds their children.
    assert not coarse.region[: cc // 2].any()
    assert coarse.region[cc // 2:].all()
    assert fine.region.shape == (2 * cc, 2 * cc, 2 * cc)
    assert fine.region[:cc].all() and not fine.region[cc:].any()
    # No active basis may remain whose support is fully inside the marked area.
    print("  ok")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_subdivision_mask()
    test_prolongation_exact()
    test_sparse_matches_dense()
    test_hb_refinement_preserves_field()
    test_multi_level_region_consistency()
    print("ALL HIERARCHICAL B-SPLINE TESTS PASSED")
