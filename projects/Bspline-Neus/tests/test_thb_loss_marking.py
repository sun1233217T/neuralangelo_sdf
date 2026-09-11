"""Loss-guided refinement marking tests (E4).

Run from the repository root:

    python projects/Bspline-Neus/tests/test_thb_loss_marking.py

Checks:
1. mark_cells_by_loss: topk budget, score normalization, min_count masking,
   all-zero statistics -> empty marking.
2. BSplineSDFWrapper.accumulate_loss: per-sample error mass lands in the
   correct cells; out-of-bounds samples are dropped without boolean indexing.
3. _mark_cells dispatches to loss marking and ranks the high-error cells.
4. maybe_refine keeps the statistics so the RGB wrapper (refining right
   after) reuses the identical marking; buffers rebuild on the new level.
5. Accumulation stops once max_levels is reached.
6. Loss mode without statistics falls back to the SDF band (never no-ops).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

hier = importlib.import_module("projects.Bspline-Neus.bspline_field.hierarchical")
modules = importlib.import_module("projects.Bspline-Neus.utils.modules")
mark_cells_by_loss = hier.mark_cells_by_loss
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


def _loss_hier_cfg(**overrides):
    cfg = dict(
        enabled=True, base_grid_size=8, max_levels=2,
        refine_iters=[10], refine_sdf_band=0.4, transfer_mode="hb",
        refine_mark_mode="loss", refine_max_marked_fraction=0.25,
    )
    cfg.update(overrides)
    return cfg


def test_mark_cells_by_loss():
    print("[test 1] mark_cells_by_loss: budget / normalization / min_count")
    cc = 6
    accum = torch.zeros((cc,) * 3)
    count = torch.zeros((cc,) * 3)
    # High total mass but many visits (low per-visit score).
    accum[0, 0, 0] = 10.0
    count[0, 0, 0] = 100.0
    # Lower mass, single visit (high per-visit score — the thin-structure case).
    accum[1, 1, 1] = 2.0
    count[1, 1, 1] = 1.0
    accum[2, 2, 2] = 1.0
    count[2, 2, 2] = 1.0
    # Below min_count: must never be marked despite nonzero accum.
    accum[3, 3, 3] = 100.0
    count[3, 3, 3] = 0.5

    marked = mark_cells_by_loss(accum, count, max_fraction=0.25, min_count=1.0)
    numel = cc ** 3
    assert int(marked.sum()) == 3, f"expected 3 marked, got {int(marked.sum())}"
    assert marked[1, 1, 1] and marked[2, 2, 2], "high per-visit score should win"
    assert marked[0, 0, 0], "third best score should fill the budget"
    assert not marked[3, 3, 3], "min_count violation must not be marked"
    assert int(marked.sum()) <= int(numel * 0.25)

    empty = mark_cells_by_loss(torch.zeros((cc,) * 3), torch.zeros((cc,) * 3),
                               max_fraction=0.25)
    assert int(empty.sum()) == 0, "zero statistics must mark nothing"
    print("  budget, normalization and min_count all respected")


def test_accumulate_loss_scatter():
    print("[test 2] accumulate_loss: scatter correctness + OOB handling")
    wrapper = BSplineSDFWrapper(_cfg(hierarchical=_loss_hier_cfg()))
    assert wrapper.loss_marking_enabled
    level = wrapper.hier_field.levels[0]
    cc = level.cell_count  # 8 - 2 = 6; step = 2/6
    # Two in-bounds points (cells (0,0,0) and (5,5,5)) + one out-of-bounds.
    points = torch.tensor([
        [[[-0.9, -0.9, -0.9], [0.9, 0.9, 0.9], [5.0, 0.0, 0.0]]],
    ])  # (1, 1, 3, 3)
    weights = torch.tensor([[[1.0, 0.5, 1.0]]])  # (1, 1, 3)
    rgb = torch.ones(1, 1, 3)
    target = torch.zeros(1, 1, 3)  # per-ray err = 1.0
    wrapper.accumulate_loss(points, weights, rgb, target)

    accum, count = wrapper._loss_accum, wrapper._loss_count
    assert accum is not None and tuple(accum.shape) == (cc,) * 3
    assert torch.isclose(accum[0, 0, 0], torch.tensor(1.0))
    assert torch.isclose(accum[5, 5, 5], torch.tensor(0.5))
    assert torch.isclose(count[0, 0, 0], torch.tensor(1.0))
    assert torch.isclose(count[5, 5, 5], torch.tensor(1.0))
    assert float(accum.sum()) == 1.5, "out-of-bounds sample must be dropped"
    assert float(count.sum()) == 2.0
    print(f"  accum sum {float(accum.sum()):.3f}, count sum {float(count.sum()):.3f}")


def test_mark_cells_dispatch():
    print("[test 3] _mark_cells: loss dispatch ranks high-error cells")
    wrapper = BSplineSDFWrapper(_cfg(hierarchical=_loss_hier_cfg()))
    cc = wrapper.hier_field.levels[0].cell_count
    # Concentrate error in two specific cells.
    pts = []
    for cell in [(0, 0, 0), (5, 5, 5)]:
        lo = wrapper.hier_field.levels[0]._lower
        st = wrapper.hier_field.levels[0].step
        center = lo + (torch.tensor(cell, dtype=torch.float32) + 0.5) * st
        pts.append(center)
    points = torch.stack(pts).reshape(1, 1, 2, 3)
    weights = torch.ones(1, 1, 2)
    rgb = torch.ones(1, 1, 3)
    target = torch.zeros(1, 1, 3)
    for _ in range(3):
        wrapper.accumulate_loss(points, weights, rgb, target)

    corners = wrapper.hier_field.levels[0].cell_corner_points()
    marked = wrapper._mark_cells(0, corners, 0)
    assert tuple(marked.shape) == (cc,) * 3
    assert marked[0, 0, 0] and marked[5, 5, 5], "high-error cells must be marked"
    print(f"  marked {int(marked.sum())} cells, both hot cells included")


def test_maybe_refine_integration():
    print("[test 4] maybe_refine: loss marking + RGB statistics reuse")
    wrapper = BSplineSDFWrapper(_cfg(hierarchical=_loss_hier_cfg(max_levels=3)))
    wrapper.accumulate_loss(
        torch.zeros(1, 1, 1, 3), torch.ones(1, 1, 1),
        torch.ones(1, 1, 3), torch.zeros(1, 1, 3))
    corners = wrapper.hier_field.levels[0].cell_corner_points()
    marked_before = wrapper._mark_cells(0, corners, 0)
    info = wrapper.maybe_refine(10)
    assert info is not None and info.get("refined"), "refine should fire at iter 10"
    assert wrapper.hier_field.num_levels == 2
    # After refinement the level-0 region has shrunk (marked cells moved to
    # the finer level), so a re-marking on level 0 will differ.  What must NOT
    # change is the loss statistics buffer — the RGB wrapper refines right
    # after and reuses them (regression: resetting the buffers in maybe_refine
    # silently dropped the RGB hierarchy to band marking).
    assert wrapper._loss_accum is not None, "loss stats must survive refinement"
    # The stats still match the pre-refinement finest level's shape (they are
    # rebuilt lazily on the new finest level by accumulate_loss's shape check).
    # The next accumulation rebuilds the buffers on the new finest level
    # (refinement doubles the grid, so the old shape never matches).
    wrapper.accumulate_loss(
        torch.zeros(1, 1, 1, 3), torch.ones(1, 1, 1),
        torch.ones(1, 1, 3), torch.zeros(1, 1, 3))
    cc_fine = wrapper.hier_field.levels[-1].cell_count
    assert tuple(wrapper._loss_accum.shape) == (cc_fine,) * 3
    print(f"  refined, info={info}")


def test_no_accumulate_past_max_levels():
    print("[test 5] no accumulation once max_levels is reached")
    wrapper = BSplineSDFWrapper(_cfg(hierarchical=_loss_hier_cfg()))  # max_levels=2
    wrapper.accumulate_loss(
        torch.zeros(1, 1, 1, 3), torch.ones(1, 1, 1),
        torch.ones(1, 1, 3), torch.zeros(1, 1, 3))
    assert wrapper._loss_accum is not None
    info = wrapper.maybe_refine(10)
    assert info is not None and info.get("refined")
    assert wrapper.hier_field.num_levels == 2  # == max_levels
    wrapper._loss_accum = None  # simulate post-refine consumption
    wrapper.accumulate_loss(
        torch.zeros(1, 1, 1, 3), torch.ones(1, 1, 1),
        torch.ones(1, 1, 3), torch.zeros(1, 1, 3))
    assert wrapper._loss_accum is None, "no accumulation after the last level"
    print("  accumulation stops at max_levels")


def test_fallback_to_band():
    print("[test 6] loss mode without statistics falls back to SDF band")
    wrapper = BSplineSDFWrapper(_cfg(hierarchical=_loss_hier_cfg()))
    assert wrapper._loss_accum is None
    info = wrapper.maybe_refine(10)
    assert info is not None and info.get("refined"), \
        "band fallback must still refine"
    assert info["marked_cells"] > 0
    print(f"  fallback band marking refined {info['marked_cells']} cells")


if __name__ == "__main__":
    test_mark_cells_by_loss()
    test_accumulate_loss_scatter()
    test_mark_cells_dispatch()
    test_maybe_refine_integration()
    test_no_accumulate_past_max_levels()
    test_fallback_to_band()
    print("ALL LOSS-MARKING TESTS PASSED")
