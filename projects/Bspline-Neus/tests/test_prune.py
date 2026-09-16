"""Test activity-based pruning: region shrink, field continuity on the
retained domain, and structure-sharing remap compatibility."""
import sys
sys.path.insert(0, ".")
import torch
from importlib import import_module

hier_mod = import_module("projects.Bspline-Neus.bspline_field.hierarchical")
HierarchicalBSplineField = hier_mod.HierarchicalBSplineField

torch.manual_seed(42)

for mode in ("hb", "thb"):
    field = HierarchicalBSplineField(
        base_grid_size=8, spline_degree=2, bounds=[[-1, 1]] * 3,
        channels=1, max_levels=3, transfer_mode=mode,
    )
    field.refine(lambda l, c: torch.ones((6, 6, 6), dtype=torch.bool))
    assert field.num_levels == 2
    field.levels[0].values.data.normal_(0, 0.1)
    field.levels[1].values.data.normal_(0, 0.01)

    level = field.levels[1]
    cc = level.cell_count
    n_before = int(level.region.sum())

    # Keep only the left half of the region (plus margin handled by caller).
    keep = torch.zeros((cc,) * 3, dtype=torch.bool)
    keep[: cc // 2] = True

    pts = torch.rand(200, 3) * 2 - 1
    # Keep clear of the keep/prune boundary by more than the deg-2 support
    # (3 cells; cell size 2/16 = 0.125) so retained-domain values must hold.
    pts[:, 0] = -0.45 - pts[:, 0].abs() * 0.5  # x in [-0.95, -0.45]
    before = field.evaluate(pts).clone()

    info = field.prune_finest(keep)
    n_after = int(level.region.sum())
    assert info["changed"] and info["removed"] == n_before - n_after > 0
    print(f"[{mode}] pruned {info['removed']} cells ({n_before} -> {n_after})")

    after = field.evaluate(pts)
    max_diff = (before - after).abs().max().item()
    print(f"[{mode}] field max diff on retained domain: {max_diff:.2e}")
    assert max_diff < 1e-4, f"[{mode}] field changed on retained domain!"

    # No-op when keep covers everything.
    info2 = field.prune_finest(torch.ones((cc,) * 3, dtype=torch.bool))
    assert not info2["changed"], "expected no-op"

    # Guard: pruning everything must be refused (empty active set would
    # disconnect the level's values from the autograd graph).
    info3 = field.prune_finest(torch.zeros((cc,) * 3, dtype=torch.bool))
    assert not info3["changed"] and "empty" in info3.get("reason", ""), \
        "empty-prune guard failed"

    # old_index_grid present for structure-sharing remap.
    assert "old_index_grid" in info and info["level_idx"] == 1

print("prune_finest: all tests passed")
