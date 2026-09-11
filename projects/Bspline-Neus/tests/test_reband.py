"""Test re-banding: region update, coefficient preservation, structure sharing."""
import sys
sys.path.insert(0, ".")
import torch
from importlib import import_module

hier_mod = import_module("projects.Bspline-Neus.bspline_field.hierarchical")
HierarchicalBSplineField = hier_mod.HierarchicalBSplineField

torch.manual_seed(42)

# Create a 2-level hierarchy
field = HierarchicalBSplineField(
    base_grid_size=8, spline_degree=2, bounds=[[-1, 1]] * 3,
    channels=1, max_levels=3, transfer_mode="hb",
)

# Force a refinement to get 2 levels
field.refine(lambda l, c: torch.ones((6, 6, 6), dtype=torch.bool))
assert field.num_levels == 2
print(f"After refine: {field.num_levels} levels, "
      f"L1 active={field.levels[1].num_active}")

# Set some values so we can check preservation
field.levels[0].values.data.normal_(0, 0.1)
field.levels[1].values.data.normal_(0, 0.01)

# Save original values for comparison
orig_l0 = field.levels[0].values.data.clone()
orig_l1 = field.levels[1].values.data.clone()
orig_l1_active = int(field.levels[1].num_active)

# Evaluate field at some points before rebanding
pts = torch.randn(100, 3) * 0.5
before = field.evaluate(pts).clone()

# Reband with a tight band (should remove some cells)
info = field.reband_finest(band=0.5)
print(f"Reband: changed={info.get('changed')}, "
      f"added={info.get('added', 0)}, removed={info.get('removed', 0)}")

if info.get("changed"):
    # L0 should be untouched
    assert torch.equal(field.levels[0].values.data, orig_l0), "L0 changed!"

    # L1 region should have changed
    print(f"L1 active: {orig_l1_active} -> {field.levels[1].num_active}")

    # Field values should be approximately preserved (HB: coarse levels still
    # contribute, new fine cells start at zero)
    after = field.evaluate(pts)
    diff = (after - before).abs().max().item()
    print(f"Max field change: {diff:.6f}")
    assert diff < 1.0, f"Field changed too much: {diff}"

# Reband with a very wide band — should add cells back
info2 = field.reband_finest(band=10.0)
print(f"Re-band (wide): changed={info2.get('changed')}, "
      f"added={info2.get('added', 0)}, removed={info2.get('removed', 0)}")

# Test sync_from_owner_reband for shared hierarchy.
# Create the color field BEFORE refinement so it tracks the owner's levels.
field2 = HierarchicalBSplineField(
    base_grid_size=8, spline_degree=2, bounds=[[-1, 1]] * 3,
    channels=1, max_levels=3, transfer_mode="hb",
)
color_field = HierarchicalBSplineField(
    base_grid_size=8, spline_degree=2, bounds=[[-1, 1]] * 3,
    channels=4, max_levels=3, transfer_mode="hb",
    structure_owner=field2,
)
# Refine both (owner first, then shared).
sdf_info = field2.refine(lambda l, c: torch.ones((6, 6, 6), dtype=torch.bool))
color_field.refine(
    lambda l, c: torch.ones((6, 6, 6), dtype=torch.bool),
    owner_refine_info=sdf_info.get("owner_refine_info"),
)
assert color_field.num_levels == field2.num_levels == 2
print(f"Shared fields: SDF {field2.num_levels} levels, "
      f"color {color_field.num_levels} levels, "
      f"color L1 channels={color_field.levels[1].values.shape[-1]}")

# Reband the owner, then sync the shared hierarchy.
field2.levels[0].values.data.normal_(0, 0.1)
color_field.levels[1].values.data.normal_(0, 0.01)
reband_info = field2.reband_finest(band=0.5)
print(f"Owner reband: changed={reband_info.get('changed')}")
if reband_info.get("changed"):
    sync_info = color_field.sync_from_owner_reband(reband_info)
    print(f"Color sync: changed={sync_info.get('changed')}")
    assert sync_info.get("changed")
    # Color values should have been remapped to the new index grid.
    new_active = int(color_field.levels[1].index_grid.ge(0).sum())
    assert color_field.levels[1].values.shape[0] == new_active
    print(f"Color L1 values remapped: {new_active} active")

print("ALL REBAND TESTS PASSED")
