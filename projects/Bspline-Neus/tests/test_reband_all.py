"""Test full-hierarchy re-banding (reband_all)."""
import sys
sys.path.insert(0, ".")
import torch
from importlib import import_module

hier_mod = import_module("projects.Bspline-Neus.bspline_field.hierarchical")
HierarchicalBSplineField = hier_mod.HierarchicalBSplineField

torch.manual_seed(42)

# Create a 3-level hierarchy with known field
field = HierarchicalBSplineField(
    base_grid_size=8, spline_degree=2, bounds=[[-1, 1]] * 3,
    channels=1, max_levels=3, transfer_mode="hb",
)

# Refine twice to get 3 levels (partial marking so L0/L1 retain cells)
mark0 = torch.zeros((6, 6, 6), dtype=torch.bool)
mark0[2:4, 2:4, 2:4] = True  # mark center block
field.refine(lambda l, c: mark0)
mark1 = torch.zeros((12, 12, 12), dtype=torch.bool)
mark1[4:8, 4:8, 4:8] = True
field.refine(lambda l, c: mark1)
assert field.num_levels == 3
print(f"Levels: {field.num_levels}")
for l, lv in enumerate(field.levels):
    print(f"  L{l}: grid={lv.grid_size}, active={lv.num_active}, "
          f"region_cells={int(lv.region.sum())}")

# Set known values
for lv in field.levels:
    lv.values.data.normal_(0, 0.1)

# Evaluate field at test points
pts = torch.randn(500, 3) * 0.5
before = field.evaluate(pts).clone()

# Reband all levels with a band that will change regions
# The field has random values, so |SDF| < band will select different cells
info = field.reband_all(bands=[0.05, 0.05, 0.05])
print(f"\nReband_all: changed={info['changed']}")
for s in info["per_level"]:
    if s["added"] or s["removed"]:
        print(f"  L{s['level']}: +{s['added']}/-{s['removed']}")

# Field should change (cells were removed/added), but coarser levels
# still provide the base field
after = field.evaluate(pts)
diff = (after - before).abs()
print(f"\nField change: mean={diff.mean():.6f}, max={diff.max():.6f}")

# The field change should be bounded (not NaN or extreme)
assert not torch.isnan(after).any(), "NaN in field after reband!"
assert diff.max() < 10.0, f"Field changed too much: {diff.max()}"

# Verify structure consistency
for l, lv in enumerate(field.levels):
    active = lv.index_grid >= 0
    assert lv.values.shape[0] == int(active.sum()), \
        f"L{l}: values shape {lv.values.shape[0]} != active count {int(active.sum())}"
    print(f"  L{l}: active={lv.num_active}, region_cells={int(lv.region.sum())}, "
          f"values={lv.values.shape}")

# Test sync_from_owner_reband_all with a shared hierarchy
field2 = HierarchicalBSplineField(
    base_grid_size=8, spline_degree=2, bounds=[[-1, 1]] * 3,
    channels=1, max_levels=3, transfer_mode="hb",
)
color = HierarchicalBSplineField(
    base_grid_size=8, spline_degree=2, bounds=[[-1, 1]] * 3,
    channels=4, max_levels=3, transfer_mode="hb",
    structure_owner=field2,
)
# Refine both (partial marking)
mark0 = torch.zeros((6, 6, 6), dtype=torch.bool)
mark0[2:4, 2:4, 2:4] = True
sdf_info = field2.refine(lambda l, c: mark0)
color.refine(lambda l, c: mark0,
             owner_refine_info=sdf_info.get("owner_refine_info"))
mark1 = torch.zeros((12, 12, 12), dtype=torch.bool)
mark1[4:8, 4:8, 4:8] = True
sdf_info2 = field2.refine(lambda l, c: mark1)
color.refine(lambda l, c: mark1,
             owner_refine_info=sdf_info2.get("owner_refine_info"))
assert color.num_levels == 3

# Reband the owner, then sync the color
info = field2.reband_all(bands=[0.05, 0.05, 0.05])
print(f"\nShared hierarchy reband_all: changed={info['changed']}")
sync_info = color.sync_from_owner_reband_all(info)
print(f"Color sync: changed={sync_info.get('changed')}")

# Verify color levels are consistent
for l, lv in enumerate(color.levels):
    active = lv.index_grid >= 0
    assert lv.values.shape[0] == int(active.sum()), \
        f"Color L{l}: values shape mismatch"
    assert lv.values.shape[-1] == 4, f"Color L{l}: wrong channels"

print("\nALL REBAND_ALL TESTS PASSED")
