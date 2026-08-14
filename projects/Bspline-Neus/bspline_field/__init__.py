"""Pure-PyTorch B-spline field subset used by projects.Bspline-Neus.

This package is a stripped-down version of the B-spline field implementation
from bijectiveImplicitShell-nerf.  It intentionally avoids importing the CUDA
extension so that the lightweight point-sampling adapter can run without a
compiled bspline_field_cuda kernel.
"""

from .field import (
    BSplineDensityField,
    BSplineField,
    BSplineRGBField,
    BSplineSDField,
    BSplineSHRGBField,
    RadianceField,
    basis_1d,
    spherical_harmonics_basis,
)
from .hierarchical import (
    HierarchicalBSplineField,
    SparseBSplineLevel,
    basis_support_mask,
    children_region,
    mark_cells_by_sdf_band,
    mark_cells_by_value_band,
    prolongate_dense,
    subdivision_mask_1d,
)

__all__ = [
    "BSplineField",
    "BSplineDensityField",
    "BSplineSDField",
    "BSplineRGBField",
    "BSplineSHRGBField",
    "RadianceField",
    "basis_1d",
    "spherical_harmonics_basis",
    "HierarchicalBSplineField",
    "SparseBSplineLevel",
    "basis_support_mask",
    "children_region",
    "mark_cells_by_sdf_band",
    "mark_cells_by_value_band",
    "prolongate_dense",
    "subdivision_mask_1d",
]
