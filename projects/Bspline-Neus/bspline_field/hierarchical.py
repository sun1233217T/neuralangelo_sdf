"""Hierarchical (adaptive, eventually truncated) B-spline fields in pure PyTorch.

This module implements the sparse hierarchical B-spline machinery for
Bspline-Neus:

- ``subdivision_mask_1d`` / ``prolongate_dense``: exact dyadic refinement
  (knot insertion for uniform B-splines).  A degree-p uniform B-spline basis
  function satisfies beta(x) = sum_k m_k beta(2x - k) with
  m_k = binom(p+1, k) / 2^p, k = 0..p+1.  This makes refinement *exact*:
  prolongating the control grid to the next level leaves the field unchanged.
- ``SparseBSplineLevel``: one hierarchy level.  Control points ("basis
  functions") are active only where the level is responsible; inactive ones
  cost no memory.  Storage is a dense int32 ``index_grid`` (G^3, -1 =
  inactive) plus a sparse ``values`` parameter of shape (N, C).
- ``HierarchicalBSplineField``: a sum of levels with a ``refine`` entry point
  that shrinks the coarse region, creates/extends the next level, and
  transfers coefficients so the represented field is unchanged.

Refinement semantics (HB mode)
------------------------------
Level l owns a set of "region" cells.  A basis function is *active* iff every
cell of its (p+1)^3 support lies in the region.  When cells are marked for
refinement they leave the coarse region and their 2^3 children enter the
next level's region.  Coarse basis functions whose support is fully marked
are removed and their coefficients are transferred to their children via the
subdivision mask, which preserves the field exactly.  Straddling coarse basis
functions stay active and untouched (HB).  THB additionally truncates them at
evaluation time (Phase 4, ``evaluate_thb``) and transfers the *full*
prolongation into the refined region.

Grid-size bookkeeping matches ``BSplineField``: ``cell_count = G - p`` and
doubling the cells gives ``G_{l+1} = 2 * G_l - p``.
"""

from __future__ import annotations

import math
from itertools import product
from typing import Any, Callable, Sequence

import torch
from torch import Tensor, nn

from .field import _as_float_tensor, _coerce_spline_degree, basis_1d, basis_1d_with_deriv

Bounds = Sequence[Sequence[float]]


# ---------------------------------------------------------------------------
# Subdivision / prolongation
# ---------------------------------------------------------------------------

def subdivision_mask_1d(degree: int, *, dtype=None, device=None) -> Tensor:
    """1D dyadic subdivision mask for uniform B-splines.

    m_k = binom(p+1, k) / 2^p for k = 0..p+1.  Sums to 2.
    """
    p = _coerce_spline_degree(degree)
    coeffs = [math.comb(p + 1, k) / (2.0 ** p) for k in range(p + 2)]
    return torch.tensor(coeffs, dtype=dtype or torch.get_default_dtype(), device=device)


def prolongate_dense(values: Tensor, degree: int) -> Tensor:
    """Prolongate a dense control grid ``(G, G, G, C)`` to the next level.

    The fine grid has ``G_f = 2 * G - p`` control points per axis (double the
    cells).  With the convention of ``BSplineField`` (cell i uses control
    points i..i+p), the field is f(x) = sum_i P_i M(x - i) where M is the
    cardinal B-spline with support [-p, 1].  The refinement equation gives

        Q_j = sum_k m_k * P_{(j + p - k) / 2}   (k = 0..p+1, j+p-k even)

    i.e. each coarse coefficient P_i contributes to fine indices
    j = 2i + k - p (k = 0..p+1) with weight m_k.  Contributions landing
    outside [0, G_f) only influence the exact domain boundary and are dropped.

    The represented field is unchanged *inside the bounds*.
    """
    p = _coerce_spline_degree(degree)
    if values.ndim != 4 or values.shape[0] != values.shape[1] or values.shape[1] != values.shape[2]:
        raise ValueError("values must have shape (G, G, G, C).")
    G = values.shape[0]
    C = values.shape[-1]
    G_f = 2 * G - p
    mask = subdivision_mask_1d(p, dtype=values.dtype, device=values.device)
    out = values.new_zeros((G_f, G_f, G_f, C))
    idx = torch.cartesian_prod(*[torch.arange(G, device=values.device)] * 3)
    vals = values.reshape(-1, C)
    for k0, k1, k2 in product(range(p + 2), repeat=3):
        w = mask[k0] * mask[k1] * mask[k2]
        j0 = 2 * idx[:, 0] + (k0 - p)
        j1 = 2 * idx[:, 1] + (k1 - p)
        j2 = 2 * idx[:, 2] + (k2 - p)
        valid = (j0 >= 0) & (j1 >= 0) & (j2 >= 0) & (j0 < G_f) & (j1 < G_f) & (j2 < G_f)
        if not valid.any():
            continue
        flat = (j0[valid] * G_f + j1[valid]) * G_f + j2[valid]
        out.reshape(-1, C).index_add_(0, flat, w * vals[valid])
    return out


# ---------------------------------------------------------------------------
# Region / basis masks
# ---------------------------------------------------------------------------

def basis_support_mask(region: Tensor, degree: int) -> Tensor:
    """Boolean (G, G, G) mask of basis functions whose support lies in ``region``.

    ``region`` is a boolean (cc, cc, cc) tensor over cells
    (``cc = G - degree``).  Basis function i has support cells
    ``i_d - o`` for o in [0, degree] per axis; support cells outside the cell
    grid (boundary truncation) impose no constraint, matching the convention
    of ``BSplineField`` where boundary basis functions are partially
    supported inside the domain.
    """
    p = _coerce_spline_degree(degree)
    if region.ndim != 3 or region.shape[0] != region.shape[1] or region.shape[1] != region.shape[2]:
        raise ValueError("region must have shape (cc, cc, cc).")
    cc = region.shape[0]
    G = cc + p
    mask = torch.ones((G, G, G), dtype=torch.bool, device=region.device)
    for o0, o1, o2 in product(range(p + 1), repeat=3):
        constraint = torch.ones_like(mask)
        constraint[o0:o0 + cc, o1:o1 + cc, o2:o2 + cc] = region
        mask &= constraint
    return mask


def children_region(marked: Tensor) -> Tensor:
    """Map marked level-l cells to their 2x children cells at level l+1."""
    out = marked
    for dim in range(3):
        out = out.repeat_interleave(2, dim=dim)
    return out


def parent_cells(omega_fine: Tensor) -> Tensor:
    """Level-l cells fully covered by the level-(l+1) cell set ``omega_fine``.

    A parent cell is covered iff all of its 2^3 children are.
    """
    cc_f = omega_fine.shape[0]
    if cc_f % 2 != 0:
        raise ValueError("fine region cell count must be even.")
    blocks = omega_fine.reshape(cc_f // 2, 2, cc_f // 2, 2, cc_f // 2, 2)
    return blocks.amin(dim=(1, 3, 5)).to(torch.bool)


def build_transition_tables(degree: int, *, dtype=None, device=None) -> Tensor:
    """THB truncation transition tables, shape ``(8, S, S)`` with ``S=(p+1)^3``.

    Truncated basis values obey the fine-to-coarse recursion

        V^l_i = sum_{j not in R^{l+1}} M[j, i] * V^{l+1}_j,

    where the children of coarse basis i are the fine indices 2i+k-p
    (k in [0, p+1]^3, weights m_k from the subdivision mask).  For a fixed
    cell parity pi = c' - 2c in {0,1}^3 the mapping between the (p+1)^3
    support slots of adjacent levels is constant, so the recursion becomes a
    per-parity linear map:

        b_axis = 2 * a_axis + k_axis - p - pi_axis
        A_pi[a, b] = sum_{k: slot(k) = b} m_k  (tensor product over axes)

    The parity case index is pi0*4 + pi1*2 + pi2.
    """
    p = _coerce_spline_degree(degree)
    s1 = p + 1
    mask = subdivision_mask_1d(p, dtype=dtype, device=device)
    A1 = torch.zeros(2, s1, s1, dtype=dtype or torch.get_default_dtype(), device=device)
    for pi in (0, 1):
        for a in range(s1):
            for k in range(p + 2):
                b = 2 * a + k - p - pi
                if 0 <= b < s1:
                    A1[pi, a, b] += mask[k]
    S = s1 ** 3
    tables = torch.zeros(8, S, S, dtype=A1.dtype, device=A1.device)
    for case in range(8):
        pi0, pi1, pi2 = (case >> 2) & 1, (case >> 1) & 1, case & 1
        tables[case] = torch.einsum(
            "ra,sb,tc->rstabc", A1[pi0], A1[pi1], A1[pi2]
        ).reshape(S, S)
    return tables


# ---------------------------------------------------------------------------
# Level structure (shared between SDF/color hierarchies)
# ---------------------------------------------------------------------------

class LevelStructure(nn.Module):
    """Container for the structural tensors of one sparse B-spline level.

    Separating structure from ``values`` lets two hierarchies (e.g. SDF and
    color) share the same grid topology while keeping their own coefficients.
    The owning ``HierarchicalBSplineField`` registers these as buffers so they
    are saved/moved with the model; a referencing level only holds the object
    reference and delegates attribute access to it.
    """

    TENSOR_NAMES = (
        "_lower", "_upper", "step", "region", "omega", "rmask",
        "_support_offsets_flat", "_support_axis_r", "_support_axis_s",
        "_support_axis_t", "index_grid",
    )

    def __init__(
        self,
        grid_size: int,
        spline_degree: int,
        bounds: Bounds,
        *,
        region: Tensor | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        self.spline_degree = _coerce_spline_degree(spline_degree)
        self.support_size = self.spline_degree + 1
        self.cell_count = self.grid_size - self.spline_degree
        if self.cell_count < 1:
            raise ValueError("grid_size too small for the given spline degree.")

        dtype = dtype or torch.get_default_dtype()
        bounds_t = _as_float_tensor(bounds, dtype=dtype, device=device)
        if bounds_t.shape != (3, 2):
            raise ValueError("bounds must have shape (3, 2).")
        self.register_buffer("_lower", bounds_t[:, 0].clone())
        self.register_buffer("_upper", bounds_t[:, 1].clone())
        self.register_buffer("step", (bounds_t[:, 1] - bounds_t[:, 0]) / self.cell_count)

        if region is None:
            region = torch.ones(
                (self.cell_count,) * 3, dtype=torch.bool, device=device
            )
        else:
            region = torch.as_tensor(region, dtype=torch.bool, device=device)
            if tuple(region.shape) != (self.cell_count,) * 3:
                raise ValueError(f"region must have shape {(self.cell_count,) * 3}.")
        self.register_buffer("region", region.clone())
        self.register_buffer("omega", self.region.clone())
        self.register_buffer("rmask", basis_support_mask(self.region, self.spline_degree))

        axis_ids = torch.arange(self.support_size, dtype=torch.long)
        triplets = torch.cartesian_prod(axis_ids, axis_ids, axis_ids)
        G2 = self.grid_size * self.grid_size
        self.register_buffer(
            "_support_offsets_flat",
            (
                triplets[:, 0] * G2 + triplets[:, 1] * self.grid_size + triplets[:, 2]
            ).to(device),
        )
        self.register_buffer("_support_axis_r", triplets[:, 0].to(device))
        self.register_buffer("_support_axis_s", triplets[:, 1].to(device))
        self.register_buffer("_support_axis_t", triplets[:, 2].to(device))

        active = basis_support_mask(self.region, self.spline_degree)
        index_grid = torch.full(
            (self.grid_size,) * 3, -1, dtype=torch.int32, device=device
        )
        num_active = int(active.sum().item())
        index_grid[active] = torch.arange(
            num_active, dtype=torch.int32, device=device
        )
        self.register_buffer("index_grid", index_grid)

    def update_region_and_masks(self, region: Tensor, omega: Tensor | None = None) -> None:
        """Replace region/omega/rmask/index_grid for a new topology.

        Used during refinement of the owning hierarchy.  Referencing levels
        see the change automatically because they access the same object.
        """
        self.region = region.clone()
        self.omega = region.clone() if omega is None else omega.clone()
        self.rmask = basis_support_mask(self.region, self.spline_degree)

        active = basis_support_mask(self.region, self.spline_degree)
        self.index_grid = torch.full(
            (self.grid_size,) * 3, -1, dtype=torch.int32, device=self.index_grid.device
        )
        num_active = int(active.sum().item())
        if num_active > 0:
            self.index_grid[active] = torch.arange(
                num_active, dtype=torch.int32, device=self.index_grid.device
            )

    def update_omega(self, omega: Tensor) -> None:
        """Update cumulative-domain buffers used by THB truncation."""
        omega = torch.as_tensor(omega, dtype=torch.bool, device=self.region.device)
        if tuple(omega.shape) != (self.cell_count,) * 3:
            raise ValueError("omega has wrong shape.")
        self.omega = omega.clone()
        self.rmask = basis_support_mask(self.omega, self.spline_degree)


# ---------------------------------------------------------------------------
# Sparse level
# ---------------------------------------------------------------------------

class SparseBSplineLevel(nn.Module):
    """One sparse level of a hierarchical B-spline field.

    Parameters
    ----------
    structure : LevelStructure
        Shared structural buffers.  Created automatically if omitted.
    channels : int
        Value dimension per control point (1 for SDF, 3*sh_dim for color).
    """

    STRUCTURAL_FIELDS = frozenset(LevelStructure.TENSOR_NAMES)

    def __init__(
        self,
        grid_size: int | None = None,
        spline_degree: int | None = None,
        bounds: Bounds | None = None,
        channels: int | None = None,
        *,
        structure: LevelStructure | None = None,
        region: Tensor | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if structure is None:
            if grid_size is None or spline_degree is None or bounds is None:
                raise ValueError(
                    "SparseBSplineLevel requires either a structure or "
                    "(grid_size, spline_degree, bounds)."
                )
            structure = LevelStructure(
                grid_size, spline_degree, bounds, region=region,
                dtype=dtype, device=device,
            )
        if channels is None:
            raise ValueError("SparseBSplineLevel requires channels.")
        # Store the structure object without registering it as a PyTorch submodule,
        # so a shared structure is not duplicated in the state dict of the color
        # hierarchy that merely references the SDF owner's structures.
        object.__setattr__(self, "_structure_ref", structure)
        self.channels = int(channels)

        active = basis_support_mask(self._structure_ref.region, self._structure_ref.spline_degree)
        num_active = int(active.sum().item())
        self.values = nn.Parameter(
            torch.zeros(
                num_active, self.channels,
                dtype=dtype or torch.get_default_dtype(),
                device=self._structure_ref.index_grid.device,
            )
        )

    def __getattr__(self, name: str) -> Any:
        if name in self.STRUCTURAL_FIELDS:
            return getattr(self._structure_ref, name)
        # Fall back to nn.Module's handling of parameters/buffers/submodules.
        return super().__getattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.STRUCTURAL_FIELDS:
            if not hasattr(self, "_structure_ref"):
                # During __init__ the structure may not exist yet; fall through.
                super().__setattr__(name, value)
                return
            setattr(self._structure_ref, name, value)
            return
        super().__setattr__(name, value)

    def _apply(self, fn, recurse=True):
        if self._structure_ref is not None:
            self._structure_ref._apply(fn)
        return super()._apply(fn, recurse)

    # -- geometry helpers ----------------------------------------------------

    @property
    def grid_size(self) -> int:
        return self._structure_ref.grid_size

    @property
    def spline_degree(self) -> int:
        return self._structure_ref.spline_degree

    @property
    def support_size(self) -> int:
        return self._structure_ref.support_size

    @property
    def cell_count(self) -> int:
        return self._structure_ref.cell_count

    @property
    def num_active(self) -> int:
        return self.values.shape[0]

    def cell_corner_points(self) -> Tensor:
        """Lattice points of the cell grid, shape ``((cc+1)^3, 3)``."""
        cc = self.cell_count
        axes = [
            self._lower[d] + self.step[d] * torch.arange(
                cc + 1, dtype=self.step.dtype, device=self.step.device
            )
            for d in range(3)
        ]
        gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
        return torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    # -- evaluation ----------------------------------------------------------

    def _prepare_points(self, points: Any) -> tuple[Tensor, tuple[int, ...]]:
        pts = _as_float_tensor(points, dtype=self.values.dtype, device=self.values.device)
        scalar_input = tuple(pts.shape) == (3,)
        if scalar_input:
            pts = pts.reshape(1, 3)
        if pts.ndim < 1 or pts.shape[-1] != 3:
            raise ValueError("points must have shape (3,) or (..., 3).")
        output_shape = tuple(pts.shape[:-1])
        return pts.reshape(-1, 3), output_shape

    def evaluate(self, points: Any) -> Tensor:
        """Evaluate the level at ``points`` -> ``(..., C)``."""
        flat, output_shape = self._prepare_points(points)
        if self.values.shape[0] == 0:
            return self.values.new_zeros(flat.shape[0], self.channels).reshape(
                output_shape + (self.channels,)
            )
        # Mirror BSplineField: map to cell coordinates, clamp the upper edge.
        local = (flat - self._lower) / self.step
        upper = torch.nextafter(
            torch.full((3,), float(self.cell_count), dtype=local.dtype, device=local.device),
            torch.zeros(3, dtype=local.dtype, device=local.device),
        )
        local = torch.minimum(torch.maximum(local, torch.zeros_like(local)), upper)
        ijk = torch.floor(local).to(torch.long)
        uvw = local - ijk.to(local.dtype)

        basis_x = basis_1d(uvw[:, 0], degree=self.spline_degree)
        basis_y = basis_1d(uvw[:, 1], degree=self.spline_degree)
        basis_z = basis_1d(uvw[:, 2], degree=self.spline_degree)

        i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]
        G2 = self.grid_size * self.grid_size
        base = i * G2 + j * self.grid_size + k
        cp_flat = base[:, None] + self._support_offsets_flat[None, :]  # (Np, S)
        slots = self.index_grid.reshape(-1)[cp_flat]  # (Np, S)
        valid = slots >= 0

        weights = (
            basis_x.index_select(1, self._support_axis_r)
            * basis_y.index_select(1, self._support_axis_s)
            * basis_z.index_select(1, self._support_axis_t)
        )
        weights = weights * valid.to(weights.dtype)
        gathered = self.values.index_select(0, slots.clamp_min(0).reshape(-1))
        gathered = gathered.reshape(slots.shape[0], slots.shape[1], self.channels)
        out = (gathered * weights.unsqueeze(-1)).sum(dim=1)  # (Np, C)
        return out.reshape(output_shape + (self.channels,))

    def evaluate_gradient(self, points: Any) -> Tensor:
        """Spatial gradient via autograd, shape ``(..., C, 3)``."""
        flat, output_shape = self._prepare_points(points)
        pts = flat.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            values = self.evaluate(pts)  # (Np, C)
            grads = []
            for c in range(self.channels):
                g = torch.autograd.grad(values[:, c].sum(), pts, create_graph=True)[0]
                grads.append(g)
            grad = torch.stack(grads, dim=1)  # (Np, C, 3)
        return grad.reshape(output_shape + (self.channels, 3))

    def evaluate_with_deriv(self, points: Any) -> tuple[Tensor, Tensor, Tensor]:
        """解析求值：返回 (values (...,C), grads (...,C,3), hess_diag (...,C,3))。

        与 ``evaluate`` 同构，但三轴基函数换成 ``basis_1d_with_deriv`` 的
        值/一阶/二阶导数基，按 (value, gx, gy, gz, hxx, hyy, hzz) 组出
        7 组 (Np, S) 权重；gather 只做一次，逐组加权求和以控制峰值显存。
        """
        flat, output_shape = self._prepare_points(points)
        if self.values.shape[0] == 0:
            values = self.values.new_zeros(flat.shape[0], self.channels)
            grads = self.values.new_zeros(flat.shape[0], self.channels, 3)
            hess = self.values.new_zeros(flat.shape[0], self.channels, 3)
            return (
                values.reshape(output_shape + (self.channels,)),
                grads.reshape(output_shape + (self.channels, 3)),
                hess.reshape(output_shape + (self.channels, 3)),
            )
        # Mirror BSplineField: map to cell coordinates, clamp the upper edge.
        local = (flat - self._lower) / self.step
        upper = torch.nextafter(
            torch.full((3,), float(self.cell_count), dtype=local.dtype, device=local.device),
            torch.zeros(3, dtype=local.dtype, device=local.device),
        )
        local = torch.minimum(torch.maximum(local, torch.zeros_like(local)), upper)
        ijk = torch.floor(local).to(torch.long)
        uvw = local - ijk.to(local.dtype)

        basis_x, basis_dx, basis_d2x = basis_1d_with_deriv(uvw[:, 0], degree=self.spline_degree)
        basis_y, basis_dy, basis_d2y = basis_1d_with_deriv(uvw[:, 1], degree=self.spline_degree)
        basis_z, basis_dz, basis_d2z = basis_1d_with_deriv(uvw[:, 2], degree=self.spline_degree)

        i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]
        G2 = self.grid_size * self.grid_size
        base = i * G2 + j * self.grid_size + k
        cp_flat = base[:, None] + self._support_offsets_flat[None, :]  # (Np, S)
        slots = self.index_grid.reshape(-1)[cp_flat]  # (Np, S)
        valid = slots >= 0
        valid_f = valid.to(flat.dtype)

        r = self._support_axis_r
        s = self._support_axis_s
        t = self._support_axis_t
        bx = basis_x.index_select(1, r)
        by = basis_y.index_select(1, s)
        bz = basis_z.index_select(1, t)
        inv_step = 1.0 / self.step
        inv_step2 = inv_step * inv_step
        weight_groups = (
            bx * by * bz,
            basis_dx.index_select(1, r) * by * bz * inv_step[0],
            bx * basis_dy.index_select(1, s) * bz * inv_step[1],
            bx * by * basis_dz.index_select(1, t) * inv_step[2],
            basis_d2x.index_select(1, r) * by * bz * inv_step2[0],
            bx * basis_d2y.index_select(1, s) * bz * inv_step2[1],
            bx * by * basis_d2z.index_select(1, t) * inv_step2[2],
        )
        weight_groups = tuple(w * valid_f for w in weight_groups)

        gathered = self.values.index_select(0, slots.clamp_min(0).reshape(-1))
        gathered = gathered.reshape(slots.shape[0], slots.shape[1], self.channels)
        outs = [(gathered * w.unsqueeze(-1)).sum(dim=1) for w in weight_groups]  # 7 x (Np, C)
        values = outs[0]
        grads = torch.stack(outs[1:4], dim=-1)  # (Np, C, 3)
        hess = torch.stack(outs[4:7], dim=-1)  # (Np, C, 3)
        return (
            values.reshape(output_shape + (self.channels,)),
            grads.reshape(output_shape + (self.channels, 3)),
            hess.reshape(output_shape + (self.channels, 3)),
        )

    # -- refinement bookkeeping ----------------------------------------------

    @torch.no_grad()
    def rebuild(
        self,
        new_region: Tensor,
        active_mask: Tensor | None = None,
        transferred_values: Tensor | None = None,
        *,
        carry_over: bool = True,
    ) -> None:
        """Rebuild sparse storage for a new region / active-basis set.

        ``new_region`` updates the responsibility cells buffer.  ``active_mask``
        explicitly selects the active basis functions (HB semantics: a basis
        stays active while its support is not *fully* refined, so it may
        straddle the region boundary); when omitted it defaults to
        ``basis_support_mask(new_region)`` (correct for the finest level).

        ``transferred_values`` is a dense ``(G, G, G, C)`` tensor with
        initialization for newly activated control points (from coefficient
        transfer).  With ``carry_over=True``, control points that stay active
        keep their current values; with ``False`` every active control point
        is (re)initialized from ``transferred_values`` (used when the level is
        created and previous values are meaningless).
        """
        new_region = torch.as_tensor(new_region, dtype=torch.bool, device=self.region.device)
        if tuple(new_region.shape) != (self.cell_count,) * 3:
            raise ValueError("new_region has wrong shape.")
        if active_mask is None:
            active = basis_support_mask(new_region, self.spline_degree)
        else:
            active = torch.as_tensor(active_mask, dtype=torch.bool, device=self.region.device)
            if tuple(active.shape) != (self.grid_size,) * 3:
                raise ValueError("active_mask has wrong shape.")

        num_active = int(active.sum().item())
        index_grid = torch.full_like(self.index_grid, -1)
        index_grid[active] = torch.arange(
            num_active, dtype=self.index_grid.dtype, device=self.region.device
        )
        new_values = self.values.new_zeros(num_active, self.channels)
        if transferred_values is not None:
            new_values.copy_(transferred_values[active])

        if carry_over:
            old_active = self.index_grid >= 0
            keep = active & old_active
            if keep.any():
                src_slots = self.index_grid[keep]
                dst_slots = index_grid[keep]
                new_values[dst_slots] = self.values.detach()[src_slots]

        self.region = new_region
        self.index_grid = index_grid
        self.values = nn.Parameter(new_values)

    @torch.no_grad()
    def update_omega(self, omega: Tensor) -> None:
        """Update the cumulative-domain buffers used by THB truncation."""
        omega = torch.as_tensor(omega, dtype=torch.bool, device=self.region.device)
        if tuple(omega.shape) != (self.cell_count,) * 3:
            raise ValueError("omega has wrong shape.")
        self.omega = omega.clone()
        self.rmask = basis_support_mask(self.omega, self.spline_degree)


# ---------------------------------------------------------------------------
# Hierarchical field
# ---------------------------------------------------------------------------

class HierarchicalBSplineField(nn.Module):
    """Sum of sparse B-spline levels with exact dyadic refinement.

    ``evaluate`` returns the sum over all levels.  ``refine(mark)`` moves the
    marked cells of the current finest level into a new (or existing) finer
    level while preserving the represented field (HB transfer).  Set
    ``transfer_mode="thb"`` to transfer the full prolongation instead and
    switch ``evaluate`` to truncated basis functions (``evaluate_thb``),
    which keeps refinement exact while removing the coarse support that
    overlaps finer levels.
    """

    def __init__(
        self,
        base_grid_size: int,
        spline_degree: int,
        bounds: Bounds,
        channels: int | Sequence[int],
        *,
        max_levels: int = 1,
        transfer_mode: str = "hb",
        transition_compute_dtype: str = "fp32",
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        structure_owner: "HierarchicalBSplineField" | None = None,
    ) -> None:
        super().__init__()
        if max_levels < 1:
            raise ValueError("max_levels must be >= 1.")
        if transfer_mode not in ("hb", "thb"):
            raise ValueError("transfer_mode must be 'hb' or 'thb'.")
        _TRANSITION_DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
        if transition_compute_dtype not in _TRANSITION_DTYPES:
            raise ValueError(
                f"transition_compute_dtype must be one of {sorted(_TRANSITION_DTYPES)}."
            )
        self.spline_degree = _coerce_spline_degree(spline_degree)
        # Per-level channel counts.  A plain int broadcasts to all levels.
        if isinstance(channels, (list, tuple)):
            self.channels_per_level = [int(c) for c in channels]
        else:
            self.channels_per_level = [int(channels)] * max_levels
        # Pad with the last value if fewer entries than max_levels.
        while len(self.channels_per_level) < max_levels:
            self.channels_per_level.append(self.channels_per_level[-1])
        self.channels = self.channels_per_level[0]  # level-0 channels (compat)
        self.max_levels = int(max_levels)
        self.transfer_mode = transfer_mode
        # Compute dtype for the THB transition bmm.  "fp32" (default) keeps
        # the exact original behavior; "fp16"/"bf16" gather a half-precision
        # table (halving the dominant memory traffic) and run the bmm in half
        # precision, casting the result back to fp32 for the accumulation.
        self.transition_compute_dtype = transition_compute_dtype
        self._transition_half_dtype = _TRANSITION_DTYPES[transition_compute_dtype]
        dtype = dtype or torch.get_default_dtype()

        bounds_t = _as_float_tensor(bounds, dtype=dtype, device=device)
        self.register_buffer("_bounds", bounds_t)
        self.register_buffer(
            "_transitions",
            build_transition_tables(self.spline_degree, dtype=dtype, device=device),
        )
        if self._transition_half_dtype is not None:
            self.register_buffer(
                "_transitions_half",
                self._transitions.to(self._transition_half_dtype),
                # Derived deterministically from ``_transitions``; keep it out
                # of the state dict so resuming fp32-era checkpoints with a
                # strict load does not fail on the missing key.
                persistent=False,
            )

        # Keep the owner reference as a plain attribute, not a PyTorch submodule,
        # so the sharing hierarchy's state dict does not duplicate the owner's
        # buffers when it is saved/loaded as part of the same model.
        object.__setattr__(self, "structure_owner", structure_owner)
        # Owned structural buffers.  When sharing, this stays empty and we
        # reference the owner's structures.
        self._structures = nn.ModuleList()

        if structure_owner is None:
            structure0 = LevelStructure(
                base_grid_size, self.spline_degree, bounds_t,
                dtype=dtype, device=device,
            )
            self._structures.append(structure0)
        else:
            if structure_owner.spline_degree != self.spline_degree:
                raise ValueError("structure_owner must have the same spline_degree.")
            structure0 = structure_owner._structures[0]

        level0 = SparseBSplineLevel(channels=self._channels_at(0), structure=structure0)
        self.levels = nn.ModuleList([level0])

    # -- structure helpers ----------------------------------------------------

    def _channels_at(self, level: int) -> int:
        """Channel count for the given level (supports per-level dims)."""
        idx = min(level, len(self.channels_per_level) - 1)
        return self.channels_per_level[idx]

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def grid_size_at(self, level: int) -> int:
        G = self.levels[0].grid_size
        for _ in range(level):
            G = 2 * G - self.spline_degree
        return G

    def finest_grid_size(self) -> int:
        return self.grid_size_at(self.num_levels - 1)

    # -- evaluation -------------------------------------------------------------

    def evaluate(self, points: Any) -> Tensor:
        if self.transfer_mode == "thb":
            return self.evaluate_thb(points)
        out = None
        for level in self.levels:
            value = level.evaluate(points)
            out = value if out is None else out + value
        return out

    def evaluate_per_level(self, points: Any) -> Tensor:
        """Evaluate each level separately and return concatenated features.

        Used when levels have different channel counts (per-level feature
        dims).  Returns a single tensor of shape (..., sum(channels)) with
        each level's contribution concatenated along the last axis.
        Levels that don't exist yet contribute zeros so the output dimension
        is always sum(channels_per_level[:max_levels]).
        Only valid for HB mode (THB truncation mixes levels).
        """
        if self.transfer_mode == "thb":
            raise NotImplementedError(
                "evaluate_per_level is not supported with THB transfer."
            )
        parts = []
        for l in range(self.max_levels):
            if l < self.num_levels:
                parts.append(self.levels[l].evaluate(points))
            else:
                ch = self._channels_at(l)
                shape = points.shape[:-1] + (ch,)
                parts.append(torch.zeros(
                    shape, dtype=points.dtype, device=points.device
                ))
        return torch.cat(parts, dim=-1)

    def evaluate_thb(self, points: Any) -> Tensor:
        """Evaluate the sum of *truncated* basis functions (THB).

        For every level the (p+1)^3 support-slot values ``V^l`` of the
        truncated basis obey the fine-to-coarse recursion

            V^l = A_pi^T (V^{l+1} * [slot not in R^{l+1}]),   V^{L-1} = b^{L-1},

        where ``A_pi`` is the per-parity transition table (see
        ``build_transition_tables``), the parity pi = c' - 2c in {0,1}^3
        relates the cell indices of adjacent levels, and R^{l+1} is the set
        of fine basis functions whose support lies fully inside the finer
        cumulative domain (``SparseBSplineLevel.rmask``).  Each level
        contributes ``sum_slots values[slot] * V^l[slot]`` over its active
        (slot >= 0) basis functions.  With the full-prolongation transfer
        (``transfer_mode="thb"``) the represented field is preserved exactly
        under refinement.  A single level degenerates to plain evaluation.
        """
        flat, output_shape = self.levels[0]._prepare_points(points)
        Np = flat.shape[0]
        S = (self.spline_degree + 1) ** 3
        cps: list[Tensor] = []
        weights: list[Tensor] = []
        uvs: list[Tensor] = []
        for lv in self.levels:
            local = (flat - lv._lower) / lv.step
            upper = torch.nextafter(
                torch.full((3,), float(lv.cell_count), dtype=local.dtype, device=local.device),
                torch.zeros(3, dtype=local.dtype, device=local.device),
            )
            local = torch.minimum(torch.maximum(local, torch.zeros_like(local)), upper)
            ijk = torch.floor(local).to(torch.long)
            uvw = local - ijk.to(local.dtype)
            basis_x = basis_1d(uvw[:, 0], degree=lv.spline_degree)
            basis_y = basis_1d(uvw[:, 1], degree=lv.spline_degree)
            basis_z = basis_1d(uvw[:, 2], degree=lv.spline_degree)
            G2 = lv.grid_size * lv.grid_size
            base = ijk[:, 0] * G2 + ijk[:, 1] * lv.grid_size + ijk[:, 2]
            cps.append(base[:, None] + lv._support_offsets_flat[None, :])  # (Np, S)
            weights.append(
                basis_x.index_select(1, lv._support_axis_r)
                * basis_y.index_select(1, lv._support_axis_s)
                * basis_z.index_select(1, lv._support_axis_t)
            )  # (Np, S)
            uvs.append(uvw)

        out = flat.new_zeros(Np, self.channels)
        V = weights[-1]
        last = self.num_levels - 1
        for l in range(last, -1, -1):
            if l < last:
                fine = self.levels[l + 1]
                keep = (~fine.rmask.reshape(-1))[cps[l + 1]]  # (Np, S) bool
                W = V * keep.to(V.dtype)
                # Parity between level l and l+1: c' = 2c + pi, and
                # pi = floor(2 * uvw^l) per axis.
                pi = torch.floor(2.0 * uvs[l]).to(torch.long).clamp_(0, 1)
                case = pi[:, 0] * 4 + pi[:, 1] * 2 + pi[:, 2]
                # Vectorized per-point transition: gather the (S, S) table
                # for each point's parity case and apply it with a single
                # bmm.  The previous per-case loop branched on `rows.any()`
                # and used masked assignments — each branch forced a device
                # sync (aten::nonzero), which dominated the per-iteration
                # fixed overhead.
                half = self._transition_half_dtype
                if half is None:
                    T = self._transitions.index_select(0, case)  # (Np, S, S)
                    V = torch.bmm(W.unsqueeze(1), T.transpose(1, 2)).squeeze(1)
                else:
                    # Half-precision transition: halves the table-gather and
                    # bmm traffic; the result is cast back to fp32 for the
                    # accumulation.  CUDA-graph safe (plain ops, no sync).
                    T = self._transitions_half.index_select(0, case)  # (Np, S, S)
                    V = torch.bmm(W.to(half).unsqueeze(1), T.transpose(1, 2))
                    V = V.squeeze(1).to(W.dtype)
            lv = self.levels[l]
            if lv.values.shape[0] == 0:
                continue
            slots = lv.index_grid.reshape(-1)[cps[l]]
            valid = slots >= 0
            gathered = lv.values.index_select(0, slots.clamp_min(0).reshape(-1))
            gathered = gathered.reshape(Np, S, self.channels)
            out = out + (gathered * (V * valid.to(V.dtype)).unsqueeze(-1)).sum(dim=1)
        return out.reshape(output_shape + (self.channels,))

    def evaluate_gradient(self, points: Any) -> Tensor:
        flat, output_shape = self.levels[0]._prepare_points(points)
        pts = flat.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            values = self.evaluate(pts)  # (Np, C)
            grads = []
            for c in range(self.channels):
                g = torch.autograd.grad(values[:, c].sum(), pts, create_graph=True)[0]
                grads.append(g)
            grad = torch.stack(grads, dim=1)
        return grad.reshape(output_shape + (self.channels, 3))

    def evaluate_with_deriv(self, points: Any) -> tuple[Tensor, Tensor, Tensor]:
        """解析求值：返回 (values (...,C), grads (...,C,3), hess_diag (...,C,3))。

        HB 模式逐级解析求导后求和；THB 模式走 ``evaluate_thb_with_deriv``
        （截断递推与求导可交换，10 组权重共用同一个线性递推）。
        """
        if self.transfer_mode == "thb":
            values, grads, hess_diag, _ = self.evaluate_thb_with_deriv(points)
            return values, grads, hess_diag
        values = grads = hess = None
        for level in self.levels:
            v, g, h = level.evaluate_with_deriv(points)
            values = v if values is None else values + v
            grads = g if grads is None else grads + g
            hess = h if hess is None else hess + h
        return values, grads, hess

    def evaluate_with_full_hessian(self, points: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """解析求值：返回 (values, grads, hess_diag, hess_off)。

        与 ``evaluate_with_deriv`` 相同，但额外返回非对角 Hessian
        (∂²f/∂xy, ∂²f/∂xz, ∂²f/∂yz)，用于 Mean Curvature 计算。
        """
        if self.transfer_mode == "thb":
            return self.evaluate_thb_with_deriv(points)
        raise NotImplementedError(
            "Full Hessian is only implemented for THB mode."
        )

    def evaluate_thb_with_deriv(self, points: Any) -> tuple[Tensor, Tensor, Tensor]:
        """THB 模式下的解析 value/gradient/对角 Hessian。

        与 ``evaluate_thb`` 相同的 fine-to-coarse 截断递推，但携带的 V
        变成 (Np, S, 7)：(value, gx, gy, gz, hxx, hyy, hzz) 七组权重。
        截断递推 V^l = A_pi^T (V^{l+1} * keep) 中 A_pi 与 keep 对查询点是
        分段常数，所以梯度/Hessian 权重组走同一个线性递推即可。
        各级 gather 只做一次，贡献用一次 einsum 累加。
        """
        flat, output_shape = self.levels[0]._prepare_points(points)
        Np = flat.shape[0]
        S = (self.spline_degree + 1) ** 3
        cps: list[Tensor] = []
        weights7: list[Tensor] = []
        uvs: list[Tensor] = []
        for lv in self.levels:
            local = (flat - lv._lower) / lv.step
            upper = torch.nextafter(
                torch.full((3,), float(lv.cell_count), dtype=local.dtype, device=local.device),
                torch.zeros(3, dtype=local.dtype, device=local.device),
            )
            local = torch.minimum(torch.maximum(local, torch.zeros_like(local)), upper)
            ijk = torch.floor(local).to(torch.long)
            uvw = local - ijk.to(local.dtype)
            basis_x, basis_dx, basis_d2x = basis_1d_with_deriv(uvw[:, 0], degree=lv.spline_degree)
            basis_y, basis_dy, basis_d2y = basis_1d_with_deriv(uvw[:, 1], degree=lv.spline_degree)
            basis_z, basis_dz, basis_d2z = basis_1d_with_deriv(uvw[:, 2], degree=lv.spline_degree)
            G2 = lv.grid_size * lv.grid_size
            base = ijk[:, 0] * G2 + ijk[:, 1] * lv.grid_size + ijk[:, 2]
            cps.append(base[:, None] + lv._support_offsets_flat[None, :])  # (Np, S)
            r = lv._support_axis_r
            s = lv._support_axis_s
            t = lv._support_axis_t
            bx = basis_x.index_select(1, r)
            by = basis_y.index_select(1, s)
            bz = basis_z.index_select(1, t)
            # 每级用自己的 step 做链式因子 1/step 与 1/step^2。
            inv_step = 1.0 / lv.step
            inv_step2 = inv_step * inv_step
            # 非对角 Hessian 的链式因子是 1/(step_i * step_j)。
            weights7.append(
                torch.stack(
                    (
                        bx * by * bz,
                        basis_dx.index_select(1, r) * by * bz * inv_step[0],
                        bx * basis_dy.index_select(1, s) * bz * inv_step[1],
                        bx * by * basis_dz.index_select(1, t) * inv_step[2],
                        basis_d2x.index_select(1, r) * by * bz * inv_step2[0],
                        bx * basis_d2y.index_select(1, s) * bz * inv_step2[1],
                        bx * by * basis_d2z.index_select(1, t) * inv_step2[2],
                        # Off-diagonal Hessian: ∂²f/∂xy, ∂²f/∂xz, ∂²f/∂yz
                        basis_dx.index_select(1, r) * basis_dy.index_select(1, s) * bz
                        * inv_step[0] * inv_step[1],
                        basis_dx.index_select(1, r) * by * basis_dz.index_select(1, t)
                        * inv_step[0] * inv_step[2],
                        bx * basis_dy.index_select(1, s) * basis_dz.index_select(1, t)
                        * inv_step[1] * inv_step[2],
                    ),
                    dim=-1,
                )
            )  # (Np, S, 10)
            uvs.append(uvw)

        out7 = flat.new_zeros(Np, self.channels, 10)
        V = weights7[-1]  # (Np, S, 10)
        last = self.num_levels - 1
        for l in range(last, -1, -1):
            if l < last:
                fine = self.levels[l + 1]
                keep = (~fine.rmask.reshape(-1))[cps[l + 1]]  # (Np, S) bool
                W = V * keep.to(V.dtype).unsqueeze(-1)  # (Np, S, 10)
                # 与 evaluate_thb 相同的 parity/transition，按 10 组批量 bmm。
                pi = torch.floor(2.0 * uvs[l]).to(torch.long).clamp(0, 1)
                case = pi[:, 0] * 4 + pi[:, 1] * 2 + pi[:, 2]
                half = self._transition_half_dtype
                if half is None:
                    T = self._transitions.index_select(0, case)  # (Np, S, S)
                    Wp = W.permute(0, 2, 1)  # (Np, 10, S)
                    V = torch.bmm(Wp, T.transpose(1, 2)).permute(0, 2, 1)  # (Np, S, 10)
                else:
                    # 半精度转移（gather/bmm 流量减半），输出 cast 回 fp32。
                    T = self._transitions_half.index_select(0, case)  # (Np, S, S)
                    Wp = W.permute(0, 2, 1).to(half)  # (Np, 10, S)
                    V = torch.bmm(Wp, T.transpose(1, 2)).permute(0, 2, 1).to(W.dtype)
            lv = self.levels[l]
            if lv.values.shape[0] == 0:
                continue
            slots = lv.index_grid.reshape(-1)[cps[l]]
            valid = slots >= 0
            gathered = lv.values.index_select(0, slots.clamp_min(0).reshape(-1))
            gathered = gathered.reshape(Np, S, self.channels)
            Wv = V * valid.to(V.dtype).unsqueeze(-1)  # (Np, S, 10)
            out7 = out7 + torch.einsum("nsc,nsg->ncg", gathered, Wv)

        values = out7[..., 0]
        grads = out7[..., 1:4]
        hess_diag = out7[..., 4:7]
        hess_off = out7[..., 7:10]
        return (
            values.reshape(output_shape + (self.channels,)),
            grads.reshape(output_shape + (self.channels, 3)),
            hess_diag.reshape(output_shape + (self.channels, 3)),
            hess_off.reshape(output_shape + (self.channels, 3)),
        )

    # -- checkpoint structure restore -----------------------------------------

    @torch.no_grad()
    def restore_levels_from_state_dict(self, state_dict: dict, prefix: str) -> int:
        """Grow/rebuild the hierarchy to match a checkpoint's state dict.

        ``load_state_dict(strict=False)`` silently drops levels that do not
        exist in the current (freshly constructed) field, which loses all
        refined structure when loading a post-refinement checkpoint into a
        new model.  Call this *before* loading: it appends any missing
        levels and copies the structural buffers (``region``, ``index_grid``,
        ``omega``, ``rmask``) and ``values`` so that the subsequent state-dict
        load finds matching keys and shapes.  ``prefix`` is the key prefix up
        to and including the field (e.g. ``"module.neural_sdf.hier_field."``).

        Supports both old-style keys ``levels.{i}.{buffer}`` and new-style
        keys ``_structures.{i}.{buffer}``.

        Returns the number of levels found in the checkpoint (0 = none).
        """
        # Detect checkpoint style and level count.
        def _structure_key(level_idx: int, name: str) -> str:
            old_key = f"{prefix}levels.{level_idx}.{name}"
            if old_key in state_dict:
                return old_key
            return f"{prefix}_structures.{level_idx}.{name}"

        idx = 0
        while (
            _structure_key(idx, "region") in state_dict
            or f"{prefix}levels.{idx}.values" in state_dict
        ):
            idx += 1
        n_ckpt = idx
        if n_ckpt == 0:
            return 0
        device = self.levels[0].values.device
        dtype = self.levels[0].values.dtype

        # Grow to the required number of levels.
        while self.num_levels < n_ckpt:
            lvl_idx = self.num_levels
            if self.structure_owner is None:
                structure = LevelStructure(
                    self.grid_size_at(lvl_idx), self.spline_degree, self._bounds,
                    dtype=dtype, device=device,
                )
                self._structures.append(structure)
            else:
                structure = self.structure_owner._structures[lvl_idx]
            self.levels.append(
                SparseBSplineLevel(channels=self._channels_at(lvl_idx), structure=structure)
            )

        # Restore structural buffers and values.
        # When sharing structure with another hierarchy, the owner is
        # responsible for restoring the structural tensors; we only need to
        # restore our own coefficient values and grow our level list.
        for i in range(self.num_levels):
            lvl = self.levels[i]
            if self.structure_owner is None:
                for name in ("region", "index_grid", "omega", "rmask"):
                    key = _structure_key(i, name)
                    if key in state_dict:
                        setattr(lvl, name, state_dict[key].to(device).clone())
            # Old-style checkpoints store values under levels.{i}.values.
            vkey = f"{prefix}levels.{i}.values"
            if vkey in state_dict:
                lvl.values = nn.Parameter(
                    state_dict[vkey].to(device=device, dtype=dtype).clone()
                )
        # Older checkpoints may lack omega/rmask: recompute from the regions.
        if self.structure_owner is None and _structure_key(0, "omega") not in state_dict:
            omegas = self._omegas()
            for i, lv in enumerate(self.levels):
                lv.update_omega(omegas[i])
        return n_ckpt

    # -- refinement ---------------------------------------------------------------

    def _omegas(self) -> list[Tensor]:
        """Cumulative responsibility domain per level: omega_l = region_l | Omega_{l+1}."""
        omegas = [None] * self.num_levels
        omega = None
        for idx in range(self.num_levels - 1, -1, -1):
            region = self.levels[idx].region
            omega = region if omega is None else (region | parent_cells(omega))
            omegas[idx] = omega
        return omegas

    def _active_mask_at(self, level_idx: int, omegas: list[Tensor]) -> Tensor:
        """HB selection: support inside Omega^l but not fully inside Omega^{l+1}."""
        p = self.spline_degree
        active = basis_support_mask(omegas[level_idx], p)
        if level_idx + 1 < self.num_levels:
            refined = basis_support_mask(parent_cells(omegas[level_idx + 1]), p)
            active = active & ~refined
        return active

    @torch.no_grad()
    def refine(
        self,
        mark_fn: Callable[[int, Tensor], Tensor],
        *,
        owner_refine_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Refine the current finest level.

        ``mark_fn(level_index, corner_points)`` receives the finest level
        index and its ``((cc+1)^3, 3)`` lattice points, and must return a
        boolean ``(cc, cc, cc)`` tensor marking cells to refine.  Only cells
        currently in the level's region are considered.

        Returns a small info dict; a no-op when already at ``max_levels``.
        """
        if self.structure_owner is None:
            return self._refine_owner(mark_fn)
        return self._refine_shared(owner_refine_info)

    @torch.no_grad()
    def _refine_owner(
        self,
        mark_fn: Callable[[int, Tensor], Tensor],
    ) -> dict[str, Any]:
        """Normal refine path for the structure-owning hierarchy."""
        level_idx = self.num_levels - 1
        if level_idx + 1 >= self.max_levels:
            return {"refined": False, "reason": "max_levels reached"}
        level = self.levels[level_idx]
        corners = level.cell_corner_points()
        marked = mark_fn(level_idx, corners)
        marked = torch.as_tensor(marked, dtype=torch.bool, device=level.region.device)
        if tuple(marked.shape) != (level.cell_count,) * 3:
            raise ValueError("mark_fn must return a (cc, cc, cc) boolean tensor.")
        marked = marked & level.region
        num_marked = int(marked.sum().item())
        if num_marked == 0:
            return {"refined": False, "reason": "no cells marked"}

        p = self.spline_degree
        new_region_coarse = level.region & ~marked
        child_region = children_region(marked)

        # Snapshot pre-refinement topology so structure-sharing hierarchies
        # (e.g. the color field sharing the SDF grid) can compute their own
        # coefficient transfers.
        old_rmasks = [lv.rmask.clone() for lv in self.levels]
        old_index_grid = level.index_grid.clone()

        # Coefficient transfer.
        #   HB:  only coarse basis functions whose support is fully marked are
        #        removed; their coefficients are transferred to their children.
        #   THB: transfer the exact fine-level coefficients of the whole
        #        represented field via the shadow chain (see
        #        ``_thb_full_transfer``); exact with truncated evaluation.
        if self.transfer_mode == "thb":
            transfer = self._thb_full_transfer(level_idx)
        else:
            transfer_parents = basis_support_mask(marked, p)
            transfer = self._transfer_coefficients(
                level, transfer_parents, child_region.shape[0] + p,
                target_channels=self._channels_at(level_idx + 1),
            )

        # Create or extend the fine level.  Refining the finest level always
        # creates a new one; the extend branch is kept for future
        # re-refinement of coarser levels.
        if level_idx + 1 < self.num_levels:
            fine = self.levels[level_idx + 1]
            fine_region = fine.region | child_region
            fine_transfer = self._pad_transfer_to_level(transfer, fine.grid_size)
            # Finest-level active mask = support inside its region.
            fine_active = basis_support_mask(fine_region, p)
            fine.rebuild(fine_region, fine_active, fine_transfer)
        else:
            fine_structure = LevelStructure(
                self.grid_size_at(level_idx + 1), p, self._bounds,
                region=child_region, dtype=level.values.dtype,
                device=level.values.device,
            )
            self._structures.append(fine_structure)
            fine = SparseBSplineLevel(
                channels=self._channels_at(level_idx + 1), structure=fine_structure,
            )
            fine.rebuild(child_region, basis_support_mask(child_region, p), transfer,
                         carry_over=False)
            self.levels.append(fine)

        # Shrink the coarse level: keep every basis function whose support is
        # NOT fully inside the marked set (straddling basis stay active).
        removed = basis_support_mask(marked, p)
        coarse_active = (level.index_grid >= 0) & ~removed
        level.rebuild(new_region_coarse, coarse_active)

        # Refresh cumulative domains (omega / rmask) for THB truncation.
        omegas = self._omegas()
        for idx, lv in enumerate(self.levels):
            lv.update_omega(omegas[idx])

        owner_info = {
            "level_idx": level_idx,
            "marked": marked,
            "child_region": child_region,
            "new_region_coarse": new_region_coarse,
            "old_index_grid": old_index_grid,
            "old_rmasks": old_rmasks,
        }
        return {
            "refined": True,
            "level": level_idx + 1,
            "marked_cells": num_marked,
            "fine_active": self.levels[level_idx + 1].num_active,
            "owner_refine_info": owner_info,
        }

    @torch.no_grad()
    def _refine_shared(
        self,
        owner_refine_info: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Refine path for a hierarchy that shares structure with its owner.

        The owner has already updated the structural buffers, so we only need
        to compute the coefficient transfer for our own channels and update
        the ``values`` parameters accordingly.
        """
        if owner_refine_info is None:
            raise ValueError("A structure-sharing hierarchy requires owner_refine_info.")
        if self.structure_owner is None:
            raise RuntimeError("_refine_shared called on a non-sharing hierarchy.")

        level_idx = owner_refine_info["level_idx"]
        if level_idx + 1 >= self.max_levels:
            return {"refined": False, "reason": "max_levels reached"}

        level = self.levels[level_idx]
        marked = owner_refine_info["marked"]
        child_region = owner_refine_info["child_region"]
        new_region_coarse = owner_refine_info["new_region_coarse"]
        old_index_grid = owner_refine_info["old_index_grid"]
        old_rmasks = owner_refine_info["old_rmasks"]
        num_marked = int(marked.sum().item())
        p = self.spline_degree

        # Coefficient transfer using the owner's pre-refinement topology.
        if self.transfer_mode == "thb":
            transfer = self._thb_full_transfer(level_idx, old_rmasks=old_rmasks)
        else:
            transfer_parents = basis_support_mask(marked, p)
            transfer = self._transfer_coefficients(
                level, transfer_parents, child_region.shape[0] + p,
                old_index_grid=old_index_grid,
                target_channels=self._channels_at(level_idx + 1),
            )

        # Create or extend the fine level sharing the owner's new structure.
        if level_idx + 1 < self.num_levels:
            fine = self.levels[level_idx + 1]
            fine_transfer = self._pad_transfer_to_level(transfer, fine.grid_size)
            fine_active = fine.index_grid >= 0
            fine.values = nn.Parameter(fine_transfer[fine_active].contiguous())
        else:
            owner_fine_structure = self.structure_owner._structures[level_idx + 1]
            fine = SparseBSplineLevel(
                channels=self._channels_at(level_idx + 1), structure=owner_fine_structure,
            )
            fine_active = fine.index_grid >= 0
            fine.values = nn.Parameter(transfer[fine_active].contiguous())
            self.levels.append(fine)

        # Shrink the coarse level values.  The structure has already been
        # updated by the owner, so ``level.index_grid`` is the *new* topology.
        # We must map old active coefficients through the owner's pre-refinement
        # index grid; ``rebuild(..., carry_over=True)`` would use the new grid
        # as both source and destination, so we rebuild the dense old value grid
        # and reindex into the new active set directly.
        coarse_active = level.index_grid >= 0
        old_index_grid = owner_refine_info["old_index_grid"]
        G = level.grid_size
        C_coarse = level.values.shape[-1]
        old_values_dense = level.values.new_zeros(G, G, G, C_coarse)
        old_mask = old_index_grid >= 0
        if old_mask.any():
            old_values_dense[old_mask] = level.values.detach()
        new_num_active = int(coarse_active.sum().item())
        new_coarse_values = level.values.new_zeros(new_num_active, C_coarse)
        if new_num_active > 0:
            new_coarse_values[:] = old_values_dense[coarse_active]
        level.values = nn.Parameter(new_coarse_values)

        return {
            "refined": True,
            "level": level_idx + 1,
            "marked_cells": num_marked,
            "fine_active": self.levels[level_idx + 1].num_active,
        }

    # -- re-banding (dynamic finest-level re-evaluation) -------------------------

    @torch.no_grad()
    def reband_finest(
        self,
        band: float,
        max_fraction: float | None = None,
        include_centers: bool = True,
    ) -> dict[str, Any]:
        """Re-evaluate the finest level's region based on current field values.

        Adds cells near the surface that are missing, removes cells far from
        it.  For THB the transfer preserves the field exactly; for HB new
        cells start at zero (the coarser levels still contribute additively).

        Returns a dict with ``changed``, ``added``, ``removed``, and
        ``old_index_grid`` (needed by structure-sharing hierarchies to remap
        their own values).
        """
        if self.num_levels < 2:
            return {"changed": False, "reason": "need at least 2 levels"}

        level_idx = self.num_levels - 1
        level = self.levels[level_idx]
        p = self.spline_degree

        # Evaluate current field at all finest-level cell corners.
        corners = level.cell_corner_points()
        new_region = mark_cells_by_sdf_band(
            self, level_idx, corners, float(band),
            include_centers=include_centers, max_fraction=max_fraction,
        )

        # Constrain by parent region (ancestor constraint: finest can only
        # cover cells whose parent is in the parent level's region).
        parent_expanded = children_region(self.levels[level_idx - 1].region)
        new_region &= parent_expanded

        if torch.equal(new_region, level.region):
            return {"changed": False}

        to_add = int((new_region & ~level.region).sum())
        to_remove = int((level.region & ~new_region).sum())

        # Snapshot for structure-sharing hierarchies.
        old_index_grid = level.index_grid.clone()

        # For THB: compute exact fine-level coefficients of the current field
        # so newly added cells get the correct initialization (the coarse
        # basis functions get truncated in the newly covered areas).
        transfer = None
        if self.transfer_mode == "thb":
            transfer = self._thb_full_transfer(level_idx - 1)

        # Rebuild the finest level with the new region.
        new_active = basis_support_mask(new_region, p)
        level.rebuild(new_region, new_active, transferred_values=transfer,
                      carry_over=True)

        # Update THB cumulative-domain buffers.
        if self.transfer_mode == "thb":
            omegas = self._omegas()
            for idx, lv in enumerate(self.levels):
                lv.update_omega(omegas[idx])

        return {
            "changed": True,
            "added": to_add,
            "removed": to_remove,
            "old_index_grid": old_index_grid,
            "level_idx": level_idx,
        }

    @torch.no_grad()
    def prune_finest(self, keep_mask: torch.Tensor) -> dict[str, Any]:
        """Shrink the finest level's region to ``region & keep_mask``.

        Used for activity-based pruning: cells that never received any
        refinement-marking signal (e.g. the interior "bubble web" no camera
        ray ever crosses) are deactivated, falling back to the coarser
        level's smoother field.  For THB the transfer preserves the field on
        the retained domain.

        ``keep_mask`` is a boolean (cc, cc, cc) tensor on the finest level's
        cell lattice.  Returns the same info dict as :meth:`reband_finest`
        so structure-sharing hierarchies can remap via
        :meth:`sync_from_owner_reband`.
        """
        if self.num_levels < 2:
            return {"changed": False, "reason": "need at least 2 levels"}

        level_idx = self.num_levels - 1
        level = self.levels[level_idx]
        p = self.spline_degree

        new_region = level.region & keep_mask.to(level.region.device)
        if torch.equal(new_region, level.region):
            return {"changed": False}

        to_remove = int((level.region & ~new_region).sum())
        old_index_grid = level.index_grid.clone()

        transfer = None
        if self.transfer_mode == "thb":
            transfer = self._thb_full_transfer(level_idx - 1)

        new_active = basis_support_mask(new_region, p)
        if not bool(new_active.any()):
            # An empty active set disconnects this level's values from the
            # autograd graph (evaluation skips empty levels), which breaks
            # CUDA-graph capture and freezes the level.  Refuse to prune.
            return {"changed": False, "reason": "prune would empty the level"}
        level.rebuild(new_region, new_active, transferred_values=transfer,
                      carry_over=True)

        if self.transfer_mode == "thb":
            omegas = self._omegas()
            for idx, lv in enumerate(self.levels):
                lv.update_omega(omegas[idx])

        return {
            "changed": True,
            "added": 0,
            "removed": to_remove,
            "old_index_grid": old_index_grid,
            "level_idx": level_idx,
        }

    @torch.no_grad()
    def sync_from_owner_reband(self, reband_info: dict[str, Any]) -> dict[str, Any]:
        """Remap values after the owner hierarchy re-banded the finest level.

        The shared LevelStructure has already been updated by the owner;
        this level's ``values`` parameter still references the old index
        grid.  We rebuild the dense value grid from the old topology and
        reindex into the new active set.  New cells get zero (HB mode:
        coarser levels still contribute, so zero is the correct init).
        """
        if not reband_info.get("changed"):
            return {"changed": False}
        level_idx = reband_info["level_idx"]
        old_index_grid = reband_info["old_index_grid"]
        level = self.levels[level_idx]
        C = level.values.shape[-1]
        G = level.grid_size

        # Save current values into a dense grid using the OLD topology.
        old_mask = old_index_grid >= 0
        old_values_dense = level.values.new_zeros(G, G, G, C)
        if old_mask.any():
            old_values_dense[old_mask] = level.values.detach()

        # Read out using the NEW (already updated) index grid.
        new_active = level.index_grid >= 0
        new_num = int(new_active.sum().item())
        new_values = level.values.new_zeros(new_num, C)
        if new_num > 0:
            new_values[:] = old_values_dense[new_active]
        level.values = nn.Parameter(new_values)

        return {
            "changed": True,
            "added": reband_info["added"],
            "removed": reband_info["removed"],
        }

    # -- full-hierarchy re-banding (all levels) ---------------------------------

    @torch.no_grad()
    def reband_all(
        self,
        bands: list[float],
        max_fractions: list[float] | None = None,
    ) -> dict[str, Any]:
        """Re-evaluate ALL level regions based on current field values.

        Analogous to 3DGS's split-and-prune: each level's region is
        recomputed from the current SDF, adding cells near the surface and
        removing cells far from it, at every level of the hierarchy.

        Phase 1 computes new regions for all levels using the current field
        (avoiding circular dependencies).  Phase 2 applies changes from
        coarse to fine, updating THB truncation buffers after each level so
        subsequent transfers use the latest structure.

        Coefficient continuity:
        - THB (SDF): exact transfer from the parent for newly added cells;
          removed cells are absorbed by the coarser levels via truncation.
        - HB (color): new cells start at zero; removed cells' contribution
          is lost (acceptable since HB sums levels additively).

        ``bands`` has one entry per level.  ``max_fractions`` optionally caps
        the fraction of cells that can be active at each level.

        Returns a dict with per-level add/remove counts and the old index
        grids (needed by structure-sharing hierarchies to remap values).
        """
        n = self.num_levels
        if n < 2:
            return {"changed": False, "reason": "need at least 2 levels"}
        if len(bands) < n:
            bands = list(bands) + [bands[-1]] * (n - len(bands))

        # Phase 1: compute new regions for all levels using the CURRENT field.
        new_regions = []
        for l in range(n):
            level = self.levels[l]
            corners = level.cell_corner_points()
            mf = max_fractions[l] if max_fractions else None
            region = mark_cells_by_sdf_band(
                self, l, corners, float(bands[l]),
                include_centers=True, max_fraction=mf,
            )
            # Ancestor constraint: can only cover cells whose parent is in
            # the parent level's (new) region.
            if l > 0:
                parent_expanded = children_region(new_regions[l - 1])
                region &= parent_expanded
            new_regions.append(region)

        # Phase 2: apply changes from coarse to fine.
        old_index_grids = [lv.index_grid.clone() for lv in self.levels]
        per_level_stats = []
        any_changed = False

        for l in range(n):
            level = self.levels[l]
            if torch.equal(new_regions[l], level.region):
                per_level_stats.append({"level": l, "added": 0, "removed": 0})
                continue

            added = int((new_regions[l] & ~level.region).sum())
            removed = int((level.region & ~new_regions[l]).sum())
            any_changed = True

            # Compute transfer for newly added cells (THB only).
            transfer = None
            if self.transfer_mode == "thb" and l > 0 and added > 0:
                transfer = self._thb_full_transfer(l - 1)

            # Rebuild values and structure.  ``rebuild`` handles updating
            # the shared LevelStructure (region, index_grid) itself — calling
            # ``update_region_and_masks`` first would clobber the old
            # index_grid that ``rebuild`` needs for carry-over.
            new_active = basis_support_mask(new_regions[l], self.spline_degree)
            level.rebuild(
                new_regions[l], new_active,
                transferred_values=transfer, carry_over=True,
            )

            # Update THB truncation buffers so the next level's transfer
            # uses the latest structure.
            if self.transfer_mode == "thb":
                omegas = self._omegas()
                for idx, lv in enumerate(self.levels):
                    lv.update_omega(omegas[idx])

            per_level_stats.append({"level": l, "added": added, "removed": removed})

        return {
            "changed": any_changed,
            "per_level": per_level_stats,
            "old_index_grids": old_index_grids,
        }

    @torch.no_grad()
    def sync_from_owner_reband_all(
        self, reband_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Remap values at ALL levels after the owner's reband_all.

        Like ``sync_from_owner_reband`` but handles every level, not just
        the finest.  Each level's values are remapped from the old index
        grid to the new (already updated) one.
        """
        if not reband_info.get("changed"):
            return {"changed": False}
        old_index_grids = reband_info["old_index_grids"]

        for l, level in enumerate(self.levels):
            old_index_grid = old_index_grids[l]
            old_mask = old_index_grid >= 0
            new_active = level.index_grid >= 0
            new_num = int(new_active.sum().item())
            old_num = int(old_mask.sum().item())
            if new_num == old_num and torch.equal(
                level.index_grid, old_index_grid
            ):
                continue  # no structural change at this level

            C = level.values.shape[-1]
            G = level.grid_size
            old_values_dense = level.values.new_zeros(G, G, G, C)
            if old_mask.any():
                old_values_dense[old_mask] = level.values.detach()
            new_values = level.values.new_zeros(new_num, C)
            if new_num > 0:
                new_values[:] = old_values_dense[new_active]
            level.values = nn.Parameter(new_values)

        return {"changed": True, "per_level": reband_info["per_level"]}

    def _transfer_coefficients(
        self,
        level: SparseBSplineLevel,
        parent_mask: Tensor,
        fine_grid_size: int,
        dense_values: Tensor | None = None,
        old_index_grid: Tensor | None = None,
        target_channels: int | None = None,
    ) -> Tensor:
        """Dense (G_f, G_f, G_f, C_out) transfer grid from marked parent basis.

        With ``dense_values=None`` the parent coefficients are looked up from
        the level's sparse storage (``parent_mask`` is intersected with the
        active set).  Otherwise ``dense_values`` is a dense ``(G, G, G, C)``
        coefficient grid aligned with the level's control points and
        ``parent_mask`` is used as-is (used for the THB shadow chain, whose
        coefficients live on *inactive* control points).

        ``old_index_grid`` supports structure sharing: a referencing hierarchy
        may need the owner's pre-refinement topology to compute its own
        coefficient transfer.

        ``target_channels`` overrides the output channel count.  When the
        target level has more channels than the source, extra channels are
        zero-padded; fewer channels are truncated.  Default: same as source.
        """
        p = self.spline_degree
        mask = subdivision_mask_1d(p, dtype=level.values.dtype, device=level.values.device)
        G_f = fine_grid_size
        index_grid = level.index_grid if old_index_grid is None else old_index_grid
        if dense_values is None:
            parents = parent_mask & (index_grid >= 0)
        else:
            parents = parent_mask
        C_in = level.values.shape[-1] if dense_values is None else dense_values.shape[-1]
        C_out = int(target_channels) if target_channels is not None else C_in
        transfer = level.values.new_zeros((G_f, G_f, G_f, C_out))
        if not parents.any():
            return transfer
        idx = parents.nonzero(as_tuple=False)  # (Np, 3)
        if dense_values is None:
            slots = index_grid[parents]
            vals = level.values.detach()[slots]  # (Np, C_in)
        else:
            vals = dense_values[parents]
        # Pad or truncate channels to match the target level.
        if C_out > C_in:
            vals = torch.cat(
                [vals, vals.new_zeros(vals.shape[0], C_out - C_in)], dim=-1
            )
        elif C_out < C_in:
            vals = vals[:, :C_out]
        G = level.grid_size
        for k0, k1, k2 in product(range(p + 2), repeat=3):
            w = mask[k0] * mask[k1] * mask[k2]
            # Children of coarse basis i are fine indices 2i+k-p (see
            # prolongate_dense for the index convention).
            j0 = 2 * idx[:, 0] + (k0 - p)
            j1 = 2 * idx[:, 1] + (k1 - p)
            j2 = 2 * idx[:, 2] + (k2 - p)
            valid = (j0 >= 0) & (j1 >= 0) & (j2 >= 0) & (j0 < G_f) & (j1 < G_f) & (j2 < G_f)
            if not valid.any():
                continue
            flat = (j0[valid] * G_f + j1[valid]) * G_f + j2[valid]
            transfer.reshape(-1, C_out).index_add_(0, flat, w * vals[valid])
        return transfer

    def _thb_full_transfer(
        self,
        level_idx: int,
        old_rmasks: list[Tensor] | None = None,
    ) -> Tensor:
        """Exact THB transfer grid for refining ``level_idx``.

        Plain prolongation of the refined level's active coefficients is NOT
        exact under truncated evaluation: coarser straddling basis functions
        reach fine basis functions inside the new region through chains that
        cross the R-set boundaries (a level-(l+1) basis inside R^{l+1} can
        have parents outside R^l).  The exact fine coefficients of the
        represented field satisfy the "shadow" recursion

            G^{l+1} = prol(active coeffs of level l) + prol(shadow^l),
            shadow^{l+1} = G^{l+1} on the complement of R^{l+1},

        with shadow^0 = 0.  The recursion is recomputed from the current
        coefficients at every refine (no persistent state); the returned grid
        is G^{level_idx+1}.  See ``evaluate_thb`` for the matching truncation.

        ``old_rmasks`` lets a structure-sharing hierarchy compute the transfer
        using the owner's pre-refinement R-masks.
        """
        shadow: Tensor | None = None
        transfer: Tensor | None = None
        for lam in range(level_idx + 1):
            lv = self.levels[lam]
            rmask = old_rmasks[lam] if old_rmasks is not None else lv.rmask
            G_f = self.grid_size_at(lam + 1)
            all_true = torch.ones(
                (lv.grid_size,) * 3, dtype=torch.bool, device=lv.region.device
            )
            acc = self._transfer_coefficients(lv, all_true, G_f)
            if shadow is not None:
                acc = acc + self._transfer_coefficients(
                    lv, ~rmask, G_f, dense_values=shadow
                )
            if lam < level_idx:
                # Chains must avoid R^{lam+1}: keep only the complement.
                next_rmask = (
                    old_rmasks[lam + 1]
                    if old_rmasks is not None
                    else self.levels[lam + 1].rmask
                )
                shadow = acc * (~next_rmask).unsqueeze(-1)
            transfer = acc
        return transfer

    @staticmethod
    def _pad_transfer_to_level(transfer: Tensor, grid_size: int) -> Tensor:
        G_t = transfer.shape[0]
        if G_t == grid_size:
            return transfer
        if G_t > grid_size:
            return transfer[:grid_size, :grid_size, :grid_size].contiguous()
        out = transfer.new_zeros((grid_size, grid_size, grid_size, transfer.shape[-1]))
        out[:G_t, :G_t, :G_t] = transfer
        return out


# ---------------------------------------------------------------------------
# SDF-band marking helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def mark_cells_by_value_band(
    eval_fn: Callable[[Tensor], Tensor],
    cell_count: int,
    corner_points: Tensor,
    band: float,
    chunk: int = 2_000_000,
    extra_points: Tensor | None = None,
    max_fraction: float | None = None,
    region: Tensor | None = None,
) -> Tensor:
    """Mark cells whose corner values come within ``band`` of zero.

    ``eval_fn`` maps points to ``(..., C)`` values (channel 0 is used);
    evaluation happens in chunks to bound the memory footprint.  A cell is
    marked if any of its 8 corners has ``|value| < band``.  When
    ``extra_points`` (a ``(cc, cc, cc)`` lattice, e.g. cell centers) is
    given, cells whose extra-point value falls within ``band`` are marked
    as well — this catches thin structures that pass through cell interiors
    while all 8 corners lie outside the band.

    ``region`` (optional, ``(cc, cc, cc)`` bool) restricts marking to cells
    inside the parent level's active region.  Applied BEFORE the budget cap
    so that budget slots are not wasted on ineligible cells.

    ``max_fraction`` caps the marked-cell count at that fraction of all
    cells: when the band would mark more (e.g. an early, still-blurry SDF
    whose band shell fills a whole volume), only the ``max_fraction`` cells
    closest to the surface (smallest probed |value|) are kept.  This guards
    the refinement-time memory budget — an exploding marked set degenerates
    the sparse hierarchy into dense grids and can page GPU memory.
    """
    cc = int(cell_count)
    value_chunks = [
        eval_fn(pts)[..., 0] for pts in corner_points.split(chunk, dim=0)
    ]
    values = torch.cat(value_chunks, dim=0).reshape(cc + 1, cc + 1, cc + 1)
    abs_values = values.abs()
    near = abs_values < float(band)
    # Cell (i, j, k) corners: [i:i+2, j:j+2, k:k+2].
    corner_min = torch.minimum(
        torch.minimum(
            torch.minimum(abs_values[:-1, :-1, :-1], abs_values[1:, :-1, :-1]),
            torch.minimum(abs_values[:-1, 1:, :-1], abs_values[1:, 1:, :-1]),
        ),
        torch.minimum(
            torch.minimum(abs_values[:-1, :-1, 1:], abs_values[1:, :-1, 1:]),
            torch.minimum(abs_values[:-1, 1:, 1:], abs_values[1:, 1:, 1:]),
        ),
    )
    marked = corner_min < float(band)
    if extra_points is not None:
        extra_chunks = [
            eval_fn(pts)[..., 0] for pts in extra_points.split(chunk, dim=0)
        ]
        extra_abs = torch.cat(extra_chunks, dim=0).reshape(cc, cc, cc).abs()
        marked |= extra_abs < float(band)
        corner_min = torch.minimum(corner_min, extra_abs)
    # Pre-filter by parent region so budget slots are not wasted on
    # ineligible cells.
    if region is not None:
        marked = marked & region
        corner_min = torch.where(
            region, corner_min, torch.full_like(corner_min, float("inf"))
        )
    if max_fraction is not None:
        budget = int(marked.numel() * float(max_fraction))
        if 0 < budget < int(marked.sum()):
            # Host sync is fine here: marking runs at refinement time,
            # outside the CUDA-graphed training step.  topk enforces the
            # budget strictly (a value threshold would let exact ties —
            # e.g. near-constant early SDF plateaus — defeat the cap).
            keep = torch.topk(corner_min.reshape(-1), budget, largest=False).indices
            marked = torch.zeros(
                corner_min.numel(), dtype=torch.bool, device=corner_min.device
            )
            marked[keep] = True
            marked = marked.reshape(corner_min.shape)
    return marked


@torch.no_grad()
def cell_center_points(level: "SparseBSplineLevel") -> Tensor:
    """Cell-center points of the level's cell grid, shape ``((cc)^3, 3)``."""
    cc = level.cell_count
    axes = [
        level._lower[d] + level.step[d] * (
            torch.arange(cc, dtype=level.step.dtype, device=level.step.device) + 0.5
        )
        for d in range(3)
    ]
    gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)


def mark_cells_by_sdf_band(
    field: HierarchicalBSplineField,
    level_index: int,
    corner_points: Tensor,
    band: float,
    chunk: int = 2_000_000,
    include_centers: bool = False,
    max_fraction: float | None = None,
    region: Tensor | None = None,
) -> Tensor:
    """Mark cells of ``level_index`` whose corner SDF values are within ``band``.

    With ``include_centers=True`` the cell centers are probed as well, so
    thin structures crossing a cell interior (all corners outside the band)
    are still marked.  ``max_fraction`` caps the marked-cell count (cells
    closest to the surface are kept), guarding the refinement-time memory
    budget against band explosions on blurry early geometry.
    ``region`` restricts candidates to the parent level's active region
    before the budget is applied.
    """
    level = field.levels[level_index]
    extra = cell_center_points(level) if include_centers else None
    return mark_cells_by_value_band(
        field.evaluate, level.cell_count, corner_points, band, chunk,
        extra_points=extra, max_fraction=max_fraction, region=region,
    )


# ---------------------------------------------------------------------------
# Loss-guided marking helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def mark_cells_by_loss(
    accum: Tensor,
    count: Tensor,
    max_fraction: float,
    min_count: float = 1.0,
    eps: float = 1e-8,
    region: Tensor | None = None,
) -> Tensor:
    """Mark cells with the highest accumulated render-loss score.

    ``accum`` / ``count`` are ``(cc, cc, cc)`` buffers filled by scattering
    per-sample render error mass into the level's cell grid during training
    (see ``BSplineSDFWrapper.accumulate_loss``).  The per-cell score is the
    mean scattered error mass per visit, ``accum / (count + eps)``, so rarely
    visited cells with high error (thin structures) outrank frequently
    visited cells with moderate error.  Cells with fewer than ``min_count``
    visits score zero — no signal, no refinement.

    ``region`` (optional, ``(cc, cc, cc)`` bool) restricts marking to cells
    inside the parent level's active region, applied BEFORE the budget cap.

    Unlike band marking there is no natural absolute threshold for a loss
    score, so marking is always a strict ``topk`` with budget
    ``max_fraction * numel`` (the same safety valve the band path uses).
    """
    if accum.shape != count.shape or accum.ndim != 3:
        raise ValueError("accum and count must both be (cc, cc, cc) tensors.")
    score = accum / (count + eps)
    score = torch.where(count >= float(min_count), score, torch.zeros_like(score))
    # Pre-filter by parent region so budget slots go only to eligible cells.
    if region is not None:
        score = score * region.to(score.dtype)
    budget = int(score.numel() * float(max_fraction))
    marked = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
    if budget > 0 and bool((score > 0).any()):
        keep = torch.topk(score.reshape(-1), min(budget, int((score > 0).sum())),
                          largest=True).indices
        marked[keep] = True
    return marked.reshape(score.shape)


@torch.no_grad()
def sample_loss_score_at_points(
    accum: Tensor,
    count: Tensor,
    source_level: SparseBSplineLevel,
    points: Tensor,
    min_count: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Sample a loss score defined on ``source_level`` cells at ``points``.

    Used when the SDF and color hierarchies have different cell grids (e.g.
    different spline degrees): the SDF stores the accumulated loss score
    on its own cell lattice, and the color grid samples it at its own corner
    points.  Out-of-bounds or unvisited source cells score zero.
    """
    score = accum / (count + eps)
    score = torch.where(
        count >= float(min_count), score, torch.zeros_like(score)
    )
    cc = source_level.cell_count
    pts = points.reshape(-1, 3).to(source_level.values.dtype)
    idx = ((pts - source_level._lower) / source_level.step).floor().long()
    valid = ((idx >= 0) & (idx < cc)).all(dim=-1)
    idx.clamp_(0, cc - 1)
    flat = (idx[:, 0] * cc + idx[:, 1]) * cc + idx[:, 2]
    out = torch.zeros(points.shape[0], dtype=score.dtype, device=score.device)
    out[valid] = score.reshape(-1)[flat[valid]]
    return out


@torch.no_grad()
def mark_cells_by_loss_sampled(
    corner_points: Tensor,
    extra_points: Tensor | None,
    cell_count: int,
    score_fn: Callable[[Tensor], Tensor],
    max_fraction: float,
) -> Tensor:
    """Mark cells by sampling a loss score at the cell lattice.

    ``score_fn`` maps any (N, 3) point tensor to per-point scores.  A cell is
    marked if any of its 8 corners (or its center when ``extra_points`` is
    given) has a high score, so thin structures crossing a cell interior are
    caught.  Marking is a strict ``topk`` under ``max_fraction``.
    """
    cc = int(cell_count)
    corner_scores = score_fn(corner_points).reshape(cc + 1, cc + 1, cc + 1)
    # Cell (i,j,k) owns corners [i:i+2, j:j+2, k:k+2]; take max so a cell
    # containing any high-error point is marked.
    cell_scores = torch.maximum(
        torch.maximum(
            torch.maximum(corner_scores[:-1, :-1, :-1], corner_scores[1:, :-1, :-1]),
            torch.maximum(corner_scores[:-1, 1:, :-1], corner_scores[1:, 1:, :-1]),
        ),
        torch.maximum(
            torch.maximum(corner_scores[:-1, :-1, 1:], corner_scores[1:, :-1, 1:]),
            torch.maximum(corner_scores[:-1, 1:, 1:], corner_scores[1:, 1:, 1:]),
        ),
    )
    if extra_points is not None:
        center_scores = score_fn(extra_points).reshape(cc, cc, cc)
        cell_scores = torch.maximum(cell_scores, center_scores)
    budget = int(cell_scores.numel() * float(max_fraction))
    marked = torch.zeros(cc, cc, cc, dtype=torch.bool, device=cell_scores.device)
    if budget > 0 and bool(cell_scores.any()):
        keep = torch.topk(
            cell_scores.reshape(-1),
            min(budget, int((cell_scores > 0).sum())),
            largest=True,
        ).indices
        marked.view(-1)[keep] = True
    return marked
