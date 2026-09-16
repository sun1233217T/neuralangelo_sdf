'''
-----------------------------------------------------------------------------
B-spline SDF/RGB wrappers for Neuralangelo.

These modules replace projects.neuralangelo.utils.modules.NeuralSDF and
NeuralRGB with tensor-product B-spline fields.  They keep the same forward
signatures so that projects.neuralangelo.model.Model can use them with minimal
changes.
-----------------------------------------------------------------------------
'''

import math
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import _load_ply_points_xyz, infer_scene_bounds_from_points
from ..bspline_field.field import (
    BSplineSDField,
    BSplineSHRGBField,
    _DEFAULT_BOUNDS,
    _SH_C0,
    _inverse_sigmoid,
    _split_rgb_values,
    spherical_harmonics_basis,
)
from ..bspline_field.hierarchical import (
    HierarchicalBSplineField,
    cell_center_points,
    mark_cells_by_loss,
    mark_cells_by_loss_sampled,
    mark_cells_by_sdf_band,
    mark_cells_by_value_band,
    sample_loss_score_at_points,
)


def _as_float_tensor(value: Any, *, dtype=None, device=None) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value
        if dtype is not None or device is not None:
            tensor = tensor.to(dtype=dtype or tensor.dtype, device=device or tensor.device)
        if not torch.is_floating_point(tensor):
            tensor = tensor.to(dtype=dtype or torch.get_default_dtype())
        return tensor
    return torch.as_tensor(value, dtype=dtype or torch.get_default_dtype(), device=device)


def _coerce_bounds(bounds: Any, *, dtype=None, device=None) -> torch.Tensor:
    tensor = _as_float_tensor(bounds, dtype=dtype, device=device)
    if tensor.shape != (3, 2):
        raise ValueError("bounds must have shape (3, 2).")
    if torch.any(tensor[:, 1] <= tensor[:, 0]).item():
        raise ValueError("Each bounds entry must satisfy min < max.")
    return tensor


class BSplineSDFWrapper(nn.Module):
    """B-spline SDF wrapper compatible with projects.neuralangelo.model.Model.

    Replaces :class:`projects.neuralangelo.utils.modules.NeuralSDF`.

    Two backends are supported:

    - Dense (default): the SDF is stored as a raw (unactivated) control grid
      ``raw_sdf_grid`` evaluated through :class:`BSplineSDField`.
    - Hierarchical (``cfg_bspline.hierarchical.enabled``): a sparse
      hierarchical B-spline field (:class:`HierarchicalBSplineField`) whose
      level-0 grid is initialized exactly like the dense grid.  Levels are
      added by :meth:`maybe_refine` during training.

    In both cases a learnable inverse standard deviation ``raw_sdf_inv_std``
    (log-space, softplus applied) drives the NeuS CDF.
    """

    def __init__(self, cfg_bspline):
        super().__init__()
        self.cfg_bspline = cfg_bspline
        self.spline_degree = int(cfg_bspline.sdf_spline_degree)
        self.bounds = _coerce_bounds(cfg_bspline.bounds)
        # Optional NeuS sharpness floor/ceiling: inv_std() clamps the softplus
        # output.  Floor prevents the surface from becoming too blurry; ceiling
        # prevents inv_std from exploding to a near-step-function and starving
        # SDF gradients.
        self.inv_s_floor = getattr(cfg_bspline, "sdf_inv_s_floor", None)
        self.inv_s_ceil = getattr(cfg_bspline, "sdf_inv_s_ceil", None)

        hier_cfg = getattr(cfg_bspline, "hierarchical", None)
        self.hierarchical_enabled = bool(
            hier_cfg is not None and hier_cfg.get("enabled", False)
        )
        self.hier_cfg = hier_cfg if self.hierarchical_enabled else None
        # Loss-guided refinement marking (E4): per-cell render-loss statistics
        # scattered from training rays, consumed by _mark_cells at refine time.
        # Plain attributes (not buffers): transient statistics, rebuilt after
        # every refine and after resume; never checkpointed.
        self.loss_marking_enabled = bool(
            self.hierarchical_enabled
            and str(self.hier_cfg.get("refine_mark_mode", "sdf_band")) in ("loss", "hybrid")
        )
        self._loss_accum = None
        self._loss_count = None
        # Gradient-driven marking: accumulate per-control-point gradient
        # magnitude as an alternative importance signal for refinement.
        # Analogous to 3DGS's position-gradient-magnitude splitting criterion.
        self.grad_marking_enabled = bool(
            self.hierarchical_enabled
            and str(self.hier_cfg.get("refine_mark_mode", "sdf_band")) in ("grad", "loss_grad")
        )
        self._grad_accum = None  # per-level list of (N_active,) gradient accumulators
        self._grad_hooks = []

        if self.hierarchical_enabled:
            self.grid_size = int(hier_cfg.get("base_grid_size", 34))
            max_levels = int(hier_cfg.get("max_levels", 1))
            transfer_mode = str(hier_cfg.get("transfer_mode", "hb"))
            self.hier_field = HierarchicalBSplineField(
                self.grid_size,
                self.spline_degree,
                self.bounds,
                channels=1,
                max_levels=max_levels,
                transfer_mode=transfer_mode,
                transition_compute_dtype=str(hier_cfg.get("transition_dtype", "fp32")),
            )
            init_grid = self._init_sdf_grid(cfg_bspline)
            self.hier_field.levels[0].values.data = (
                init_grid.reshape(-1, 1).to(self.hier_field.levels[0].values)
            )
        else:
            self.grid_size = int(cfg_bspline.grid_size)
            # Raw SDF control grid.
            raw_sdf_grid = self._init_sdf_grid(cfg_bspline)
            self.register_parameter("raw_sdf_grid", nn.Parameter(raw_sdf_grid))

        # Inverse standard deviation in log-space.
        raw_inv_std = math.log(math.expm1(float(cfg_bspline.sdf_inv_s_init)))
        self.register_parameter(
            "raw_sdf_inv_std",
            nn.Parameter(torch.tensor(raw_inv_std, dtype=torch.get_default_dtype())),
        )

        if not self.hierarchical_enabled:
            # Stateless field object used for evaluation.  The actual learnable
            # parameters are owned by this wrapper; we temporarily assign them to
            # the field before each forward call.
            self._field = BSplineSDField(
                grid_size=self.grid_size,
                control_coefficients=torch.zeros_like(self.raw_sdf_grid),
                bounds=self.bounds,
                spline_degree=self.spline_degree,
                trainable=False,
                inv_std=1.0,
            )

    def _init_sdf_grid(self, cfg_bspline) -> torch.Tensor:
        mode = str(getattr(cfg_bspline, "sdf_init_mode", "constant")).lower()
        grid_size = self.grid_size
        device = getattr(cfg_bspline, "device", None)
        dtype = getattr(cfg_bspline, "dtype", torch.float32)

        if mode == "constant":
            return torch.full(
                (grid_size, grid_size, grid_size),
                fill_value=float(cfg_bspline.sdf_init),
                dtype=dtype,
                device=device,
            )

        if mode == "pointcloud":
            return self._init_sdf_grid_from_pointcloud(cfg_bspline, dtype, device)

        if mode != "sphere":
            raise ValueError(f"Unknown sdf_init_mode: {cfg_bspline.sdf_init_mode!r}")

        # Sphere initialization matching the B-spline cell spacing.
        bounds_cpu = self.bounds.detach().cpu().to(torch.float64)
        lower = bounds_cpu[:, 0]
        upper = bounds_cpu[:, 1]
        center = _as_float_tensor(
            getattr(cfg_bspline, "sdf_init_sphere_center", [0.0, 0.0, 0.0]),
            dtype=torch.float64,
            device="cpu",
        )
        radius = getattr(cfg_bspline, "sdf_init_sphere_radius", None)
        if radius is None:
            radius = float(((upper - lower) * 0.5).min().item() * 0.6)
        else:
            radius = float(radius)

        extent = upper - lower
        cell_count = grid_size - self.spline_degree
        step = extent / float(cell_count)
        coords_1d = [lower[d] + step[d] * torch.arange(grid_size, dtype=torch.float64) for d in range(3)]
        gx, gy, gz = torch.meshgrid(*coords_1d, indexing="ij")
        pts = torch.stack([gx, gy, gz], dim=-1)
        sdf = torch.linalg.norm(pts - center, dim=-1) - radius
        sdf = sdf.clamp(-2.0 * radius, 2.0 * radius)
        return sdf.to(dtype=dtype, device=device)

    def _init_sdf_grid_from_pointcloud(self, cfg_bspline, dtype, device) -> torch.Tensor:
        """Initialize the SDF control grid from a sparse point cloud PLY.

        The PLY vertices are assumed to be in the same world coordinate frame
        used by the dataset (i.e. before the ``readjust`` transform).  We
        recompute the same quantile-based AABB that ``data.py`` uses for
        ``auto_bounds`` and map the points into the B-spline normalized frame
        ``[-B, B]^3`` (typically ``[-2, 2]^3``).  The initial SDF is then the
        signed distance to the bounding sphere of the normalized point cloud,
        which is a much better geometric prior than a tiny sphere at the
        origin for scenes whose object is not centered at (0, 0, 0).

        Config keys consumed from ``cfg_bspline``:
            sdf_init_ply_path: path to the PLY file (absolute or relative to cwd).
            sdf_init_ply_quantile_low / high: quantiles for AABB (default 0.01/0.99).
            sdf_init_ply_padding_ratio: padding added around the AABB (default 0.15).
            sdf_init_ply_force_cube: whether to force a cube AABB (default True).
            sdf_init_ply_radius_percentile: percentile of point norms used for
                                            the bounding sphere radius (default 0.95).
        """
        ply_path = Path(getattr(cfg_bspline, "sdf_init_ply_path", "points.ply"))
        if not ply_path.is_absolute():
            # Try relative to the current working directory first, then relative
            # to the project root used by the dataset loaders.
            cwd_path = Path.cwd() / ply_path
            if cwd_path.exists():
                ply_path = cwd_path
            else:
                ply_path = Path.cwd() / ply_path
        if not ply_path.exists():
            raise FileNotFoundError(f"SDF point-cloud init could not find PLY: {ply_path}")

        points_world = _load_ply_points_xyz(ply_path)
        if points_world.shape[0] == 0:
            raise ValueError(f"Point cloud PLY contains no vertices: {ply_path}")

        quantile_low = float(getattr(cfg_bspline, "sdf_init_ply_quantile_low", 0.01))
        quantile_high = float(getattr(cfg_bspline, "sdf_init_ply_quantile_high", 0.99))
        padding_ratio = float(getattr(cfg_bspline, "sdf_init_ply_padding_ratio", 0.15))
        force_cube = bool(getattr(cfg_bspline, "sdf_init_ply_force_cube", True))
        radius_percentile = float(getattr(cfg_bspline, "sdf_init_ply_radius_percentile", 0.95))

        world_bounds = infer_scene_bounds_from_points(
            ply_path.parent,
            ply_filename=ply_path.name,
            metadata_bounds=None,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            padding_ratio=padding_ratio,
            force_cube=force_cube,
        )
        world_bounds = np.array(world_bounds, dtype=np.float64)
        world_lower = world_bounds[:, 0]
        world_upper = world_bounds[:, 1]
        world_center = 0.5 * (world_lower + world_upper)
        world_extent = float(np.max(world_upper - world_lower))
        if world_extent <= 0.0:
            raise ValueError(f"Degenerate point cloud AABB in {ply_path}")

        bounds_cpu = self.bounds.detach().cpu().to(torch.float64)
        bspline_lower = bounds_cpu[:, 0]
        bspline_upper = bounds_cpu[:, 1]
        bspline_extent = float(torch.max(bspline_upper - bspline_lower).item())
        if bspline_extent <= 0.0:
            raise ValueError("B-spline bounds have zero extent.")

        # Map world -> normalized B-spline frame.
        scale = world_extent / bspline_extent
        points_norm = (points_world.astype(np.float64) - world_center) / scale

        # Bounding sphere in the normalized frame.
        center_cfg = getattr(cfg_bspline, "sdf_init_sphere_center", None)
        if center_cfg is None or str(center_cfg).lower() == "auto":
            center_cfg = points_norm.mean(axis=0).tolist()
        center = _as_float_tensor(center_cfg, dtype=torch.float64, device="cpu")
        norms = np.linalg.norm(points_norm - center.numpy(), axis=1)
        radius = float(np.quantile(norms, radius_percentile))
        if radius <= 0.0:
            radius = float(np.max(norms))
        if radius <= 0.0:
            raise ValueError(f"Point cloud degenerated to a single point: {ply_path}")

        # Generate control-grid lattice in the normalized frame.
        grid_size = self.grid_size
        cell_count = grid_size - self.spline_degree
        step = (bspline_upper - bspline_lower) / float(cell_count)
        coords_1d = [
            bspline_lower[d] + step[d] * torch.arange(grid_size, dtype=torch.float64)
            for d in range(3)
        ]
        gx, gy, gz = torch.meshgrid(*coords_1d, indexing="ij")
        pts = torch.stack([gx, gy, gz], dim=-1)
        sdf = torch.linalg.norm(pts - center, dim=-1) - radius
        sdf = sdf.clamp(-2.0 * radius, 2.0 * radius)
        return sdf.to(dtype=dtype, device=device)

    def inv_std(self) -> torch.Tensor:
        s = F.softplus(self.raw_sdf_inv_std)
        if self.inv_s_floor is not None:
            s = s.clamp_min(float(self.inv_s_floor))
        if self.inv_s_ceil is not None:
            s = s.clamp_max(float(self.inv_s_ceil))
        return s

    def _prepare_field(self):
        self._field.control_grid = self.raw_sdf_grid
        self._field._inv_std = self.inv_std()

    def _apply(self, fn, recurse=True):
        # Plain tensor attributes do not follow module.to()/.cuda(); move
        # ``bounds`` along so _clamp_points does not pay a synchronous
        # host-to-device copy on every forward (which also breaks CUDA-graph
        # capture).
        module = super()._apply(fn, recurse)
        self.bounds = fn(self.bounds)
        return module

    def _evaluate_sdf(self, points_3D):
        """Raw SDF evaluation dispatching to the active backend."""
        if self.hierarchical_enabled:
            return self.hier_field.evaluate(points_3D)  # (..., 1)
        self._prepare_field()
        return self._field.evaluate(points_3D).unsqueeze(-1)  # [...,1]

    def _evaluate_sdf_with_deriv(self, points_3D):
        """Analytic value/gradient/diagonal-Hessian, dispatching to the backend.

        Returns (values (...,1), grads (...,1,3), hess_diag (...,1,3)).
        The field is linear in the control points, so the outputs keep a
        first-order graph into the control-point parameters without any
        nested autograd (CUDA-graph capture safe).
        """
        if self.hierarchical_enabled:
            return self.hier_field.evaluate_with_deriv(points_3D)
        self._prepare_field()
        values, grads, hess = self._field.evaluate_with_deriv(points_3D)
        return values.unsqueeze(-1), grads.unsqueeze(-2), hess.unsqueeze(-2)

    def _evaluate_sdf_full_hessian(self, points_3D):
        """Analytic value/gradient/full-Hessian (diagonal + off-diagonal).

        Returns (values, grads, hess_diag, hess_off) where hess_off contains
        (∂²f/∂xy, ∂²f/∂xz, ∂²f/∂yz).  The off-diagonal terms are already
        computed by the THB evaluation; this method simply exposes them.
        """
        if not self.hierarchical_enabled:
            raise NotImplementedError("Full Hessian requires hierarchical THB mode.")
        return self.hier_field.evaluate_with_full_hessian(points_3D)

    def evaluate_mean_curvature(self, points_3D):
        """Compute mean curvature of the SDF zero level set at query points.

        Mean curvature of the implicit surface f(x)=0:
            H = (∇²f·|∇f|² - ∇f·Hess·∇f) / |∇f|³

        This is the true geometric curvature of the surface, unlike the raw
        Laplacian which conflates surface curvature with gradient magnitude
        variation.  Small artifacts ("bubbles") have high mean curvature
        (H ~ 2/r for sphere radius r), while real surfaces have low H.

        Requires hierarchical THB mode (full Hessian support).
        Returns (H (...,), grads (...,3)) — mean curvature and SDF gradient.
        """
        if not self.hierarchical_enabled:
            raise NotImplementedError("Mean curvature requires hierarchical THB mode.")
        points_3D = self._clamp_points(points_3D)
        values, grads, hess_diag, hess_off = self.hier_field.evaluate_with_full_hessian(points_3D)
        # grads: (...,1,3), hess_diag: (...,1,3), hess_off: (...,1,3)
        g = grads[..., 0, :]  # (...,3)
        g2 = (g * g).sum(dim=-1, keepdim=False)  # |∇f|²
        gn = g2.clamp_min(1e-8)  # avoid division by zero
        # ∇²f (Laplacian) = sum of diagonal Hessian
        lap = hess_diag[..., 0, :].sum(dim=-1, keepdim=False)  # (...,)
        # ∇f·Hess·∇f = Σ_i Σ_j g_i * H_ij * g_j
        # = gx²*hxx + gy²*hyy + gz²*hzz + 2*(gx*gy*hxy + gx*gz*hxz + gy*gz*hyz)
        gx, gy, gz = g[..., 0], g[..., 1], g[..., 2]
        hxx, hyy, hzz = hess_diag[..., 0, 0], hess_diag[..., 0, 1], hess_diag[..., 0, 2]
        hxy, hxz, hyz = hess_off[..., 0, 0], hess_off[..., 0, 1], hess_off[..., 0, 2]
        gHg = (gx * gx * hxx + gy * gy * hyy + gz * gz * hzz
               + 2.0 * (gx * gy * hxy + gx * gz * hxz + gy * gz * hyz))
        # H = (∇²f·|∇f|² - ∇f·Hess·∇f) / |∇f|³
        H = (lap * g2 - gHg) / (gn * g2.sqrt())
        return H, g

    def _clamp_points(self, points_3D):
        """Clamp query points to the field bounds to avoid out-of-bounds errors."""
        shape = (1,) * (points_3D.ndim - 1) + (3,)
        lower = self.bounds[:, 0].reshape(shape).to(points_3D)
        upper = self.bounds[:, 1].reshape(shape).to(points_3D)
        # Use a tiny epsilon so points stay strictly inside the valid support.
        eps = 1e-5
        return torch.clamp(points_3D, min=lower + eps, max=upper - eps)

    def forward(self, points_3D, with_sdf=True, with_feat=True):
        points_3D = self._clamp_points(points_3D)
        sdf = self._evaluate_sdf(points_3D)
        feat = None
        if with_feat:
            feat = torch.zeros(
                *points_3D.shape[:-1], 0,
                device=points_3D.device,
                dtype=points_3D.dtype,
            )
        return sdf, feat

    def sdf(self, points_3D):
        return self.forward(points_3D, with_sdf=True, with_feat=False)[0]

    def compute_gradients(self, x, training=False, sdf=None):
        """Compute gradient and (training-only) diagonal Hessian w.r.t. points.

        Uses the field's analytic derivatives instead of nested autograd:
        the B-spline field is linear in its control points, so the returned
        gradient/Hessian keep a first-order graph into the control-point
        parameters and the outer ``loss.backward()`` stays a plain
        first-order backward (no ``create_graph=True`` second-order pass,
        which dominated the training-step backward and broke CUDA-graph
        capture).
        """
        original_shape = x.shape
        x = self._clamp_points(x)
        x_flat = x.reshape(-1, 3).detach()
        if training:
            with torch.enable_grad():
                _, grads, hess = self._evaluate_sdf_with_deriv(x_flat)
            gradient = grads[..., 0, :]
            hessian = hess[..., 0, :]
        else:
            with torch.no_grad():
                _, grads, _ = self._evaluate_sdf_with_deriv(x_flat)
            gradient = grads[..., 0, :].detach()
            hessian = None
        gradient = gradient.reshape(original_shape)
        if hessian is not None:
            hessian = hessian.reshape(original_shape)
        return gradient, hessian

    # -- Gradient-driven refinement marking -------------------------------------

    def _register_grad_hooks(self):
        """Register gradient hooks on all level values to accumulate |grad|.

        The hook fires during backward and accumulates the per-control-point
        gradient L2 norm into a persistent buffer.  At refinement time, the
        accumulated magnitude serves as the importance score: control points
        that are actively being updated are the ones that need finer
        resolution (analogous to 3DGS's position-gradient splitting).
        """
        for h in self._grad_hooks:
            h.remove()
        self._grad_hooks = []
        if not self.grad_marking_enabled:
            return
        if self.hier_field.num_levels >= self.hier_field.max_levels:
            return
        self._grad_accum = [None] * self.hier_field.num_levels
        for l, level in enumerate(self.hier_field.levels):
            if level.values.requires_grad:
                hook = level.values.register_hook(
                    lambda grad, lvl=l: self._accumulate_grad(lvl, grad)
                )
                self._grad_hooks.append(hook)

    @torch.no_grad()
    def _accumulate_grad(self, level_idx, grad):
        """Accumulate gradient L2 norm per control point (called by hook)."""
        if self._grad_accum is None:
            return
        acc = self._grad_accum[level_idx]
        if acc is None or acc.shape[0] != grad.shape[0]:
            # Lazily allocate or re-allocate after refinement changes the
            # number of active control points.
            acc = torch.zeros(grad.shape[0], dtype=torch.float32, device=grad.device)
            self._grad_accum[level_idx] = acc
        acc.add_(grad.detach().norm(dim=-1))

    @torch.no_grad()
    def grad_mark_cells(self, level_idx, max_fraction):
        """Mark cells whose control points have the highest accumulated gradient.

        Maps per-control-point gradient magnitudes to per-cell scores by
        scattering each control point's magnitude to its (p+1)^3 support cells.
        """
        if self._grad_accum is None or self._grad_accum[level_idx] is None:
            return None
        level = self.hier_field.levels[level_idx]
        p = self.spline_degree
        cc = level.cell_count
        G = level.grid_size

        # Scatter control-point gradient magnitudes to their support cells.
        grad_mag = self._grad_accum[level_idx]  # (N_active,)
        # Get the grid indices of active control points.
        active_mask = level.index_grid >= 0
        cp_indices = torch.nonzero(active_mask, as_tuple=False)  # (N_active, 3)

        cell_score = torch.zeros((cc,) * 3, dtype=torch.float32, device=grad_mag.device)
        cell_count = torch.zeros((cc,) * 3, dtype=torch.float32, device=grad_mag.device)
        # Each control point at grid position (i,j,k) covers cells
        # [i-o, j-o, k-o] for o in [0, p].
        for o0, o1, o2 in product(range(p + 1), repeat=3):
            ci = cp_indices[:, 0] - o0
            cj = cp_indices[:, 1] - o1
            ck = cp_indices[:, 2] - o2
            valid = (ci >= 0) & (ci < cc) & (cj >= 0) & (cj < cc) & (ck >= 0) & (ck < cc)
            if not valid.any():
                continue
            flat = (ci[valid] * cc + cj[valid]) * cc + ck[valid]
            cell_score.reshape(-1).index_add_(0, flat, grad_mag[valid])
            cell_count.reshape(-1).index_add_(0, flat, torch.ones_like(grad_mag[valid]))

        # Score = mean gradient magnitude per cell.
        score = cell_score / (cell_count + 1e-8)
        score = torch.where(cell_count > 0, score, torch.zeros_like(score))

        budget = int(score.numel() * float(max_fraction))
        marked = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
        if budget > 0 and bool((score > 0).any()):
            keep = torch.topk(score.reshape(-1), min(budget, int((score > 0).sum())),
                              largest=True).indices
            marked[keep] = True
        return marked.reshape(score.shape)
    def _get_anchor_offsets(self, radius, device):
        """Cached neighbor offset tensor for surface-anchor scatter.

        Pre-computed at first use to avoid creating tensors during CUDA
        graph capture (CPU→GPU transfer is not capturable).
        """
        cache_key = (radius, str(device))
        if not hasattr(self, "_anchor_offsets_cache"):
            self._anchor_offsets_cache = {}
        if cache_key not in self._anchor_offsets_cache:
            offsets = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        offsets.append([dx, dy, dz])
            self._anchor_offsets_cache[cache_key] = torch.tensor(
                offsets, dtype=torch.long, device=device
            )
        return self._anchor_offsets_cache[cache_key]

    @torch.no_grad()
    def accumulate_loss(self, points, weights, rgb, target, sdfs=None):
        """Scatter per-ray render error into the finest level's cell grid.

        ``points`` (B,R,N,3) and ``weights`` (B,R,N) are the (detached) object
        samples of the training rays; ``rgb``/``target`` are the per-ray
        rendered/GT colors (B,R,3).

        When ``sdfs`` (B,R,N) is provided and ``refine_surface_anchor`` is
        enabled in the hierarchical config, the error is scattered to the
        estimated SDF zero-crossing point along each ray instead of being
        distributed by compositing weights.  This anchors the refinement
        marking precisely at the surface, rather than at the compositing
        weight peak (which is systematically offset to the outside).

        Graph-capture safe: no boolean indexing / host syncs.
        """
        if not self.loss_marking_enabled:
            return
        if self.hier_field.num_levels >= self.hier_field.max_levels \
                and not self.hier_cfg.get("prune_iters", []):
            # No further refinement and no activity-based pruning scheduled:
            # the statistics would never be consumed.
            return
        level = self.hier_field.levels[-1]
        cc = level.cell_count
        if self._loss_accum is None or tuple(self._loss_accum.shape) != (cc,) * 3:
            device = level.values.device
            self._loss_accum = torch.zeros((cc,) * 3, dtype=torch.float32, device=device)
            self._loss_count = torch.zeros((cc,) * 3, dtype=torch.float32, device=device)
        err = (rgb.detach() - target).abs().mean(dim=-1)  # (B, R)

        use_anchor = (
            sdfs is not None
            and bool(self.hier_cfg.get("refine_surface_anchor", False))
        )
        if use_anchor:
            # --- Surface crossing anchoring ---
            # sdfs: (B,R,N), points: (B,R,N,3)
            B, R, N = sdfs.shape
            # Find sign changes along each ray.
            sign_prod = sdfs[..., :-1] * sdfs[..., 1:]  # (B,R,N-1)
            has_crossing = sign_prod < 0  # (B,R,N-1)
            # First crossing index per ray.
            first_idx = has_crossing.float().argmax(dim=-1)  # (B,R)
            any_crossing = has_crossing.any(dim=-1)  # (B,R)
            # Linear interpolation: t = sdf_i / (sdf_i - sdf_{i+1})
            sdf_i = torch.gather(sdfs, 2, first_idx.unsqueeze(-1)).squeeze(-1)
            sdf_next = torch.gather(sdfs, 2, (first_idx + 1).unsqueeze(-1)).squeeze(-1)
            t = sdf_i / (sdf_i - sdf_next + 1e-8)  # (B,R)
            # Gather the two bracketing points: (B,R,3)
            idx4d = first_idx[:, :, None, None].expand(-1, -1, 1, 3)
            idx4d_next = (first_idx + 1)[:, :, None, None].expand(-1, -1, 1, 3)
            p_i = torch.gather(points, 2, idx4d).squeeze(2)  # (B,R,3)
            p_next = torch.gather(points, 2, idx4d_next).squeeze(2)  # (B,R,3)
            crossing = p_i + t.unsqueeze(-1) * (p_next - p_i)  # (B,R,3)

            # Scatter to the crossing cell AND its neighbors within radius.
            # radius=0: exact cell only (original behavior).
            # radius=1: +6 face neighbors.  radius=2: +26 neighbors (3³-1).
            anchor_radius = int(self.hier_cfg.get("refine_surface_anchor_radius", 0))
            pts_cross = crossing.reshape(-1, 3).to(dtype=torch.float32)
            err_cross = err.reshape(-1)
            valid_cross = any_crossing.reshape(-1)

            # Compute base cell indices for all crossing points.
            base_idx = ((pts_cross - level._lower) / level.step).floor().long()
            in_bounds_base = ((base_idx >= 0) & (base_idx < cc)).all(dim=-1)
            valid_base = valid_cross * in_bounds_base.to(err_cross.dtype)

            if anchor_radius <= 0:
                idx = base_idx
                mass = err_cross
                valid = valid_base
            else:
                # Use cached neighbor offsets (pre-computed to avoid creating
                # tensors during CUDA graph capture, which is not capturable).
                offsets_t = self._get_anchor_offsets(anchor_radius, pts_cross.device)
                # Expand: (M, 1, 3) + (1, K, 3) → (M, K, 3) cell indices
                idx_all = base_idx.unsqueeze(1) + offsets_t.unsqueeze(0)  # (M,K,3)
                M, K = idx_all.shape[0], idx_all.shape[1]
                idx_flat = idx_all.reshape(-1, 3)
                in_bounds_all = ((idx_flat >= 0) & (idx_flat < cc)).all(dim=-1)
                # Each neighbor gets 1/(K+1) of the error mass.
                mass_all = (err_cross / (K + 1)).unsqueeze(1).expand(-1, K).reshape(-1)
                valid_all = (valid_base.unsqueeze(1).expand(-1, K).reshape(-1)
                             * in_bounds_all.to(err_cross.dtype))
                # Also include the center cell itself with full weight.
                mass = torch.cat([err_cross, mass_all])
                valid = torch.cat([valid_base, valid_all])
                idx = torch.cat([base_idx, idx_flat])

            # Scatter to cells.
            in_bounds = ((idx >= 0) & (idx < cc)).all(dim=-1).to(mass.dtype)
            valid = valid * in_bounds
            idx = idx.clamp(0, cc - 1)
            flat = (idx[:, 0] * cc + idx[:, 1]) * cc + idx[:, 2]
            self._loss_accum.reshape(-1).index_add_(0, flat, mass * valid)
            self._loss_count.reshape(-1).index_add_(0, flat, valid)
            return
        else:
            # --- Original compositing-weight approach ---
            mass = (weights.reshape(*err.shape, -1) * err.unsqueeze(-1)).reshape(-1)
            pts = points.reshape(-1, 3).to(dtype=torch.float32)
            valid = torch.ones_like(mass)

        # Out-of-place floor: points may be a view into graph-pool memory.
        idx = ((pts - level._lower) / level.step).floor().long()  # (M, 3)
        in_bounds = ((idx >= 0) & (idx < cc)).all(dim=-1).to(mass.dtype)
        valid = valid * in_bounds
        idx.clamp_(0, cc - 1)
        flat = (idx[:, 0] * cc + idx[:, 1]) * cc + idx[:, 2]
        self._loss_accum.reshape(-1).index_add_(0, flat, mass * valid)
        self._loss_count.reshape(-1).index_add_(0, flat, valid)

    @torch.no_grad()
    def _mark_cells(self, level_idx, corners, round_idx):
        """Refinement marking dispatch shared by the SDF and RGB wrappers.

        ``refine_mark_mode: loss`` ranks cells by the accumulated render-loss
        statistics (topk under ``refine_max_marked_fraction``); anything else
        keeps the SDF-band marking.  Falls back to the band when no loss
        statistics are available (e.g. a resume landing exactly on a refine
        iteration) so the schedule never silently no-ops.
        """
        max_fraction = self.hier_cfg.get("refine_max_marked_fraction", None)
        mode = str(self.hier_cfg.get("refine_mark_mode", "sdf_band"))
        level = self.hier_field.levels[level_idx]
        region = level.region  # parent-level active region for pre-filtering

        # Gradient-driven marking: use accumulated |grad| per control point.
        if mode in ("grad", "loss_grad"):
            marked_grad = self.grad_mark_cells(level_idx, max_fraction or 0.1)
            if marked_grad is not None:
                marked_grad = marked_grad & region  # constrain to parent region
                if mode == "grad":
                    return marked_grad
            else:
                print("[Bspline-Neus] grad marking requested but no gradient "
                      "statistics; falling back to loss/band.")

        if mode in ("loss", "hybrid", "loss_grad"):
            cc = level.cell_count
            if (
                self._loss_accum is not None
                and tuple(self._loss_accum.shape) == (cc,) * 3
                and bool(self._loss_count.any())
            ):
                if max_fraction is None:
                    raise ValueError(
                        "refine_mark_mode=loss/hybrid requires refine_max_marked_fraction "
                        "(loss scores have no natural absolute threshold)."
                    )
                min_count = float(self.hier_cfg.get("refine_loss_min_count", 1.0))
                marked_loss = mark_cells_by_loss(
                    self._loss_accum, self._loss_count, max_fraction, min_count,
                    region=region,
                )
                if mode == "loss":
                    return marked_loss
            else:
                marked_loss = None
                print("[Bspline-Neus] loss/hybrid marking requested but no statistics "
                      "available; falling back to SDF band.")
        band = self.hier_cfg.get("refine_sdf_band", 0.05)
        if isinstance(band, (list, tuple)):
            band = band[min(round_idx, len(band) - 1)]
        mark_centers = bool(self.hier_cfg.get("refine_mark_centers", False))
        marked_band = mark_cells_by_sdf_band(
            self.hier_field, level_idx, corners, float(band),
            include_centers=mark_centers,
            max_fraction=max_fraction,
            region=region,
        )
        if mode == "hybrid" and marked_loss is not None:
            return marked_loss | marked_band
        if mode == "loss_grad" and marked_grad is not None:
            if marked_loss is not None:
                return marked_grad | marked_loss  # union of both signals
            return marked_grad
        return marked_band

    @torch.no_grad()
    def loss_mark_cells_for_target(
        self,
        level_idx: int,
        target_level,
        corner_points: torch.Tensor,
        extra_points: torch.Tensor | None,
        max_fraction: float,
        min_count: float = 1.0,
    ) -> torch.Tensor | None:
        """Produce a loss-based marking for a (possibly different) target grid.

        When the target hierarchy has the same cell grid as the SDF (the
        default), the cached loss score tensor is used directly.  When the
        grids differ (e.g. different spline degrees for SDF and color) the
        score is sampled at the target cell lattice.  Returns ``None`` when no
        loss statistics are available, signalling a fallback to band marking.
        """
        if (
            self._loss_accum is None
            or tuple(self._loss_accum.shape) != (self.hier_field.levels[level_idx].cell_count,) * 3
            or not bool(self._loss_count.any())
        ):
            return None
        source_level = self.hier_field.levels[level_idx]
        if source_level.cell_count == target_level.cell_count:
            return mark_cells_by_loss(
                self._loss_accum, self._loss_count, max_fraction, min_count,
                region=target_level.region,
            )
        score_fn = lambda pts: sample_loss_score_at_points(
            self._loss_accum, self._loss_count, source_level, pts, min_count
        )
        return mark_cells_by_loss_sampled(
            corner_points, extra_points, target_level.cell_count,
            score_fn, max_fraction,
        )

    @torch.no_grad()
    def maybe_refine(self, iteration):
        """Refine the hierarchical field when ``iteration`` hits the schedule.

        Returns the refinement info dict, or ``None`` when nothing happened.
        """
        if not self.hierarchical_enabled:
            return None
        refine_iters = [int(i) for i in self.hier_cfg.get("refine_iters", [])]
        if iteration not in refine_iters:
            return None
        round_idx = refine_iters.index(iteration)
        if self.hier_field.num_levels >= round_idx + 2:
            # This round already happened (e.g. resumed from a checkpoint
            # saved at exactly this iteration); do not refine twice.
            return None
        info = self.hier_field.refine(
            lambda level_idx, corners: self._mark_cells(level_idx, corners, round_idx)
        )
        # NOTE: the loss statistics are intentionally NOT reset here — the RGB
        # wrapper's maybe_refine runs right after this one and reuses them via
        # _mark_cells to mark the identical cell set.  They are rebuilt lazily
        # by accumulate_loss's shape check instead (refinement always doubles
        # the grid, so the pre-refine buffers never match the new finest level).
        # Re-register gradient hooks: refinement replaces the value tensors,
        # so old hooks are on dead parameters.
        if self.grad_marking_enabled and info.get("refined"):
            self._register_grad_hooks()
        return info

    @torch.no_grad()
    def maybe_reband(self, iteration):
        """Re-evaluate the finest level's region at scheduled iterations.

        Unlike refinement (which adds new levels), re-banding adjusts the
        existing finest level's coverage to match the current SDF surface.
        Returns the reband info dict, or ``None`` when nothing happened.
        """
        if not self.hierarchical_enabled:
            return None
        reband_iters = [int(i) for i in self.hier_cfg.get("reband_iters", [])]
        if iteration not in reband_iters:
            return None
        if self.hier_field.num_levels < 2:
            return None
        band = self.hier_cfg.get("reband_band", None)
        if band is None:
            # Default: use the last refine band value.
            bands = self.hier_cfg.get("refine_sdf_band", 0.05)
            band = bands[-1] if isinstance(bands, (list, tuple)) else bands
        max_frac = self.hier_cfg.get("reband_max_fraction", None)
        mark_centers = bool(self.hier_cfg.get("refine_mark_centers", True))
        info = self.hier_field.reband_finest(
            float(band), max_fraction=max_frac, include_centers=mark_centers,
        )
        if info.get("changed"):
            print(f"[Bspline-Neus] rebanding at iter {iteration}: "
                  f"added={info['added']}, removed={info['removed']}")
        return info

    @torch.no_grad()
    def maybe_reband_all(self, iteration):
        """Re-evaluate ALL level regions at scheduled iterations.

        Unlike ``maybe_reband`` (finest level only), this rebuilds every
        level's region from the current SDF — analogous to 3DGS's periodic
        split-and-prune.  Config keys:
            reband_all_iters: list of iterations to fire.
            reband_all_bands: per-level band values (one per level).
            reband_all_max_fractions: optional per-level caps.
        """
        if not self.hierarchical_enabled:
            return None
        reband_iters = [int(i) for i in self.hier_cfg.get("reband_all_iters", [])]
        if iteration not in reband_iters:
            return None
        if self.hier_field.num_levels < 2:
            return None
        n = self.hier_field.num_levels
        bands = self.hier_cfg.get("reband_all_bands", None)
        if bands is None:
            # Default: same band for all levels (the last refine band).
            default_bands = self.hier_cfg.get("refine_sdf_band", 0.05)
            last = default_bands[-1] if isinstance(default_bands, (list, tuple)) else default_bands
            bands = [float(last)] * n
        max_fracs = self.hier_cfg.get("reband_all_max_fractions", None)
        info = self.hier_field.reband_all(bands, max_fractions=max_fracs)
        if info.get("changed"):
            stats = "  ".join(
                f"L{s['level']}:+{s['added']}/-{s['removed']}"
                for s in info["per_level"] if s["added"] or s["removed"]
            )
            print(f"[Bspline-Neus] reband_all at iter {iteration}: {stats}")
        return info

    @torch.no_grad()
    def maybe_prune(self, iteration):
        """Activity-based pruning of the finest level at scheduled iterations.

        Cells that never received any loss-marking hits (zero accumulated
        count) since the last refinement are deactivated, subject to a
        dilation margin so the B-spline support of every marked cell stays
        intact (deg-2 needs a 1-ring; the default margin of 3 is
        conservative).  This removes the interior "bubble web" that no
        camera ray ever crosses.  Config keys:
            prune_iters: list of iterations to fire.
            prune_margin: dilation radius around hit cells (default 3).
        """
        if not self.hierarchical_enabled:
            return None
        prune_iters = [int(i) for i in self.hier_cfg.get("prune_iters", [])]
        if iteration not in prune_iters:
            return None
        level = self.hier_field.levels[-1]
        cc = level.cell_count
        if (
            self._loss_count is None
            or tuple(self._loss_count.shape) != (cc,) * 3
            or not bool(self._loss_count.any())
        ):
            print("[Bspline-Neus] prune requested but no loss statistics; skipping.")
            return None
        margin = int(self.hier_cfg.get("prune_margin", 3))
        live = (self._loss_count > 0).float()[None, None]
        if margin > 0:
            keep = F.max_pool3d(live, kernel_size=2 * margin + 1, stride=1,
                                padding=margin)[0, 0] > 0
        else:
            keep = live[0, 0] > 0
        info = self.hier_field.prune_finest(keep)
        if info.get("changed"):
            print(f"[Bspline-Neus] pruning at iter {iteration}: "
                  f"removed={info['removed']}, "
                  f"active={int(self.hier_field.levels[-1].num_active)}")
            # Region change replaces value tensors: re-register grad hooks.
            if self.grad_marking_enabled:
                self._register_grad_hooks()
        else:
            print(f"[Bspline-Neus] pruning at iter {iteration}: no-op "
                  f"({info.get('reason', 'nothing to remove')})")
        return info


class BSplineRGBWrapper(nn.Module):
    """B-spline RGB wrapper compatible with projects.neuralangelo.model.Model.

    Replaces :class:`projects.neuralangelo.utils.modules.NeuralRGB`.

    Three backends are supported:

    - ``color_mode == "sh"`` (default): a B-spline field that outputs
      spherical-harmonics coefficients (``3 * sh_dim`` channels per control
      point), then projects to RGB with the view direction.  Dense and
      hierarchical variants are supported.
    - ``color_mode == "mlp"``: a B-spline field that outputs a learned
      spatial-feature vector, which is decoded by a tiny MLP together with the
      3D point and view direction.  This relaxes the fixed SH/linear structure
      and lets the network learn RGB-channel correlations and view-dependent
      effects.  Only hierarchical is supported in this mode.
    """

    def __init__(self, cfg_bspline, sdf_wrapper=None):
        super().__init__()
        self.cfg_bspline = cfg_bspline
        self.bounds = _coerce_bounds(cfg_bspline.bounds)
        self.color_mode = str(getattr(cfg_bspline, "color_mode", "sh")).lower()
        self.color_spline_degree = cfg_bspline.color_spline_degree
        self.color_sh_degree = int(getattr(cfg_bspline, "color_sh_degree", 2))
        self.color_sh_basis_dim = (self.color_sh_degree + 1) ** 2
        self._mlp = None

        hier_cfg = getattr(cfg_bspline, "hierarchical", None)
        self.hierarchical_enabled = bool(
            hier_cfg is not None
            and hier_cfg.get("enabled", False)
            and hier_cfg.get("color_enabled", False)
        )
        self.hier_cfg = hier_cfg if self.hierarchical_enabled else None

        device = getattr(cfg_bspline, "device", None)
        dtype = getattr(cfg_bspline, "dtype", torch.float32)
        color_init = getattr(cfg_bspline, "color_init", [0.5, 0.5, 0.5])
        color_r, color_g, color_b = _split_rgb_values(color_init, name="color_init")

        # Optionally share the grid topology with the SDF hierarchy.  This is
        # only safe when both fields use the same base grid size and spline
        # degree; otherwise the color hierarchy keeps its own structure.
        structure_owner = None
        if (
            self.hierarchical_enabled
            and hier_cfg is not None
            and bool(hier_cfg.get("share_structure", True))
            and sdf_wrapper is not None
            and getattr(sdf_wrapper, "hierarchical_enabled", False)
        ):
            if (
                int(hier_cfg.get("base_grid_size", 34)) == sdf_wrapper.grid_size
                and int(self.color_spline_degree) == int(sdf_wrapper.spline_degree)
                and int(hier_cfg.get("max_levels", 1)) <= sdf_wrapper.hier_field.max_levels
            ):
                structure_owner = sdf_wrapper.hier_field
            else:
                print(
                    "[Bspline-Neus] SDF/color structure parameters differ; "
                    "color hierarchy will own its own structure."
                )

        # Gradient scaling for SH bands > 0, mirroring the source project.
        self.color_sh_rest_lr_multiplier = float(
            getattr(cfg_bspline, "color_sh_rest_lr_multiplier", 1.0)
        )
        self._sh_hook_handles = []
        self._sh_channel_scale_cache = {}

        if self.color_mode == "mlp":
            if not self.hierarchical_enabled:
                raise NotImplementedError(
                    "color_mode='mlp' currently requires a hierarchical color field."
                )
            self.grid_size = int(hier_cfg.get("base_grid_size", 34))
            self.color_feature_dim = int(getattr(cfg_bspline, "color_feature_dim", 16))
            # Optional per-level feature dims (e.g. [8, 8, 12, 16, 24]).
            # When set, overrides the uniform color_feature_dim.  The total
            # feature dimension is the sum, and levels are concatenated
            # (not summed) before the MLP.
            color_feature_dims = getattr(cfg_bspline, "color_feature_dims", None)
            if color_feature_dims is not None:
                color_feature_dims = [int(d) for d in color_feature_dims]
                self._per_level_dims = True
                total_feature_dim = sum(color_feature_dims)
            else:
                self._per_level_dims = False
                total_feature_dim = self.color_feature_dim
            hidden_dims = list(getattr(cfg_bspline, "color_mlp_hidden_dims", [64, 64]))
            self.color_mlp_pos_freqs = int(getattr(cfg_bspline, "color_mlp_pos_freqs", 0))
            self.color_mlp_view_freqs = int(getattr(cfg_bspline, "color_mlp_view_freqs", 0))
            self.color_mlp_use_normal = bool(getattr(cfg_bspline, "color_mlp_use_normal", False))
            self.hier_field = HierarchicalBSplineField(
                self.grid_size,
                int(self.color_spline_degree),
                self.bounds,
                channels=color_feature_dims if color_feature_dims else self.color_feature_dim,
                max_levels=int(hier_cfg.get("max_levels", 1)),
                transfer_mode=str(
                    hier_cfg.get("color_transfer_mode") or hier_cfg.get("transfer_mode", "hb")
                ),
                dtype=dtype,
                device=device,
                structure_owner=structure_owner,
            )
            # Initialize feature grid near zero; MLP outputs raw RGB logits.
            self.hier_field.levels[0].values.data.zero_()
            pos_dim = 3 + 2 * 3 * self.color_mlp_pos_freqs if self.color_mlp_pos_freqs > 0 else 3
            view_dim = 3 + 2 * 3 * self.color_mlp_view_freqs if self.color_mlp_view_freqs > 0 else 3
            normal_dim = 3 if self.color_mlp_use_normal else 0
            mlp_in_dim = total_feature_dim + pos_dim + view_dim + normal_dim
            self._mlp = self._build_color_mlp(hidden_dims, in_dim=mlp_in_dim)
            return

        if self.hierarchical_enabled:
            self.grid_size = int(hier_cfg.get("base_grid_size", 34))
            channels = 3 * self.color_sh_basis_dim
            self.hier_field = HierarchicalBSplineField(
                self.grid_size,
                int(self.color_spline_degree),
                self.bounds,
                channels=channels,
                max_levels=int(hier_cfg.get("max_levels", 1)),
                transfer_mode=str(
                    hier_cfg.get("color_transfer_mode") or hier_cfg.get("transfer_mode", "hb")
                ),
                dtype=dtype,
                device=device,
                structure_owner=structure_owner,
            )
            init_values = torch.zeros_like(self.hier_field.levels[0].values)
            sh = self.color_sh_basis_dim
            init_values[:, 0 * sh] = _inverse_sigmoid(color_r) / _SH_C0
            init_values[:, 1 * sh] = _inverse_sigmoid(color_g) / _SH_C0
            init_values[:, 2 * sh] = _inverse_sigmoid(color_b) / _SH_C0
            self.hier_field.levels[0].values.data = init_values
            self._register_sh_hooks()
            return

        self.grid_size = int(cfg_bspline.grid_size)
        color_grid = torch.zeros(
            (self.grid_size, self.grid_size, self.grid_size, 3, self.color_sh_basis_dim),
            dtype=dtype,
            device=device,
        )
        color_grid[..., 0, 0].fill_(_inverse_sigmoid(color_r) / _SH_C0)
        color_grid[..., 1, 0].fill_(_inverse_sigmoid(color_g) / _SH_C0)
        color_grid[..., 2, 0].fill_(_inverse_sigmoid(color_b) / _SH_C0)
        self.register_parameter("raw_color_grid", nn.Parameter(color_grid))

        if self.color_sh_basis_dim > 1 and self.color_sh_rest_lr_multiplier != 1.0:
            self.raw_color_grid.register_hook(self._scale_color_sh_rest_grad)

        # Stateless field object.
        self._field = BSplineSHRGBField(
            grid_size=self.grid_size,
            control_coefficients=torch.zeros(
                3, self.color_sh_basis_dim, self.grid_size, self.grid_size, self.grid_size,
                dtype=dtype,
                device=device,
            ),
            bounds=self.bounds,
            spline_degree=self.color_spline_degree,
            sh_degree=self.color_sh_degree,
            trainable=False,
        )

    @staticmethod
    def _fourier_features(x, num_freqs):
        """NeRF-style positional encoding: [x, sin(pi*2^k*x), cos(pi*2^k*x)].

        Args:
            x: Tensor of shape [..., D].
            num_freqs: Number of frequency octaves.
        Returns:
            Tensor of shape [..., D + 2*num_freqs*D].
        """
        if num_freqs <= 0:
            return x
        freqs = math.pi * (2.0 ** torch.arange(num_freqs, dtype=x.dtype, device=x.device))
        # x[..., None, :] * freqs[:, None] -> [..., num_freqs, D]
        args = x[..., None, :] * freqs[:, None]
        return torch.cat([x, torch.sin(args).flatten(-2, -1), torch.cos(args).flatten(-2, -1)], dim=-1)

    @staticmethod
    def _build_color_mlp(hidden_dims: list[int], in_dim: int = 22, out_dim: int = 3) -> nn.Module:
        """Tiny MLP for decoding spatial features + position + view direction."""
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def _scale_color_sh_rest_grad(self, grad: torch.Tensor) -> torch.Tensor:
        if grad is None or self.color_sh_basis_dim <= 1:
            return grad
        multiplier = self.color_sh_rest_lr_multiplier
        if math.isclose(multiplier, 1.0):
            return grad
        scaled = grad.clone()
        scaled[..., 1:] = scaled[..., 1:] * multiplier
        return scaled

    def _scale_hier_sh_rest_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """SH-band gradient scaling for hierarchical ``(N, 3*sh_dim)`` values."""
        if grad is None or self.color_sh_basis_dim <= 1:
            return grad
        multiplier = self.color_sh_rest_lr_multiplier
        if math.isclose(multiplier, 1.0):
            return grad
        # Per-channel scale vector, cached per (channels, device, dtype):
        # multiplying by it is a single dense kernel — no Python-list advanced
        # indexing (H2D index copy), so the hook is CUDA-graph capturable.
        key = (grad.shape[1], grad.device, grad.dtype)
        scale = self._sh_channel_scale_cache.get(key)
        if scale is None:
            sh = self.color_sh_basis_dim
            channels = torch.arange(grad.shape[1], device=grad.device)
            is_dc = (channels % sh) == 0
            scale = torch.where(
                is_dc,
                torch.ones((), device=grad.device, dtype=grad.dtype),
                torch.full((), multiplier, device=grad.device, dtype=grad.dtype),
            )
            self._sh_channel_scale_cache[key] = scale
        return grad * scale

    def _register_sh_hooks(self):
        """(Re)register SH-rest gradient hooks on all level value tensors."""
        for handle in self._sh_hook_handles:
            handle.remove()
        self._sh_hook_handles = []
        # Learned MLP features have no SH/DC channel ordering. This method is
        # also called after refinement and checkpoint restoration in MLP mode.
        if self.color_mode == "mlp":
            return
        if self.color_sh_basis_dim <= 1 or self.color_sh_rest_lr_multiplier == 1.0:
            return
        self._sh_channel_scale_cache = {}
        for level in self.hier_field.levels:
            self._sh_hook_handles.append(
                level.values.register_hook(self._scale_hier_sh_rest_grad)
            )

    def _prepare_field(self):
        self._field.control_grid = self.raw_color_grid.permute(3, 4, 0, 1, 2).contiguous()

    def _apply(self, fn, recurse=True):
        # See BSplineSDFWrapper._apply — keep ``bounds`` on the model device.
        module = super()._apply(fn, recurse)
        self.bounds = fn(self.bounds)
        return module

    def _clamp_points(self, points_3D):
        """Clamp query points to the field bounds to avoid out-of-bounds errors."""
        shape = (1,) * (points_3D.ndim - 1) + (3,)
        lower = self.bounds[:, 0].reshape(shape).to(points_3D)
        upper = self.bounds[:, 1].reshape(shape).to(points_3D)
        eps = 1e-5
        return torch.clamp(points_3D, min=lower + eps, max=upper - eps)

    def forward(self, points_3D, normals, rays_unit, feats, app):
        del feats, app  # MLP mode uses points/rays_unit (+ optionally normals)
        points_3D = self._clamp_points(points_3D)
        if self.color_mode == "mlp":
            if getattr(self, "_per_level_dims", False):
                features = self.hier_field.evaluate_per_level(points_3D)
            else:
                features = self.hier_field.evaluate(points_3D)  # (..., feature_dim)
            # Concatenate feature, encoded query position, and encoded view direction.
            # Both points_3D and rays_unit share the leading sample shape.
            pos_enc = self._fourier_features(points_3D, self.color_mlp_pos_freqs)
            view_enc = self._fourier_features(rays_unit, self.color_mlp_view_freqs)
            inputs = [features, pos_enc, view_enc]
            if self.color_mlp_use_normal:
                inputs.append(normals)
            mlp_in = torch.cat(inputs, dim=-1)
            return torch.sigmoid(self._mlp(mlp_in))
        del normals
        if self.hierarchical_enabled:
            sh = self.hier_field.evaluate(points_3D)  # (..., 3*sh_dim)
            sh = sh.unflatten(-1, (3, self.color_sh_basis_dim))  # (..., 3, sh_dim)
            basis = spherical_harmonics_basis(rays_unit, self.color_sh_degree)
            rgb_logits = (sh * basis.unsqueeze(-2)).sum(dim=-1)
            return torch.sigmoid(rgb_logits)
        self._prepare_field()
        rgb = self._field.evaluate(points_3D, rays_unit)
        return rgb

    @torch.no_grad()
    def maybe_refine(self, iteration, sdf_wrapper=None, sdf_refine_info=None):
        """Refine the color hierarchy on the SDF schedule.

        The marked cells are computed from the SDF field (via ``sdf_wrapper``)
        so that both hierarchies cover the same surface band.  When the color
        hierarchy shares structure with the SDF hierarchy, ``sdf_refine_info``
        (the dict returned by ``sdf_wrapper.maybe_refine``) must be passed so
        the color field can update its coefficients without recomputing the
        grid topology.  Returns the refinement info dict or ``None``.
        """
        if not self.hierarchical_enabled:
            return None
        refine_iters = [int(i) for i in self.hier_cfg.get("refine_iters", [])]
        if iteration not in refine_iters:
            return None
        if sdf_wrapper is None or not sdf_wrapper.hierarchical_enabled:
            raise ValueError(
                "Hierarchical color refinement requires a hierarchical SDF wrapper."
            )
        round_idx = refine_iters.index(iteration)
        if self.hier_field.num_levels >= round_idx + 2:
            # This round already happened (e.g. resumed from a checkpoint
            # saved at exactly this iteration); do not refine twice.
            return None
        # The SDF wrapper owns the marking decision (SDF band or accumulated
        # render loss).  When the color grid differs from the SDF grid (e.g.
        # different spline degrees) the loss score is sampled onto the color
        # cell lattice; otherwise the source marking tensor is used directly.
        def mark_fn(level_idx, corners):
            level = self.hier_field.levels[level_idx]
            region = level.region  # pre-filter to parent's active region
            mode = str(sdf_wrapper.hier_cfg.get("refine_mark_mode", "sdf_band"))
            max_fraction = self.hier_cfg.get("refine_max_marked_fraction", None)
            mark_centers = bool(self.hier_cfg.get("refine_mark_centers", False))
            if mode in ("loss", "hybrid"):
                min_count = float(self.hier_cfg.get("refine_loss_min_count", 1.0))
                extra = cell_center_points(level) if mark_centers else None
                marked_loss = sdf_wrapper.loss_mark_cells_for_target(
                    level_idx, level, corners, extra, max_fraction, min_count
                )
                if mode == "loss" and marked_loss is not None:
                    return marked_loss
                if mode == "hybrid" and marked_loss is not None:
                    # Also include SDF-band cells; union covers loss + geometry surface.
                    band = self.hier_cfg.get("refine_sdf_band", 0.05)
                    if isinstance(band, (list, tuple)):
                        band = band[min(round_idx, len(band) - 1)]
                    extra = cell_center_points(level) if mark_centers else None
                    marked_band = mark_cells_by_value_band(
                        sdf_wrapper.sdf, level.cell_count, corners, float(band),
                        extra_points=extra, max_fraction=max_fraction,
                        region=region,
                    )
                    return marked_loss | marked_band
                print("[Bspline-Neus] loss/hybrid marking requested but no statistics "
                      "available for color; falling back to SDF band.")
            band = self.hier_cfg.get("refine_sdf_band", 0.05)
            if isinstance(band, (list, tuple)):
                band = band[min(round_idx, len(band) - 1)]
            extra = cell_center_points(level) if mark_centers else None
            return mark_cells_by_value_band(
                sdf_wrapper.sdf, level.cell_count, corners, float(band),
                extra_points=extra, max_fraction=max_fraction,
                region=region,
            )

        # When sharing structure with the SDF hierarchy, the SDF refine info dict
        # (or its inner ``owner_refine_info`` payload) must be forwarded so the
        # color field updates its coefficients using the owner's topology snapshot.
        owner_refine_info = sdf_refine_info
        if self.hier_field.structure_owner is not None:
            if isinstance(sdf_refine_info, dict) and "owner_refine_info" in sdf_refine_info:
                owner_refine_info = sdf_refine_info["owner_refine_info"]
            if owner_refine_info is None:
                raise ValueError(
                    "A structure-sharing color hierarchy requires the SDF refine info "
                    "(returned by sdf_wrapper.maybe_refine)."
                )

        info = self.hier_field.refine(mark_fn, owner_refine_info=owner_refine_info)
        if info.get("refined"):
            self._register_sh_hooks()
        return info

    @torch.no_grad()
    def maybe_reband(self, iteration, sdf_reband_info=None):
        """Sync color values after the SDF hierarchy re-bands the finest level.

        Only needed when sharing structure with the SDF hierarchy — the
        shared LevelStructure is already updated by the SDF's reband, but
        this hierarchy's ``values`` must be remapped to the new index grid.
        """
        if not self.hierarchical_enabled:
            return None
        reband_iters = [int(i) for i in self.hier_cfg.get("reband_iters", [])]
        if iteration not in reband_iters:
            return None
        if self.hier_field.structure_owner is not None:
            if sdf_reband_info is None or not sdf_reband_info.get("changed"):
                return None
            info = self.hier_field.sync_from_owner_reband(sdf_reband_info)
            if info.get("changed"):
                self._register_sh_hooks()
            return info
        # Non-shared color hierarchy re-bands independently.
        if self.hier_field.num_levels < 2:
            return None
        band = self.hier_cfg.get("reband_band", None)
        if band is None:
            bands = self.hier_cfg.get("refine_sdf_band", 0.05)
            band = bands[-1] if isinstance(bands, (list, tuple)) else bands
        max_frac = self.hier_cfg.get("reband_max_fraction", None)
        info = self.hier_field.reband_finest(
            float(band), max_fraction=max_frac, include_centers=True,
        )
        if info.get("changed"):
            self._register_sh_hooks()
        return info

    @torch.no_grad()
    def maybe_reband_all(self, iteration, sdf_reband_info=None):
        """Sync color values after the SDF hierarchy's reband_all.

        When sharing structure, the SDF's reband_all has already updated the
        shared LevelStructures; this remaps color values at every level.
        """
        if not self.hierarchical_enabled:
            return None
        reband_iters = [int(i) for i in self.hier_cfg.get("reband_all_iters", [])]
        if iteration not in reband_iters:
            return None
        if self.hier_field.structure_owner is not None:
            if sdf_reband_info is None or not sdf_reband_info.get("changed"):
                return None
            info = self.hier_field.sync_from_owner_reband_all(sdf_reband_info)
            if info.get("changed"):
                self._register_sh_hooks()
            return info
        # Non-shared: reband independently (same bands as SDF).
        if self.hier_field.num_levels < 2:
            return None
        n = self.hier_field.num_levels
        bands = self.hier_cfg.get("reband_all_bands", None)
        if bands is None:
            default_bands = self.hier_cfg.get("refine_sdf_band", 0.05)
            last = default_bands[-1] if isinstance(default_bands, (list, tuple)) else default_bands
            bands = [float(last)] * n
        max_fracs = self.hier_cfg.get("reband_all_max_fractions", None)
        info = self.hier_field.reband_all(bands, max_fractions=max_fracs)
        if info.get("changed"):
            self._register_sh_hooks()
        return info

    @torch.no_grad()
    def maybe_prune(self, iteration, sdf_prune_info=None):
        """Sync color values after the SDF hierarchy prunes the finest level.

        Mirrors :meth:`maybe_reband`: when sharing structure, the shared
        LevelStructure has already been updated by the owner's prune and only
        this hierarchy's ``values`` need remapping.  Without structure
        sharing there is no activity signal on the color side, so pruning is
        a no-op (the SDF-driven stats live in the SDF wrapper).
        """
        if not self.hierarchical_enabled:
            return None
        if self.hier_field.structure_owner is not None:
            # Shared structure: follow the owner's schedule (the SDF wrapper
            # owns the prune statistics and the iteration check).
            if sdf_prune_info is None or not sdf_prune_info.get("changed"):
                return None
            info = self.hier_field.sync_from_owner_reband(sdf_prune_info)
            if info.get("changed"):
                self._register_sh_hooks()
            return info
        return None
