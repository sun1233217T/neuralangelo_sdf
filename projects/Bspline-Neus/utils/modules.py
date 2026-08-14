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
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # Optional NeuS sharpness floor: inv_std() clamps the softplus output
        # at this value.  Gradients through clamp_min vanish below the floor,
        # which effectively pins inv_std at the floor (E2 experiment).
        self.inv_s_floor = getattr(cfg_bspline, "sdf_inv_s_floor", None)

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
            and str(self.hier_cfg.get("refine_mark_mode", "sdf_band")) == "loss"
        )
        self._loss_accum = None
        self._loss_count = None

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

    def inv_std(self) -> torch.Tensor:
        s = F.softplus(self.raw_sdf_inv_std)
        if self.inv_s_floor is not None:
            s = s.clamp_min(float(self.inv_s_floor))
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

    @torch.no_grad()
    def accumulate_loss(self, points, weights, rgb, target):
        """Scatter per-ray render error into the finest level's cell grid.

        ``points`` (B,R,N,3) and ``weights`` (B,R,N) are the (detached) object
        samples of the training rays; ``rgb``/``target`` are the per-ray
        rendered/GT colors (B,R,3).  Each sample receives the mass
        ``weight * |rgb-target|`` of its ray, accumulated additively per cell
        together with a visit count.  The buffers feed ``mark_cells_by_loss``
        at refinement time.  No-op unless loss marking is enabled and further
        refinement rounds remain (after the last refine the statistics would
        never be consumed, so the buffers are not even allocated).

        Graph-capture safe: no boolean indexing / host syncs — out-of-bounds
        samples are zeroed via a float mask instead of filtered.
        """
        if not self.loss_marking_enabled:
            return
        if self.hier_field.num_levels >= self.hier_field.max_levels:
            return
        level = self.hier_field.levels[-1]
        cc = level.cell_count
        if self._loss_accum is None or tuple(self._loss_accum.shape) != (cc,) * 3:
            device = level.values.device
            self._loss_accum = torch.zeros((cc,) * 3, dtype=torch.float32, device=device)
            self._loss_count = torch.zeros((cc,) * 3, dtype=torch.float32, device=device)
        err = (rgb.detach() - target).abs().mean(dim=-1)  # (B, R)
        mass = (weights.reshape(*err.shape, -1) * err.unsqueeze(-1)).reshape(-1)
        pts = points.reshape(-1, 3).to(dtype=torch.float32)
        # Out-of-place floor: points may be a view into graph-pool memory.
        idx = ((pts - level._lower) / level.step).floor().long()  # (M, 3)
        valid = ((idx >= 0) & (idx < cc)).all(dim=-1).to(mass.dtype)
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
        if mode == "loss":
            level = self.hier_field.levels[level_idx]
            cc = level.cell_count
            if (
                self._loss_accum is not None
                and tuple(self._loss_accum.shape) == (cc,) * 3
                and bool(self._loss_count.any())
            ):
                if max_fraction is None:
                    raise ValueError(
                        "refine_mark_mode=loss requires refine_max_marked_fraction "
                        "(loss scores have no natural absolute threshold)."
                    )
                min_count = float(self.hier_cfg.get("refine_loss_min_count", 1.0))
                return mark_cells_by_loss(
                    self._loss_accum, self._loss_count, max_fraction, min_count
                )
            print("[Bspline-Neus] loss marking requested but no statistics "
                  "available; falling back to SDF band.")
        band = self.hier_cfg.get("refine_sdf_band", 0.05)
        if isinstance(band, (list, tuple)):
            band = band[min(round_idx, len(band) - 1)]
        mark_centers = bool(self.hier_cfg.get("refine_mark_centers", False))
        return mark_cells_by_sdf_band(
            self.hier_field, level_idx, corners, float(band),
            include_centers=mark_centers,
            max_fraction=max_fraction,
        )

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
                self._loss_accum, self._loss_count, max_fraction, min_count
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

    def __init__(self, cfg_bspline):
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
            hidden_dims = list(getattr(cfg_bspline, "color_mlp_hidden_dims", [64, 64]))
            self.color_mlp_pos_freqs = int(getattr(cfg_bspline, "color_mlp_pos_freqs", 0))
            self.color_mlp_view_freqs = int(getattr(cfg_bspline, "color_mlp_view_freqs", 0))
            self.hier_field = HierarchicalBSplineField(
                self.grid_size,
                int(self.color_spline_degree),
                self.bounds,
                channels=self.color_feature_dim,
                max_levels=int(hier_cfg.get("max_levels", 1)),
                transfer_mode=str(
                    hier_cfg.get("color_transfer_mode") or hier_cfg.get("transfer_mode", "hb")
                ),
                dtype=dtype,
                device=device,
            )
            # Initialize feature grid near zero; MLP outputs raw RGB logits.
            self.hier_field.levels[0].values.data.zero_()
            pos_dim = 3 + 2 * 3 * self.color_mlp_pos_freqs if self.color_mlp_pos_freqs > 0 else 3
            view_dim = 3 + 2 * 3 * self.color_mlp_view_freqs if self.color_mlp_view_freqs > 0 else 3
            mlp_in_dim = self.color_feature_dim + pos_dim + view_dim
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
        del normals, feats, app  # SH mode ignores these; MLP mode uses points/rays_unit directly
        points_3D = self._clamp_points(points_3D)
        if self.color_mode == "mlp":
            features = self.hier_field.evaluate(points_3D)  # (..., feature_dim)
            # Concatenate feature, encoded query position, and encoded view direction.
            # Both points_3D and rays_unit share the leading sample shape.
            pos_enc = self._fourier_features(points_3D, self.color_mlp_pos_freqs)
            view_enc = self._fourier_features(rays_unit, self.color_mlp_view_freqs)
            mlp_in = torch.cat([features, pos_enc, view_enc], dim=-1)
            return torch.sigmoid(self._mlp(mlp_in))
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
    def maybe_refine(self, iteration, sdf_wrapper=None):
        """Refine the color hierarchy on the SDF schedule.

        The marked cells are computed from the SDF field (via ``sdf_wrapper``)
        so that both hierarchies cover the same surface band.  Returns the
        refinement info dict or ``None``.
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
            mode = str(sdf_wrapper.hier_cfg.get("refine_mark_mode", "sdf_band"))
            max_fraction = self.hier_cfg.get("refine_max_marked_fraction", None)
            mark_centers = bool(self.hier_cfg.get("refine_mark_centers", False))
            if mode == "loss":
                min_count = float(self.hier_cfg.get("refine_loss_min_count", 1.0))
                extra = cell_center_points(level) if mark_centers else None
                marked = sdf_wrapper.loss_mark_cells_for_target(
                    level_idx, level, corners, extra, max_fraction, min_count
                )
                if marked is not None:
                    return marked
                print("[Bspline-Neus] loss marking requested but no statistics "
                      "available for color; falling back to SDF band.")
            band = self.hier_cfg.get("refine_sdf_band", 0.05)
            if isinstance(band, (list, tuple)):
                band = band[min(round_idx, len(band) - 1)]
            extra = cell_center_points(level) if mark_centers else None
            return mark_cells_by_value_band(
                sdf_wrapper.sdf, level.cell_count, corners, float(band),
                extra_points=extra, max_fraction=max_fraction,
            )

        info = self.hier_field.refine(mark_fn)
        if info.get("refined"):
            self._register_sh_hooks()
        return info
