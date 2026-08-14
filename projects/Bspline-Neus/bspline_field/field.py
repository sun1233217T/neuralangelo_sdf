from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch.autograd.profiler import record_function
from torch import Tensor, nn

Bounds = Sequence[Sequence[float]]

_DEFAULT_BOUNDS: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
)

_GAUSS_LEGENDRE_5_NODES = (
    -0.9061798459386640,
    -0.5384693101056831,
    0.0,
    0.5384693101056831,
    0.9061798459386640,
)
_GAUSS_LEGENDRE_5_WEIGHTS = (
    0.2369268850561891,
    0.4786286704993665,
    0.5688888888888889,
    0.4786286704993665,
    0.2369268850561891,
)
_SH_C0 = 0.28209479177387814
_SH_C1 = 0.4886025119029199
_SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
_SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def _as_float_tensor(
    value: Any,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    if torch.is_tensor(value):
        tensor = value
        if dtype is not None or device is not None:
            tensor = tensor.to(dtype=dtype or tensor.dtype, device=device or tensor.device)
        if not torch.is_floating_point(tensor):
            tensor = tensor.to(dtype=dtype or torch.get_default_dtype())
        return tensor
    return torch.as_tensor(
        value,
        dtype=dtype or torch.get_default_dtype(),
        device=device,
    )


def _inverse_sigmoid(value: float) -> float:
    clipped = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


def _coerce_spline_degree(spline_degree: int) -> int:
    degree = int(spline_degree)
    if degree < 0 or degree > 3:
        raise ValueError("Only uniform B-spline degrees 0 through 3 are currently supported.")
    return degree


def _coerce_sh_degree(sh_degree: int) -> int:
    degree = int(sh_degree)
    if degree < 0 or degree > 3:
        raise ValueError("Only real spherical harmonics degrees 0 through 3 are currently supported.")
    return degree


def spherical_harmonics_basis(ray_directions: Any, sh_degree: int) -> Tensor:
    directions = _as_float_tensor(ray_directions)
    scalar_input = tuple(directions.shape) == (3,)
    if scalar_input:
        directions = directions.reshape(1, 3)
    if directions.ndim < 1 or directions.shape[-1] != 3:
        raise ValueError("ray_directions must have shape (3,) or (..., 3).")

    degree = _coerce_sh_degree(sh_degree)
    eps = torch.finfo(directions.dtype).eps
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(eps)
    x, y, z = directions.unbind(dim=-1)

    basis = [torch.full_like(x, _SH_C0)]
    if degree >= 1:
        basis.extend(
            (
                -_SH_C1 * y,
                _SH_C1 * z,
                -_SH_C1 * x,
            )
        )
    if degree >= 2:
        basis.extend(
            (
                _SH_C2[0] * x * y,
                _SH_C2[1] * y * z,
                _SH_C2[2] * (2.0 * z * z - x * x - y * y),
                _SH_C2[3] * x * z,
                _SH_C2[4] * (x * x - y * y),
            )
        )
    if degree >= 3:
        basis.extend(
            (
                _SH_C3[0] * y * (3.0 * x * x - y * y),
                _SH_C3[1] * x * y * z,
                _SH_C3[2] * y * (4.0 * z * z - x * x - y * y),
                _SH_C3[3] * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y),
                _SH_C3[4] * x * (4.0 * z * z - x * x - y * y),
                _SH_C3[5] * z * (x * x - y * y),
                _SH_C3[6] * x * (x * x - 3.0 * y * y),
            )
        )

    values = torch.stack(basis, dim=-1)
    if scalar_input:
        return values[0]
    return values


def _split_rgb_values(value: Any, *, name: str, allow_scalar: bool = False) -> tuple[Any, Any, Any]:
    if torch.is_tensor(value):
        if value.ndim >= 1 and value.shape[0] == 3:
            return value[0], value[1], value[2]
        if allow_scalar and value.ndim == 0:
            return value, value, value

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 3:
            return value[0], value[1], value[2]

    if allow_scalar:
        return value, value, value

    raise ValueError(f"{name} must provide exactly three channel values.")


def basis_1d(u: Any, degree: int = 3) -> Tensor:
    """Return uniform B-spline basis weights on a single cell."""

    # 这里对应 C++ 里的 basisFunc：
    # 输入区间内局部坐标 u，输出 4 个三次 B-spline 基函数值。
    uu = _as_float_tensor(u)
    degree = _coerce_spline_degree(degree)
    uu = _as_float_tensor(u)
    if degree == 0:
        return torch.ones_like(uu).unsqueeze(dim=-1)
    if degree == 1:
        return torch.stack(
            (
                1.0 - uu,
                uu,
            ),
            dim=-1,
        )
    u2 = uu * uu
    if degree == 2:
        return torch.stack(
            (
                (1.0 - uu) ** 2,
                -2.0 * u2 + 2.0 * uu + 1.0,
                u2,
            ),
            dim=-1,
        ) / 2.0
    u3 = u2 * uu
    return torch.stack(
        (
            (1.0 - uu) ** 3,
            3.0 * u3 - 6.0 * u2 + 4.0,
            -3.0 * u3 + 3.0 * u2 + 3.0 * uu + 1.0,
            u3,
        ),
        dim=-1,
    ) / 6.0


def basis_1d_with_deriv(u: Any, degree: int = 3) -> tuple[Tensor, Tensor, Tensor]:
    """Return (w, dw, d2w)：基函数值及其对局部坐标 u 的一/二阶闭式导数。

    与 ``basis_1d`` 完全对应；对全局坐标的导数由调用方乘链式因子
    du/dx = 1/step（一阶）和 1/step^2（二阶）。
    """
    uu = _as_float_tensor(u)
    degree = _coerce_spline_degree(degree)
    if degree == 0:
        w = torch.ones_like(uu).unsqueeze(dim=-1)
        zeros = torch.zeros_like(w)
        return w, zeros, zeros
    if degree == 1:
        w = torch.stack(
            (
                1.0 - uu,
                uu,
            ),
            dim=-1,
        )
        dw = torch.stack(
            (
                -torch.ones_like(uu),
                torch.ones_like(uu),
            ),
            dim=-1,
        )
        return w, dw, torch.zeros_like(w)
    u2 = uu * uu
    if degree == 2:
        w = torch.stack(
            (
                (1.0 - uu) ** 2,
                -2.0 * u2 + 2.0 * uu + 1.0,
                u2,
            ),
            dim=-1,
        ) / 2.0
        dw = torch.stack(
            (
                -2.0 * (1.0 - uu),
                -4.0 * uu + 2.0,
                2.0 * uu,
            ),
            dim=-1,
        ) / 2.0
        d2w = torch.stack(
            (
                2.0 * torch.ones_like(uu),
                -4.0 * torch.ones_like(uu),
                2.0 * torch.ones_like(uu),
            ),
            dim=-1,
        ) / 2.0
        return w, dw, d2w
    u3 = u2 * uu
    one_minus_u = 1.0 - uu
    w = torch.stack(
        (
            one_minus_u ** 3,
            3.0 * u3 - 6.0 * u2 + 4.0,
            -3.0 * u3 + 3.0 * u2 + 3.0 * uu + 1.0,
            u3,
        ),
        dim=-1,
    ) / 6.0
    dw = torch.stack(
        (
            -3.0 * one_minus_u ** 2,
            9.0 * u2 - 12.0 * uu,
            -9.0 * u2 + 6.0 * uu + 3.0,
            3.0 * u2,
        ),
        dim=-1,
    ) / 6.0
    d2w = torch.stack(
        (
            6.0 * one_minus_u,
            18.0 * uu - 12.0,
            -18.0 * uu + 6.0,
            6.0 * uu,
        ),
        dim=-1,
    ) / 6.0
    return w, dw, d2w


class BSplineField(nn.Module):
    """3D tensor-product B-spline scalar field backed by torch tensors."""

    def __init__(
        self,
        grid_size: int,
        control_coefficients: Any,
        bounds: Bounds = _DEFAULT_BOUNDS,
        *,
        spline_degree: int = 3,
        trainable: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        self.grid_size = int(grid_size)
        self.spline_degree = _coerce_spline_degree(spline_degree)
        self.support_size = self.spline_degree + 1
        if self.grid_size < self.support_size:
            raise ValueError(
                f"grid_size must be at least {self.support_size} for degree-{self.spline_degree} B-splines."
            )

        # 和原 C++ 一样，真正的单元数是 grid_size - 3。
        # 原因是每个采样点会访问当前位置周围 4 个控制点。
        self.cell_count = self.grid_size - self.spline_degree

        control_grid = self._coerce_control_grid(
            control_coefficients,
            dtype=dtype,
            device=device,
        )
        bounds_tensor = self._coerce_bounds(
            bounds,
            dtype=control_grid.dtype,
            device=control_grid.device,
        )

        self.register_buffer("_bounds", bounds_tensor)
        self.register_buffer("_lower", bounds_tensor[:, 0].clone())
        self.register_buffer("_upper", bounds_tensor[:, 1].clone())
        self.register_buffer("_extent", self._upper - self._lower)
        self.register_buffer("_max_extent", self._extent.amax())
        self.register_buffer("step", self._extent / self.cell_count)
        support_offsets_flat, support_axis_r, support_axis_s, support_axis_t = self._build_support_index_buffers()
        support_offsets_flat = support_offsets_flat.to(device=control_grid.device)
        support_axis_r = support_axis_r.to(device=control_grid.device)
        support_axis_s = support_axis_s.to(device=control_grid.device)
        support_axis_t = support_axis_t.to(device=control_grid.device)
        self.register_buffer("_support_offsets_flat", support_offsets_flat)
        self.register_buffer("_support_axis_r", support_axis_r)
        self.register_buffer("_support_axis_s", support_axis_s)
        self.register_buffer("_support_axis_t", support_axis_t)

        if trainable:
            self.control_grid = nn.Parameter(control_grid)
        else:
            self.register_buffer("control_grid", control_grid)

    @classmethod
    def zeros(
        cls,
        grid_size: int,
        bounds: Bounds = _DEFAULT_BOUNDS,
        fill_value: float = 0.0,
        *,
        spline_degree: int = 3,
        trainable: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> "BSplineField":
        coeffs = torch.full(
            (grid_size, grid_size, grid_size),
            fill_value=fill_value,
            dtype=dtype or torch.get_default_dtype(),
            device=device,
        )
        return cls(
            grid_size=grid_size,
            control_coefficients=coeffs,
            bounds=bounds,
            spline_degree=spline_degree,
            trainable=trainable,
        )

    @classmethod
    def from_flat_coefficients(
        cls,
        grid_size: int,
        control_coefficients: Any,
        bounds: Bounds = _DEFAULT_BOUNDS,
        *,
        spline_degree: int = 3,
        trainable: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> "BSplineField":
        return cls(
            grid_size=grid_size,
            control_coefficients=control_coefficients,
            bounds=bounds,
            spline_degree=spline_degree,
            trainable=trainable,
            dtype=dtype,
            device=device,
        )

    @property
    def bounds(self) -> Tensor:
        return self._bounds

    def _coerce_bounds(
        self,
        bounds: Bounds,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> Tensor:
        tensor = _as_float_tensor(bounds, dtype=dtype, device=device)
        if tensor.shape != (3, 2):
            raise ValueError("bounds must have shape (3, 2).")
        if torch.any(tensor[:, 1] <= tensor[:, 0]).item():
            raise ValueError("Each bounds entry must satisfy min < max.")
        return tensor

    def _coerce_control_grid(
        self,
        control_coefficients: Any,
        *,
        dtype: torch.dtype | None,
        device: torch.device | str | None,
    ) -> Tensor:
        coeffs = _as_float_tensor(control_coefficients, dtype=dtype, device=device)
        expected_shape = (self.grid_size, self.grid_size, self.grid_size)
        expected_size = self.grid_size ** 3

        if tuple(coeffs.shape) == expected_shape:
            return coeffs.clone()

        if coeffs.ndim == 1 and coeffs.numel() == expected_size:
            return coeffs.reshape(expected_shape).clone()

        raise ValueError(
            "control_coefficients must have shape "
            f"{expected_shape} or be a flat array of length {expected_size}."
        )

    def _build_support_index_buffers(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        axis_ids = torch.arange(self.support_size, dtype=torch.long)
        support_triplets = torch.cartesian_prod(axis_ids, axis_ids, axis_ids)
        grid_size_sq = self.grid_size * self.grid_size
        support_offsets_flat = (
            support_triplets[:, 0] * grid_size_sq
            + support_triplets[:, 1] * self.grid_size
            + support_triplets[:, 2]
        )
        return (
            support_offsets_flat,
            support_triplets[:, 0],
            support_triplets[:, 1],
            support_triplets[:, 2],
        )

    def set_control_coefficients(self, control_coefficients: Any) -> None:
        new_grid = self._coerce_control_grid(
            control_coefficients,
            dtype=self.control_grid.dtype,
            device=self.control_grid.device,
        )
        with torch.no_grad():
            self.control_grid.copy_(new_grid)

    def flat_control_coefficients(self) -> Tensor:
        # 按 C++ 的线性索引顺序展平：
        # idx = i * grid_size^2 + j * grid_size + k
        return self.control_grid.reshape(-1)

    def forward(self, x: Any, y: Any | None = None, z: Any | None = None) -> Tensor:
        if y is None and z is None:
            return self.evaluate(x)
        if y is None or z is None:
            raise TypeError("Pass either a single (..., 3) points tensor or x, y, z together.")
        return self.evaluate_xyz(x, y, z)

    def evaluate_xyz(self, x: Any, y: Any, z: Any) -> Tensor:
        # 支持 torch 广播，这样可以直接传标量、向量或网格。
        tx = _as_float_tensor(x, dtype=self.control_grid.dtype, device=self.control_grid.device)
        ty = _as_float_tensor(y, dtype=self.control_grid.dtype, device=self.control_grid.device)
        tz = _as_float_tensor(z, dtype=self.control_grid.dtype, device=self.control_grid.device)
        bx, by, bz = torch.broadcast_tensors(tx, ty, tz)
        points = torch.stack((bx, by, bz), dim=-1)
        return self.evaluate(points)

    def _prepare_point_evaluation_context(
        self,
        points: Any,
    ) -> tuple[tuple[int, ...], bool, Tensor, Tensor]:
        query_points, output_shape, scalar_input = self._prepare_points(points)
        query_points = self._validate_points(query_points)

        # 先把全局坐标映射到“格子坐标系”。
        # local_coordinates = (点 - 边界下界) / 单元步长
        local_coordinates = (query_points - self._lower) / self.step
        local_coordinates = torch.minimum(
            local_coordinates,
            torch.nextafter(
                torch.full(
                    (3,),
                    float(self.cell_count),
                    dtype=local_coordinates.dtype,
                    device=local_coordinates.device,
                ),
                torch.zeros(3, dtype=local_coordinates.dtype, device=local_coordinates.device),
            ),
        )
        # 这里对上边界做一个极小量回退，避免恰好落在 xmax/ymax/zmax 时
        # floor 后跑到最后一个合法单元之外。
        ijk = torch.floor(local_coordinates).to(torch.long)
        uvw = local_coordinates - ijk.to(local_coordinates.dtype)

        # 分别计算 x/y/z 三个方向上的 1D 基函数值，
        # 再在后面做 3D 张量积。
        basis_x = basis_1d(uvw[:, 0], degree=self.spline_degree)
        basis_y = basis_1d(uvw[:, 1], degree=self.spline_degree)
        basis_z = basis_1d(uvw[:, 2], degree=self.spline_degree)

        i = ijk[:, 0]
        j = ijk[:, 1]
        k = ijk[:, 2]

        grid_size_sq = self.grid_size * self.grid_size
        base_flat_indices = i * grid_size_sq + j * self.grid_size + k
        support_flat_indices = base_flat_indices[:, None] + self._support_offsets_flat[None, :]

        basis_weights = (
            basis_x.index_select(1, self._support_axis_r)
            * basis_y.index_select(1, self._support_axis_s)
            * basis_z.index_select(1, self._support_axis_t)
        )
        return output_shape, scalar_input, support_flat_indices, basis_weights

    def _prepare_point_evaluation_context_with_deriv(
        self,
        points: Any,
    ) -> tuple[tuple[int, ...], bool, Tensor, tuple[Tensor, ...]]:
        """与 _prepare_point_evaluation_context 同构，额外给出 6 组导数权重。

        返回 (output_shape, scalar_input, support_flat_indices, weight_groups)，
        weight_groups 按 (value, gx, gy, gz, hxx, hyy, hzz) 顺序，均为 (N, S)。
        对全局坐标的偏导只需把对应轴的 1D 基换成导数基，再乘链式因子
        1/step（一阶）或 1/step^2（二阶对角 Hessian）。

        注意：这里刻意跳过 ``_validate_points`` 的 .item() 越界检查
        （host sync，CUDA graph capture 不安全），只做静默 clamp；
        上层包装（``BSplineSDFWrapper._clamp_points``）已保证点在体内。
        """
        query_points, output_shape, scalar_input = self._prepare_points(points)
        query_points = torch.minimum(torch.maximum(query_points, self._lower), self._upper)

        # 同 _prepare_point_evaluation_context：全局坐标 -> 格子坐标系。
        local_coordinates = (query_points - self._lower) / self.step
        local_coordinates = torch.minimum(
            local_coordinates,
            torch.nextafter(
                torch.full(
                    (3,),
                    float(self.cell_count),
                    dtype=local_coordinates.dtype,
                    device=local_coordinates.device,
                ),
                torch.zeros(3, dtype=local_coordinates.dtype, device=local_coordinates.device),
            ),
        )
        ijk = torch.floor(local_coordinates).to(torch.long)
        uvw = local_coordinates - ijk.to(local_coordinates.dtype)

        # 三轴同时算基函数值和对 u 的一/二阶导数。
        basis_x, basis_dx, basis_d2x = basis_1d_with_deriv(uvw[:, 0], degree=self.spline_degree)
        basis_y, basis_dy, basis_d2y = basis_1d_with_deriv(uvw[:, 1], degree=self.spline_degree)
        basis_z, basis_dz, basis_d2z = basis_1d_with_deriv(uvw[:, 2], degree=self.spline_degree)

        i = ijk[:, 0]
        j = ijk[:, 1]
        k = ijk[:, 2]

        grid_size_sq = self.grid_size * self.grid_size
        base_flat_indices = i * grid_size_sq + j * self.grid_size + k
        support_flat_indices = base_flat_indices[:, None] + self._support_offsets_flat[None, :]

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
        return output_shape, scalar_input, support_flat_indices, weight_groups

    def _evaluate_control_tensor_with_deriv(
        self,
        control_coefficients: Any,
        points: Any,
    ) -> tuple[Tensor, Tensor, Tensor]:
        control_tensor = _as_float_tensor(
            control_coefficients,
            dtype=self.control_grid.dtype,
            device=self.control_grid.device,
        )
        if tuple(control_tensor.shape[-3:]) != (self.grid_size, self.grid_size, self.grid_size):
            raise ValueError(
                "control_coefficients must have trailing shape "
                f"({self.grid_size}, {self.grid_size}, {self.grid_size})."
            )

        output_shape, scalar_input, support_flat_indices, weight_groups = (
            self._prepare_point_evaluation_context_with_deriv(points)
        )

        feature_shape = tuple(control_tensor.shape[:-3])
        num_points, support_size = support_flat_indices.shape
        control_flat = control_tensor.reshape(-1, self.grid_size * self.grid_size * self.grid_size)
        flat_support_indices = support_flat_indices.reshape(-1)
        control_values = control_flat.index_select(1, flat_support_indices)
        control_values = control_values.reshape(control_flat.shape[0], num_points, support_size)

        # gather 只做一次；逐组循环加权求和，避免堆出 (C, N, S, 7) 大张量。
        reduced = [
            torch.sum(control_values * weights.unsqueeze(0), dim=-1).transpose(0, 1)
            for weights in weight_groups
        ]  # 7 x (N, *feature_shape)

        values = reduced[0].reshape((num_points,) + feature_shape)
        gradients = torch.stack(reduced[1:4], dim=-1).reshape((num_points,) + feature_shape + (3,))
        hessian_diag = torch.stack(reduced[4:7], dim=-1).reshape((num_points,) + feature_shape + (3,))

        if scalar_input:
            return values[0], gradients[0], hessian_diag[0]
        return (
            values.reshape(output_shape + feature_shape),
            gradients.reshape(output_shape + feature_shape + (3,)),
            hessian_diag.reshape(output_shape + feature_shape + (3,)),
        )

    def evaluate_with_deriv(self, points: Any) -> tuple[Tensor, Tensor, Tensor]:
        """解析求值：返回 (values, gradients, hessian_diag)。

        values 形状 (..., *feature_shape)（与 evaluate 一致）；
        gradients / hessian_diag 形状 (..., *feature_shape, 3)，其中
        hessian_diag 是对角 Hessian（∂²f/∂x², ∂²f/∂y², ∂²f/∂z²）。
        场对控制点线性，所以返回值自然保留对控制点参数的一阶梯度图。
        """
        return self._evaluate_control_tensor_with_deriv(self.control_grid, points)

    def _evaluate_control_tensor(self, control_coefficients: Any, points: Any) -> Tensor:
        with record_function("field.evaluate_control.coerce"):
            control_tensor = _as_float_tensor(
                control_coefficients,
                dtype=self.control_grid.dtype,
                device=self.control_grid.device,
            )
            if tuple(control_tensor.shape[-3:]) != (self.grid_size, self.grid_size, self.grid_size):
                raise ValueError(
                    "control_coefficients must have trailing shape "
                    f"({self.grid_size}, {self.grid_size}, {self.grid_size})."
                )

        with record_function("field.evaluate_control.prepare_context"):
            output_shape, scalar_input, support_flat_indices, basis_weights = self._prepare_point_evaluation_context(points)

        with record_function("field.evaluate_control.gather_support"):
            feature_shape = tuple(control_tensor.shape[:-3])
            num_points, support_size = support_flat_indices.shape
            control_flat = control_tensor.reshape(-1, self.grid_size * self.grid_size * self.grid_size)
            flat_support_indices = support_flat_indices.reshape(-1)
            control_values = control_flat.index_select(1, flat_support_indices)
            control_values = control_values.reshape(control_flat.shape[0], num_points, support_size)

        with record_function("field.evaluate_control.reduce"):
            weighted_values = torch.sum(control_values * basis_weights.unsqueeze(0), dim=-1).transpose(0, 1)
            values = weighted_values.reshape((num_points,) + feature_shape)

        if scalar_input:
            return values[0]
        return values.reshape(output_shape + feature_shape)

    def evaluate(self, points: Any) -> Tensor:
        return self._evaluate_control_tensor(self.control_grid, points)

    def evaluate_gradient(self, points: Any) -> Tensor:
        query_points, output_shape, scalar_input = self._prepare_points(points)
        points_for_grad = query_points.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            values = self.evaluate(points_for_grad)
            gradients = torch.autograd.grad(values.sum(), points_for_grad, create_graph=True)[0]
        if scalar_input:
            return gradients[0]
        return gradients.reshape(output_shape + (3,))

    def _prepare_rays(self, ray_origins: Any, ray_dirs: Any) -> tuple[Tensor, Tensor]:
        origins = _as_float_tensor(
            ray_origins,
            dtype=self.control_grid.dtype,
            device=self.control_grid.device,
        )
        dirs = _as_float_tensor(
            ray_dirs,
            dtype=self.control_grid.dtype,
            device=self.control_grid.device,
        )

        if tuple(origins.shape) == (3,):
            origins = origins.reshape(1, 3)
        if tuple(dirs.shape) == (3,):
            dirs = dirs.reshape(1, 3)

        if origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("ray_origins must have shape (3,) or (N, 3).")
        if dirs.ndim != 2 or dirs.shape[1] != 3:
            raise ValueError("ray_dirs must have shape (3,) or (N, 3).")

        if origins.shape[0] == 1 and dirs.shape[0] > 1:
            origins = origins.expand(dirs.shape[0], 3)
        elif dirs.shape[0] == 1 and origins.shape[0] > 1:
            dirs = dirs.expand(origins.shape[0], 3)
        elif origins.shape[0] != dirs.shape[0]:
            raise ValueError("ray_origins and ray_dirs must broadcast to the same batch size.")

        return origins.contiguous(), dirs.contiguous()

    def _coerce_ray_bounds(self, value: Any | None, num_rays: int, default: float) -> Tensor:
        if value is None:
            return torch.full(
                (num_rays,),
                fill_value=default,
                dtype=self.control_grid.dtype,
                device=self.control_grid.device,
            )

        bound = _as_float_tensor(
            value,
            dtype=self.control_grid.dtype,
            device=self.control_grid.device,
        )
        if bound.ndim == 0:
            return bound.reshape(1).expand(num_rays).contiguous()
        if bound.ndim == 1 and bound.shape == (1,):
            return bound.expand(num_rays).contiguous()
        if bound.ndim == 1 and bound.shape == (num_rays,):
            return bound.contiguous()
        raise ValueError(f"ray bounds must be scalar or shape ({num_rays},).")

    def _coerce_segment_matrix(self, value: Any, num_rays: int, name: str) -> Tensor:
        tensor = _as_float_tensor(
            value,
            dtype=self.control_grid.dtype,
            device=self.control_grid.device,
        )
        if tensor.ndim == 1:
            if num_rays == 1:
                return tensor.reshape(1, -1).contiguous()
            if tensor.shape == (num_rays,):
                return tensor.reshape(num_rays, 1).contiguous()
        if tensor.ndim == 2 and tensor.shape[0] == num_rays:
            return tensor.contiguous()
        raise ValueError(f"{name} must have shape ({num_rays}, M) or broadcast from a matching vector.")

    def _coerce_valid_mask(self, valid_mask: Any | None, shape: torch.Size, device: torch.device) -> Tensor:
        if valid_mask is None:
            return torch.ones(shape, dtype=torch.bool, device=device)
        mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)
        if mask.shape != shape:
            raise ValueError(f"valid_mask must have shape {tuple(shape)}.")
        return mask.contiguous()

    def _ray_box_intersections(
        self,
        ray_origins: Tensor,
        ray_dirs: Tensor,
        t_min: Tensor,
        t_max: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        eps = torch.finfo(ray_origins.dtype).eps * 16.0
        parallel = ray_dirs.abs() <= eps
        safe_dirs = torch.where(parallel, torch.ones_like(ray_dirs), ray_dirs)

        t0 = (self._lower - ray_origins) / safe_dirs
        t1 = (self._upper - ray_origins) / safe_dirs
        axis_entry = torch.minimum(t0, t1)
        axis_exit = torch.maximum(t0, t1)

        inside = (ray_origins >= self._lower) & (ray_origins <= self._upper)
        neg_inf = torch.full_like(axis_entry, float("-inf"))
        pos_inf = torch.full_like(axis_exit, float("inf"))
        axis_entry = torch.where(parallel & inside, neg_inf, axis_entry)
        axis_exit = torch.where(parallel & inside, pos_inf, axis_exit)
        axis_entry = torch.where(parallel & ~inside, pos_inf, axis_entry)
        axis_exit = torch.where(parallel & ~inside, neg_inf, axis_exit)

        t_entry = torch.maximum(axis_entry.max(dim=1).values, t_min)
        t_exit = torch.minimum(axis_exit.min(dim=1).values, t_max)
        hit_mask = t_exit > t_entry
        return t_entry, t_exit, hit_mask

    def _collect_single_ray_segments(
        self,
        ray_origin: Tensor,
        ray_dir: Tensor,
        t_entry: Tensor,
        t_exit: Tensor,
    ) -> tuple[Tensor, Tensor]:
        eps = torch.finfo(ray_origin.dtype).eps * 16.0
        boundaries = [t_entry.reshape(1), t_exit.reshape(1)]

        for axis in range(3):
            if ray_dir[axis].abs() <= eps:
                continue

            plane_ids = torch.arange(
                1,
                self.cell_count,
                dtype=ray_origin.dtype,
                device=ray_origin.device,
            )
            if plane_ids.numel() == 0:
                continue

            plane_positions = self._lower[axis] + plane_ids * self.step[axis]
            t_planes = (plane_positions - ray_origin[axis]) / ray_dir[axis]
            mask = (t_planes > t_entry + eps) & (t_planes < t_exit - eps)
            if mask.any():
                boundaries.append(t_planes[mask])

        boundaries = torch.sort(torch.cat(boundaries)).values
        if boundaries.numel() > 1:
            keep = torch.ones(boundaries.numel(), dtype=torch.bool, device=boundaries.device)
            keep[1:] = (boundaries[1:] - boundaries[:-1]).abs() > eps
            boundaries = boundaries[keep]

        if boundaries.numel() < 2:
            empty = torch.empty(0, dtype=ray_origin.dtype, device=ray_origin.device)
            return empty, empty

        t_starts = boundaries[:-1]
        t_ends = boundaries[1:]
        valid = t_ends > t_starts + eps
        return t_starts[valid], t_ends[valid]

    def _stack_segment_boundaries(
        self,
        ray_origins: Tensor,
        ray_dirs: Tensor,
        t_entry: Tensor,
        t_exit: Tensor,
        hit_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        num_rays = ray_origins.shape[0]
        if num_rays == 0:
            empty = torch.empty((0, 0), dtype=ray_origins.dtype, device=ray_origins.device)
            return empty, empty, torch.empty((0, 0), dtype=torch.bool, device=ray_origins.device)

        with record_function("field.segment_boundaries.prepare_candidates"):
            eps = torch.finfo(ray_origins.dtype).eps * 16.0
            inf = torch.full((num_rays, 1), float("inf"), dtype=ray_origins.dtype, device=ray_origins.device)

            entry_candidates = torch.where(hit_mask[:, None], t_entry[:, None], inf)
            exit_candidates = torch.where(hit_mask[:, None], t_exit[:, None], inf)
            candidate_values = [entry_candidates, exit_candidates]
            candidate_valid = [hit_mask[:, None], hit_mask[:, None]]

            plane_ids = torch.arange(
                1,
                self.cell_count,
                dtype=ray_origins.dtype,
                device=ray_origins.device,
            )
            if plane_ids.numel() > 0:
                for axis in range(3):
                    plane_positions = self._lower[axis] + plane_ids * self.step[axis]
                    ray_dir_axis = ray_dirs[:, axis : axis + 1]
                    ray_origin_axis = ray_origins[:, axis : axis + 1]
                    non_parallel = ray_dir_axis.abs() > eps
                    safe_dir_axis = torch.where(non_parallel, ray_dir_axis, torch.ones_like(ray_dir_axis))
                    t_planes = (plane_positions[None, :] - ray_origin_axis) / safe_dir_axis
                    valid_planes = (
                        hit_mask[:, None]
                        & non_parallel
                        & (t_planes > t_entry[:, None] + eps)
                        & (t_planes < t_exit[:, None] - eps)
                    )
                    candidate_values.append(torch.where(valid_planes, t_planes, torch.full_like(t_planes, float("inf"))))
                    candidate_valid.append(valid_planes)

        with record_function("field.segment_boundaries.sort_and_unique"):
            boundaries = torch.cat(candidate_values, dim=1)
            boundary_valid = torch.cat(candidate_valid, dim=1)
            sorted_boundaries, sort_indices = torch.sort(boundaries, dim=1)
            sorted_valid = torch.gather(boundary_valid, 1, sort_indices)

            keep = sorted_valid.clone()
            if keep.shape[1] > 1:
                distinct = (sorted_boundaries[:, 1:] - sorted_boundaries[:, :-1]).abs() > eps
                keep[:, 1:] = sorted_valid[:, 1:] & (~sorted_valid[:, :-1] | distinct)

            boundary_counts = keep.sum(dim=1)
        max_boundaries = int(boundary_counts.max().item()) if boundary_counts.numel() > 0 else 0
        if max_boundaries < 2:
            empty = torch.empty((num_rays, 0), dtype=ray_origins.dtype, device=ray_origins.device)
            return empty, empty, torch.empty((num_rays, 0), dtype=torch.bool, device=ray_origins.device)

        with record_function("field.segment_boundaries.pack"):
            packed_boundaries = torch.zeros(
                (num_rays, max_boundaries),
                dtype=ray_origins.dtype,
                device=ray_origins.device,
            )
            boundary_positions = torch.cumsum(keep.to(torch.long), dim=1) - 1
            row_indices = torch.arange(num_rays, device=ray_origins.device)[:, None].expand_as(keep)
            packed_boundaries[row_indices[keep], boundary_positions[keep]] = sorted_boundaries[keep]

            t_starts = packed_boundaries[:, :-1]
            t_ends = packed_boundaries[:, 1:]
            segment_positions = torch.arange(max_boundaries - 1, device=ray_origins.device)[None, :]
            valid_mask = segment_positions < (boundary_counts - 1).clamp_min(0)[:, None]
            valid_mask = valid_mask & (t_ends > t_starts + eps)

            t_starts = torch.where(valid_mask, t_starts, torch.zeros_like(t_starts))
            t_ends = torch.where(valid_mask, t_ends, torch.zeros_like(t_ends))
        return t_starts, t_ends, valid_mask

    def _prepare_segment_integration_context(
        self,
        ray_origins: Tensor,
        ray_dirs: Tensor,
        t_starts: Tensor,
        t_ends: Tensor,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        with record_function("field.segment_context.quadrature_setup"):
            nodes = torch.tensor(
                _GAUSS_LEGENDRE_5_NODES,
                dtype=ray_origins.dtype,
                device=ray_origins.device,
            )
            weights = torch.tensor(
                _GAUSS_LEGENDRE_5_WEIGHTS,
                dtype=ray_origins.dtype,
                device=ray_origins.device,
            )
            empty_indices = torch.empty(0, dtype=torch.long, device=ray_origins.device)
            empty_lengths = torch.empty(0, dtype=ray_origins.dtype, device=ray_origins.device)
            empty_points = torch.empty((0, 3), dtype=ray_origins.dtype, device=ray_origins.device)

        if valid_mask.numel() == 0 or not valid_mask.any().item():
            return {
                "weights": weights,
                "flat_segment_indices": empty_indices,
                "valid_rows": empty_indices,
                "valid_half_lengths": empty_lengths,
                "sample_points_flat": empty_points,
            }

        with record_function("field.segment_context.compact_valid_segments"):
            half_lengths = 0.5 * (t_ends - t_starts)
            centers = 0.5 * (t_starts + t_ends)
            valid_rows, valid_cols = valid_mask.nonzero(as_tuple=True)
            valid_half_lengths = half_lengths[valid_rows, valid_cols]
            valid_centers = centers[valid_rows, valid_cols]
            valid_origins = ray_origins[valid_rows]
            valid_dirs = ray_dirs[valid_rows]

        with record_function("field.segment_context.sample_points"):
            t_samples = valid_centers[:, None] + valid_half_lengths[:, None] * nodes[None, :]
            sample_points = valid_origins[:, None, :] + t_samples[..., None] * valid_dirs[:, None, :]
            flat_segment_indices = valid_rows * t_starts.shape[1] + valid_cols
        return {
            "weights": weights,
            "flat_segment_indices": flat_segment_indices,
            "valid_rows": valid_rows,
            "valid_half_lengths": valid_half_lengths,
            "sample_points_flat": sample_points.reshape(-1, 3),
        }

    def _scatter_segment_values(
        self,
        valid_values: Tensor,
        flat_segment_indices: Tensor,
        output_shape: torch.Size,
    ) -> Tensor:
        flat_size = output_shape[0] * output_shape[1]
        if valid_values.ndim == 1:
            scattered = torch.zeros(flat_size, dtype=valid_values.dtype, device=valid_values.device)
            if flat_segment_indices.numel() > 0:
                scattered = scattered.scatter(0, flat_segment_indices, valid_values)
            return scattered.reshape(output_shape)

        trailing_shape = valid_values.shape[1:]
        scattered = torch.zeros((flat_size,) + trailing_shape, dtype=valid_values.dtype, device=valid_values.device)
        if flat_segment_indices.numel() > 0:
            index = flat_segment_indices.reshape((-1,) + (1,) * len(trailing_shape)).expand(valid_values.shape)
            scattered = scattered.scatter(0, index, valid_values)
        return scattered.reshape(tuple(output_shape) + trailing_shape)

    def _integrate_segment_sample_values(
        self,
        sample_values: Tensor,
        output_shape: torch.Size,
        integration_context: dict[str, Tensor],
    ) -> Tensor:
        flat_segment_indices = integration_context["flat_segment_indices"]
        if flat_segment_indices.numel() == 0:
            return self._scatter_segment_values(
                sample_values.new_zeros((0,) + sample_values.shape[2:]),
                flat_segment_indices,
                output_shape,
            )

        weights = integration_context["weights"]
        valid_half_lengths = integration_context["valid_half_lengths"]
        weight_shape = (1, weights.shape[0]) + (1,) * (sample_values.ndim - 2)
        integrated = torch.sum(sample_values * weights.view(weight_shape), dim=1)
        half_length_shape = (valid_half_lengths.shape[0],) + (1,) * (integrated.ndim - 1)
        valid_values = integrated * valid_half_lengths.view(half_length_shape)
        return self._scatter_segment_values(valid_values, flat_segment_indices, output_shape)

    def _integrate_segment_density(
        self,
        ray_origins: Tensor,
        ray_dirs: Tensor,
        t_starts: Tensor,
        t_ends: Tensor,
        valid_mask: Tensor,
        *,
        integration_context: dict[str, Tensor] | None = None,
    ) -> Tensor:
        with record_function("field.integrate_density.prepare_context"):
            context = integration_context
            if context is None:
                context = self._prepare_segment_integration_context(ray_origins, ray_dirs, t_starts, t_ends, valid_mask)

        if context["flat_segment_indices"].numel() == 0:
            return torch.zeros_like(t_starts)

        with record_function("field.integrate_density.evaluate_sigma"):
            num_nodes = context["weights"].shape[0]
            sigma_samples = self.evaluate(context["sample_points_flat"]).reshape(-1, num_nodes)
        with record_function("field.integrate_density.reduce_samples"):
            return self._integrate_segment_sample_values(sigma_samples, t_starts.shape, context)

    def integrate_on_ray_segments(
        self,
        ray_origins: Any,
        ray_dirs: Any,
        t_starts: Any,
        t_ends: Any,
        *,
        valid_mask: Any | None = None,
        integration_context: dict[str, Tensor] | None = None,
    ) -> Tensor:
        ray_origins_t, ray_dirs_t = self._prepare_rays(ray_origins, ray_dirs)
        num_rays = ray_origins_t.shape[0]
        t_starts_t = self._coerce_segment_matrix(t_starts, num_rays, "t_starts")
        t_ends_t = self._coerce_segment_matrix(t_ends, num_rays, "t_ends")
        if t_starts_t.shape != t_ends_t.shape:
            raise ValueError("t_starts and t_ends must have the same shape.")

        valid_mask_t = self._coerce_valid_mask(valid_mask, t_starts_t.shape, ray_origins_t.device)
        return self._integrate_segment_density(
            ray_origins_t,
            ray_dirs_t,
            t_starts_t,
            t_ends_t,
            valid_mask_t,
            integration_context=integration_context,
        )

    def integrate_normal_on_ray_segments(
        self,
        ray_origins: Any,
        ray_dirs: Any,
        t_starts: Any,
        t_ends: Any,
        *,
        valid_mask: Any | None = None,
        integration_context: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        ray_origins_t, ray_dirs_t = self._prepare_rays(ray_origins, ray_dirs)
        num_rays = ray_origins_t.shape[0]
        t_starts_t = self._coerce_segment_matrix(t_starts, num_rays, "t_starts")
        t_ends_t = self._coerce_segment_matrix(t_ends, num_rays, "t_ends")
        if t_starts_t.shape != t_ends_t.shape:
            raise ValueError("t_starts and t_ends must have the same shape.")

        valid_mask_t = self._coerce_valid_mask(valid_mask, t_starts_t.shape, ray_origins_t.device)
        segment_lengths = torch.where(valid_mask_t, t_ends_t - t_starts_t, torch.zeros_like(t_starts_t))
        context = integration_context
        if context is None:
            context = self._prepare_segment_integration_context(
                ray_origins_t,
                ray_dirs_t,
                t_starts_t,
                t_ends_t,
                valid_mask_t,
            )

        if context["flat_segment_indices"].numel() == 0:
            normal_integrals = torch.zeros(
                tuple(t_starts_t.shape) + (3,),
                dtype=ray_origins_t.dtype,
                device=ray_origins_t.device,
            )
        else:
            num_nodes = context["weights"].shape[0]
            gradients = self.evaluate_gradient(context["sample_points_flat"]).reshape(-1, num_nodes, 3)
            gradient_norm = torch.linalg.norm(gradients, dim=-1, keepdim=True)
            normal_samples = gradients / gradient_norm.clamp_min(torch.finfo(gradients.dtype).eps)
            normal_integrals = self._integrate_segment_sample_values(normal_samples, t_starts_t.shape, context)

        safe_lengths = segment_lengths.clamp_min(torch.finfo(segment_lengths.dtype).eps)
        normal_means = torch.where(
            valid_mask_t[..., None],
            normal_integrals / safe_lengths[..., None],
            torch.zeros_like(normal_integrals),
        )
        return {
            "normal_segment_integrals": normal_integrals,
            "normal_segment_means": normal_means,
        }

    def integrate_rays(
        self,
        ray_origins: Any,
        ray_dirs: Any,
        *,
        t_min: Any | None = 0.0,
        t_max: Any | None = None,
        return_integration_context: bool = False,
    ) -> dict[str, Any]:
        """对给定射线做分段密度积分，返回可用于体渲染合成的中间量。"""

        with record_function("field.integrate_rays.prepare_inputs"):
            ray_origins_t, ray_dirs_t = self._prepare_rays(ray_origins, ray_dirs)
            num_rays = ray_origins_t.shape[0]
            t_min_t = self._coerce_ray_bounds(t_min, num_rays, default=0.0)
            t_max_t = self._coerce_ray_bounds(t_max, num_rays, default=float("inf"))

        with record_function("field.integrate_rays.intersections"):
            t_entry, t_exit, hit_mask = self._ray_box_intersections(
                ray_origins_t,
                ray_dirs_t,
                t_min_t,
                t_max_t,
            )
        with record_function("field.integrate_rays.segment_boundaries"):
            t_starts, t_ends, valid_mask = self._stack_segment_boundaries(
                ray_origins_t,
                ray_dirs_t,
                t_entry,
                t_exit,
                hit_mask,
            )

        with record_function("field.integrate_rays.segment_geometry"):
            segment_lengths = torch.where(valid_mask, t_ends - t_starts, torch.zeros_like(t_starts))
            mid_t = 0.5 * (t_starts + t_ends)
            midpoints = ray_origins_t[:, None, :] + mid_t[..., None] * ray_dirs_t[:, None, :]
        with record_function("field.integrate_rays.integration_context"):
            integration_context = self._prepare_segment_integration_context(
                ray_origins_t,
                ray_dirs_t,
                t_starts,
                t_ends,
                valid_mask,
            )
        with record_function("field.integrate_rays.segment_density"):
            tau = self._integrate_segment_density(
                ray_origins_t,
                ray_dirs_t,
                t_starts,
                t_ends,
                valid_mask,
                integration_context=integration_context,
            )

        with record_function("field.integrate_rays.attenuation_alpha"):
            attenuation = torch.where(valid_mask, torch.exp(-tau), torch.ones_like(tau))
            alpha = torch.where(valid_mask, 1.0 - attenuation, torch.zeros_like(tau))

        with record_function("field.integrate_rays.transmittance_weights"):
            if valid_mask.shape[1] == 0:
                transmittance = torch.zeros_like(tau)
                remaining_transmittance = torch.ones(num_rays, dtype=tau.dtype, device=tau.device)
                weights = torch.zeros_like(tau)
            else:
                tau_prefix = torch.cumsum(tau, dim=1)
                exclusive_tau = torch.cat(
                    [torch.zeros((num_rays, 1), dtype=tau.dtype, device=tau.device), tau_prefix[:, :-1]],
                    dim=1,
                )
                transmittance = torch.exp(-exclusive_tau)
                weights = transmittance * alpha
                remaining_transmittance = torch.exp(-tau_prefix[:, -1])

        opacity = 1.0 - remaining_transmittance
        result: dict[str, Any] = {
            "ray_origins": ray_origins_t,
            "ray_dirs": ray_dirs_t,
            "t_entry": t_entry,
            "t_exit": t_exit,
            "hit_mask": hit_mask,
            "t_starts": t_starts,
            "t_ends": t_ends,
            "mid_t": mid_t,
            "midpoints": midpoints,
            "segment_lengths": segment_lengths,
            "tau": tau,
            "alpha": alpha,
            "transmittance": transmittance,
            "weights": weights,
            "remaining_transmittance": remaining_transmittance,
            "opacity": opacity,
            "valid_mask": valid_mask,
        }
        if return_integration_context:
            result["integration_context"] = integration_context
        return result

    def _coerce_segment_colors(self, segment_colors: Any, num_rays: int, max_segments: int) -> Tensor:
        colors = _as_float_tensor(
            segment_colors,
            dtype=self.control_grid.dtype,
            device=self.control_grid.device,
        )

        if colors.ndim == 0:
            return colors.reshape(1, 1).expand(num_rays, max_segments)
        if colors.ndim == 1:
            if colors.shape[0] == max_segments:
                return colors.reshape(1, max_segments).expand(num_rays, max_segments)
            return colors.reshape(1, 1, -1).expand(num_rays, max_segments, -1)
        if colors.ndim == 2:
            if colors.shape == (num_rays, max_segments):
                return colors
            if colors.shape[0] == num_rays:
                return colors[:, None, :].expand(num_rays, max_segments, colors.shape[1])
            if colors.shape[0] == max_segments:
                return colors[None, :, :].expand(num_rays, max_segments, colors.shape[1])
        if colors.ndim == 3 and colors.shape[:2] == (num_rays, max_segments):
            return colors

        raise ValueError(
            "segment_colors must be broadcastable to (num_rays, max_segments) or "
            "(num_rays, max_segments, channels)."
        )

    def _render_from_segment_colors(
        self,
        weights: Tensor,
        colors: Tensor,
        remaining_transmittance: Tensor,
        background: Any | None,
    ) -> Tensor:
        if colors.ndim == 2:
            rendered = torch.sum(weights * colors, dim=1)
            if background is not None:
                background_t = _as_float_tensor(
                    background,
                    dtype=rendered.dtype,
                    device=rendered.device,
                )
                if background_t.ndim == 0:
                    rendered = rendered + remaining_transmittance * background_t
                elif background_t.ndim == 1 and background_t.shape == (weights.shape[0],):
                    rendered = rendered + remaining_transmittance * background_t
                else:
                    raise ValueError("Scalar rendering only supports scalar background or shape (num_rays,).")
            return rendered

        rendered = torch.sum(weights[..., None] * colors, dim=1)
        if background is not None:
            background_t = _as_float_tensor(
                background,
                dtype=rendered.dtype,
                device=rendered.device,
            )
            if background_t.ndim == 1:
                background_t = background_t.reshape(1, -1).expand(weights.shape[0], -1)
            elif background_t.ndim != 2 or background_t.shape[0] != weights.shape[0]:
                raise ValueError(
                    "Vector rendering only supports background with shape (channels,) or (num_rays, channels)."
                )
            rendered = rendered + remaining_transmittance[:, None] * background_t
        return rendered

    def volume_render_rays(
        self,
        ray_origins: Any,
        ray_dirs: Any,
        segment_colors: Any | None = None,
        *,
        color_field: "BSplineRGBField" | None = None,
        t_min: Any | None = 0.0,
        t_max: Any | None = None,
        background: Any | None = None,
        return_normals: bool = False,
        normal_background: Any | None = None,
    ) -> dict[str, Tensor]:
        if (segment_colors is None) == (color_field is None):
            raise ValueError("Provide exactly one of segment_colors or color_field.")

        with record_function("field.volume_render.integrate_rays"):
            ray_samples = self.integrate_rays(
                ray_origins,
                ray_dirs,
                t_min=t_min,
                t_max=t_max,
                return_integration_context=color_field is not None,
            )
        weights = ray_samples["weights"]
        num_rays, max_segments = weights.shape
        normal_outputs: dict[str, Tensor] = {}
        if return_normals:
            with record_function("field.volume_render.normal_integrals"):
                integration_context = ray_samples.get("integration_context")
                normal_samples = self.integrate_normal_on_ray_segments(
                    ray_samples["ray_origins"],
                    ray_samples["ray_dirs"],
                    ray_samples["t_starts"],
                    ray_samples["t_ends"],
                    valid_mask=ray_samples["valid_mask"],
                    integration_context=integration_context,
                )
            with record_function("field.volume_render.normal_composite"):
                rendered_normals = self._render_from_segment_colors(
                    weights,
                    normal_samples["normal_segment_means"],
                    ray_samples["remaining_transmittance"],
                    torch.zeros(3, dtype=weights.dtype, device=weights.device)
                    if normal_background is None
                    else normal_background,
                )
            normal_outputs = {
                **normal_samples,
                "rendered_normals": rendered_normals,
                "normals": rendered_normals,
                "normal": rendered_normals,
            }
        if color_field is not None:
            with record_function("field.volume_render.color_integrals"):
                integration_context = ray_samples.pop("integration_context", None)
                color_samples = color_field.integrate_on_ray_segments(
                    ray_samples["ray_origins"],
                    ray_samples["ray_dirs"],
                    ray_samples["t_starts"],
                    ray_samples["t_ends"],
                    valid_mask=ray_samples["valid_mask"],
                    integration_context=integration_context,
                )
                colors = color_samples["rgb_segment_means"]
            with record_function("field.volume_render.composite"):
                rendered = self._render_from_segment_colors(
                    weights,
                    colors,
                    ray_samples["remaining_transmittance"],
                    background,
                )
            return {
                **ray_samples,
                **color_samples,
                **normal_outputs,
                "segment_colors": colors,
                "rendered": rendered,
            }

        with record_function("field.volume_render.coerce_segment_colors"):
            colors = self._coerce_segment_colors(segment_colors, num_rays, max_segments)
        with record_function("field.volume_render.composite"):
            rendered = self._render_from_segment_colors(
                weights,
                colors,
                ray_samples["remaining_transmittance"],
                background,
            )

        return {
            **ray_samples,
            **normal_outputs,
            "segment_colors": colors,
            "rendered": rendered,
        }

    def _prepare_points(self, points: Any) -> tuple[Tensor, tuple[int, ...], bool]:
        array = _as_float_tensor(
            points,
            dtype=self.control_grid.dtype,
            device=self.control_grid.device,
        )
        if tuple(array.shape) == (3,):
            # 单点输入统一改成 (1, 3)，便于后续走批量路径。
            return array.reshape(1, 3), (), True
        if array.ndim < 1 or array.shape[-1] != 3:
            raise ValueError("points must have shape (3,) or (..., 3).")
        # 先压平成二维批量形式，最后再恢复成原来的广播输出形状。
        output_shape = tuple(array.shape[:-1])
        return array.reshape(-1, 3), output_shape, False

    def _validate_points(self, points: Tensor) -> Tensor:
        # 允许边界上的微小浮点误差，并将其钳回到合法范围内；
        # 只有明显越界的点才继续报错。
        eps = torch.finfo(points.dtype).eps
        tolerance = torch.clamp_min(self._max_extent * (64.0 * eps), 1e-6)
        lower_limit = self._lower - tolerance
        upper_limit = self._upper + tolerance
        out_of_bounds = (points < lower_limit) | (points > upper_limit)
        if torch.any(out_of_bounds).item():
            raise ValueError(
                "All query points must lie inside the field bounds "
                f"{tuple(map(tuple, self.bounds.tolist()))}."
            )
        return torch.minimum(torch.maximum(points, self._lower), self._upper)


class BSplineRGBField(nn.Module):
    """RGB field composed from three scalar B-spline fields."""

    def __init__(
        self,
        grid_size: int,
        control_coefficients: Any,
        bounds: Bounds = _DEFAULT_BOUNDS,
        *,
        spline_degree: int | Sequence[int] = 3,
        trainable: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        coeff_r, coeff_g, coeff_b = _split_rgb_values(
            control_coefficients,
            name="control_coefficients",
        )
        degree_r, degree_g, degree_b = (
            _coerce_spline_degree(value)
            for value in _split_rgb_values(spline_degree, name="spline_degree", allow_scalar=True)
        )

        self.fields = nn.ModuleList(
            [
                BSplineField(
                    grid_size=grid_size,
                    control_coefficients=coeff_r,
                    bounds=bounds,
                    spline_degree=degree_r,
                    trainable=trainable,
                    dtype=dtype,
                    device=device,
                ),
                BSplineField(
                    grid_size=grid_size,
                    control_coefficients=coeff_g,
                    bounds=bounds,
                    spline_degree=degree_g,
                    trainable=trainable,
                    dtype=dtype,
                    device=device,
                ),
                BSplineField(
                    grid_size=grid_size,
                    control_coefficients=coeff_b,
                    bounds=bounds,
                    spline_degree=degree_b,
                    trainable=trainable,
                    dtype=dtype,
                    device=device,
                ),
            ]
        )
        self.grid_size = int(grid_size)
        self.spline_degree = (degree_r, degree_g, degree_b)

    @classmethod
    def zeros(
        cls,
        grid_size: int,
        bounds: Bounds = _DEFAULT_BOUNDS,
        fill_value: float | Sequence[float] = 0.0,
        *,
        spline_degree: int | Sequence[int] = 3,
        trainable: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> "BSplineRGBField":
        fill_r, fill_g, fill_b = _split_rgb_values(fill_value, name="fill_value", allow_scalar=True)
        coeffs = torch.stack(
            [
                torch.full(
                    (grid_size, grid_size, grid_size),
                    fill_value=float(fill_r),
                    dtype=dtype or torch.get_default_dtype(),
                    device=device,
                ),
                torch.full(
                    (grid_size, grid_size, grid_size),
                    fill_value=float(fill_g),
                    dtype=dtype or torch.get_default_dtype(),
                    device=device,
                ),
                torch.full(
                    (grid_size, grid_size, grid_size),
                    fill_value=float(fill_b),
                    dtype=dtype or torch.get_default_dtype(),
                    device=device,
                ),
            ],
            dim=0,
        )
        return cls(
            grid_size=grid_size,
            control_coefficients=coeffs,
            bounds=bounds,
            spline_degree=spline_degree,
            trainable=trainable,
            dtype=dtype,
            device=device,
        )

    @property
    def bounds(self) -> Tensor:
        return self.fields[0].bounds

    @property
    def control_grid(self) -> Tensor:
        return torch.stack([field.control_grid for field in self.fields], dim=0)

    def set_control_coefficients(self, control_coefficients: Any) -> None:
        coeff_r, coeff_g, coeff_b = _split_rgb_values(
            control_coefficients,
            name="control_coefficients",
        )
        for field, coeffs in zip(self.fields, (coeff_r, coeff_g, coeff_b)):
            field.set_control_coefficients(coeffs)

    def flat_control_coefficients(self) -> Tensor:
        return torch.stack([field.flat_control_coefficients() for field in self.fields], dim=0)

    def forward(self, x: Any, y: Any | None = None, z: Any | None = None) -> Tensor:
        if y is None and z is None:
            return self.evaluate(x)
        if y is None or z is None:
            raise TypeError("Pass either a single (..., 3) points tensor or x, y, z together.")
        return self.evaluate_xyz(x, y, z)

    def evaluate_xyz(self, x: Any, y: Any, z: Any) -> Tensor:
        return torch.stack([field.evaluate_xyz(x, y, z) for field in self.fields], dim=-1)

    def evaluate(self, points: Any) -> Tensor:
        return torch.stack([field.evaluate(points) for field in self.fields], dim=-1)

    def integrate_on_ray_segments(
        self,
        ray_origins: Any,
        ray_dirs: Any,
        t_starts: Any,
        t_ends: Any,
        *,
        valid_mask: Any | None = None,
        integration_context: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        base_field = self.fields[0]
        ray_origins_t, ray_dirs_t = base_field._prepare_rays(ray_origins, ray_dirs)
        num_rays = ray_origins_t.shape[0]
        t_starts_t = base_field._coerce_segment_matrix(t_starts, num_rays, "t_starts")
        t_ends_t = base_field._coerce_segment_matrix(t_ends, num_rays, "t_ends")
        if t_starts_t.shape != t_ends_t.shape:
            raise ValueError("t_starts and t_ends must have the same shape.")

        valid_mask_t = base_field._coerce_valid_mask(valid_mask, t_starts_t.shape, ray_origins_t.device)
        segment_lengths = torch.where(valid_mask_t, t_ends_t - t_starts_t, torch.zeros_like(t_starts_t))
        context = integration_context
        if context is None:
            context = base_field._prepare_segment_integration_context(
                ray_origins_t,
                ray_dirs_t,
                t_starts_t,
                t_ends_t,
                valid_mask_t,
            )

        if context["flat_segment_indices"].numel() == 0:
            rgb_integrals = torch.zeros(
                tuple(t_starts_t.shape) + (3,),
                dtype=ray_origins_t.dtype,
                device=ray_origins_t.device,
            )
        else:
            num_nodes = context["weights"].shape[0]
            rgb_samples = self.evaluate(context["sample_points_flat"]).reshape(-1, num_nodes, 3)
            rgb_integrals = base_field._integrate_segment_sample_values(rgb_samples, t_starts_t.shape, context)
        safe_lengths = segment_lengths.clamp_min(torch.finfo(segment_lengths.dtype).eps)
        rgb_means = torch.where(
            valid_mask_t[..., None],
            rgb_integrals / safe_lengths[..., None],
            torch.zeros_like(rgb_integrals),
        )
        return {
            "rgb_segment_integrals": rgb_integrals,
            "rgb_segment_means": rgb_means,
        }

    def render_rays(
        self,
        density_field: BSplineField,
        ray_origins: Any,
        ray_dirs: Any,
        *,
        t_min: Any | None = 0.0,
        t_max: Any | None = None,
        background: Any | None = None,
        return_normals: bool = False,
        normal_background: Any | None = None,
    ) -> dict[str, Tensor]:
        return density_field.volume_render_rays(
            ray_origins,
            ray_dirs,
            color_field=self,
            t_min=t_min,
            t_max=t_max,
            background=background,
            return_normals=return_normals,
            normal_background=normal_background,
        )


class BSplineSHRGBField(nn.Module):
    """Direction-dependent RGB field using spherical harmonics at each B-spline control point."""

    def __init__(
        self,
        grid_size: int,
        control_coefficients: Any,
        bounds: Bounds = _DEFAULT_BOUNDS,
        *,
        spline_degree: int | Sequence[int] = 3,
        sh_degree: int = 2,
        trainable: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        degree_r, degree_g, degree_b = tuple(
            _coerce_spline_degree(value)
            for value in _split_rgb_values(spline_degree, name="spline_degree", allow_scalar=True)
        )
        if not (degree_r == degree_g == degree_b):
            raise ValueError("BSplineSHRGBField uses one shared spatial field, so RGB spline degrees must match.")

        self.sh_degree = _coerce_sh_degree(sh_degree)
        self.sh_basis_dim = (self.sh_degree + 1) ** 2
        self.feature_dim = 3 * self.sh_basis_dim
        control_grid = self._coerce_feature_control_grid(
            control_coefficients,
            grid_size=grid_size,
            dtype=dtype or torch.get_default_dtype(),
            device=device,
        )

        self.field = BSplineField.zeros(
            grid_size=grid_size,
            bounds=bounds,
            fill_value=0.0,
            spline_degree=degree_r,
            trainable=False,
            dtype=control_grid.dtype,
            device=control_grid.device,
        )
        if trainable:
            self.control_grid = nn.Parameter(control_grid)
        else:
            self.register_buffer("control_grid", control_grid)

        self.grid_size = int(grid_size)
        self.spline_degree = (degree_r, degree_g, degree_b)

    @classmethod
    def zeros(
        cls,
        grid_size: int,
        bounds: Bounds = _DEFAULT_BOUNDS,
        fill_value: float | Sequence[float] = 0.0,
        *,
        spline_degree: int | Sequence[int] = 3,
        sh_degree: int = 2,
        trainable: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> "BSplineSHRGBField":
        basis_dim = (_coerce_sh_degree(sh_degree) + 1) ** 2
        fill_r, fill_g, fill_b = _split_rgb_values(fill_value, name="fill_value", allow_scalar=True)
        coeffs = torch.zeros(
            (3, basis_dim, grid_size, grid_size, grid_size),
            dtype=dtype or torch.get_default_dtype(),
            device=device,
        )
        coeffs[0, 0].fill_(_inverse_sigmoid(float(fill_r)) / _SH_C0)
        coeffs[1, 0].fill_(_inverse_sigmoid(float(fill_g)) / _SH_C0)
        coeffs[2, 0].fill_(_inverse_sigmoid(float(fill_b)) / _SH_C0)
        return cls(
            grid_size=grid_size,
            control_coefficients=coeffs,
            bounds=bounds,
            spline_degree=spline_degree,
            sh_degree=sh_degree,
            trainable=trainable,
            dtype=dtype,
            device=device,
        )

    @property
    def bounds(self) -> Tensor:
        return self.field.bounds

    def _coerce_feature_control_grid(
        self,
        control_coefficients: Any,
        *,
        grid_size: int,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> Tensor:
        if torch.is_tensor(control_coefficients) or not (
            isinstance(control_coefficients, Sequence) and not isinstance(control_coefficients, (str, bytes))
        ):
            control_grid = _as_float_tensor(
                control_coefficients,
                dtype=dtype,
                device=device,
            )
        else:
            coeff_r, coeff_g, coeff_b = _split_rgb_values(
                control_coefficients,
                name="control_coefficients",
            )
            control_grid = torch.stack(
                [
                    _as_float_tensor(coeff_r, dtype=dtype, device=device),
                    _as_float_tensor(coeff_g, dtype=dtype, device=device),
                    _as_float_tensor(coeff_b, dtype=dtype, device=device),
                ],
                dim=0,
            )

        if tuple(control_grid.shape) == (self.feature_dim, grid_size, grid_size, grid_size):
            return control_grid.reshape(3, self.sh_basis_dim, grid_size, grid_size, grid_size)
        if tuple(control_grid.shape) == (3, self.sh_basis_dim, grid_size, grid_size, grid_size):
            return control_grid
        raise ValueError(
            "control_coefficients must have shape "
            f"({self.feature_dim}, {grid_size}, {grid_size}, {grid_size}) or "
            f"(3, {self.sh_basis_dim}, {grid_size}, {grid_size}, {grid_size})."
        )

    def flat_control_coefficients(self) -> Tensor:
        return self.control_grid.reshape(3, self.sh_basis_dim, -1)

    def set_control_coefficients(self, control_coefficients: Any) -> None:
        updated = self._coerce_feature_control_grid(
            control_coefficients,
            grid_size=self.grid_size,
            dtype=self.control_grid.dtype,
            device=self.control_grid.device,
        )
        with torch.no_grad():
            self.control_grid.copy_(updated)

    def evaluate_sh_coefficients(self, points: Any) -> Tensor:
        with record_function("field.sh.evaluate_coefficients"):
            return self.field._evaluate_control_tensor(self.control_grid, points)

    def _direction_basis(self, ray_directions: Any, target_shape: tuple[int, ...]) -> Tensor:
        with record_function("field.sh.direction_basis"):
            directions = _as_float_tensor(
                ray_directions,
                dtype=self.control_grid.dtype,
                device=self.control_grid.device,
            )
            if tuple(directions.shape) == (3,):
                directions = directions.reshape((1,) * len(target_shape) + (3,))
            if directions.ndim < 1 or directions.shape[-1] != 3:
                raise ValueError("ray_directions must have shape (3,) or (..., 3).")
            try:
                directions = torch.broadcast_to(directions, target_shape + (3,))
            except RuntimeError as exc:
                raise ValueError(
                    "ray_directions must broadcast to the same leading shape as the queried points."
                ) from exc
            return spherical_harmonics_basis(directions, self.sh_degree)

    def forward(self, points: Any, ray_directions: Any) -> Tensor:
        return self.evaluate(points, ray_directions)

    def evaluate(self, points: Any, ray_directions: Any) -> Tensor:
        with record_function("field.sh.evaluate.coefficients"):
            sh_coefficients = self.evaluate_sh_coefficients(points)
            output_shape = tuple(sh_coefficients.shape[:-2])
        with record_function("field.sh.evaluate.direction_basis"):
            direction_basis = self._direction_basis(ray_directions, output_shape)
        with record_function("field.sh.evaluate.rgb_logits"):
            rgb_logits = torch.sum(sh_coefficients * direction_basis.unsqueeze(-2), dim=-1)
        with record_function("field.sh.evaluate.rgb_sigmoid"):
            return torch.sigmoid(rgb_logits)

    def integrate_on_ray_segments(
        self,
        ray_origins: Any,
        ray_dirs: Any,
        t_starts: Any,
        t_ends: Any,
        *,
        valid_mask: Any | None = None,
        integration_context: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        base_field = self.field
        with record_function("field.sh.integrate.prepare_inputs"):
            ray_origins_t, ray_dirs_t = base_field._prepare_rays(ray_origins, ray_dirs)
            num_rays = ray_origins_t.shape[0]
            t_starts_t = base_field._coerce_segment_matrix(t_starts, num_rays, "t_starts")
            t_ends_t = base_field._coerce_segment_matrix(t_ends, num_rays, "t_ends")
        if t_starts_t.shape != t_ends_t.shape:
            raise ValueError("t_starts and t_ends must have the same shape.")

        with record_function("field.sh.integrate.segment_lengths"):
            valid_mask_t = base_field._coerce_valid_mask(valid_mask, t_starts_t.shape, ray_origins_t.device)
            segment_lengths = torch.where(valid_mask_t, t_ends_t - t_starts_t, torch.zeros_like(t_starts_t))
        with record_function("field.sh.integrate.context"):
            context = integration_context
            if context is None:
                context = base_field._prepare_segment_integration_context(
                    ray_origins_t,
                    ray_dirs_t,
                    t_starts_t,
                    t_ends_t,
                    valid_mask_t,
                )

        if context["flat_segment_indices"].numel() == 0:
            rgb_integrals = torch.zeros(
                tuple(t_starts_t.shape) + (3,),
                dtype=ray_origins_t.dtype,
                device=ray_origins_t.device,
            )
        else:
            with record_function("field.sh.integrate.coefficients"):
                num_nodes = context["weights"].shape[0]
                sh_coeff_samples = self.evaluate_sh_coefficients(context["sample_points_flat"])
            with record_function("field.sh.integrate.direction_basis"):
                valid_ray_dirs = ray_dirs_t[context["valid_rows"]]
                direction_basis = spherical_harmonics_basis(valid_ray_dirs, self.sh_degree)
                direction_basis = direction_basis[:, None, :].expand(-1, num_nodes, -1).reshape(-1, self.sh_basis_dim)
            with record_function("field.sh.integrate.rgb_samples"):
                rgb_logits = torch.sum(sh_coeff_samples * direction_basis[:, None, :], dim=-1)
                rgb_samples = torch.sigmoid(rgb_logits).reshape(-1, num_nodes, 3)
            with record_function("field.sh.integrate.reduce_samples"):
                rgb_integrals = base_field._integrate_segment_sample_values(rgb_samples, t_starts_t.shape, context)

        with record_function("field.sh.integrate.segment_means"):
            safe_lengths = segment_lengths.clamp_min(torch.finfo(segment_lengths.dtype).eps)
            rgb_means = torch.where(
                valid_mask_t[..., None],
                rgb_integrals / safe_lengths[..., None],
                torch.zeros_like(rgb_integrals),
            )
        return {
            "rgb_segment_integrals": rgb_integrals,
            "rgb_segment_means": rgb_means,
        }

    def render_rays(
        self,
        density_field: BSplineField,
        ray_origins: Any,
        ray_dirs: Any,
        *,
        t_min: Any | None = 0.0,
        t_max: Any | None = None,
        background: Any | None = None,
        return_normals: bool = False,
        normal_background: Any | None = None,
    ) -> dict[str, Tensor]:
        return density_field.volume_render_rays(
            ray_origins,
            ray_dirs,
            color_field=self,
            t_min=t_min,
            t_max=t_max,
            background=background,
            return_normals=return_normals,
            normal_background=normal_background,
        )


class BSplineSDField(BSplineField):
    """B-spline signed distance field with NeuS-style unbiased segment opacity.

    The control grid stores raw signed-distance values (no activation).  For a
    segment ``[t_i, t_{i+1}]`` along a ray, opacity is derived from the SDF
    values at the two endpoints using the NeuS CDF formulation:

        Φ_s(f) = sigmoid(s * f)
        α_i    = max( (Φ_s(f(t_i)) - Φ_s(f(t_{i+1}))) / Φ_s(f(t_i)), 0 )

    where *s* is a learnable (or fixed) inverse standard deviation.
    """

    def __init__(
        self,
        grid_size: int,
        control_coefficients: Any,
        bounds: Bounds = _DEFAULT_BOUNDS,
        *,
        spline_degree: int = 3,
        trainable: bool = True,
        inv_std: float = 1.0,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(
            grid_size=grid_size,
            control_coefficients=control_coefficients,
            bounds=bounds,
            spline_degree=spline_degree,
            trainable=trainable,
            dtype=dtype,
            device=device,
        )
        inv_std_t = _as_float_tensor(inv_std, dtype=self.control_grid.dtype, device=self.control_grid.device)
        self.register_buffer("_inv_std", inv_std_t.reshape(()).clone())

    def set_inv_std(self, inv_std: Any) -> None:
        """Replace the NeuS inverse standard deviation buffer."""
        self._inv_std = _as_float_tensor(
            inv_std,
            dtype=self.control_grid.dtype,
            device=self.control_grid.device,
        ).reshape(()).clone()

    def inv_std(self) -> Tensor:
        """Return the current inverse standard deviation ``s``."""
        return self._inv_std

    def evaluate_sdf(self, points: Any) -> Tensor:
        """Evaluate the raw SDF at the given points."""
        return self.evaluate(points)

    def evaluate_sdf_gradient(self, points: Any) -> Tensor:
        """Evaluate the spatial gradient of the SDF at the given points."""
        return self.evaluate_gradient(points)

    def _neus_cdf(self, sdf: Tensor) -> Tensor:
        """NeuS CDF ``Φ_s(f) = sigmoid(s * f)``."""
        return torch.sigmoid(self._inv_std * sdf)

    def _neus_alpha(self, f_start: Tensor, f_end: Tensor) -> Tensor:
        """Segment opacity from endpoint SDF values (clamped)."""
        cdf_start = self._neus_cdf(f_start)
        cdf_end = self._neus_cdf(f_end)
        denom = cdf_start.clamp_min(torch.finfo(cdf_start.dtype).eps)
        alpha = (cdf_start - cdf_end) / denom
        return alpha.clamp_min(0.0)

    def integrate_sdf_rays(
        self,
        ray_origins: Any,
        ray_dirs: Any,
        *,
        t_min: Any | None = 0.0,
        t_max: Any | None = None,
    ) -> dict[str, Tensor]:
        """March rays through the SDF grid and return NeuS segment weights."""
        ray_origins_t, ray_dirs_t = self._prepare_rays(ray_origins, ray_dirs)
        num_rays = ray_origins_t.shape[0]
        t_min_t = self._coerce_ray_bounds(t_min, num_rays, default=0.0)
        t_max_t = self._coerce_ray_bounds(t_max, num_rays, default=float("inf"))

        t_entry, t_exit, hit_mask = self._ray_box_intersections(
            ray_origins_t,
            ray_dirs_t,
            t_min_t,
            t_max_t,
        )
        t_starts, t_ends, valid_mask = self._stack_segment_boundaries(
            ray_origins_t,
            ray_dirs_t,
            t_entry,
            t_exit,
            hit_mask,
        )

        max_segments = t_starts.shape[1]
        if max_segments == 0:
            alpha = torch.zeros_like(t_starts)
            transmittance = torch.zeros_like(t_starts)
            weights = torch.zeros_like(t_starts)
            remaining_transmittance = torch.ones(num_rays, dtype=t_starts.dtype, device=t_starts.device)
        else:
            start_points = ray_origins_t[:, None, :] + t_starts[..., None] * ray_dirs_t[:, None, :]
            end_points = ray_origins_t[:, None, :] + t_ends[..., None] * ray_dirs_t[:, None, :]

            valid_rows, valid_cols = valid_mask.nonzero(as_tuple=True)
            if valid_rows.numel() > 0:
                valid_start_points = start_points[valid_rows, valid_cols, :]
                valid_end_points = end_points[valid_rows, valid_cols, :]
                valid_f_start = self.evaluate_sdf(valid_start_points)
                valid_f_end = self.evaluate_sdf(valid_end_points)
                flat_size = num_rays * max_segments
                flat_indices = valid_rows * max_segments + valid_cols
                f_start = torch.zeros(flat_size, dtype=t_starts.dtype, device=t_starts.device).scatter(
                    0, flat_indices, valid_f_start
                ).reshape(t_starts.shape)
                f_end = torch.zeros(flat_size, dtype=t_ends.dtype, device=t_ends.device).scatter(
                    0, flat_indices, valid_f_end
                ).reshape(t_ends.shape)
            else:
                f_start = torch.zeros_like(t_starts)
                f_end = torch.zeros_like(t_ends)

            alpha = self._neus_alpha(f_start, f_end)
            alpha = torch.where(valid_mask, alpha, torch.zeros_like(alpha))

            log_1_minus_alpha = torch.log((1.0 - alpha).clamp_min(torch.finfo(alpha.dtype).eps))
            log_transmittance_prefix = torch.cumsum(log_1_minus_alpha, dim=1)
            transmittance = torch.exp(
                torch.cat(
                    [
                        torch.zeros((num_rays, 1), dtype=log_transmittance_prefix.dtype, device=log_transmittance_prefix.device),
                        log_transmittance_prefix[:, :-1],
                    ],
                    dim=1,
                )
            )
            weights = transmittance * alpha
            remaining_transmittance = torch.exp(log_transmittance_prefix[:, -1])

        opacity = 1.0 - remaining_transmittance
        mid_t = 0.5 * (t_starts + t_ends)
        return {
            "ray_origins": ray_origins_t,
            "ray_dirs": ray_dirs_t,
            "t_entry": t_entry,
            "t_exit": t_exit,
            "hit_mask": hit_mask,
            "t_starts": t_starts,
            "t_ends": t_ends,
            "valid_mask": valid_mask,
            "segment_lengths": torch.where(valid_mask, t_ends - t_starts, torch.zeros_like(t_starts)),
            "mid_t": mid_t,
            "midpoints": ray_origins_t[:, None, :] + mid_t[..., None] * ray_dirs_t[:, None, :],
            "alpha": alpha,
            "transmittance": transmittance,
            "weights": weights,
            "remaining_transmittance": remaining_transmittance,
            "opacity": opacity,
            "depth": (weights * mid_t).sum(dim=1),
        }

    def volume_render_sdf_rays(
        self,
        ray_origins: Any,
        ray_dirs: Any,
        *,
        color_field: "BSplineSHRGBField" | None = None,
        t_min: Any | None = 0.0,
        t_max: Any | None = None,
        background: Any | None = None,
        return_normals: bool = False,
        normal_background: Any | None = None,
    ) -> dict[str, Tensor]:
        """NeuS-style volume rendering using a separate color field."""
        if color_field is None:
            raise ValueError("color_field is required for SDF volume rendering.")

        ray_samples = self.integrate_sdf_rays(
            ray_origins,
            ray_dirs,
            t_min=t_min,
            t_max=t_max,
        )
        weights = ray_samples["weights"]
        num_rays = weights.shape[0]

        normal_outputs: dict[str, Tensor] = {}
        if return_normals:
            integration_context = self._prepare_segment_integration_context(
                ray_samples["ray_origins"],
                ray_samples["ray_dirs"],
                ray_samples["t_starts"],
                ray_samples["t_ends"],
                ray_samples["valid_mask"],
            )
            normal_samples = self.integrate_normal_on_ray_segments(
                ray_samples["ray_origins"],
                ray_samples["ray_dirs"],
                ray_samples["t_starts"],
                ray_samples["t_ends"],
                valid_mask=ray_samples["valid_mask"],
                integration_context=integration_context,
            )
            rendered_normals = self._render_from_segment_colors(
                weights,
                normal_samples["normal_segment_means"],
                ray_samples["remaining_transmittance"],
                (
                    torch.zeros(3, dtype=weights.dtype, device=weights.device)
                    if normal_background is None
                    else normal_background
                ),
            )
            normal_outputs = {
                **normal_samples,
                "rendered_normals": rendered_normals,
                "normals": rendered_normals,
                "normal": rendered_normals,
            }

        color_samples = color_field.integrate_on_ray_segments(
            ray_samples["ray_origins"],
            ray_samples["ray_dirs"],
            ray_samples["t_starts"],
            ray_samples["t_ends"],
            valid_mask=ray_samples["valid_mask"],
            integration_context=None,
        )
        colors = color_samples["rgb_segment_means"]
        rendered = self._render_from_segment_colors(
            weights,
            colors,
            ray_samples["remaining_transmittance"],
            background,
        )

        return {
            **ray_samples,
            **color_samples,
            **normal_outputs,
            "segment_colors": colors,
            "rendered": rendered,
            "rgb": rendered,
        }


BSplineDensityField = BSplineField
RadianceField = BSplineSHRGBField
