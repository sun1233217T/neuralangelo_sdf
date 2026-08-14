'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from projects.neuralangelo.data import Dataset as NeuralangeloDataset


_PLY_DTYPE_MAP = {
    "char": np.int8, "uchar": np.uint8, "short": np.int16, "ushort": np.uint16,
    "int": np.int32, "uint": np.uint32, "float": np.float32, "double": np.float64,
    "int8": np.int8, "uint8": np.uint8, "int16": np.int16, "uint16": np.uint16,
    "int32": np.int32, "uint32": np.uint32, "float32": np.float32, "float64": np.float64,
}


def _load_ply_points_xyz(ply_path):
    """Minimal PLY XYZ loader (matches bijectiveImplicitShell-nerf)."""
    with open(ply_path, "rb") as ply_file:
        format_name = None
        vertex_count = None
        vertex_properties = []
        in_vertex_element = False

        while True:
            line = ply_file.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in PLY header: {ply_path}")
            decoded = line.decode("ascii", errors="strict").strip()
            if not decoded:
                continue
            tokens = decoded.split()
            keyword = tokens[0]

            if keyword == "format":
                format_name = tokens[1]
            elif keyword == "element":
                in_vertex_element = tokens[1] == "vertex"
                if in_vertex_element:
                    vertex_count = int(tokens[2])
                    vertex_properties = []
            elif keyword == "property" and in_vertex_element:
                if tokens[1] == "list":
                    raise ValueError("List-valued vertex properties are not supported.")
                vertex_properties.append((tokens[2], tokens[1]))
            elif keyword == "end_header":
                break

        property_names = [name for name, _ in vertex_properties]
        for axis in ("x", "y", "z"):
            if axis not in property_names:
                raise ValueError(f"PLY file missing required '{axis}' property: {ply_path}")

        if format_name == "ascii":
            usecols = [property_names.index("x"), property_names.index("y"), property_names.index("z")]
            points = np.loadtxt(ply_file, dtype=np.float32, usecols=usecols, max_rows=vertex_count)
            return np.atleast_2d(points).astype(np.float32, copy=False)

        byteorder = "<" if format_name == "binary_little_endian" else ">"
        dtype_fields = []
        for property_name, dtype_name in vertex_properties:
            base_dtype = _PLY_DTYPE_MAP[dtype_name]
            dtype_fields.append((property_name, np.dtype(base_dtype).newbyteorder(byteorder)))
        vertex_dtype = np.dtype(dtype_fields)
        vertex_data = np.fromfile(ply_file, dtype=vertex_dtype, count=vertex_count)
        return np.stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1).astype(np.float32, copy=False)


def infer_scene_bounds_from_points(
    scene_root,
    *,
    metadata_bounds=None,
    ply_filename="points.ply",
    quantile_low=0.0,
    quantile_high=1.0,
    padding_ratio=0.05,
    force_cube=False,
):
    """Compute a quantile-based axis-aligned bounding box from a PLY point cloud.

    This replicates the logic in bijectiveImplicitShell-nerf/dataset.py.
    """
    if not 0.0 <= float(quantile_low) < float(quantile_high) <= 1.0:
        raise ValueError("quantile_low and quantile_high must satisfy 0 <= low < high <= 1.")
    if padding_ratio < 0.0:
        raise ValueError("padding_ratio must be non-negative.")

    ply_path = Path(scene_root) / ply_filename
    if not ply_path.exists():
        raise FileNotFoundError(f"Could not find point cloud PLY file: {ply_path}")

    points = _load_ply_points_xyz(ply_path)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"Point cloud PLY did not contain any XYZ vertices: {ply_path}")

    if metadata_bounds is not None:
        metadata_array = np.asarray(metadata_bounds, dtype=np.float32)
        metadata_lower = metadata_array[:, 0]
        metadata_upper = metadata_array[:, 1]
        inside_mask = np.all((points >= metadata_lower) & (points <= metadata_upper), axis=1)
        if bool(np.any(inside_mask)):
            points = points[inside_mask]

    if quantile_low > 0.0 or quantile_high < 1.0:
        lower = np.quantile(points, quantile_low, axis=0).astype(np.float32, copy=False)
        upper = np.quantile(points, quantile_high, axis=0).astype(np.float32, copy=False)
    else:
        lower = points.min(axis=0)
        upper = points.max(axis=0)

    extent = np.maximum(upper - lower, 1.0e-6)
    padding = extent * float(padding_ratio)
    lower = lower - padding
    upper = upper + padding

    if force_cube:
        center = 0.5 * (lower + upper)
        half_extent = 0.5 * float(np.max(upper - lower))
        lower = center - half_extent
        upper = center + half_extent

    return tuple((float(lower[axis]), float(upper[axis])) for axis in range(3))


def compute_readjust_and_bounds(
    scene_root,
    meta,
    target_half_bound=2.0,
    ply_filename="points.ply",
    quantile_low=0.03,
    quantile_high=0.99,
    padding_ratio=0.15,
    force_cube=True,
):
    """Compute readjust center/scale and normalized B-spline bounds from points.ply.

    The returned bounds are a symmetric cube ``[[-B, B], ...]`` in the normalized
    coordinate frame used by Neuralangelo.  ``readjust`` should be placed under
    ``data.readjust``; ``bounds`` should be placed under ``model.object.bspline.bounds``.
    """
    metadata_bounds = None
    if "aabb_range" in meta:
        metadata_bounds = tuple(tuple(axis) for axis in meta["aabb_range"])

    world_bounds = infer_scene_bounds_from_points(
        scene_root,
        metadata_bounds=metadata_bounds,
        ply_filename=ply_filename,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
        padding_ratio=padding_ratio,
        force_cube=force_cube,
    )

    world_bounds_arr = np.array(world_bounds, dtype=np.float32)
    lower = world_bounds_arr[:, 0]
    upper = world_bounds_arr[:, 1]
    world_center = 0.5 * (lower + upper)
    max_extent = float(np.max(upper - lower))

    meta_center = np.array(meta.get("sphere_center", [0.0, 0.0, 0.0]), dtype=np.float32)
    meta_radius = float(meta.get("sphere_radius", 1.0))
    if meta_radius <= 0.0:
        meta_radius = 1.0

    readjust = {
        "center": (world_center - meta_center).tolist(),
        "scale": max_extent / (2.0 * target_half_bound * meta_radius),
    }
    bounds = [[-target_half_bound, target_half_bound]] * 3

    return readjust, bounds


class Dataset(NeuralangeloDataset):
    """Dataset class for Bspline-Neus.

    Reuses the Neuralangelo dataset loader and optionally auto-computes a tight
    scene bounding box from ``points.ply`` to reduce the effective world extent.
    """

    def __init__(self, cfg, is_inference=False):
        cfg_data = cfg.data
        # Optionally compute readjust / B-spline bounds from points.ply before
        # the parent class normalizes cameras.
        auto_bounds = getattr(cfg_data, "auto_bounds", None)
        if auto_bounds is not None and bool(auto_bounds.get("enabled", False)):
            root = cfg_data.root
            meta_fname = f"{root}/transforms.json"
            with open(meta_fname) as file:
                meta = json.load(file)

            readjust, bounds = compute_readjust_and_bounds(
                root,
                meta,
                target_half_bound=float(auto_bounds.get("target_half_bound", 2.0)),
                ply_filename=str(auto_bounds.get("ply_filename", "points.ply")),
                quantile_low=float(auto_bounds.get("quantile_low", 0.03)),
                quantile_high=float(auto_bounds.get("quantile_high", 0.99)),
                padding_ratio=float(auto_bounds.get("padding_ratio", 0.15)),
                force_cube=bool(auto_bounds.get("force_cube", True)),
            )

            # Merge with any user-provided readjust values.
            # The parent dataset expects attribute-style access
            # (getattr(readjust, "center") / getattr(readjust, "scale")), so we
            # must keep this as a namespace-like object rather than a plain dict.
            existing_readjust = getattr(cfg_data, "readjust", None)
            if isinstance(existing_readjust, dict):
                existing_readjust = SimpleNamespace(**existing_readjust)
            elif existing_readjust is None:
                existing_readjust = SimpleNamespace(center=[0.0, 0.0, 0.0], scale=1.0)
            else:
                center = list(getattr(existing_readjust, "center", [0.0, 0.0, 0.0]))
                scale = float(getattr(existing_readjust, "scale", 1.0))
                existing_readjust = SimpleNamespace(center=center, scale=scale)

            existing_readjust.center = readjust["center"]
            existing_readjust.scale = readjust["scale"]
            cfg_data.readjust = existing_readjust

            # Update the model's B-spline bounds in the shared config object.
            cfg.model.object.bspline.bounds = bounds

            if is_inference or getattr(cfg, "local_rank", 0) == 0:
                print(f"[Bspline-Neus] auto_bounds: world_bounds -> {bounds}")
                print(f"[Bspline-Neus] auto_bounds: readjust -> {existing_readjust}")

        super().__init__(cfg, is_inference=is_inference)
