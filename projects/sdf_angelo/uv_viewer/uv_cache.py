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

from dataclasses import dataclass
import io
import os
import numpy as np
import torch
import torch.nn.functional as torch_F

from projects.sdf_angelo.utils.mesh import _get_nvdiffrast, _rasterize_uv


@dataclass
class UVCache:
    tex_size: int
    coords: torch.Tensor  # [N,2] int64, (y,x)
    points: torch.Tensor  # [N,3] float32, normalized coords
    normals: torch.Tensor  # [N,3] float32
    feats: torch.Tensor  # [N,C] float32
    device: torch.device


def build_uv_cache(mesh, neural_sdf, sphere_center, sphere_radius, texture_size=1024,
                   raster_mode="gpu", batch_size=65536, device=None,
                   project_to_surface=False, project_iters=1, project_step=1.0,
                   project_max_step=0.0):
    if not hasattr(mesh, "visual") or mesh.visual is None or getattr(mesh.visual, "uv", None) is None:
        raise ValueError("Mesh does not contain UV coordinates.")
    if device is None:
        device = next(neural_sdf.parameters()).device
    vertices_world = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uvs = np.asarray(mesh.visual.uv, dtype=np.float32)
    if uvs.shape[0] != vertices_world.shape[0]:
        raise ValueError("UVs must be per-vertex for rasterization.")
    vertices = (vertices_world - sphere_center[None, :]) / float(sphere_radius)

    if raster_mode == "gpu":
        coords, points = _rasterize_uv_gpu(vertices, faces, uvs, texture_size, device)
    elif raster_mode == "cpu":
        coords, points = _rasterize_uv_cpu(vertices, faces, uvs, texture_size, device)
    else:
        raise ValueError(f"Unknown raster_mode: {raster_mode}")
    if points.numel() == 0:
        raise ValueError("No valid texels found after UV rasterization.")

    if project_to_surface:
        points = _project_points_to_surface(
            points,
            neural_sdf,
            batch_size=batch_size,
            iters=project_iters,
            step=project_step,
            max_step=project_max_step,
        )

    feats, normals = _compute_feats_normals(points, neural_sdf, batch_size=batch_size)
    return UVCache(
        tex_size=int(texture_size),
        coords=coords,
        points=points,
        normals=normals,
        feats=feats,
        device=device,
    )


def save_uv_bundle(path, uv_cache, neural_rgb, appear_embed, sphere_center, sphere_radius,
                   mesh_obj=None, mesh_bounds=None, init_camera=None, print_sizes=True):
    if uv_cache is None:
        raise ValueError("uv_cache is required for bundle saving.")
    bundle = {
        "version": 1,
        "uv_cache": {
            "tex_size": int(uv_cache.tex_size),
            "coords": uv_cache.coords.detach().cpu(),
            "points": uv_cache.points.detach().cpu(),
            "normals": uv_cache.normals.detach().cpu(),
            "feats": uv_cache.feats.detach().cpu(),
        },
        "sphere_center": np.asarray(sphere_center, dtype=np.float32),
        "sphere_radius": float(sphere_radius),
        "mesh_obj": mesh_obj,
        "mesh_bounds": np.asarray(mesh_bounds, dtype=np.float32) if mesh_bounds is not None else None,
        "init_camera": np.asarray(init_camera, dtype=np.float32) if init_camera is not None else None,
        "neural_rgb": _clone_module_to_cpu(neural_rgb),
        "appear_embed": _clone_module_to_cpu(appear_embed) if appear_embed is not None else None,
    }
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if print_sizes:
        _print_bundle_sizes(
            uv_cache=uv_cache,
            neural_rgb=neural_rgb,
            appear_embed=appear_embed,
            sphere_center=sphere_center,
            sphere_radius=sphere_radius,
            mesh_obj=mesh_obj,
            mesh_bounds=mesh_bounds,
            init_camera=init_camera,
        )
    torch.save(bundle, path)


def load_uv_bundle(path, device=None):
    bundle = _torch_load_compat(path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uv = bundle["uv_cache"]
    uv_cache = UVCache(
        tex_size=int(uv["tex_size"]),
        coords=uv["coords"].to(device=device),
        points=uv["points"].to(device=device),
        normals=uv["normals"].to(device=device),
        feats=uv["feats"].to(device=device),
        device=device,
    )
    neural_rgb = bundle["neural_rgb"]
    if neural_rgb is None:
        raise ValueError("Bundle missing neural_rgb.")
    neural_rgb = neural_rgb.to(device=device)
    neural_rgb.eval()
    appear_embed = bundle.get("appear_embed")
    if appear_embed is not None:
        appear_embed = appear_embed.to(device=device)
        appear_embed.eval()
    meta = {
        "version": bundle.get("version", 0),
        "sphere_center": bundle.get("sphere_center"),
        "sphere_radius": bundle.get("sphere_radius"),
        "mesh_obj": bundle.get("mesh_obj"),
        "mesh_bounds": bundle.get("mesh_bounds"),
        "init_camera": bundle.get("init_camera"),
    }
    return uv_cache, neural_rgb, appear_embed, meta


def _clone_module_to_cpu(module):
    if module is None:
        return None
    buffer = io.BytesIO()
    torch.save(module, buffer)
    buffer.seek(0)
    clone = _torch_load_compat(buffer)
    clone.eval()
    return clone


def _torch_load_compat(source):
    try:
        return torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(source, map_location="cpu")


def _tensor_bytes(tensor):
    if tensor is None:
        return 0
    return int(tensor.numel()) * int(tensor.element_size())


def _module_bytes(module):
    if module is None:
        return 0
    total = 0
    for param in module.parameters(recurse=True):
        total += int(param.numel()) * int(param.element_size())
    for buf in module.buffers(recurse=True):
        total += int(buf.numel()) * int(buf.element_size())
    return total


def _array_bytes(arr):
    if arr is None:
        return 0
    if hasattr(arr, "nbytes"):
        return int(arr.nbytes)
    return 0


def _string_bytes(value):
    if value is None:
        return 0
    if isinstance(value, (bytes, bytearray)):
        return int(len(value))
    if isinstance(value, str):
        return int(len(value.encode("utf-8")))
    return 0


def _format_bytes(num_bytes):
    num = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(num)}B"
            return f"{num:.2f}{unit}"
        num /= 1024.0
    return f"{num_bytes}B"


def _print_bundle_sizes(uv_cache, neural_rgb, appear_embed,
                        sphere_center, sphere_radius,
                        mesh_obj=None, mesh_bounds=None, init_camera=None):
    sizes = {
        "uv_cache.coords": _tensor_bytes(uv_cache.coords),
        "uv_cache.points": _tensor_bytes(uv_cache.points),
        "uv_cache.normals": _tensor_bytes(uv_cache.normals),
        "uv_cache.feats": _tensor_bytes(uv_cache.feats),
        "neural_rgb": _module_bytes(neural_rgb),
        "appear_embed": _module_bytes(appear_embed),
        "mesh_obj": _string_bytes(mesh_obj),
        "mesh_bounds": _array_bytes(mesh_bounds),
        "init_camera": _array_bytes(init_camera),
        "sphere_center": _array_bytes(np.asarray(sphere_center, dtype=np.float32)),
        "sphere_radius": _array_bytes(np.asarray([sphere_radius], dtype=np.float32)),
    }
    sizes["uv_cache.total"] = (
        sizes["uv_cache.coords"]
        + sizes["uv_cache.points"]
        + sizes["uv_cache.normals"]
        + sizes["uv_cache.feats"]
    )
    total = sum(sizes.values())
    print("UV bundle sizes:")
    for key in (
        "uv_cache.coords",
        "uv_cache.points",
        "uv_cache.normals",
        "uv_cache.feats",
        "uv_cache.total",
        "neural_rgb",
        "appear_embed",
        "mesh_obj",
        "mesh_bounds",
        "init_camera",
        "sphere_center",
        "sphere_radius",
    ):
        print(f"  {key}: {_format_bytes(sizes[key])}")
    print(f"  total_estimate: {_format_bytes(total)}")


def _rasterize_uv_gpu(vertices, faces, uvs, texture_size, device):
    dr = _get_nvdiffrast()
    height = width = int(texture_size)
    uv = torch.from_numpy(uvs).to(device=device, dtype=torch.float32).clone()
    uv[:, 1] = 1.0 - uv[:, 1]
    pos = torch.zeros((1, uv.shape[0], 4), device=device, dtype=torch.float32)
    pos[0, :, 0:2] = uv * 2.0 - 1.0
    pos[0, :, 2] = 0.0
    pos[0, :, 3] = 1.0
    tri = torch.from_numpy(faces).to(device=device, dtype=torch.int32)
    ctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(ctx, pos, tri, resolution=[height, width])
    tri_id = rast[..., 3]
    if tri_id.min().item() < 0:
        mask = tri_id >= 0
    else:
        mask = tri_id > 0
    mask_2d = mask[0]
    idx = torch.nonzero(mask_2d, as_tuple=False)
    if idx.numel() == 0:
        return idx, torch.empty((0, 3), device=device, dtype=torch.float32)
    verts = torch.from_numpy(vertices).to(device=device, dtype=torch.float32).unsqueeze(0)
    attr, _ = dr.interpolate(verts, rast, tri)
    points = attr[0][mask_2d]
    return idx.to(dtype=torch.int64), points


def _rasterize_uv_cpu(vertices, faces, uvs, texture_size, device):
    face_idx, bary = _rasterize_uv(uvs, faces, texture_size)
    valid = face_idx >= 0
    ys, xs = np.where(valid)
    if ys.size == 0:
        coords = torch.zeros((0, 2), device=device, dtype=torch.int64)
        points = torch.zeros((0, 3), device=device, dtype=torch.float32)
        return coords, points
    face_ids = face_idx[ys, xs]
    bary_vals = bary[ys, xs]
    tri = vertices[faces[face_ids]]
    points_np = (tri * bary_vals[:, :, None]).sum(axis=1)
    coords = torch.from_numpy(np.stack([ys, xs], axis=1)).to(device=device, dtype=torch.int64)
    points = torch.from_numpy(points_np).to(device=device, dtype=torch.float32)
    return coords, points


def _project_points_to_surface(points, neural_sdf, batch_size=65536, iters=1, step=1.0, max_step=0.0):
    total = points.shape[0]
    if total == 0:
        return points
    iters = max(int(iters), 1)
    step = float(step)
    max_step = float(max_step)
    if max_step <= 0.0:
        max_step = None
    with torch.no_grad():
        for _ in range(iters):
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                pts = points[start:end]
                sdfs, _ = neural_sdf(pts)
                grads, _ = neural_sdf.compute_gradients(pts, training=False, sdf=sdfs)
                normals = torch_F.normalize(grads, dim=-1)
                sdf_vals = sdfs.squeeze(-1) if sdfs.ndim > 1 else sdfs
                step_vec = sdf_vals[:, None] * normals * step
                if max_step is not None:
                    step_norm = torch.linalg.norm(step_vec, dim=-1, keepdim=True).clamp_min(1e-6)
                    scale = torch.clamp(max_step / step_norm, max=1.0)
                    step_vec = step_vec * scale
                points[start:end] = pts - step_vec
    return points


def _compute_feats_normals(points, neural_sdf, batch_size=65536):
    device = points.device
    total = points.shape[0]
    feats = None
    normals = torch.empty((total, 3), device=device, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            pts = points[start:end]
            sdfs, feat = neural_sdf(pts)
            grads, _ = neural_sdf.compute_gradients(pts, training=False, sdf=sdfs)
            normals[start:end] = torch_F.normalize(grads, dim=-1)
            if feats is None:
                feats = torch.empty((total, feat.shape[-1]), device=device, dtype=feat.dtype)
            feats[start:end] = feat
    return feats, normals
