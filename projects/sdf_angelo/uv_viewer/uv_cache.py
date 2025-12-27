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
                   raster_mode="gpu", batch_size=65536, device=None):
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

    feats, normals = _compute_feats_normals(points, neural_sdf, batch_size=batch_size)
    return UVCache(
        tex_size=int(texture_size),
        coords=coords,
        points=points,
        normals=normals,
        feats=feats,
        device=device,
    )


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
