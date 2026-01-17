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

import argparse
import importlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as torch_F
import trimesh
from PIL import Image

sys.path.append(os.getcwd())

from imaginaire.config import Config, recursive_update_strict, parse_cmdline_arguments  # noqa: E402
from imaginaire.utils.distributed import (  # noqa: E402
    init_dist,
    get_world_size,
    get_rank,
    is_master,
    master_only_print as print,
    dist_reduce_tensor,
)
from imaginaire.utils.gpu_affinity import set_affinity  # noqa: E402
from imaginaire.trainers.utils.get_trainer import get_trainer  # noqa: E402
from projects.sdf_angelo.uv_viewer.uv_cache import build_uv_cache  # noqa: E402
from projects.sdf_angelo.utils.mesh import _get_nvdiffrast, _dilate_texture  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="UV mesh PSNR evaluation")
    parser.add_argument("--config", required=True, help="Path to the training config file.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--mesh", required=True, help="Path to mesh file (OBJ/PLY).")
    parser.add_argument("--color_mode", choices=["uv", "vertex"], default="uv",
                        help="Mesh color source: uv (neural texture) or vertex (PLY colors).")
    parser.add_argument("--sanitize_mesh", action="store_true",
                        help="Drop invalid faces (out-of-range or non-finite vertices).")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--texture_size", default=1024, type=int, help="UV texture resolution.")
    parser.add_argument("--batch_size", default=65536, type=int, help="Batch size for SDF/RGB queries.")
    parser.add_argument("--uv_padding", default=8, type=int,
                        help="Texel padding iterations for UV texture (0 to disable).")
    parser.add_argument("--uv_raster", choices=["gpu", "cpu"], default="gpu",
                        help="Rasterization backend for UV cache.")
    parser.add_argument("--mesh_space", choices=["world", "normalized"], default="world",
                        help="Mesh coordinate space (world or normalized).")
    parser.add_argument("--appear_idx", default=None, type=int,
                        help="Fixed appearance index for all views. Use -1 for zero embedding.")
    parser.add_argument("--max_views", default=0, type=int, help="Limit number of views (0 = all).")
    parser.add_argument("--ignore_alpha", action="store_true", help="Ignore alpha channel if present.")
    parser.add_argument("--mask_psnr", action="store_true",
                        help="Compute PSNR only on rendered pixels.")
    parser.add_argument("--log_interval", default=10, type=int, help="Log every N views.")
    parser.add_argument("--flip_y", dest="flip_y", action="store_true",
                        help="Flip screen-space Y (default: enabled).")
    parser.add_argument("--no_flip_y", dest="flip_y", action="store_false",
                        help="Disable screen-space Y flip.")
    parser.add_argument("--debug_dir", default="", type=str,
                        help="Output directory for debug renders (empty to disable).")
    parser.add_argument("--debug_every", default=0, type=int,
                        help="Save debug images every N views (0 to disable).")
    parser.add_argument("--debug_max_views", default=20, type=int,
                        help="Maximum debug views to save per rank.")
    parser.add_argument("--debug_indices", default="", type=str,
                        help="Comma-separated view indices to save.")
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument('--single_gpu', action='store_true')
    parser.set_defaults(flip_y=True)
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd


def _load_mesh(mesh_path):
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    mesh_loaded = trimesh.load(mesh_path, process=False)
    if isinstance(mesh_loaded, trimesh.Scene):
        if not mesh_loaded.geometry:
            raise ValueError(f"Mesh scene is empty: {mesh_path}")
        mesh = trimesh.util.concatenate(tuple(mesh_loaded.geometry.values()))
    else:
        mesh = mesh_loaded
    return mesh


def _get_vertex_colors(mesh):
    if not hasattr(mesh, "visual") or mesh.visual is None:
        return None
    colors = getattr(mesh.visual, "vertex_colors", None)
    if colors is None:
        return None
    colors = np.asarray(colors)
    if colors.ndim != 2 or colors.shape[0] != len(mesh.vertices):
        return None
    if colors.shape[1] < 3:
        return None
    colors = colors[:, :3].astype(np.float32)
    if colors.size == 0:
        return None
    if colors.max() > 1.0:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def _compute_valid_face_mask(vertices, faces):
    if faces.size == 0:
        return np.zeros((0,), dtype=bool)
    num_vertices = vertices.shape[0]
    in_range = (faces >= 0) & (faces < num_vertices)
    face_mask = in_range.all(axis=1)
    if not face_mask.any():
        return face_mask
    finite = np.isfinite(vertices).all(axis=1)
    valid_idx = np.where(face_mask)[0]
    faces_in_range = faces[valid_idx]
    finite_ok = finite[faces_in_range].all(axis=1)
    face_mask[valid_idx] = finite_ok
    return face_mask


def _validate_or_sanitize_mesh(mesh, sanitize=False):
    if not hasattr(mesh, "faces"):
        raise ValueError("Mesh has no faces attribute; ensure the input is a triangle mesh.")
    faces = np.asarray(mesh.faces)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Mesh faces must have shape [F,3].")
    if faces.size == 0:
        raise ValueError("Mesh has no faces; point clouds are not supported for rasterization.")
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    face_mask = _compute_valid_face_mask(vertices, faces)
    if face_mask.all():
        return mesh
    if not sanitize:
        bad = np.count_nonzero(~face_mask)
        min_idx = int(faces.min()) if faces.size else 0
        max_idx = int(faces.max()) if faces.size else 0
        raise ValueError(
            f"Mesh has {bad} invalid faces (min_idx={min_idx}, max_idx={max_idx}, "
            f"num_vertices={len(vertices)}). Use --sanitize_mesh to drop them."
        )
    mesh.update_faces(face_mask)
    mesh.remove_unreferenced_vertices()
    return mesh


def _get_scene_normalization(cfg, meta):
    center = np.array(meta.get("sphere_center", [0.0, 0.0, 0.0]), dtype=np.float32)
    radius = float(meta.get("sphere_radius", 1.0))
    readjust = getattr(cfg.data, "readjust", None)
    if readjust is not None:
        center += np.array(getattr(readjust, "center", [0.0, 0.0, 0.0]), dtype=np.float32)
        radius *= float(getattr(readjust, "scale", 1.0))
    return center, radius


def _build_dataset(cfg, split):
    lib_data = importlib.import_module(cfg.data.type)
    is_inference = split != "train"
    return lib_data.Dataset(cfg, is_inference=is_inference)


def _load_full_sample(dataset, idx, ignore_alpha=False):
    image, image_size_raw = dataset.images[idx] if dataset.preload else dataset.get_image(idx)
    image, alpha = dataset.preprocess_image(image)
    intr, pose = dataset.cameras[idx] if dataset.preload else dataset.get_camera(idx)
    intr, pose = dataset.preprocess_camera(intr, pose, image_size_raw)
    if alpha is not None and not ignore_alpha:
        if dataset.white_bkgd:
            image = image * alpha + (1 - alpha)
        else:
            image = image * alpha
    return image, intr, pose


def _camera_center_from_pose(pose):
    if pose.ndim == 3:
        pose = pose[0]
    R = pose[:3, :3]
    t = pose[:3, 3]
    return -R.t().matmul(t)


def _make_app_value(appear_embed, appear_idx, device):
    if appear_embed is None:
        return None
    dim = appear_embed.embedding_dim
    if appear_idx is None:
        return torch.zeros((1, 1, 1, dim), device=device)
    idx = torch.tensor([int(appear_idx)], device=device)
    return appear_embed(idx).view(1, 1, 1, -1)


def _parse_indices(text):
    if not text:
        return set()
    result = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            if len(parts) != 2:
                continue
            start = parts[0].strip()
            end = parts[1].strip()
            if not start or not end:
                continue
            try:
                lo = int(start)
                hi = int(end)
            except ValueError:
                continue
            if hi < lo:
                lo, hi = hi, lo
            result.update(range(lo, hi + 1))
        else:
            try:
                result.add(int(token))
            except ValueError:
                continue
    return result


def _to_uint8_image(tensor):
    arr = (tensor.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def _to_uint8_mask(mask, height, width):
    m = mask.detach()
    if m.ndim > 2:
        m = m.squeeze()
    if m.ndim == 2:
        if m.shape != (height, width) and m.numel() == height * width:
            m = m.view(height, width)
    elif m.ndim == 1:
        if m.numel() == height * width:
            m = m.view(height, width)
        else:
            m = m.view(1, -1)
    else:
        m = m.view(1, 1)
    arr = (m.clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
    return Image.fromarray(arr, mode="L")


def _save_debug_images(out_dir, view_idx, render, target, mask, debug_info):
    base = os.path.join(out_dir, f"view_{view_idx:06d}")
    _to_uint8_image(render).save(f"{base}_render.png")
    _to_uint8_image(target).save(f"{base}_gt.png")
    _to_uint8_image((render - target).abs()).save(f"{base}_error.png")
    _to_uint8_mask(mask, render.shape[1], render.shape[2]).save(f"{base}_mask.png")
    with open(f"{base}_info.json", "w") as file:
        json.dump(debug_info, file, indent=2)


def _build_uv_mask(uv_cache):
    tex_size = int(uv_cache.tex_size)
    mask = torch.zeros((tex_size, tex_size), device=uv_cache.device, dtype=torch.bool)
    coords = uv_cache.coords
    if coords.numel() > 0:
        mask[coords[:, 0], coords[:, 1]] = True
    return mask


def render_view_dependent_texture(uv_cache, neural_rgb, appear_embed, cam_pos_world,
                                  sphere_center, sphere_radius, batch_size, appear_idx,
                                  pad_iters=0, pad_mask=None):
    device = uv_cache.device
    cam = cam_pos_world.to(device=device, dtype=torch.float32)
    center = torch.as_tensor(sphere_center, device=device, dtype=torch.float32)
    radius = float(sphere_radius)
    cam_norm = (cam - center) / radius
    points = uv_cache.points
    normals = uv_cache.normals
    feats = uv_cache.feats
    coords = uv_cache.coords
    rays = points - cam_norm[None, :]
    rays_unit = torch_F.normalize(rays, dim=-1)
    total = points.shape[0]
    rgbs = torch.empty((total, 3), device=device, dtype=torch.float32)
    app = _make_app_value(appear_embed, appear_idx, device)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            pts = points[start:end]
            nrm = normals[start:end]
            ray = rays_unit[start:end]
            feat = feats[start:end]
            pts_b = pts.view(1, 1, -1, 3)
            nrm_b = nrm.view(1, 1, -1, 3)
            ray_b = ray.view(1, 1, -1, 3)
            feat_b = feat.view(1, 1, -1, feat.shape[-1])
            rgb_b = neural_rgb(pts_b, nrm_b, ray_b, feat_b, app=app)
            rgbs[start:end] = rgb_b.view(-1, 3)
    texture = torch.zeros((uv_cache.tex_size, uv_cache.tex_size, 3),
                          device=device, dtype=rgbs.dtype)
    texture[coords[:, 0], coords[:, 1]] = rgbs
    if pad_iters > 0 and pad_mask is not None:
        texture = _dilate_texture(texture, pad_mask, pad_iters)
    return texture.clamp(0.0, 1.0)


class MeshRenderer:

    def __init__(self, vertices, faces, device, uvs=None, vertex_colors=None, flip_y=True):
        use_uv = uvs is not None
        use_vertex = vertex_colors is not None
        if use_uv == use_vertex:
            raise ValueError("Provide exactly one of uvs or vertex_colors.")
        dr = _get_nvdiffrast()
        self.dr = dr
        self.ctx = dr.RasterizeCudaContext()
        self.device = device
        self.flip_y = bool(flip_y)
        self.mode = "uv" if use_uv else "vertex"
        self.vertices = torch.as_tensor(vertices, device=device, dtype=torch.float32)
        self.faces = torch.as_tensor(faces, device=device, dtype=torch.int32)
        self.uv_attr = None
        self.color_attr = None
        if use_uv:
            uv = torch.as_tensor(uvs, device=device, dtype=torch.float32).clone()
            uv[:, 1] = 1.0 - uv[:, 1]
            self.uv_attr = uv.unsqueeze(0)
        else:
            color = torch.as_tensor(vertex_colors, device=device, dtype=torch.float32)
            self.color_attr = color.unsqueeze(0)

    def render(self, intr, pose, image_size, background, texture=None):
        height, width = int(image_size[0]), int(image_size[1])
        intr = intr.to(device=self.device, dtype=torch.float32)
        pose = pose.to(device=self.device, dtype=torch.float32)
        verts = self.vertices
        R = pose[:3, :3]
        t = pose[:3, 3]
        pts_cam = verts @ R.t() + t
        z = pts_cam[:, 2]
        valid_z = z > 1e-6
        if not valid_z.any():
            bg = self._background_tensor(background, height, width)
            mask = torch.zeros((1, height, width), device=self.device, dtype=torch.float32)
            return bg, mask
        min_z = z[valid_z].min().item()
        max_z = z[valid_z].max().item()
        near = max(min_z * 0.9, 1e-4)
        far = max(max_z * 1.1, near + 1e-4)
        uvw = pts_cam @ intr.t()
        z_safe = torch.where(valid_z, z, torch.full_like(z, near))
        u = uvw[:, 0] / z_safe
        v = uvw[:, 1] / z_safe
        denom_w = max(width - 1, 1)
        denom_h = max(height - 1, 1)
        uv = torch.stack([u / float(denom_w), v / float(denom_h)], dim=-1)
        if self.flip_y:
            uv[:, 1] = 1.0 - uv[:, 1]
        pos = torch.empty((1, verts.shape[0], 4), device=self.device, dtype=torch.float32)
        pos[0, :, 0:2] = uv * 2.0 - 1.0
        pos[0, :, 2] = (2.0 * z - (far + near)) / (far - near)
        pos[0, :, 3] = 1.0
        invalid = ~valid_z
        if invalid.any():
            pos[0, invalid, 0:3] = 2.0
        rast, _ = self.dr.rasterize(self.ctx, pos, self.faces, resolution=[height, width])
        if self.mode == "uv":
            if texture is None:
                raise ValueError("Texture is required for UV rendering.")
            texc, _ = self.dr.interpolate(self.uv_attr, rast, self.faces)
            tex = texture.unsqueeze(0).contiguous()
            color = self.dr.texture(tex, texc, filter_mode="linear", boundary_mode="clamp")
        else:
            color, _ = self.dr.interpolate(self.color_attr, rast, self.faces)
        tri_id = rast[..., 3]
        if tri_id.min().item() < 0:
            mask = tri_id >= 0
        else:
            mask = tri_id > 0
        mask = mask.to(dtype=torch.float32)
        bg = self._background_tensor(background, height, width)
        color = color * mask.unsqueeze(-1) + bg.permute(1, 2, 0) * (1.0 - mask.unsqueeze(-1))
        color = color.permute(0, 3, 1, 2).contiguous()[0]
        return color, mask.unsqueeze(0)

    def _background_tensor(self, background, height, width):
        if isinstance(background, (tuple, list, np.ndarray)):
            bg = torch.tensor(background, device=self.device, dtype=torch.float32)
            if bg.numel() == 1:
                bg = bg.repeat(3)
        else:
            bg = torch.full((3,), float(background), device=self.device, dtype=torch.float32)
        return bg.view(3, 1, 1).expand(3, height, width)


def compute_psnr(pred, target, mask=None):
    if mask is None:
        mse = torch_F.mse_loss(pred, target)
    else:
        mask = mask.expand_as(pred)
        denom = mask.sum().clamp_min(1.0)
        mse = ((pred - target) ** 2 * mask).sum() / denom
    return -10.0 * torch.log10(mse.clamp_min(1e-10))


def main():
    args, cfg_cmd = parse_args()
    set_affinity(args.local_rank)
    cfg = Config(args.config)
    cfg_cmd = parse_cmdline_arguments(cfg_cmd)
    recursive_update_strict(cfg, cfg_cmd)

    if not args.single_gpu:
        os.environ["NCLL_BLOCKING_WAIT"] = "0"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank, rank=-1, world_size=-1)
    print(f"Running mesh PSNR ({args.color_mode}) with {get_world_size()} GPUs.")
    cfg.logdir = ''

    trainer = None
    if args.color_mode == "uv":
        trainer = get_trainer(cfg, is_inference=True, seed=0)
        trainer.checkpointer.load(args.checkpoint, load_opt=False, load_sch=False)
        trainer.model.eval()
        trainer.current_iteration = trainer.checkpointer.eval_iteration
        if cfg.model.object.sdf.encoding.coarse2fine.enabled:
            trainer.model_module.neural_sdf.set_active_levels(trainer.current_iteration)
            if cfg.model.object.sdf.gradient.mode == "numerical":
                trainer.model_module.neural_sdf.set_normal_epsilon()
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("Vertex color rendering requires CUDA (nvdiffrast).")
        torch.cuda.set_device(int(args.local_rank))

    meta_fname = os.path.join(cfg.data.root, "transforms.json")
    with open(meta_fname, "r") as file:
        meta = json.load(file)

    mesh = _load_mesh(args.mesh)
    mesh = _validate_or_sanitize_mesh(mesh, sanitize=args.sanitize_mesh)
    if args.color_mode == "uv":
        if not hasattr(mesh, "visual") or mesh.visual is None or getattr(mesh.visual, "uv", None) is None:
            raise ValueError("Input mesh does not contain UV coordinates.")
    else:
        vertex_colors = _get_vertex_colors(mesh)
        if vertex_colors is None:
            raise ValueError("Input mesh does not contain per-vertex colors.")

    scene_center, scene_radius = _get_scene_normalization(cfg, meta)
    if args.mesh_space == "world":
        sphere_center = scene_center
        sphere_radius = scene_radius
        vertices_norm = (np.asarray(mesh.vertices, dtype=np.float32) - scene_center[None, :]) / scene_radius
    else:
        sphere_center = np.zeros(3, dtype=np.float32)
        sphere_radius = 1.0
        vertices_norm = np.asarray(mesh.vertices, dtype=np.float32)

    faces = np.asarray(mesh.faces, dtype=np.int32)
    if args.color_mode == "uv":
        uv_cache = build_uv_cache(
            mesh,
            trainer.model_module.neural_sdf,
            sphere_center=sphere_center,
            sphere_radius=sphere_radius,
            texture_size=args.texture_size,
            raster_mode=args.uv_raster,
            batch_size=args.batch_size,
        )
        pad_mask = _build_uv_mask(uv_cache) if args.uv_padding > 0 else None
        uvs = np.asarray(mesh.visual.uv, dtype=np.float32)
        if uvs.shape[0] != vertices_norm.shape[0]:
            raise ValueError("UVs must be per-vertex and match mesh vertices.")
        device = uv_cache.device
        renderer = MeshRenderer(vertices_norm, faces, device=device, uvs=uvs, flip_y=args.flip_y)
    else:
        if vertex_colors.shape[0] != vertices_norm.shape[0]:
            raise ValueError("Vertex colors must match mesh vertices.")
        device = torch.device("cuda", int(args.local_rank))
        renderer = MeshRenderer(vertices_norm, faces, device=device,
                                vertex_colors=vertex_colors, flip_y=args.flip_y)
    dataset = _build_dataset(cfg, args.split)
    total_views = len(dataset)
    if args.max_views and args.max_views > 0:
        total_views = min(total_views, int(args.max_views))
    indices = list(range(total_views))
    if get_world_size() > 1:
        indices = indices[get_rank()::get_world_size()]

    debug_dir = args.debug_dir
    debug_every = int(args.debug_every)
    debug_max_views = int(args.debug_max_views)
    debug_indices = _parse_indices(args.debug_indices)
    debug_saved = 0
    if debug_dir:
        if get_world_size() > 1:
            debug_dir = os.path.join(debug_dir, f"rank_{get_rank()}")
        os.makedirs(debug_dir, exist_ok=True)

    psnr_sum = 0.0
    count = 0
    for view_idx in indices:
        image, intr, pose = _load_full_sample(dataset, view_idx, ignore_alpha=args.ignore_alpha)
        image = image.to(device=device, dtype=torch.float32)
        intr = intr.to(device=device, dtype=torch.float32)
        pose = pose.to(device=device, dtype=torch.float32)
        cam_center_norm = _camera_center_from_pose(pose)
        cam_center_world = None
        if args.mesh_space == "world":
            cam_center_world = cam_center_norm * float(scene_radius) + torch.as_tensor(
                scene_center, device=device, dtype=torch.float32
            )
        if args.color_mode == "uv":
            cam_pos_world = cam_center_world if cam_center_world is not None else cam_center_norm
            if args.appear_idx is None:
                appear_idx = view_idx
            elif int(args.appear_idx) < 0:
                appear_idx = None
            else:
                appear_idx = int(args.appear_idx)
            texture = render_view_dependent_texture(
                uv_cache=uv_cache,
                neural_rgb=trainer.model_module.neural_rgb,
                appear_embed=trainer.model_module.appear_embed,
                cam_pos_world=cam_pos_world,
                sphere_center=sphere_center,
                sphere_radius=sphere_radius,
                batch_size=args.batch_size,
                appear_idx=appear_idx,
                pad_iters=args.uv_padding,
                pad_mask=pad_mask,
            )
            render, mask = renderer.render(
                texture=texture,
                intr=intr,
                pose=pose,
                image_size=image.shape[-2:],
                background=1.0 if cfg.model.background.white else 0.0,
            )
        else:
            render, mask = renderer.render(
                intr=intr,
                pose=pose,
                image_size=image.shape[-2:],
                background=1.0 if cfg.model.background.white else 0.0,
            )
        if args.mask_psnr:
            psnr = compute_psnr(render, image, mask=mask)
        else:
            psnr = compute_psnr(render, image, mask=None)
        psnr_sum += psnr.item()
        count += 1
        if debug_dir and debug_saved < debug_max_views:
            if debug_indices:
                should_save = view_idx in debug_indices
            elif debug_every > 0:
                should_save = (count % debug_every == 0)
            else:
                should_save = True
            if should_save:
                debug_info = {
                    "view_idx": int(view_idx),
                    "color_mode": args.color_mode,
                    "mesh_space": args.mesh_space,
                    "psnr": float(psnr.item()),
                    "camera_center_norm": cam_center_norm.detach().cpu().tolist(),
                    "camera_center_world": cam_center_world.detach().cpu().tolist()
                    if cam_center_world is not None else None,
                    "intr": intr.detach().cpu().tolist(),
                    "pose": pose.detach().cpu().tolist(),
                    "image_size": [int(image.shape[-2]), int(image.shape[-1])],
                }
                _save_debug_images(debug_dir, view_idx, render, image, mask, debug_info)
                debug_saved += 1
        if is_master() and args.log_interval > 0 and count % args.log_interval == 0:
            print(f"[{count}/{len(indices)}] PSNR: {psnr.item():.4f}")

    psnr_tensor = torch.tensor([psnr_sum], device=device, dtype=torch.float32)
    count_tensor = torch.tensor([count], device=device, dtype=torch.float32)
    psnr_tensor = dist_reduce_tensor(psnr_tensor, reduce="sum")
    count_tensor = dist_reduce_tensor(count_tensor, reduce="sum")
    if is_master():
        denom = max(count_tensor.item(), 1.0)
        avg_psnr = (psnr_tensor / denom).item()
        print(f"Average PSNR ({args.split}): {avg_psnr:.4f} over {int(denom)} views.")


if __name__ == "__main__":
    main()
