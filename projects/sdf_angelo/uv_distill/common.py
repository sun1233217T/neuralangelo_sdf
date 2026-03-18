import importlib
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as torch_F
import trimesh
from PIL import Image

from projects.sdf_angelo.utils.mesh import _dilate_texture, _get_nvdiffrast
from projects.sdf_angelo.uv_viewer.uv_cache import build_uv_cache, load_uv_bundle


@dataclass
class CameraRecord:
    index: int
    kind: str
    intr: torch.Tensor
    pose: torch.Tensor
    image_size: tuple
    camera_center_norm: torch.Tensor
    camera_center_world: torch.Tensor
    source_indices: tuple


def load_mesh(mesh_path):
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    mesh_loaded = trimesh.load(mesh_path, process=False)
    if isinstance(mesh_loaded, trimesh.Scene):
        if not mesh_loaded.geometry:
            raise ValueError(f"Mesh scene is empty: {mesh_path}")
        return trimesh.util.concatenate(tuple(mesh_loaded.geometry.values()))
    return mesh_loaded


def build_dataset(cfg, split="train"):
    lib_data = importlib.import_module(cfg.data.type)
    is_inference = split != "train"
    return lib_data.Dataset(cfg, is_inference=is_inference)


def load_scene_normalization(cfg):
    meta_fname = os.path.join(cfg.data.root, "transforms.json")
    with open(meta_fname, "r") as file:
        meta = json.load(file)
    center = np.array(meta.get("sphere_center", [0.0, 0.0, 0.0]), dtype=np.float32)
    radius = float(meta.get("sphere_radius", 1.0))
    readjust = getattr(cfg.data, "readjust", None)
    if readjust is not None:
        center += np.array(getattr(readjust, "center", [0.0, 0.0, 0.0]), dtype=np.float32)
        radius *= float(getattr(readjust, "scale", 1.0))
    return center, radius, meta


def camera_center_from_pose(pose):
    if pose.ndim == 3:
        pose = pose[0]
    R = pose[:3, :3]
    t = pose[:3, 3]
    return -R.t().matmul(t)


def pose_to_c2w(pose):
    c2w = torch.eye(4, dtype=pose.dtype)
    R = pose[:3, :3]
    t = pose[:3, 3]
    c2w[:3, :3] = R.t()
    c2w[:3, 3] = -R.t().matmul(t)
    return c2w


def c2w_to_pose(c2w):
    w2c = torch.eye(4, dtype=c2w.dtype)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    w2c[:3, :3] = R.t()
    w2c[:3, 3] = -R.t().matmul(t)
    return w2c[:3]


def load_original_camera_records(dataset, sphere_center, sphere_radius):
    records = []
    for idx in range(len(dataset.list)):
        intr, pose = dataset.cameras[idx] if dataset.preload else dataset.get_camera(idx)
        _, image_size_raw = dataset.images[idx] if dataset.preload else dataset.get_image(idx)
        intr, pose = dataset.preprocess_camera(intr, pose, image_size_raw)
        center_norm = camera_center_from_pose(pose)
        center_world = center_norm * float(sphere_radius) + torch.as_tensor(sphere_center, dtype=torch.float32)
        records.append(
            CameraRecord(
                index=idx,
                kind="original",
                intr=intr.clone(),
                pose=pose.clone(),
                image_size=(int(dataset.H), int(dataset.W)),
                camera_center_norm=center_norm.clone(),
                camera_center_world=center_world.clone(),
                source_indices=(idx,),
            )
        )
    return records


def _make_app_value(appear_embed, appear_idx, device):
    if appear_embed is None:
        return None
    dim = appear_embed.embedding_dim
    if appear_idx is None or int(appear_idx) < 0:
        return torch.zeros((1, dim), device=device)
    idx = torch.tensor([int(appear_idx)], device=device)
    return appear_embed(idx).detach()


def save_geometry_cache(path, uv_cache, sphere_center, sphere_radius):
    payload = {
        "tex_size": int(uv_cache.tex_size),
        "coords": uv_cache.coords.detach().cpu().to(dtype=torch.int32),
        "points": uv_cache.points.detach().cpu().to(dtype=torch.float16),
        "normals": uv_cache.normals.detach().cpu().to(dtype=torch.float16),
        "sphere_center": np.asarray(sphere_center, dtype=np.float32),
        "sphere_radius": float(sphere_radius),
    }
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(payload, path)


def load_geometry_cache(path, device=None):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "tex_size": int(payload["tex_size"]),
        "coords": payload["coords"].to(device=device, dtype=torch.int64),
        "points": payload["points"].to(device=device, dtype=torch.float32),
        "normals": payload["normals"].to(device=device, dtype=torch.float32),
        "sphere_center": np.asarray(payload["sphere_center"], dtype=np.float32),
        "sphere_radius": float(payload["sphere_radius"]),
    }


def render_teacher_texture(uv_cache, neural_rgb, appear_embed, cam_pos_world,
                           sphere_center, sphere_radius, batch_size=65536,
                           appear_idx=None, pad_iters=0):
    device = uv_cache.device
    cam_norm = (torch.as_tensor(cam_pos_world, device=device, dtype=torch.float32)
                - torch.as_tensor(sphere_center, device=device, dtype=torch.float32)) / float(sphere_radius)
    points = uv_cache.points
    normals = uv_cache.normals
    coords = uv_cache.coords
    rays = points - cam_norm[None, :]
    rays_unit = torch_F.normalize(rays, dim=-1)
    total = points.shape[0]
    rgbs = torch.empty((total, 3), device=device, dtype=torch.float32)
    app_value = _make_app_value(appear_embed, appear_idx, device=device)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            pts = points[start:end].view(1, 1, -1, 3)
            nrm = normals[start:end].view(1, 1, -1, 3)
            ray = rays_unit[start:end].view(1, 1, -1, 3)
            feat = uv_cache.feats[start:end].view(1, 1, -1, uv_cache.feats.shape[-1])
            app = None
            if app_value is not None:
                app = app_value.view(1, 1, 1, -1)
            rgb = neural_rgb(pts, nrm, ray, feat, app=app)
            rgbs[start:end] = rgb.view(-1, 3)
    texture = torch.zeros((uv_cache.tex_size, uv_cache.tex_size, 3),
                          device=device, dtype=torch.float32)
    texture[coords[:, 0], coords[:, 1]] = rgbs
    if pad_iters > 0 and coords.numel() > 0:
        mask = torch.zeros((uv_cache.tex_size, uv_cache.tex_size), device=device, dtype=torch.bool)
        mask[coords[:, 0], coords[:, 1]] = True
        texture = _dilate_texture(texture, mask, pad_iters)
    return texture.clamp(0.0, 1.0)


class DepthVisibilityRenderer:

    def __init__(self, vertices_norm, faces, device):
        self.device = device
        self.vertices = torch.as_tensor(vertices_norm, device=device, dtype=torch.float32)
        self.faces = torch.as_tensor(faces, device=device, dtype=torch.int32)
        self.dr = _get_nvdiffrast()
        self.ctx = self.dr.RasterizeCudaContext()

    def render_depth(self, intr, pose, image_size):
        height, width = int(image_size[0]), int(image_size[1])
        intr = intr.to(device=self.device, dtype=torch.float32)
        pose = pose.to(device=self.device, dtype=torch.float32)
        verts = self.vertices
        pts_cam = verts @ pose[:3, :3].t() + pose[:3, 3]
        z = pts_cam[:, 2]
        valid_z = z > 1e-6
        if not valid_z.any():
            depth = torch.zeros((height, width), device=self.device, dtype=torch.float32)
            mask = torch.zeros((height, width), device=self.device, dtype=torch.bool)
            return depth, mask
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
        uv[:, 1] = 1.0 - uv[:, 1]
        pos = torch.empty((1, verts.shape[0], 4), device=self.device, dtype=torch.float32)
        pos[0, :, 0:2] = uv * 2.0 - 1.0
        pos[0, :, 2] = (2.0 * z - (far + near)) / (far - near)
        pos[0, :, 3] = 1.0
        invalid = ~valid_z
        if invalid.any():
            pos[0, invalid, 0:3] = 2.0
        rast, _ = self.dr.rasterize(self.ctx, pos, self.faces, resolution=[height, width])
        tri_id = rast[..., 3]
        if tri_id.min().item() < 0:
            mask = tri_id >= 0
        else:
            mask = tri_id > 0
        z_attr = z.view(1, -1, 1).contiguous()
        depth, _ = self.dr.interpolate(z_attr, rast, self.faces)
        return depth[0, ..., 0].contiguous(), mask[0].contiguous()

    def compute_visible_mask(self, points, normals, intr, pose, image_size,
                             camera_center_norm, chunk_size=262144,
                             depth_tol_abs=1e-3, depth_tol_rel=1e-2, front_only=True,
                             use_depth_test=True, return_debug=False):
        if use_depth_test:
            depth_map, depth_valid = self.render_depth(intr, pose, image_size)
        else:
            depth_map = None
            depth_valid = None
        height, width = int(image_size[0]), int(image_size[1])
        intr = intr.to(device=self.device, dtype=torch.float32)
        pose = pose.to(device=self.device, dtype=torch.float32)
        camera_center_norm = torch.as_tensor(camera_center_norm, device=self.device, dtype=torch.float32)
        total = points.shape[0]
        visible = torch.zeros((total,), device=self.device, dtype=torch.bool)
        debug = None
        if return_debug:
            debug = {
                "valid_z": torch.zeros((total,), device=self.device, dtype=torch.bool),
                "inside": torch.zeros((total,), device=self.device, dtype=torch.bool),
                "depth_valid": torch.zeros((total,), device=self.device, dtype=torch.bool),
                "depth_pass": torch.zeros((total,), device=self.device, dtype=torch.bool),
                "front_pass": torch.zeros((total,), device=self.device, dtype=torch.bool),
            }
        denom_w = max(width - 1, 1)
        denom_h = max(height - 1, 1)
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            pts = points[start:end]
            nrm = normals[start:end]
            pts_cam = pts @ pose[:3, :3].t() + pose[:3, 3]
            z = pts_cam[:, 2]
            valid_z = z > 1e-6
            uvw = pts_cam @ intr.t()
            z_safe = torch.where(valid_z, z, torch.ones_like(z))
            x = uvw[:, 0] / z_safe
            y = uvw[:, 1] / z_safe
            x_int = torch.round(x).to(dtype=torch.int64)
            y_int = torch.round(y).to(dtype=torch.int64)
            inside = valid_z & (x_int >= 0) & (x_int < width) & (y_int >= 0) & (y_int < height)
            if return_debug:
                debug["valid_z"][start:end] = valid_z
                debug["inside"][start:end] = inside
            if not inside.any():
                continue
            idx_inside = torch.nonzero(inside, as_tuple=False).squeeze(1)
            global_idx_inside = start + idx_inside
            if use_depth_test:
                sample_depth = depth_map[y_int[idx_inside], x_int[idx_inside]]
                sample_valid = depth_valid[y_int[idx_inside], x_int[idx_inside]]
                if return_debug:
                    debug["depth_valid"][global_idx_inside] = sample_valid
                tol = float(depth_tol_abs) + float(depth_tol_rel) * sample_depth.abs()
                depth_ok = sample_valid & ((z[idx_inside] - sample_depth).abs() <= tol)
                if return_debug:
                    debug["depth_pass"][global_idx_inside] = depth_ok
            else:
                depth_ok = torch.ones_like(idx_inside, dtype=torch.bool, device=self.device)
                if return_debug:
                    debug["depth_valid"][global_idx_inside] = True
                    debug["depth_pass"][global_idx_inside] = True
            if front_only:
                view_to_camera = torch_F.normalize(camera_center_norm[None, :] - pts[idx_inside], dim=-1)
                front_ok = (nrm[idx_inside] * view_to_camera).sum(dim=-1) > 0
                if return_debug:
                    debug["front_pass"][global_idx_inside] = front_ok
                final_ok = depth_ok & front_ok
            else:
                if return_debug:
                    debug["front_pass"][global_idx_inside] = True
                final_ok = depth_ok
            visible[start:end][idx_inside] = final_ok
        if return_debug:
            debug["visible"] = visible
            return visible, debug
        return visible


def save_rgb_png(path, tensor):
    arr = (tensor.clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
    Image.fromarray(arr).save(path)


def save_mask_png(path, mask):
    arr = mask.to(dtype=torch.uint8).mul_(255).cpu().numpy()
    Image.fromarray(arr, mode="L").save(path)


def load_teacher_context(args, cfg=None):
    mesh_space = getattr(args, "mesh_space", "world")
    if args.uv_bundle:
        device = torch.device(f"cuda:{int(args.local_rank)}" if torch.cuda.is_available() else "cpu")
        uv_cache, neural_rgb, appear_embed, meta = load_uv_bundle(args.uv_bundle, device=device)
        sphere_center = np.asarray(meta.get("sphere_center", [0.0, 0.0, 0.0]), dtype=np.float32)
        sphere_radius = float(meta.get("sphere_radius", 1.0))
        if mesh_space == "world":
            mesh_center = sphere_center.copy()
            mesh_radius = sphere_radius
        elif mesh_space == "normalized":
            mesh_center = np.zeros(3, dtype=np.float32)
            mesh_radius = 1.0
        else:
            raise ValueError(f"Unsupported mesh_space: {mesh_space}")
        return {
            "uv_cache": uv_cache,
            "neural_rgb": neural_rgb,
            "appear_embed": appear_embed,
            "sphere_center": sphere_center,
            "sphere_radius": sphere_radius,
            "mesh_center": mesh_center,
            "mesh_radius": mesh_radius,
            "mesh_space": mesh_space,
            "device": device,
        }
    if cfg is None:
        raise ValueError("cfg is required when --uv_bundle is not provided.")
    from imaginaire.trainers.utils.get_trainer import get_trainer
    trainer = get_trainer(cfg, is_inference=True, seed=0)
    trainer.checkpointer.load(args.checkpoint, load_opt=False, load_sch=False)
    trainer.model.eval()
    trainer.current_iteration = trainer.checkpointer.eval_iteration
    if cfg.model.object.sdf.encoding.coarse2fine.enabled:
        trainer.model_module.neural_sdf.set_active_levels(trainer.current_iteration)
        if cfg.model.object.sdf.gradient.mode == "numerical":
            trainer.model_module.neural_sdf.set_normal_epsilon()
    sphere_center, sphere_radius, _ = load_scene_normalization(cfg)
    if mesh_space == "world":
        mesh_center = sphere_center
        mesh_radius = sphere_radius
    elif mesh_space == "normalized":
        mesh_center = np.zeros(3, dtype=np.float32)
        mesh_radius = 1.0
    else:
        raise ValueError(f"Unsupported mesh_space: {mesh_space}")
    mesh = load_mesh(args.mesh)
    uv_cache = build_uv_cache(
        mesh,
        trainer.model_module.neural_sdf,
        sphere_center=mesh_center,
        sphere_radius=mesh_radius,
        texture_size=args.texture_size,
        raster_mode=args.uv_raster,
        batch_size=args.batch_size,
        project_to_surface=getattr(args, "uv_project_to_surface", False),
        project_iters=getattr(args, "uv_project_iters", 1),
        project_step=getattr(args, "uv_project_step", 1.0),
        project_max_step=getattr(args, "uv_project_max_step", 0.0),
    )
    return {
        "uv_cache": uv_cache,
        "neural_rgb": trainer.model_module.neural_rgb,
        "appear_embed": trainer.model_module.appear_embed,
        "sphere_center": sphere_center,
        "sphere_radius": sphere_radius,
        "mesh_center": np.asarray(mesh_center, dtype=np.float32),
        "mesh_radius": float(mesh_radius),
        "mesh_space": mesh_space,
        "device": next(trainer.model_module.neural_rgb.parameters()).device,
    }
