import json
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as torch_F
import wandb
from tqdm import tqdm   

from imaginaire.utils.distributed import barrier, is_dist, is_master, master_only_print
from projects.nerf.utils import camera as nerf_camera
from projects.sdf_GS_gengeration.submodels.ECGS.gaussian_model import GaussianModel as ECGSGaussianModel


class NeuralsdfAssistantModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.neural_sdf.sdf(x)

    def forward_color(self, x, *args, **kwargs):
        if hasattr(self.model, "neural_rgb"):
            return self.model.neural_rgb.forward(x, *args, **kwargs)
        if hasattr(self.model, "neural_gs"):
            return self.model.neural_gs.forward(x, *args, **kwargs)
        raise AttributeError("Model has no neural_rgb or neural_gs for color prediction.")


class SDF_GS_Trainer(nn.Module):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.model_module = model.module if hasattr(model, "module") else model
        self.full_cfg = cfg
        self.cfg = getattr(cfg, "GS_pretrainer", None)
        self.enabled = bool(self.cfg and self._cfg_get(self.cfg, "enabled", False))
        self.device = next(self.model_module.neural_sdf.parameters()).device
        self.global_step = 0
        self.optimizer = None
        self.scheduler = None

        self.gaussian = None
        self.gaussian_xyz = None
        self.gaussian_scales = None
        self.gaussian_rots = None
        self.gaussian_aabb_min = None
        self.gaussian_aabb_max = None
        self.pts_xyz = None
        self.pts_normals = None
        self._dumped_points = False

        if not self.enabled:
            return

        self._setup_from_cfg()

    def _setup_from_cfg(self):
        self.num_iters = int(self._cfg_get(self.cfg, "num_iters", 0))
        self.batch_size = int(self._cfg_get(self.cfg, "batch_size", 65536))
        self.radius_std = float(self._cfg_get(self.cfg, "radius_std", 0.0))
        self.sdf_mode = str(self._cfg_get(self.cfg, "sdf_mode", "zero")).lower()
        self.loss_type = str(self._cfg_get(self.cfg, "loss_type", "l1")).lower()
        self.huber_delta = float(self._cfg_get(self.cfg, "huber_delta", 0.1))
        self.opacity_thresh = float(self._cfg_get(self.cfg, "opacity_thresh", 0.0))
        self.max_gaussians = int(self._cfg_get(self.cfg, "max_gaussians", 0))
        self.input_pts = self._cfg_get(self.cfg, "input_pts", None)
        self.pts_sample_std = float(self._cfg_get(self.cfg, "sample_std", 0.05))
        self.pts_samples_per_point = int(self._cfg_get(self.cfg, "N_samples", 1))
        self.pts_max_points = int(self._cfg_get(self.cfg, "max_points", 0))
        self.normalize = bool(self._cfg_get(self.cfg, "normalize", True))
        self.normalize_mode = str(self._cfg_get(self.cfg, "normalize_mode", "dataset")).lower()
        self.gs_cameras_path = self._cfg_get(self.cfg, "gs_cameras_path", None)
        self.align_allow_reflection = bool(self._cfg_get(self.cfg, "align_allow_reflection", False))
        self.apply_gl_to_cv = bool(self._cfg_get(self.cfg, "apply_gl_to_cv", True))
        self.eikonal_weight = float(self._cfg_get(self.cfg, "eikonal_weight", 0.0))
        self.clamping_distance = float(self._cfg_get(self.cfg, "clamping_distance", 0.0))
        self.log_interval = int(self._cfg_get(self.cfg, "log_interval", 100))
        self.log_wandb = bool(self._cfg_get(self.cfg, "log_wandb", False))
        self.wandb_step_offset = int(self._cfg_get(self.cfg, "wandb_step_offset", 0))
        self.force_full_levels = bool(self._cfg_get(self.cfg, "force_full_levels", False))
        self.dump_points_path = self._cfg_get(self.cfg, "dump_points_path", None)
        self.dump_points_num = int(self._cfg_get(self.cfg, "dump_points_num", 0))
        self.data_root = self._cfg_get(self.cfg, "data_root", None)
        if self.data_root is None and hasattr(self.full_cfg, "data"):
            self.data_root = getattr(self.full_cfg.data, "root", None)
        self.readjust = getattr(self.full_cfg.data, "readjust", None) if hasattr(self.full_cfg, "data") else None

        if self.sdf_mode not in ("zero", "ellipsoid", "random"):
            master_only_print(f"Unknown sdf_mode {self.sdf_mode}, fallback to zero.")
            self.sdf_mode = "zero"
        if self.normalize_mode not in ("dataset", "camera_align"):
            master_only_print(f"Unknown normalize_mode {self.normalize_mode}, fallback to dataset.")
            self.normalize_mode = "dataset"
        if self.pts_samples_per_point <= 0:
            master_only_print("GS pretrainer: N_samples <= 0, fallback to 1.")
            self.pts_samples_per_point = 1

        if self.apply_gl_to_cv:
            self._coord_flip = torch.tensor([1.0, -1.0, -1.0], device=self.device)
            self._coord_flip_mat = torch.diag(self._coord_flip)
        else:
            self._coord_flip = None
            self._coord_flip_mat = None

    def _cfg_get(self, cfg, key, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _resolve_gaussian_path(self):
        gaussian_path = self._cfg_get(self.cfg, "gaussian_path", None)
        if not gaussian_path:
            return None
        if os.path.isabs(gaussian_path):
            return gaussian_path
        candidates = []
        if self.data_root:
            candidates.append(os.path.join(self.data_root, gaussian_path))
        candidates.append(os.path.join(os.getcwd(), gaussian_path))
        for path in candidates:
            if os.path.exists(path):
                return path
        return gaussian_path

    def _resolve_input_pts(self):
        pts_path = self._cfg_get(self.cfg, "input_pts", None)
        if not pts_path:
            return None
        if os.path.isabs(pts_path):
            return pts_path
        candidates = []
        if self.data_root:
            candidates.append(os.path.join(self.data_root, pts_path))
        candidates.append(os.path.join(os.getcwd(), pts_path))
        for path in candidates:
            if os.path.exists(path):
                return path
        return pts_path

    def _resolve_dump_points_path(self):
        path = self.dump_points_path
        if not path:
            return None
        if os.path.isabs(path):
            return path
        logdir = getattr(self.full_cfg, "logdir", None)
        if logdir:
            return os.path.join(logdir, path)
        return os.path.join(os.getcwd(), path)

    def _resolve_gs_cameras_path(self, gaussian_path):
        if self.gs_cameras_path:
            if os.path.isabs(self.gs_cameras_path):
                return self.gs_cameras_path
            if self.data_root:
                candidate = os.path.join(self.data_root, self.gs_cameras_path)
                if os.path.exists(candidate):
                    return candidate
            return os.path.join(os.getcwd(), self.gs_cameras_path)
        if not gaussian_path:
            return None
        logdir = os.path.dirname(os.path.dirname(os.path.dirname(gaussian_path)))
        candidate = os.path.join(logdir, "cameras.json")
        if os.path.exists(candidate):
            return candidate
        return None

    def _apply_gl_to_cv_xyz(self, xyz):
        if not self.apply_gl_to_cv:
            return xyz
        return xyz * self._coord_flip.to(xyz)

    def _apply_gl_to_cv_rot(self, rots):
        if not self.apply_gl_to_cv:
            return rots
        rot_mat = self._quat_to_matrix(rots)
        rot_mat = self._coord_flip_mat.to(rot_mat) @ rot_mat @ self._coord_flip_mat.to(rot_mat)
        rots = nerf_camera.quaternion.R_to_q(rot_mat)
        return torch_F.normalize(rots, dim=-1)

    def _load_pts_file(self, pts_path):
        try:
            data = np.loadtxt(pts_path)
        except Exception as exc:
            self._log(f"GS pretrainer: failed to read input_pts {pts_path}: {exc}")
            return None
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[1] < 6:
            self._log(f"GS pretrainer: input_pts needs at least 6 columns (x y z nx ny nz): {pts_path}")
            return None
        points = data[:, :3].astype(np.float32)
        normals = data[:, 3:6].astype(np.float32)
        mask = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1)
        norms = np.linalg.norm(normals, axis=1)
        mask &= norms > 1e-8
        if not np.all(mask):
            removed = int(np.size(mask) - np.count_nonzero(mask))
            self._log(f"GS pretrainer: filtered {removed} invalid points from {pts_path}")
            points = points[mask]
            normals = normals[mask]
            norms = norms[mask]
        normals = normals / norms[:, None]
        return points, normals

    def _load_gs_camera_centers(self, gaussian_path):
        cam_path = self._resolve_gs_cameras_path(gaussian_path)
        if not cam_path or not os.path.isfile(cam_path):
            self._log(f"GS pretrainer camera align: cameras.json not found: {cam_path}")
            return None
        with open(cam_path, "r") as file:
            data = json.load(file)
        centers = {}
        if isinstance(data, list) and data and "position" in data[0]:
            for cam in data:
                name = os.path.basename(cam.get("img_name", ""))
                pos = cam.get("position", None)
                if name and pos is not None:
                    centers[name] = np.array(pos, dtype=np.float32)
        elif isinstance(data, dict) and "frames" in data:
            for frame in data["frames"]:
                name = os.path.basename(frame.get("file_path", ""))
                mat = np.array(frame.get("transform_matrix", []), dtype=np.float32)
                if name and mat.shape == (4, 4):
                    centers[name] = mat[:3, 3]
        if not centers:
            self._log(f"GS pretrainer camera align: no centers found in {cam_path}")
            return None
        return centers

    def _load_sdf_camera_centers(self):
        if not self.data_root:
            self._log("GS pretrainer camera align: data_root is empty.")
            return None
        meta_path = os.path.join(self.data_root, "transforms.json")
        if not os.path.isfile(meta_path):
            self._log(f"GS pretrainer camera align: transforms.json not found: {meta_path}")
            return None
        with open(meta_path, "r") as file:
            meta = json.load(file)
        centers = {}
        for frame in meta.get("frames", []):
            name = os.path.basename(frame.get("file_path", ""))
            mat = np.array(frame.get("transform_matrix", []), dtype=np.float32)
            if not name or mat.shape != (4, 4):
                continue
            c2w_gl = mat
            c2w_cv = c2w_gl * np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float32)
            pos = c2w_cv[:3, 3]
            centers[name] = pos
        if not centers:
            self._log("GS pretrainer camera align: no frames found in transforms.json")
            return None
        return centers

    @staticmethod
    def _compute_similarity_transform(src, dst, allow_reflection=False):
        if src.shape != dst.shape or src.shape[0] < 3:
            return None
        num = src.shape[0]
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        src_centered = src - src_mean
        dst_centered = dst - dst_mean
        cov = (dst_centered.T @ src_centered) / num
        u, s, vt = np.linalg.svd(cov)
        d = np.ones(3, dtype=np.float32)
        if np.linalg.det(u) * np.linalg.det(vt) < 0:
            d[-1] = -1.0
        r = u @ np.diag(d) @ vt
        if not allow_reflection and np.linalg.det(r) < 0:
            d[-1] *= -1.0
            r = u @ np.diag(d) @ vt
        var_src = (src_centered ** 2).sum() / num
        if var_src <= 0:
            return None
        scale = (s * d).sum() / var_src
        t = dst_mean - scale * (r @ src_mean)
        return r.astype(np.float32), float(scale), t.astype(np.float32)

    def _align_gaussians_by_cameras(self, xyz, scales, rots, gaussian_path):
        gs_centers = self._load_gs_camera_centers(gaussian_path)
        sdf_centers = self._load_sdf_camera_centers()
        if not gs_centers or not sdf_centers:
            return None
        common = sorted(set(gs_centers.keys()) & set(sdf_centers.keys()))
        if len(common) < 3:
            self._log(f"GS pretrainer camera align: not enough common frames ({len(common)}).")
            return None
        src = np.stack([gs_centers[k] for k in common], axis=0)
        dst = np.stack([sdf_centers[k] for k in common], axis=0)
        result = self._compute_similarity_transform(src, dst, allow_reflection=self.align_allow_reflection)
        if result is None:
            self._log("GS pretrainer camera align: failed to compute similarity transform.")
            return None
        r, scale, t = result
        self._log(f"GS pretrainer camera align: scale={scale:.6f}, t={t.tolist()}, det={float(np.linalg.det(r)):.3f}, cams={len(common)}")
        r_t = torch.as_tensor(r, dtype=xyz.dtype, device=xyz.device)
        t_t = torch.as_tensor(t, dtype=xyz.dtype, device=xyz.device)
        xyz = (xyz @ r_t.T) * scale + t_t
        scales = scales * scale
        rot_mat = self._quat_to_matrix(rots)
        rot_mat = r_t @ rot_mat
        rots = nerf_camera.quaternion.R_to_q(rot_mat)
        rots = torch_F.normalize(rots, dim=-1)
        return xyz, scales, rots

    def _align_points_by_cameras(self, xyz, normals, pts_path):
        gs_centers = self._load_gs_camera_centers(pts_path)
        sdf_centers = self._load_sdf_camera_centers()
        if not gs_centers or not sdf_centers:
            return None
        common = sorted(set(gs_centers.keys()) & set(sdf_centers.keys()))
        if len(common) < 3:
            self._log(f"GS pretrainer camera align: not enough common frames ({len(common)}).")
            return None
        src = np.stack([gs_centers[k] for k in common], axis=0)
        dst = np.stack([sdf_centers[k] for k in common], axis=0)
        result = self._compute_similarity_transform(src, dst, allow_reflection=self.align_allow_reflection)
        if result is None:
            self._log("GS pretrainer camera align: failed to compute similarity transform.")
            return None
        r, scale, t = result
        self._log(f"GS pretrainer camera align: scale={scale:.6f}, t={t.tolist()}, det={float(np.linalg.det(r)):.3f}, cams={len(common)}")
        r_t = torch.as_tensor(r, dtype=xyz.dtype, device=xyz.device)
        t_t = torch.as_tensor(t, dtype=xyz.dtype, device=xyz.device)
        xyz = (xyz @ r_t.T) * scale + t_t
        normals = torch_F.normalize(normals @ r_t.T, dim=-1)
        return xyz, normals

    def _get_scene_normalization(self):
        center = np.zeros(3, dtype=np.float32)
        scale = np.array(1.0, dtype=np.float32)
        if self.data_root:
            meta_path = os.path.join(self.data_root, "transforms.json")
            if os.path.isfile(meta_path):
                with open(meta_path, "r") as file:
                    meta = json.load(file)
                center = np.array(meta.get("sphere_center", [0.0, 0.0, 0.0]), dtype=np.float32)
                scale = np.array(meta.get("sphere_radius", 1.0), dtype=np.float32)
        if self.readjust is not None:
            center += np.array(getattr(self.readjust, "center", [0.0]), dtype=np.float32)
            scale *= np.array(getattr(self.readjust, "scale", 1.0), dtype=np.float32)
        return center, scale

    def _prepare_gaussian_cache(self, gaussian_path=None):
        xyz = self.gaussian.get_xyz.detach()
        scales = self.gaussian.get_scaling.detach()
        rots = self.gaussian.get_rotation.detach()
        opacity = self.gaussian.get_opacity.detach()

        if self.opacity_thresh > 0:
            mask = opacity.squeeze(-1) >= self.opacity_thresh
            xyz = xyz[mask]
            scales = scales[mask]
            rots = rots[mask]

        if self.normalize_mode == "camera_align":
            aligned = self._align_gaussians_by_cameras(xyz, scales, rots, gaussian_path)
            if aligned is not None:
                xyz, scales, rots = aligned
            else:
                self._log("GS pretrainer camera align failed, fallback to dataset normalization.")
                self.normalize_mode = "dataset"

        if self.normalize_mode == "dataset":
            xyz = self._apply_gl_to_cv_xyz(xyz)
            rots = self._apply_gl_to_cv_rot(rots)
            if self.normalize:
                center, scale = self._get_scene_normalization()
                center_t = torch.as_tensor(center, dtype=xyz.dtype, device=xyz.device)
                scale_t = torch.as_tensor(scale, dtype=xyz.dtype, device=xyz.device)
                scale_t = torch.where(scale_t == 0, torch.ones_like(scale_t), scale_t)
                xyz = (xyz - center_t) / scale_t
                scales = scales / scale_t

        if self.max_gaussians > 0 and xyz.shape[0] > self.max_gaussians:
            idx = torch.randperm(xyz.shape[0], device=xyz.device)[:self.max_gaussians]
            xyz = xyz.index_select(0, idx)
            scales = scales.index_select(0, idx)
            rots = rots.index_select(0, idx)

        self.gaussian_xyz = xyz
        self.gaussian_scales = scales
        self.gaussian_rots = rots
        if xyz.numel() > 0:
            self.gaussian_aabb_min = xyz.min(dim=0).values
            self.gaussian_aabb_max = xyz.max(dim=0).values
        else:
            self.gaussian_aabb_min = None
            self.gaussian_aabb_max = None

    def _prepare_pointcloud_cache(self, pts_path=None):
        xyz = self.pts_xyz
        normals = self.pts_normals
        if xyz is None or normals is None or xyz.numel() == 0:
            return

        if self.normalize_mode == "camera_align":
            aligned = self._align_points_by_cameras(xyz, normals, pts_path)
            if aligned is not None:
                xyz, normals = aligned
            else:
                self._log("GS pretrainer camera align failed, fallback to dataset normalization.")
                self.normalize_mode = "dataset"

        if self.normalize_mode == "dataset":
            xyz = self._apply_gl_to_cv_xyz(xyz)
            normals = self._apply_gl_to_cv_xyz(normals)
            if self.normalize:
                center, scale = self._get_scene_normalization()
                center_t = torch.as_tensor(center, dtype=xyz.dtype, device=xyz.device)
                scale_t = torch.as_tensor(scale, dtype=xyz.dtype, device=xyz.device)
                scale_t = torch.where(scale_t == 0, torch.ones_like(scale_t), scale_t)
                xyz = (xyz - center_t) / scale_t

        normals = torch_F.normalize(normals, dim=-1)

        if self.pts_max_points > 0 and xyz.shape[0] > self.pts_max_points:
            idx = torch.randperm(xyz.shape[0], device=xyz.device)[:self.pts_max_points]
            xyz = xyz.index_select(0, idx)
            normals = normals.index_select(0, idx)

        self.pts_xyz = xyz
        self.pts_normals = normals

    def _dump_points(self):
        if self._dumped_points or not self.dump_points_path:
            return
        if not is_master():
            return
        if self.gaussian_xyz is not None and self.gaussian_xyz.numel() > 0:
            xyz = self.gaussian_xyz
        elif self.pts_xyz is not None and self.pts_xyz.numel() > 0:
            xyz = self.pts_xyz
        else:
            return
        dump_path = self._resolve_dump_points_path()
        if not dump_path:
            return
        if self.dump_points_num > 0 and xyz.shape[0] > self.dump_points_num:
            idx = torch.randperm(xyz.shape[0], device=xyz.device)[:self.dump_points_num]
            xyz = xyz.index_select(0, idx)
        xyz_cpu = xyz.detach().float().cpu().numpy()
        dump_dir = os.path.dirname(dump_path)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        with open(dump_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {xyz_cpu.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for p in xyz_cpu:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        self._dumped_points = True
        self._log(f"GS pretrainer dumped normalized points to: {dump_path}")

    def _set_full_levels(self):
        sdf_model = self.model_module.neural_sdf
        cfg_encoding = getattr(sdf_model.cfg_sdf, "encoding", None)
        if cfg_encoding is None:
            return
        levels = int(getattr(cfg_encoding, "levels", 0))
        if levels <= 0:
            return
        sdf_model.active_levels = levels
        sdf_model.anneal_levels = levels
        grad_cfg = None
        if hasattr(self.full_cfg, "model") and hasattr(self.full_cfg.model, "object"):
            grad_cfg = getattr(self.full_cfg.model.object.sdf, "gradient", None)
        if grad_cfg is not None and getattr(grad_cfg, "mode", "") == "numerical":
            sdf_model.set_normal_epsilon()

    def _update_sdf_c2f(self, current_iter):
        sdf_model = self.model_module.neural_sdf
        cfg_encoding = getattr(sdf_model.cfg_sdf, "encoding", None)
        c2f_cfg = getattr(cfg_encoding, "coarse2fine", None) if cfg_encoding is not None else None
        if not c2f_cfg or not getattr(c2f_cfg, "enabled", False):
            return
        if not hasattr(sdf_model, "warm_up_end"):
            warm_up_end = 0
            if hasattr(self.full_cfg, "optim") and hasattr(self.full_cfg.optim, "sched"):
                warm_up_end = int(getattr(self.full_cfg.optim.sched, "warm_up_end", 0))
            sdf_model.warm_up_end = warm_up_end
        sdf_model.set_active_levels(current_iter)
        grad_cfg = None
        if hasattr(self.full_cfg, "model") and hasattr(self.full_cfg.model, "object"):
            grad_cfg = getattr(self.full_cfg.model.object.sdf, "gradient", None)
        if grad_cfg is not None and getattr(grad_cfg, "mode", "") == "numerical":
            sdf_model.set_normal_epsilon()

    def _quat_to_matrix(self, quat):
        quat = torch_F.normalize(quat, dim=-1)
        w, x, y, z = quat.unbind(-1)
        ww, xx, yy, zz = w * w, x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
        row1 = torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1)
        row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)
        return torch.stack([row0, row1, row2], dim=-2)

    def _sample_batch(self, batch_size):
        num_gaussians = self.gaussian_xyz.shape[0]
        idx = torch.randint(0, num_gaussians, (batch_size,), device=self.gaussian_xyz.device)
        centers = self.gaussian_xyz.index_select(0, idx)
        scales = self.gaussian_scales.index_select(0, idx)
        rots = self.gaussian_rots.index_select(0, idx)

        direction = torch.randn(batch_size, 3, device=centers.device)
        direction = torch_F.normalize(direction, dim=-1)
        if self.sdf_mode in ("ellipsoid", "random"):
            radius = 1.0 + torch.randn(batch_size, 1, device=centers.device) * self.radius_std
            radius = radius.clamp_min(1e-3)
        else:
            radius = torch.ones(batch_size, 1, device=centers.device)

        local = direction * scales * radius
        rot_mat = self._quat_to_matrix(rots)
        surface_points = torch.bmm(rot_mat, local.unsqueeze(-1)).squeeze(-1) + centers

        if self.sdf_mode in ("ellipsoid", "random"):
            surface_dist = (direction * scales).norm(dim=-1)
            sdf_gt_surface = (radius.squeeze(-1) - 1.0) * surface_dist
        else:
            sdf_gt_surface = torch.zeros(batch_size, device=centers.device)

        if self.sdf_mode == "random":
            if self.gaussian_aabb_min is None or self.gaussian_aabb_max is None:
                self.gaussian_aabb_min = self.gaussian_xyz.min(dim=0).values
                self.gaussian_aabb_max = self.gaussian_xyz.max(dim=0).values
            aabb_min = self.gaussian_aabb_min
            aabb_max = self.gaussian_aabb_max
            rand = torch.rand(batch_size, 3, device=centers.device)
            random_points = aabb_min + (aabb_max - aabb_min) * rand
            sdf_gt_random = sdf_gt_surface.new_full((batch_size,), float("nan"))
            points = torch.cat([surface_points, random_points], dim=0)
            sdf_gt = torch.cat([sdf_gt_surface, sdf_gt_random], dim=0)
        else:
            points = surface_points
            sdf_gt = sdf_gt_surface
        return {"points": points, "sdf_gt": sdf_gt}

    def _sample_batch_pts(self, batch_size):
        num_points = self.pts_xyz.shape[0]
        samples_per_point = self.pts_samples_per_point
        if samples_per_point <= 1:
            idx = torch.randint(0, num_points, (batch_size,), device=self.pts_xyz.device)
            base = self.pts_xyz.index_select(0, idx)
            normals = self.pts_normals.index_select(0, idx)
            noise = torch.randn(batch_size, 3, device=base.device) * self.pts_sample_std
            points = base + noise * normals
            diffs = points - base
            dists = diffs.norm(dim=-1)
            signs = torch.sign((diffs * normals).sum(dim=-1))
            sdf_gt = dists * signs
            return {"points": points, "sdf_gt": sdf_gt}

        num_base = (batch_size + samples_per_point - 1) // samples_per_point
        idx = torch.randint(0, num_points, (num_base,), device=self.pts_xyz.device)
        base = self.pts_xyz.index_select(0, idx)
        normals = self.pts_normals.index_select(0, idx)
        base = base[:, None, :].expand(num_base, samples_per_point, 3)
        normals = normals[:, None, :].expand(num_base, samples_per_point, 3)
        noise = torch.randn(num_base, samples_per_point, 3, device=base.device) * self.pts_sample_std
        points = base + noise * normals
        diffs = points - base
        dists = diffs.norm(dim=-1)
        signs = torch.sign((diffs * normals).sum(dim=-1))
        sdf_gt = dists * signs
        points = points.reshape(-1, 3)
        sdf_gt = sdf_gt.reshape(-1)
        if points.shape[0] > batch_size:
            points = points[:batch_size]
            sdf_gt = sdf_gt[:batch_size]
        return {"points": points, "sdf_gt": sdf_gt}

    @staticmethod
    def _huber_loss(values, delta):
        abs_val = values.abs()
        delta_t = abs_val.new_tensor(max(float(delta), 1e-6))
        quadratic = torch.where(abs_val < delta_t, abs_val, delta_t)
        linear = abs_val - quadratic
        return 0.5 * (quadratic ** 2) / delta_t + linear

    def _compute_loss(self, points, sdf_gt):
        points = points.detach().requires_grad_(self.eikonal_weight > 0)
        sdf_pred = self.model_module.neural_sdf.sdf(points).squeeze(-1)
        if self.clamping_distance > 0:
            sdf_pred = torch.clamp(sdf_pred, -self.clamping_distance, self.clamping_distance)
            sdf_gt = torch.clamp(sdf_gt, -self.clamping_distance, self.clamping_distance)
        valid_mask = torch.isfinite(sdf_gt)
        if valid_mask.any():
            sdf_pred_valid = sdf_pred[valid_mask]
            sdf_gt_valid = sdf_gt[valid_mask]
            if self.loss_type == "huber":
                sdf_loss = self._huber_loss(sdf_pred_valid - sdf_gt_valid, self.huber_delta).mean()
            elif self.loss_type == "l2":
                sdf_loss = torch_F.mse_loss(sdf_pred_valid, sdf_gt_valid)
            else:
                sdf_loss = (sdf_pred_valid - sdf_gt_valid).abs().mean()
        else:
            sdf_loss = sdf_pred.new_tensor(0.0)
        loss = sdf_loss
        metrics = {"sdf_loss": sdf_loss}
        if self.eikonal_weight > 0:
            gradients = torch.autograd.grad(sdf_pred.sum(), points, create_graph=True)[0]
            eikonal = ((gradients.norm(dim=-1) - 1.0) ** 2).mean()
            loss = loss + self.eikonal_weight * eikonal
            metrics["eikonal_loss"] = eikonal
        return loss, metrics

    def training_setup(self):
        optim_cfg = self._cfg_get(self.cfg, "optim", None)
        if optim_cfg is None:
            lr = float(self._cfg_get(self.cfg, "lr", 1e-3))
            weight_decay = float(self._cfg_get(self.cfg, "weight_decay", 0.0))
            self.optimizer = torch.optim.Adam(self.model_module.neural_sdf.parameters(),
                                              lr=lr, weight_decay=weight_decay)
            self.scheduler = None
            return

        optim_type = self._cfg_get(optim_cfg, "type", "Adam")
        optim_params = self._cfg_get(optim_cfg, "params", {}) or {}
        optimizer_cls = getattr(torch.optim, optim_type)
        self.optimizer = optimizer_cls(self.model_module.neural_sdf.parameters(), **optim_params)
        self.scheduler = self._build_scheduler(optim_cfg)

    def _build_scheduler(self, optim_cfg):
        sched_cfg = self._cfg_get(optim_cfg, "sched", None)
        if sched_cfg is None:
            return None
        sched_type = str(self._cfg_get(sched_cfg, "type", "constant")).lower()
        if sched_type == "constant":
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
        if sched_type == "linear_warmup":
            warmup = int(self._cfg_get(sched_cfg, "warmup", 100))
            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda x: x * 1.0 / warmup if x < warmup else 1.0
            )
        if sched_type == "two_steps_with_warmup":
            warm_up_end = int(self._cfg_get(sched_cfg, "warm_up_end", 0))
            two_steps = self._cfg_get(sched_cfg, "two_steps", [0, 0])
            gamma = float(self._cfg_get(sched_cfg, "gamma", 10.0))

            def sch(x):
                if x < warm_up_end:
                    return x / max(warm_up_end, 1)
                if x > two_steps[1]:
                    return 1.0 / gamma ** 2
                if x > two_steps[0]:
                    return 1.0 / gamma
                return 1.0

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda x: sch(x))
        if sched_type == "cos_with_warmup":
            alpha = float(self._cfg_get(sched_cfg, "alpha", 0.0))
            max_iter = int(self._cfg_get(sched_cfg, "max_iter", self.num_iters))
            warm_up_end = int(self._cfg_get(sched_cfg, "warm_up_end", 0))

            def sch(x):
                if x < warm_up_end:
                    return x / max(warm_up_end, 1)
                progress = (x - warm_up_end) / max(max_iter - warm_up_end, 1)
                learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
                return learning_factor

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda x: sch(x))
        return None

    def _broadcast_sdf_params(self):
        if not is_dist():
            return
        for param in self.model_module.neural_sdf.parameters():
            dist.broadcast(param.data, src=0)
        for buf in self.model_module.neural_sdf.buffers():
            dist.broadcast(buf.data, src=0)

    def _log(self, message):
        if is_master():
            master_only_print(message)

    def _maybe_train(self):
        if not self.enabled:
            return
        if self.num_iters <= 0:
            self._log("GS pretrainer skipped: num_iters <= 0.")
            return
        sample_fn = None
        if self.input_pts:
            pts_path = self._resolve_input_pts()
            if not pts_path or not os.path.isfile(pts_path):
                self._log(f"GS pretrainer skipped: input_pts not found: {pts_path}")
                return
            loaded = self._load_pts_file(pts_path)
            if loaded is None:
                self._log("GS pretrainer skipped: failed to load input_pts.")
                return
            points, normals = loaded
            if points.size == 0:
                self._log("GS pretrainer skipped: empty input_pts.")
                return
            self.pts_xyz = torch.as_tensor(points, device=self.device)
            self.pts_normals = torch.as_tensor(normals, device=self.device)
            self._prepare_pointcloud_cache(pts_path=pts_path)
            self._dump_points()
            if self.pts_xyz is None or self.pts_xyz.numel() == 0:
                self._log("GS pretrainer skipped: empty point cloud after preprocessing.")
                return
            sample_fn = self._sample_batch_pts
            self._log(f"GS pretrainer using input_pts: {pts_path}")
        else:
            gaussian_path = self._resolve_gaussian_path()
            if not gaussian_path or not os.path.isfile(gaussian_path):
                self._log(f"GS pretrainer skipped: gaussian_path not found: {gaussian_path}")
                return
            self.gaussian = ECGSGaussianModel(3)
            self.gaussian.load_ply(gaussian_path)
            self._prepare_gaussian_cache(gaussian_path=gaussian_path)
            self._dump_points()
            if self.gaussian_xyz is None or self.gaussian_xyz.numel() == 0:
                self._log("GS pretrainer skipped: empty gaussian cache.")
                return
            sample_fn = self._sample_batch

        self.training_setup()
        self.model_module.neural_sdf.train()
        tbar = tqdm(range(self.num_iters), desc="GS SDF Pretrain", disable=not is_master())
        if self.force_full_levels:
            self._set_full_levels()
        for _ in range(self.num_iters):
            if not self.force_full_levels:
                self._update_sdf_c2f(self.global_step)
            batch = sample_fn(self.batch_size)
            loss, metrics = self._compute_loss(batch["points"], batch["sdf_gt"])
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            if self.log_interval and self.global_step % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                log_data = {
                    "gs_pretrain/loss": loss.item(),
                    "gs_pretrain/lr": lr,
                }
                if "eikonal_loss" in metrics:
                    log_data["gs_pretrain/eikonal"] = metrics["eikonal_loss"].item()
                if self.log_wandb and wandb.run is not None:
                    wandb.log(log_data, step=self.wandb_step_offset + self.global_step)
                else:
                    self._log(f"GS pretrain step {self.global_step} loss {loss.item():.6f} lr {lr:.3e}")
            self.global_step += 1
            tbar.update(1)
            #print all loss:
            tbar.set_postfix({k: f"{v.item():.6f}" for k, v in metrics.items()})

    def run(self):
        if not self.enabled:
            return
        if not is_dist():
            self._maybe_train()
            return
        if is_master():
            self._maybe_train()
        barrier()
        self._broadcast_sdf_params()
        barrier()
