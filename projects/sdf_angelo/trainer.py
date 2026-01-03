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
import math
import os

import numpy as np
import torch
import torch.nn.functional as torch_F
import wandb

from imaginaire.utils.distributed import master_only
from imaginaire.utils.visualization import wandb_image
from projects.nerf.trainers.base import BaseTrainer
from projects.sdf_angelo.utils.misc import get_scheduler, eikonal_loss, curvature_loss, sdf_shift_loss
from projects.sdf_angelo.scripts.read_write_model import read_points3D_binary, read_points3D_text

from mtools import debug


class Trainer(BaseTrainer):

    def __init__(self, cfg, is_inference=True, seed=0):
        super().__init__(cfg, is_inference=is_inference, seed=seed)
        self.metrics = dict()
        self.warm_up_end = cfg.optim.sched.warm_up_end
        self.cfg_gradient = cfg.model.object.sdf.gradient
        if cfg.model.object.sdf.encoding.type == "hashgrid" and cfg.model.object.sdf.encoding.coarse2fine.enabled:
            self.c2f_step = cfg.model.object.sdf.encoding.coarse2fine.step
            self.model.module.neural_sdf.warm_up_end = self.warm_up_end
        self._init_pointcloud_sdf(cfg)

    def _init_loss(self, cfg):
        self.criteria["render"] = torch.nn.L1Loss()

    def setup_scheduler(self, cfg, optim):
        return get_scheduler(cfg.optim, optim)

    def _init_pointcloud_sdf(self, cfg):
        pc_cfg = getattr(cfg.trainer, "pointcloud_sdf", None)
        self.pc_sdf_enabled = bool(getattr(pc_cfg, "enabled", False)) if pc_cfg else False
        self.pc_sdf_base_weight = float(getattr(cfg.trainer.loss_weight, "pc_sdf", 0.0))
        self.pc_sdf_weight = 0.0
        self.pc_sdf_points = None
        self.pc_sdf_num_samples = 0
        self.pc_sdf_warmup_iters = 0
        self.pc_sdf_end_iter = 0
        self.pc_sdf_loss_type = "huber"
        self.pc_sdf_huber_delta = 0.1
        self.pc_sdf_max_error = None
        if not self.pc_sdf_enabled or self.pc_sdf_base_weight <= 0.0:
            if "pc_sdf" in self.weights:
                self.weights["pc_sdf"] = 0.0
            return

        self.pc_sdf_num_samples = int(getattr(pc_cfg, "num_samples", 10000))
        self.pc_sdf_warmup_iters = int(getattr(pc_cfg, "warmup_iters", 0))
        self.pc_sdf_end_iter = int(getattr(pc_cfg, "end_iter", 0))
        self.pc_sdf_loss_type = str(getattr(pc_cfg, "loss_type", "huber")).lower()
        self.pc_sdf_huber_delta = float(getattr(pc_cfg, "huber_delta", 0.1))
        self.pc_sdf_max_error = getattr(pc_cfg, "max_error", None)
        self.pc_sdf_points = self._load_pointcloud_sdf_points(cfg, pc_cfg)
        if self.pc_sdf_points is None or self.pc_sdf_points.numel() == 0:
            print("Point cloud SDF warmup disabled: no usable points loaded.")
            self.pc_sdf_enabled = False
            if "pc_sdf" in self.weights:
                self.weights["pc_sdf"] = 0.0

    def _load_pointcloud_sdf_points(self, cfg, pc_cfg):
        data_root = cfg.data.root
        pc_rel_path = getattr(pc_cfg, "path", "sparse/points3D.bin")
        pc_path = os.path.join(data_root, pc_rel_path)
        points3d = None
        if os.path.isfile(pc_path):
            points3d = read_points3D_binary(pc_path)
        else:
            alt_path = os.path.join(data_root, "sparse", "points3D.txt")
            if os.path.isfile(alt_path):
                points3d = read_points3D_text(alt_path)
        if not points3d:
            print(f"Point cloud file not found or empty: {pc_path}")
            return None

        xyzs = np.stack([pt.xyz for pt in points3d.values()], axis=0).astype(np.float32)
        errors = np.array([float(pt.error) for pt in points3d.values()], dtype=np.float32)
        if self.pc_sdf_max_error is not None:
            max_error = float(self.pc_sdf_max_error)
            if max_error > 0:
                mask = errors <= max_error
                xyzs = xyzs[mask]
        if xyzs.size == 0:
            print("Point cloud SDF warmup disabled: all points filtered by error.")
            return None

        center, radius = self._get_scene_normalization(cfg)
        if radius <= 0:
            print("Point cloud SDF warmup disabled: invalid sphere radius.")
            return None
        xyzs = (xyzs - center[None, :]) / radius
        return torch.from_numpy(xyzs).float()

    def _get_scene_normalization(self, cfg):
        meta_path = os.path.join(cfg.data.root, "transforms.json")
        center = np.zeros(3, dtype=np.float32)
        radius = 1.0
        if os.path.isfile(meta_path):
            with open(meta_path, "r") as file:
                meta = json.load(file)
            center = np.array(meta.get("sphere_center", [0.0, 0.0, 0.0]), dtype=np.float32)
            radius = float(meta.get("sphere_radius", 1.0))
        readjust = getattr(cfg.data, "readjust", None)
        if readjust is not None:
            center += np.array(getattr(readjust, "center", [0.0, 0.0, 0.0]), dtype=np.float32)
            radius *= float(getattr(readjust, "scale", 1.0))
        return center, radius

    def _sample_pointcloud_sdf_points(self, device):
        if self.pc_sdf_points is None or self.pc_sdf_points.numel() == 0:
            return None
        num_points = self.pc_sdf_points.shape[0]
        if num_points == 0 or self.pc_sdf_num_samples <= 0:
            return None
        idx = torch.randint(0, num_points, (self.pc_sdf_num_samples,), device=self.pc_sdf_points.device)
        pts = self.pc_sdf_points.index_select(0, idx)
        return pts.to(device, non_blocking=True)

    @staticmethod
    def _huber_loss(values, delta):
        abs_val = values.abs()
        delta_t = abs_val.new_tensor(max(float(delta), 1e-6))
        quadratic = torch.where(abs_val < delta_t, abs_val, delta_t)
        linear = abs_val - quadratic
        return 0.5 * (quadratic ** 2) / delta_t + linear

    def _get_pc_sdf_weight(self, current_iteration):
        if not self.pc_sdf_enabled:
            return 0.0
        if self.pc_sdf_end_iter > 0 and current_iteration >= self.pc_sdf_end_iter:
            return 0.0
        if self.pc_sdf_warmup_iters > 0:
            warmup_ratio = min(float(current_iteration) / float(self.pc_sdf_warmup_iters), 1.0)
        else:
            warmup_ratio = 1.0
        return self.pc_sdf_base_weight * warmup_ratio

    def _compute_loss(self, data, mode=None):
        if mode == "train":
            # Compute loss only on randomly sampled rays.
            self.losses["render"] = self.criteria["render"](data["rgb"], data["image_sampled"]) * 3  # FIXME:sumRGB?!
            self.metrics["psnr"] = -10 * torch_F.mse_loss(data["rgb"], data["image_sampled"]).log10()
            if "eikonal" in self.weights.keys():
                self.losses["eikonal"] = eikonal_loss(data["gradients"], outside=data["outside"])
            if "curvature" in self.weights:
                self.losses["curvature"] = curvature_loss(data["hessians"], outside=data["outside"])
            if "sdf_shift" in self.weights and "sdf_offsets" in data:
                # self.losses["sdf_shift"] = sdf_shift_loss(data["sdf_offsets"],data["rgb_offsets"],data["image_sampled"])
                self.losses["sdf_shift"] = sdf_shift_loss(data["sdf_offsets"], data["rgb_offsets"], data["image_sampled"], data["rgb"])
            if "sdf_render" in self.weights and "surface_rgb" in data:
                self.losses["sdf_render"] = self.criteria["render"](data["surface_rgb"], data["image_sampled"])
            self.losses.pop("pc_sdf", None)
            self.pc_sdf_weight = self._get_pc_sdf_weight(self.current_iteration)
            if "pc_sdf" in self.weights:
                self.weights["pc_sdf"] = self.pc_sdf_weight
            if self.pc_sdf_weight > 0:
                pts = self._sample_pointcloud_sdf_points(data["rgb"].device)
                if pts is not None:
                    sdf = self.model_module.neural_sdf.sdf(pts).squeeze(-1)
                    if self.pc_sdf_loss_type == "huber":
                        self.losses["pc_sdf"] = self._huber_loss(sdf, self.pc_sdf_huber_delta).mean()
                    else:
                        self.losses["pc_sdf"] = sdf.abs().mean()

        else:
            # Compute loss on the entire image.
            self.losses["render"] = self.criteria["render"](data["rgb_map"], data["image"])
            self.metrics["psnr"] = -10 * torch_F.mse_loss(data["rgb_map"], data["image"]).log10()
            self.pc_sdf_weight = 0.0
            if "pc_sdf" in self.weights:
                self.weights["pc_sdf"] = 0.0
            self.losses.pop("pc_sdf", None)
        # for key in self.losses:
        #     print(f"{mode} loss {key}: {self.losses[key].item()} weight: {self.weights.get(key, 'N/A')}")

    def get_curvature_weight(self, current_iteration, init_weight):
        if "curvature" in self.weights:
            if current_iteration <= self.warm_up_end:
                self.weights["curvature"] = current_iteration / self.warm_up_end * init_weight
            else:
                model = self.model_module
                decay_factor = model.neural_sdf.growth_rate ** (model.neural_sdf.anneal_levels - 1)
                self.weights["curvature"] = init_weight / decay_factor

    def _start_of_iteration(self, data, current_iteration):
        model = self.model_module
        self.progress = model.progress = current_iteration / self.cfg.max_iter
        model.current_iteration = current_iteration
        if self.cfg.model.object.sdf.encoding.coarse2fine.enabled:
            model.neural_sdf.set_active_levels(current_iteration)
            if self.cfg_gradient.mode == "numerical":
                model.neural_sdf.set_normal_epsilon()
                self.get_curvature_weight(current_iteration, self.cfg.trainer.loss_weight.curvature)
        elif self.cfg_gradient.mode == "numerical":
            model.neural_sdf.set_normal_epsilon()

        return super()._start_of_iteration(data, current_iteration)

    @master_only
    def log_wandb_scalars(self, data, mode=None):
        super().log_wandb_scalars(data, mode=mode)
        scalars = {
            f"{mode}/PSNR": self.metrics["psnr"].detach(),
            f"{mode}/s-var": self.model_module.s_var.item(),
        }
        if "curvature" in self.weights:
            scalars[f"{mode}/curvature_weight"] = self.weights["curvature"]
        if "eikonal" in self.weights:
            scalars[f"{mode}/eikonal_weight"] = self.weights["eikonal"]
        if mode == "train" and self.cfg_gradient.mode == "numerical":
            scalars[f"{mode}/epsilon"] = self.model.module.neural_sdf.normal_eps
        if self.cfg.model.object.sdf.encoding.coarse2fine.enabled:
            scalars[f"{mode}/active_levels"] = self.model.module.neural_sdf.active_levels
        if mode == "train" and "pc_sdf" in self.weights:
            scalars[f"{mode}/pc_sdf_weight"] = self.pc_sdf_weight
        wandb.log(scalars, step=self.current_iteration)

    @master_only
    def log_wandb_images(self, data, mode=None, max_samples=None):
        images = {"iteration": self.current_iteration, "epoch": self.current_epoch}
        if mode == "val":
            images_error = (data["rgb_map"] - data["image"]).abs()
            images.update({
                f"{mode}/vis/rgb_target": wandb_image(data["image"]),
                f"{mode}/vis/rgb_render": wandb_image(data["rgb_map"]),
                f"{mode}/vis/rgb_error": wandb_image(images_error),
                f"{mode}/vis/normal": wandb_image(data["normal_map"], from_range=(-1, 1)),
                f"{mode}/vis/inv_depth": wandb_image(1 / (data["depth_map"] + 1e-8) * self.cfg.trainer.depth_vis_scale),
                f"{mode}/vis/opacity": wandb_image(data["opacity_map"]),
                f"{mode}/vis_sdf/rgb_render": wandb_image(data["sdf_rgb_map"]),
                f"{mode}/vis_sdf/normal": wandb_image(data["sdf_normal_map"], from_range=(-1, 1)),
                f"{mode}/vis_sdf/inv_depth": wandb_image(1 / (data["sdf_depth_map"] + 1e-8) * self.cfg.trainer.depth_vis_scale),
            })
        wandb.log(images, step=self.current_iteration)

    def train(self, cfg, data_loader, single_gpu=False, profile=False, show_pbar=False):
        self._apply_epoch_config(cfg, data_loader)
        self.progress = self.model_module.progress = self.current_iteration / self.cfg.max_iter
        super().train(cfg, data_loader, single_gpu, profile, show_pbar)

    def _apply_epoch_config(self, cfg, data_loader):
        if not getattr(cfg.trainer, "epoch_based", False):
            return
        batch_size = cfg.data.train.batch_size
        samples_per_epoch = getattr(cfg.trainer, "samples_per_epoch", None)
        iters_per_epoch = getattr(cfg.trainer, "iters_per_epoch", None)
        if iters_per_epoch is None:
            if samples_per_epoch is not None:
                iters_per_epoch = int(math.ceil(samples_per_epoch / float(batch_size)))
            else:
                iters_per_epoch = len(data_loader)
        cfg.trainer.iters_per_epoch = iters_per_epoch

        def to_iter(epoch_val):
            if epoch_val is None:
                return None
            return int(epoch_val * iters_per_epoch)

        if getattr(cfg, "max_epoch", None) is not None:
            cfg.max_iter = to_iter(cfg.max_epoch)
        if getattr(cfg, "wandb_scalar_epoch", None) is not None:
            cfg.wandb_scalar_iter = to_iter(cfg.wandb_scalar_epoch)
        if getattr(cfg, "wandb_image_epoch", None) is not None:
            cfg.wandb_image_iter = to_iter(cfg.wandb_image_epoch)
        if getattr(cfg, "validation_epoch", None) is not None:
            cfg.validation_iter = to_iter(cfg.validation_epoch)
        if getattr(cfg.checkpoint, "save_epoch", None) is not None:
            cfg.checkpoint.save_iter = to_iter(cfg.checkpoint.save_epoch)

        sched = cfg.optim.sched
        if getattr(sched, "warm_up_epoch", None) is not None:
            sched.warm_up_end = to_iter(sched.warm_up_epoch)
        if getattr(sched, "two_steps_epoch", None) is not None:
            sched.two_steps = [to_iter(v) for v in sched.two_steps_epoch]
        if getattr(sched, "max_epoch", None) is not None:
            sched.max_iter = to_iter(sched.max_epoch)

        c2f = cfg.model.object.sdf.encoding.coarse2fine
        if getattr(c2f, "step_epoch", None) is not None:
            c2f.step = to_iter(c2f.step_epoch)

        self.warm_up_end = cfg.optim.sched.warm_up_end
        if hasattr(self.model.module, "neural_sdf"):
            self.model.module.neural_sdf.warm_up_end = self.warm_up_end
