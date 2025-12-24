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

import math
import torch
import torch.nn.functional as torch_F
import wandb

from imaginaire.utils.distributed import master_only
from imaginaire.utils.visualization import wandb_image
from projects.nerf.trainers.base import BaseTrainer
from projects.sdf_angelo.utils.misc import get_scheduler, eikonal_loss, curvature_loss, sdf_shift_loss

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

    def _init_loss(self, cfg):
        self.criteria["render"] = torch.nn.L1Loss()

    def setup_scheduler(self, cfg, optim):
        return get_scheduler(cfg.optim, optim)

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

        else:
            # Compute loss on the entire image.
            self.losses["render"] = self.criteria["render"](data["rgb_map"], data["image"])
            self.metrics["psnr"] = -10 * torch_F.mse_loss(data["rgb_map"], data["image"]).log10()
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
