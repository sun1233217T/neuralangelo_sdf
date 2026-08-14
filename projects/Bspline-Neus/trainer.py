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

import torch
import torch.nn.functional as torch_F
import wandb

from imaginaire.utils.distributed import master_only
from imaginaire.utils.visualization import wandb_image
from projects.neuralangelo.trainer import Trainer as NeuralangeloTrainer
from projects.neuralangelo.utils.misc import eikonal_loss, curvature_loss


class Trainer(NeuralangeloTrainer):

    def __init__(self, cfg, is_inference=True, seed=0):
        super().__init__(cfg, is_inference=is_inference, seed=seed)
        self.metrics = dict()
        self.warm_up_end = cfg.optim.sched.warm_up_end
        # Fired by the checkpointer after the model weights are loaded (see
        # imaginaire.trainers.base.Checkpointer.load).
        self.checkpointer.post_model_load_hook = self._on_checkpoint_model_loaded
        # Optional CUDA-graph training step (see graph_runner.py).  Only
        # available when AMP, EMA and gradient accumulation are all off.
        self._graph_runner = None
        cfg_graph = getattr(cfg.trainer, "cudagraph", None)
        if cfg_graph is not None and bool(cfg_graph.get("enabled", False)):
            from imaginaire.utils.distributed import get_world_size as _gws
            unsupported = []
            if getattr(cfg.trainer.amp_config, "enabled", False):
                unsupported.append("amp")
            if getattr(cfg.trainer.ema_config, "enabled", False):
                unsupported.append("ema")
            if int(getattr(cfg.trainer, "grad_accum_iter", 1)) != 1:
                unsupported.append("grad_accum")
            if _gws() > 1:
                unsupported.append("multi-gpu")
            if unsupported:
                print(f"[graph] cudagraph.enabled=True but unsupported with: "
                      f"{', '.join(unsupported)}; using eager training")
            else:
                from .graph_runner import GraphTrainStep
                self._graph_runner = GraphTrainStep(self, cfg_graph)
                print("[graph] CUDA-graph training step enabled "
                      f"(capture_iter={self._graph_runner.capture_iter})")

    def _on_checkpoint_model_loaded(self):
        """Rebuild the optimizer when a checkpoint restored finer levels.

        Structure restore creates brand-new ``nn.Parameter`` tensors for the
        refined levels, so the optimizer built at trainer construction time
        still references the old (dead) tensors and its group layout would
        mismatch the checkpoint's optimizer state.  Rebuilding here — before
        the checkpointer loads the optimizer state — fixes both.
        """
        if getattr(self.model_module, "_hier_structure_restored", False):
            self.model_module._hier_structure_restored = False
            self._rebuild_optimizer()
        # The restored level parameters do not carry the SH-rest gradient
        # scaling hooks (they are new tensors); re-register on all levels.
        rgb = getattr(self.model_module, "neural_rgb", None)
        if rgb is not None and getattr(rgb, "hierarchical_enabled", False):
            rgb._register_sh_hooks()

    @torch.no_grad()
    def test(self, data_loader, output_dir=None, inference_args=None, mode="test", show_pbar=False):
        """Run evaluation and release reserved CUDA memory afterwards.

        Full-image inference (especially at 800x800) can transiently allocate
        many GBs of activation memory.  PyTorch's caching allocator normally
        keeps that reserved for reuse, which makes Windows Task Manager report
        very high GPU memory usage.  We call :func:`torch.cuda.empty_cache`
        after validation so the OS sees the memory as free again.
        """
        data_all = super().test(data_loader, output_dir=output_dir, inference_args=inference_args,
                                mode=mode, show_pbar=show_pbar)
        if mode == "val":
            torch.cuda.empty_cache()
        return data_all

    def _init_loss(self, cfg):
        self.criteria["render"] = torch.nn.L1Loss()

    def _compute_loss(self, data, mode=None):
        if mode == "train":
            # Compute loss only on randomly sampled rays.
            self.losses["render"] = self.criteria["render"](data["rgb"], data["image_sampled"]) * 3
            self.metrics["psnr"] = -10 * torch_F.mse_loss(data["rgb"], data["image_sampled"]).log10()
            if "hier_points" in data:
                # Loss-guided refinement statistics (eager path; the CUDA-graph
                # runner performs the same scatter inside the captured step).
                self.model_module.neural_sdf.accumulate_loss(
                    data["hier_points"], data["hier_weights"],
                    data["rgb"], data["image_sampled"])
            if "eikonal" in self.weights.keys():
                self.losses["eikonal"] = eikonal_loss(data["gradients"], outside=data["outside"])
            if "curvature" in self.weights:
                self.losses["curvature"] = curvature_loss(data["hessians"], outside=data["outside"])
        else:
            # Compute loss on the entire image.
            self.losses["render"] = self.criteria["render"](data["rgb_map"], data["image"])
            self.metrics["psnr"] = -10 * torch_F.mse_loss(data["rgb_map"], data["image"]).log10()

    def get_curvature_weight(self, current_iteration, init_weight):
        # B-spline fields do not use hash-grid coarse-to-fine level growth.
        # Keep the warm-up ramp and then decay based on training progress.
        if "curvature" in self.weights:
            if current_iteration <= self.warm_up_end:
                self.weights["curvature"] = current_iteration / self.warm_up_end * init_weight
            else:
                progress = min(current_iteration / self.cfg.max_iter, 1.0)
                self.weights["curvature"] = init_weight * (1.0 - progress * 0.5)

    def _start_of_iteration(self, data, current_iteration):
        model = self.model_module
        if torch.is_tensor(getattr(model, "progress", None)):
            # CUDA-graph mode: update the shared 0-dim tensor in place so the
            # captured graph reads the current progress.
            model.progress.fill_(current_iteration / self.cfg.max_iter)
            self.progress = model.progress
        else:
            self.progress = model.progress = current_iteration / self.cfg.max_iter
        # Skip hash-grid coarse2fine handling (not applicable to B-spline fields).
        # Only update numerical-gradient curvature weight if requested.
        self.get_curvature_weight(current_iteration, self.cfg.trainer.loss_weight.curvature)
        # Hierarchical B-spline refinement: grow the hierarchy on schedule and
        # rebuild the optimizer around the new parameter tensors.  The color
        # hierarchy refines with the same SDF-band cells so both stay aligned.
        # Guard on grad mode: the shared `start_of_iteration` is also called
        # from `test()` (validation, under @torch.no_grad) with the same
        # iteration counter, which would fire refinement again whenever a
        # validation lands on a scheduled iteration.  (Model.train()/eval()
        # state is not reliable here: validation runs between the counter
        # increment and the next start_of_iteration, so the model is still in
        # eval mode when the next training iteration's hook executes.)
        sdf_info = rgb_info = None
        if torch.is_grad_enabled():
            sdf_info = model.neural_sdf.maybe_refine(current_iteration)
            rgb_info = model.neural_rgb.maybe_refine(current_iteration, sdf_wrapper=model.neural_sdf)
        refined = (sdf_info is not None and sdf_info.get("refined")) or \
                  (rgb_info is not None and rgb_info.get("refined"))
        if refined:
            print(f"[Bspline-Neus] refinement at iter {current_iteration}: "
                  f"sdf={sdf_info}, rgb={rgb_info}")
            self._rebuild_optimizer()
            if self._graph_runner is not None:
                # New parameter tensors: the captured graph is stale.
                self._graph_runner.invalidate()
        # Forward to the nerf base trainer for AMP/iteration bookkeeping.
        return super(NeuralangeloTrainer, self)._start_of_iteration(data, current_iteration)

    def _rebuild_optimizer(self):
        """Recreate optimizer/scheduler after the parameter set changed.

        Refinement replaces the value tensors of the hierarchical levels, so
        the old optimizer would keep references to dead parameters.  Learning
        rate scheduling continues from the current iteration.
        """
        self.optim = self.setup_optimizer(self.cfg, self.model_module)
        self.sched = self.setup_scheduler(self.cfg, self.optim)
        if self.cfg.optim.sched.iteration_mode:
            self.sched.last_epoch = self.current_iteration
        else:
            self.sched.last_epoch = self.current_epoch
        # The checkpointer keeps references to optim/sched for state saving.
        self.checkpointer.optim = self.optim
        self.checkpointer.sched = self.sched

    @master_only
    def log_wandb_scalars(self, data, mode=None):
        super(NeuralangeloTrainer, self).log_wandb_scalars(data, mode=mode)
        scalars = {
            f"{mode}/PSNR": self.metrics["psnr"].detach(),
            f"{mode}/inv_std": self.model_module.neural_sdf.inv_std().item(),
        }
        if "curvature" in self.weights:
            scalars[f"{mode}/curvature_weight"] = self.weights["curvature"]
        if "eikonal" in self.weights:
            scalars[f"{mode}/eikonal_weight"] = self.weights["eikonal"]
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
            })
        wandb.log(images, step=self.current_iteration)

    def train_step(self, data, last_iter_in_epoch=False):
        if self._graph_runner is not None and self._graph_runner.train_step(data):
            return
        super().train_step(data, last_iter_in_epoch=last_iter_in_epoch)

    def train(self, cfg, data_loader, single_gpu=False, profile=False, show_pbar=False):
        self.progress = self.model_module.progress = self.current_iteration / self.cfg.max_iter
        super().train(cfg, data_loader, single_gpu, profile, show_pbar)
