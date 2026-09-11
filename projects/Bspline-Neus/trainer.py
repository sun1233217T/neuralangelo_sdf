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

    def _eikonal_loss(self, gradients, outside, sdfs=None):
        """Eikonal loss with optional SDF-band weighting.

        When ``trainer.loss_weight.eikonal_band > 0`` and per-sample SDF
        values are available, the eikonal constraint is only applied to
        samples within ``|sdf| < band`` of the surface, normalized by the
        number of in-band samples.  This reduces regularizer pressure on
        empty-space samples that carry no geometric signal.
        """
        band = float(getattr(self.cfg.trainer.loss_weight, "eikonal_band", 0.0))
        if band > 0 and sdfs is not None:
            weight = (sdfs.detach().abs() < band).float()
            gradient_error = (gradients.norm(dim=-1) - 1.0) ** 2
            gradient_error = gradient_error.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            if outside is not None:
                weight = weight * (~outside).float()
            return (gradient_error * weight).sum() / (weight.sum() + 1e-8)
        return eikonal_loss(gradients, outside=outside)

    def _init_loss(self, cfg):
        self.criteria["render"] = torch.nn.L1Loss()

    def _mean_curvature_loss(self, gradients, hessians, sdfs, outside=None):
        """Mean curvature regularization via normalized Laplacian.

        H ≈ ∇²f / |∇f|  (normalized Laplacian, approximates mean curvature).

        Only penalizes curvature above ``threshold`` (bubble-scale) and
        only within ``|sdf| < band`` of the surface.  Works with data
        already in the render output — no extra SDF evaluation needed.
        """
        mc_params = getattr(self.cfg.trainer, "mean_curvature_params", {})
        threshold = float(getattr(mc_params, "threshold", 20.0))
        band = float(getattr(mc_params, "band", 0.05))

        lap = hessians.sum(dim=-1)  # (B,R,N) — Laplacian from diagonal
        gn = gradients.norm(dim=-1)  # (B,R,N) — |∇f|
        H = lap / gn.clamp_min(1e-6)
        H = H.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        # Threshold: only penalize curvature above threshold (bubbles).
        mc_err = (H.abs() - threshold).clamp_min(0.0) ** 2
        # Band: only apply near surface.
        w = (sdfs.abs() < band).to(mc_err.dtype)
        if outside is not None:
            w = w * (~outside).to(mc_err.dtype)
        return (mc_err * w).sum() / (w.sum() + 1e-8)

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
                    data["rgb"], data["image_sampled"],
                    sdfs=data.get("hier_sdfs"))
            if "eikonal" in self.weights.keys():
                self.losses["eikonal"] = self._eikonal_loss(
                    data["gradients"], data["outside"], data.get("sdfs"))
            if "curvature" in self.weights:
                self.losses["curvature"] = curvature_loss(data["hessians"], outside=data["outside"])
            if "mean_curvature" in self.weights:
                self.losses["mean_curvature"] = self._mean_curvature_loss(
                    data["gradients"], data["hessians"], data["sdfs"], outside=data["outside"])
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
            rgb_info = model.neural_rgb.maybe_refine(
                current_iteration,
                sdf_wrapper=model.neural_sdf,
                sdf_refine_info=sdf_info,
            )
        refined = (sdf_info is not None and sdf_info.get("refined")) or \
                  (rgb_info is not None and rgb_info.get("refined"))
        if refined:
            print(f"[Bspline-Neus] refinement at iter {current_iteration}: "
                  f"sdf={sdf_info}, rgb={rgb_info}")
            self._rebuild_optimizer()
            if self._graph_runner is not None:
                # New parameter tensors: the captured graph is stale.
                self._graph_runner.invalidate()

        # Re-banding: periodically re-evaluate level regions.
        # Only fires when no refinement happened at this iteration.
        if not refined and torch.is_grad_enabled():
            # Finest-level re-banding (legacy, single-level).
            reband_sdf = model.neural_sdf.maybe_reband(current_iteration)
            reband_rgb = model.neural_rgb.maybe_reband(
                current_iteration, sdf_reband_info=reband_sdf,
            )
            # Full-hierarchy re-banding (all levels, 3DGS-style split/prune).
            reband_all_sdf = model.neural_sdf.maybe_reband_all(current_iteration)
            reband_all_rgb = model.neural_rgb.maybe_reband_all(
                current_iteration, sdf_reband_info=reband_all_sdf,
            )
            rebanded = any(
                r is not None and r.get("changed")
                for r in (reband_sdf, reband_rgb, reband_all_sdf, reband_all_rgb)
            )
            if rebanded:
                self._rebuild_optimizer()
                if self._graph_runner is not None:
                    self._graph_runner.invalidate()

        # Forward to the nerf base trainer for AMP/iteration bookkeeping.
        return super(NeuralangeloTrainer, self)._start_of_iteration(data, current_iteration)

    def _rebuild_optimizer(self):
        """Recreate optimizer/scheduler after the parameter set changed.

        Refinement replaces the value tensors of the refined and new levels,
        so the old optimizer would keep references to dead parameters.  This
        rebuild preserves Adam state (exp_avg, exp_avg_sq, step) for
        parameters that survive the refinement unchanged (same tensor object),
        which includes all coarser levels, the color MLP, inv_std, and the
        background NeRF.  Only genuinely new tensors start with fresh state.
        """
        # Save Adam state for surviving parameters (keyed by tensor identity).
        old_state = {}
        if hasattr(self, "optim") and self.optim is not None:
            for group in self.optim.param_groups:
                for p in group["params"]:
                    if p in self.optim.state:
                        state = self.optim.state[p]
                        old_state[id(p)] = {
                            k: v.clone() if torch.is_tensor(v) else v
                            for k, v in state.items()
                        }

        self.optim = self.setup_optimizer(self.cfg, self.model_module)
        self.sched = self.setup_scheduler(self.cfg, self.optim)

        # Restore Adam state for parameters that are the same tensor objects.
        if old_state:
            n_restored = 0
            for group in self.optim.param_groups:
                for p in group["params"]:
                    if id(p) in old_state:
                        self.optim.state[p] = old_state[id(p)]
                        n_restored += 1
            print(f"[Bspline-Neus] optimizer rebuilt: {n_restored} params "
                  f"restored Adam state, "
                  f"{sum(len(g['params']) for g in self.optim.param_groups) - n_restored} fresh")

        if self.cfg.optim.sched.iteration_mode:
            self.sched.last_epoch = self.current_iteration
        else:
            self.sched.last_epoch = self.current_epoch
        # The checkpointer keeps references to optim/sched for state saving.
        self.checkpointer.optim = self.optim
        self.checkpointer.sched = self.sched

    def _extra_step(self, data):
        """Snapshot per-level gradient RMS right after backward.

        Called by the base trainer between ``backward()`` and the optimizer
        step/zero-grad, so this is the only point where per-parameter grads
        are still available.  Only computed on logging iterations (the
        reductions on the multi-million-element L4 tensors are cheap but not
        free).  Consumed by :meth:`log_wandb_scalars`.
        """
        if self.current_iteration % self.cfg.wandb_scalar_iter != 0:
            return
        grad_stats = {}
        for name in ("neural_sdf", "neural_rgb"):
            hier = getattr(getattr(self.model_module, name, None), "hier_field", None)
            if hier is None:
                continue
            for level_idx, level in enumerate(hier.levels):
                grad = level.values.grad
                if grad is not None:
                    grad_stats[f"{name}_L{level_idx}"] = (
                        grad.detach().pow(2).mean().sqrt().item()
                    )
        self._level_grad_rms = grad_stats

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
        for key, value in getattr(self, "_level_grad_rms", {}).items():
            scalars[f"{mode}/grad_rms/{key}"] = value
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
        # Register gradient-accumulation hooks for gradient-driven marking.
        if getattr(self.model_module.neural_sdf, "grad_marking_enabled", False):
            self.model_module.neural_sdf._register_grad_hooks()
        super().train(cfg, data_loader, single_gpu, profile, show_pbar)
