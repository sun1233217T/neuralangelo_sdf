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

from collections import defaultdict
from functools import partial
import torch
import torch.nn.functional as torch_F

from imaginaire.models.base import Model as BaseModel
from projects.neuralangelo.model import Model as NeuralangeloModel
from projects.neuralangelo.utils.modules import BackgroundNeRF
from projects.nerf.utils import nerf_util, camera, render
from projects.neuralangelo.utils import misc
from .utils.modules import BSplineSDFWrapper, BSplineRGBWrapper


class Model(NeuralangeloModel):
    """NeuS model with B-spline SDF and SH color fields.

    This inherits the rendering/compositing logic from
    :class:`projects.neuralangelo.model.Model` and only replaces the MLP-based
    ``neural_sdf`` and ``neural_rgb`` with B-spline field wrappers.
    """

    def __init__(self, cfg_model, cfg_data):
        # Call BaseModel init directly so that we can replace the model building
        # logic without executing NeuralangeloModel.build_model first.
        BaseModel.__init__(self, cfg_model, cfg_data)
        self.cfg_render = cfg_model.render
        self.white_background = cfg_model.background.white
        self.with_background = cfg_model.background.enabled
        self.with_appear_embed = cfg_model.appear_embed.enabled
        self.anneal_end = cfg_model.object.s_var.anneal_end
        self.outside_val = 1000. * (-1 if getattr(cfg_model.object, "inside_out", False) else 1)
        self.image_size_train = cfg_data.train.image_size
        self.image_size_val = cfg_data.val.image_size
        # B-spline specific configuration.
        self.bspline_cfg = cfg_model.object.bspline
        # Define models.
        self.build_model(cfg_model, cfg_data)
        # Define functions.
        self.ray_generator = partial(nerf_util.ray_generator,
                                     camera_ndc=False,
                                     num_rays=cfg_model.render.rand_rays)
        self.sample_dists_from_pdf = partial(nerf_util.sample_dists_from_pdf,
                                             intvs_fine=cfg_model.render.num_samples.fine)
        self.to_full_val_image = partial(misc.to_full_image, image_size=cfg_data.val.image_size)

    def build_model(self, cfg_model, cfg_data):
        cfg_bspline = cfg_model.object.bspline
        # Appearance encoding.
        if cfg_model.appear_embed.enabled:
            assert cfg_data.num_images is not None
            self.appear_embed = torch.nn.Embedding(cfg_data.num_images, cfg_model.appear_embed.dim)
            if cfg_model.background.enabled:
                self.appear_embed_outside = torch.nn.Embedding(cfg_data.num_images, cfg_model.appear_embed.dim)
            else:
                self.appear_embed_outside = None
        else:
            self.appear_embed = self.appear_embed_outside = None

        # B-spline fields replace NeuralSDF/NeuralRGB.
        self.neural_sdf = BSplineSDFWrapper(cfg_bspline)
        self.neural_rgb = BSplineRGBWrapper(cfg_bspline, sdf_wrapper=self.neural_sdf)

        # Optional background NeRF (reused from neuralangelo).
        if cfg_model.background.enabled:
            self.background_nerf = BackgroundNeRF(cfg_model.background, appear_embed=cfg_model.appear_embed)
        else:
            self.background_nerf = None

        # ``s_var`` is replaced by ``neural_sdf.raw_sdf_inv_std``.
        # Keep a dummy scalar so that any code referencing it does not crash,
        # but the actual inverse std comes from the B-spline wrapper.
        self.register_buffer("s_var", torch.tensor(0.0, dtype=torch.float32))

    @torch.no_grad()
    def get_dist_bounds(self, center, ray_unit):
        """Intersect rays with the B-spline bounding sphere.

        The parent implementation hard-codes ``radius=1.``, which corresponds
        to the canonical unit sphere used by the original MLP-based Neuralangelo.
        For B-spline fields the active world extent is controlled by
        ``target_half_bound`` (the half-length of the B-spline bounds cube),
        so we intersect with that radius instead.  This ensures ray samples
        stay inside the field bounds after ``readjust.scale`` has mapped the
        scene to fill ``[-target_half_bound, target_half_bound]^3``.
        """
        target_radius = float(getattr(self.bspline_cfg, "target_half_bound", 2.0))
        dist_near, dist_far = nerf_util.intersect_with_sphere(center, ray_unit, radius=target_radius)
        dist_near.relu_()
        outside = dist_near.isnan()
        # masked_fill_ instead of boolean-mask assignment (nonzero sync,
        # breaks CUDA-graph capture); same as the parent class fix.
        dist_near.masked_fill_(outside, 1)
        dist_far.masked_fill_(outside, 1.2)
        return dist_near, dist_far, outside

    def compute_neus_alphas(self, ray_unit, sdfs, gradients, dists, dist_far=None, progress=1., eps=1e-5,
                            points=None):
        """NeuS alpha computation using the B-spline SDF inverse std.

        When ``points`` are provided and the field is hierarchical, the
        per-level inv_std is evaluated at every sample (finest covering
        level); otherwise the scalar inv_std is broadcast as before.
        """
        sdfs = sdfs[..., 0]  # [B,R,N]
        if points is not None and getattr(self.neural_sdf, "hierarchical_enabled", False):
            inv_s = self.neural_sdf.inv_std_at(points)  # [B,R,N]
        else:
            inv_s = self.neural_sdf.inv_std()
        true_cos = (ray_unit[..., None, :] * gradients).sum(dim=-1, keepdim=False)  # [B,R,N]
        iter_cos = self._get_iter_cos(true_cos, progress=progress)  # [B,R,N]
        if dist_far is None:
            dist_far = torch.empty_like(dists[..., :1, :]).fill_(1e10)  # [B,R,1,1]
        dists = torch.cat([dists, dist_far], dim=2)  # [B,R,N+1,1]
        dist_intvs = dists[..., 1:, 0] - dists[..., :-1, 0]  # [B,R,N]
        est_prev_sdf = sdfs - iter_cos * dist_intvs * 0.5  # [B,R,N]
        est_next_sdf = sdfs + iter_cos * dist_intvs * 0.5  # [B,R,N]
        prev_cdf = (est_prev_sdf * inv_s).sigmoid()  # [B,R,N]
        next_cdf = (est_next_sdf * inv_s).sigmoid()  # [B,R,N]
        alphas = ((prev_cdf - next_cdf) / (prev_cdf + eps)).clip_(0.0, 1.0)  # [B,R,N]
        return alphas

    def _get_iter_cos(self, true_cos, progress=1.):
        # Tensor-safe variant: when CUDA-graph training is active, ``progress``
        # is a 0-dim CUDA tensor updated in place each iteration so the captured
        # graph reads the current value instead of a baked-in Python float.
        if torch.is_tensor(progress):
            anneal_ratio = torch.clamp(progress / self.anneal_end, max=1.0)
        else:
            anneal_ratio = min(progress / self.anneal_end, 1.)
        # The anneal strategy below keeps the cos value alive at the beginning of training iterations.
        return -((-true_cos * 0.5 + 0.5).relu() * (1.0 - anneal_ratio) +
                 (-true_cos).relu() * anneal_ratio)  # always non-positive

    def render_rays(self, center, ray_unit, sample_idx=None, stratified=False):
        """Extend the parent with loss-guided refinement statistics.

        In training mode (and only when the SDF wrapper collects loss-guided
        marking statistics) the object sample points and their compositing
        weights are exposed as ``hier_points``/``hier_weights`` so the caller
        (eager ``_compute_loss`` or the captured graph step) can scatter the
        per-ray render error into the finest level's cell grid.
        """
        output = super().render_rays(center, ray_unit, sample_idx=sample_idx,
                                     stratified=stratified)
        if self.training and getattr(self.neural_sdf, "loss_marking_enabled", False):
            n_obj = output["gradients"].shape[2]
            dists_obj = output["dists"][:, :, :n_obj]  # [B,R,No,1]
            points = camera.get_3D_points_from_dist(center, ray_unit, dists_obj)
            output["hier_points"] = points.detach()  # [B,R,No,3]
            output["hier_weights"] = output["weights"][:, :, :n_obj, 0].detach()  # [B,R,No]
            # SDF values at each sample point for surface-crossing anchoring.
            output["hier_sdfs"] = output["sdfs"].detach()  # [B,R,No]
        return output

    def render_image(self, pose, intr, image_size, stratified=False, sample_idx=None):
        """Render full images with reduced memory footprint for inference.

        The parent implementation keeps every per-sample tensor (``dists``,
        ``weights``, ``gradients``, ``rgbs``, ``sdfs``) for all rays of the
        entire validation image.  For B-spline fields with dense hierarchical
        sampling this consumes many GBs of GPU memory per image.  Here we only
        retain the tensors required by :meth:`inference`:
        ``rgb``, ``opacity``, ``depth`` and ``gradient``.
        """
        output = defaultdict(list)
        for center, ray, _ in self.ray_generator(pose, intr, image_size, full_image=True):
            ray_unit = torch_F.normalize(ray, dim=-1)  # [B,R,3]
            output_batch = self.render_rays(center, ray_unit, sample_idx=sample_idx, stratified=stratified)
            if not self.training:
                dist = render.composite(output_batch["dists"], output_batch["weights"])  # [B,R,1]
                depth = dist / ray.norm(dim=-1, keepdim=True)
                output_batch.update(depth=depth)
            # Only keep the tensors that inference() actually uses.
            keep_keys = {"rgb", "opacity", "depth", "gradient"}
            for key, value in output_batch.items():
                if value is not None and key in keep_keys:
                    output[key].append(value.detach())
        for key, value in output.items():
            output[key] = torch.cat(value, dim=1)
        return output

    def get_param_groups(self, cfg_optim):
        """Return parameter groups with B-spline-specific learning rates.

        Per-level LR scaling is controlled by optional config lists:
        ``sdf_level_lr_scale`` and ``color_feature_level_lr_scale``.  Each
        list maps level index to a multiplier on the group LR.  Missing
        entries default to 1.0.  When absent (None), all levels share the
        group LR as before.  Because the trainer rebuilds the optimizer at
        every refinement, new levels automatically pick up their group's LR.
        """
        base_lr = cfg_optim.params.lr
        color_lr = float(getattr(self.bspline_cfg, "color_lr", base_lr))
        param_groups = []

        # --- SDF field ---
        sdf_level_scale = getattr(self.bspline_cfg, "sdf_level_lr_scale", None)
        if self.neural_sdf.hierarchical_enabled and sdf_level_scale is not None:
            sdf_level_scale = [float(s) for s in sdf_level_scale]
            # inv_std and any non-level SDF params stay at base LR.
            non_level = [
                p for n, p in self.neural_sdf.named_parameters()
                if p.requires_grad and not n.startswith("hier_field.levels.")
            ]
            if non_level:
                param_groups.append({"params": non_level, "lr": base_lr})
            for l, level in enumerate(self.neural_sdf.hier_field.levels):
                scale = sdf_level_scale[l] if l < len(sdf_level_scale) else 1.0
                if level.values.requires_grad:
                    param_groups.append({"params": [level.values], "lr": base_lr * scale})
        else:
            sdf_params = [p for p in self.neural_sdf.parameters() if p.requires_grad]
            if sdf_params:
                param_groups.append({"params": sdf_params, "lr": base_lr})

        # --- Color field ---
        color_params = [p for p in self.neural_rgb.parameters() if p.requires_grad]
        if color_params:
            bspline_cfg = self.bspline_cfg
            if str(getattr(bspline_cfg, "color_mode", "sh")).lower() == "mlp":
                feature_lr = float(getattr(bspline_cfg, "color_feature_lr", color_lr))
                mlp_lr = float(getattr(bspline_cfg, "color_mlp_lr", color_lr))
                feat_level_scale = getattr(bspline_cfg, "color_feature_level_lr_scale", None)

                if feat_level_scale is not None and self.neural_rgb.hierarchical_enabled:
                    feat_level_scale = [float(s) for s in feat_level_scale]
                    for l, level in enumerate(self.neural_rgb.hier_field.levels):
                        scale = feat_level_scale[l] if l < len(feat_level_scale) else 1.0
                        if level.values.requires_grad:
                            param_groups.append(
                                {"params": [level.values], "lr": feature_lr * scale})
                else:
                    feature_params = [
                        p for n, p in self.neural_rgb.named_parameters()
                        if p.requires_grad and n.startswith("hier_field.")
                    ]
                    if feature_params:
                        param_groups.append({"params": feature_params, "lr": feature_lr})

                mlp_params = [
                    p for n, p in self.neural_rgb.named_parameters()
                    if p.requires_grad and n.startswith("_mlp.")
                ]
                if mlp_params:
                    param_groups.append({"params": mlp_params, "lr": mlp_lr})
                # Fallback for any unexpected color params (should be empty).
                other_color_params = [
                    p for n, p in self.neural_rgb.named_parameters()
                    if p.requires_grad
                    and not n.startswith("hier_field.")
                    and not n.startswith("_mlp.")
                ]
                if other_color_params:
                    param_groups.append({"params": other_color_params, "lr": color_lr})
            else:
                param_groups.append({"params": color_params, "lr": color_lr})

        # Everything else (background NeRF, appearance embeddings) uses base lr.
        other_params = [
            param
            for name, param in self.named_parameters()
            if param.requires_grad
            and not name.startswith(("neural_sdf.", "neural_rgb."))
        ]
        if other_params:
            param_groups.append({"params": other_params, "lr": base_lr})

        return param_groups

    def prepare_load_state_dict(self, state_dict):
        """Grow hierarchical fields to the checkpoint's depth before loading.

        Refined levels only exist in the checkpoint, not in a freshly built
        model, so a plain load would silently drop them (the checkpointer
        filters unknown keys with ``strict=False``).  The checkpointer calls
        this hook with the *unfiltered* checkpoint dict before any key/shape
        filtering, making resume and evaluation across refinement boundaries
        exact.
        """
        restored = False
        # Migrate pre-T3 checkpoints: the scalar inv_std becomes per-level.
        for prefix in ("neural_sdf.", "module.neural_sdf."):
            old_key = f"{prefix}raw_sdf_inv_std"
            new_key = f"{prefix}raw_sdf_inv_std_levels"
            if old_key in state_dict and new_key not in state_dict:
                old = state_dict[old_key]
                levels = self.neural_sdf.hier_field.max_levels if \
                    getattr(self.neural_sdf, "hierarchical_enabled", False) else 1
                state_dict[new_key] = old.reshape(()).expand(levels).clone()
            state_dict.pop(old_key, None)
        for mod_name in ("neural_sdf", "neural_rgb"):
            field = getattr(getattr(self, mod_name, None), "hier_field", None)
            if field is None:
                continue
            for prefix in (f"{mod_name}.hier_field.", f"module.{mod_name}.hier_field."):
                if field.restore_levels_from_state_dict(state_dict, prefix):
                    restored = True
                    break
            # A hierarchy that shares its structure with another field does not
            # own those buffers, so any structure keys from an older checkpoint
            # (where it owned its own structure) must be stripped before the
            # strict state-dict load, otherwise they appear as unexpected keys.
            if getattr(field, "structure_owner", None) is not None:
                struct_names = {"region", "index_grid", "omega", "rmask"}
                for key in list(state_dict.keys()):
                    if not key.startswith(f"{mod_name}.hier_field.") and not key.startswith(
                        f"module.{mod_name}.hier_field."
                    ):
                        continue
                    local = key.split("hier_field.", 1)[-1]
                    if local.startswith("_structures."):
                        del state_dict[key]
                    elif local.startswith("levels."):
                        parts = local.split(".")
                        if len(parts) == 3 and parts[2] in struct_names:
                            del state_dict[key]
        # Read by the trainer's post-checkpoint-load hook to decide whether
        # the optimizer must be rebuilt around the restored parameters.
        self._hier_structure_restored = restored

    def load_state_dict(self, state_dict, strict=True):
        self.prepare_load_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict)
