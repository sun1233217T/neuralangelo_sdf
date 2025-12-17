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

from functools import partial
import torch
import torch.nn.functional as torch_F
from collections import defaultdict

from imaginaire.models.base import Model as BaseModel
from projects.nerf.utils import nerf_util, camera, render
from projects.sdf_angelo.utils import misc
from projects.sdf_angelo.utils.modules import NeuralSDF, NeuralRGB, BackgroundNeRF

from mtools import debug


class Model(BaseModel):

    def __init__(self, cfg_model, cfg_data):
        super().__init__(cfg_model, cfg_data)
        self.cfg_render = cfg_model.render
        self.white_background = cfg_model.background.white
        self.with_background = cfg_model.background.enabled
        self.with_appear_embed = cfg_model.appear_embed.enabled
        self.anneal_end = cfg_model.object.s_var.anneal_end
        self.outside_val = 1000. * (-1 if cfg_model.object.sdf.mlp.inside_out else 1)
        self.image_size_train = cfg_data.train.image_size
        self.image_size_val = cfg_data.val.image_size
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
        # appearance encoding
        if cfg_model.appear_embed.enabled:
            assert cfg_data.num_images is not None
            self.appear_embed = torch.nn.Embedding(cfg_data.num_images, cfg_model.appear_embed.dim)
            if cfg_model.background.enabled:
                self.appear_embed_outside = torch.nn.Embedding(cfg_data.num_images, cfg_model.appear_embed.dim)
            else:
                self.appear_embed_outside = None
        else:
            self.appear_embed = self.appear_embed_outside = None
        self.neural_sdf = NeuralSDF(cfg_model.object.sdf)
        self.neural_rgb = NeuralRGB(cfg_model.object.rgb, feat_dim=cfg_model.object.sdf.mlp.hidden_dim,
                                    appear_embed=cfg_model.appear_embed)
        if cfg_model.background.enabled:
            self.background_nerf = BackgroundNeRF(cfg_model.background, appear_embed=cfg_model.appear_embed)
        else:
            self.background_nerf = None
        self.s_var = torch.nn.Parameter(torch.tensor(cfg_model.object.s_var.init_val, dtype=torch.float32))

    def forward(self, data):
        # Randomly sample and render the pixels.
        output = self.render_pixels_surface(data["pose"], data["intr"], image_size=self.image_size_train,
                                    stratified=self.cfg_render.stratified, sample_idx=data["idx"],
                                    ray_idx=data["ray_idx"])
        return output

    @torch.no_grad()
    def inference(self, data):
        self.eval()
        # Render the full images.
        import time
        # time0 = time.time()
        # output1 = self.render_image(data["pose"], data["intr"], image_size=self.image_size_val,
        #                            stratified=False, sample_idx=data["idx"])  # [B,N,C]
        time1 = time.time()
        output = self.rander_image_surface(data["pose"], data["intr"], image_size=self.image_size_val,
                                   stratified=False, sample_idx=data["idx"])  # [B,N,C]
        time2 = time.time()
        print(f"Render surface time: {time2 - time1:.4f}s")
        # debug()
        # Get full rendered RGB and depth images.
        rot = data["pose"][..., :3, :3]  # [B,3,3]
        normal_cam = -output["gradient"] @ rot.transpose(-1, -2)  # [B,HW,3]
        output.update(
            rgb_map=self.to_full_val_image(output["rgb"]),  # [B,3,H,W]
            opacity_map=self.to_full_val_image(output["opacity"]),  # [B,1,H,W]
            depth_map=self.to_full_val_image(output["depth"]),  # [B,1,H,W]
            normal_map=self.to_full_val_image(normal_cam),  # [B,3,H,W]
        )
        return output

    def render_image(self, pose, intr, image_size, stratified=False, sample_idx=None):
        """ Render the rays given the camera intrinsics and poses.
        Args:
            pose (tensor [batch,3,4]): Camera poses ([R,t]).
            intr (tensor [batch,3,3]): Camera intrinsics.
            stratified (bool): Whether to stratify the depth sampling.
            sample_idx (tensor [batch]): Data sample index.
        Returns:
            output: A dictionary containing the outputs.
        """
        output = defaultdict(list)
        for center, ray, _ in self.ray_generator(pose, intr, image_size, full_image=True):
            ray_unit = torch_F.normalize(ray, dim=-1)  # [B,R,3]
            output_batch = self.render_rays(center, ray_unit, sample_idx=sample_idx, stratified=stratified)
            if not self.training:
                dist = render.composite(output_batch["dists"], output_batch["weights"])  # [B,R,1]
                depth = dist / ray.norm(dim=-1, keepdim=True)
                output_batch.update(depth=depth)
            for key, value in output_batch.items():
                if value is not None:
                    output[key].append(value.detach())
        # Concat each item (list) in output into one tensor. Concatenate along the ray dimension (1)
        for key, value in output.items():
            output[key] = torch.cat(value, dim=1)
        return output

    def render_pixels(self, pose, intr, image_size, stratified=False, sample_idx=None, ray_idx=None):
        center, ray = camera.get_center_and_ray(pose, intr, image_size)  # [B,HW,3]
        center = nerf_util.slice_by_ray_idx(center, ray_idx)  # [B,R,3]
        ray = nerf_util.slice_by_ray_idx(ray, ray_idx)  # [B,R,3]
        ray_unit = torch_F.normalize(ray, dim=-1)  # [B,R,3]
        output = self.render_rays(center, ray_unit, sample_idx=sample_idx, stratified=stratified)
        return output

    def render_rays(self, center, ray_unit, sample_idx=None, stratified=False):
        with torch.no_grad():
            near, far, outside = self.get_dist_bounds(center, ray_unit)
        app, app_outside = self.get_appearance_embedding(sample_idx, ray_unit.shape[1])
        output_object = self.render_rays_object(center, ray_unit, near, far, outside, app, stratified=stratified)
        if self.with_background:
            output_background = self.render_rays_background(center, ray_unit, far, app_outside, stratified=stratified)
            # Concatenate object and background samples.
            rgbs = torch.cat([output_object["rgbs"], output_background["rgbs"]], dim=2)  # [B,R,No+Nb,3]
            dists = torch.cat([output_object["dists"], output_background["dists"]], dim=2)  # [B,R,No+Nb,1]
            alphas = torch.cat([output_object["alphas"], output_background["alphas"]], dim=2)  # [B,R,No+Nb]
        else:
            rgbs = output_object["rgbs"]  # [B,R,No,3]
            dists = output_object["dists"]  # [B,R,No,1]
            alphas = output_object["alphas"]  # [B,R,No]
        weights = render.alpha_compositing_weights(alphas)  # [B,R,No+Nb,1]
        # Compute weights and composite samples.
        rgb = render.composite(rgbs, weights)  # [B,R,3]
        if self.white_background:
            opacity_all = render.composite(1., weights)  # [B,R,1]
            rgb = rgb + (1 - opacity_all)
        # Collect output.
        output = dict(
            rgb=rgb,  # [B,R,3]
            opacity=output_object["opacity"],  # [B,R,1]/None
            outside=outside,  # [B,R,1]
            dists=dists,  # [B,R,No+Nb,1]
            weights=weights,  # [B,R,No+Nb,1]
            gradient=output_object["gradient"],  # [B,R,3]/None
            gradients=output_object["gradients"],  # [B,R,No,3]
            hessians=output_object["hessians"],  # [B,R,No,3]/None
        )
        return output
    
    def render_pixels_surface(self, pose, intr, image_size, stratified=False, sample_idx=None, ray_idx=None):
        center, ray = camera.get_center_and_ray(pose, intr, image_size)  # [B,HW,3]
        center = nerf_util.slice_by_ray_idx(center, ray_idx)  # [B,R,3]
        ray = nerf_util.slice_by_ray_idx(ray, ray_idx)  # [B,R,3]
        ray_unit = torch_F.normalize(ray, dim=-1)  # [B,R,3]
        output = self.render_ray_surface(center, ray_unit, sample_idx=sample_idx, stratified=stratified)
        return output
    
    def rander_image_surface(self, pose, intr, image_size, stratified=False, sample_idx=None):
        """ Render the rays given the camera intrinsics and poses.
        Args:
            pose (tensor [batch,3,4]): Camera poses ([R,t]).
            intr (tensor [batch,3,3]): Camera intrinsics.
            stratified (bool): Whether to stratify the depth sampling.
            sample_idx (tensor [batch]): Data sample index.
        Returns:
            output: A dictionary containing the outputs.
        """

        output = defaultdict(list)
        for center, ray, _ in self.ray_generator(pose, intr, image_size, full_image=True):
            ray_unit = torch_F.normalize(ray, dim=-1)  # [B,R,3]
            output_batch = self.render_ray_surface(center, ray_unit, sample_idx=sample_idx, stratified=stratified)
            for key, value in output_batch.items():
                if value is not None:
                    output[key].append(value.detach())
        # Concat each item (list) in output into one tensor. Concatenate along the ray dimension (1)
        for key, value in output.items():
            output[key] = torch.cat(value, dim=1)
        return output
    
    def render_ray_surface(self, center, ray_unit, sample_idx=None, stratified=False, sample_eps=1e-3):
        app, app_outside = self.get_appearance_embedding(sample_idx, ray_unit.shape[1])
        with torch.no_grad():
            near, far, outside = self.get_dist_bounds(center, ray_unit)
            surf_pts, hit_mask, t_vals = self.det_surface(center, ray_unit, near, far)
        # Only evaluate valid hits to avoid NaNs and unnecessary compute.
        outside_bool = outside.squeeze(-1) if outside.dim() > 2 else outside  # [B,R]
        valid = hit_mask & (~outside_bool)  # [B,R]
        # Pre-allocate outputs.
        rgbs = torch.zeros(*center.shape[:2], 3, device=center.device, dtype=center.dtype)
        rgb_offsets = torch.zeros(*center.shape[:2], 4, 3, device=center.device, dtype=center.dtype)  # +u,-u,+v,-v
        sdf_offsets = torch.zeros(*center.shape[:2], 4, device=center.device, dtype=center.dtype)
        gradients = torch.zeros_like(rgbs)
        grads_v = torch.zeros_like(rgbs)
        hessians = torch.zeros_like(rgbs) if self.training else None
        # sample_points: [B,R,5,3] (surface, +u, -u, +v, -v)
        sample_points = torch.zeros(*center.shape[:2], 5, 3, device=center.device, dtype=center.dtype)
        if valid.any():
            pts_valid = surf_pts[valid]  # [Nh,3]
            rays_valid = ray_unit[valid]  # [Nh,3]
            if app is not None:
                app_valid = app[..., 0, :][valid]  # [Nh,C]
            else:
                app_valid = None
            # Build orthonormal basis (u,v) orthogonal to each ray for lateral sampling.
            helper = torch.tensor([0.0, 0.0, 1.0], device=rays_valid.device, dtype=rays_valid.dtype).expand_as(rays_valid)
            alt_helper = torch.tensor([1.0, 0.0, 0.0], device=rays_valid.device, dtype=rays_valid.dtype)
            helper = torch.where((rays_valid.abs().sum(dim=-1, keepdim=True) < 1e-6).expand_as(helper), alt_helper, helper)
            u = torch_F.normalize(torch.cross(rays_valid, helper, dim=-1), dim=-1)
            v = torch_F.normalize(torch.cross(rays_valid, u, dim=-1), dim=-1)
            pts_u_plus = pts_valid + u * sample_eps
            pts_u_minus = pts_valid - u * sample_eps
            pts_v_plus = pts_valid + v * sample_eps
            pts_v_minus = pts_valid - v * sample_eps
            # Arrange as [5,Nh,3]: surface, +u, -u, +v, -v then scatter.
            sample_stack = torch.stack([pts_valid, pts_u_plus, pts_u_minus, pts_v_plus, pts_v_minus], dim=0)  # [5,Nh,3]
            # For downstream outputs keep per-ray grouping.
            sample_points[valid] = sample_stack.permute(1, 0, 2)  # [Nh,5,3]
            valid_num = pts_valid.shape[0]
            # Repeat per-hit attributes for lateral samples.
            sample_points_flat = sample_stack.view(-1, 3).contiguous()  # [5Nh,3] ordered blocks
            rays_repeat = torch.cat([rays_valid] * 5, dim=0)  # [5Nh,3]
            if app_valid is not None:
                app_repeat = torch.cat([app_valid] * 5, dim=0)  # [5Nh,C]
            else:
                app_repeat = None
            sdfs_v, feats_v = self.neural_sdf.forward(sample_points_flat)  # [5Nh,1],[5Nh,K]
            sdfs_v = sdfs_v.squeeze(-1)  # [5Nh]
            sdfs_v = sdfs_v.view(-1, 1)  # keep 2D for compute_gradients signature
            grads_v, hess_v = self.neural_sdf.compute_gradients(sample_points_flat, training=self.training, sdf=sdfs_v)
            normals_v = torch_F.normalize(grads_v, dim=-1)  # [5Nh,3]
            rgbs_v = self.neural_rgb.forward(sample_points_flat, normals_v, rays_repeat, feats_v, app=app_repeat)  # [5Nh,3]
            # Scatter back.
            rgbs[valid] = rgbs_v[:valid_num]
            # Offsets: blocks of size valid_num in order [+u, -u, +v, -v].
            sdf_offsets[valid, 0] = sdfs_v[valid_num:2*valid_num].squeeze(-1)
            sdf_offsets[valid, 1] = sdfs_v[2*valid_num:3*valid_num].squeeze(-1)
            sdf_offsets[valid, 2] = sdfs_v[3*valid_num:4*valid_num].squeeze(-1)
            sdf_offsets[valid, 3] = sdfs_v[4*valid_num:5*valid_num].squeeze(-1)
            rgb_offsets[valid, 0] = rgbs_v[valid_num:2*valid_num]
            rgb_offsets[valid, 1] = rgbs_v[2*valid_num:3*valid_num]
            rgb_offsets[valid, 2] = rgbs_v[3*valid_num:4*valid_num]
            rgb_offsets[valid, 3] = rgbs_v[4*valid_num:5*valid_num]
            gradients[valid] = grads_v[:valid_num]
            if hessians is not None and hess_v is not None:
                hessians[valid] = hess_v[:valid_num]
        # Mask out invalid rays in final maps.
        valid_f = valid.float()[..., None]
        opacity = valid_f  # [B,R,1]
        depth = t_vals[..., None] * valid_f  # [B,R,1]
        rgbs = rgbs * valid_f
        gradients = gradients * valid_f
        if hessians is not None:
            hessians = hessians * valid_f
        # Collect output.
        output = dict(
            rgb=rgbs,  # [B,R,3]
            opacity=opacity,  # [B,R,1]/None
            # surf_pts=surf_pts,  # [B,R,3]
            # hit_mask=hit_mask,  # [B,R]
            depth = depth, # [B,R]
            outside=outside,  # [B,R]
            # t_vals=t_vals,  # [B,R]
            gradients=grads_v.unsqueeze(0),  # [B,R,3]， 这玩意用来算SDF梯度模长1的约束的， 
            gradient = gradients,  # [B,R,3]/None， 这玩意用来算法线的
            hessians=hessians,  # [B,R,3]/None， 这玩意用来算 curvature_loss，对 hessian.sum(dim=-1)（近似拉普拉斯）取绝对值求平均，用来鼓励平滑/低曲率的 SDF。
            sample_points=sample_points,  # [B,R,5,3]: surface, +u, -u, +v, -v
            sdf_offsets=sdf_offsets,      # [B,R,4]: +u, -u, +v, -v
            rgb_offsets=rgb_offsets,      # [B,R,4,3]
        )
        return output 


    def render_rays_object(self, center, ray_unit, near, far, outside, app, stratified=False):
        with torch.no_grad():
            dists = self.sample_dists_all(center, ray_unit, near, far, stratified=stratified)  # [B,R,N,3]
        points = camera.get_3D_points_from_dist(center, ray_unit, dists)  # [B,R,N,3]
        sdfs, feats = self.neural_sdf.forward(points)  # [B,R,N,1],[B,R,N,K]
        # surf_pts, hit_mask, t_vals = self.det_surface(center, ray_unit, near, far)
        # debug()
        sdfs[outside[..., None].expand_as(sdfs)] = self.outside_val
        # Compute 1st- and 2nd-order gradients.
        rays_unit = ray_unit[..., None, :].expand_as(points).contiguous()  # [B,R,N,3]
        gradients, hessians = self.neural_sdf.compute_gradients(points, training=self.training, sdf=sdfs)
        normals = torch_F.normalize(gradients, dim=-1)  # [B,R,N,3]
        rgbs = self.neural_rgb.forward(points, normals, rays_unit, feats, app=app)  # [B,R,N,3]
        # SDF volume rendering.
        alphas = self.compute_neus_alphas(ray_unit, sdfs, gradients, dists, dist_far=far[..., None],
                                          progress=self.progress)  # [B,R,N]
        if not self.training:
            weights = render.alpha_compositing_weights(alphas)  # [B,R,N,1]
            opacity = render.composite(1., weights)  # [B,R,1]
            gradient = render.composite(gradients, weights)  # [B,R,3]
        else:
            opacity = None
            gradient = None
        # Collect output.
        output = dict(
            rgbs=rgbs,  # [B,R,N,3]
            sdfs=sdfs[..., 0],  # [B,R,N]
            dists=dists,  # [B,R,N,1]
            alphas=alphas,  # [B,R,N]
            opacity=opacity,  # [B,R,3]/None
            gradient=gradient,  # [B,R,3]/None
            gradients=gradients,  # [B,R,N,3]
            hessians=hessians,  # [B,R,N,3]/None
        )
        return output

    def render_rays_background(self, center, ray_unit, far, app_outside, stratified=False):
        with torch.no_grad():
            dists = self.sample_dists_background(ray_unit, far, stratified=stratified)
        points = camera.get_3D_points_from_dist(center, ray_unit, dists)  # [B,R,N,3]
        rays_unit = ray_unit[..., None, :].expand_as(points)  # [B,R,N,3]
        rgbs, densities = self.background_nerf.forward(points, rays_unit, app_outside)  # [B,R,N,3]
        alphas = render.volume_rendering_alphas_dist(densities, dists)  # [B,R,N]
        # Collect output.
        output = dict(
            rgbs=rgbs,  # [B,R,3]
            dists=dists,  # [B,R,N,1]
            alphas=alphas,  # [B,R,N]
        )
        return output

    @torch.no_grad()
    def get_dist_bounds(self, center, ray_unit):
        dist_near, dist_far = nerf_util.intersect_with_sphere(center, ray_unit, radius=1.)
        dist_near.relu_()  # Distance (and thus depth) should be non-negative.
        outside = dist_near.isnan()
        dist_near[outside], dist_far[outside] = 1, 1.2  # Dummy distances. Density will be set to 0.
        return dist_near, dist_far, outside

    def get_appearance_embedding(self, sample_idx, num_rays):
        if self.with_appear_embed:
            # Object appearance embedding.
            num_samples_all = self.cfg_render.num_samples.coarse + \
                self.cfg_render.num_samples.fine * self.cfg_render.num_sample_hierarchy
            app = self.appear_embed(sample_idx)[:, None, None]  # [B,1,1,C]
            app = app.expand(-1, num_rays, num_samples_all, -1)  # [B,R,N,C]
            # Background appearance embedding.
            if self.with_background:
                app_outside = self.appear_embed_outside(sample_idx)[:, None, None]  # [B,1,1,C]
                app_outside = app_outside.expand(-1, num_rays, self.cfg_render.num_samples.background, -1)  # [B,R,N,C]
            else:
                app_outside = None
        else:
            app = app_outside = None
        return app, app_outside

    @torch.no_grad()
    def sample_dists_all(self, center, ray_unit, near, far, stratified=False):
        dists = nerf_util.sample_dists(ray_unit.shape[:2], dist_range=(near[..., None], far[..., None]),
                                       intvs=self.cfg_render.num_samples.coarse, stratified=stratified)
        if self.cfg_render.num_sample_hierarchy > 0:
            points = camera.get_3D_points_from_dist(center, ray_unit, dists)  # [B,R,N,3]
            sdfs = self.neural_sdf.sdf(points)  # [B,R,N]
        for h in range(self.cfg_render.num_sample_hierarchy):
            dists_fine = self.sample_dists_hierarchical(dists, sdfs, inv_s=(64 * 2 ** h))  # [B,R,Nf,1]
            dists = torch.cat([dists, dists_fine], dim=2)  # [B,R,N+Nf,1]
            dists, sort_idx = dists.sort(dim=2)
            if h != self.cfg_render.num_sample_hierarchy - 1:
                points_fine = camera.get_3D_points_from_dist(center, ray_unit, dists_fine)  # [B,R,Nf,3]
                sdfs_fine = self.neural_sdf.sdf(points_fine)  # [B,R,Nf]
                sdfs = torch.cat([sdfs, sdfs_fine], dim=2)  # [B,R,N+Nf]
                sdfs = sdfs.gather(dim=2, index=sort_idx.expand_as(sdfs))  # [B,R,N+Nf,1]
        return dists

    def sample_dists_hierarchical(self, dists, sdfs, inv_s, robust=True, eps=1e-5):
        sdfs = sdfs[..., 0]  # [B,R,N]
        prev_sdfs, next_sdfs = sdfs[..., :-1], sdfs[..., 1:]  # [B,R,N-1]
        prev_dists, next_dists = dists[..., :-1, 0], dists[..., 1:, 0]  # [B,R,N-1]
        mid_sdfs = (prev_sdfs + next_sdfs) * 0.5  # [B,R,N-1]
        cos_val = (next_sdfs - prev_sdfs) / (next_dists - prev_dists + 1e-5)  # [B,R,N-1]
        if robust:
            prev_cos_val = torch.cat([torch.zeros_like(cos_val)[..., :1], cos_val[..., :-1]], dim=-1)  # [B,R,N-1]
            cos_val = torch.stack([prev_cos_val, cos_val], dim=-1).min(dim=-1).values  # [B,R,N-1]
        dist_intvs = dists[..., 1:, 0] - dists[..., :-1, 0]  # [B,R,N-1]
        est_prev_sdf = mid_sdfs - cos_val * dist_intvs * 0.5  # [B,R,N-1]
        est_next_sdf = mid_sdfs + cos_val * dist_intvs * 0.5  # [B,R,N-1]
        prev_cdf = (est_prev_sdf * inv_s).sigmoid()  # [B,R,N-1]
        next_cdf = (est_next_sdf * inv_s).sigmoid()  # [B,R,N-1]
        alphas = ((prev_cdf - next_cdf) / (prev_cdf + eps)).clip_(0.0, 1.0)  # [B,R,N-1]
        weights = render.alpha_compositing_weights(alphas)  # [B,R,N-1,1]
        dists_fine = self.sample_dists_from_pdf(dists, weights=weights[..., 0])  # [B,R,Nf,1]
        return dists_fine

    def sample_dists_background(self, ray_unit, far, stratified=False, eps=1e-5):
        inv_dists = nerf_util.sample_dists(ray_unit.shape[:2], dist_range=(1, 0),
                                           intvs=self.cfg_render.num_samples.background, stratified=stratified)
        dists = far[..., None] / (inv_dists + eps)  # [B,R,N,1]
        return dists

    @torch.no_grad()
    def det_surface(self, center, ray_unit, near, far, max_steps=128, eps=1e-4, step_scale=0.9, refine_steps=3):
        """Sphere tracing to find the first SDF zero crossing along rays, with optional bisection refinement on sign flips.
        Args:
            center (tensor [B,R,3]): Ray origins.
            ray_unit (tensor [B,R,3]): Normalized ray directions.
            near/far (tensor [B,R]): Per-ray bounds (from get_dist_bounds).
            max_steps (int): Maximum sphere tracing iterations.
            eps (float): Convergence threshold on |SDF|.
            step_scale (float): Safety factor on step size to avoid overshooting.
            refine_steps (int): Bisection steps when a sign flip is observed.
        Returns:
            surface_points (tensor [B,R,3]): NaN where miss or out-of-bounds.
            hit_mask (tensor [B,R]): True where a surface is found within bounds.
            t_vals (tensor [B,R]): Final traced distance values.
        """
        # Normalize near/far to shape [B,R].
        if near.dim() > 2:
            near = near[..., 0]
        if far.dim() > 2:
            far = far[..., -1]
        near = near.squeeze(-1)
        far = far.squeeze(-1)
        if near.shape != center.shape[:2] or far.shape != center.shape[:2]:
            raise ValueError(f"det_surface expects near/far shape {center.shape[:2]}, got {near.shape}, {far.shape}")
        # Initialize travel distance along each ray.
        t_vals = near.clone()  # [B,R]
        hit_mask = torch.zeros_like(near, dtype=torch.bool)  # [B,R]
        surface_points = torch.full_like(center, float("nan"))  # [B,R,3]
        # Early skip rays that start outside the unit sphere intersection range.
        valid = far > near
        prev_t = None
        prev_sdf = None
        for _ in range(max_steps):
            if not valid.any() or hit_mask.all():
                break
            pts = center + ray_unit * t_vals[..., None]  # [B,R,3]
            sdfs = self.neural_sdf.sdf(pts)[..., 0]  # [B,R]
            converged = (sdfs.abs() < eps) & valid & (~hit_mask)
            if converged.any():
                surface_points[converged] = pts[converged]
                hit_mask[converged] = True
            unfinished = valid & (~hit_mask)
            # Optional sign-flip refinement (bisection) to reduce oversteps near sharp features.
            if prev_sdf is not None and refine_steps > 0:
                sign_flip = (unfinished & (sdfs * prev_sdf < 0))
                if sign_flip.any():
                    # Order the bracket so that low/high follow the sign of prev_sdf.
                    t_low = torch.where(prev_sdf < 0, prev_t, t_vals)
                    t_high = torch.where(prev_sdf < 0, t_vals, prev_t)
                    sdf_low = torch.where(prev_sdf < 0, prev_sdf, sdfs)
                    sdf_high = torch.where(prev_sdf < 0, sdfs, prev_sdf)
                    for _ in range(refine_steps):
                        t_mid = 0.5 * (t_low + t_high)
                        pts_mid = center + ray_unit * t_mid[..., None]
                        sdf_mid = self.neural_sdf.sdf(pts_mid)[..., 0]
                        go_low = (sdf_low * sdf_mid < 0)
                        t_high = torch.where(go_low, t_mid, t_high)
                        sdf_high = torch.where(go_low, sdf_mid, sdf_high)
                        t_low = torch.where(go_low, t_low, t_mid)
                        sdf_low = torch.where(go_low, sdf_low, sdf_mid)
                    # Take mid as refined hit.
                    t_vals = torch.where(sign_flip, t_mid, t_vals)
                    sdfs = torch.where(sign_flip, sdf_mid, sdfs)
                    pts = torch.where(sign_flip[..., None], pts_mid, pts)
                    surface_points[sign_flip] = pts_mid[sign_flip]
                    hit_mask[sign_flip] = True
            # Update steps for unfinished rays.
            if not unfinished.any():
                break
            step = sdfs.abs().clamp_min(eps) * step_scale  # [B,R]
            t_vals = t_vals + step * unfinished.float()
            # Invalidate rays that marched past far bound.
            valid = unfinished & (t_vals <= far)
            # Cache previous values for refinement.
            prev_t = t_vals.clone()
            prev_sdf = sdfs.clone()
        return surface_points, hit_mask, t_vals

    def compute_neus_alphas(self, ray_unit, sdfs, gradients, dists, dist_far=None, progress=1., eps=1e-5):
        sdfs = sdfs[..., 0]  # [B,R,N]
        # SDF volume rendering in NeuS.
        inv_s = self.s_var.exp()
        true_cos = (ray_unit[..., None, :] * gradients).sum(dim=-1, keepdim=False)  # [B,R,N]
        iter_cos = self._get_iter_cos(true_cos, progress=progress)  # [B,R,N]
        # Estimate signed distances at section points
        if dist_far is None:
            dist_far = torch.empty_like(dists[..., :1, :]).fill_(1e10)  # [B,R,1,1]
        dists = torch.cat([dists, dist_far], dim=2)  # [B,R,N+1,1]
        dist_intvs = dists[..., 1:, 0] - dists[..., :-1, 0]  # [B,R,N]
        est_prev_sdf = sdfs - iter_cos * dist_intvs * 0.5  # [B,R,N]
        est_next_sdf = sdfs + iter_cos * dist_intvs * 0.5  # [B,R,N]
        prev_cdf = (est_prev_sdf * inv_s).sigmoid()  # [B,R,N]
        next_cdf = (est_next_sdf * inv_s).sigmoid()  # [B,R,N]
        alphas = ((prev_cdf - next_cdf) / (prev_cdf + eps)).clip_(0.0, 1.0)  # [B,R,N]
        # weights = render.alpha_compositing_weights(alphas)  # [B,R,N,1]
        return alphas

    def _get_iter_cos(self, true_cos, progress=1.):
        anneal_ratio = min(progress / self.anneal_end, 1.)
        # The anneal strategy below keeps the cos value alive at the beginning of training iterations.
        return -((-true_cos * 0.5 + 0.5).relu() * (1.0 - anneal_ratio) +
                 (-true_cos).relu() * anneal_ratio)  # always non-positive
