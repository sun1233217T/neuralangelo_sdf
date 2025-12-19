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

from imaginaire.models.base import Model as BaseModel
from projects.nerf.utils import nerf_util, camera
from projects.sdf_GS_gengeration.utils.modules import NeuralSDF, NeuralGS
from projects.sdf_GS_gengeration.utils.gs_render import GaussianModel, Camera, render_analyse


class Model(BaseModel):

    def __init__(self, cfg_model, cfg_data):
        super().__init__(cfg_model, cfg_data)
        self.white_background = cfg_model.background.white
        # Define models.
        self.build_model(cfg_model, cfg_data)

    def build_model(self, cfg_model, cfg_data):
        self.neural_sdf = NeuralSDF(cfg_model.object.sdf)
        self.neural_gs = NeuralGS(cfg_model.object.rgb, feat_dim=cfg_model.object.sdf.mlp.hidden_dim,
                                    appear_embed=cfg_model.appear_embed, output_dim=10)

    def forward(self, data):
        output = self.render_gaussian_image(data, training=True)
        return output

    @torch.no_grad()
    def inference(self, data):
        self.eval()
        output = self.render_gaussian_image(data, training=False)
        return output

    def render_gaussian_image(self, data, training=True):
        """Generate Gaussians on traced surface points and render full image with Gaussian rasterizer."""
        pose = data["pose"]  # [B,3,4] (w2c)
        intr = data["intr"]  # [B,3,3]
        H, W = self._get_image_hw(data)
        image_size = (H, W)

        # Generate rays for full image.
        center, ray = camera.get_center_and_ray(pose, intr, image_size)  # [B,HW,3]
        ray_unit = torch_F.normalize(ray, dim=-1)  # [B,HW,3]
        ray_norm = ray.norm(dim=-1, keepdim=False)  # [B,HW]

        # Trace to surface to obtain per-pixel 3D position.
        with torch.no_grad():
            near, far, outside = self.get_dist_bounds(center, ray_unit)
            surface_points, hit_mask, t_vals = self.det_surface(center, ray_unit, near, far)

        # Run SDF network on surface points.
        points = surface_points[..., None, :]  # [B,HW,1,3]
        points = torch.nan_to_num(points, nan=0.0)
        sdfs, feats = self.neural_sdf.forward(points)
        gradients, hessians = self.neural_sdf.compute_gradients(points, training=training, sdf=sdfs)
        normals = torch_F.normalize(gradients, dim=-1)
        rays_unit_exp = ray_unit[..., None, :].expand_as(points)

        # NeuralGS outputs: [rgb(3), quat(4), scale(3)].
        gauss_params = self.neural_gs.forward(points, normals, rays_unit_exp, feats, app=None)
        colors = gauss_params[..., :3]  # already sigmoid
        quat = gauss_params[..., 3:7]
        scales = gauss_params[..., 7:]

        quat_norm = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)
        rot_mats = self.quaternion_to_matrix(quat_norm)  # [B,HW,1,3,3]
        scales = torch_F.softplus(scales) + 1e-4

        # Flatten per batch for rasterizer.
        B = pose.shape[0]
        rendered_images = []
        for b in range(B):
            hit = hit_mask[b].view(-1)  # [HW]
            if hit.sum() == 0:
                bg = torch.ones(3, H, W, device=pose.device) if self.white_background else torch.zeros(3, H, W, device=pose.device)
                rendered_images.append(bg)
                continue
            pts_b = points[b].view(-1, 3)[hit]  # [N,3]
            color_b = colors[b].view(-1, 3)[hit]  # [N,3]
            rot_b = rot_mats[b].view(-1, 3, 3)[hit]  # [N,3,3]
            scale_b = scales[b].view(-1, 3)[hit]  # [N,3]

            pc = GaussianModel(pts_b, opacity=None, scaling=scale_b, rotation=rot_b, color=color_b)
            cam = self.build_camera_from_data(data, b)
            bg_color = torch.ones(3, device=pts_b.device) if self.white_background else torch.zeros(3, device=pts_b.device)
            render_out = render_analyse(cam, pc, bg_color=bg_color)
            img = render_out["render"]  # [3,H,W]
            rendered_images.append(img)

        rgb_map = torch.stack(rendered_images, dim=0)  # [B,3,H,W]
        # Depth and normal maps from SDF tracing.
        depth = (t_vals / (ray_norm + 1e-8)) * hit_mask  # [B,HW]
        depth_map = depth.view(B, H, W).unsqueeze(1)
        normal_flat = normals.squeeze(2) * hit_mask[..., None]  # [B,HW,3]
        normal_map = normal_flat.view(B, H, W, 3).permute(0, 3, 1, 2)
        hit_mask_map = hit_mask.view(B, H, W).unsqueeze(1).float()

        output = dict(
            rgb_map=rgb_map,
            depth_map=depth_map,
            normal_map=normal_map,
            hit_mask=hit_mask_map,
            gradients=gradients.squeeze(2),
            hessians=hessians.squeeze(2) if (training and hessians is not None) else None,
            outside=outside,
            surface_points=surface_points,
            t_vals=t_vals,
        )
        return output

    def _get_image_hw(self, data):
        h = data["image_height"]
        w = data["image_width"]
        if torch.is_tensor(h):
            h_val = int(h[0].item())
            w_val = int(w[0].item())
        else:
            h_val = int(h)
            w_val = int(w)
        return h_val, w_val

    def build_camera_from_data(self, data, b_idx):
        h_field = data["image_height"]
        w_field = data["image_width"]
        fovx_field = data["FoVx"]
        fovy_field = data["FoVy"]
        H = int(h_field[b_idx].item()) if torch.is_tensor(h_field) else int(h_field)
        W = int(w_field[b_idx].item()) if torch.is_tensor(w_field) else int(w_field)
        FoVx = float(fovx_field[b_idx].item()) if torch.is_tensor(fovx_field) else float(fovx_field)
        FoVy = float(fovy_field[b_idx].item()) if torch.is_tensor(fovy_field) else float(fovy_field)
        world_view = data["world_view_transform"][b_idx].to(self.device()).contiguous()
        full_proj = data["full_proj_transform"][b_idx].to(self.device()).contiguous()
        cam_center = data["camera_center"][b_idx].to(self.device())
        return Camera(
            image_height=H,
            image_width=W,
            FoVx=FoVx,
            FoVy=FoVy,
            world_view_transform=world_view,
            full_proj_transform=full_proj,
            camera_center=cam_center,
        )

    def quaternion_to_matrix(self, quat):
        """Convert normalized quaternion [...,4] to rotation matrix [...,3,3]."""
        # quat: (..., 4) in (w, x, y, z) order
        w, x, y, z = quat.unbind(-1)
        ww, xx, yy, zz = w * w, x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
        row1 = torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1)
        row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)
        return torch.stack([row0, row1, row2], dim=-2)

    @torch.no_grad()
    def get_dist_bounds(self, center, ray_unit):
        dist_near, dist_far = nerf_util.intersect_with_sphere(center, ray_unit, radius=1.)
        dist_near.relu_()  # Distance (and thus depth) should be non-negative.
        outside = dist_near.isnan()
        dist_near[outside], dist_far[outside] = 1, 1.2  # Dummy distances. Density will be set to 0.
        return dist_near, dist_far, outside
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
