#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_plane_rasterization_analyse import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization_analyse import GaussianRasterizer as PlaneGaussianRasterizer

# def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
#     # depth: (H, W), bg_color: (3), alpha: (H, W)
#     # normal_ref: (3, H, W)
#     intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
#     st = max(int(scale/2)-1,0)
#     if offset is not None:
#         offset = offset[st::scale,st::scale]
#     normal_ref = normal_from_depth_image(depth[st::scale,st::scale], 
#                                             intrinsic_matrix.to(depth.device), 
#                                             extrinsic_matrix.to(depth.device), offset)

#     normal_ref = normal_ref.permute(2,0,1)
#     return normal_ref
class GaussianModel:
    def __init__(self, xyz, opacity, scaling, rotation, color):
        self.xyz = xyz # (N, 3)
        # self.opacity = opacity # (N, 1) # force to 1
        self.scaling = scaling # (N, 3)
        self.rotation = rotation # (N, 4)
        self.color = color # (N, 3)

class Camera:
    def __init__(self, image_height, image_width, FoVx, FoVy, world_view_transform, full_proj_transform, camera_center):
        self.image_height = image_height
        self.image_width = image_width
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = camera_center

def render_analyse(viewpoint_camera: Camera, pc : GaussianModel, bg_color : torch.Tensor, scaling_modifier = 1.0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.xyz, dtype=pc.xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros_like(pc.xyz, dtype=pc.xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.xyz
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    # opacity = pc.opacity
    opacity = torch.ones((pc.xyz.shape[0], 1), dtype=pc.xyz.dtype, device=pc.xyz.device)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.

    cov3D_precomp = None
    scales = pc.scaling
    rotations = pc.rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = pc.color

    return_dict = None
    raster_settings = PlaneGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=0,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            render_geo=False,
            debug=False
        )

    rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)


    rendered_image, radii, out_observe, _, _, app_opacity, rendered_opacity = rasterizer(
        means3D = means3D,
        means2D = means2D,
        means2D_abs = means2D_abs,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    rendered_opacity = rendered_opacity.clamp(1e-6, 1 - 1e-6)
    
    return_dict =  {"render": rendered_image,
                    "viewspace_points": screenspace_points,
                    "viewspace_points_abs": screenspace_points_abs,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "out_observe": out_observe,
                    "app_opacity" : app_opacity,
                    "render_opacity": rendered_opacity}

    return return_dict
