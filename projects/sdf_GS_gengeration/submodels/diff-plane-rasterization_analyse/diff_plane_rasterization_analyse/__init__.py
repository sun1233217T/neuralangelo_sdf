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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C
import math

def s_curve(x: torch.Tensor, x0 = 0.8, k = 10.0) -> torch.Tensor:
    """S曲线函数，用于将输入张量映射到 [0, 1] 区间。
    参数:
        x (torch.Tensor): 输入张量，范围为 [0, 1]。
        x0 (float): 拐点位置，范围为 (0, 1)。
        k (float): 控制曲线陡峭程度的参数，k > 0。

    返回:
        torch.Tensor: 输出张量，范围为 [0, 1]。
    """
    # 计算标准Logistic函数
    logistic = 1 / (1 + torch.exp(-k * (x - x0)))
    
    # 计算边界值
    logistic_0 = 1 / (1 + math.exp(k * x0))
    logistic_1 = 1 / (1 + math.exp(-k * (1 - x0)))
    
    # 线性变换，使 f(0) = 0, f(1) = 1
    f_x = (logistic - logistic_0) / (logistic_1 - logistic_0)
    
    return f_x

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    means2D_abs,
    sh,
    colors_precomp,
    opacities,
    H_app_opacity,
    scales,
    rotations,
    cov3Ds_precomp,
    all_map,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        means2D_abs,
        sh,
        colors_precomp,
        opacities,
        H_app_opacity,
        scales,
        rotations,
        cov3Ds_precomp,
        all_map,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        means2D_abs,
        sh,
        colors_precomp,
        opacities,
        H_app_opacity,
        scales,
        rotations,
        cov3Ds_precomp,
        all_maps,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            all_maps,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.render_geo,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, out_observe, out_all_map, geomBuffer, binningBuffer, imgBuffer, app_opacity, color_alpha = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, out_observe, out_all_map, out_plane_depth, geomBuffer, binningBuffer, imgBuffer, app_opacity, color_alpha = _C.rasterize_gaussians(*args)

        if H_app_opacity is not None:
            H_app_opacity = H_app_opacity * 0.99
            H_app_opacity_mask = app_opacity > H_app_opacity
            H_app_opacity = torch.where(H_app_opacity_mask, app_opacity, H_app_opacity)

            H_app_opacity_mask = s_curve(app_opacity - H_app_opacity)
            H_app_opacity_mask = torch.clamp(H_app_opacity_mask, 0.0, 1.0)
        else:
            H_app_opacity_mask = torch.zeros_like(app_opacity)
            H_app_opacity = torch.zeros_like(app_opacity)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(out_all_map, colors_precomp, all_maps, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, H_app_opacity, H_app_opacity_mask)
        return color, radii, out_observe, out_all_map, out_plane_depth, app_opacity, color_alpha

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_out_observe, grad_out_all_map, grad_out_plane_depth, grad_out_app_opacity, grad_out_color_alpha):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        all_map_pixels, colors_precomp, all_maps, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, H_app_opacity, H_app_opacity_mask = ctx.saved_tensors

        dl_dH_H = grad_out_app_opacity * H_app_opacity * H_app_opacity_mask
        dl_dpixopa = grad_out_color_alpha

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                all_map_pixels,
                means3D, 
                radii, 
                colors_precomp, 
                all_maps,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                grad_out_all_map,
                grad_out_plane_depth,
                dl_dH_H,
                dl_dpixopa,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.render_geo,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_means2D_abs, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, gard_all_map = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_means2D_abs, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, gard_all_map = _C.rasterize_gaussians_backward(*args)
        # print(f"grad_means2D {grad_means2D.sum()}, grad_means2D_abs {grad_means2D_abs.sum()}")

        grads = (
            grad_means3D,
            grad_means2D,
            grad_means2D_abs,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            None,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            gard_all_map,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    render_geo : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, means2D_abs, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, all_map=None, H_app_opacity = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if all_map is None:
            all_map = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            means2D_abs,
            shs,
            colors_precomp,
            opacities,
            H_app_opacity,
            scales, 
            rotations,
            cov3D_precomp,
            all_map,
            raster_settings, 
        )

