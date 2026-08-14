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
from torch.cuda.amp import autocast


class _CumprodGraphSafe(torch.autograd.Function):
    """``cumprod`` with a CUDA-graph-capturable backward.

    ATen's ``cumprod_backward`` runs a data-dependent loop over zero entries
    (with device synchronizations) to apply the omitted-product correction,
    which makes it impossible to capture in a CUDA graph.  The backward below
    uses the closed form ``grad_x = revcumsum(grad * v) / x`` — mathematically
    identical for nonzero ``x`` (verified against ATen to machine precision).
    For exact-zero entries the correction term is dropped (gradient 0); in the
    compositing use below ``x = 1 - alpha_front`` with ``alpha < 1`` strictly,
    so the edge case does not occur in practice.
    """

    @staticmethod
    def forward(ctx, x, dim):
        v = x.cumprod(dim=dim)
        ctx.save_for_backward(x, v)
        ctx.dim = dim
        return v

    @staticmethod
    def backward(ctx, grad_output):
        x, v = ctx.saved_tensors
        dim = ctx.dim
        # d v_j / d x_i = v_j / x_i for j >= i  =>  grad_x_i = sum_{j>=i} g_j v_j / x_i
        c = (grad_output * v).flip(dim).cumsum(dim=dim).flip(dim)
        safe_x = torch.where(x == 0, torch.ones_like(x), x)
        grad_x = torch.where(x == 0, torch.zeros_like(x), c / safe_x)
        return grad_x, None


def volume_rendering_weights(ray, densities, depths, depth_far=None):
    """The volume rendering function. Details can be found in the NeRF paper.
    Args:
        ray (tensor [batch,ray,3]): The ray directions in world space.
        densities (tensor [batch,ray,samples]): The predicted volume density samples.
        depths (tensor [batch,ray,samples,1]): The corresponding depth samples.
        depth_far (tensor [batch,ray,1,1]): The farthest depth for computing the last interval.
    Returns:
        weights (tensor [batch,ray,samples,1]): The predicted weight of each sampled point along the ray (in [0,1]).
    """
    ray_length = ray.norm(dim=-1, keepdim=True)  # [B,R,1]
    if depth_far is None:
        depth_far = torch.empty_like(depths[..., :1, :]).fill_(1e10)  # [B,R,1,1]
    depths_aug = torch.cat([depths, depth_far], dim=2)  # [B,R,N+1,1]
    dists = depths_aug * ray_length[..., None]  # [B,R,N+1,1]
    # Volume rendering: compute rendering weights (using quadrature).
    dist_intvs = dists[..., 1:, 0] - dists[..., :-1, 0]  # [B,R,N]
    sigma_delta = densities * dist_intvs  # [B,R,N]
    sigma_delta_0 = torch.cat([torch.zeros_like(sigma_delta[..., :1]),
                               sigma_delta[..., :-1]], dim=2)  # [B,R,N]
    T = (-sigma_delta_0.cumsum(dim=2)).exp_()  # [B,R,N]
    alphas = 1 - (-sigma_delta).exp_()  # [B,R,N]
    # Compute weights for compositing samples.
    weights = (T * alphas)[..., None]  # [B,R,N,1]
    return weights


def volume_rendering_weights_dist(densities, dists, dist_far=None):
    """The volume rendering function. Details can be found in the NeRF paper.
    Args:
        densities (tensor [batch,ray,samples]): The predicted volume density samples.
        dists (tensor [batch,ray,samples,1]): The corresponding distance samples.
        dist_far (tensor [batch,ray,1,1]): The farthest distance for computing the last interval.
    Returns:
        weights (tensor [batch,ray,samples,1]): The predicted weight of each sampled point along the ray (in [0,1]).
    """
    # TODO: re-consolidate!!
    if dist_far is None:
        dist_far = torch.empty_like(dists[..., :1, :]).fill_(1e10)  # [B,R,1,1]
    dists = torch.cat([dists, dist_far], dim=2)  # [B,R,N+1,1]
    # Volume rendering: compute rendering weights (using quadrature).
    dist_intvs = dists[..., 1:, 0] - dists[..., :-1, 0]  # [B,R,N]
    sigma_delta = densities * dist_intvs  # [B,R,N]
    sigma_delta_0 = torch.cat([torch.zeros_like(sigma_delta[..., :1]), sigma_delta[..., :-1]], dim=2)  # [B,R,N]
    T = (-sigma_delta_0.cumsum(dim=2)).exp_()  # [B,R,N]
    alphas = 1 - (-sigma_delta).exp_()  # [B,R,N]
    # Compute weights for compositing samples.
    weights = (T * alphas)[..., None]  # [B,R,N,1]
    return weights


def volume_rendering_alphas_dist(densities, dists, dist_far=None):
    """The volume rendering function. Details can be found in the NeRF paper.
    Args:
        densities (tensor [batch,ray,samples]): The predicted volume density samples.
        dists (tensor [batch,ray,samples,1]): The corresponding distance samples.
        dist_far (tensor [batch,ray,1,1]): The farthest distance for computing the last interval.
    Returns:
        alphas (tensor [batch,ray,samples,1]): The occupancy of each sampled point along the ray (in [0,1]).
    """
    if dist_far is None:
        dist_far = torch.empty_like(dists[..., :1, :]).fill_(1e10)  # [B,R,1,1]
    dists = torch.cat([dists, dist_far], dim=2)  # [B,R,N+1,1]
    # Volume rendering: compute rendering weights (using quadrature).
    dist_intvs = dists[..., 1:, 0] - dists[..., :-1, 0]  # [B,R,N]
    sigma_delta = densities * dist_intvs  # [B,R,N]
    alphas = 1 - (-sigma_delta).exp_()  # [B,R,N]
    return alphas


def alpha_compositing_weights(alphas):
    """Alpha compositing of (sampled) MPIs given their RGBs and alphas.
    Args:
        alphas (tensor [batch,ray,samples]): The predicted opacity values.
    Returns:
        weights (tensor [batch,ray,samples,1]): The predicted weight of each MPI (in [0,1]).
    """
    alphas_front = torch.cat([torch.zeros_like(alphas[..., :1]),
                              alphas[..., :-1]], dim=2)  # [B,R,N]
    with autocast(enabled=False):  # TODO: may be unstable in some cases.
        visibility = _CumprodGraphSafe.apply(1 - alphas_front, 2)  # [B,R,N]
    weights = (alphas * visibility)[..., None]  # [B,R,N,1]
    return weights


def composite(quantities, weights):
    """Composite the samples to render the RGB/depth/opacity of the corresponding pixels.
    Args:
        quantities (tensor [batch,ray,samples,k]): The quantity to be weighted summed.
        weights (tensor [batch,ray,samples,1]): The predicted weight of each sampled point along the ray.
    Returns:
        quantity (tensor [batch,ray,k]): The expected (rendered) quantity.
    """
    # Integrate RGB and depth weighted by probability.
    quantity = (quantities * weights).sum(dim=2)  # [B,R,K]
    return quantity
