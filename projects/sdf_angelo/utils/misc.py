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
import numpy as np
import torch
import torch.nn.functional as torch_F
import imaginaire.trainers.utils
from torch.optim import lr_scheduler

flip_mat = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])


def get_scheduler(cfg_opt, opt):
    """Return the scheduler object.

    Args:
        cfg_opt (obj): Config for the specific optimization module (gen/dis).
        opt (obj): PyTorch optimizer object.

    Returns:
        (obj): Scheduler
    """
    if cfg_opt.sched.type == 'two_steps_with_warmup':
        warm_up_end = cfg_opt.sched.warm_up_end
        two_steps = cfg_opt.sched.two_steps
        gamma = cfg_opt.sched.gamma

        def sch(x):
            if x < warm_up_end:
                return x / warm_up_end
            else:
                if x > two_steps[1]:
                    return 1.0 / gamma ** 2
                elif x > two_steps[0]:
                    return 1.0 / gamma
                else:
                    return 1.0

        scheduler = lr_scheduler.LambdaLR(opt, lambda x: sch(x))
    elif cfg_opt.sched.type == 'cos_with_warmup':
        alpha = cfg_opt.sched.alpha
        max_iter = cfg_opt.sched.max_iter
        warm_up_end = cfg_opt.sched.warm_up_end

        def sch(x):
            if x < warm_up_end:
                return x / warm_up_end
            else:
                progress = (x - warm_up_end) / (max_iter - warm_up_end)
                learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
                return learning_factor

        scheduler = lr_scheduler.LambdaLR(opt, lambda x: sch(x))
    else:
        return imaginaire.trainers.utils.get_scheduler()
    return scheduler


def eikonal_loss(gradients, outside=None):
    gradient_error = (gradients.norm(dim=-1) - 1.0) ** 2  # [B,R,N]
    gradient_error = gradient_error.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # [B,R,N]
    if outside is not None and outside.shape == gradient_error.shape:
        return (gradient_error * (~outside).float()).mean()
    else:
        return gradient_error.mean()


def curvature_loss(hessian, outside=None):
    laplacian = hessian.sum(dim=-1).abs()  # [B,R,N]
    laplacian = laplacian.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # [B,R,N]
    if outside is not None:
        return (laplacian * (~outside).float()).mean()
    else:
        return laplacian.mean()


def _sample_reg_points(model, batch_size, device, num_points=64, need_hessian=False):
    """Uniform samples in unit sphere for regularization."""
    dir = torch.randn(batch_size, num_points, 3, device=device)
    dir = torch_F.normalize(dir, dim=-1)
    radii = torch.rand(batch_size, num_points, 1, device=device) ** (1.0 / 3.0)
    pts = dir * radii  # [B,N,3]
    sdf = model.neural_sdf.sdf(pts)  # [B,N,1]
    grads, hess = model.neural_sdf.compute_gradients(pts, training=need_hessian, sdf=sdf)
    return grads, hess


def eikonal_loss_total(data, model, num_rand=64, rand_weight=0.5):
    """Eikonal on rendered samples plus random global samples."""
    loss_main = eikonal_loss(data["gradients"], outside=data.get("outside", None))
    grads_rand, _ = _sample_reg_points(model, batch_size=data["rgb"].shape[0], device=data["rgb"].device,
                                       num_points=num_rand, need_hessian=False)
    loss_rand = eikonal_loss(grads_rand)
    return loss_main + loss_rand * rand_weight


def curvature_loss_total(data, model, num_rand=32, rand_weight=0.5):
    """Curvature on rendered samples plus random global samples."""
    loss_main = curvature_loss(data["hessians"], outside=data.get("outside", None))
    _, hess_rand = _sample_reg_points(model, batch_size=data["rgb"].shape[0], device=data["rgb"].device,
                                      num_points=num_rand, need_hessian=True)
    if hess_rand is not None:
        loss_main = loss_main + curvature_loss(hess_rand) * rand_weight
    return loss_main


def sdf_shift_loss_continuous(
    data, model,
    base_eps=1e-3,
    step_min=1e-3,
    step_max=5e-3,
    temp=0.01,
    consistency_weight=0.5,
):
    if "surf_pts" not in data:
        return torch.tensor(0.0, device=data["rgb"].device, dtype=data["rgb"].dtype)

    mask = data.get("hit_mask", data["opacity"].squeeze(-1) > 0)  # [B,R]
    if not mask.any():
        return torch.tensor(0.0, device=data["rgb"].device, dtype=data["rgb"].dtype)

    rgb    = data["rgb"]           # 必须与 surf_pts 在同一计算图里才有意义
    rgb_gt = data["image_sampled"]
    surf_pts = data["surf_pts"]

    # 注意：仅当 rgb 是由这个 surf_pts 计算出来的，这里才会得到非零梯度
    # if not surf_pts.requires_grad:
    #     surf_pts = surf_pts.detach().requires_grad_(True)

    mask_f = mask.float()[..., None]
    color_err = ((rgb - rgb_gt) * mask_f).pow(2).sum()

    grad_x = torch.autograd.grad(
        color_err, surf_pts,
        create_graph=False,   # 避免二阶
        retain_graph=True,
        allow_unused=True
    )[0]
    if grad_x is None:
        return torch.tensor(0.0, device=rgb.device, dtype=rgb.dtype)

    # 用方向做路由信号：detach
    dir = torch_F.normalize(grad_x, dim=-1).detach()

    # step 的 clamp 建议作用在 step 本身
    step = ((rgb - rgb_gt).detach().norm(dim=-1, keepdim=True) * base_eps).clamp(step_min, step_max)

    pts_offset = surf_pts - step * dir   # 梯度下降方向

    sdf_offset = model.neural_sdf.sdf(pts_offset)[..., 0]  # [B,R]
    m = mask.float()
    den = m.sum().clamp_min(1.0)
    loss_shift = (sdf_offset.abs() * m).sum() / den

    if "sdf_surface" in data:
        sdf_surface = data["sdf_surface"]
        consistency = ((sdf_surface - sdf_offset.detach()) ** 2 + (sdf_offset - sdf_surface.detach()) ** 2)
        loss_shift = loss_shift + (consistency * m).sum() / den * consistency_weight

    return loss_shift



def sdf_shift_loss(sdf_offsets, rgb_offsets, rgb_target, rgb_center):
    """
    Pick the best match among center and lateral samples (up/down/left/right) based on RGB,
    and encourage its SDF to be near zero (center assumed zero).
    Args:
        sdf_offsets (tensor [B,R,4]): SDF at offsets [+u,-u,+v,-v].
        rgb_offsets (tensor [B,R,4,3]): RGB at offsets.
        rgb_target (tensor [B,R,3]): Ground-truth RGB at the center pixel.
        rgb_center (tensor [B,R,3]): Rendered RGB at the center sample.
    Returns:
        scalar loss.
    """
    # Color distance: center + 4 offsets.
    color_dist_center = (rgb_center - rgb_target).abs().mean(dim=-1, keepdim=True)  # [B,R,1]
    color_dist_offsets = (rgb_offsets - rgb_target[..., None, :]).abs().mean(dim=-1)  # [B,R,4]
    color_dist = torch.cat([color_dist_center, color_dist_offsets], dim=-1)  # [B,R,5]
    best_idx = color_dist.argmin(dim=-1, keepdim=True)  # [B,R,1]
    # SDF list: center assumed 0, offsets as provided.
    sdf_center = torch.zeros_like(sdf_offsets[..., :1])  # [B,R,1]
    sdf_all = torch.cat([sdf_center, sdf_offsets], dim=-1)  # [B,R,5]
    sdf_best = sdf_all.gather(dim=-1, index=best_idx).squeeze(-1)  # [B,R]
    loss = sdf_best.abs().mean()
    return loss + color_dist.mean()

def sdf_shift_loss_old(sdf_front, sdf_back, rgb_front, rgb_back, image_sampled):

    # 用颜色误差判断哪个更“像表面”（注意 detach：只用来做路由，不让它自己学崩）
    err_f = (rgb_front - image_sampled).pow(2).mean(dim=-1)  # [Nh]
    err_b = (rgb_back - image_sampled).pow(2).mean(dim=-1)  # [Nh]

    # 硬选择：front 更好则选 front，否则选 back
    choose_front = (err_f < err_b).float().detach()  # [Nh]

    # 让更好的那个点的 |sdf| -> 0，相当于把 0 面往那边拉
    loss_shift = (choose_front * sdf_front.abs() + (1.0 - choose_front) * sdf_back.abs()).mean()
    return loss_shift + err_f.mean() + err_b.mean()


def get_activation(activ, **kwargs):
    func = dict(
        identity=lambda x: x,
        relu=torch_F.relu,
        relu_=torch_F.relu_,
        abs=torch.abs,
        abs_=torch.abs_,
        sigmoid=torch.sigmoid,
        sigmoid_=torch.sigmoid_,
        exp=torch.exp,
        exp_=torch.exp_,
        softplus=torch_F.softplus,
        silu=torch_F.silu,
        silu_=partial(torch_F.silu, inplace=True),
    )[activ]
    return partial(func, **kwargs)


def to_full_image(image, image_size=None, from_vec=True):
    # if from_vec is True: [B,HW,...,K] --> [B,K,H,W,...]
    # if from_vec is False: [B,H,W,...,K] --> [B,K,H,W,...]
    if from_vec:
        assert image_size is not None
        image = image.unflatten(dim=1, sizes=image_size)
    image = image.moveaxis(-1, 1)
    return image
