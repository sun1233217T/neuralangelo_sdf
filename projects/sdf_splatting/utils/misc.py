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

from mtools import debug

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
    if outside is not None:
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


def neighbor_shift_loss(rgb_render, rgb_target, sdf_net, target_points, shift_inter=5):
    """ Shift Loss for SDF Splatting

    Args:
        rgb_render: [B,R,3]
        rgb_target: [B,R,3]
        target_points: [B,R,3]
        shift_inter: interval for shift
    Returns:
        loss value
    """
    shift_inter = int(shift_inter)

    # Use only base positions that have all shift candidates available.
    lengths = [rgb_render[:, i::shift_inter, :].shape[1] for i in range(shift_inter)]
    num_base = min(lengths) if lengths else 0
    if num_base == 0:
        return torch.zeros((), device=rgb_render.device, dtype=rgb_render.dtype)
    target_base_rgb = rgb_target[:, ::shift_inter, :][:, :num_base, :]  # [B,Nb,3]
    candidates = []
    for i in range(shift_inter):
        cur_rgb = rgb_render[:, i::shift_inter, :][:, :num_base, :]
        candidates.append(cur_rgb)
    candidates = torch.stack(candidates, dim=0)  # [K,B,Nb,3]
    diff = (candidates - target_base_rgb.unsqueeze(0)).abs().mean(dim=-1)  # [K,B,Nb]
    offset = diff.argmin(dim=0)  # [B,Nb]

    base_idx = torch.arange(0, num_base * shift_inter, shift_inter, device=rgb_render.device)
    idx = base_idx.view(1, -1) + offset  # [B,Nb]
    target_base_points = torch.gather(target_points, 1, idx[..., None].expand(-1, -1, 3))  # [B,Nb,3]


    sdf_out = sdf_net(target_base_points.reshape(-1, 3))
    if isinstance(sdf_out, (tuple, list)):
        sdf_out = sdf_out[0]
    target_base_points_sdf = sdf_out.view(*target_base_points.shape[:-1], 1)  # [B,R,Nb,1]
    loss = target_base_points_sdf.abs().mean()
    return loss
