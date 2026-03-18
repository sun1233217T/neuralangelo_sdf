from dataclasses import replace

import numpy as np
import torch

from projects.sdf_angelo.uv_distill.common import CameraRecord, c2w_to_pose, pose_to_c2w


def _matrix_to_quaternion(R):
    m = R
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = torch.stack([qw, qx, qy, qz])
    return q / q.norm().clamp_min(1e-8)


def _quaternion_to_matrix(q):
    q = q / q.norm().clamp_min(1e-8)
    w, x, y, z = q
    row0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)])
    row1 = torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)])
    row2 = torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)])
    return torch.stack([row0, row1, row2], dim=0).to(dtype=q.dtype)


def _slerp(q0, q1, t):
    q0 = q0 / q0.norm().clamp_min(1e-8)
    q1 = q1 / q1.norm().clamp_min(1e-8)
    dot = (q0 * q1).sum()
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / q.norm().clamp_min(1e-8)
    theta_0 = torch.acos(dot.clamp(-1.0, 1.0))
    sin_theta_0 = torch.sin(theta_0).clamp_min(1e-8)
    theta = theta_0 * t
    sin_theta = torch.sin(theta)
    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


def _random_axis_angle(max_angle_deg, generator):
    angle = torch.rand(1, generator=generator).item() * np.deg2rad(max_angle_deg)
    axis = torch.randn(3, generator=generator)
    axis = axis / axis.norm().clamp_min(1e-8)
    half = angle * 0.5
    half_t = torch.tensor([half], dtype=torch.float32)
    return torch.cat([torch.cos(half_t), axis.to(dtype=torch.float32) * torch.sin(half_t)])


def sample_augmented_views(original_records, sphere_radius,
                           jitter_per_view=0, jitter_translation_ratio=0.01,
                           jitter_rotation_deg=2.0, interp_steps=0, seed=0):
    if not original_records:
        return []
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    augmented = []
    next_index = len(original_records)
    world_radius = float(sphere_radius)
    for record in original_records:
        if jitter_per_view <= 0:
            continue
        c2w = pose_to_c2w(record.pose)
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        for _ in range(int(jitter_per_view)):
            local_offset = torch.randn(3, generator=generator, dtype=torch.float32) * float(jitter_translation_ratio)
            world_offset = R @ local_offset
            dq = _random_axis_angle(float(jitter_rotation_deg), generator)
            R_j = _quaternion_to_matrix(_slerp(torch.tensor([1.0, 0.0, 0.0, 0.0]), dq, 1.0))
            c2w_new = c2w.clone()
            c2w_new[:3, :3] = R @ R_j
            c2w_new[:3, 3] = t + world_offset
            pose_new = c2w_to_pose(c2w_new)
            center_norm = c2w_new[:3, 3]
            augmented.append(
                CameraRecord(
                    index=next_index,
                    kind="jitter",
                    intr=record.intr.clone(),
                    pose=pose_new,
                    image_size=record.image_size,
                    camera_center_norm=center_norm.clone(),
                    camera_center_world=record.camera_center_world.clone() + world_offset * world_radius,
                    source_indices=record.source_indices,
                )
            )
            next_index += 1
    if interp_steps > 0:
        for i in range(len(original_records) - 1):
            rec0 = original_records[i]
            rec1 = original_records[i + 1]
            c2w0 = pose_to_c2w(rec0.pose)
            c2w1 = pose_to_c2w(rec1.pose)
            q0 = _matrix_to_quaternion(c2w0[:3, :3])
            q1 = _matrix_to_quaternion(c2w1[:3, :3])
            for step in range(1, int(interp_steps) + 1):
                alpha = step / float(interp_steps + 1)
                q = _slerp(q0, q1, alpha)
                R = _quaternion_to_matrix(q)
                center_norm = (1.0 - alpha) * c2w0[:3, 3] + alpha * c2w1[:3, 3]
                c2w_new = torch.eye(4, dtype=c2w0.dtype)
                c2w_new[:3, :3] = R
                c2w_new[:3, 3] = center_norm
                intr = (1.0 - alpha) * rec0.intr + alpha * rec1.intr
                center_world = (1.0 - alpha) * rec0.camera_center_world + alpha * rec1.camera_center_world
                augmented.append(
                    CameraRecord(
                        index=next_index,
                        kind="interp",
                        intr=intr,
                        pose=c2w_to_pose(c2w_new),
                        image_size=rec0.image_size,
                        camera_center_norm=center_norm.clone(),
                        camera_center_world=center_world.clone(),
                        source_indices=(rec0.index, rec1.index),
                    )
                )
                next_index += 1
    return augmented
