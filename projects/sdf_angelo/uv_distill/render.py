import os

import torch

from projects.sdf_angelo.utils.mesh import _dilate_texture
from projects.sdf_angelo.uv_distill.model import UVResidualStudent


def load_student_checkpoint(checkpoint_path, texture_size, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = checkpoint.get("args", {})
    model = UVResidualStudent(
        texture_size=texture_size,
        latent_dim=int(train_args.get("latent_dim", 8)),
        latent_scale=int(train_args.get("latent_scale", 4)),
        hidden_dim=int(train_args.get("hidden_dim", 64)),
        num_layers=int(train_args.get("num_layers", 3)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


class StudentTextureRenderer:

    def __init__(self, model, geometry, base_rgb_u8):
        self.model = model
        self.device = next(model.parameters()).device
        self.tex_size = int(geometry["tex_size"])
        self.coords = geometry["coords"].to(device=self.device, dtype=torch.int64)
        self.points = geometry["points"].to(device=self.device, dtype=torch.float32)
        self.normals = geometry["normals"].to(device=self.device, dtype=torch.float32)
        self.all_sparse_idx = torch.arange(self.points.shape[0], device=self.device, dtype=torch.int64)
        if base_rgb_u8.device != self.device:
            base_rgb_u8 = base_rgb_u8.to(self.device, non_blocking=True)
        self.base_sparse_rgb = base_rgb_u8[self.coords[:, 0], self.coords[:, 1]].float() / 255.0
        self.all_uv = torch.stack([
            (self.coords[:, 1].float() + 0.5) / float(self.tex_size),
            (self.coords[:, 0].float() + 0.5) / float(self.tex_size),
        ], dim=-1)
        self.atlas_mask = torch.zeros((self.tex_size, self.tex_size), device=self.device, dtype=torch.bool)
        self.atlas_mask[self.coords[:, 0], self.coords[:, 1]] = True

    def _to_device_tensor(self, tensor, *, dtype=None):
        if tensor.device != self.device:
            tensor = tensor.to(self.device, non_blocking=True)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor

    @torch.no_grad()
    def predict_sparse(self, sparse_idx, uv, base_rgb, camera_center_norm, batch_size=262144):
        sparse_idx = self._to_device_tensor(sparse_idx, dtype=torch.int64)
        uv = self._to_device_tensor(uv, dtype=torch.float32)
        base_rgb = self._to_device_tensor(base_rgb)
        if base_rgb.dtype == torch.uint8:
            base_rgb = base_rgb.float() / 255.0
        else:
            base_rgb = base_rgb.to(dtype=torch.float32)
        camera_center_norm = self._to_device_tensor(camera_center_norm, dtype=torch.float32).view(1, 3)
        total = int(sparse_idx.shape[0])
        pred_rgb = torch.empty((total, 3), device=self.device, dtype=torch.float32)
        for start in range(0, total, int(batch_size)):
            end = min(start + int(batch_size), total)
            batch_idx = sparse_idx[start:end]
            batch_uv = uv[start:end]
            batch_base = base_rgb[start:end]
            batch_points = self.points[batch_idx]
            batch_normals = self.normals[batch_idx]
            batch_cam = camera_center_norm.expand(end - start, -1)
            delta = self.model(batch_uv, batch_points, batch_normals, batch_base, batch_cam)
            pred_rgb[start:end] = (batch_base + delta).clamp(0.0, 1.0)
        return pred_rgb

    @torch.no_grad()
    def render_texture(self, camera_center_norm, batch_size=262144, pad_iters=0):
        sparse_rgb = self.predict_sparse(
            sparse_idx=self.all_sparse_idx,
            uv=self.all_uv,
            base_rgb=self.base_sparse_rgb,
            camera_center_norm=camera_center_norm,
            batch_size=batch_size,
        )
        texture = torch.zeros((self.tex_size, self.tex_size, 3), device=self.device, dtype=torch.float32)
        texture[self.coords[:, 0], self.coords[:, 1]] = sparse_rgb
        if pad_iters > 0:
            texture = _dilate_texture(texture, self.atlas_mask, int(pad_iters))
        return texture


def checkpoint_size_mb(checkpoint_path):
    return float(os.path.getsize(checkpoint_path)) / (1024.0 * 1024.0)
