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

import base64
import threading
import time
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as torch_F
from PIL import Image


class TextureGenerator:

    def __init__(self, uv_cache, neural_rgb, appear_embed, sphere_center, sphere_radius,
                 update_fps=1.0, batch_size=65536, appear_idx=None):
        self.uv_cache = uv_cache
        self.neural_rgb = neural_rgb
        self.appear_embed = appear_embed
        self.sphere_center = torch.tensor(
            np.asarray(sphere_center, dtype=np.float32),
            device=uv_cache.device,
            dtype=torch.float32,
        )
        self.sphere_radius = float(sphere_radius)
        self.batch_size = int(batch_size)
        update_fps = float(update_fps)
        self.update_interval = 0.0 if update_fps < 0 else 1.0 / max(update_fps, 1e-6)
        self.appear_idx = appear_idx
        self._app_value = self._init_app_value()

        self._lock = threading.Lock()
        self._texture_b64 = None
        self._texture_ready = threading.Event()
        self._stop = threading.Event()
        self._camera_lock = threading.Lock()
        self._camera_pos_world = None
        self._thread = None
        self._stats_lock = threading.Lock()
        self._update_count = 0
        self._last_update_time = None
        self._last_update_fps = 0.0
        self._last_render_ms = 0.0

    def _init_app_value(self):
        if self.appear_embed is None:
            return None
        dim = self.appear_embed.embedding_dim
        device = self.uv_cache.device
        if self.appear_idx is None:
            return torch.zeros((1, dim), device=device)
        idx = torch.tensor([int(self.appear_idx)], device=device)
        return self.appear_embed(idx).detach()

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def update_camera(self, cam_pos_world):
        with self._camera_lock:
            self._camera_pos_world = np.asarray(cam_pos_world, dtype=np.float32)

    def get_texture_b64(self):
        if not self._texture_ready.wait(timeout=0.1):
            return None
        with self._lock:
            return self._texture_b64

    def get_stats(self):
        with self._stats_lock:
            return {
                "updates": self._update_count,
                "update_fps": self._last_update_fps,
                "last_ms": self._last_render_ms,
            }

    def _get_camera(self):
        with self._camera_lock:
            if self._camera_pos_world is None:
                return None
            return self._camera_pos_world.copy()

    def _run(self):
        next_time = time.time()
        while not self._stop.is_set():
            now = time.time()
            if self.update_interval > 0 and now < next_time:
                time.sleep(next_time - now)
            cam_pos = self._get_camera()
            if cam_pos is None:
                if self.update_interval > 0:
                    next_time = time.time() + self.update_interval
                else:
                    time.sleep(0.02)
                continue
            render_start = time.time()
            texture = self._render_texture(cam_pos)
            render_ms = (time.time() - render_start) * 1000.0
            tex_b64 = _encode_texture_b64(texture)
            now = time.time()
            with self._stats_lock:
                self._update_count += 1
                if self._last_update_time is not None:
                    dt = max(now - self._last_update_time, 1e-6)
                    inst_fps = 1.0 / dt
                    if self._last_update_fps == 0.0:
                        self._last_update_fps = inst_fps
                    else:
                        self._last_update_fps = 0.8 * self._last_update_fps + 0.2 * inst_fps
                self._last_update_time = now
                self._last_render_ms = render_ms
            with self._lock:
                self._texture_b64 = tex_b64
                self._texture_ready.set()
            next_time = time.time() + self.update_interval

    def _render_texture(self, cam_pos_world):
        device = self.uv_cache.device
        cam_norm = (torch.from_numpy(cam_pos_world).to(device=device, dtype=torch.float32)
                    - self.sphere_center) / self.sphere_radius
        points = self.uv_cache.points
        normals = self.uv_cache.normals
        feats = self.uv_cache.feats
        coords = self.uv_cache.coords
        rays = points - cam_norm[None, :]
        rays_unit = torch_F.normalize(rays, dim=-1)
        total = points.shape[0]
        rgbs = torch.empty((total, 3), device=device, dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, total, self.batch_size):
                end = min(start + self.batch_size, total)
                app = self._get_app(end - start)
                rgbs[start:end] = self.neural_rgb(
                    points[start:end],
                    normals[start:end],
                    rays_unit[start:end],
                    feats[start:end],
                    app=app,
                )
        texture = torch.zeros((self.uv_cache.tex_size, self.uv_cache.tex_size, 3),
                              device=device, dtype=torch.float32)
        texture[coords[:, 0], coords[:, 1]] = rgbs
        texture = (texture.clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
        return texture

    def _get_app(self, batch_size):
        if self.appear_embed is None:
            return None
        if self._app_value is None:
            return None
        return self._app_value.expand(batch_size, -1)


def _encode_texture_b64(texture):
    if not isinstance(texture, np.ndarray):
        raise ValueError("Texture must be a numpy array.")
    image = Image.fromarray(texture)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
