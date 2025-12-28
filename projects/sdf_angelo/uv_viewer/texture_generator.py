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
import queue
import threading
import time
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as torch_F
from PIL import Image


class TextureGenerator:

    def __init__(self, uv_cache, neural_rgb, appear_embed, sphere_center, sphere_radius,
                 update_fps=1.0, batch_size=65536, appear_idx=None,
                 encode_base64=True, include_raw=False, async_encode=True,
                 encode_queue_size=1):
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
        self.encode_base64 = bool(encode_base64)
        self.include_raw = bool(include_raw)
        if not self.encode_base64 and not self.include_raw:
            raise ValueError("At least one of encode_base64/include_raw must be True.")
        self.async_encode = bool(async_encode)
        self.encode_queue_size = max(int(encode_queue_size), 1)
        self.appear_idx = appear_idx
        self._app_value = self._init_app_value()

        self._lock = threading.Lock()
        self._texture_b64 = None
        self._texture_bytes = None
        self._texture_ready = threading.Event()
        self._stop = threading.Event()
        self._camera_lock = threading.Lock()
        self._camera_pos_world = None
        self._thread = None
        self._encode_thread = None
        self._encode_queue = queue.Queue(maxsize=self.encode_queue_size) if self.async_encode else None
        self._stats_lock = threading.Lock()
        self._update_count = 0
        self._last_update_time = None
        self._last_update_fps = 0.0
        self._last_render_ms = 0.0
        self._last_encode_ms = 0.0
        self._last_total_ms = 0.0

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
        if self.async_encode and self._encode_thread is None:
            self._encode_thread = threading.Thread(target=self._encode_loop, daemon=True)
            self._encode_thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._encode_thread is not None:
            self._encode_thread.join(timeout=1.0)

    def update_camera(self, cam_pos_world):
        with self._camera_lock:
            self._camera_pos_world = np.asarray(cam_pos_world, dtype=np.float32)

    def get_texture_b64(self):
        if not self._texture_ready.wait(timeout=0.1):
            return None
        with self._lock:
            return self._texture_b64

    def get_texture_bytes(self):
        if not self._texture_ready.wait(timeout=0.1):
            return None
        with self._lock:
            return self._texture_bytes

    def get_stats(self):
        with self._stats_lock:
            return {
                "updates": self._update_count,
                "update_fps": self._last_update_fps,
                "render_ms": self._last_render_ms,
                "encode_ms": self._last_encode_ms,
                "total_ms": self._last_total_ms,
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
            if self.async_encode:
                self._enqueue_texture(texture, render_ms)
            else:
                self._encode_and_store(texture, render_ms)
            next_time = time.time() + self.update_interval

    def _enqueue_texture(self, texture, render_ms):
        if self._encode_queue is None:
            return
        try:
            self._encode_queue.put_nowait((texture, render_ms))
        except queue.Full:
            try:
                _ = self._encode_queue.get_nowait()
                self._encode_queue.task_done()
            except queue.Empty:
                pass
            try:
                self._encode_queue.put_nowait((texture, render_ms))
            except queue.Full:
                pass

    def _encode_loop(self):
        if self._encode_queue is None:
            return
        while not self._stop.is_set() or not self._encode_queue.empty():
            try:
                texture, render_ms = self._encode_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._encode_and_store(texture, render_ms)
            self._encode_queue.task_done()

    def _encode_and_store(self, texture, render_ms):
        encode_ms = 0.0
        tex_b64 = None
        tex_bytes = None
        if self.include_raw:
            pack_start = time.time()
            tex_bytes = _pack_texture_rgba(texture)
            encode_ms += (time.time() - pack_start) * 1000.0
        if self.encode_base64:
            encode_start = time.time()
            tex_b64 = _encode_texture_b64(texture)
            encode_ms += (time.time() - encode_start) * 1000.0
        total_ms = render_ms + encode_ms
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
            self._last_encode_ms = encode_ms
            self._last_total_ms = total_ms
        with self._lock:
            self._texture_b64 = tex_b64
            self._texture_bytes = tex_bytes
            self._texture_ready.set()

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


def _pack_texture_rgba(texture):
    if not isinstance(texture, np.ndarray):
        raise ValueError("Texture must be a numpy array.")
    if texture.ndim != 3 or texture.shape[2] != 3:
        raise ValueError("Texture must have shape [H,W,3].")
    height, width, _ = texture.shape
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = texture
    rgba[..., 3] = 255
    return rgba.tobytes()
