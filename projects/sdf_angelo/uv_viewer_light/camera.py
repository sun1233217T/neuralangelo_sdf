import math
from dataclasses import dataclass

import numpy as np


def _normalize(vec):
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return vec.copy()
    return vec / norm


@dataclass
class OrbitCamera:
    target: np.ndarray
    distance: float
    yaw: float = 0.0
    pitch: float = 0.0
    fov_y_deg: float = 50.0
    near: float = 0.01
    far: float = 100.0
    world_up: np.ndarray = None

    def __post_init__(self):
        self.target = np.asarray(self.target, dtype=np.float32)
        self.distance = float(max(self.distance, 1e-3))
        self.world_up = np.asarray(
            [0.0, 1.0, 0.0] if self.world_up is None else self.world_up,
            dtype=np.float32,
        )
        self.pitch = float(np.clip(self.pitch, -1.54, 1.54))

    @classmethod
    def from_bounds(cls, bounds):
        bounds = np.asarray(bounds, dtype=np.float32)
        center = bounds.mean(axis=0)
        size = bounds[1] - bounds[0]
        distance = max(float(np.linalg.norm(size)) * 1.2, 1.0)
        far = max(distance * 8.0, 10.0)
        return cls(target=center, distance=distance, near=distance * 0.01, far=far)

    def reset_from_bounds(self, bounds):
        fresh = self.from_bounds(bounds)
        self.target = fresh.target
        self.distance = fresh.distance
        self.yaw = fresh.yaw
        self.pitch = fresh.pitch
        self.fov_y_deg = fresh.fov_y_deg
        self.near = fresh.near
        self.far = fresh.far

    def get_position(self):
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        sy = math.sin(self.yaw)
        cy = math.cos(self.yaw)
        offset = np.array([cp * sy, sp, cp * cy], dtype=np.float32) * self.distance
        return self.target + offset

    def get_forward(self):
        return _normalize(self.target - self.get_position())

    def get_basis(self):
        forward = self.get_forward()
        right = np.cross(forward, self.world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = _normalize(right)
        up = _normalize(np.cross(right, forward))
        return right, up, forward

    def orbit(self, delta_x, delta_y, sensitivity=0.005):
        self.yaw += float(delta_x) * float(sensitivity)
        self.pitch -= float(delta_y) * float(sensitivity)
        self.pitch = float(np.clip(self.pitch, -1.54, 1.54))

    def zoom(self, amount, zoom_scale=0.1):
        factor = 1.0 - float(amount) * float(zoom_scale)
        factor = float(np.clip(factor, 0.1, 10.0))
        self.distance = float(np.clip(self.distance * factor, 0.05, 1e5))
        self.near = min(self.near, self.distance * 0.2)
        self.far = max(self.far, self.distance * 4.0)

    def pan(self, delta_x, delta_y, viewport_size):
        width = max(int(viewport_size[0]), 1)
        height = max(int(viewport_size[1]), 1)
        right, up, _ = self.get_basis()
        view_height = 2.0 * self.distance * math.tan(math.radians(self.fov_y_deg) * 0.5)
        view_width = view_height * (float(width) / float(height))
        move = (
            right * (-float(delta_x) / float(width)) * view_width
            + up * (float(delta_y) / float(height)) * view_height
        )
        self.target = self.target + move.astype(np.float32)
