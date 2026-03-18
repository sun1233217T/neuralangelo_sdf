import math
from dataclasses import dataclass

import numpy as np


def _normalize(vec):
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return vec.copy()
    return vec / norm


def _rotate_axis_angle(vec, axis, angle):
    axis = _normalize(axis)
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    return (
        vec * c
        + np.cross(axis, vec) * s
        + axis * np.dot(axis, vec) * (1.0 - c)
    ).astype(np.float32)


@dataclass
class OrbitCamera:
    target: np.ndarray
    offset: np.ndarray
    up: np.ndarray
    fov_y_deg: float = 45.0
    near: float = 0.01
    far: float = 100.0

    def __post_init__(self):
        self.target = np.asarray(self.target, dtype=np.float32)
        self.offset = np.asarray(self.offset, dtype=np.float32)
        self.up = _normalize(np.asarray(self.up, dtype=np.float32))
        if np.linalg.norm(self.offset) < 1e-6:
            self.offset = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    @classmethod
    def from_bounds(cls, bounds):
        bounds = np.asarray(bounds, dtype=np.float32)
        center = bounds.mean(axis=0)
        size = bounds[1] - bounds[0]
        max_size = max(float(np.max(size)), 1e-6)
        radius = max(max_size * 2.2, 2.0)
        theta = math.pi / 4.0
        phi = math.pi / 2.2
        offset = np.array([
            radius * math.sin(phi) * math.sin(theta),
            radius * math.cos(phi),
            radius * math.sin(phi) * math.cos(theta),
        ], dtype=np.float32)
        far = max(radius * 8.0, 10.0)
        return cls(target=center, offset=offset, up=np.array([0.0, 1.0, 0.0], dtype=np.float32), near=radius * 0.01, far=far)

    def reset_from_bounds(self, bounds):
        fresh = self.from_bounds(bounds)
        self.target = fresh.target
        self.offset = fresh.offset
        self.up = fresh.up
        self.fov_y_deg = fresh.fov_y_deg
        self.near = fresh.near
        self.far = fresh.far

    def set_from_position(self, position):
        position = np.asarray(position, dtype=np.float32)
        offset = position - self.target
        if np.linalg.norm(offset) >= 1e-6:
            self.offset = offset.astype(np.float32)

    def get_position(self):
        return self.target + self.offset

    def get_distance(self):
        return float(np.linalg.norm(self.offset))

    def get_view_direction(self):
        return _normalize(self.offset)

    def get_forward(self):
        return -self.get_view_direction()

    def get_basis(self):
        view_dir = self.get_view_direction()
        right = np.cross(self.up, view_dir)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = _normalize(right)
        up = _normalize(np.cross(view_dir, right))
        return right, up, view_dir

    def project_trackball(self, screen_pos, viewport_size):
        width = max(int(viewport_size[0]), 1)
        height = max(int(viewport_size[1]), 1)
        x = (float(screen_pos[0]) / float(width)) * 2.0 - 1.0
        y = 1.0 - (float(screen_pos[1]) / float(height)) * 2.0
        len2 = x * x + y * y
        if len2 <= 1.0:
            z = math.sqrt(1.0 - len2)
            return np.array([x, y, z], dtype=np.float32)
        norm = 1.0 / math.sqrt(len2)
        return np.array([x * norm, y * norm, 0.0], dtype=np.float32)

    def trackball_rotate(self, start_trackball, curr_trackball, invert=False):
        start = _normalize(np.asarray(start_trackball, dtype=np.float32))
        curr = _normalize(np.asarray(curr_trackball, dtype=np.float32))
        axis_cam = np.cross(start, curr)
        axis_norm = float(np.linalg.norm(axis_cam))
        if axis_norm < 1e-8:
            return
        dot = float(np.clip(np.dot(start, curr), -1.0, 1.0))
        angle = math.acos(dot)
        if abs(angle) < 1e-8:
            return
        right, up, view_dir = self.get_basis()
        axis_world = (
            right * axis_cam[0]
            + up * axis_cam[1]
            + view_dir * axis_cam[2]
        )
        if invert:
            angle = -angle
        self.offset = _rotate_axis_angle(self.offset, axis_world, angle)
        self.up = _rotate_axis_angle(self.up, axis_world, angle)
        self.up = _normalize(self.up)

    def zoom(self, amount, zoom_scale=0.08):
        scale = 1.0 - float(amount) * float(zoom_scale)
        scale = float(np.clip(scale, 0.1, 10.0))
        self.offset = (self.offset * scale).astype(np.float32)

    def pan(self, delta_x, delta_y, viewport_size):
        width = max(int(viewport_size[0]), 1)
        height = max(int(viewport_size[1]), 1)
        right, up, _ = self.get_basis()
        distance = self.get_distance()
        view_height = 2.0 * distance * math.tan(math.radians(self.fov_y_deg) * 0.5)
        view_width = view_height * (float(width) / float(height))
        move = (
            right * (-float(delta_x) / float(width)) * view_width
            + up * (float(delta_y) / float(height)) * view_height
        )
        self.target = (self.target + move).astype(np.float32)
