import json
import os
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

from projects.sdf_angelo.uv_distill.common import load_geometry_cache


class TeacherViewStore:

    def __init__(self, dataset_dir, device=None, cache_size=4, cache_device=None,
                 use_packed_views=True, build_packed_views=True, pin_memory=None):
        self.dataset_dir = dataset_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cache_device is None:
            cache_device = self.device
        self.cache_device = torch.device(cache_device)
        self.use_packed_views = bool(use_packed_views)
        self.build_packed_views = bool(build_packed_views)
        if pin_memory is None:
            pin_memory = self.cache_device.type == "cpu" and self.device.type == "cuda"
        self.pin_memory = bool(pin_memory)

        meta_path = os.path.join(dataset_dir, "meta.json")
        with open(meta_path, "r") as file:
            self.meta = json.load(file)
        self.views = self.meta["views"]
        self.geometry = load_geometry_cache(os.path.join(dataset_dir, "geometry.pt"), device=self.device)
        self.tex_size = int(self.geometry["tex_size"])
        self.coords_dtype = torch.int16 if self.tex_size <= 32767 else torch.int32

        self.base_rgb_u8 = self._load_rgb_u8(os.path.join(dataset_dir, "base", "base_median.png"))
        self.base_valid_cpu = self._load_mask(os.path.join(dataset_dir, "base", "base_valid.png"))
        self.base_rgb_u8 = self.base_rgb_u8.to(self.device, non_blocking=True)
        self.base_valid = self.base_valid_cpu.to(self.device, non_blocking=True)

        self.flat_to_sparse_cpu = self._build_sparse_map_cpu()
        self.cache_size = max(int(cache_size), 1)
        self._cache = OrderedDict()
        self.packed_dir = os.path.join(self.dataset_dir, "packed_views_v1")
        if self.use_packed_views:
            os.makedirs(self.packed_dir, exist_ok=True)

    def _build_sparse_map_cpu(self):
        mapping = torch.full((self.tex_size, self.tex_size), -1, dtype=torch.int32)
        coords = self.geometry["coords"].detach().cpu().to(dtype=torch.int64)
        mapping[coords[:, 0], coords[:, 1]] = torch.arange(coords.shape[0], dtype=torch.int32)
        return mapping

    @staticmethod
    def _load_rgb_u8(path):
        image = Image.open(path).convert("RGB")
        arr = np.asarray(image, dtype=np.uint8).copy()
        return torch.from_numpy(arr)

    @staticmethod
    def _load_mask(path):
        image = Image.open(path).convert("L")
        arr = np.asarray(image, dtype=np.uint8).copy()
        return torch.from_numpy(arr > 127)

    def _packed_view_path(self, view_id):
        return os.path.join(self.packed_dir, f"view_{int(view_id):06d}.pt")

    def ensure_cache_capacity(self, size):
        self.cache_size = max(self.cache_size, int(size))

    def get_view_ids(self, include_kinds=None):
        if include_kinds is None:
            return [view["id"] for view in self.views]
        include = set(include_kinds)
        return [view["id"] for view in self.views if view["kind"] in include]

    def split_view_ids(self, include_kinds=None, holdout_every=0):
        ids = self.get_view_ids(include_kinds=include_kinds)
        if holdout_every <= 0:
            return ids, []
        train_ids = []
        val_ids = []
        for idx, view_id in enumerate(ids):
            if (idx + 1) % int(holdout_every) == 0:
                val_ids.append(view_id)
            else:
                train_ids.append(view_id)
        return train_ids, val_ids

    def _save_packed_view(self, view_id, payload):
        if not self.use_packed_views:
            return
        torch.save(payload, self._packed_view_path(view_id))

    def _build_packed_view(self, view_id):
        view = self.views[int(view_id)]
        rgb_path = os.path.join(self.dataset_dir, view["rgb_path"])
        mask_path = os.path.join(self.dataset_dir, view["mask_path"])
        rgb_u8 = self._load_rgb_u8(rgb_path)
        mask = self._load_mask(mask_path)
        coords_yx = torch.nonzero(mask, as_tuple=False).to(dtype=torch.int64)
        if coords_yx.numel() == 0:
            payload = {
                "coords_yx": torch.empty((0, 2), dtype=self.coords_dtype),
                "sparse_idx": torch.empty((0,), dtype=torch.int32),
                "target_rgb_u8": torch.empty((0, 3), dtype=torch.uint8),
                "base_rgb_u8": torch.empty((0, 3), dtype=torch.uint8),
                "uv": torch.empty((0, 2), dtype=torch.float16),
                "base_pool_idx": torch.empty((0,), dtype=torch.int32),
            }
            self._save_packed_view(view_id, payload)
            return payload

        sparse_idx = self.flat_to_sparse_cpu[coords_yx[:, 0], coords_yx[:, 1]].to(dtype=torch.int64)
        keep = sparse_idx >= 0
        if not keep.all():
            coords_yx = coords_yx[keep]
            sparse_idx = sparse_idx[keep]
        target_rgb_u8 = rgb_u8[coords_yx[:, 0], coords_yx[:, 1]]
        base_rgb_u8 = self.base_rgb_u8[coords_yx[:, 0], coords_yx[:, 1]]
        base_keep = self.base_valid_cpu[coords_yx[:, 0], coords_yx[:, 1]]
        base_pool_idx = torch.nonzero(base_keep, as_tuple=False).squeeze(1).to(dtype=torch.int32)
        uv = torch.stack([
            (coords_yx[:, 1].float() + 0.5) / float(self.tex_size),
            (coords_yx[:, 0].float() + 0.5) / float(self.tex_size),
        ], dim=-1).to(dtype=torch.float16)
        payload = {
            "coords_yx": coords_yx.to(dtype=self.coords_dtype),
            "sparse_idx": sparse_idx.to(dtype=torch.int32),
            "target_rgb_u8": target_rgb_u8.to(dtype=torch.uint8),
            "base_rgb_u8": base_rgb_u8.to(dtype=torch.uint8),
            "uv": uv,
            "base_pool_idx": base_pool_idx,
        }
        self._save_packed_view(view_id, payload)
        return payload

    def _load_packed_view(self, view_id):
        packed_path = self._packed_view_path(view_id)
        if self.use_packed_views and os.path.isfile(packed_path):
            try:
                payload = torch.load(packed_path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(packed_path, map_location="cpu")
            required_keys = {"coords_yx", "sparse_idx", "target_rgb_u8", "base_rgb_u8", "uv", "base_pool_idx"}
            if required_keys.issubset(payload.keys()):
                return payload
            if self.build_packed_views:
                return self._build_packed_view(view_id)
            return payload
        if self.build_packed_views:
            return self._build_packed_view(view_id)
        return self._build_packed_view(view_id)

    def _move_cache_tensor(self, tensor, *, dtype=None):
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if self.cache_device.type == "cpu":
            tensor = tensor.contiguous()
            if self.pin_memory:
                tensor = tensor.pin_memory()
            return tensor
        return tensor.to(device=self.cache_device, non_blocking=True)

    def _load_view_payload(self, view_id):
        view = self.views[int(view_id)]
        packed = self._load_packed_view(view_id)
        cam_center = torch.tensor(view["camera_center_norm"], dtype=torch.float32, device=self.device)
        return {
            "coords_yx": self._move_cache_tensor(packed["coords_yx"], dtype=torch.int64),
            "sparse_idx": self._move_cache_tensor(packed["sparse_idx"], dtype=torch.int64),
            "target_rgb_u8": self._move_cache_tensor(packed["target_rgb_u8"], dtype=torch.uint8),
            "base_rgb_u8": self._move_cache_tensor(packed["base_rgb_u8"], dtype=torch.uint8),
            "uv": self._move_cache_tensor(packed["uv"], dtype=torch.float16),
            "base_pool_idx": self._move_cache_tensor(packed["base_pool_idx"], dtype=torch.int64),
            "camera_center_norm": cam_center,
            "view": view,
        }

    def get_view_payload(self, view_id):
        key = int(view_id)
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        value = self._load_view_payload(key)
        self._cache[key] = value
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    def preload_view_ids(self, view_ids, limit=0, verbose=False):
        if limit > 0:
            view_ids = view_ids[:int(limit)]
        total = len(view_ids)
        for idx, view_id in enumerate(view_ids):
            self.get_view_payload(view_id)
            if verbose and (idx == 0 or (idx + 1) % 10 == 0 or (idx + 1) == total):
                print(f"[preload] {idx + 1}/{total} views ready")
        return total

    def sample_view_texels(self, view_id, num_samples, restrict_to_base_valid=False, rng=None):
        payload = self.get_view_payload(view_id)
        if restrict_to_base_valid:
            source = payload["base_pool_idx"]
            valid_count = source.shape[0]
        else:
            source = None
            valid_count = payload["coords_yx"].shape[0]
        if valid_count == 0:
            raise ValueError(f"View {view_id} has no valid texels after masking.")

        count = int(num_samples)
        if isinstance(rng, torch.Generator):
            idx = torch.randint(0, valid_count, (count,), device=self.cache_device, generator=rng)
        elif rng is None:
            idx = torch.randint(0, valid_count, (count,), device=self.cache_device)
        else:
            idx = torch.as_tensor(
                rng.integers(0, valid_count, size=count, endpoint=False),
                dtype=torch.int64,
                device=self.cache_device,
            )
        if source is not None:
            idx = source[idx]

        coords_yx = payload["coords_yx"][idx]
        sparse_idx = payload["sparse_idx"][idx]
        target_rgb_u8 = payload["target_rgb_u8"][idx]
        base_rgb_u8 = payload["base_rgb_u8"][idx]
        uv = payload["uv"][idx]
        if self.cache_device.type != self.device.type:
            coords_yx = coords_yx.to(self.device, non_blocking=True)
            sparse_idx = sparse_idx.to(self.device, non_blocking=True)
            target_rgb_u8 = target_rgb_u8.to(self.device, non_blocking=True)
            base_rgb_u8 = base_rgb_u8.to(self.device, non_blocking=True)
            uv = uv.to(self.device, non_blocking=True)

        base_rgb = base_rgb_u8.float().mul_(1.0 / 255.0)
        target_rgb = target_rgb_u8.float().mul_(1.0 / 255.0)
        points = self.geometry["points"][sparse_idx]
        normals = self.geometry["normals"][sparse_idx]
        camera_center = payload["camera_center_norm"].expand(points.shape[0], -1)
        return {
            "view_id": int(view_id),
            "uv": uv.float(),
            "points": points,
            "normals": normals,
            "base_rgb": base_rgb,
            "target_rgb": target_rgb,
            "target_delta": target_rgb - base_rgb,
            "camera_center_norm": camera_center,
            "coords_yx": coords_yx,
        }
