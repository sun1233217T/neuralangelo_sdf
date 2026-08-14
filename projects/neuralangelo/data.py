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

import json
import numpy as np
import torch
import torchvision.transforms.functional as torchvision_F
from PIL import Image, ImageFile

from projects.nerf.datasets import base
from projects.nerf.utils import camera

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(base.Dataset):

    def __init__(self, cfg, is_inference=False):
        super().__init__(cfg, is_inference=is_inference, is_test=False)
        cfg_data = cfg.data
        self.root = cfg_data.root
        self.preload = cfg_data.preload
        self.H, self.W = cfg_data.val.image_size if is_inference else cfg_data.train.image_size
        meta_fname = f"{cfg_data.root}/transforms.json"
        with open(meta_fname) as file:
            self.meta = json.load(file)
        self.list = self.meta["frames"]
        if cfg_data[self.split].subset:
            subset = cfg_data[self.split].subset
            subset_idx = np.linspace(0, len(self.list), subset+1)[:-1].astype(int)
            self.list = [self.list[i] for i in subset_idx]
        self.num_rays = cfg.model.render.rand_rays
        self.readjust = getattr(cfg_data, "readjust", None)
        # Edge-aware ray sampling options (disabled by default).
        edge_cfg = getattr(cfg_data, "edge_sample", {})
        self.edge_sample = bool(edge_cfg.get("enabled", False)) and self.split == "train"
        self.edge_ratio = float(edge_cfg.get("ratio", 0.5))
        self.edge_power = float(edge_cfg.get("power", 2.0))
        self.edge_eps = float(edge_cfg.get("eps", 1e-6))
        # Preload dataset if possible.
        if cfg_data.preload:
            self.images = self.preload_threading(self.get_image, cfg_data.num_workers)
            self.cameras = self.preload_threading(self.get_camera, cfg_data.num_workers, data_str="cameras")

    def _sample_edge_aware_rays(self, image):
        """Sample ray indices with a mix of edge importance and uniform sampling.

        Args:
            image: Tensor of shape [3, H, W] in [0, 1].
        Returns:
            ray_idx: Long tensor of shape [num_rays] with flat indices in [0, H*W).
        """
        gray = image.mean(0)  # [H, W]
        # Central-difference gradient magnitude.
        gx = torch.zeros_like(gray)
        gy = torch.zeros_like(gray)
        gx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
        gy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
        grad = torch.sqrt(gx ** 2 + gy ** 2)
        # Convert gradient magnitude to a probability map.
        flat = grad.flatten()
        importance = flat ** self.edge_power + self.edge_eps
        p_edge = importance / importance.sum()
        p_uniform = torch.ones_like(p_edge) / p_edge.numel()
        p_mix = self.edge_ratio * p_edge + (1.0 - self.edge_ratio) * p_uniform
        return torch.multinomial(p_mix, self.num_rays, replacement=False)

    def __getitem__(self, idx):
        """Process raw data and return processed data in a dictionary.

        Args:
            idx: The index of the sample of the dataset.
        Returns: A dictionary containing the data.
                 idx (scalar): The index of the sample of the dataset.
                 image (R tensor): Image idx for per-image embedding.
                 image (Rx3 tensor): Image with pixel values in [0,1] for supervision.
                 intr (3x3 tensor): The camera intrinsics of `image`.
                 pose (3x4 tensor): The camera extrinsics [R,t] of `image`.
        """
        # Keep track of sample index for convenience.
        sample = dict(idx=idx)
        # Get the images.
        image, image_size_raw = self.images[idx] if self.preload else self.get_image(idx)
        image = self.preprocess_image(image)
        # Get the cameras (intrinsics and pose).
        intr, pose = self.cameras[idx] if self.preload else self.get_camera(idx)
        intr, pose = self.preprocess_camera(intr, pose, image_size_raw)
        # Precompute inverse intrinsics: the forward pass would otherwise run a
        # cuSOLVER inverse every iteration (device sync, not graph-capturable).
        intr_inv = intr.inverse()
        # Pre-sample ray indices.
        if self.split == "train":
            if self.edge_sample:
                ray_idx = self._sample_edge_aware_rays(image)
            else:
                ray_idx = torch.randperm(self.H * self.W)[:self.num_rays]  # [R]
            image_sampled = image.flatten(1, 2)[:, ray_idx].t()  # [R,3]
            sample.update(
                ray_idx=ray_idx,
                image_sampled=image_sampled,
                intr=intr,
                intr_inv=intr_inv,
                pose=pose,
            )
        else:  # keep image during inference
            sample.update(
                image=image,
                intr=intr,
                intr_inv=intr_inv,
                pose=pose,
            )
        return sample

    def get_image(self, idx):
        fpath = self.list[idx]["file_path"]
        image_fname = f"{self.root}/{fpath}"
        image = Image.open(image_fname)
        image.load()
        image_size_raw = image.size
        return image, image_size_raw

    def preprocess_image(self, image):
        # Resize the image.
        image = image.resize((self.W, self.H))
        image = torchvision_F.to_tensor(image)
        rgb = image[:3]
        return rgb

    def get_camera(self, idx):
        # Camera intrinsics.
        intr = torch.tensor([[self.meta["fl_x"], self.meta["sk_x"], self.meta["cx"]],
                             [self.meta["sk_y"], self.meta["fl_y"], self.meta["cy"]],
                             [0, 0, 1]]).float()
        # Camera pose.
        c2w_gl = torch.tensor(self.list[idx]["transform_matrix"], dtype=torch.float32)
        c2w = self._gl_to_cv(c2w_gl)
        # center scene
        center = np.array(self.meta["sphere_center"])
        center += np.array(getattr(self.readjust, "center", [0])) if self.readjust else 0.
        c2w[:3, -1] -= center
        # scale scene
        scale = np.array(self.meta["sphere_radius"])
        scale *= getattr(self.readjust, "scale", 1.) if self.readjust else 1.
        c2w[:3, -1] /= scale
        w2c = camera.Pose().invert(c2w[:3])
        return intr, w2c

    def preprocess_camera(self, intr, pose, image_size_raw):
        # Adjust the intrinsics according to the resized image.
        intr = intr.clone()
        raw_W, raw_H = image_size_raw
        intr[0] *= self.W / raw_W
        intr[1] *= self.H / raw_H
        return intr, pose

    def _gl_to_cv(self, gl):
        # convert to CV convention used in Imaginaire
        cv = gl * torch.tensor([1, -1, -1, 1])
        return cv
