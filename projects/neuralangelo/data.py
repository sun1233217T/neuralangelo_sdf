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
import os
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
        # Optionally exclude validation frames from the training set so that
        # val PSNR measures generalization to unseen views, not memorization.
        val_subset = cfg_data.val.subset
        if (
            self.split == "train"
            and val_subset
            and getattr(cfg_data.val, "exclude_from_train", False)
        ):
            val_idx = set(np.linspace(0, len(self.list), val_subset + 1)[:-1].astype(int).tolist())
            self.list = [f for i, f in enumerate(self.list) if i not in val_idx]
        elif cfg_data[self.split].subset:
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
        # Optional object masks (e.g. DTU provides per-frame masks under
        # root/mask/).  When enabled, training rays additionally carry a
        # per-ray foreground indicator used by the background-opacity loss.
        mask_cfg = getattr(cfg_data, "mask", {})
        self.use_mask = bool(mask_cfg.get("enabled", False))
        if self.use_mask and cfg_data.preload:
            self.masks = self.preload_threading(self.get_mask, cfg_data.num_workers, data_str="masks")
        # Optional GT depth supervision (e.g. Replica provides per-frame
        # 16-bit depth PNGs next to the color frames).  When enabled,
        # training rays additionally carry per-ray GT depth converted to
        # ray distance in the model's normalized units.
        depth_cfg = getattr(cfg_data, "depth", {})
        self.use_depth = bool(depth_cfg.get("enabled", False)) and self.split == "train"
        self.depth_scale = float(depth_cfg.get("scale", 6553.5))  # raw -> meters
        # GT surfaces beyond this normalized distance are outside the
        # B-spline field (bounds are +/-2); supervising them would push the
        # field to fake shells at its boundary.  Treat them as invalid.
        self.depth_max_norm = depth_cfg.get("max_norm", None)
        if self.use_depth and cfg_data.preload:
            self.depths = self.preload_threading(self.get_depth, cfg_data.num_workers, data_str="depths")
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
            if self.use_mask:
                mask, _ = self.masks[idx] if self.preload else self.get_mask(idx)
                mask = torchvision_F.to_tensor(mask.resize((self.W, self.H)))[0]  # [H,W]
                mask_sampled = mask.flatten(0, 1)[ray_idx]  # [R]
                sample.update(mask_sampled=mask_sampled)
            if self.use_depth:
                depth, _ = self.depths[idx] if self.preload else self.get_depth(idx)
                depth_sampled = depth.flatten(0, 1)[ray_idx]  # [R]
                sample.update(depth_sampled=depth_sampled)
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

    def get_mask(self, idx):
        """Load the object mask for frame ``idx`` (mask/{frame:03d}.png)."""
        fpath = self.list[idx]["file_path"]
        frame_no = int(os.path.splitext(os.path.basename(fpath))[0])
        mask_fname = f"{self.root}/mask/{frame_no:03d}.png"
        mask = Image.open(mask_fname).convert("L")
        mask.load()
        mask_size_raw = mask.size
        return mask, mask_size_raw

    def get_depth(self, idx):
        """Load GT depth for frame ``idx`` as a [H,W] float tensor.

        iMAP/NICE-SLAM Replica layout: ``results/depthNNNNNN.png`` next to
        ``results/frameNNNNNN.jpg``, 16-bit, depth_m = raw / depth_scale.
        Returned depth is converted from camera-space z to ray distance
        (t) and from meters to the model's normalized units
        (divided by sphere_radius), matching the renderer's composited depth.
        Invalid pixels (raw == 0) stay 0 and are masked out in the loss.
        """
        fpath = self.list[idx]["file_path"]
        stem = os.path.splitext(os.path.basename(fpath))[0]  # frameNNNNNN
        depth_fname = os.path.join(self.root, os.path.dirname(fpath),
                                   stem.replace("frame", "depth", 1) + ".png")
        raw = np.asarray(Image.open(depth_fname), dtype=np.float32) / self.depth_scale
        H0, W0 = raw.shape
        # Resize to the training resolution (nearest, keeps depth values).
        if (W0, H0) != (self.W, self.H):
            ys = (np.arange(self.H) + 0.5) * H0 / self.H - 0.5
            xs = (np.arange(self.W) + 0.5) * W0 / self.W - 0.5
            raw = raw[np.clip(ys.round().astype(int), 0, H0 - 1)][:, np.clip(xs.round().astype(int), 0, W0 - 1)]
        # z -> ray distance: t = z * sqrt(1 + u^2 + v^2) with the resized intr.
        fx = self.meta["fl_x"] * self.W / W0
        fy = self.meta["fl_y"] * self.H / H0
        cx = self.meta["cx"] * self.W / W0
        cy = self.meta["cy"] * self.H / H0
        u = (np.arange(self.W) - cx) / fx
        v = (np.arange(self.H) - cy) / fy
        factor = np.sqrt(1.0 + u[None, :] ** 2 + v[:, None] ** 2).astype(np.float32)
        depth_t = raw * factor
        # meters -> normalized model units
        depth_t = depth_t / float(self.meta["sphere_radius"])
        if self.depth_max_norm is not None:
            depth_t = depth_t.copy()
            depth_t[depth_t > float(self.depth_max_norm)] = 0.0
        return torch.from_numpy(depth_t), (W0, H0)

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
