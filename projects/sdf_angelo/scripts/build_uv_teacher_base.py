import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.append(os.getcwd())

from projects.sdf_angelo.utils.mesh import _dilate_texture  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Build visibility-aware median UV base texture")
    parser.add_argument("--dataset_dir", required=True, help="Teacher dataset directory.")
    parser.add_argument("--tile_size", default=256, type=int)
    parser.add_argument("--fill_iters", default=16, type=int)
    return parser.parse_args()


def _read_tile(path, box, mode):
    image = Image.open(path).convert(mode)
    if box is not None:
        image = image.crop(box)
    return np.asarray(image)


def main():
    args = parse_args()
    meta_path = os.path.join(args.dataset_dir, "meta.json")
    with open(meta_path, "r") as file:
        meta = json.load(file)
    base_views = [view for view in meta["views"] if view.get("is_base_view", False)]
    if not base_views:
        raise ValueError("No base views found in teacher dataset.")
    tex_size = int(meta["texture_size"])
    tile = max(int(args.tile_size), 1)
    base = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
    valid = np.zeros((tex_size, tex_size), dtype=bool)
    for y0 in range(0, tex_size, tile):
        for x0 in range(0, tex_size, tile):
            y1 = min(y0 + tile, tex_size)
            x1 = min(x0 + tile, tex_size)
            box = (x0, y0, x1, y1)
            rgb_stack = []
            mask_stack = []
            for view in base_views:
                rgb_path = os.path.join(args.dataset_dir, view["rgb_path"])
                mask_path = os.path.join(args.dataset_dir, view["mask_path"])
                rgb_stack.append(_read_tile(rgb_path, box, "RGB"))
                mask_stack.append(_read_tile(mask_path, box, "L") > 127)
            rgbs = np.stack(rgb_stack, axis=0).astype(np.float32)
            masks = np.stack(mask_stack, axis=0)
            valid_tile = masks.any(axis=0)
            med_tile = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
            for channel in range(3):
                vals = np.where(masks, rgbs[..., channel], np.nan)
                med = np.nanmedian(vals, axis=0)
                med = np.nan_to_num(med, nan=0.0)
                med_tile[..., channel] = np.clip(np.round(med), 0.0, 255.0).astype(np.uint8)
            base[y0:y1, x0:x1] = med_tile
            valid[y0:y1, x0:x1] = valid_tile
            print(f"Processed tile x={x0}:{x1} y={y0}:{y1}")

    raw_dir = os.path.join(args.dataset_dir, "base")
    os.makedirs(raw_dir, exist_ok=True)
    Image.fromarray(base).save(os.path.join(raw_dir, "base_median_raw.png"))
    Image.fromarray((valid.astype(np.uint8) * 255), mode="L").save(os.path.join(raw_dir, "base_valid.png"))
    if args.fill_iters > 0:
        tex_t = torch.from_numpy(base).float() / 255.0
        mask_t = torch.from_numpy(valid)
        tex_t = _dilate_texture(tex_t, mask_t, int(args.fill_iters))
        base = (tex_t.clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
    Image.fromarray(base).save(os.path.join(raw_dir, "base_median.png"))


if __name__ == "__main__":
    main()

