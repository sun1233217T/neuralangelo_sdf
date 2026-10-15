"""Extract alpha-channel masks from blender (NeRF-synthetic) RGBA images.

Writes mask/NNN.png (L mode, 255=object) next to the scene's transforms.json,
named to match the dataset loader's get_mask() convention (digits parsed from
the frame filename).

Usage:
    python projects/Bspline-Neus/scripts/blender_extract_masks.py \
        --scene datasets/nerf_synthetic/drums
"""
import argparse
import os
import re

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    args = ap.parse_args()
    scene = args.scene

    mask_dir = os.path.join(scene, "mask")
    os.makedirs(mask_dir, exist_ok=True)
    n = 0
    for split in ("train", "val", "test"):
        sdir = os.path.join(scene, split)
        if not os.path.isdir(sdir):
            continue
        for fname in sorted(os.listdir(sdir)):
            if not fname.endswith(".png"):
                continue
            m = re.search(r"(\d+)\.png$", fname)
            frame_no = int(m.group(1))
            img = Image.open(os.path.join(sdir, fname))
            if img.mode != "RGBA":
                continue
            alpha = np.asarray(img)[..., 3]
            mask = (alpha > 127).astype(np.uint8) * 255
            Image.fromarray(mask, mode="L").save(
                os.path.join(mask_dir, f"{frame_no:03d}.png"))
            n += 1
    print(f"[blender_extract_masks] {scene}: {n} masks -> {mask_dir}")


if __name__ == "__main__":
    main()
