"""Convert NeRF-synthetic (Blender) scenes to neuralangelo transforms.json.

Blender format stores only camera_angle_x; our loader needs fl_x/fl_y/cx/cy.
Poses are c2w in OpenGL/Blender convention — exactly what the loader expects
(it converts GL->CV itself).

Two outputs:
    <scene>/transforms.json           train split (model training)
    <scene>_heldout/transforms.json   test split (true novel-view eval)

Bounds: cameras orbit at ~4 blender units, object spans ~1.3.  We normalize
by sphere_radius so the object maps well inside the +/-2 field.

Usage:
    python projects/Bspline-Neus/scripts/blender_to_json.py \
        --scene datasets/nerf_synthetic/lego
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def convert_split(scene, split, sphere_center, sphere_radius, out_dir=None,
                  path_prefix=""):
    with open(os.path.join(scene, f"transforms_{split}.json")) as f:
        meta = json.load(f)
    frames = []
    W = H = None
    for fr in meta["frames"]:
        fp = path_prefix + fr["file_path"].lstrip("./")
        img_path = os.path.join(scene, fp)
        if not os.path.splitext(img_path)[1]:
            img_path += ".png"
            fp += ".png"
        if W is None:
            with Image.open(img_path) as im:
                W, H = im.size
        frames.append({"file_path": fp,
                       "transform_matrix": np.asarray(fr["transform_matrix"]).tolist()})
    angle = meta["camera_angle_x"]
    fl = 0.5 * W / np.tan(0.5 * angle)
    out = {
        "fl_x": fl, "fl_y": fl, "sk_x": 0.0, "sk_y": 0.0,
        "cx": W / 2, "cy": H / 2, "w": W, "h": H,
        "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0,
        "p1": 0.0, "p2": 0.0, "is_fisheye": False,
        "camera_angle_x": angle,
        "sphere_center": sphere_center,
        "sphere_radius": sphere_radius,
        "note": f"blender_to_json.py split={split}",
        "frames": frames,
    }
    out_dir = out_dir or scene
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "transforms.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[blender_to_json] {split}: {len(frames)} frames, {W}x{H}, fl={fl:.1f} -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--sphere_radius", type=float, default=1.0,
                    help="world units mapped to normalized 1.0 (object ~1.3 "
                         "blender units fits comfortably in the +/-2 field)")
    args = ap.parse_args()
    scene = args.scene
    convert_split(scene, "train", [0.0, 0.0, 0.0], args.sphere_radius)
    # held-out: test split, paths relative back to the scene dir
    convert_split(scene, "test", [0.0, 0.0, 0.0], args.sphere_radius,
                  out_dir=scene + "_heldout", path_prefix="../" + os.path.basename(scene) + "/")


if __name__ == "__main__":
    main()
