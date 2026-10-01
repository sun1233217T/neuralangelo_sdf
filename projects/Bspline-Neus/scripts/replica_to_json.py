"""Convert NICE-SLAM/iMAP-format Replica scenes to neuralangelo transforms.json.

Expected layout after unzipping Replica.zip (datasets/Replica/<scene>):
    frameXXXXXX.jpg       color frames (1200x680)
    depthXXXXXX.png       depth (unused for RGB-only training)
    traj.txt              one 4x4 c2w (row-major, 16 floats) per frame,
                          OpenGL/NeRF camera convention (iMAP-style)

Our loader stores GL c2w in transforms.json and converts GL->CV itself
(_gl_to_cv), so traj.txt poses are written through unchanged.

Scene bounds: cameras are INSIDE the room, so the B-spline field must cover
the trajectory plus the room walls. We take the camera-position bbox and pad
it by --pad on each side (walls are typically < 1.5 m from the camera cloud).

Usage:
    python projects/Bspline-Neus/scripts/replica_to_json.py \
        --scene datasets/Replica/office0 [--stride 10] [--pad 1.0]
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from PIL import Image

# iMAP/NICE-SLAM Replica intrinsics (680x1200).
FX = FY = 600.0
CX, CY = 599.5, 339.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--stride", type=int, default=10,
                    help="use every Nth frame (2000 -> 200 with default)")
    ap.add_argument("--offset", type=int, default=0,
                    help="start frame index; use --offset 5 with --stride 10 to "
                         "select only held-out views of a stride-10 training set")
    ap.add_argument("--out_dir", default=None,
                    help="output dir for transforms.json (default: scene itself). "
                         "file_path entries stay relative to --scene.")
    ap.add_argument("--pad", type=float, default=1.0,
                    help="meters of padding around the camera-position bbox")
    ap.add_argument("--target_half_bound", type=float, default=2.0)
    ap.add_argument("--norm_from", default=None,
                    help="path to a training transforms.json; copy its "
                         "sphere_center/sphere_radius instead of recomputing")
    ap.add_argument("--mesh", default=None,
                    help="PLY mesh; derive bounds from its AABB + --pad_margin "
                         "instead of the camera cloud (fixes the +y overflow)")
    ap.add_argument("--pad_margin", type=float, default=0.2,
                    help="meters of margin around the mesh AABB (with --mesh)")
    args = ap.parse_args()
    scene = args.scene

    traj_path = os.path.join(scene, "traj.txt")
    assert os.path.isfile(traj_path), f"missing {traj_path}"
    traj = np.loadtxt(traj_path, dtype=np.float64).reshape(-1, 4, 4)

    imgs = sorted(glob.glob(os.path.join(scene, "results", "frame*.jpg")) +
                  glob.glob(os.path.join(scene, "results", "frame*.png")),
                  key=lambda p: int(re.search(r"(\d+)\.(jpg|png)$", p).group(1)))
    assert len(imgs) == len(traj), f"{len(imgs)} images vs {len(traj)} poses"
    with Image.open(imgs[0]) as im:
        W, H = im.size
    assert (W, H) == (1200, 680), f"unexpected image size {W}x{H}"

    sel = np.arange(args.offset, len(imgs), args.stride)
    c2w = traj[sel]
    imgs = [imgs[i] for i in sel]

    # bounds from camera cloud (+pad), cube forced around its center
    pos = c2w[:, :3, 3]
    lower = pos.min(axis=0) - args.pad
    upper = pos.max(axis=0) + args.pad
    center = 0.5 * (lower + upper)
    half = float(np.max(upper - lower)) / 2.0
    sphere_radius = half / args.target_half_bound

    if args.mesh is not None:
        import trimesh
        mv = np.asarray(trimesh.load(args.mesh, process=False).vertices,
                        dtype=np.float64)
        lo = mv.min(axis=0) - args.pad_margin
        hi = mv.max(axis=0) + args.pad_margin
        center = 0.5 * (lo + hi)
        half = float(np.max(hi - lo)) / 2.0
        sphere_radius = half / args.target_half_bound
        print(f"[replica_to_json] bounds from mesh AABB +{args.pad_margin} m: "
              f"{np.round(lo, 2).tolist()} .. {np.round(hi, 2).tolist()}")

    # Held-out evaluation must reuse the training normalization exactly.
    if args.norm_from is not None:
        with open(args.norm_from) as f:
            train_meta = json.load(f)
        center = np.array(train_meta["sphere_center"], dtype=np.float64)
        sphere_radius = float(train_meta["sphere_radius"])
        print(f"[replica_to_json] normalization taken from {args.norm_from}")

    # sanity: trajectory smoothness + camera-forward spread
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    fwd = c2w[:, :3, 2]  # GL: camera looks along -z; record spread only
    print(f"[replica_to_json] {len(imgs)} frames kept (stride {args.stride})")
    print(f"  cam pos bbox: {np.round(lower, 2).tolist()} .. {np.round(upper, 2).tolist()}")
    print(f"  step: mean {step.mean():.3f} m, max {step.max():.3f} m")
    print(f"  forward(-z) mean dir: {np.round((-fwd).mean(axis=0), 2).tolist()}")
    print(f"  sphere_center={np.round(center, 4).tolist()}, "
          f"sphere_radius={sphere_radius:.4f} (half {half:.2f} m -> "
          f"+/-{args.target_half_bound} normalized)")

    frames = [{"file_path": f"results/{os.path.basename(p)}",
               "transform_matrix": c2w[i].tolist()}
              for i, p in enumerate(imgs)]
    meta = {
        "fl_x": FX, "fl_y": FY, "sk_x": 0.0, "sk_y": 0.0,
        "cx": CX, "cy": CY, "w": W, "h": H,
        "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0,
        "p1": 0.0, "p2": 0.0, "is_fisheye": False,
        "camera_angle_x": float(2.0 * np.arctan(W / (2.0 * FX))),
        "sphere_center": center.tolist(),
        "sphere_radius": sphere_radius,
        "note": f"replica_to_json.py: stride={args.stride}, "
                f"offset={args.offset}, pad={args.pad} m",
        "frames": frames,
    }
    out_dir = args.out_dir if args.out_dir else scene
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "transforms.json")
    # file_path must resolve from out_dir (loader does root/file_path)
    if os.path.normpath(out_dir) != os.path.normpath(scene):
        for fr in frames:
            fr["file_path"] = os.path.relpath(
                os.path.join(scene, fr["file_path"]), out_dir).replace("\\", "/")
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[replica_to_json] wrote {out}")


if __name__ == "__main__":
    main()
