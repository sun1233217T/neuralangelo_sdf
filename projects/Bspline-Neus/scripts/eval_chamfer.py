"""Chamfer-distance evaluation against the Replica GT mesh.

Loads a reconstructed mesh (normalized B-spline coordinates), maps it back to
world units using the scene's sphere_center/sphere_radius from
transforms.json, samples surface points on both meshes and reports
accuracy (rec->GT), completeness (GT->rec) and within-threshold ratios.

Usage:
    python projects/Bspline-Neus/scripts/eval_chamfer.py \
        --transforms datasets/Replica/Replica/office0_s2/transforms.json \
        --rec meshout/office0_Y8_500k.ply \
        --gt datasets/Replica/Replica/office0_mesh.ply
"""
import argparse
import json

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transforms", required=True,
                    help="transforms.json of the training scene (for normalization)")
    ap.add_argument("--rec", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--n_samples", type=int, default=200000)
    ap.add_argument("--crop_to_gt_margin", type=float, default=None,
                    help="meters; crop the rec mesh to the GT bbox expanded by "
                         "this margin before sampling (removes outer shells "
                         "outside the room)")
    args = ap.parse_args()

    with open(args.transforms) as f:
        meta = json.load(f)
    center = np.array(meta["sphere_center"], dtype=np.float64)
    radius = float(meta["sphere_radius"])

    rec = trimesh.load(args.rec, process=False)
    gt = trimesh.load(args.gt, process=False)
    rec_v = np.asarray(rec.vertices, dtype=np.float64) * radius + center
    rec.vertices = rec_v
    print(f"[chamfer] rec: {len(rec.vertices)} verts, gt: {len(gt.vertices)} verts")
    print(f"[chamfer] rec bounds: {np.round(rec_v.min(0), 2).tolist()} .. "
          f"{np.round(rec_v.max(0), 2).tolist()}")
    gt_v = np.asarray(gt.vertices, dtype=np.float64)
    print(f"[chamfer] gt  bounds: {np.round(gt_v.min(0), 2).tolist()} .. "
          f"{np.round(gt_v.max(0), 2).tolist()}")

    if args.crop_to_gt_margin is not None:
        lo = gt_v.min(0) - args.crop_to_gt_margin
        hi = gt_v.max(0) + args.crop_to_gt_margin
        inside = np.all((rec_v >= lo) & (rec_v <= hi), axis=1)
        face_ok = inside[np.asarray(rec.faces)].all(axis=1)
        rec.update_faces(face_ok)
        rec.remove_unreferenced_vertices()
        rec_v = np.asarray(rec.vertices, dtype=np.float64)
        print(f"[chamfer] rec cropped to GT bbox +{args.crop_to_gt_margin} m: "
              f"{len(rec_v)} verts left")

    rec_pts, _ = trimesh.sample.sample_surface(rec, args.n_samples, seed=0)
    gt_pts, _ = trimesh.sample.sample_surface(gt, args.n_samples, seed=0)

    tree_gt = cKDTree(gt_pts)
    tree_rec = cKDTree(rec_pts)
    acc, _ = tree_gt.query(rec_pts)      # rec -> GT
    comp, _ = tree_rec.query(gt_pts)     # GT -> rec

    def report(name, d):
        print(f"  {name}: mean {d.mean()*100:.2f} cm | median "
              f"{np.median(d)*100:.2f} cm | <2cm {(d < 0.02).mean()*100:.1f}% "
              f"| <5cm {(d < 0.05).mean()*100:.1f}% | <10cm {(d < 0.10).mean()*100:.1f}%")

    print("[chamfer] accuracy (rec -> GT):")
    report("acc", acc)
    print("[chamfer] completeness (GT -> rec):")
    report("comp", comp)
    chamfer = 0.5 * (acc.mean() + comp.mean())
    print(f"[chamfer] Chamfer (mean of means): {chamfer*100:.2f} cm")


if __name__ == "__main__":
    main()
