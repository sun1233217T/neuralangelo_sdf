"""Chamfer evaluation for NeRF-synthetic scenes against GT point clouds.

GT point clouds (meshout/blender_gt/<scene>_pcd.ply) come from the SplatSDF
release (sampled from the original .blend assets).  Our meshes are in the
normalized B-spline frame; for blender scenes sphere_radius=1.0 and
center=0, so the frames coincide (verified via bounds).

Protocol: sample points on the rec mesh, crop rec points to the GT bbox
expanded by a margin (removes boundary shells / background junk far outside
the object), then symmetric Chamfer (rec->GT and GT->rec, GT->rec computed
against the *cropped* rec point set).

Usage:
    python projects/Bspline-Neus/scripts/eval_blender_chamfer.py SCENE MESH [MESH ...]
"""
import sys
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

scene = sys.argv[1]
gt_path = f"meshout/blender_gt/{scene}_pcd.ply"
gt = np.asarray(o3d.io.read_point_cloud(gt_path).points)
gt_min, gt_max = gt.min(0), gt.max(0)
margin = 0.2
print(f"[{scene}] GT: {len(gt)} pts, bounds {np.round(gt_min,2)} .. {np.round(gt_max,2)}")

gt_tree = cKDTree(gt)
for mesh_path in sys.argv[2:]:
    m = trimesh.load(mesh_path, process=False)
    pts = m.sample(300000)
    inb = ((pts >= gt_min - margin) & (pts <= gt_max + margin)).all(1)
    pts_c = pts[inb]
    if len(pts_c) == 0:
        print(f"  {mesh_path}: 0 pts survived crop!")
        continue
    d2s = gt_tree.query(pts_c, k=1, workers=-1)[0]
    rec_tree = cKDTree(pts_c)
    s2d = rec_tree.query(gt, k=1, workers=-1)[0]
    overall = (d2s.mean() + s2d.mean()) / 2
    print(f"  {mesh_path}")
    print(f"    kept {len(pts) - len(pts_c) and len(pts_c)}/{len(pts)} pts after crop")
    print(f"    d2s: mean={d2s.mean():.4f} p50={np.percentile(d2s,50):.4f} p90={np.percentile(d2s,90):.4f}")
    print(f"    s2d: mean={s2d.mean():.4f} p50={np.percentile(s2d,50):.4f} p90={np.percentile(s2d,90):.4f}")
    print(f"    overall: {overall:.4f}  (normalized units; scene span ~{np.round((gt_max-gt_min).max(),2)})")
