import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())

from imaginaire.config import Config, parse_cmdline_arguments, recursive_update_strict  # noqa: E402
from imaginaire.utils.distributed import init_dist  # noqa: E402
from projects.sdf_angelo.uv_distill.camera_sampling import sample_augmented_views  # noqa: E402
from projects.sdf_angelo.uv_distill.common import (  # noqa: E402
    DepthVisibilityRenderer,
    build_dataset,
    load_mesh,
    load_original_camera_records,
    load_scene_normalization,
    load_teacher_context,
    render_teacher_texture,
    save_geometry_cache,
    save_mask_png,
    save_rgb_png,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Export offline UV teacher dataset")
    parser.add_argument("--config", required=False, help="Path to config.")
    parser.add_argument("--checkpoint", required=False, help="Checkpoint path.")
    parser.add_argument("--mesh", required=True, help="UV mesh path.")
    parser.add_argument("--uv_bundle", default="", type=str,
                        help="Optional uv bundle. If set, checkpoint is not required for teacher rendering.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--mesh_space", choices=["world", "normalized"], default="world")
    parser.add_argument("--texture_size", default=4096, type=int)
    parser.add_argument("--batch_size", default=65536, type=int)
    parser.add_argument("--uv_raster", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--uv_project_to_surface", action="store_true")
    parser.add_argument("--uv_project_iters", default=1, type=int)
    parser.add_argument("--uv_project_step", default=1.0, type=float)
    parser.add_argument("--uv_project_max_step", default=0.0, type=float)
    parser.add_argument("--appear_idx", default=None, type=int)
    parser.add_argument("--pad_iters", default=0, type=int)
    parser.add_argument("--max_views", default=0, type=int)
    parser.add_argument("--jitter_per_view", default=0, type=int)
    parser.add_argument("--jitter_translation_ratio", default=0.01, type=float)
    parser.add_argument("--jitter_rotation_deg", default=2.0, type=float)
    parser.add_argument("--interp_steps", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--visibility_chunk", default=262144, type=int)
    parser.add_argument("--depth_tol_abs", default=1e-3, type=float)
    parser.add_argument("--depth_tol_rel", default=1e-2, type=float)
    parser.add_argument("--skip_depth_test", action="store_true",
                        help="Skip mesh depth validation and keep only projection/front-facing checks.")
    parser.add_argument("--no_front_only", action="store_true")
    parser.add_argument("--debug_visibility", action="store_true",
                        help="Export intermediate atlas-space visibility masks.")
    parser.add_argument("--debug_every", default=1, type=int,
                        help="Export debug visibility artifacts every N views.")
    parser.add_argument("--save_geometry_only", action="store_true")
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument('--single_gpu', action='store_true')
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd


def main():
    args, cfg_cmd = parse_args()
    try:
        from imaginaire.utils.gpu_affinity import set_affinity  # noqa: E402
        set_affinity(args.local_rank)
    except Exception as exc:
        print(f"Warning: set_affinity failed, continuing without GPU affinity: {exc}")
    cfg = None
    if args.config:
        cfg = Config(args.config)
        cfg_cmd = parse_cmdline_arguments(cfg_cmd)
        recursive_update_strict(cfg, cfg_cmd)
        if not args.single_gpu:
            os.environ["NCLL_BLOCKING_WAIT"] = "0"
            os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
            cfg.local_rank = args.local_rank
            init_dist(cfg.local_rank, rank=-1, world_size=-1)
        cfg.logdir = ""
    teacher = load_teacher_context(args, cfg=cfg)
    os.makedirs(args.output_dir, exist_ok=True)
    geometry_path = os.path.join(args.output_dir, "geometry.pt")
    save_geometry_cache(
        geometry_path,
        teacher["uv_cache"],
        sphere_center=teacher["sphere_center"],
        sphere_radius=teacher["sphere_radius"],
    )
    atlas_valid = torch.zeros((teacher["uv_cache"].tex_size, teacher["uv_cache"].tex_size),
                              device=teacher["device"], dtype=torch.bool)
    atlas_valid[teacher["uv_cache"].coords[:, 0], teacher["uv_cache"].coords[:, 1]] = True
    save_mask_png(os.path.join(args.output_dir, "atlas_valid_mask.png"), atlas_valid)
    if args.save_geometry_only:
        return

    if cfg is None:
        raise ValueError("--config is required when exporting camera-conditioned teacher views.")

    dataset = build_dataset(cfg, split=args.split)
    sphere_center, sphere_radius, _ = load_scene_normalization(cfg)
    original_records = load_original_camera_records(dataset, sphere_center, sphere_radius)
    if args.max_views > 0:
        original_records = original_records[:int(args.max_views)]
    sampled_records = list(original_records)
    sampled_records.extend(
        sample_augmented_views(
            original_records,
            sphere_radius=sphere_radius,
            jitter_per_view=args.jitter_per_view,
            jitter_translation_ratio=args.jitter_translation_ratio,
            jitter_rotation_deg=args.jitter_rotation_deg,
            interp_steps=args.interp_steps,
            seed=args.seed,
        )
    )

    mesh = load_mesh(args.mesh)
    vertices_norm = (np.asarray(mesh.vertices, dtype=np.float32) - teacher["mesh_center"][None, :]) / float(teacher["mesh_radius"])
    visibility = DepthVisibilityRenderer(vertices_norm, np.asarray(mesh.faces, dtype=np.int32), teacher["device"])

    rgb_dir = os.path.join(args.output_dir, "teacher")
    os.makedirs(rgb_dir, exist_ok=True)
    debug_dir = os.path.join(args.output_dir, "visibility_debug")
    if args.debug_visibility:
        os.makedirs(debug_dir, exist_ok=True)
    views_meta = []
    for export_id, record in enumerate(sampled_records):
        texture = render_teacher_texture(
            teacher["uv_cache"],
            teacher["neural_rgb"],
            teacher["appear_embed"],
            cam_pos_world=record.camera_center_world.cpu().numpy(),
            sphere_center=teacher["sphere_center"],
            sphere_radius=teacher["sphere_radius"],
            batch_size=args.batch_size,
            appear_idx=args.appear_idx,
            pad_iters=args.pad_iters,
        )
        if args.debug_visibility and (export_id % max(int(args.debug_every), 1) == 0):
            visible, debug = visibility.compute_visible_mask(
                teacher["uv_cache"].points,
                teacher["uv_cache"].normals,
                intr=record.intr,
                pose=record.pose,
                image_size=record.image_size,
                camera_center_norm=record.camera_center_norm,
                chunk_size=args.visibility_chunk,
                depth_tol_abs=args.depth_tol_abs,
                depth_tol_rel=args.depth_tol_rel,
                front_only=not args.no_front_only,
                use_depth_test=not args.skip_depth_test,
                return_debug=True,
            )
        else:
            debug = None
            visible = visibility.compute_visible_mask(
                teacher["uv_cache"].points,
                teacher["uv_cache"].normals,
                intr=record.intr,
                pose=record.pose,
                image_size=record.image_size,
                camera_center_norm=record.camera_center_norm,
                chunk_size=args.visibility_chunk,
                depth_tol_abs=args.depth_tol_abs,
                depth_tol_rel=args.depth_tol_rel,
                front_only=not args.no_front_only,
                use_depth_test=not args.skip_depth_test,
            )
        dense_mask = torch.zeros((teacher["uv_cache"].tex_size, teacher["uv_cache"].tex_size),
                                 device=teacher["device"], dtype=torch.bool)
        dense_mask[teacher["uv_cache"].coords[:, 0], teacher["uv_cache"].coords[:, 1]] = visible
        rgb_rel = os.path.join("teacher", f"view_{export_id:06d}_rgb.png")
        mask_rel = os.path.join("teacher", f"view_{export_id:06d}_mask.png")
        save_rgb_png(os.path.join(args.output_dir, rgb_rel), texture)
        save_mask_png(os.path.join(args.output_dir, mask_rel), dense_mask)
        visible_count = int(visible.sum().item())
        atlas_count = int(teacher["uv_cache"].coords.shape[0])
        if debug is not None:
            visible_rgb = texture * dense_mask[..., None].float()
            save_rgb_png(os.path.join(debug_dir, f"view_{export_id:06d}_visible_rgb.png"), visible_rgb)
            for key in ("valid_z", "inside", "depth_valid", "depth_pass", "front_pass", "visible"):
                dense_debug = torch.zeros((teacher["uv_cache"].tex_size, teacher["uv_cache"].tex_size),
                                         device=teacher["device"], dtype=torch.bool)
                dense_debug[teacher["uv_cache"].coords[:, 0], teacher["uv_cache"].coords[:, 1]] = debug[key]
                save_mask_png(os.path.join(debug_dir, f"view_{export_id:06d}_{key}.png"), dense_debug)
            debug_stats = {
                "valid_z": int(debug["valid_z"].sum().item()),
                "inside": int(debug["inside"].sum().item()),
                "depth_valid": int(debug["depth_valid"].sum().item()),
                "depth_pass": int(debug["depth_pass"].sum().item()),
                "front_pass": int(debug["front_pass"].sum().item()),
                "visible": visible_count,
                "atlas_valid": atlas_count,
            }
            with open(os.path.join(debug_dir, f"view_{export_id:06d}_stats.json"), "w") as file:
                json.dump(debug_stats, file, indent=2)
        views_meta.append({
            "id": export_id,
            "dataset_index": int(record.index),
            "kind": record.kind,
            "source_indices": [int(v) for v in record.source_indices],
            "is_base_view": bool(record.kind == "original"),
            "image_size": [int(record.image_size[0]), int(record.image_size[1])],
            "intr": record.intr.cpu().numpy().tolist(),
            "pose": record.pose.cpu().numpy().tolist(),
            "camera_center_norm": record.camera_center_norm.cpu().numpy().tolist(),
            "camera_center_world": record.camera_center_world.cpu().numpy().tolist(),
            "rgb_path": rgb_rel,
            "mask_path": mask_rel,
            "visible_count": visible_count,
            "atlas_valid_count": atlas_count,
        })
        print(f"[{export_id + 1}/{len(sampled_records)}] exported {record.kind} view {record.index}")

    meta = {
        "texture_size": int(teacher["uv_cache"].tex_size),
        "sphere_center": teacher["sphere_center"].tolist(),
        "sphere_radius": float(teacher["sphere_radius"]),
        "mesh_space": teacher["mesh_space"],
        "appear_idx": args.appear_idx,
        "split": args.split,
        "use_depth_test": bool(not args.skip_depth_test),
        "front_only": bool(not args.no_front_only),
        "views": views_meta,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as file:
        json.dump(meta, file, indent=2)


if __name__ == "__main__":
    main()
