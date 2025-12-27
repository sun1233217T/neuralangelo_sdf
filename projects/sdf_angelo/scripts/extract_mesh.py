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

import argparse
import json
import os
import sys
import numpy as np
import trimesh
from functools import partial

sys.path.append(os.getcwd())
from imaginaire.config import Config, recursive_update_strict, parse_cmdline_arguments  # noqa: E402
from imaginaire.utils.distributed import init_dist, get_world_size, is_master, master_only_print as print  # noqa: E402
from imaginaire.utils.gpu_affinity import set_affinity  # noqa: E402
from imaginaire.trainers.utils.get_trainer import get_trainer  # noqa: E402
from projects.sdf_angelo.utils.mesh import extract_mesh, extract_texture, bake_uv_texture, prepare_mesh_for_uv  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", required=True, help="Path to the training config file.")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path.")
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument("--resolution", default=512, type=int, help="Marching cubes resolution")
    parser.add_argument("--block_res", default=64, type=int, help="Block-wise resolution for marching cubes")
    parser.add_argument("--input_mesh", default="", type=str,
                        help="Use a pre-cropped mesh instead of extracting from SDF")
    parser.add_argument("--input_mesh_space", choices=["world", "normalized"], default="world",
                        help="Coordinate space of input mesh")
    parser.add_argument("--bounds_scale", default=1.0, type=float,
                        help="Uniform scale factor applied to extraction bounds (centered)")
    parser.add_argument("--output_file", default="mesh.ply", type=str, help="Output file name")
    parser.add_argument("--textured", action="store_true", help="Export mesh with texture")
    parser.add_argument("--uv_textured", action="store_true",
                        help="Export mesh with UV texture (xatlas + neural bake)")
    parser.add_argument("--texture_size", default=2048, type=int, help="UV texture resolution")
    parser.add_argument("--bake_batch", default=65536, type=int, help="Batch size for neural texture bake")
    parser.add_argument("--uv_raster", choices=["gpu", "cpu"], default="gpu",
                        help="Rasterization backend for UV bake")
    parser.add_argument("--uv_remesh", action="store_true",
                        help="Apply isotropic remesh with pymeshlab during UV preprocess")
    parser.add_argument("--uv_remesh_target_len", default=0.0, type=float,
                        help="Target edge length for isotropic remesh (0 to auto)")
    parser.add_argument("--uv_remesh_iters", default=10, type=int,
                        help="Iterations for isotropic remesh")
    parser.add_argument("--uv_remesh_position", choices=["before", "after"], default="after",
                        help="Run remesh before or after final target-face simplification")
    parser.add_argument("--simplify_only", action="store_true",
                        help="Export simplified mesh without UV baking")
    parser.add_argument("--skip_uv_preprocess", action="store_true",
                        help="Skip UV preprocess (simplify/LCC)")
    parser.add_argument("--uv_target_faces", default=0, type=int,
                        help="Optional face count target for UV bake decimation")
    parser.add_argument("--uv_max_faces", default=500000, type=int,
                        help="Auto-decimate when face count exceeds this threshold (0 to disable)")
    parser.add_argument("--uv_no_lcc", action="store_true",
                        help="Disable largest connected component filtering before UV unwrap")
    parser.add_argument("--keep_lcc", action="store_true",
                        help="Keep only largest connected component. May remove thin structures.")
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd


def main():
    args, cfg_cmd = parse_args()
    set_affinity(args.local_rank)
    cfg = Config(args.config)

    cfg_cmd = parse_cmdline_arguments(cfg_cmd)
    recursive_update_strict(cfg, cfg_cmd)

    # If args.single_gpu is set to True, we will disable distributed data parallel.
    if not args.single_gpu:
        # this disables nccl timeout
        os.environ["NCLL_BLOCKING_WAIT"] = "0"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank, rank=-1, world_size=-1)
    print(f"Running mesh extraction with {get_world_size()} GPUs.")

    cfg.logdir = ''

    # Initialize data loaders and models.
    trainer = get_trainer(cfg, is_inference=True, seed=0)
    # Load checkpoint.
    trainer.checkpointer.load(args.checkpoint, load_opt=False, load_sch=False)
    trainer.model.eval()

    # Set the coarse-to-fine levels.
    trainer.current_iteration = trainer.checkpointer.eval_iteration
    if cfg.model.object.sdf.encoding.coarse2fine.enabled:
        trainer.model_module.neural_sdf.set_active_levels(trainer.current_iteration)
        if cfg.model.object.sdf.gradient.mode == "numerical":
            trainer.model_module.neural_sdf.set_normal_epsilon()

    meta_fname = f"{cfg.data.root}/transforms.json"
    with open(meta_fname) as file:
        meta = json.load(file)

    if "aabb_range" in meta:
        bounds = (np.array(meta["aabb_range"]) - np.array(meta["sphere_center"])[..., None]) / meta["sphere_radius"]
    else:
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])

    if args.bounds_scale != 1.0:
        if args.bounds_scale <= 0:
            raise ValueError("bounds_scale must be > 0")
        center = bounds.mean(axis=1)
        half = (bounds[:, 1] - bounds[:, 0]) * 0.5 * args.bounds_scale
        bounds = np.stack([center - half, center + half], axis=1)
        print(f"Scaled bounds by {args.bounds_scale}: {bounds.tolist()}")
    sphere_center = np.array(meta["sphere_center"], dtype=np.float32)
    sphere_radius = float(meta["sphere_radius"])

    mesh = None
    if args.input_mesh:
        if not os.path.isfile(args.input_mesh):
            raise FileNotFoundError(f"Input mesh not found: {args.input_mesh}")
        mesh_loaded = trimesh.load(args.input_mesh, process=False)
        if isinstance(mesh_loaded, trimesh.Scene):
            if not mesh_loaded.geometry:
                raise ValueError(f"Input mesh scene is empty: {args.input_mesh}")
            mesh = trimesh.util.concatenate(tuple(mesh_loaded.geometry.values()))
        else:
            mesh = mesh_loaded
        if args.input_mesh_space == "world":
            mesh.vertices = (mesh.vertices - sphere_center) / sphere_radius
        print(f"Loaded input mesh: {args.input_mesh}")

    sdf_func = lambda x: -trainer.model_module.neural_sdf.sdf(x)  # noqa: E731
    texture_func = None
    if args.textured and not args.uv_textured and not args.simplify_only and not args.input_mesh:
        texture_func = partial(extract_texture, neural_sdf=trainer.model_module.neural_sdf,
                               neural_rgb=trainer.model_module.neural_rgb,
                               appear_embed=trainer.model_module.appear_embed)
    if mesh is None:
        mesh = extract_mesh(sdf_func=sdf_func, bounds=bounds, intv=(2.0 / args.resolution),
                            block_res=args.block_res, texture_func=texture_func, filter_lcc=args.keep_lcc)

    if is_master():
        if args.simplify_only:
            if args.uv_textured:
                print("Simplify-only mode enabled; skipping UV bake.")
            if args.skip_uv_preprocess:
                print("Simplify-only mode: skipping UV preprocess.")
            else:
                mesh = prepare_mesh_for_uv(
                    mesh,
                    target_faces=args.uv_target_faces,
                    max_faces=args.uv_max_faces,
                    keep_lcc=not args.uv_no_lcc,
                    remesh=args.uv_remesh,
                    remesh_target_len=args.uv_remesh_target_len,
                    remesh_iters=args.uv_remesh_iters,
                    remesh_position=args.uv_remesh_position,
                )
        elif args.uv_textured:
            mesh = bake_uv_texture(
                mesh,
                neural_rgb=trainer.model_module.neural_rgb,
                neural_sdf=trainer.model_module.neural_sdf,
                appear_embed=trainer.model_module.appear_embed,
                texture_size=args.texture_size,
                batch_size=args.bake_batch,
                target_faces=args.uv_target_faces,
                max_faces=args.uv_max_faces,
                keep_lcc=not args.uv_no_lcc,
                raster_mode=args.uv_raster,
                preprocess=not args.skip_uv_preprocess,
                remesh=args.uv_remesh,
                remesh_target_len=args.uv_remesh_target_len,
                remesh_iters=args.uv_remesh_iters,
                remesh_position=args.uv_remesh_position,
            )
        print(f"vertices: {len(mesh.vertices)}")
        print(f"faces: {len(mesh.faces)}")
        if args.textured and not args.uv_textured and not args.simplify_only and not args.input_mesh:
            print(f"colors: {len(mesh.visual.vertex_colors)}")
        if args.textured and args.uv_textured and not args.simplify_only:
            print("Both --textured and --uv_textured are set. Using UV texture output.")
        # center and scale
        mesh.vertices = mesh.vertices * sphere_radius + sphere_center
        mesh.update_faces(mesh.nondegenerate_faces())
        output_file = args.output_file
        if args.uv_textured and not args.simplify_only:
            ext = os.path.splitext(output_file)[1].lower()
            if ext in {".ply", ".stl", ".off", ".obj"}:
                if ext != ".obj":
                    output_file = os.path.splitext(output_file)[0] + ".obj"
                    print(f"UV textures require OBJ or GLB output. Writing to {output_file}")
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        mesh.export(output_file)


if __name__ == "__main__":
    main()
