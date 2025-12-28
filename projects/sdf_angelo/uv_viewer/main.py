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
import torch
import webview

sys.path.append(os.getcwd())

from imaginaire.config import Config, recursive_update_strict, parse_cmdline_arguments  # noqa: E402
from imaginaire.utils.distributed import init_dist, get_world_size, is_master, master_only_print as print  # noqa: E402
from imaginaire.utils.gpu_affinity import set_affinity  # noqa: E402
from imaginaire.trainers.utils.get_trainer import get_trainer  # noqa: E402
from projects.sdf_angelo.uv_viewer.uv_cache import build_uv_cache  # noqa: E402
from projects.sdf_angelo.uv_viewer.texture_generator import TextureGenerator  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="UV Mesh Viewer")
    parser.add_argument("--config", required=True, help="Path to the training config file.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--mesh", required=True, help="Path to UV mesh (OBJ).")
    parser.add_argument("--texture_size", default=1024, type=int, help="UV texture resolution.")
    parser.add_argument("--update_fps", default=1.0, type=float,
                        help="Texture update rate in FPS. Use -1 to remove rate limit.")
    parser.add_argument("--batch_size", default=65536, type=int, help="Batch size for SDF/RGB queries.")
    parser.add_argument("--uv_raster", choices=["gpu", "cpu"], default="gpu",
                        help="Rasterization backend for UV cache.")
    parser.add_argument("--appear_idx", default=None, type=int,
                        help="Optional appear_embed index. Omit to use zero embedding.")
    parser.add_argument("--async_encode", dest="async_encode", action="store_true",
                        help="Enable async encoding (default).")
    parser.add_argument("--no_async_encode", dest="async_encode", action="store_false",
                        help="Disable async encoding.")
    parser.add_argument("--window_width", default=1280, type=int, help="Viewer window width.")
    parser.add_argument("--window_height", default=720, type=int, help="Viewer window height.")
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument('--single_gpu', action='store_true')
    parser.set_defaults(async_encode=True)
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd


class WebAPI:

    def __init__(self, generator, mesh_obj, update_ms):
        self.generator = generator
        self.mesh_obj = mesh_obj
        self.update_ms = update_ms

    def get_init(self):
        return {
            "mesh_obj": self.mesh_obj,
            "update_ms": int(self.update_ms),
            "texture_size": int(self.generator.uv_cache.tex_size),
            "texture_transport": "png",
        }

    def get_texture(self):
        stats = self.generator.get_stats()
        stats.update(_get_cuda_stats())
        return {
            "data_url": self.generator.get_texture_b64(),
            "stats": stats,
        }

    def set_camera(self, camera):
        if not camera:
            return
        self.generator.update_camera([camera["x"], camera["y"], camera["z"]])


def _load_mesh(mesh_path):
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    mesh_loaded = trimesh.load(mesh_path, process=False)
    if isinstance(mesh_loaded, trimesh.Scene):
        if not mesh_loaded.geometry:
            raise ValueError(f"Mesh scene is empty: {mesh_path}")
        mesh = trimesh.util.concatenate(tuple(mesh_loaded.geometry.values()))
    else:
        mesh = mesh_loaded
    return mesh


def _initial_camera(mesh):
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    size = bounds[1] - bounds[0]
    radius = np.linalg.norm(size) * 0.6
    cam_pos = center + np.array([0.0, 0.0, max(radius, 1.0) * 2.5], dtype=np.float32)
    return cam_pos


def _get_cuda_stats():
    if not torch.cuda.is_available():
        return {
            "gpu_allocated_mb": None,
            "gpu_reserved_mb": None,
            "gpu_max_allocated_mb": None,
        }
    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return {
        "gpu_allocated_mb": alloc,
        "gpu_reserved_mb": reserved,
        "gpu_max_allocated_mb": max_alloc,
    }


def main():
    args, cfg_cmd = parse_args()
    set_affinity(args.local_rank)
    cfg = Config(args.config)
    cfg_cmd = parse_cmdline_arguments(cfg_cmd)
    recursive_update_strict(cfg, cfg_cmd)

    if not args.single_gpu:
        os.environ["NCLL_BLOCKING_WAIT"] = "0"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank, rank=-1, world_size=-1)
    print(f"Launching UV viewer with {get_world_size()} GPUs.")
    cfg.logdir = ''

    trainer = get_trainer(cfg, is_inference=True, seed=0)
    trainer.checkpointer.load(args.checkpoint, load_opt=False, load_sch=False)
    trainer.model.eval()
    trainer.current_iteration = trainer.checkpointer.eval_iteration
    if cfg.model.object.sdf.encoding.coarse2fine.enabled:
        trainer.model_module.neural_sdf.set_active_levels(trainer.current_iteration)
        if cfg.model.object.sdf.gradient.mode == "numerical":
            trainer.model_module.neural_sdf.set_normal_epsilon()

    meta_fname = f"{cfg.data.root}/transforms.json"
    with open(meta_fname) as file:
        meta = json.load(file)
    sphere_center = np.array(meta["sphere_center"], dtype=np.float32)
    sphere_radius = float(meta["sphere_radius"])

    mesh = _load_mesh(args.mesh)
    if not hasattr(mesh, "visual") or mesh.visual is None or getattr(mesh.visual, "uv", None) is None:
        raise ValueError("Input mesh does not contain UV coordinates.")

    if is_master():
        print(f"Mesh vertices: {len(mesh.vertices)}")
        print(f"Mesh faces: {len(mesh.faces)}")
        print(f"UV texture size: {args.texture_size}")

    uv_cache = build_uv_cache(
        mesh,
        trainer.model_module.neural_sdf,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
        texture_size=args.texture_size,
        raster_mode=args.uv_raster,
        batch_size=args.batch_size,
    )

    generator = TextureGenerator(
        uv_cache=uv_cache,
        neural_rgb=trainer.model_module.neural_rgb,
        appear_embed=trainer.model_module.appear_embed,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
        update_fps=args.update_fps,
        batch_size=args.batch_size,
        appear_idx=args.appear_idx,
        encode_base64=True,
        include_raw=False,
        async_encode=args.async_encode,
    )

    cam_pos = _initial_camera(mesh)
    generator.update_camera(cam_pos)
    generator.start()

    mesh_obj = mesh.export(file_type="obj")
    if isinstance(mesh_obj, bytes):
        mesh_obj = mesh_obj.decode("utf-8", errors="ignore")

    update_ms = 0 if args.update_fps < 0 else 1000.0 / max(args.update_fps, 1e-6)
    api = WebAPI(generator, mesh_obj, update_ms)
    index_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
    window = webview.create_window(
        "UV Viewer",
        url=index_path,
        js_api=api,
        width=args.window_width,
        height=args.window_height,
    )

    def _on_closed():
        generator.stop()

    window.events.closed += _on_closed
    webview.start()


if __name__ == "__main__":
    main()
