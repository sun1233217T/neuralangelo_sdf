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


'''
python projects/sdf_angelo/scripts/extract_mesh.py --single_gpu\
  --config projects/sdf_angelo/configs/dtu-win.yaml --checkpoint logs/test/sdf_fast_train_baseline/epoch_04081_iteration_000200000_checkpoint.pt \
  --depth_visible --alpha_threshold 0.5 --resolution 1024\
  --output_file meshout/sdf_fast_train_baseline.ply

'''

import argparse
import json
import os
import sys
import numpy as np
import trimesh
import torch
from functools import partial
from PIL import Image

sys.path.append(os.getcwd())
from imaginaire.config import Config, recursive_update_strict, parse_cmdline_arguments  # noqa: E402
from imaginaire.utils.distributed import init_dist, get_world_size, is_master, master_only_print as print  # noqa: E402
from imaginaire.utils.gpu_affinity import set_affinity  # noqa: E402
from imaginaire.trainers.utils.get_trainer import get_trainer  # noqa: E402
from projects.sdf_angelo.utils.mesh import (  # noqa: E402
    extract_mesh,
    extract_texture,
    bake_uv_texture,
    prepare_mesh_for_uv,
    _get_nvdiffrast,
)


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
    parser.add_argument("--uv_padding", default=2, type=int,
                        help="Texel padding iterations for UV texture (0 to disable)")
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
    parser.add_argument("--visible_only", action="store_true",
                        help="Filter mesh by camera frustum and RGBA alpha mask.")
    parser.add_argument("--alpha_threshold", default=0.5, type=float,
                        help="Alpha threshold in [0,1] for RGBA visibility mask.")
    parser.add_argument("--depth_visible", action="store_true",
                        help="Use GPU rasterization for depth visibility (requires nvdiffrast).")
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd


def _gl_to_cv(gl):
    return gl * np.array([1, -1, -1, 1], dtype=np.float32)


def _build_intrinsics(meta):
    fl_x = float(meta["fl_x"])
    fl_y = float(meta["fl_y"])
    cx = float(meta["cx"])
    cy = float(meta["cy"])
    sk_x = float(meta.get("sk_x", 0.0))
    sk_y = float(meta.get("sk_y", 0.0))
    intr = np.array([
        [fl_x, sk_x, cx],
        [sk_y, fl_y, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    w = int(meta.get("w", 0))
    h = int(meta.get("h", 0))
    return intr, w, h


def _load_alpha_mask(image_path, alpha_threshold):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path)
    image.load()
    image = image.convert("RGBA")
    rgba = np.asarray(image)
    alpha = rgba[..., 3]
    if alpha_threshold <= 0:
        mask = np.ones(alpha.shape, dtype=bool)
    else:
        thresh = int(round(alpha_threshold * 255.0))
        mask = alpha >= thresh
    return mask, image.size


def _filter_mesh_visibility(mesh, meta, cfg, alpha_threshold=0.5, device=None, depth_visible=False):
    if mesh is None or len(mesh.vertices) == 0:
        return mesh
    if not (0.0 <= alpha_threshold <= 1.0):
        raise ValueError("alpha_threshold must be in [0, 1].")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if depth_visible:
        return _filter_mesh_visibility_depth(mesh, meta, cfg, alpha_threshold, device)

    intr_base, meta_w, meta_h = _build_intrinsics(meta)
    center = np.array(meta.get("sphere_center", [0.0, 0.0, 0.0]), dtype=np.float32)
    radius = float(meta.get("sphere_radius", 1.0))
    readjust = getattr(cfg.data, "readjust", None)
    if readjust is not None:
        center += np.array(getattr(readjust, "center", [0.0, 0.0, 0.0]), dtype=np.float32)
        radius *= float(getattr(readjust, "scale", 1.0))
    if radius <= 0:
        raise ValueError("sphere_radius must be > 0 for visibility filtering.")

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    verts_t = torch.from_numpy(vertices).to(device=device)
    visible = torch.zeros((verts_t.shape[0],), dtype=torch.bool, device=device)
    frames = meta.get("frames", [])
    if len(frames) == 0:
        print("No frames found in transforms.json; skipping visibility filter.")
        return mesh

    print(f"Filtering mesh visibility using {len(frames)} frames...")
    for frame in frames:
        image_path = os.path.join(cfg.data.root, frame["file_path"])
        alpha_mask, image_size = _load_alpha_mask(image_path, alpha_threshold)
        img_w, img_h = image_size
        intr = intr_base.copy()
        if meta_w > 0 and meta_h > 0 and (img_w != meta_w or img_h != meta_h):
            scale_x = img_w / float(meta_w)
            scale_y = img_h / float(meta_h)
            intr[0] *= scale_x
            intr[1] *= scale_y

        c2w_gl = np.asarray(frame["transform_matrix"], dtype=np.float32)
        c2w = _gl_to_cv(c2w_gl)
        c2w[:3, 3] -= center
        c2w[:3, 3] /= radius
        w2c = np.linalg.inv(c2w)
        R = torch.from_numpy(w2c[:3, :3]).to(device=device)
        t = torch.from_numpy(w2c[:3, 3]).to(device=device)
        intr_t = torch.from_numpy(intr).to(device=device)

        pts_cam = verts_t @ R.t() + t
        z = pts_cam[:, 2]
        valid_z = z > 1e-6
        if not valid_z.any():
            continue
        valid_idx = torch.nonzero(valid_z, as_tuple=False).squeeze(1)
        pts_cam = pts_cam[valid_idx]
        uvw = pts_cam @ intr_t.t()
        x = uvw[:, 0] / uvw[:, 2]
        y = uvw[:, 1] / uvw[:, 2]
        x_int = torch.round(x).to(dtype=torch.int64)
        y_int = torch.round(y).to(dtype=torch.int64)
        inside = (x_int >= 0) & (x_int < img_w) & (y_int >= 0) & (y_int < img_h)
        if not inside.any():
            continue
        inside_idx = valid_idx[inside]
        alpha_t = torch.from_numpy(alpha_mask).to(device=device)
        alpha_ok = alpha_t[y_int[inside], x_int[inside]]
        if alpha_ok.any():
            visible[inside_idx[alpha_ok]] = True
        if visible.all():
            break

    visible_cpu = visible.cpu().numpy()
    face_mask = visible_cpu[mesh.faces].any(axis=1)
    if not np.any(face_mask):
        print("Visibility filter removed all faces.")
        return trimesh.Trimesh()
    mesh.update_faces(face_mask)
    mesh.remove_unreferenced_vertices()
    print(f"Visible vertices: {len(mesh.vertices)}")
    print(f"Visible faces: {len(mesh.faces)}")
    return mesh


def _filter_mesh_visibility_depth(mesh, meta, cfg, alpha_threshold=0.5, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("depth_visible requires a CUDA device.")

    dr = _get_nvdiffrast()
    ctx = dr.RasterizeCudaContext()
    intr_base, meta_w, meta_h = _build_intrinsics(meta)
    center = np.array(meta.get("sphere_center", [0.0, 0.0, 0.0]), dtype=np.float32)
    radius = float(meta.get("sphere_radius", 1.0))
    readjust = getattr(cfg.data, "readjust", None)
    if readjust is not None:
        center += np.array(getattr(readjust, "center", [0.0, 0.0, 0.0]), dtype=np.float32)
        radius *= float(getattr(readjust, "scale", 1.0))
    if radius <= 0:
        raise ValueError("sphere_radius must be > 0 for visibility filtering.")

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    verts_t = torch.from_numpy(vertices).to(device=device, dtype=torch.float32)
    tri = torch.from_numpy(faces).to(device=device, dtype=torch.int32)
    visible_faces = torch.zeros((faces.shape[0],), dtype=torch.bool, device=device)
    frames = meta.get("frames", [])
    if len(frames) == 0:
        print("No frames found in transforms.json; skipping visibility filter.")
        return mesh

    print(f"Filtering mesh visibility (depth) using {len(frames)} frames...")
    for frame in frames:
        image_path = os.path.join(cfg.data.root, frame["file_path"])
        alpha_mask, image_size = _load_alpha_mask(image_path, alpha_threshold)
        img_w, img_h = image_size
        denom_w = max(img_w - 1, 1)
        denom_h = max(img_h - 1, 1)
        intr = intr_base.copy()
        if meta_w > 0 and meta_h > 0 and (img_w != meta_w or img_h != meta_h):
            scale_x = img_w / float(meta_w)
            scale_y = img_h / float(meta_h)
            intr[0] *= scale_x
            intr[1] *= scale_y

        c2w_gl = np.asarray(frame["transform_matrix"], dtype=np.float32)
        c2w = _gl_to_cv(c2w_gl)
        c2w[:3, 3] -= center
        c2w[:3, 3] /= radius
        w2c = np.linalg.inv(c2w)
        R = torch.from_numpy(w2c[:3, :3]).to(device=device)
        t = torch.from_numpy(w2c[:3, 3]).to(device=device)
        intr_t = torch.from_numpy(intr).to(device=device)

        pts_cam = verts_t @ R.t() + t
        z = pts_cam[:, 2]
        valid_z = z > 1e-6
        if not valid_z.any():
            continue
        min_z = z[valid_z].min().item()
        max_z = z[valid_z].max().item()
        near = max(min_z * 0.9, 1e-4)
        far = max(max_z * 1.1, near + 1e-4)

        uvw = pts_cam @ intr_t.t()
        z_safe = torch.where(valid_z, z, torch.full_like(z, near))
        u = uvw[:, 0] / z_safe
        v = uvw[:, 1] / z_safe
        uv = torch.stack([u / float(denom_w), v / float(denom_h)], dim=-1)
        uv[:, 1] = 1.0 - uv[:, 1]
        pos = torch.empty((1, verts_t.shape[0], 4), device=device, dtype=torch.float32)
        pos[0, :, 0:2] = uv * 2.0 - 1.0
        pos[0, :, 2] = (2.0 * z - (far + near)) / (far - near)
        pos[0, :, 3] = 1.0
        invalid = ~valid_z
        if invalid.any():
            pos[0, invalid, 0:3] = 2.0

        rast, _ = dr.rasterize(ctx, pos, tri, resolution=[img_h, img_w])
        tri_id = rast[..., 3]
        if tri_id.min().item() < 0:
            mask = tri_id >= 0
            tri_idx = tri_id.to(torch.int64)
        else:
            mask = tri_id > 0
            tri_idx = tri_id.to(torch.int64) - 1
        alpha_t = torch.from_numpy(alpha_mask).to(device=device)
        mask = mask & alpha_t.unsqueeze(0)
        if not mask.any():
            continue
        tri_visible = tri_idx[mask]
        visible_faces[tri_visible] = True
        if visible_faces.all():
            break

    visible_faces_cpu = visible_faces.cpu().numpy()
    if not np.any(visible_faces_cpu):
        print("Visibility filter removed all faces.")
        return trimesh.Trimesh()
    mesh.update_faces(visible_faces_cpu)
    mesh.remove_unreferenced_vertices()
    print(f"Visible vertices: {len(mesh.vertices)}")
    print(f"Visible faces: {len(mesh.faces)}")
    return mesh


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
        if args.visible_only or args.depth_visible:
            mesh = _filter_mesh_visibility(
                mesh,
                meta,
                cfg,
                alpha_threshold=args.alpha_threshold,
                device=next(trainer.model_module.neural_sdf.parameters()).device,
                depth_visible=args.depth_visible,
            )
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
                pad_iters=args.uv_padding,
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
