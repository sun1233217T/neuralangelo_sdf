"""Mesh extraction for Bspline-Neus checkpoints (dense and hierarchical).

Adapted from ``projects/neuralangelo/scripts/extract_mesh.py``: drops the
hash-grid coarse2fine hooks and reads the extraction bounds from
``model.object.bspline.bounds`` (normalized space) instead of
``transforms.json``.

Run from the repository root, e.g.:

    python projects/Bspline-Neus/scripts/extract_mesh.py --single_gpu \
        --config projects/Bspline-Neus/configs/dtu_scan24_thb_short.yaml \
        --checkpoint logs/<run>/epoch_00833_iteration_000020000_checkpoint.pt \
        --resolution 256 --output_file meshout/scan24_thb_20k.ply
"""

import argparse
import os
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from imaginaire.config import Config, recursive_update_strict, parse_cmdline_arguments  # noqa: E402
from imaginaire.utils.distributed import init_dist, get_world_size, is_master, master_only_print as print  # noqa: E402
from imaginaire.utils.gpu_affinity import set_affinity  # noqa: E402
from imaginaire.trainers.utils.get_trainer import get_trainer  # noqa: E402
from projects.neuralangelo.utils import mesh as _mesh_module  # noqa: E402
from projects.neuralangelo.utils.mesh import extract_mesh, extract_texture  # noqa: E402


def _filter_points_within_bounds(old_mesh, radius):
    """Replace neuralangelo's unit-sphere filter with the actual field bounds.

    The original ``filter_points_outside_bounding_sphere`` hard-codes
    ``norm(v) < 1.0`` which discards vertices outside the unit sphere.  Our
    B-spline field lives on [-B, B]^3 with B = target_half_bound (typically
    2.0), so the mesh legitimately extends past |v|=1 and was being silently
    truncated.  We filter by the actual circumscribed sphere radius instead.
    """
    import numpy as np
    mask = np.linalg.norm(old_mesh.vertices, axis=-1) < radius
    if np.any(mask):
        indices = np.ones(len(old_mesh.vertices), dtype=int) * -1
        indices[mask] = np.arange(mask.sum())
        faces_mask = mask[old_mesh.faces[:, 0]] & mask[old_mesh.faces[:, 1]] & mask[old_mesh.faces[:, 2]]
        new_faces = indices[old_mesh.faces[faces_mask]]
        new_vertices = old_mesh.vertices[mask]
        new_colors = old_mesh.visual.vertex_colors[mask]
        new_mesh = type(old_mesh)(new_vertices, new_faces, vertex_colors=new_colors)
    else:
        import trimesh
        new_mesh = trimesh.Trimesh()
    return new_mesh


def parse_args():
    parser = argparse.ArgumentParser(description="Bspline-Neus mesh extraction")
    parser.add_argument("--config", required=True, help="Path to the training config file.")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path.")
    parser.add_argument("--local_rank", type=int, default=os.getenv("LOCAL_RANK", 0))
    parser.add_argument("--single_gpu", action="store_true")
    parser.add_argument("--resolution", default=512, type=int, help="Marching cubes resolution")
    parser.add_argument("--block_res", default=64, type=int, help="Block-wise resolution")
    parser.add_argument("--output_file", default="mesh.ply", type=str, help="Output file name")
    parser.add_argument("--textured", action="store_true", help="Export mesh with texture")
    parser.add_argument("--keep_lcc", action="store_true",
                        help="Keep only largest connected component.")
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd


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
    print(f"Running mesh extraction with {get_world_size()} GPUs.")

    cfg.logdir = ""

    trainer = get_trainer(cfg, is_inference=True, seed=0)
    # The post-model-load hook rebuilds the optimizer when structure restore
    # grows the hierarchical fields, and the rebuild reads current_iteration
    # for the scheduler.  Set a placeholder; the true value is assigned below.
    trainer.current_iteration = 0
    # The checkpointer calls the model's prepare_load_state_dict hook, which
    # grows the hierarchical fields to the checkpoint's depth before any
    # key/shape filtering, so refined levels survive the load.
    trainer.checkpointer.load(args.checkpoint, load_opt=False, load_sch=False)
    trainer.model.eval()
    trainer.current_iteration = trainer.checkpointer.eval_iteration

    bounds = cfg.model.object.bspline.bounds  # normalized-space AABB
    # The B-spline field lives on [-B, B]^3; the circumscribed sphere radius
    # is B*sqrt(3).  Monkey-patch neuralangelo's hard-coded unit-sphere filter.
    half_bound = max(abs(float(b)) for axis in bounds for b in axis)
    _mesh_module.filter_points_outside_bounding_sphere = partial(
        _filter_points_within_bounds, radius=half_bound * 1.7320508 + 1e-4
    )

    sdf_func = lambda x: -trainer.model_module.neural_sdf.sdf(x)  # noqa: E731
    texture_func = partial(extract_texture, neural_sdf=trainer.model_module.neural_sdf,
                           neural_rgb=trainer.model_module.neural_rgb,
                           appear_embed=trainer.model_module.appear_embed) if args.textured else None
    extent = max(float(b[1]) - float(b[0]) for b in bounds)
    mesh = extract_mesh(sdf_func=sdf_func, bounds=bounds, intv=(extent / args.resolution),
                        block_res=args.block_res, texture_func=texture_func,
                        filter_lcc=args.keep_lcc)

    if is_master():
        print(f"vertices: {len(mesh.vertices)}")
        print(f"faces: {len(mesh.faces)}")
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        print(f"Saving mesh to {args.output_file}")
        mesh.export(args.output_file)
        print("Done!")


if __name__ == "__main__":
    main()
