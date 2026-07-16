#!/usr/bin/env python
"""Run a full eval pipeline for existing sdf_angelo checkpoints on DTU scan24.

Pipeline per checkpoint:
  1. extract alpha-visible mesh
  2. extract depth-visible mesh
  3. DTU Chamfer eval
  4. extract UV textured mesh
  5. UV mesh PSNR on val split

Outputs are written to meshout/eval_current_state/<exp_name>/.
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACT_SCRIPT = PROJECT_ROOT / "projects/sdf_angelo/scripts/extract_mesh.py"
EVAL_SCRIPT = PROJECT_ROOT / "projects/sdf_angelo/scripts/eval_DTU_mesh.py"
UV_EVAL_SCRIPT = PROJECT_ROOT / "projects/sdf_angelo/scripts/uv_mesh_psnr.py"

DEFAULT_RESOLUTION = 1024
DEFAULT_BLOCK_RES = 128
DEFAULT_UV_TEXTURE_SIZE = 2048
DEFAULT_UV_TARGET_FACES = 50000
DEFAULT_UV_WELD_TOL = 1e-4


def _run(cmd, env=None, cwd=None, capture=False):
    print("\n" + "=" * 80)
    print(" ".join(str(c) for c in cmd))
    print("=" * 80)
    if capture:
        result = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = result.stdout
        print(output)
    else:
        result = subprocess.run(cmd, cwd=cwd, env=env)
        output = ""
    if result.returncode != 0:
        print(f"[WARN] command failed with code {result.returncode}")
    return result.returncode == 0, output


def _get_scan_id_from_root(root):
    m = re.search(r"scan(\d+)", str(root))
    if m:
        return int(m.group(1))
    return None


def eval_one(
    exp_name,
    config_path,
    checkpoint_path,
    data_root,
    dataset_dir,
    output_dir,
    resolution=DEFAULT_RESOLUTION,
    block_res=DEFAULT_BLOCK_RES,
    uv_texture_size=DEFAULT_UV_TEXTURE_SIZE,
    uv_target_faces=DEFAULT_UV_TARGET_FACES,
    gpu_id=None,
    skip_uv=False,
    skip_dtuv=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scan_id = _get_scan_id_from_root(data_root)
    if scan_id is None:
        raise ValueError(f"Could not infer scan id from data_root: {data_root}")

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    base = output_dir / exp_name
    vis_mesh = base.with_name(base.name + "_vis.obj")
    eval_mesh = base.with_name(base.name + "_eval.obj")
    uv_mesh = base.with_name(base.name + "_eval_uv.obj")
    uv_debug_dir = base.with_name(base.name + "_uv_debug")
    results_json = output_dir / f"results_{scan_id}_{exp_name}.json"
    psnr_log = uv_debug_dir / "psnr.txt"

    # 1. alpha-visible mesh
    if not vis_mesh.exists():
        ok, _ = _run([
            sys.executable, str(EXTRACT_SCRIPT),
            "--single_gpu",
            f"--config={config_path}",
            f"--checkpoint={checkpoint_path}",
            f"--resolution={resolution}",
            f"--block_res={block_res}",
            "--visible_only",
            f"--output_file={vis_mesh}",
            f"--data.root={data_root}",
        ], env=env, cwd=PROJECT_ROOT)
        if not ok:
            return {"exp_name": exp_name, "status": "failed_alpha_visible"}
    else:
        print(f"[skip] alpha-visible mesh exists: {vis_mesh}")

    # 2. depth-visible mesh
    if not eval_mesh.exists():
        ok, _ = _run([
            sys.executable, str(EXTRACT_SCRIPT),
            "--single_gpu",
            f"--config={config_path}",
            f"--checkpoint={checkpoint_path}",
            f"--output_file={eval_mesh}",
            "--input_mesh_space=world",
            f"--input_mesh={vis_mesh}",
            "--depth_visible",
            "--depth_percentile=95",
            "--depth_trim_low=1",
            "--depth_trim_high=99",
            "--depth_smooth_kernel=3",
            "--depth_margin_stat=median",
            "--depth_margin_ratio=0.01",
            f"--data.root={data_root}",
        ], env=env, cwd=PROJECT_ROOT)
        if not ok:
            return {"exp_name": exp_name, "status": "failed_depth_visible"}
    else:
        print(f"[skip] depth-visible mesh exists: {eval_mesh}")

    # 3. DTU eval
    if not skip_dtuv and (not results_json.exists() or results_json.stat().st_size == 0):
        ok, dtu_output = _run([
            sys.executable, str(EVAL_SCRIPT),
            f"--data={eval_mesh}",
            f"--scan={scan_id}",
            "--mode=mesh",
            f"--dataset_dir={dataset_dir}",
            f"--vis_out_dir={output_dir}",
            "--downsample_method=voxel",
            "--downsample_density=0.2",
            "--patch_size=60",
            "--max_dist=20",
            "--visualize_threshold=10",
            "--apply_scale_mat",
            f"--scale_mat_path={Path(data_root) / 'cameras.npz'}",
        ], env=env, cwd=PROJECT_ROOT, capture=True)
        if ok:
            # eval_DTU_mesh.py prints "mean_d2s mean_s2d overall" as the last line.
            m = re.search(r"(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s*$", dtu_output)
            if m:
                with open(results_json, "w") as f:
                    json.dump({
                        "mean_d2s": float(m.group(1)),
                        "mean_s2d": float(m.group(2)),
                        "overall": float(m.group(3)),
                    }, f, indent=2)
            else:
                print(f"[warn] could not parse DTU output for {exp_name}")
        else:
            return {"exp_name": exp_name, "status": "failed_dtu_eval"}
    else:
        print(f"[skip] DTU results exist or skipped: {results_json}")

    # 4. UV textured mesh
    if not skip_uv and not uv_mesh.exists():
        ok, _ = _run([
            sys.executable, str(EXTRACT_SCRIPT),
            "--single_gpu",
            f"--config={config_path}",
            f"--checkpoint={checkpoint_path}",
            f"--input_mesh={eval_mesh}",
            "--input_mesh_space=world",
            f"--output_file={uv_mesh}",
            "--uv_textured",
            "--uv_raster=gpu",
            f"--texture_size={uv_texture_size}",
            f"--uv_target_faces={uv_target_faces}",
            "--uv_remesh",
            "--uv_remesh_position=after",
            "--uv_no_lcc",
            f"--uv_weld_tol={DEFAULT_UV_WELD_TOL}",
            f"--data.root={data_root}",
        ], env=env, cwd=PROJECT_ROOT)
        if not ok:
            return {"exp_name": exp_name, "status": "failed_uv_mesh"}
    else:
        print(f"[skip] UV mesh exists or skipped: {uv_mesh}")

    # 5. UV PSNR
    if not skip_uv and (not psnr_log.exists() or psnr_log.stat().st_size == 0):
        uv_debug_dir.mkdir(parents=True, exist_ok=True)
        ok, _ = _run([
            sys.executable, str(UV_EVAL_SCRIPT),
            "--single_gpu",
            f"--config={config_path}",
            f"--checkpoint={checkpoint_path}",
            f"--mesh={uv_mesh}",
            "--color_mode=uv",
            "--no_flip_y",
            "--split=val",
            "--ignore_alpha",
            "--uv_project_to_surface",
            "--uv_project_iters=1",
            "--uv_project_step=1.0",
            "--uv_project_max_step=0.0",
            f"--texture_size={uv_texture_size}",
            "--mesh_space=world",
            "--uv_raster=gpu",
            f"--debug_dir={uv_debug_dir}",
            "--debug_every=10",
            "--debug_max_views=20",
            f"--data.root={data_root}",
        ], env=env, cwd=PROJECT_ROOT)
        if not ok:
            return {"exp_name": exp_name, "status": "failed_uv_psnr"}
    else:
        print(f"[skip] UV PSNR exists or skipped: {psnr_log}")

    # Collect results
    record = {"exp_name": exp_name, "status": "ok"}
    try:
        import trimesh
        if vis_mesh.exists():
            record["vis_faces"] = len(trimesh.load(vis_mesh, process=False).faces)
        if eval_mesh.exists():
            record["eval_faces"] = len(trimesh.load(eval_mesh, process=False).faces)
        if uv_mesh.exists():
            record["uv_faces"] = len(trimesh.load(uv_mesh, process=False).faces)
    except Exception as e:
        print(f"[warn] counting faces: {e}")
    if results_json.exists():
        try:
            with open(results_json) as f:
                record.update(json.load(f))
        except Exception as e:
            print(f"[warn] reading {results_json}: {e}")
    if psnr_log.exists():
        try:
            lines = psnr_log.read_text(errors="ignore").splitlines()
            for line in reversed(lines):
                m = re.search(r"Average PSNR \(([^)]+)\):\s*([\d.]+)\s*over\s*(\d+)", line)
                if m:
                    record["uv_psnr_split"] = m.group(1)
                    record["uv_psnr"] = float(m.group(2))
                    record["uv_psnr_views"] = int(m.group(3))
                    break
        except Exception as e:
            print(f"[warn] reading {psnr_log}: {e}")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/home/sunny/data/DTU/scan24")
    parser.add_argument("--dataset_dir", default="/home/sunny/data/DTU/EVAL")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "meshout/eval_current_state"))
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--block_res", type=int, default=DEFAULT_BLOCK_RES)
    parser.add_argument("--uv_texture_size", type=int, default=DEFAULT_UV_TEXTURE_SIZE)
    parser.add_argument("--uv_target_faces", type=int, default=DEFAULT_UV_TARGET_FACES)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--skip_uv", action="store_true")
    parser.add_argument("--skip_dtuv", action="store_true")
    parser.add_argument("--experiments", default="", help="Comma-separated experiment names under logs/test")
    args = parser.parse_args()

    log_root = PROJECT_ROOT / "logs/test"
    if args.experiments:
        exp_names = [x.strip() for x in args.experiments.split(",") if x.strip()]
    else:
        # Default set of interesting experiments.
        exp_names = [
            "baseline",
            "sdf_fast_train_baseline",
            "sdf_utlurfast",
            "sdf_fast_alpha_depth_embaded",
        ]

    records = []
    for exp_name in exp_names:
        exp_dir = log_root / exp_name
        if not exp_dir.is_dir():
            print(f"[skip] experiment dir not found: {exp_dir}")
            continue
        config_path = exp_dir / "config.yaml"
        ckpts = sorted(exp_dir.glob("epoch_*_checkpoint.pt"))
        if not ckpts:
            print(f"[skip] no checkpoint found in {exp_dir}")
            continue
        # Use the last checkpoint.
        checkpoint_path = ckpts[-1]
        print(f"\n\n########## Evaluating {exp_name} : {checkpoint_path.name} ##########")
        record = eval_one(
            exp_name=exp_name,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            data_root=args.data_root,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            resolution=args.resolution,
            block_res=args.block_res,
            uv_texture_size=args.uv_texture_size,
            uv_target_faces=args.uv_target_faces,
            gpu_id=args.gpu_id,
            skip_uv=args.skip_uv,
            skip_dtuv=args.skip_dtuv,
        )
        record["checkpoint"] = str(checkpoint_path)
        records.append(record)

    summary_path = Path(args.output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSummary written to {summary_path}")
    for r in records:
        print(r)


if __name__ == "__main__":
    main()
