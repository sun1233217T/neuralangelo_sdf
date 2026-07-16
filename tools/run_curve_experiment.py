#!/usr/bin/env python
"""Train baseline vs ours and plot convergence curves for DTU scan24.

Pipeline:
  1. Train baseline (no pc_sdf) to max_iter, saving every save_iter.
  2. Train ours (with pc_sdf) to max_iter, saving every save_iter.
  3. For each saved checkpoint, evaluate:
       - teacher implicit PSNR/SSIM/LPIPS on train split
       - Chamfer distance via mesh extraction + DTU eval
  4. Save metrics table and plot curves.
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Use the neuralangelo conda environment Python explicitly; fallback to sys.executable.
PYTHON = Path(os.environ.get("PYTHON", "/home/sunny/miniconda3/envs/neuralangelo/bin/python"))
TRAIN_PY = PROJECT_ROOT / "train.py"
EXTRACT_SCRIPT = PROJECT_ROOT / "projects/sdf_angelo/scripts/extract_mesh.py"
EVAL_SCRIPT = PROJECT_ROOT / "projects/sdf_angelo/scripts/eval_DTU_mesh.py"
TEACHER_SCRIPT = PROJECT_ROOT / "projects/sdf_angelo/scripts/eval_teacher_psnr.py"

DEFAULT_DATA_ROOT = "/home/sunny/data/DTU/scan24"
DEFAULT_DATASET_DIR = "/home/sunny/data/DTU/EVAL"
DEFAULT_MAX_ITER = 100000
DEFAULT_SAVE_ITER = 20000
DEFAULT_MESH_RES = 1024


def _run(cmd, env, cwd=None):
    if cwd is None:
        cwd = PROJECT_ROOT
    print("\n" + "=" * 80)
    print(" ".join(str(c) for c in cmd))
    print("=" * 80)
    result = subprocess.run(cmd, cwd=cwd, env=env)
    return result.returncode == 0


def _extract_and_eval_chamfer(checkpoint_path, config_path, data_root, dataset_dir,
                               output_dir, gpu_id=None):
    """Extract depth-visible mesh and run DTU Chamfer eval."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = checkpoint_path.stem
    vis_mesh = output_dir / f"{name}_vis.obj"
    eval_mesh = output_dir / f"{name}_eval.obj"

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # alpha-visible mesh
    if not vis_mesh.exists():
        ok = _run([
            str(PYTHON), str(EXTRACT_SCRIPT),
            "--single_gpu",
            f"--config={config_path}",
            f"--checkpoint={checkpoint_path}",
            "--resolution", str(DEFAULT_MESH_RES),
            "--block_res", "128",
            "--visible_only",
            f"--output_file={vis_mesh}",
            f"--data.root={data_root}",
        ], env=env)
        if not ok:
            return None

    # depth-visible mesh
    if not eval_mesh.exists():
        ok = _run([
            str(PYTHON), str(EXTRACT_SCRIPT),
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
        ], env=env)
        if not ok:
            return None

    # DTU eval
    ok, output = _run_capture([
        str(PYTHON), str(EVAL_SCRIPT),
        f"--data={eval_mesh}",
        "--scan=24",
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
    ], env=env)
    if not ok:
        return None
    m = re.search(r"(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s*$", output)
    if not m:
        print(f"[warn] could not parse Chamfer output for {name}")
        return None
    return {
        "mean_d2s": float(m.group(1)),
        "mean_s2d": float(m.group(2)),
        "overall": float(m.group(3)),
    }


def _run_capture(cmd, env, cwd=None):
    if cwd is None:
        cwd = PROJECT_ROOT
    result = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    print(result.stdout)
    return result.returncode == 0, result.stdout


def _eval_teacher(checkpoint_path, config_path, data_root, output_dir, gpu_id=None,
                    split="train"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "teacher_psnr.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    ok = _run([
        str(PYTHON), str(TEACHER_SCRIPT),
        "--single_gpu",
        f"--config={config_path}",
        f"--checkpoint={checkpoint_path}",
        f"--split={split}",
        "--num_views=0",
        f"--data_root={data_root}",
        f"--output_dir={output_dir}",
    ], env=env)
    if not ok or not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)


def _parse_iter(checkpoint_path):
    m = re.search(r"iteration_(\d+)", str(checkpoint_path))
    return int(m.group(1)) if m else -1


def train_experiment(exp_name, config_path, logdir, max_iter, save_iter, data_root, gpu_id):
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    # Check if already trained.
    final_ckpt = logdir / f"epoch_{max_iter//save_iter:05}_iteration_{max_iter:09}_checkpoint.pt"
    if final_ckpt.exists():
        print(f"[skip] training exists: {final_ckpt}")
        return True

    cmd = [
        str(PYTHON), str(TRAIN_PY),
        "--single_gpu",
        f"--logdir={logdir}",
        f"--config={config_path}",
        f"--data.root={data_root}",
        f"--max_iter={max_iter}",
        f"--checkpoint.save_iter={save_iter}",
        f"--checkpoint.save_latest_iter={save_iter}",
        "--show_pbar",
        "--resume",
    ]
    return _run(cmd, env=env)


def evaluate_experiment(exp_name, logdir, config_path, data_root, dataset_dir,
                        output_dir, gpu_id, split="train"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(Path(logdir).glob("epoch_*_iteration_*_checkpoint.pt"))
    records = []
    for ckpt in ckpts:
        it = _parse_iter(ckpt)
        if it < 0:
            continue
        print(f"\n##### Evaluating {exp_name} @ {it} #####")
        teacher = _eval_teacher(ckpt, config_path, data_root,
                                output_dir / f"iter_{it:09d}_teacher_{split}",
                                gpu_id, split=split)
        chamfer = _extract_and_eval_chamfer(ckpt, config_path, data_root, dataset_dir,
                                            output_dir / f"iter_{it:09d}_mesh", gpu_id)
        rec = {
            "exp_name": exp_name,
            "iteration": it,
            "checkpoint": str(ckpt),
        }
        if teacher:
            rec.update({
                "teacher_psnr": teacher.get("avg_psnr"),
                "teacher_ssim": teacher.get("avg_ssim"),
                "teacher_lpips": teacher.get("avg_lpips"),
            })
        if chamfer:
            rec.update(chamfer)
        records.append(rec)

    csv_path = output_dir / f"{exp_name}_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        if records:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    print(f"Saved {csv_path}")
    return records


def plot_curves(all_records, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exps = sorted(set(r["exp_name"] for r in all_records))
    iters = sorted(set(r["iteration"] for r in all_records))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [
        ("teacher_psnr", "Teacher PSNR (dB)", "upper left"),
        ("teacher_lpips", "Teacher LPIPS", "upper right"),
        ("overall", "Chamfer Distance", "upper right"),
    ]
    for ax, (key, ylabel, legend_loc) in zip(axes, metrics):
        for exp in exps:
            rows = [r for r in all_records if r["exp_name"] == exp and key in r and r[key] is not None]
            rows = sorted(rows, key=lambda x: x["iteration"])
            if not rows:
                continue
            xs = [r["iteration"] for r in rows]
            ys = [r[key] for r in rows]
            ax.plot(xs, ys, marker="o", label=exp)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(loc=legend_loc)
        ax.grid(True, alpha=0.3)
    plot_path = output_dir / "convergence_curves.png"
    try:
        fig.savefig(plot_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    except Exception as e:
        print(f"[warn] savefig failed with {type(e).__name__}: {e}; retrying without bbox_inches")
        fig.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--max_iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--save_iter", type=int, default=DEFAULT_SAVE_ITER)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--split", default="train",
                        help="Split for teacher PSNR evaluation (train or val)")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_plot", action="store_true")
    args = parser.parse_args()

    experiments = [
        ("curve_baseline", PROJECT_ROOT / "projects/sdf_angelo/configs/curve_baseline.yaml"),
        ("curve_ours", PROJECT_ROOT / "projects/sdf_angelo/configs/curve_ours.yaml"),
    ]

    root = PROJECT_ROOT / "logs/curve_experiment"
    eval_root = PROJECT_ROOT / "meshout/curve_experiment"

    all_records = []
    for exp_name, config_path in experiments:
        logdir = root / exp_name
        eval_dir = eval_root / exp_name

        if not args.skip_train:
            ok = train_experiment(exp_name, config_path, logdir, args.max_iter,
                                  args.save_iter, args.data_root, args.gpu_id)
            if not ok:
                print(f"[error] training failed for {exp_name}")
                continue

        if not args.skip_eval:
            records = evaluate_experiment(exp_name, logdir, config_path, args.data_root,
                                          args.dataset_dir, eval_dir, args.gpu_id,
                                          split=args.split)
            all_records.extend(records)

    if all_records and not args.skip_plot:
        plot_curves(all_records, eval_root)

    summary_path = eval_root / "all_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nAll metrics: {summary_path}")


if __name__ == "__main__":
    main()
