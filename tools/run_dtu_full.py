#!/usr/bin/env python
"""Full DTU benchmark runner for baseline vs. pc_sdf (ours).

This script prepares configs, dispatches training/evaluation across multiple GPUs,
and aggregates results for all 15 standard DTU scans.

Typical workflow on a multi-GPU server:
    python tools/run_dtu_full.py train --gpus 0,1,2,3
    python tools/run_dtu_full.py eval  --gpus 0,1,2,3
    python tools/run_dtu_full.py aggregate

All paths are configurable via command-line arguments.
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time


def _is_nan(v):
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get(
    "PYTHON",
    "/home/sunny/miniconda3/envs/neuralangelo/bin/python"
))
TRAIN_PY = PROJECT_ROOT / "train.py"
EXTRACT_SCRIPT_SDF_ANGELO = PROJECT_ROOT / "projects/sdf_angelo/scripts/extract_mesh.py"
EXTRACT_SCRIPT_NEURALANGELO = PROJECT_ROOT / "projects/neuralangelo/scripts/extract_mesh.py"
EVAL_SCRIPT = PROJECT_ROOT / "projects/sdf_angelo/scripts/eval_DTU_mesh.py"
TEACHER_SCRIPT = PROJECT_ROOT / "projects/sdf_angelo/scripts/eval_teacher_psnr.py"

DTU_SCANS = [
    "scan24", "scan37", "scan40", "scan55", "scan63",
    "scan65", "scan69", "scan83", "scan97", "scan105",
    "scan106", "scan110", "scan114", "scan118", "scan122",
]

METHOD_CONFIGS = {
    "baseline": PROJECT_ROOT / "projects/sdf_angelo/configs/dtu_baseline.yaml",
    "ours": PROJECT_ROOT / "projects/sdf_angelo/configs/dtu_ours.yaml",
    "neuralangelo": PROJECT_ROOT / "projects/neuralangelo/configs/dtu_generic.yaml",
}

# Method-specific mesh extraction script.
EXTRACT_SCRIPTS = {
    "baseline": EXTRACT_SCRIPT_SDF_ANGELO,
    "ours": EXTRACT_SCRIPT_SDF_ANGELO,
    "neuralangelo": EXTRACT_SCRIPT_NEURALANGELO,
}

DEFAULT_MAX_ITER = 120000
DEFAULT_SAVE_ITER = 20000
DEFAULT_MESH_RES = 1024


def _run(cmd, env=None, cwd=None, capture=False):
    """Run a command and optionally capture output."""
    if cwd is None:
        cwd = PROJECT_ROOT
    if env is None:
        env = os.environ.copy()
    cmd_str = " ".join(str(c) for c in cmd)
    print("\n" + "=" * 80)
    print(cmd_str)
    print("=" * 80)
    if capture:
        result = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        print(result.stdout)
        return result.returncode == 0, result.stdout
    result = subprocess.run(cmd, cwd=cwd, env=env)
    return result.returncode == 0


def parse_scans(scans_str, dtu_root):
    """Resolve scan list from comma-separated string or auto-detect from DTU root."""
    if scans_str:
        scans = [s.strip() for s in scans_str.split(",") if s.strip()]
    else:
        scans = []
        for entry in os.listdir(dtu_root):
            scan_root = Path(dtu_root) / entry
            if not scan_root.is_dir():
                continue
            if (scan_root / "transforms.json").is_file():
                scans.append(entry)
    # Keep only standard DTU scans if they appear, and sort by numeric scan ID.
    scans = [s for s in scans if s in DTU_SCANS]
    return sorted(scans, key=lambda s: scan_id_from_name(s) or 0)


def parse_gpus(gpus_str):
    """Parse GPU list; fall back to GPU 0 if unspecified/invalid."""
    if not gpus_str:
        try:
            import torch
            return list(range(torch.cuda.device_count())) or [0]
        except Exception:
            return [0]
    return [int(g.strip()) for g in gpus_str.split(",") if g.strip()]


def dispatch_jobs(jobs, worker, gpus):
    """Dispatch jobs to GPUs; each job is a tuple passed to worker(gpu, *job)."""
    if not jobs:
        print("No jobs to dispatch.")
        return []
    gpus = list(gpus)
    reserved = set()
    results = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        future_to_job = {}
        while jobs or future_to_job:
            available = [g for g in gpus if g not in reserved]
            while available and jobs:
                gpu = available.pop(0)
                job = jobs.pop(0)
                future = executor.submit(worker, gpu, *job)
                future_to_job[future] = (gpu, job)
                reserved.add(gpu)
            done = [f for f in future_to_job if f.done()]
            for future in done:
                gpu, job = future_to_job.pop(future)
                reserved.discard(gpu)
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"[error] job {job} on GPU {gpu} failed: {e}")
            if jobs or future_to_job:
                time.sleep(5)
    return results


def iter_from_checkpoint(checkpoint_path):
    m = re.search(r"iteration_(\d+)", str(checkpoint_path))
    return int(m.group(1)) if m else -1


def scan_id_from_name(scan_name):
    m = re.search(r"(\d+)", scan_name)
    return int(m.group(1)) if m else None


def build_train_command(method, scan, args):
    """Build training command for one scan/method."""
    logdir = Path(args.logdir) / method / scan
    data_root = Path(args.dtu_root) / scan
    config_path = METHOD_CONFIGS[method]
    cmd = [
        str(PYTHON), str(TRAIN_PY),
        "--single_gpu",
        f"--logdir={logdir}",
        f"--config={config_path}",
        f"--data.root={data_root}",
        f"--max_iter={args.max_iter}",
        f"--checkpoint.save_iter={args.save_iter}",
        f"--checkpoint.save_latest_iter={args.save_iter}",
        "--show_pbar",
    ]
    if args.resume:
        cmd.append("--resume")
    return cmd


def train_worker(gpu, method, scan, args):
    """Train one scan/method on a specific GPU."""
    logdir = Path(args.logdir) / method / scan
    final_ckpt = logdir / f"epoch_{args.max_iter // args.save_iter:05d}_iteration_{args.max_iter:09d}_checkpoint.pt"
    if final_ckpt.exists():
        print(f"[skip] final checkpoint exists: {final_ckpt}")
        return {"method": method, "scan": scan, "status": "skipped", "gpu": gpu}

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = build_train_command(method, scan, args)
    ok = _run(cmd, env=env)
    return {"method": method, "scan": scan, "status": "success" if ok else "failed", "gpu": gpu}


def cmd_train(args):
    """Train all scan/method combinations."""
    scans = parse_scans(args.scans, args.dtu_root)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    gpus = parse_gpus(args.gpus)
    print(f"Training scans: {scans}")
    print(f"Methods: {methods}")
    print(f"GPUs: {gpus}")

    jobs = [(m, s) for m in methods for s in scans]
    results = dispatch_jobs(jobs, lambda gpu, m, s: train_worker(gpu, m, s, args), gpus)

    failed = [r for r in results if r.get("status") == "failed"]
    if failed:
        print(f"[warning] {len(failed)} training jobs failed.")
    return results



def eval_teacher(checkpoint_path, config_path, data_root, output_dir, split="train", gpu_id=None):
    """Evaluate teacher PSNR/SSIM/LPIPS for a checkpoint."""
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


def eval_chamfer(checkpoint_path, config_path, scan_id, data_root, dataset_dir,
                 output_dir, method="baseline", gpu_id=None, mesh_resolution=DEFAULT_MESH_RES):
    """Extract mesh and compute DTU Chamfer distance for a checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = checkpoint_path.stem
    result_json = output_dir / f"results_{scan_id:03d}.json"
    if result_json.exists():
        with open(result_json) as f:
            return json.load(f)

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    extract_script = EXTRACT_SCRIPTS.get(method, EXTRACT_SCRIPT_SDF_ANGELO)

    if method == "neuralangelo":
        # Original Neuralangelo extraction: full SDF mesh, keep largest connected component.
        eval_mesh = output_dir / f"{name}_eval.ply"
        if not eval_mesh.exists():
            ok = _run([
                str(PYTHON), str(extract_script),
                "--single_gpu",
                f"--config={config_path}",
                f"--checkpoint={checkpoint_path}",
                "--resolution", str(mesh_resolution),
                "--block_res", "128",
                "--keep_lcc",
                f"--output_file={eval_mesh}",
                f"--data.root={data_root}",
            ], env=env)
            if not ok:
                return None
    else:
        # SDF-Angelo extraction with alpha/depth visibility filtering.
        vis_mesh = output_dir / f"{name}_vis.obj"
        eval_mesh = output_dir / f"{name}_eval.obj"

        # 1) alpha-visible mesh extraction
        if not vis_mesh.exists():
            ok = _run([
                str(PYTHON), str(extract_script),
                "--single_gpu",
                f"--config={config_path}",
                f"--checkpoint={checkpoint_path}",
                "--resolution", str(mesh_resolution),
                "--block_res", "128",
                "--visible_only",
                f"--output_file={vis_mesh}",
                f"--data.root={data_root}",
            ], env=env)
            if not ok:
                return None

        # 2) depth-visible mesh extraction
        if not eval_mesh.exists():
            ok = _run([
                str(PYTHON), str(extract_script),
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

    # 3) DTU Chamfer evaluation
    ok, output = _run([
        str(PYTHON), str(EVAL_SCRIPT),
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
    ], env=env, capture=True)
    if not ok:
        return None
    if result_json.exists():
        with open(result_json) as f:
            return json.load(f)
    # Fallback parsing if JSON not written.
    m = re.search(r"(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s*$", output, re.MULTILINE)
    if not m:
        print(f"[warn] could not parse Chamfer output for {checkpoint_path}")
        return None
    return {
        "mean_d2s": float(m.group(1)),
        "mean_s2d": float(m.group(2)),
        "overall": float(m.group(3)),
    }


def eval_worker(gpu, method, scan, checkpoint, args):
    """Run all evaluations for one checkpoint."""
    scan_id = scan_id_from_name(scan)
    data_root = Path(args.dtu_root) / scan
    config_path = METHOD_CONFIGS[method]
    method_output = Path(args.output_dir) / method / scan
    it = iter_from_checkpoint(checkpoint)

    print(f"\n##### Evaluating {method}/{scan} @ {it} on GPU {gpu} #####")
    teacher_train = eval_teacher(
        checkpoint, config_path, data_root,
        method_output / f"iter_{it:09d}_teacher_train",
        split="train", gpu_id=gpu,
    )
    teacher_val = None
    if args.eval_val:
        teacher_val = eval_teacher(
            checkpoint, config_path, data_root,
            method_output / f"iter_{it:09d}_teacher_val",
            split="val", gpu_id=gpu,
        )
    chamfer = eval_chamfer(
        checkpoint, config_path, scan_id, data_root, args.eval_dir,
        method_output / f"iter_{it:09d}_mesh",
        method=method, gpu_id=gpu, mesh_resolution=args.mesh_resolution,
    )

    record = {
        "method": method,
        "scan": scan,
        "scan_id": scan_id,
        "iteration": it,
        "checkpoint": str(checkpoint),
    }
    if teacher_train:
        record.update({
            "train_psnr": teacher_train.get("avg_psnr"),
            "train_ssim": teacher_train.get("avg_ssim"),
            "train_lpips": teacher_train.get("avg_lpips"),
        })
    if teacher_val:
        record.update({
            "val_psnr": teacher_val.get("avg_psnr"),
            "val_ssim": teacher_val.get("avg_ssim"),
            "val_lpips": teacher_val.get("avg_lpips"),
        })
    if chamfer:
        record.update({
            "chamfer_d2s": chamfer.get("mean_d2s"),
            "chamfer_s2d": chamfer.get("mean_s2d"),
            "chamfer_overall": chamfer.get("overall"),
        })
    return record


def collect_checkpoints(logdir, methods, scans, max_iter=None):
    """Collect all checkpoints for evaluation."""
    checkpoints = []
    for method in methods:
        for scan in scans:
            method_logdir = Path(logdir) / method / scan
            if not method_logdir.exists():
                continue
            for ckpt in sorted(method_logdir.glob("epoch_*_iteration_*_checkpoint.pt")):
                it = iter_from_checkpoint(ckpt)
                if it < 0:
                    continue
                if max_iter is not None and it > max_iter:
                    continue
                checkpoints.append((method, scan, ckpt))
    return checkpoints


def cmd_eval(args):
    """Evaluate all trained checkpoints."""
    scans = parse_scans(args.scans, args.dtu_root)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    gpus = parse_gpus(args.gpus)
    print(f"Evaluating scans: {scans}")
    print(f"Methods: {methods}")
    print(f"GPUs: {gpus}")

    checkpoints = collect_checkpoints(args.logdir, methods, scans, max_iter=args.max_iter)
    print(f"Found {len(checkpoints)} checkpoints to evaluate.")
    if not checkpoints:
        print("No checkpoints found. Train first.")
        return []

    results = dispatch_jobs(checkpoints, lambda gpu, m, s, c: eval_worker(gpu, m, s, c, args), gpus)

    # Save per-evaluation records.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "all_eval_records.csv"
    if results:
        keys = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved evaluation records: {csv_path}")
    return results



def _mean(values):
    vals = [float(v) for v in values if v is not None and v != ""]
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else None


def load_records(output_dir):
    csv_path = Path(output_dir) / "all_eval_records.csv"
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def format_markdown_table(rows):
    if not rows:
        return ""
    keys = list(rows[0].keys())
    int_keys = {"iteration", "scan_id"}
    lines = []
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
    for r in rows:
        vals = []
        for k in keys:
            v = r.get(k, "")
            if v is None or _is_nan(v):
                v = ""
            elif k in int_keys:
                v = str(int(v))
            elif isinstance(v, float):
                v = f"{v:.4f}"
            vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def cmd_aggregate(args):
    """Aggregate evaluation results into tables and plots."""
    records = load_records(args.output_dir)
    if not records:
        print("No evaluation records found. Run `eval` first.")
        return

    numeric_keys = ["iteration", "train_psnr", "train_ssim", "train_lpips",
                    "val_psnr", "val_ssim", "val_lpips",
                    "chamfer_d2s", "chamfer_s2d", "chamfer_overall"]
    for r in records:
        for k in numeric_keys:
            v = r.get(k)
            if v is None or v == "":
                r[k] = None
            else:
                try:
                    r[k] = float(v)
                except ValueError:
                    r[k] = None

    methods = sorted(set(r["method"] for r in records))
    scans = sorted(set(r["scan"] for r in records), key=lambda s: scan_id_from_name(s) or 0)

    # Final-iteration record per (method, scan).
    final_records = {}
    for r in records:
        key = (r["method"], r["scan"])
        if key not in final_records or (r["iteration"] or -1) > (final_records[key]["iteration"] or -1):
            final_records[key] = r

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Final per-scan summary.
    summary_rows = []
    for method in methods:
        for scan in scans:
            key = (method, scan)
            if key in final_records:
                summary_rows.append(final_records[key])
    if summary_rows:
        summary_csv = output_dir / "final_summary.csv"
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved final summary: {summary_csv}")

    # Method averages at final iteration.
    avg_rows = []
    for method in methods:
        vals = [r for r in final_records.values() if r["method"] == method]
        avg_rows.append({
            "method": method,
            "train_psnr": _mean([v["train_psnr"] for v in vals]),
            "train_ssim": _mean([v["train_ssim"] for v in vals]),
            "train_lpips": _mean([v["train_lpips"] for v in vals]),
            "val_psnr": _mean([v["val_psnr"] for v in vals]),
            "val_ssim": _mean([v["val_ssim"] for v in vals]),
            "val_lpips": _mean([v["val_lpips"] for v in vals]),
            "chamfer_d2s": _mean([v["chamfer_d2s"] for v in vals]),
            "chamfer_s2d": _mean([v["chamfer_s2d"] for v in vals]),
            "chamfer_overall": _mean([v["chamfer_overall"] for v in vals]),
        })
    avg_csv = output_dir / "method_averages.csv"
    with open(avg_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=avg_rows[0].keys())
        writer.writeheader()
        writer.writerows(avg_rows)
    print(f"Saved method averages: {avg_csv}")

    # Markdown report.
    report_path = output_dir / "RESULTS.md"
    with open(report_path, "w") as f:
        f.write("# DTU Full Benchmark Results\n\n")
        f.write(f"- max_iter: {args.max_iter}\n")
        f.write(f"- save_iter: {args.save_iter}\n")
        f.write(f"- mesh_resolution: {args.mesh_resolution}\n")
        f.write(f"- methods: {', '.join(methods)}\n")
        f.write(f"- scans: {', '.join(scans)}\n\n")

        f.write("## Method Averages (final checkpoint)\n\n")
        f.write(format_markdown_table(avg_rows))
        f.write("\n")

        f.write("## Per-Scan Final Results\n\n")
        f.write(format_markdown_table(summary_rows))
        f.write("\n")

        f.write("## Notes\n\n")
        f.write("- `baseline` / `ours` use the `projects.sdf_angelo` trainer and visibility-filtered mesh extraction.\n")
        f.write("- `neuralangelo` uses the original `projects.neuralangelo` trainer and mesh extraction (`--keep_lcc`).\n")
        f.write("- Train PSNR is computed on sampled rays, so train SSIM/LPIPS are not meaningful and left blank.\n")
    print(f"Saved report: {report_path}")

    # Convergence plots.
    plot_convergence(records, methods, output_dir)


def plot_convergence(records, methods, output_dir):
    """Plot per-method metrics averaged across scans over iterations."""
    metrics = [
        ("train_psnr", "Train PSNR (dB)", "upper left"),
        ("val_psnr", "Val PSNR (dB)", "upper left"),
        ("chamfer_overall", "Chamfer Distance (mm)", "upper right"),
    ]

    n_plots = sum(1 for key, _, _ in metrics if any(r.get(key) is not None for r in records))
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    ax_idx = 0
    for key, ylabel, legend_loc in metrics:
        if not any(r.get(key) is not None for r in records):
            continue
        ax = axes[ax_idx]
        ax_idx += 1
        for method in methods:
            rows = [r for r in records if r["method"] == method and r.get(key) is not None]
            if not rows:
                continue
            iters = sorted(set(int(r["iteration"]) for r in rows))
            means = []
            for it in iters:
                vals = [r[key] for r in rows if int(r["iteration"]) == it]
                m = _mean(vals)
                if m is not None:
                    means.append(m)
                else:
                    means.append(float("nan"))
            ax.plot(iters, means, marker="o", label=method)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(loc=legend_loc)
        ax.grid(True, alpha=0.3)

    plot_path = output_dir / "convergence_curves.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved convergence plot: {plot_path}")



def main():
    parser = argparse.ArgumentParser(description="Full DTU benchmark runner")
    parser.add_argument("--dtu_root", default="/home/sunny/data/DTU",
                        help="Root directory containing scanXX folders")
    parser.add_argument("--eval_dir", default="/home/sunny/data/DTU/EVAL",
                        help="DTU evaluation directory (contains ObsMask, Points, etc.)")
    parser.add_argument("--logdir", default="logs/dtu_full",
                        help="Directory to store training logs/checkpoints")
    parser.add_argument("--output_dir", default="meshout/dtu_full",
                        help="Directory to store evaluation outputs and final results")
    parser.add_argument("--methods", default="baseline,ours,neuralangelo",
                        help="Comma-separated method names to run (must be keys of METHOD_CONFIGS)")
    parser.add_argument("--scans", default="",
                        help="Comma-separated scan names; auto-detect if empty")
    parser.add_argument("--max_iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--save_iter", type=int, default=DEFAULT_SAVE_ITER)
    parser.add_argument("--mesh_resolution", type=int, default=DEFAULT_MESH_RES)
    parser.add_argument("--gpus", default="",
                        help="Comma-separated GPU IDs for dispatch; auto-detect if empty")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume training if checkpoints exist")
    parser.add_argument("--eval_val", action="store_true", default=False,
                        help="Also evaluate val PSNR/SSIM/LPIPS (slower)")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="Train all scan/method combinations")
    subparsers.add_parser("eval", help="Evaluate all trained checkpoints")
    subparsers.add_parser("aggregate", help="Aggregate evaluation results")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "aggregate":
        cmd_aggregate(args)


if __name__ == "__main__":
    main()
