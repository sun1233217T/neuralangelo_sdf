#!/usr/bin/env python
"""Run teacher implicit eval + UV mesh PSNR with all val views for selected checkpoints."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEACHER_SCRIPT = PROJECT_ROOT / "projects/sdf_angelo/scripts/eval_teacher_psnr.py"
UV_SCRIPT = PROJECT_ROOT / "projects/sdf_angelo/scripts/uv_mesh_psnr.py"


def _run(cmd, env):
    print("\n" + "=" * 80)
    print(" ".join(str(c) for c in cmd))
    print("=" * 80)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="baseline,sdf_fast_train_baseline,sdf_utlurfast,sdf_fast_alpha_depth_embaded")
    parser.add_argument("--data_root", default="/home/sunny/data/DTU/scan24")
    parser.add_argument("--split", default="val")
    parser.add_argument("--num_views", type=int, default=0, help="0 = all")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "meshout/eval_current_state"))
    parser.add_argument("--gpu_id", type=int, default=None)
    args = parser.parse_args()

    env = os.environ.copy()
    if args.gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    log_root = PROJECT_ROOT / "logs/test"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for exp_name in [x.strip() for x in args.experiments.split(",") if x.strip()]:
        exp_dir = log_root / exp_name
        ckpts = sorted(exp_dir.glob("epoch_*_checkpoint.pt"))
        if not ckpts:
            print(f"[skip] no checkpoint for {exp_name}")
            continue
        checkpoint = ckpts[-1]
        config = exp_dir / "config.yaml"

        exp_out = output_dir / f"{exp_name}_teacher_uv"
        exp_out.mkdir(parents=True, exist_ok=True)
        teacher_json = exp_out / "teacher_psnr.json"
        uv_log = exp_out / "uv_psnr.txt"

        # 1. Teacher implicit eval
        if not teacher_json.exists():
            ok = _run([
                sys.executable, str(TEACHER_SCRIPT),
                "--single_gpu",
                f"--config={config}",
                f"--checkpoint={checkpoint}",
                f"--split={args.split}",
                f"--num_views={args.num_views}",
                f"--data_root={args.data_root}",
                f"--output_dir={exp_out}/teacher_debug",
                "--save_images",
            ], env=env)
            if not ok:
                print(f"[warn] teacher eval failed for {exp_name}")
        else:
            print(f"[skip] teacher eval exists: {teacher_json}")

        # 2. UV mesh PSNR on the same split with all views
        uv_mesh = output_dir / f"{exp_name}_eval_uv.obj"
        if uv_mesh.exists() and not uv_log.exists():
            ok = _run([
                sys.executable, str(UV_SCRIPT),
                "--single_gpu",
                f"--config={config}",
                f"--checkpoint={checkpoint}",
                f"--mesh={uv_mesh}",
                "--color_mode=uv",
                "--no_flip_y",
                f"--split={args.split}",
                "--ignore_alpha",
                "--uv_project_to_surface",
                "--uv_project_iters=1",
                "--uv_project_step=1.0",
                "--uv_project_max_step=0.0",
                "--texture_size=2048",
                "--mesh_space=world",
                "--uv_raster=gpu",
                f"--max_views={args.num_views}",
                f"--debug_dir={exp_out}/uv_debug",
                "--debug_every=10",
                f"--psnr_log={uv_log}",
                "--data.val.subset=",
                f"--data.root={args.data_root}",
            ], env=env)
            if ok:
                # uv_mesh_psnr.py prints to stdout; capture by redirecting not trivial here.
                pass
            else:
                print(f"[warn] uv psnr failed for {exp_name}")
        else:
            print(f"[skip] uv psnr: mesh={uv_mesh.exists()}, log={uv_log.exists()}")

        rec = {"exp_name": exp_name, "checkpoint": str(checkpoint)}
        if teacher_json.exists():
            with open(teacher_json) as f:
                teacher_data = json.load(f)
            rec["teacher_psnr"] = teacher_data.get("avg_psnr")
            rec["teacher_ssim"] = teacher_data.get("avg_ssim")
            rec["teacher_lpips"] = teacher_data.get("avg_lpips")
            rec["teacher_views"] = teacher_data.get("num_views")
        if uv_log.exists():
            lines = uv_log.read_text(errors="ignore").splitlines()
            for line in reversed(lines):
                if "Average PSNR" in line:
                    import re
                    m = re.search(r"Average PSNR \(([^)]+)\):\s*([\d.]+)\s*over\s*(\d+)", line)
                    if m:
                        rec["uv_psnr_split"] = m.group(1)
                        rec["uv_psnr"] = float(m.group(2))
                        rec["uv_psnr_views"] = int(m.group(3))
                        break
        results.append(rec)

    summary_path = output_dir / "teacher_uv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary: {summary_path}")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
