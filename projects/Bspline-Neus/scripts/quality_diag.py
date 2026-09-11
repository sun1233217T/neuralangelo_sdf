"""Batch validation PSNR + per-image error attribution for Bspline-Neus.

Loads one trainer instance and evaluates a list of checkpoints in ascending
iteration order (hierarchical structure restore only grows, so ascending
order is required).  For the last checkpoint it additionally dumps per-image
GT / render / error-heatmap composites and an error decomposition JSON into
``--out_dir``.

Run from the repository root, e.g.:

    python projects/Bspline-Neus/scripts/quality_diag.py --single_gpu \
        --config projects/Bspline-Neus/configs/dtu_scan24_thb.yaml \
        --checkpoints logs/<run>/epoch_..._checkpoint.pt[,more.pt] \
        --out_dir logs/quality_diag --save_viz
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Workaround for duplicate OpenMP runtimes on Windows (libomp vs libiomp5md).
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from imaginaire.config import Config, recursive_update_strict, parse_cmdline_arguments  # noqa: E402
from imaginaire.utils.distributed import get_world_size, is_master, master_only_print as print  # noqa: E402
from imaginaire.utils.gpu_affinity import set_affinity  # noqa: E402
from imaginaire.trainers.utils.get_trainer import get_trainer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Bspline-Neus quality diagnostics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", required=True,
                        help="Comma-separated checkpoint paths, ascending iteration order.")
    parser.add_argument("--out_dir", default="logs/quality_diag")
    parser.add_argument("--viz_prefix", default="val",
                        help="Filename prefix for per-sample PNG/npz outputs.")
    parser.add_argument("--save_viz", action="store_true",
                        help="Dump per-image composites + error decomposition for the last checkpoint.")
    parser.add_argument("--err_gain", type=float, default=4.0,
                        help="Amplification factor for the error heatmap.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"],
                        help="Dataset split to evaluate on.")
    parser.add_argument("--local_rank", type=int, default=os.getenv("LOCAL_RANK", 0))
    parser.add_argument("--single_gpu", action="store_true")
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd


def _to_np(t):
    return t.detach().float().cpu().numpy()


def _psnr_from_mse(mse):
    return float(-10.0 * np.log10(max(mse, 1e-12)))


def _sobel(gray):
    """Simple 3x3 Sobel gradient magnitude (numpy, no scipy dependency)."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    p = np.pad(gray, 1, mode="edge")

    def conv(k):
        out = np.zeros_like(gray)
        for i in range(3):
            for j in range(3):
                out += k[i, j] * p[i:i + gray.shape[0], j:j + gray.shape[1]]
        return out

    return np.hypot(conv(kx), conv(ky))


def _bucket_stats(se, mask):
    """MSE / PSNR / pixel count / SE share for a boolean mask."""
    n = int(mask.sum())
    if n == 0:
        return dict(pixels=0, mse=None, psnr=None, se_share=0.0)
    se_sum = float(se[mask].sum())
    mse = se_sum / n
    return dict(pixels=n, mse=mse, psnr=_psnr_from_mse(mse),
                se_share=se_sum / max(float(se.sum()), 1e-12))


def analyze_sample(idx, gt, rd, opacity, fg_mask, err_gain, out_path, maps_path=None):
    """Per-image attribution: buckets + composite PNG (+ optional npz maps)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = np.clip(gt, 0.0, 1.0)
    rd_clipped = rd  # keep raw render for PSNR (matches trainer protocol)
    se = ((rd_clipped - gt) ** 2).mean(axis=0)  # [H,W] squared error
    mse = float(se.mean())
    psnr = _psnr_from_mse(mse)

    # Foreground / background split from the GT object mask.
    fg = fg_mask
    bg = ~fg

    # Edge / flat split from GT luminance gradient (top 20% = edge).
    gray = gt.mean(axis=0)
    grad = _sobel(gray)
    thr = np.quantile(grad, 0.8)
    edge = grad >= thr
    flat = ~edge

    stats = dict(
        idx=int(idx),
        overall=dict(mse=mse, psnr=psnr),
        foreground=_bucket_stats(se, fg),
        background=_bucket_stats(se, bg),
        edge=_bucket_stats(se, edge),
        flat=_bucket_stats(se, flat),
        edge_fg=_bucket_stats(se, edge & fg),
        flat_fg=_bucket_stats(se, flat & fg),
        err_grad_corr=float(np.corrcoef(se.ravel(), grad.ravel())[0, 1]),
        opacity_mean=float(opacity.mean()),
        opacity_sat_frac=float((opacity > 0.99).mean()),
        opacity_bg_mean=float(opacity[bg].mean()) if bg.any() else None,
        edge_grad_threshold=float(thr),
    )

    # Composite PNG: GT | render | error heatmap (amplified).
    err = np.abs(rd_clipped - gt).mean(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].imshow(np.clip(gt.transpose(1, 2, 0), 0, 1))
    axes[0].set_title("GT")
    axes[1].imshow(np.clip(rd.transpose(1, 2, 0), 0, 1))
    axes[1].set_title(f"render  PSNR {psnr:.2f} dB")
    im = axes[2].imshow(np.clip(err * err_gain, 0, 1), cmap="inferno", vmin=0, vmax=1)
    axes[2].set_title(f"|err| x{err_gain:g}  (fg {stats['foreground']['psnr']:.2f} / "
                      f"bg {stats['background']['psnr']:.2f}, "
                      f"edge {stats['edge']['psnr']:.2f} / flat {stats['flat']['psnr']:.2f})")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.suptitle(f"val idx {idx}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    if maps_path is not None:
        # Per-pixel maps for cross-run delta analysis (SE + GT-derived masks).
        np.savez_compressed(maps_path, se=se.astype(np.float32), fg=fg, edge=edge)
    return stats


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
        from imaginaire.utils.distributed import init_dist
        init_dist(cfg.local_rank, rank=-1, world_size=-1)
    print(f"Running diagnostics with {get_world_size()} GPUs.")

    cfg.logdir = ""
    # For train-split evaluation we need to return full images.  The dataset's
    # __getitem__ checks self.split, so monkey-patch it to "val".  To make the
    # patch visible to the DataLoader we must avoid worker subprocess copies.
    if args.split == "train":
        cfg.data.num_workers = 0
    trainer = get_trainer(cfg, is_inference=True, seed=0)
    trainer.set_data_loader(cfg, split=args.split)
    if args.split == "train":
        trainer.train_data_loader.dataset.split = "val"
    trainer.current_iteration = 0

    out_dir = Path(args.out_dir)
    if is_master():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Pick the data loader matching the requested split.
    if args.split == "train":
        data_loader = trainer.train_data_loader
    elif args.split == "val":
        data_loader = trainer.eval_data_loader
    else:
        data_loader = trainer.eval_data_loader

    checkpoints = args.checkpoints.split(",")
    trajectory = []
    data_all = None
    for ckpt_path in checkpoints:
        trainer.checkpointer.load(ckpt_path, load_opt=False, load_sch=False)
        trainer.model.eval()
        trainer.current_iteration = trainer.checkpointer.eval_iteration
        data_all = trainer.test(data_loader, mode="val", show_pbar=True)
        if is_master():
            psnr = trainer.metrics["psnr"].item()
            try:
                inv_std = trainer.model_module.neural_sdf.inv_std().item()
            except Exception:
                inv_std = float("nan")
            it = int(trainer.current_iteration)
            trajectory.append(dict(checkpoint=ckpt_path, iteration=it, psnr=psnr, inv_std=inv_std))
            print(f"[traj] iter {it}: PSNR {psnr:.4f}  inv_std {inv_std:.4f}  ({ckpt_path})")

    if not is_master():
        return

    traj_path = out_dir / "psnr_trajectory.json"
    with open(traj_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"Trajectory written to {traj_path}")

    if not args.save_viz:
        return

    # ---- Error attribution for the last checkpoint ----
    dataset = data_loader.dataset
    root = Path(cfg.data.root)
    all_stats = []
    for i, sample_idx in enumerate(data_all["idx"]):
        sample_idx = int(sample_idx)
        gt = _to_np(data_all["image"][i])        # [3,H,W] in [0,1]
        rd = _to_np(data_all["rgb_map"][i])      # [3,H,W]
        opacity = _to_np(data_all["opacity_map"][i])
        if opacity.ndim == 3:
            opacity = opacity[0]  # [H,W]

        # Map dataset index -> GT object mask (mask/NNN.png matches images/0NNN.png).
        file_path = dataset.list[sample_idx]["file_path"]
        stem = Path(file_path).stem
        mask_path = root / "mask" / f"{int(stem):03d}.png"
        if mask_path.exists():
            from PIL import Image
            m = Image.open(mask_path).convert("L").resize((gt.shape[2], gt.shape[1]),
                                                          resample=Image.NEAREST)
            fg_mask = np.asarray(m) > 127
        else:
            print(f"WARNING: mask not found for {file_path} ({mask_path}); "
                  "falling back to opacity>0.5 as foreground")
            fg_mask = opacity > 0.5

        out_png = out_dir / f"{args.viz_prefix}{sample_idx:02d}_compare.png"
        maps_path = out_dir / f"{args.viz_prefix}{sample_idx:02d}_maps.npz"
        stats = analyze_sample(sample_idx, gt, rd, opacity, fg_mask, args.err_gain, out_png,
                               maps_path=maps_path)
        stats["file_path"] = file_path
        all_stats.append(stats)
        print(f"[viz] idx {sample_idx}: PSNR {stats['overall']['psnr']:.3f}  "
              f"fg {stats['foreground']['psnr']:.3f}  bg {stats['background']['psnr']:.3f}  "
              f"edge {stats['edge']['psnr']:.3f}  flat {stats['flat']['psnr']:.3f}  "
              f"-> {out_png}")

    summary_path = out_dir / "error_decomposition.json"
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Decomposition written to {summary_path}")


if __name__ == "__main__":
    main()
