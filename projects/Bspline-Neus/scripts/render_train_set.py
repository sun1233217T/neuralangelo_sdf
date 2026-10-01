"""Render the full training set for a Bspline-Neus checkpoint.

The training dataloader returns sampled rays, so this script instantiates the
same dataset in inference (val) mode, forces it to cover all frames, and renders
each full 800x800 image via :meth:`Model.inference`.  Per-image PSNR and
edge/foreground decompositions are saved alongside GT/render/error composites.

Run from the repository root, e.g.::

    python projects/Bspline-Neus/scripts/render_train_set.py \
        --single_gpu \
        --config projects/Bspline-Neus/configs/dtu_scan24_thb_v3_color_mlp_v12_edge_pe.yaml \
        --checkpoint logs/.../epoch_*_iteration_000500000_checkpoint.pt \
        --out_dir logs/.../quality_diag_train
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from importlib import import_module

from imaginaire.config import Config
from imaginaire.utils.gpu_affinity import set_affinity
_quality_diag = import_module("projects.Bspline-Neus.scripts.quality_diag")
_to_np = _quality_diag._to_np
_psnr_from_mse = _quality_diag._psnr_from_mse
_sobel = _quality_diag._sobel
_bucket_stats = _quality_diag._bucket_stats

Dataset = import_module("projects.Bspline-Neus.data").Dataset
Model = import_module("projects.Bspline-Neus.model").Model


def parse_args():
    parser = argparse.ArgumentParser(description="Render full training set for Bspline-Neus")
    parser.add_argument("--config", required=True, help="Path to the run config.")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint.")
    parser.add_argument("--out_dir", required=True, help="Directory for PNG/JSON outputs.")
    parser.add_argument("--single_gpu", action="store_true", help="Run on a single GPU.")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (dataset images are already preloaded).")
    parser.add_argument("--max_images", type=int, default=None,
                        help="If set, only render the first N training images.")
    parser.add_argument("--err_gain", type=float, default=4.0,
                        help="Amplification factor for the error heatmap.")
    parser.add_argument("--local_rank", type=int, default=os.getenv("LOCAL_RANK", 0))
    return parser.parse_args()


def build_dataset(cfg):
    """Build the dataset in val mode but covering every training frame."""
    # The val subset defaults to 4; clear it so we render all 49 training frames.
    cfg.data.val.subset = None
    # Keep preloading threaded (num_workers >= 1) to avoid the single-thread
    # queue.join() hang in base.Dataset.preload_threading.
    if getattr(cfg.data, "num_workers", 4) == 0:
        cfg.data.num_workers = 4
    return Dataset(cfg, is_inference=True)


def load_model(cfg, checkpoint_path):
    """Build the model and restore the checkpoint (strip 'module.' prefix if any)."""
    model = Model(cfg.model, cfg.data)
    model = model.cuda()
    # The NeuS annealing schedule expects self.progress; use 1.0 for the final
    # 500k checkpoint (current_iteration / max_iter).
    model.progress = 1.0
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    state_dict = checkpoint["model"]
    # Strip a single 'module.' prefix saved by (non-DDP) trainer wrappers.
    state_dict = {
        k[7:] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def analyze_sample(idx, gt, rd, opacity, fg_mask, err_gain, out_path):
    """Compute per-image PSNR decomposition and save a GT/render/error composite."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = np.clip(gt, 0.0, 1.0)
    rd = np.clip(rd, 0.0, 1.0)
    se = ((rd - gt) ** 2).mean(axis=0)
    mse = float(se.mean())
    psnr = _psnr_from_mse(mse)

    fg = fg_mask
    bg = ~fg

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

    err = np.abs(rd - gt).mean(axis=0)
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
    fig.suptitle(f"train idx {idx}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return stats


def get_fg_mask(sample_idx, gt_image, opacity, dataset, root):
    """Load the GT object mask, falling back to opacity > 0.5."""
    from PIL import Image
    file_path = dataset.list[sample_idx]["file_path"]
    stem = Path(file_path).stem
    mask_path = root / "mask" / f"{int(stem):03d}.png"
    if mask_path.exists():
        m = Image.open(mask_path).convert("L").resize(
            (gt_image.shape[2], gt_image.shape[1]), resample=Image.NEAREST)
        return np.asarray(m) > 127
    else:
        print(f"WARNING: mask not found for {file_path} ({mask_path}); "
              "falling back to opacity > 0.5")
        return opacity > 0.5


def main():
    args = parse_args()
    set_affinity(args.local_rank)

    cfg = Config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(cfg)
    print(f"Built dataset with {len(dataset)} training frames (split={dataset.split})")

    model = load_model(cfg, args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False)

    root = Path(cfg.data.root)
    all_stats = []
    max_images = args.max_images or len(dataset)

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(data_loader, desc="Rendering train images")):
            if batch_idx >= max_images:
                break
            # Move tensors to GPU; keep idx on CPU for indexing.
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].cuda()

            output = model.inference(data)

            for b in range(data["image"].shape[0]):
                sample_idx = int(data["idx"][b].cpu())
                out_png = out_dir / f"train{sample_idx:02d}_compare.png"
                if out_png.exists():
                    # Resume support: skip images already rendered.
                    continue
                gt = _to_np(data["image"][b])  # [3,H,W]
                rd = _to_np(output["rgb_map"][b])  # [3,H,W]
                opacity = _to_np(output["opacity_map"][b])  # [1,H,W]
                if opacity.ndim == 3:
                    opacity = opacity[0]
                # Depth (normalized within opaque region, near=bright) and
                # analytic-gradient normal maps for geometric inspection.
                depth = _to_np(output["depth_map"][b])  # [1,H,W]
                if depth.ndim == 3:
                    depth = depth[0]
                normal = _to_np(output["normal_map"][b])  # [3,H,W]
                oq = opacity > 0.5
                depth_v = np.zeros_like(depth)
                if oq.any():
                    lo, hi = depth[oq].min(), depth[oq].max()
                    depth_v[oq] = 1.0 - (depth[oq] - lo) / (hi - lo + 1e-8)
                Image.fromarray((np.clip(depth_v, 0, 1) * 255).astype(np.uint8)).save(
                    out_dir / f"train{sample_idx:02d}_depth.png")
                Image.fromarray((np.clip(normal.transpose(1, 2, 0) * 0.5 + 0.5, 0, 1)
                                 * 255).astype(np.uint8)).save(
                    out_dir / f"train{sample_idx:02d}_normal.png")

                fg_mask = get_fg_mask(sample_idx, gt, opacity, dataset, root)
                out_png = out_dir / f"train{sample_idx:02d}_compare.png"
                stats = analyze_sample(
                    sample_idx, gt, rd, opacity, fg_mask, args.err_gain, out_png)
                stats["file_path"] = dataset.list[sample_idx]["file_path"]
                all_stats.append(stats)
                print(f"[viz] idx {sample_idx}: PSNR {stats['overall']['psnr']:.3f}  "
                      f"fg {stats['foreground']['psnr']:.3f}  "
                      f"edge_fg {stats['edge_fg']['psnr']:.3f}  "
                      f"-> {out_png}")

    summary = {
        "num_images": len(all_stats),
        "overall_mean_psnr": float(np.mean([s["overall"]["psnr"] for s in all_stats])),
        "foreground_mean_psnr": float(np.mean([s["foreground"]["psnr"] for s in all_stats])),
        "edge_fg_mean_psnr": float(np.mean([s["edge_fg"]["psnr"] for s in all_stats])),
        "per_image": all_stats,
    }
    summary_path = out_dir / "error_decomposition.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: overall PSNR {summary['overall_mean_psnr']:.4f}, "
          f"fg {summary['foreground_mean_psnr']:.4f}, "
          f"edge_fg {summary['edge_fg_mean_psnr']:.4f}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
