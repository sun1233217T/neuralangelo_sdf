#!/usr/bin/env python
"""Evaluate teacher implicit rendering PSNR/SSIM/LPIPS on val/train views.

This loads a trained sdf_angelo checkpoint and renders full images using the
original neural SDF + neural RGB (i.e. the teacher), without any mesh/UV
extraction.  It is the quality upper bound for the downstream UV/student
methods.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_F

sys.path.append(os.getcwd())

from imaginaire.config import Config, parse_cmdline_arguments, recursive_update_strict  # noqa: E402
from imaginaire.trainers.utils.get_trainer import get_trainer  # noqa: E402
from imaginaire.utils.distributed import init_dist  # noqa: E402
from imaginaire.utils.gpu_affinity import set_affinity  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Teacher implicit rendering evaluation")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--num_views", type=int, default=0, help="Number of views to evaluate (0 = all)")
    parser.add_argument("--data_root", default="", help="Override data.root")
    parser.add_argument("--image_size", default="", help="Override image size, e.g. 300,400")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--output_dir", default="", help="Output directory for debug images")
    parser.add_argument("--save_images", action="store_true", help="Save render/gt/error images")
    parser.add_argument("--max_views_to_save", type=int, default=8)
    parser.add_argument("--local_rank", type=int, default=os.getenv("LOCAL_RANK", 0))
    parser.add_argument("--single_gpu", action="store_true")
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd


def _setup_lpips(device):
    try:
        import lpips
        return lpips.LPIPS(net="alex").to(device)
    except Exception as e:
        print(f"[warn] LPIPS not available: {e}")
        return None


def _compute_psnr(img1, img2, mask=None):
    # img: [B,3,H,W] or [3,H,W], in [0,1]
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        if mask.shape[2:] != img1.shape[2:]:
            mask = torch_F.interpolate(mask.float(), size=img1.shape[2:], mode="nearest")
        mse = (((img1 - img2) ** 2) * mask).sum() / (mask.sum() * 3.0 + 1e-8)
    else:
        mse = torch_F.mse_loss(img1, img2)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))
    return psnr.item()


def _compute_ssim(img1, img2):
    # img: [3,H,W] numpy in [0,1]
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(img1.transpose(1, 2, 0), img2.transpose(1, 2, 0), channel_axis=2, data_range=1.0)
    except Exception as e:
        print(f"[warn] SSIM failed: {e}")
        return float("nan")


def _compute_lpips(lpips_model, img1, img2):
    # img: [3,H,W] tensor in [0,1]
    if lpips_model is None:
        return float("nan")
    with torch.no_grad():
        # LPIPS expects [-1,1]
        loss = lpips_model(img1 * 2.0 - 1.0, img2 * 2.0 - 1.0)
    return loss.item()


def _to_img(tensor):
    # [3,H,W] -> uint8 [H,W,3]
    arr = (tensor.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return arr


def main():
    args, cfg_cmd = parse_args()
    try:
        set_affinity(args.local_rank)
    except Exception as exc:
        print(f"Warning: set_affinity failed: {exc}")

    cfg = Config(args.config)
    cfg_cmd = parse_cmdline_arguments(cfg_cmd)
    recursive_update_strict(cfg, cfg_cmd)

    if not args.single_gpu:
        os.environ["NCLL_BLOCKING_WAIT"] = "0"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank, rank=-1, world_size=-1)

    # Ensure logdir exists for the checkpointer.
    cfg.logdir = args.output_dir or ""

    # Overrides.
    if args.data_root:
        cfg.data.root = args.data_root
    if args.num_views >= 0:
        cfg.data[args.split].subset = args.num_views if args.num_views > 0 else None
    if args.image_size:
        h, w = [int(x) for x in args.image_size.split(",")]
        cfg.data[args.split].image_size = [h, w]
    cfg.data[args.split].batch_size = args.batch_size

    trainer = get_trainer(cfg, is_inference=True)
    trainer.checkpointer.load(checkpoint_path=args.checkpoint, resume=False, load_opt=False, load_sch=False)
    trainer.set_data_loader(cfg, args.split, shuffle=False, drop_last=False)

    device = next(trainer.model.parameters()).device

    # Set active levels for coarse-to-fine hashgrid encoding.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    current_iter = ckpt.get("iteration", cfg.max_iter)
    model = trainer.model_module
    model.progress = current_iter / cfg.max_iter
    model.current_iteration = current_iter
    if cfg.model.object.sdf.encoding.coarse2fine.enabled:
        model.neural_sdf.set_active_levels(current_iter)
        if cfg.model.object.sdf.gradient.mode == "numerical":
            model.neural_sdf.set_normal_epsilon()
    lpips_model = _setup_lpips(device)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    records = []
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0
    save_count = 0

    if args.split == "train":
        dataloader = trainer.train_data_loader
    else:
        dataloader = trainer.eval_data_loader
    print(f"Evaluating teacher implicit rendering on {args.split} split, {len(dataloader.dataset)} views...")
    start = time.time()
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

            if args.split == "train":
                # Train split returns ray-sampled pixels; use the training forward pass.
                # Disable stratified sampling for deterministic evaluation.
                original_stratified = getattr(model.cfg_render, "stratified", None)
                model.cfg_render.stratified = False
                output = model(data)
                if original_stratified is not None:
                    model.cfg_render.stratified = original_stratified
                rgb_render = output["rgb"]  # [B,R,3]
                rgb_gt = data["image_sampled"]  # [B,R,3]
                alpha = data.get("alpha_sampled", None)

                # Compute PSNR over the entire batch of sampled rays.
                if alpha is not None:
                    alpha = alpha.float()
                    if alpha.dim() == rgb_gt.dim() - 1:
                        alpha = alpha.unsqueeze(-1)
                    mse = (((rgb_render - rgb_gt) ** 2) * alpha).sum() / (alpha.sum() * 3.0 + 1e-8)
                else:
                    mse = torch_F.mse_loss(rgb_render, rgb_gt)
                psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))
                records.append({
                    "view_idx": batch_idx,
                    "psnr": float(psnr),
                    "ssim": float("nan"),
                    "lpips": float("nan"),
                })
                total_psnr += psnr
                count += 1
            else:
                output = model.inference(data)
                rgb_render = output["rgb_map"]  # [B,3,H,W]
                rgb_gt = data["image"]  # [B,3,H,W]
                alpha = data.get("alpha", None)

                for b in range(rgb_render.shape[0]):
                    render_b = rgb_render[b]
                    gt_b = rgb_gt[b]
                    mask_b = alpha[b] if alpha is not None else None

                    psnr = _compute_psnr(render_b, gt_b)
                    ssim = _compute_ssim(render_b.cpu().numpy(), gt_b.cpu().numpy())
                    lpips_v = _compute_lpips(lpips_model, render_b, gt_b)

                    records.append({
                        "view_idx": batch_idx * args.batch_size + b,
                        "psnr": float(psnr),
                        "ssim": float(ssim),
                        "lpips": float(lpips_v),
                    })
                    total_psnr += psnr
                    total_ssim += ssim
                    total_lpips += lpips_v
                    count += 1

                    if args.save_images and output_dir is not None and save_count < args.max_views_to_save:
                        from PIL import Image
                        Image.fromarray(_to_img(render_b)).save(output_dir / f"view_{count-1:06d}_render.png")
                        Image.fromarray(_to_img(gt_b)).save(output_dir / f"view_{count-1:06d}_gt.png")
                        err = (render_b - gt_b).abs()
                        Image.fromarray(_to_img(err)).save(output_dir / f"view_{count-1:06d}_error.png")
                        if mask_b is not None:
                            Image.fromarray((mask_b.squeeze().cpu().numpy() * 255).astype(np.uint8), mode="L").save(
                                output_dir / f"view_{count-1:06d}_mask.png")
                        save_count += 1

    elapsed = time.time() - start
    if args.split == "train":
        avg_ssim = float("nan")
        avg_lpips = float("nan")
    else:
        avg_ssim = float(total_ssim / max(count, 1))
        avg_lpips = float(total_lpips / max(count, 1))
    summary = {
        "split": args.split,
        "num_views": count,
        "avg_psnr": float(total_psnr / max(count, 1)),
        "avg_ssim": avg_ssim,
        "avg_lpips": avg_lpips,
        "total_time_sec": elapsed,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "per_view": records,
    }

    print("\n===== Teacher Implicit Rendering Summary =====")
    print(f"Split: {args.split}, Views: {count}")
    print(f"Avg PSNR: {summary['avg_psnr']:.4f}")
    if args.split != "train":
        print(f"Avg SSIM: {summary['avg_ssim']:.4f}")
        print(f"Avg LPIPS: {summary['avg_lpips']:.4f}")
    print(f"Total time: {elapsed:.1f}s")

    if output_dir is not None:
        out_path = output_dir / "teacher_psnr.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
