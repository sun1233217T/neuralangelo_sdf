"""Validation PSNR evaluation for Bspline-Neus checkpoints.

Loads a checkpoint (dense or hierarchical — the checkpointer's
``prepare_load_state_dict`` hook grows the hierarchical fields to the
checkpoint's depth before key/shape filtering) and runs the trainer's own
``test(mode="val")`` path over the validation split, then prints the PSNR
computed by ``_compute_loss``.

Run from the repository root, e.g.:

    python projects/Bspline-Neus/scripts/eval_psnr.py --single_gpu \
        --config projects/Bspline-Neus/configs/dtu_scan24_thb.yaml \
        --checkpoint logs/<run>/epoch_20833_iteration_000500000_checkpoint.pt
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from imaginaire.config import Config, recursive_update_strict, parse_cmdline_arguments  # noqa: E402
from imaginaire.utils.distributed import init_dist, get_world_size, is_master, master_only_print as print  # noqa: E402
from imaginaire.utils.gpu_affinity import set_affinity  # noqa: E402
from imaginaire.trainers.utils.get_trainer import get_trainer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Bspline-Neus validation PSNR")
    parser.add_argument("--config", required=True, help="Path to the training config file.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"],
                        help="Dataset split to evaluate on.")
    parser.add_argument("--local_rank", type=int, default=os.getenv("LOCAL_RANK", 0))
    parser.add_argument("--single_gpu", action="store_true")
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
    print(f"Running validation with {get_world_size()} GPUs.")

    cfg.logdir = ""

    trainer = get_trainer(cfg, is_inference=True, seed=0)
    if args.split == "train":
        # To evaluate the train split on full images (instead of sampled rays)
        # we temporarily use the val dataloader but with train subset/config.
        from imaginaire.datasets.utils.get_dataloader import get_val_dataloader
        train_cfg = cfg.data.train
        cfg.data.val = cfg.data.train
        loader = get_val_dataloader(cfg, seed=0)
    else:
        trainer.set_data_loader(cfg, split=args.split)
        loader = getattr(trainer, "eval_data_loader")
    # The post-model-load hook rebuilds the optimizer when structure restore
    # grows the hierarchical fields, and the rebuild reads current_iteration
    # for the scheduler.  Set a placeholder; the true value is assigned below.
    trainer.current_iteration = 0
    trainer.checkpointer.load(args.checkpoint, load_opt=False, load_sch=False)
    trainer.model.eval()
    trainer.current_iteration = trainer.checkpointer.eval_iteration

    data_all = trainer.test(loader, mode="val", show_pbar=True)

    if is_master():
        psnr = trainer.metrics["psnr"].item()
        print(f"checkpoint: {args.checkpoint}")
        print(f"split: {args.split}")
        print(f"samples: {len(data_all['idx'])}")
        print(f"{args.split} PSNR: {psnr:.4f}")
        # SSIM + LPIPS from the same rendered/GT image pairs.
        if "rgb_map" in data_all and "image" in data_all:
            import numpy as np
            from skimage.metrics import structural_similarity
            rgb = data_all["rgb_map"].detach().float().clamp(0, 1).cpu().numpy()
            gt = data_all["image"].detach().float().clamp(0, 1).cpu().numpy()
            if rgb.ndim == 3:
                rgb, gt = rgb[None], gt[None]
            ssims = [structural_similarity(
                np.moveaxis(g, 0, -1), np.moveaxis(r, 0, -1),
                channel_axis=2, data_range=1.0) for r, g in zip(rgb, gt)]
            ssim = float(np.mean(ssims))
            try:
                import lpips as _lpips
                import torch as _torch
                _net = _lpips.LPIPS(net="vgg").to("cuda")
                with _torch.no_grad():
                    lp = [_net(_torch.from_numpy(r)[None].cuda() * 2 - 1,
                               _torch.from_numpy(g)[None].cuda() * 2 - 1).item()
                          for r, g in zip(rgb, gt)]
                lpips_mean = float(np.mean(lp))
            except Exception as exc:  # noqa: BLE001
                lpips_mean = float("nan")
                print(f"LPIPS unavailable: {exc}")
            print(f"{args.split} SSIM: {ssim:.4f}")
            print(f"{args.split} LPIPS: {lpips_mean:.4f}")


if __name__ == "__main__":
    main()
