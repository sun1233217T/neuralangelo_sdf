"""Freeze/continuation experiments from a trained checkpoint.

Loads a trained model, optionally freezes SDF or color parameters, runs a
short continuation training, and evaluates PSNR.  Used to isolate whether
color or geometry is the bottleneck.

Usage (from repo root):
    python projects/Bspline-Neus/scripts/freeze_experiment.py \
        --config projects/Bspline-Neus/configs/dtu_scan24_E22b_optimized_500k.yaml \
        --checkpoint logs/E22b_optimized_500k/epoch_20833_iteration_000500000_checkpoint.pt \
        --mode joint --iters 2000

Modes:
    joint    — normal training (control)
    color    — freeze SDF + inv_std, train color only
    geometry — freeze color, train SDF + inv_std
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from importlib import import_module
from imaginaire.config import Config
from imaginaire.utils.gpu_affinity import set_affinity
from projects.neuralangelo.utils.misc import eikonal_loss, curvature_loss

Dataset = import_module("projects.Bspline-Neus.data").Dataset
Model = import_module("projects.Bspline-Neus.model").Model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", choices=["joint", "color", "geometry"], default="joint")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local_rank", type=int, default=os.getenv("LOCAL_RANK", 0))
    return p.parse_args()


def load_model(cfg, checkpoint_path):
    model = Model(cfg.model, cfg.data).cuda()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt["model"].items()}
    model.prepare_load_state_dict(sd)
    model.load_state_dict(sd, strict=False)
    return model


def freeze_params(model, mode):
    """Set requires_grad based on the freeze mode."""
    if mode == "joint":
        return  # nothing frozen
    if mode == "color":
        # Train color only — freeze SDF + inv_std + background.
        for p in model.neural_sdf.parameters():
            p.requires_grad_(False)
        if model.background_nerf is not None:
            for p in model.background_nerf.parameters():
                p.requires_grad_(False)
        print("[freeze] SDF + inv_std + background frozen; training color only")
    elif mode == "geometry":
        # Train SDF + inv_std only — freeze color.
        for p in model.neural_rgb.parameters():
            p.requires_grad_(False)
        if model.background_nerf is not None:
            for p in model.background_nerf.parameters():
                p.requires_grad_(False)
        print("[freeze] Color + background frozen; training SDF + inv_std only")


def get_param_groups(model, cfg):
    """Use the model's own param group logic, filtered by requires_grad."""
    groups = model.get_param_groups(cfg.optim)
    return [{"params": [p for p in g["params"] if p.requires_grad], "lr": g["lr"]}
            for g in groups if any(p.requires_grad for p in g["params"])]


@torch.no_grad()
def evaluate(model, dataset, device, num_images=4):
    """Quick PSNR eval on a few val images."""
    model.eval()
    indices = list(range(0, len(dataset), max(1, len(dataset) // num_images)))[:num_images]
    total_mse = 0.0
    total_pixels = 0
    for idx in indices:
        data = dataset[idx]
        data = {k: (v[None].to(device) if torch.is_tensor(v) else v) for k, v in data.items()}
        output = model.inference(data)
        mse = F.mse_loss(output["rgb_map"], data["image"], reduction="sum")
        total_mse += mse.item()
        total_pixels += data["image"].numel()
    model.train()
    psnr = -10 * torch.log10(torch.tensor(total_mse / total_pixels))
    return psnr.item()


def main():
    args = parse_args()
    set_affinity(args.local_rank)
    torch.manual_seed(args.seed)

    cfg = Config(args.config)
    cfg.data.val.subset = None  # use all 49 images for eval

    train_dataset = Dataset(cfg, is_inference=False)
    val_dataset = Dataset(cfg, is_inference=True)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )

    model = load_model(cfg, args.checkpoint)
    model.train()
    # End-of-training state: fully annealed.
    model.progress = 1.0

    freeze_params(model, args.mode)

    # Build optimizer with only trainable params at endpoint LRs.
    # After 500k with two_steps gamma=10 at 300k/400k: effective LR = base/100.
    param_groups = get_param_groups(model, cfg)
    endpoint_groups = []
    for g in param_groups:
        endpoint_groups.append({"params": g["params"], "lr": g["lr"] / 100.0})
    optim = torch.optim.AdamW(endpoint_groups, weight_decay=1e-5)

    n_trainable = sum(p.numel() for g in endpoint_groups for p in g["params"])
    print(f"[freeze] mode={args.mode}, trainable params: {n_trainable:,}")
    for i, g in enumerate(endpoint_groups):
        n = sum(p.numel() for p in g["params"])
        print(f"  group {i}: lr={g['lr']:.2e}, params={n:,}")

    device = next(model.parameters()).device
    l1_loss = torch.nn.L1Loss()

    # E22b endpoint loss weights (from saved config).
    eikonal_w = 0.0005
    curvature_w = 0.0005 * 0.5  # decayed to init * 0.5 at end

    print(f"\n[freeze] Starting {args.iters}-iteration continuation...")
    print(f"[freeze] loss weights: eikonal={eikonal_w}, curvature={curvature_w}")

    # Initial eval.
    psnr_0 = evaluate(model, val_dataset, device)
    print(f"[eval] iter 0: PSNR = {psnr_0:.4f}")

    data_iter = iter(train_loader)
    for step in range(1, args.iters + 1):
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            data = next(data_iter)
        data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}

        output = model(data)
        loss_render = l1_loss(output["rgb"], data["image_sampled"]) * 3
        loss_eik = eikonal_loss(output["gradients"], outside=output["outside"]) * eikonal_w
        loss_curv = curvature_loss(output["hessians"], outside=output["outside"]) * curvature_w
        loss = loss_render + loss_eik + loss_curv

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % args.eval_every == 0 or step == args.iters:
            psnr = evaluate(model, val_dataset, device)
            print(f"[eval] iter {step}: PSNR = {psnr:.4f} (delta = {psnr - psnr_0:+.4f})")

    print(f"\n[freeze] Done. Mode={args.mode}, {args.iters} iters, "
          f"PSNR: {psnr_0:.4f} -> {psnr:.4f} ({psnr - psnr_0:+.4f})")


if __name__ == "__main__":
    main()
