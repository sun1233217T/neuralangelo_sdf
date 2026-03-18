import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as torch_F

sys.path.append(os.getcwd())

from projects.sdf_angelo.uv_distill.dataset import TeacherViewStore  # noqa: E402
from projects.sdf_angelo.uv_distill.model import UVResidualStudent  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train UV residual student")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--steps", default=20000, type=int)
    parser.add_argument("--batch_views", default=2, type=int)
    parser.add_argument("--texels_per_view", default=32768, type=int)
    parser.add_argument("--latent_dim", default=8, type=int)
    parser.add_argument("--latent_scale", default=4, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--latent_l2", default=1e-5, type=float)
    parser.add_argument("--latent_tv", default=1e-5, type=float)
    parser.add_argument("--holdout_every", default=10, type=int)
    parser.add_argument("--view_kinds", default="original,jitter,interp", type=str)
    parser.add_argument("--restrict_to_base_valid", action="store_true")
    parser.add_argument("--log_every", default=50, type=int)
    parser.add_argument("--eval_every", default=500, type=int)
    parser.add_argument("--save_every", default=1000, type=int)
    parser.add_argument("--cache_size", default=4, type=int)
    parser.add_argument("--view_cache_device", choices=["cpu", "gpu", "cuda"], default="cpu")
    parser.add_argument("--preload_views", choices=["none", "train", "all"], default="train")
    parser.add_argument("--preload_limit", default=0, type=int)
    parser.add_argument("--reuse_views_steps", default=8, type=int)
    parser.add_argument("--no_packed_views", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def _resolve_cache_device(device_name, train_device):
    if device_name in {"gpu", "cuda"}:
        return torch.device("cuda" if train_device.type == "cuda" else "cpu")
    return torch.device("cpu")


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _concat_batch(samples, device):
    if len(samples) == 1:
        out = {}
        for key, value in samples[0].items():
            if isinstance(value, torch.Tensor):
                out[key] = value.to(device)
            else:
                out[key] = value
        return out
    out = {}
    for key in samples[0].keys():
        if isinstance(samples[0][key], torch.Tensor):
            out[key] = torch.cat([sample[key] for sample in samples], dim=0).to(device)
        else:
            out[key] = [sample[key] for sample in samples]
    return out


@torch.no_grad()
def evaluate(store, model, view_ids, args, device, rng, sample_generator):
    if not view_ids:
        return {}
    model.eval()
    view_subset = view_ids if len(view_ids) <= args.batch_views else rng.choice(view_ids, size=args.batch_views, replace=False).tolist()
    losses = []
    psnrs = []
    for view_id in view_subset:
        batch = store.sample_view_texels(
            view_id,
            num_samples=args.texels_per_view,
            restrict_to_base_valid=args.restrict_to_base_valid,
            rng=sample_generator,
        )
        delta = model(
            batch["uv"].to(device),
            batch["points"].to(device),
            batch["normals"].to(device),
            batch["base_rgb"].to(device),
            batch["camera_center_norm"].to(device),
        )
        pred_rgb = (batch["base_rgb"].to(device) + delta).clamp(0.0, 1.0)
        target_rgb = batch["target_rgb"].to(device)
        loss = torch_F.l1_loss(pred_rgb, target_rgb)
        mse = torch_F.mse_loss(pred_rgb, target_rgb)
        psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))
        losses.append(loss.item())
        psnrs.append(psnr.item())
    return {
        "val_l1": float(np.mean(losses)),
        "val_psnr": float(np.mean(psnrs)),
        "val_views": len(view_subset),
    }


def main():
    args = parse_args()
    _set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_device = _resolve_cache_device(args.view_cache_device, device)
    store = TeacherViewStore(
        args.dataset_dir,
        device=device,
        cache_size=args.cache_size,
        cache_device=cache_device,
        use_packed_views=not args.no_packed_views,
        build_packed_views=not args.no_packed_views,
    )
    include_kinds = [token.strip() for token in args.view_kinds.split(",") if token.strip()]
    train_view_ids, val_view_ids = store.split_view_ids(include_kinds=include_kinds, holdout_every=args.holdout_every)
    if not train_view_ids:
        raise ValueError("No training views selected.")
    preload_ids = []
    if args.preload_views == "train":
        preload_ids = list(train_view_ids)
    elif args.preload_views == "all":
        preload_ids = sorted(set(train_view_ids + val_view_ids))
    if preload_ids:
        preload_count = len(preload_ids) if args.preload_limit <= 0 else min(len(preload_ids), int(args.preload_limit))
        store.ensure_cache_capacity(preload_count)
        print(f"Preloading {preload_count} view payloads on {store.cache_device.type}.")
        store.preload_view_ids(preload_ids, limit=args.preload_limit, verbose=True)
    model = UVResidualStudent(
        texture_size=store.geometry["tex_size"],
        latent_dim=args.latent_dim,
        latent_scale=args.latent_scale,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    sample_gen_device = store.cache_device if store.cache_device.type == "cuda" else torch.device("cpu")
    sample_generator = torch.Generator(device=sample_gen_device)
    sample_generator.manual_seed(args.seed)
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    latest_path = os.path.join(args.output_dir, "student_latest.pt")
    best_psnr = None
    active_view_ids = None
    reuse_views_steps = max(int(args.reuse_views_steps), 1)
    log_window_start = time.perf_counter()
    data_time_acc = 0.0
    window_steps = 0
    for step in range(1, int(args.steps) + 1):
        model.train()
        iter_data_start = time.perf_counter()
        if active_view_ids is None or ((step - 1) % reuse_views_steps == 0):
            active_view_ids = rng.choice(
                train_view_ids,
                size=min(len(train_view_ids), args.batch_views),
                replace=False,
            ).tolist()
        batch_samples = []
        for view_id in active_view_ids:
            batch_samples.append(
                store.sample_view_texels(
                    view_id,
                    num_samples=args.texels_per_view,
                    restrict_to_base_valid=args.restrict_to_base_valid,
                    rng=sample_generator,
                )
            )
        batch = _concat_batch(batch_samples, device=device)
        data_time_acc += time.perf_counter() - iter_data_start
        window_steps += 1
        delta = model(batch["uv"], batch["points"], batch["normals"], batch["base_rgb"], batch["camera_center_norm"])
        pred_rgb = (batch["base_rgb"] + delta).clamp(0.0, 1.0)
        target_rgb = batch["target_rgb"]
        recon_loss = torch_F.l1_loss(pred_rgb, target_rgb)
        latent_l2 = model.latent_regularization()
        latent_tv = model.latent_tv()
        loss = recon_loss + args.latent_l2 * latent_l2 + args.latent_tv * latent_tv
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step % args.log_every == 0 or step == 1:
            if device.type == "cuda":
                torch.cuda.synchronize()
            mse = torch_F.mse_loss(pred_rgb.detach(), target_rgb.detach())
            psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))
            wall_now = time.perf_counter()
            record = {
                "step": step,
                "train_l1": float(recon_loss.item()),
                "train_psnr": float(psnr.item()),
                "latent_l2": float(latent_l2.item()),
                "latent_tv": float(latent_tv.item()),
                "avg_step_ms": float(1000.0 * (wall_now - log_window_start) / max(window_steps, 1)),
                "avg_data_ms": float(1000.0 * data_time_acc / max(window_steps, 1)),
            }
            with open(log_path, "a") as file:
                file.write(json.dumps(record) + "\n")
            print(record)
            log_window_start = time.perf_counter()
            data_time_acc = 0.0
            window_steps = 0

        if step % args.eval_every == 0 and val_view_ids:
            stats = evaluate(store, model, val_view_ids, args, device, rng, sample_generator)
            stats["step"] = step
            with open(log_path, "a") as file:
                file.write(json.dumps(stats) + "\n")
            print(stats)
            if best_psnr is None or stats["val_psnr"] > best_psnr:
                best_psnr = stats["val_psnr"]
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_val_psnr": best_psnr,
                }, os.path.join(args.output_dir, "student_best.pt"))

        if step % args.save_every == 0 or step == args.steps:
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "best_val_psnr": best_psnr,
            }, latest_path)


if __name__ == "__main__":
    main()
