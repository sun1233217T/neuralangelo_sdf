import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.append(os.getcwd())

from projects.sdf_angelo.uv_distill.dataset import TeacherViewStore  # noqa: E402
from projects.sdf_angelo.uv_distill.render import (  # noqa: E402
    StudentTextureRenderer,
    checkpoint_size_mb,
    load_student_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate UV student distillation model")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--student_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_mode", choices=["uv", "image", "both"], default="uv")
    parser.add_argument("--eval_split", choices=["all", "train", "val"], default="val")
    parser.add_argument("--image_eval_split", choices=["same", "all", "train", "val"], default="same")
    parser.add_argument("--holdout_every", default=10, type=int)
    parser.add_argument("--view_kinds", default="original,jitter,interp", type=str)
    parser.add_argument("--max_views", default=0, type=int)
    parser.add_argument("--batch_size", default=262144, type=int)
    parser.add_argument("--cache_size", default=8, type=int)
    parser.add_argument("--view_cache_device", choices=["cpu", "gpu", "cuda"], default="gpu")
    parser.add_argument("--preload_views", choices=["none", "selected", "all"], default="selected")
    parser.add_argument("--preload_limit", default=0, type=int)
    parser.add_argument("--no_packed_views", action="store_true")
    parser.add_argument("--config", default="", type=str,
                        help="Required for image-space evaluation.")
    parser.add_argument("--mesh", default="", type=str,
                        help="Required for image-space evaluation.")
    parser.add_argument("--mesh_space", choices=["world", "normalized"], default="world")
    parser.add_argument("--sanitize_mesh", action="store_true")
    parser.add_argument("--uv_padding", default=2, type=int,
                        help="Padding iterations before image-space rasterization.")
    parser.add_argument("--ignore_alpha", action="store_true")
    parser.add_argument("--mask_image_psnr", action="store_true")
    parser.add_argument("--flip_y", dest="flip_y", action="store_true")
    parser.add_argument("--no_flip_y", dest="flip_y", action="store_false")
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--debug_dir", default="", type=str)
    parser.add_argument("--debug_every", default=0, type=int)
    parser.add_argument("--debug_max_views", default=10, type=int)
    parser.set_defaults(flip_y=True)
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd


def _resolve_cache_device(device_name, train_device):
    if device_name in {"gpu", "cuda"}:
        return torch.device("cuda" if train_device.type == "cuda" else "cpu")
    return torch.device("cpu")


def _parse_kinds(text):
    return [token.strip() for token in text.split(",") if token.strip()]


def _select_view_ids(store, include_kinds, eval_split, holdout_every):
    if eval_split == "all":
        return store.get_view_ids(include_kinds=include_kinds)
    train_ids, val_ids = store.split_view_ids(include_kinds=include_kinds, holdout_every=holdout_every)
    return train_ids if eval_split == "train" else val_ids


def _slice_view_ids(view_ids, max_views):
    if max_views and max_views > 0:
        return view_ids[:int(max_views)]
    return view_ids


def _sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sparse_psnr(pred_rgb, target_rgb):
    mse = (pred_rgb - target_rgb).square().mean()
    return -10.0 * torch.log10(mse.clamp_min(1e-10))


def _hwc_to_pil(tensor):
    arr = (tensor.clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
    return Image.fromarray(arr)


def _chw_to_pil(tensor):
    arr = (tensor.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def _mask_to_pil(mask):
    arr = mask.to(dtype=torch.uint8).mul(255).cpu().numpy()
    return Image.fromarray(arr, mode="L")


def _normalize_mask_chw(mask):
    while mask.ndim > 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    return mask


def _should_save_debug(index, debug_every, debug_saved, debug_max_views):
    if debug_saved >= debug_max_views:
        return False
    if debug_every <= 0:
        return index == 0
    return ((index + 1) % debug_every) == 0


def _save_uv_debug(out_dir, view_id, tex_size, coords_yx, pred_rgb, target_rgb, info):
    coords_yx = coords_yx.to(dtype=torch.int64)
    pred_dense = torch.zeros((tex_size, tex_size, 3), device=pred_rgb.device, dtype=torch.float32)
    target_dense = torch.zeros((tex_size, tex_size, 3), device=pred_rgb.device, dtype=torch.float32)
    mask_dense = torch.zeros((tex_size, tex_size), device=pred_rgb.device, dtype=torch.bool)
    pred_dense[coords_yx[:, 0], coords_yx[:, 1]] = pred_rgb
    target_dense[coords_yx[:, 0], coords_yx[:, 1]] = target_rgb
    mask_dense[coords_yx[:, 0], coords_yx[:, 1]] = True
    base = os.path.join(out_dir, f"view_{int(view_id):06d}")
    _hwc_to_pil(pred_dense).save(f"{base}_pred.png")
    _hwc_to_pil(target_dense).save(f"{base}_teacher.png")
    _hwc_to_pil((pred_dense - target_dense).abs()).save(f"{base}_error.png")
    _mask_to_pil(mask_dense).save(f"{base}_mask.png")
    with open(f"{base}_info.json", "w", encoding="utf-8") as file:
        json.dump(info, file, indent=2)


def _save_image_debug(out_dir, view_id, render, target, mask, info):
    base = os.path.join(out_dir, f"view_{int(view_id):06d}")
    _chw_to_pil(render).save(f"{base}_render.png")
    _chw_to_pil(target).save(f"{base}_gt.png")
    _chw_to_pil((render - target).abs()).save(f"{base}_error.png")
    mask_2d = _normalize_mask_chw(mask).squeeze(0) > 0.5
    _mask_to_pil(mask_2d).save(f"{base}_mask.png")
    with open(f"{base}_info.json", "w", encoding="utf-8") as file:
        json.dump(info, file, indent=2)


class ImageSpaceEvaluator:

    def __init__(self, args, cfg_cmd, geometry_meta, device):
        if not args.config:
            raise ValueError("--config is required for image-space evaluation.")
        if not args.mesh:
            raise ValueError("--mesh is required for image-space evaluation.")
        from imaginaire.config import Config, parse_cmdline_arguments, recursive_update_strict  # noqa: E402
        from projects.sdf_angelo.scripts.uv_mesh_psnr import (  # noqa: E402
            MeshRenderer,
            _build_dataset,
            _load_full_sample,
            _load_mesh,
            _validate_or_sanitize_mesh,
            compute_psnr,
        )

        self._load_full_sample = _load_full_sample
        self.compute_psnr = compute_psnr
        self.device = device
        cfg = Config(args.config)
        if cfg_cmd:
            cfg_cmd = parse_cmdline_arguments(cfg_cmd)
            recursive_update_strict(cfg, cfg_cmd)
        dataset_split = geometry_meta.get("split", "train")
        self.dataset = _build_dataset(cfg, dataset_split)
        self.background = 1.0 if cfg.model.background.white else 0.0

        mesh = _load_mesh(args.mesh)
        mesh = _validate_or_sanitize_mesh(mesh, sanitize=args.sanitize_mesh)
        if not hasattr(mesh, "visual") or mesh.visual is None or getattr(mesh.visual, "uv", None) is None:
            raise ValueError("Input mesh does not contain UV coordinates.")
        sphere_center = np.asarray(geometry_meta["sphere_center"], dtype=np.float32)
        sphere_radius = float(geometry_meta["sphere_radius"])
        if args.mesh_space == "world":
            vertices_norm = (np.asarray(mesh.vertices, dtype=np.float32) - sphere_center[None, :]) / sphere_radius
        else:
            vertices_norm = np.asarray(mesh.vertices, dtype=np.float32)
        uvs = np.asarray(mesh.visual.uv, dtype=np.float32)
        if uvs.shape[0] != vertices_norm.shape[0]:
            raise ValueError("UVs must be per-vertex and match mesh vertices.")
        faces = np.asarray(mesh.faces, dtype=np.int32)
        self.renderer = MeshRenderer(vertices_norm, faces, device=device, uvs=uvs, flip_y=args.flip_y)
        self.mask_image_psnr = bool(args.mask_image_psnr)
        self.ignore_alpha = bool(args.ignore_alpha)

    @torch.no_grad()
    def evaluate_view(self, view_meta, texture):
        dataset_index = int(view_meta["dataset_index"])
        image, intr, pose = self._load_full_sample(self.dataset, dataset_index, ignore_alpha=self.ignore_alpha)
        image = image.to(device=self.device, dtype=torch.float32)
        intr = intr.to(device=self.device, dtype=torch.float32)
        pose = pose.to(device=self.device, dtype=torch.float32)
        render, mask = self.renderer.render(
            texture=texture,
            intr=intr,
            pose=pose,
            image_size=image.shape[-2:],
            background=self.background,
        )
        mask = _normalize_mask_chw(mask)
        if self.mask_image_psnr:
            psnr = self.compute_psnr(render, image, mask=mask)
        else:
            psnr = self.compute_psnr(render, image, mask=None)
        l1 = (render - image).abs().mean()
        return {
            "render": render,
            "target": image,
            "mask": mask,
            "psnr": float(psnr.item()),
            "l1": float(l1.item()),
            "image_size": [int(image.shape[-2]), int(image.shape[-1])],
        }


def main():
    args, cfg_cmd = parse_args()
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

    uv_view_ids = _select_view_ids(
        store,
        include_kinds=_parse_kinds(args.view_kinds),
        eval_split=args.eval_split,
        holdout_every=args.holdout_every,
    )
    uv_view_ids = _slice_view_ids(uv_view_ids, args.max_views)

    image_view_ids = []
    image_eval_split = args.eval_split if args.image_eval_split == "same" else args.image_eval_split
    image_eval_split_effective = image_eval_split
    if args.eval_mode in {"image", "both"}:
        image_view_ids = _select_view_ids(
            store,
            include_kinds=["original"],
            eval_split=image_eval_split,
            holdout_every=args.holdout_every,
        )
        if not image_view_ids and image_eval_split == "val":
            image_view_ids = _select_view_ids(
                store,
                include_kinds=["original"],
                eval_split="all",
                holdout_every=args.holdout_every,
            )
            image_eval_split_effective = "all"
            print(
                "Warning: no original views landed in image val split; "
                "falling back to all original views for image-space evaluation."
            )
        image_view_ids = _slice_view_ids(image_view_ids, args.max_views)

    preload_ids = []
    if args.preload_views == "selected":
        preload_ids = sorted(set(uv_view_ids + image_view_ids))
    elif args.preload_views == "all":
        preload_ids = sorted(set(store.get_view_ids()))
    if preload_ids:
        preload_count = len(preload_ids) if args.preload_limit <= 0 else min(len(preload_ids), int(args.preload_limit))
        store.ensure_cache_capacity(preload_count)
        print(f"Preloading {preload_count} view payloads on {store.cache_device.type}.")
        store.preload_view_ids(preload_ids, limit=args.preload_limit, verbose=True)

    model, checkpoint = load_student_checkpoint(
        args.student_ckpt,
        texture_size=store.geometry["tex_size"],
        device=device,
    )
    renderer = StudentTextureRenderer(model, store.geometry, store.base_rgb_u8)

    summary = {
        "student_ckpt": args.student_ckpt,
        "student_ckpt_mb": checkpoint_size_mb(args.student_ckpt),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "best_val_psnr": checkpoint.get("best_val_psnr", None),
        "texture_size": int(store.geometry["tex_size"]),
        "dataset_dir": args.dataset_dir,
        "eval_mode": args.eval_mode,
        "eval_split": args.eval_split,
        "image_eval_split_requested": args.image_eval_split,
        "image_eval_split_effective": image_eval_split_effective,
        "holdout_every": int(args.holdout_every),
        "view_kinds": _parse_kinds(args.view_kinds),
        "device": str(device),
        "view_cache_device": str(store.cache_device),
    }

    debug_uv_dir = ""
    debug_img_dir = ""
    if args.debug_dir:
        debug_uv_dir = os.path.join(args.debug_dir, "uv")
        debug_img_dir = os.path.join(args.debug_dir, "image")
        os.makedirs(debug_uv_dir, exist_ok=True)
        os.makedirs(debug_img_dir, exist_ok=True)

    uv_log_path = os.path.join(args.output_dir, "uv_metrics.jsonl")
    image_log_path = os.path.join(args.output_dir, "image_metrics.jsonl")
    debug_saved_uv = 0
    debug_saved_img = 0

    if args.eval_mode in {"uv", "both"}:
        if not uv_view_ids:
            raise ValueError("No UV evaluation views selected.")
        uv_l1_sum = 0.0
        uv_psnr_sum = 0.0
        uv_pred_time = 0.0
        for eval_idx, view_id in enumerate(uv_view_ids):
            payload = store.get_view_payload(view_id)
            start_time = time.perf_counter()
            pred_rgb = renderer.predict_sparse(
                sparse_idx=payload["sparse_idx"],
                uv=payload["uv"],
                base_rgb=payload["base_rgb_u8"],
                camera_center_norm=payload["camera_center_norm"],
                batch_size=args.batch_size,
            )
            _sync_if_needed(device)
            pred_time_ms = 1000.0 * (time.perf_counter() - start_time)
            target_rgb = payload["target_rgb_u8"]
            if target_rgb.device != device:
                target_rgb = target_rgb.to(device, non_blocking=True)
            target_rgb = target_rgb.float() / 255.0
            l1 = (pred_rgb - target_rgb).abs().mean()
            psnr = _sparse_psnr(pred_rgb, target_rgb)
            record = {
                "view_id": int(view_id),
                "kind": store.views[int(view_id)]["kind"],
                "visible_texels": int(payload["coords_yx"].shape[0]),
                "uv_l1": float(l1.item()),
                "uv_psnr": float(psnr.item()),
                "predict_ms": pred_time_ms,
            }
            with open(uv_log_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(record) + "\n")
            uv_l1_sum += record["uv_l1"]
            uv_psnr_sum += record["uv_psnr"]
            uv_pred_time += pred_time_ms
            if args.log_every > 0 and ((eval_idx + 1) % args.log_every == 0 or eval_idx == 0):
                print(f"[uv {eval_idx + 1}/{len(uv_view_ids)}] psnr={record['uv_psnr']:.4f} l1={record['uv_l1']:.6f}")
            if debug_uv_dir and _should_save_debug(eval_idx, args.debug_every, debug_saved_uv, args.debug_max_views):
                coords_yx = payload["coords_yx"]
                if coords_yx.device != device:
                    coords_yx = coords_yx.to(device, non_blocking=True)
                debug_info = dict(record)
                debug_info["camera_center_norm"] = payload["camera_center_norm"].detach().cpu().tolist()
                _save_uv_debug(
                    debug_uv_dir,
                    view_id,
                    store.geometry["tex_size"],
                    coords_yx,
                    pred_rgb,
                    target_rgb,
                    debug_info,
                )
                debug_saved_uv += 1
        summary["uv"] = {
            "views": len(uv_view_ids),
            "avg_l1": uv_l1_sum / max(len(uv_view_ids), 1),
            "avg_psnr": uv_psnr_sum / max(len(uv_view_ids), 1),
            "avg_predict_ms": uv_pred_time / max(len(uv_view_ids), 1),
        }
        print(f"UV summary: {summary['uv']}")

    if args.eval_mode in {"image", "both"}:
        if not image_view_ids:
            raise ValueError("No image evaluation views selected. Image evaluation uses original views only.")
        image_evaluator = ImageSpaceEvaluator(
            args=args,
            cfg_cmd=cfg_cmd,
            geometry_meta={
                "split": store.meta.get("split", "train"),
                "sphere_center": store.geometry["sphere_center"],
                "sphere_radius": store.geometry["sphere_radius"],
            },
            device=device,
        )
        img_l1_sum = 0.0
        img_psnr_sum = 0.0
        tex_time_sum = 0.0
        render_time_sum = 0.0
        for eval_idx, view_id in enumerate(image_view_ids):
            view_meta = store.views[int(view_id)]
            payload = store.get_view_payload(view_id)
            tex_start = time.perf_counter()
            texture = renderer.render_texture(
                camera_center_norm=payload["camera_center_norm"],
                batch_size=args.batch_size,
                pad_iters=args.uv_padding,
            )
            _sync_if_needed(device)
            tex_ms = 1000.0 * (time.perf_counter() - tex_start)

            render_start = time.perf_counter()
            result = image_evaluator.evaluate_view(view_meta, texture)
            _sync_if_needed(device)
            render_ms = 1000.0 * (time.perf_counter() - render_start)

            record = {
                "view_id": int(view_id),
                "dataset_index": int(view_meta["dataset_index"]),
                "kind": view_meta["kind"],
                "image_l1": float(result["l1"]),
                "image_psnr": float(result["psnr"]),
                "texture_ms": tex_ms,
                "render_ms": render_ms,
                "image_size": result["image_size"],
            }
            with open(image_log_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(record) + "\n")
            img_l1_sum += record["image_l1"]
            img_psnr_sum += record["image_psnr"]
            tex_time_sum += tex_ms
            render_time_sum += render_ms
            if args.log_every > 0 and ((eval_idx + 1) % args.log_every == 0 or eval_idx == 0):
                print(f"[img {eval_idx + 1}/{len(image_view_ids)}] psnr={record['image_psnr']:.4f} l1={record['image_l1']:.6f}")
            if debug_img_dir and _should_save_debug(eval_idx, args.debug_every, debug_saved_img, args.debug_max_views):
                debug_info = dict(record)
                debug_info["camera_center_norm"] = view_meta["camera_center_norm"]
                _save_image_debug(
                    debug_img_dir,
                    view_id,
                    result["render"],
                    result["target"],
                    result["mask"],
                    debug_info,
                )
                debug_saved_img += 1
        summary["image"] = {
            "views": len(image_view_ids),
            "avg_l1": img_l1_sum / max(len(image_view_ids), 1),
            "avg_psnr": img_psnr_sum / max(len(image_view_ids), 1),
            "avg_texture_ms": tex_time_sum / max(len(image_view_ids), 1),
            "avg_render_ms": render_time_sum / max(len(image_view_ids), 1),
        }
        print(f"Image summary: {summary['image']}")

    with open(os.path.join(args.output_dir, "eval_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


if __name__ == "__main__":
    main()
