"""Observation-strength analysis: does weak observation cause cloudy depth?

Phase 1 replays all training views and accumulates per-cell anchor hits at
the finest level (observation strength, same criterion as pruning).
Phase 2 computes, per pixel of validation views:
  - the zero-crossing point and its cell's hit count / finest level,
  - the compositing-weight entropy (depth cloudiness proxy),
  - the eikonal residual |grad f| - 1 at the crossing (field convergence).
Binned statistics then show whether entropy and eikonal residual rise as
observation strength falls.

Usage:
    python projects/Bspline-Neus/scripts/obs_strength_analysis.py \
        --config ... --checkpoint ... --outdir logs/T2_thin_prune_500k/obs_strength \
        --views 0 24 --stride 2
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from importlib import import_module

from imaginaire.config import Config

Dataset = import_module("projects.Bspline-Neus.data").Dataset
Model = import_module("projects.Bspline-Neus.model").Model

OFFS = torch.tensor([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                     [0, 0, 1], [0, 0, -1]], device="cuda")


def first_crossing(center, ray_unit, dists, sdfs):
    d = dists[..., 0]
    s = sdfs
    cross = (s[:, :-1] > 0) & (s[:, 1:] <= 0)
    has = cross.any(dim=1)
    idx = cross.float().argmax(dim=1)
    ar = torch.arange(d.shape[0], device=d.device)
    d0, d1 = d[ar, idx], d[ar, idx + 1]
    s0, s1 = s[ar, idx], s[ar, idx + 1]
    t = (s0 / (s0 - s1).clamp_min(1e-8)).clamp(0, 1)
    dc = d0 + t * (d1 - d0)
    return center + ray_unit * dc[:, None], has, idx


def cell_idx(level, p):
    cc = level.cell_count
    idx = ((p - level._lower) / level.step).floor().long()
    ok = ((idx >= 0) & (idx < cc)).all(dim=-1)
    return idx, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--views", type=int, nargs="+", default=[0, 24])
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--train_views", type=int, default=49)
    args = ap.parse_args()

    cfg = Config(args.config)
    ds_train = Dataset(cfg, is_inference=False)
    ds_val = Dataset(cfg, is_inference=True)
    model = Model(cfg.model, cfg.data).cuda()
    model.progress = 1.0
    ckpt = torch.load(args.checkpoint, map_location=lambda s, l: s)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    hier = model.neural_sdf.hier_field
    finest = hier.levels[-1]
    cc = finest.cell_count

    # ---------------- phase 1: anchor hits at the finest level -------------
    hits = torch.zeros((cc,) * 3, dtype=torch.int32, device="cuda")
    st = args.stride
    for vi in range(min(args.train_views, len(ds_train))):
        data = ds_train[vi]
        data = {k: (v[None].cuda() if torch.is_tensor(v) else v) for k, v in data.items()}
        pose, intr = data["pose"], data["intr"].clone()
        H, W = int(cfg.data.val.image_size[0]), int(cfg.data.val.image_size[1])
        intr[:, 0, 0] /= st; intr[:, 1, 1] /= st
        intr[:, 0, 2] /= st; intr[:, 1, 2] /= st
        with torch.no_grad():
            for center, ray, _ in model.ray_generator(pose, intr, (H // st, W // st), full_image=True):
                ray_unit = F.normalize(ray, dim=-1)
                out = model.render_rays(center, ray_unit)
                n_obj = out["gradients"].shape[2]
                pts, has, _ = first_crossing(center[0], ray_unit[0],
                                             out["dists"][0, :, :n_obj], out["sdfs"][0, :, :n_obj])
                if not has.any():
                    continue
                idx, ok = cell_idx(finest, pts[has])
                for o in OFFS:
                    io = idx + o
                    iok = ok & ((io >= 0) & (io < cc)).all(dim=-1)
                    if iok.any():
                        ii = io[iok]
                        hits.index_put_((ii[:, 0], ii[:, 1], ii[:, 2]),
                                        torch.ones(iok.sum(), dtype=torch.int32, device="cuda"),
                                        accumulate=True)
        if vi % 10 == 0:
            print(f"phase1 view {vi}", flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    torch.save(hits.cpu(), f"{args.outdir}/hits_L4.pt")

    # ---------------- phase 2: per-pixel entropy / residual ----------------
    rows = []  # (hit_count, level, entropy, eikonal_resid)
    for vi in args.views:
        data = ds_val[vi]
        data = {k: (v[None].cuda() if torch.is_tensor(v) else v) for k, v in data.items()}
        pose, intr = data["pose"], data["intr"]
        H, W = data["image"].shape[2], data["image"].shape[3]
        with torch.no_grad():
            for center, ray, _ in model.ray_generator(pose, intr, (H, W), full_image=True):
                ray_unit = F.normalize(ray, dim=-1)
                out = model.render_rays(center, ray_unit)
                n_obj = out["gradients"].shape[2]
                dists = out["dists"][0, :, :n_obj]
                sdfs = out["sdfs"][0, :, :n_obj]
                w = out["weights"][0, :, :n_obj, 0]  # (R,No)
                pts, has, _ = first_crossing(center[0], ray_unit[0], dists, sdfs)
                if not has.any():
                    continue
                p = pts[has]
                # entropy of weights (normalized by log N)
                wn = w[has].clamp_min(1e-12)
                wn = wn / wn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                ent = -(wn * wn.log()).sum(dim=-1) / np.log(w.shape[-1])
                # eikonal residual at crossing (analytic gradient)
                _, grads, _ = model.neural_sdf._evaluate_sdf_with_deriv(p)
                eik = (grads.reshape(-1, 3).norm(dim=-1) - 1.0).abs()
                idx, ok = cell_idx(finest, p)
                hcnt = torch.zeros(p.shape[0], dtype=torch.int32, device="cuda")
                if ok.any():
                    hcnt[ok] = hits[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
                # finest level covering each point
                lev = torch.zeros(p.shape[0], dtype=torch.long, device="cuda")
                for li in reversed(range(hier.num_levels)):
                    lv = hier.levels[li]
                    idl, okl = cell_idx(lv, p)
                    inl = okl & lv.region[idl[:, 0].clamp(0, lv.cell_count - 1),
                                          idl[:, 1].clamp(0, lv.cell_count - 1),
                                          idl[:, 2].clamp(0, lv.cell_count - 1)]
                    take = inl & (lev == 0)
                    lev[take] = li
                rows.append(torch.stack([hcnt.float(), lev.float(), ent, eik], dim=-1).cpu())
    rows = torch.cat(rows)
    os.makedirs(args.outdir, exist_ok=True)
    torch.save(rows, f"{args.outdir}/rows.pt")
    torch.save(hits.cpu(), f"{args.outdir}/hits_L4.pt")

    # ---------------- binned report ----------------
    bins = [0, 1, 5, 20, 100, 1000, 1e9]
    labels = ["0", "1-4", "5-19", "20-99", "100-999", "1000+"]
    print(f"\n=== {rows.shape[0]} surface pixels ===")
    print(f"{'hits':>8} {'pixels':>8} {'entropy':>8} {'eik_res':>8}")
    ent_m, eik_m, counts = [], [], []
    for lo, hi, lb in zip(bins[:-1], bins[1:], labels):
        m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        if m.any():
            ent_m.append(rows[m, 2].mean().item())
            eik_m.append(rows[m, 3].mean().item())
            counts.append(int(m.sum()))
            print(f"{lb:>8} {int(m.sum()):>8} {rows[m, 2].mean().item():>8.4f} "
                  f"{rows[m, 3].mean().item():>8.4f}")
    print("\nby finest level:")
    for li in range(hier.num_levels):
        m = rows[:, 1] == li
        if m.any():
            print(f"  L{li}: n={int(m.sum())}, entropy={rows[m, 2].mean():.4f}, "
                  f"eik={rows[m, 3].mean():.4f}, hits_med={rows[m, 0].median():.0f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    used = [lb for lb, (lo, hi) in zip(labels, zip(bins[:-1], bins[1:]))
            if ((rows[:, 0] >= lo) & (rows[:, 0] < hi)).any()]
    ax[0].plot(used, ent_m, "o-"); ax[0].set_xlabel("anchor hits"); ax[0].set_ylabel("weight entropy")
    ax[1].plot(used, eik_m, "o-"); ax[1].set_xlabel("anchor hits"); ax[1].set_ylabel("|grad|-1 residual")
    fig.suptitle("observation strength vs depth cloudiness / field convergence")
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/obs_strength.png", dpi=120)
    print(f"saved to {args.outdir}")


if __name__ == "__main__":
    main()
