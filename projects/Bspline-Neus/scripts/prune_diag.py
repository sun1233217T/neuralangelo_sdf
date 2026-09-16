"""Pruning diagnostic: how much of each refined level is "dead weight"?

Replays all training views, finds the SDF zero-crossing point of every ray
(the same anchor the loss-guided marking uses, radius=1 = 6 face neighbors),
and accumulates per-cell hit counts at every hierarchy level.  Cells whose
region is active but which are never hit by any anchor never contribute to
the render loss -- they are the interior "bubble web" wasting budget.

Usage:
    python projects/Bspline-Neus/scripts/prune_diag.py \
        --config ... --checkpoint ... [--stride 1]
"""
import argparse
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from importlib import import_module

from imaginaire.config import Config

Dataset = import_module("projects.Bspline-Neus.data").Dataset
Model = import_module("projects.Bspline-Neus.model").Model


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
    return center + ray_unit * dc[:, None], has


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stride", type=int, default=2, help="pixel stride per view")
    args = ap.parse_args()

    cfg = Config(args.config)
    ds = Dataset(cfg, is_inference=False)  # train split = all 49 views
    print(f"dataset views: {len(ds)}")
    model = Model(cfg.model, cfg.data).cuda()
    model.progress = 1.0
    ckpt = torch.load(args.checkpoint, map_location=lambda s, l: s)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    hier = model.neural_sdf.hier_field

    # 6 face-neighbor offsets (anchor radius=1)
    offs = torch.tensor([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
                        device="cuda")

    n_levels = hier.num_levels
    hits = []
    for li in range(n_levels):
        lv = hier.levels[li]
        hits.append(torch.zeros_like(lv.region, dtype=torch.int32))

    n_views = len(ds)
    st = args.stride
    for vi in range(n_views):
        data = ds[vi]
        data = {k: (v[None].cuda() if torch.is_tensor(v) else v) for k, v in data.items()}
        pose, intr = data["pose"], data["intr"].clone()
        H, W = int(cfg.data.val.image_size[0]), int(cfg.data.val.image_size[1])
        # strided intrinsics: scale focal and principal point
        intr[:, 0, 0] /= st; intr[:, 1, 1] /= st
        intr[:, 0, 2] /= st; intr[:, 1, 2] /= st
        with torch.no_grad():
            for center, ray, _ in model.ray_generator(pose, intr, (H // st, W // st), full_image=True):
                ray_unit = F.normalize(ray, dim=-1)
                out = model.render_rays(center, ray_unit)
                n_obj = out["gradients"].shape[2]
                pts, has = first_crossing(center[0], ray_unit[0],
                                          out["dists"][0, :, :n_obj], out["sdfs"][0, :, :n_obj])
                if not has.any():
                    continue
                p = pts[has]
                for li in range(n_levels):
                    lv = hier.levels[li]
                    lower = lv._lower.detach()
                    step = lv.step.detach()
                    cc = lv.cell_count
                    idx = ((p - lower) / step).floor().long()
                    for o in offs:
                        io = idx + o
                        ok = ((io >= 0) & (io < cc)).all(dim=-1)
                        if ok.any():
                            ii = io[ok]
                            hits[li].index_put_((ii[:, 0], ii[:, 1], ii[:, 2]),
                                                torch.ones(ok.sum(), dtype=torch.int32, device="cuda"),
                                                accumulate=True)
        if vi % 8 == 0:
            print(f"view {vi}/{n_views} done", flush=True)

    print(f"\n=== dead-cell diagnostic ({n_views} train views, stride {st}) ===")
    print(f"{'level':>6} {'cc':>5} {'active':>8} {'hit':>8} {'dead':>8} {'dead%':>7}")
    for li in range(n_levels):
        lv = hier.levels[li]
        act = lv.region
        n_act = int(act.sum())
        n_hit = int((hits[li] > 0).sum())
        dead = n_act - n_hit
        print(f"{li:>6} {lv.cell_count:>5} {n_act:>8} {n_hit:>8} {dead:>8} {dead/max(n_act,1):>7.1%}")
    # save hit masks for later pruning experiments
    torch.save({f"hits_L{li}": h.cpu() for li, h in enumerate(hits)},
               "logs/prune_diag_hits.pt")
    print("hit masks saved to logs/prune_diag_hits.pt")


if __name__ == "__main__":
    main()
