"""Measure per-level gradient norms of a trained hierarchical checkpoint.

Runs K real training forwards/backwards on the train split (render, eikonal
and curvature losses exactly as the trainer computes them, at end-of-training
weights) and reports, per hierarchy level and per loss component:

  - grad RMS per element  (does the level still receive signal?)
  - grad L2 total         (where is the gradient "energy" concentrated?)
  - value RMS per element (context: relative update size)

The result grounds the choice of per-level learning-rate multipliers.

Run from repo root::

    python projects/Bspline-Neus/scripts/grad_norm_diag.py \
        --config projects/Bspline-Neus/configs/dtu_scan24_E22b_optimized_500k.yaml \
        --checkpoint logs/E22b_optimized_500k/epoch_20833_iteration_000500000_checkpoint.pt \
        --iters 64
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from torch.utils.data import DataLoader

from importlib import import_module
from imaginaire.config import Config
from imaginaire.utils.gpu_affinity import set_affinity
from projects.neuralangelo.utils.misc import eikonal_loss, curvature_loss

Dataset = import_module("projects.Bspline-Neus.data").Dataset
Model = import_module("projects.Bspline-Neus.model").Model


def parse_args():
    parser = argparse.ArgumentParser(description="Per-level gradient norm diagnostic")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--iters", type=int, default=64)
    parser.add_argument("--local_rank", type=int, default=os.getenv("LOCAL_RANK", 0))
    return parser.parse_args()


class GradAccumulator:
    """Accumulate sum-of-squares / abs / L2 of grads for a named param set."""

    def __init__(self, params):
        self.params = list(params)
        self.sq = 0.0
        self.abs_sum = 0.0
        self.l2_sq = 0.0
        self.count = 0
        self.steps = 0

    def collect(self):
        sq = abs_sum = 0.0
        count = 0
        l2_sq = 0.0
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            sq += g.pow(2).sum().item()
            abs_sum += g.abs().sum().item()
            count += g.numel()
            l2_sq += g.pow(2).sum().item()
        self.sq += sq
        self.abs_sum += abs_sum
        self.count += count
        self.l2_sq += l2_sq
        self.steps += 1

    def report(self):
        if self.steps == 0 or self.count == 0:
            return dict(rms=float("nan"), mean_abs=float("nan"), l2=float("nan"))
        rms = (self.sq / self.count) ** 0.5
        mean_abs = self.abs_sum / self.count
        l2 = (self.l2_sq / self.steps) ** 0.5  # per-step average total L2
        return dict(rms=rms, mean_abs=mean_abs, l2=l2)


def main():
    args = parse_args()
    set_affinity(args.local_rank)
    cfg = Config(args.config)

    dataset = Dataset(cfg, is_inference=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    model = Model(cfg.model, cfg.data).cuda()
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    state_dict = {k[7:] if k.startswith("module.") else k: v
                  for k, v in checkpoint["model"].items()}
    model.load_state_dict(state_dict, strict=False)
    model.train()
    iteration = int(checkpoint.get("iteration", cfg.max_iter))
    model.progress = min(iteration / cfg.max_iter, 1.0)

    sdf_hier = model.neural_sdf.hier_field
    rgb_hier = model.neural_rgb.hier_field
    n_levels = sdf_hier.num_levels

    components = ["render", "eikonal", "curvature"]
    accs = {}
    for comp in components:
        for l in range(n_levels):
            accs[(comp, f"sdf_L{l}")] = GradAccumulator([sdf_hier.levels[l].values])
            accs[(comp, f"rgb_L{l}")] = GradAccumulator([rgb_hier.levels[l].values])
        accs[(comp, "mlp")] = GradAccumulator(model.neural_rgb._mlp.parameters())
        _p = getattr(model.neural_sdf, "raw_sdf_inv_std_levels", None)
        if _p is None:
            _p = model.neural_sdf.raw_sdf_inv_std
        accs[(comp, "inv_std")] = GradAccumulator([_p])

    # Use the run's actual weights; the saved E22b config uses eikonal=0.0005.
    # Hard-coding 0.1 overstated its contribution by 200x.
    loss_weights = cfg.trainer.loss_weight
    warmup = cfg.optim.sched.warm_up_end
    curv_factor = iteration / max(warmup, 1) if iteration <= warmup else 1-model.progress*0.5
    curv_w = float(loss_weights.curvature) * curv_factor
    eik_w = float(loss_weights.eikonal)
    render_w = float(loss_weights.render)
    band = float(getattr(loss_weights, "eikonal_band", 0.0))
    print(f"Actual weighted losses: render=3*{render_w}, eikonal={eik_w}, "
          f"eikonal_band={band}, curvature={curv_w}")
    l1 = torch.nn.L1Loss()

    it = iter(loader)
    for step in range(args.iters):
        try:
            data = next(it)
        except StopIteration:
            it = iter(loader)
            data = next(it)
        data = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in data.items()}

        output = model(data)
        if band > 0:
            weight = (output["sdfs"].detach().abs() < band).float()
            weight = weight * (~output["outside"]).float()
            error = ((output["gradients"].norm(dim=-1)-1)**2).nan_to_num(
                nan=0.0, posinf=0.0, neginf=0.0)
            eik = (error*weight).sum() / (weight.sum()+1e-8)
        else:
            eik = eikonal_loss(output["gradients"], outside=output["outside"])
        losses = {
            "render": l1(output["rgb"], data["image_sampled"]) * 3 * render_w,
            "eikonal": eik * eik_w,
            "curvature": curvature_loss(output["hessians"], outside=output["outside"]) * curv_w,
        }
        for ci, (comp, loss) in enumerate(losses.items()):
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=(ci < len(losses) - 1))
            for key, acc in accs.items():
                if key[0] == comp:
                    acc.collect()

    print("=" * 96)
    print(f"Per-level gradient stats over {args.iters} training steps "
          f"(rand_rays={cfg.model.render.rand_rays}, progress={model.progress})")
    print("=" * 96)
    header = f"{'param':10s} {'N':>12s} | " + " | ".join(
        f"{c+' RMS':>12s} {c+' L2':>12s}" for c in components)
    print(header)
    print("-" * len(header))
    rows = [f"sdf_L{l}" for l in range(n_levels)] + \
           [f"rgb_L{l}" for l in range(n_levels)] + ["mlp", "inv_std"]
    for name in rows:
        if name.startswith("sdf_L"):
            n = sdf_hier.levels[int(name[-1])].values.numel()
        elif name.startswith("rgb_L"):
            n = rgb_hier.levels[int(name[-1])].values.numel()
        elif name == "mlp":
            n = sum(p.numel() for p in model.neural_rgb._mlp.parameters())
        else:
            n = 1
        cell = f"{name:10s} {n:12d} | "
        parts = []
        for comp in components:
            r = accs[(comp, name)].report()
            parts.append(f"{r['rms']:12.3e} {r['l2']:12.3e}")
        print(cell + " | ".join(parts))

    # Value RMS for context.
    print("\nvalue RMS per level:")
    for l in range(n_levels):
        vs = sdf_hier.levels[l].values.detach()
        vr = rgb_hier.levels[l].values.detach()
        print(f"  L{l}: sdf {vs.pow(2).mean().sqrt().item():.4e}  "
              f"rgb_feat {vr.pow(2).mean().sqrt().item():.4e}")


if __name__ == "__main__":
    main()
