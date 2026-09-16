"""Probe mean-curvature / Laplacian distribution of a trained SDF field.

Samples random points in the unit domain, keeps the near-surface band,
and reports quantiles of |H| (true mean curvature) and |lap|/|grad|
(normalized Laplacian) to guide threshold selection for MC regularization.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from imaginaire.config import Config  # noqa: E402
from imaginaire.utils.distributed import init_dist  # noqa: E402
from imaginaire.trainers.utils.get_trainer import get_trainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--band", type=float, default=0.05)
    args = ap.parse_args()

    cfg = Config(args.config)
    cfg.logdir = ""
    trainer = get_trainer(cfg, is_inference=True, seed=0)
    trainer.current_iteration = 0
    trainer.checkpointer.load(args.checkpoint, load_opt=False, load_sch=False)
    model = trainer.model_module
    sdf = model.neural_sdf
    assert hasattr(sdf, "evaluate_mean_curvature"), "need THB full-Hessian support"

    Hs, Ls, Ss = [], [], []
    with torch.no_grad():
        for _ in range(40):
            pts = torch.rand(65536, 3, device="cuda") * 4.0 - 2.0
            H, g = sdf.evaluate_mean_curvature(pts)
            s = sdf.sdf(pts).reshape(-1)
            keep = s.abs() < args.band
            if keep.any():
                gn = g.norm(dim=-1).clamp_min(1e-6)
                # normalized Laplacian needs diag; approximate via H + gHg term:
                # here just record H and |H|
                Hs.append(H[keep].cpu())
                Ss.append(s[keep].cpu())
    H = torch.cat(Hs).float().nan_to_num(0.0)
    aH = H.abs()
    q = torch.tensor([0.5, 0.9, 0.95, 0.99, 0.999])
    qs = torch.quantile(aH, q)
    print(f"band |sdf|<{args.band}: {len(aH)} samples")
    for qq, v in zip(q.tolist(), qs.tolist()):
        print(f"  |H| q{qq*100:.1f}: {v:.1f}")
    print(f"  |H| max: {aH.max().item():.1f}")
    # histogram tail
    for t in [50, 100, 200, 500, 1000]:
        frac = (aH > t).float().mean().item()
        print(f"  P(|H|>{t}): {frac*100:.3f}%")


if __name__ == "__main__":
    main()
