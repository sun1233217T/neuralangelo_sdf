"""Phase 2 trainer smoke test: hierarchical refinement + optimizer rebuild.

Run from the repository root:

    python projects/Bspline-Neus/tests/test_thb_trainer.py

Verifies that a scheduled refinement (iteration 1) fires inside
``start_of_iteration``, that the optimizer is rebuilt around the new
hierarchical parameters, and that training steps work before and after.
"""

import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from imaginaire.config import Config  # noqa: E402


def _make_data(cfg, B=2, H=100, W=100):
    pose = torch.eye(3, 4, device="cuda").unsqueeze(0).repeat(B, 1, 1)
    pose[:, :3, 3] = torch.tensor([0.0, 0.0, 2.0], device="cuda")
    intr = torch.tensor([
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 1.0],
    ], device="cuda").unsqueeze(0).repeat(B, 1, 1)
    idx = torch.arange(B, device="cuda")
    ray_idx = torch.stack([
        torch.randperm(H * W, device="cuda")[:cfg.model.render.rand_rays]
        for _ in range(B)
    ], dim=0)
    image_sampled = torch.rand(B, cfg.model.render.rand_rays, 3, device="cuda")
    return {
        "pose": pose, "intr": intr, "idx": idx,
        "ray_idx": ray_idx, "image_sampled": image_sampled,
    }


def test_hierarchical_trainer():
    print("[test] hierarchical trainer: refine hook + optimizer rebuild")
    cfg = Config("projects/Bspline-Neus/configs/base.yaml")
    cfg.logdir = "logs/test_thb_neus"
    cfg.data.num_images = 10
    hier = cfg.model.object.bspline.hierarchical
    hier.enabled = True
    hier.color_enabled = True
    hier.base_grid_size = 16
    hier.max_levels = 2
    hier.refine_iters = [1]
    hier.refine_sdf_band = 0.4

    trainer_lib = importlib.import_module("projects.Bspline-Neus.trainer")
    trainer = trainer_lib.Trainer(cfg, is_inference=False, seed=0)
    trainer.model.train()
    trainer.model_module.progress = 1.0

    sdf = trainer.model_module.neural_sdf
    rgb = trainer.model_module.neural_rgb
    assert sdf.hierarchical_enabled and sdf.hier_field.num_levels == 1
    assert rgb.hierarchical_enabled and rgb.hier_field.num_levels == 1
    params_before = sum(p.numel() for p in trainer.model_module.parameters())
    print(f"  param count before refine: {params_before:,}")

    # Iteration 0: plain step with the original optimizer.
    data = trainer.start_of_iteration(_make_data(cfg), current_iteration=0)
    loss = trainer.model_forward(data)
    trainer.optim.zero_grad()
    loss.backward()
    trainer.optim.step()
    old_optim = trainer.optim
    print(f"  iter 0 loss = {loss.item():.4f}")

    # Iteration 1: refinement fires; optimizer must be rebuilt.
    data = trainer.start_of_iteration(_make_data(cfg), current_iteration=1)
    assert sdf.hier_field.num_levels == 2, "SDF refinement did not add a level"
    assert rgb.hier_field.num_levels == 2, "color refinement did not add a level"
    assert torch.equal(
        rgb.hier_field.levels[1].region, sdf.hier_field.levels[1].region
    ), "color/SDF hierarchies diverged"
    assert trainer.optim is not old_optim, "optimizer was not rebuilt"
    assert trainer.checkpointer.optim is trainer.optim
    params_after = sum(p.numel() for p in trainer.model_module.parameters())
    optim_params = sum(p.numel() for g in trainer.optim.param_groups for p in g["params"])
    print(f"  refined at iter 1; param count after: {params_after:,} (optim covers {optim_params:,})")
    assert optim_params == params_after

    loss = trainer.model_forward(data)
    trainer.optim.zero_grad()
    loss.backward()
    trainer.optim.step()
    assert loss.item() == loss.item()  # not NaN
    print(f"  iter 1 loss = {loss.item():.4f}")
    print("  ok")


def test_refinement_guards():
    print("[test] refinement guards: no refine in eval mode / no double refine")
    cfg = Config("projects/Bspline-Neus/configs/base.yaml")
    cfg.logdir = "logs/test_thb_neus"
    cfg.data.num_images = 10
    hier = cfg.model.object.bspline.hierarchical
    hier.enabled = True
    hier.color_enabled = True
    hier.base_grid_size = 16
    hier.max_levels = 4
    hier.refine_iters = [1, 2, 3]
    hier.refine_sdf_band = 0.4

    trainer_lib = importlib.import_module("projects.Bspline-Neus.trainer")
    trainer = trainer_lib.Trainer(cfg, is_inference=False, seed=0)
    sdf = trainer.model_module.neural_sdf

    # Validation runs under torch.no_grad (see test()); refinement must not
    # fire there even at a scheduled iteration.  Simulate both the no_grad
    # context and the eval-mode model state left behind by a validation.
    trainer.model.eval()
    with torch.no_grad():
        trainer.start_of_iteration(_make_data(cfg), current_iteration=1)
    assert sdf.hier_field.num_levels == 1, "validation triggered refinement"
    # The real post-validation sequence: model left in eval mode, grad
    # enabled — refinement must STILL fire (regression: gating on
    # model.training let validation iterations suppress scheduled refines).
    trainer.start_of_iteration(_make_data(cfg), current_iteration=1)
    assert sdf.hier_field.num_levels == 2, "training refinement did not fire"
    # A second call at the same iteration (e.g. resume landing on the
    # schedule) must not refine again.
    trainer.start_of_iteration(_make_data(cfg), current_iteration=1)
    assert sdf.hier_field.num_levels == 2, "refinement fired twice at one iteration"
    # Later rounds still fire on schedule.
    trainer.start_of_iteration(_make_data(cfg), current_iteration=2)
    assert sdf.hier_field.num_levels == 3
    print("  ok")


if __name__ == "__main__":
    test_hierarchical_trainer()
    test_refinement_guards()
    print("\nHierarchical trainer smoke test passed!")
