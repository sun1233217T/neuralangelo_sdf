"""Break down GPU memory usage for a Bspline-Neus training checkpoint.

Run from repo root, e.g.::

    python projects/Bspline-Neus/scripts/analyze_vram.py \
        --config projects/Bspline-Neus/configs/dtu_scan24_thb_v3_color_mlp_v12_edge_pe.yaml \
        --checkpoint logs/.../epoch_20833_iteration_000500000_checkpoint.pt
"""

import argparse
import gc
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from imaginaire.config import Config, recursive_update_strict, parse_cmdline_arguments
from imaginaire.utils.distributed import init_dist, get_world_size, master_only_print as print
from imaginaire.utils.gpu_affinity import set_affinity
from imaginaire.trainers.utils.get_trainer import get_trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Bspline-Neus VRAM breakdown")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--local_rank", type=int, default=os.getenv("LOCAL_RANK", 0))
    parser.add_argument("--single_gpu", action="store_true")
    parser.add_argument("--num_iters", type=int, default=3, help="iterations for activation memory peak")
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd


def fmt_bytes(b):
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if b < 1024.0:
            return f"{b:.2f} {unit}"
        b /= 1024.0
    return f"{b:.2f} TiB"


def param_mem(module):
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    return total


def main():
    args, cfg_cmd = parse_args()
    set_affinity(args.local_rank)
    cfg = Config(args.config)
    cfg_cmd = parse_cmdline_arguments(cfg_cmd)
    recursive_update_strict(cfg, cfg_cmd)
    cfg.validation_iter = 10 ** 12

    if not args.single_gpu:
        os.environ["NCLL_BLOCKING_WAIT"] = "0"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank, rank=-1, world_size=-1)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    base_mem = torch.cuda.memory_allocated()
    print(f"Base allocated after empty_cache: {fmt_bytes(base_mem)}")

    cfg.logdir = ""
    trainer = get_trainer(cfg, is_inference=False, seed=0)
    trainer.set_data_loader(cfg, split="train")
    trainer.current_iteration = 0
    trainer.checkpointer.load(args.checkpoint, load_opt=True, load_sch=True)
    trainer.current_iteration = trainer.checkpointer.eval_iteration
    model = trainer.model
    model.train()

    raw_model = model.module if hasattr(model, "module") else model

    print("\n=== Parameter memory breakdown ===")
    total_param = 0
    for name, module in raw_model.named_children():
        mem = param_mem(module)
        total_param += mem
        print(f"  {name:30s}: {fmt_bytes(mem):>12s}")
    print(f"  {'Total model params':30s}: {fmt_bytes(total_param)}")

    # Detailed B-spline submodules.
    if hasattr(raw_model, "neural_sdf"):
        sdf = raw_model.neural_sdf
        print("\n  -- SDF hierarchy --")
        if sdf.hierarchical_enabled:
            for i, level in enumerate(sdf.hier_field.levels):
                vals = level.values
                mem = vals.numel() * vals.element_size()
                print(f"    level {i} values ({vals.shape}): {fmt_bytes(mem)}")
            # Transitions and index grids are int64 / buffers.
            trans_mem = sum(b.numel() * b.element_size() for b in sdf.hier_field.buffers())
            print(f"    hierarchy buffers: {fmt_bytes(trans_mem)}")
        else:
            print(f"    raw_sdf_grid: {fmt_bytes(param_mem(sdf))}")

    if hasattr(raw_model, "neural_rgb"):
        rgb = raw_model.neural_rgb
        print("\n  -- Color field --")
        if rgb.hierarchical_enabled:
            for i, level in enumerate(rgb.hier_field.levels):
                vals = level.values
                mem = vals.numel() * vals.element_size()
                print(f"    level {i} values ({vals.shape}): {fmt_bytes(mem)}")
            trans_mem = sum(b.numel() * b.element_size() for b in rgb.hier_field.buffers())
            print(f"    hierarchy buffers: {fmt_bytes(trans_mem)}")
        else:
            print(f"    raw_color_grid: {fmt_bytes(param_mem(rgb))}")
        if rgb._mlp is not None:
            print(f"    color MLP: {fmt_bytes(param_mem(rgb._mlp))}")

    # Optimizer state memory.
    print("\n=== Optimizer state memory ===")
    opt = trainer.optim
    print(f"  Optimizer type: {type(opt).__name__}")
    print(f"  #param groups: {len(opt.param_groups)}, #state entries: {len(opt.state)}")
    opt_state_bytes = 0
    for group in opt.param_groups:
        for p in group["params"]:
            state = opt.state.get(p, {})
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    opt_state_bytes += v.numel() * v.element_size()
    print(f"  Adam state (exp_avg + exp_avg_sq): {fmt_bytes(opt_state_bytes)}")

    # Gradient memory (rough: same as params when training).
    grad_bytes = sum(p.numel() * p.element_size() for p in raw_model.parameters() if p.requires_grad)
    print(f"  Gradient storage (est.): {fmt_bytes(grad_bytes)}")

    after_load = torch.cuda.memory_allocated()
    print(f"\nAfter model+optim load: {fmt_bytes(after_load)}")

    # Activation/peak memory during training step.
    print("\n=== Training step activation memory ===")
    batch = next(iter(trainer.train_data_loader))
    del trainer.train_data_loader
    gc.collect()
    data = trainer.start_of_iteration(batch, trainer.current_iteration)

    peak_before = torch.cuda.max_memory_allocated()
    for it in range(args.num_iters):
        torch.cuda.reset_peak_memory_stats()
        trainer.train_step(data)
        peak = torch.cuda.max_memory_allocated()
        print(f"  iter {it} peak: {fmt_bytes(peak)}  (delta from load: {fmt_bytes(peak - after_load)})")
        trainer.optim.zero_grad(set_to_none=False)

    print("\n=== Summary ===")
    final = torch.cuda.memory_allocated()
    print(f"Final allocated: {fmt_bytes(final)}")
    print(f"Peak observed: {fmt_bytes(torch.cuda.max_memory_allocated())}")
    print(f"Reserved: {fmt_bytes(torch.cuda.memory_reserved())}")


if __name__ == "__main__":
    main()
