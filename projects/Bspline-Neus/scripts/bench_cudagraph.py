"""CUDA Graph experiment for the Bspline-Neus training step.

Measures the per-iteration time of the real training step (model forward +
losses + backward + optimizer step) in eager mode vs replayed as a captured
CUDA graph.  The point of the experiment is to quantify how much of the
per-iteration fixed overhead (kernel launches, Python glue) a graph replay
could eliminate: shapes are static between refinement events, so the steady
state is graph-capturable in principle.

Scope and caveats (this is a benchmark, not a trainer integration):

- The same data batch is replayed every iteration; data loading is excluded.
- Scalar hyper-parameters that vary per iteration in real training
  (curvature weight schedule, ``model.progress``, lr schedule) are baked
  into the graph as constants.  At the 500k checkpoint used here they are
  all at their constant end-of-schedule values anyway.
- Refinement events change tensor shapes and would require re-capture.
- RNG inside the capture (stratified depth sampling) uses PyTorch's
  graph-safe RNG, so replays draw fresh samples.

Run from the repository root, e.g.:

    python projects/Bspline-Neus/scripts/bench_cudagraph.py --single_gpu \
        --config projects/Bspline-Neus/configs/dtu_scan24_thb.yaml \
        --checkpoint logs/<run>/epoch_20833_iteration_000500000_checkpoint.pt \
        --iters 100
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from imaginaire.config import Config, recursive_update_strict, parse_cmdline_arguments  # noqa: E402
from imaginaire.utils.distributed import init_dist, get_world_size, master_only_print as print  # noqa: E402
from imaginaire.utils.gpu_affinity import set_affinity  # noqa: E402
from imaginaire.trainers.utils.get_trainer import get_trainer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Bspline-Neus CUDA graph benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--local_rank", type=int, default=os.getenv("LOCAL_RANK", 0))
    parser.add_argument("--single_gpu", action="store_true")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--stage", choices=["fwd", "fwdbwd", "full"], default="full",
                        help="Bisect capture failures: forward only, forward+backward, "
                             "or the full step including the optimizer.")
    parser.add_argument("--losses", default="render,eikonal,curvature",
                        help="Comma-separated loss terms to keep (bisects which "
                             "higher-order backward path breaks capture).")
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd


def main():
    args, cfg_cmd = parse_args()
    set_affinity(args.local_rank)
    cfg = Config(args.config)
    cfg_cmd = parse_cmdline_arguments(cfg_cmd)
    recursive_update_strict(cfg, cfg_cmd)
    # Validation must never fire inside the benchmark loop.
    cfg.validation_iter = 10 ** 12

    if not args.single_gpu:
        os.environ["NCLL_BLOCKING_WAIT"] = "0"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank, rank=-1, world_size=-1)
    print(f"Running CUDA graph benchmark with {get_world_size()} GPUs.")

    cfg.logdir = ""
    trainer = get_trainer(cfg, is_inference=False, seed=0)
    trainer.set_data_loader(cfg, split="train")
    trainer.current_iteration = 0
    trainer.checkpointer.load(args.checkpoint, load_opt=False, load_sch=False)
    trainer.current_iteration = trainer.checkpointer.eval_iteration
    model = trainer.model
    model.train()

    # Grab one real training batch and normalize it through the trainer's
    # own start-of-iteration path (device transfer, AMP bookkeeping, and the
    # refinement hook — a no-op past the refinement schedule).
    batch = next(iter(trainer.train_data_loader))
    # Tear the loader down completely: its worker/pin-memory threads keep
    # issuing CUDA calls from foreign threads, which poisons graph capture.
    import gc
    del trainer.train_data_loader
    gc.collect()
    data = trainer.start_of_iteration(batch, trainer.current_iteration)

    # Bisect support: drop loss terms not requested via --losses so the
    # backward graph only traverses the kept higher-order paths.
    keep = {k.strip() for k in args.losses.split(",") if k.strip()}
    for k in list(trainer.weights.keys()):
        if k not in keep:
            trainer.weights.pop(k)
    print(f"active loss weights: {trainer.weights}")

    def eager_step():
        """The full production train step, as a reference."""
        trainer.train_step(data)

    # -- graph capture FIRST ------------------------------------------------
    # Empirically (see logs/bisect_mgc_bwd.py): capturing AFTER many eager
    # train steps fails with "legacy stream depends on capturing stream"
    # (optimizer/EMA state allocated on side streams).  The reliable recipe:
    # light side-stream warmup of the forward only, then MGC immediately,
    # eager baseline afterwards.  Route: torch.cuda.make_graphed_callables —
    # captures BOTH the forward and the backward graph (manual capture of
    # .backward() fails because the autograd engine launches from worker
    # threads).  All per-iteration inputs must be tensor arguments; everything
    # else (weights, schedule scalars) is baked in as static, which matches
    # the end-of-schedule state at 500k.
    trainer.optim = torch.optim.AdamW(trainer.optim.param_groups, capturable=True)

    class TrainStepModule(torch.nn.Module):
        """nn.Module wrapper so make_graphed_callables can discover the
        parameters to differentiate into (plain functions get none)."""

        def __init__(self, model, trainer):
            super().__init__()
            self.model = model
            self.trainer = trainer

        def forward(self, pose, intr, intr_inv, idx, ray_idx, image_sampled):
            d = {"pose": pose, "intr": intr, "intr_inv": intr_inv, "idx": idx,
                 "ray_idx": ray_idx, "image_sampled": image_sampled}
            d.update(self.model(d))
            self.trainer._compute_loss(d, mode="train")
            return self.trainer._get_total_loss()

    step_module = TrainStepModule(trainer.model_module, trainer)
    static_args = tuple(data[k] for k in ("pose", "intr", "intr_inv", "idx", "ray_idx", "image_sampled"))

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            trainer.model_forward(data)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    try:
        graphed = torch.cuda.make_graphed_callables(step_module, static_args)
    except Exception as exc:  # noqa: BLE001 - report and fall back
        import traceback
        traceback.print_exc()
        print(f"graph capture FAILED: {type(exc).__name__}: {exc}")
        return

    def graph_step():
        loss = graphed(*static_args)
        loss.backward()
        trainer.optim.step()
        trainer.optim.zero_grad(set_to_none=False)
        return loss

    for _ in range(args.warmup):
        graph_step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        graph_step()
    torch.cuda.synchronize()
    graph_dt = (time.perf_counter() - t0) / args.iters
    print(f"graph replay:      {graph_dt * 1e3:8.2f} ms/iter  ({1.0 / graph_dt:6.2f} it/s)")

    # -- breakdown: fwd replay / bwd replay / optimizer -----------------------
    def time_phase(fn):
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(args.iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / args.iters

    fwd_dt = time_phase(lambda: graphed(*static_args))
    fwdbwd_dt = time_phase(lambda: graphed(*static_args).backward())
    optim_dt = time_phase(lambda: (trainer.optim.step(),
                                   trainer.optim.zero_grad(set_to_none=False)))
    print(f"  fwd replay only: {fwd_dt * 1e3:8.2f} ms/iter")
    print(f"  fwd+bwd replay:  {fwdbwd_dt * 1e3:8.2f} ms/iter")
    print(f"  optim step:      {optim_dt * 1e3:8.2f} ms/iter")

    # -- per-op GPU time inside a replayed step (profiler on replay) ----------
    from torch.profiler import profile, ProfilerActivity
    for _ in range(3):
        graph_step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(5):
            graph_step()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15,
                                    max_name_column_width=60))

    # -- eager baseline (run after capture; see note above) -------------------
    for _ in range(args.warmup):
        eager_step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        eager_step()
    torch.cuda.synchronize()
    eager_dt = (time.perf_counter() - t0) / args.iters
    print(f"eager train_step:  {eager_dt * 1e3:8.2f} ms/iter  ({1.0 / eager_dt:6.2f} it/s)")
    print(f"speedup:           {eager_dt / graph_dt:8.2f}x")

    # Sanity: the graph must actually train — losses from the eager and
    # graph phases should be in the same range.
    loss = trainer._get_total_loss()
    print(f"total loss after graph phase: {float(loss):.6f}")


if __name__ == "__main__":
    main()
