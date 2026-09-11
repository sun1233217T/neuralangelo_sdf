"""CUDA-graph accelerated training step for Bspline-Neus.

Captures the model forward AND backward with
``torch.cuda.make_graphed_callables`` (MGC) and replays them every iteration,
eliminating the per-iteration kernel-launch / Python overhead that dominates
the eager step (measured: ~142 ms eager -> ~87 ms replay at 2048 rays).

Design constraints that shaped this module (see scripts/bench_cudagraph.py):

- Shapes are static between hierarchical refinement events; refinement
  invalidates the graph (new parameter tensors) and triggers a re-capture.
- Per-iteration schedule scalars must NOT be baked into the graph:
    * loss weights (curvature warm-up/decay) are applied EAGERLY on the
      graph's outputs — the graph returns unweighted raw quantities;
    * ``model.progress`` becomes a 0-dim CUDA tensor that the trainer
      ``fill_``s each iteration (``_get_iter_cos`` reads it via torch.clamp).
- The learning-rate schedule is untouched: the optimizer/scheduler run
  eagerly outside the graph.
- ``camera.img2cam`` needs a precomputed inverse intrinsic (the dataset
  provides ``intr_inv``); ``torch.inverse`` (cuSOLVER) is not capturable.
- Capture must happen early, right after a short side-stream warmup —
  capturing after many eager steps fails with legacy-stream dependencies.
- ``aten::cumprod`` backward is not capturable; the shared
  ``projects/nerf/utils/render.py`` uses the graph-safe replacement.

AMP, EMA and gradient accumulation are unsupported; the trainer only enables
this path when they are all off (the default for this project).
"""

import torch
import torch.nn.functional as torch_F

from projects.neuralangelo.utils.misc import eikonal_loss, curvature_loss


class GraphTrainStep:
    """Owns the captured training step and its static input/output buffers."""

    INPUT_KEYS = ("pose", "intr", "intr_inv", "idx", "ray_idx", "image_sampled")

    def __init__(self, trainer, cfg_graph):
        self.trainer = trainer
        # Capture at iteration 0: any eager train step before the capture
        # poisons it (the eager backward leaves autograd-engine/allocator
        # stream state that later trips "legacy stream depends on capturing
        # stream").  Graph replays do NOT poison re-captures, so refinement
        # re-capture works as long as every capture happens before any eager
        # backward in that shape regime... in practice: capture at iter 0 and
        # re-capture immediately after each refinement.
        self.capture_iter = int(cfg_graph.get("capture_iter", 0))
        self.graphed = None
        self.static_in = dict()
        self.failed = False  # permanent fallback to eager

    def invalidate(self):
        """Drop the captured graph; the next step re-captures.

        Hierarchical refinement replaces the level value parameters, so the
        old graph (bound to the old tensors) must be discarded.
        """
        if self.graphed is not None:
            print("[graph] invalidating captured training graph (will re-capture)")
        self.graphed = None
        self.static_in = dict()

    class _StepModule(torch.nn.Module):
        """nn.Module wrapper so MGC can discover the parameters."""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pose, intr, intr_inv, idx, ray_idx, image_sampled):
            d = {"pose": pose, "intr": intr, "intr_inv": intr_inv, "idx": idx,
                 "ray_idx": ray_idx, "image_sampled": image_sampled}
            d.update(self.model(d))
            # Loss-guided refinement statistics: scatter this batch's render
            # error into the finest level's cell grid.  The accumulation is an
            # in-place index_add_ on persistent buffers owned by the SDF
            # wrapper, which replays deterministically inside the graph.
            if "hier_points" in d:
                self.model.neural_sdf.accumulate_loss(
                    d["hier_points"], d["hier_weights"], d["rgb"], image_sampled,
                    sdfs=d.get("hier_sdfs"))
            # MGC requires every output to require grad; ``outside`` is a bool
            # mask, so pass it out as a float with a zero gradient path.
            outside_d = d["outside"].to(d["rgb"].dtype) + d["rgb"].sum() * 0.0
            return d["rgb"], d["gradients"], d["hessians"], outside_d, d["sdfs"]

    def _capture(self, data):
        trainer = self.trainer
        model = trainer.model_module
        # Make the progress schedule graph-dynamic: a 0-dim tensor the trainer
        # updates in place each iteration (read via torch.clamp in the model).
        if not torch.is_tensor(getattr(model, "progress", None)):
            device = next(model.parameters()).device
            model.progress = torch.tensor(
                float(getattr(model, "progress", 0.0) or 0.0), device=device)
        trainer.progress = model.progress

        self.static_in = {k: data[k].clone() for k in self.INPUT_KEYS}
        args = tuple(self.static_in[k] for k in self.INPUT_KEYS)
        module = self._StepModule(model)
        # Side-stream warmup, then capture immediately.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                module(*args)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.graphed = torch.cuda.make_graphed_callables(module, args)
        torch.cuda.synchronize()
        print(f"[graph] captured training step at iter {trainer.current_iteration}")

    def train_step(self, data):
        """Run one training step through the graph. Returns False if the
        caller should fall back to the eager path for this step."""
        trainer = self.trainer
        if self.failed:
            return False
        if self.graphed is None:
            if trainer.current_iteration < self.capture_iter:
                return False
            try:
                self._capture(data)
            except Exception as exc:  # noqa: BLE001 - any capture failure -> eager
                import traceback
                traceback.print_exc()
                print(f"[graph] capture FAILED ({type(exc).__name__}: {exc}); "
                      "falling back to eager training permanently")
                self.failed = True
                self.graphed = None
                torch.cuda.synchronize()
                return False

        # Load this iteration's batch into the static input buffers.
        for k in self.INPUT_KEYS:
            self.static_in[k].copy_(data[k], non_blocking=True)
        rgb, gradients, hessians, outside_d, sdfs = self.graphed(
            *tuple(self.static_in[k] for k in self.INPUT_KEYS))

        # Losses stay eager so the current schedule weights apply.
        target = self.static_in["image_sampled"]
        outside = outside_d > 0.5
        losses = dict()
        losses["render"] = trainer.criteria["render"](rgb, target) * 3
        trainer.metrics["psnr"] = -10 * torch_F.mse_loss(rgb, target).log10()
        total = losses["render"] * trainer.weights.get("render", 1.0)
        if "eikonal" in trainer.weights:
            losses["eikonal"] = trainer._eikonal_loss(gradients, outside, sdfs)
            total = total + losses["eikonal"] * trainer.weights["eikonal"]
        if "curvature" in trainer.weights:
            losses["curvature"] = curvature_loss(hessians, outside=outside)
            total = total + losses["curvature"] * trainer.weights["curvature"]
        if "mean_curvature" in trainer.weights:
            losses["mean_curvature"] = trainer._mean_curvature_loss(
                gradients, hessians, sdfs, outside=outside)
            total = total + losses["mean_curvature"] * trainer.weights["mean_curvature"]
        losses["total"] = total
        trainer.losses.clear()
        trainer.losses.update(losses)

        total.backward()  # replays the captured backward graph
        # Per-level gradient-RMS snapshot for logging (no-op off logging iters).
        trainer._extra_step(data)
        trainer.optim.step()
        trainer.optim.zero_grad(**trainer.optim_zero_grad_kwargs)
        trainer._detach_losses()
        return True
