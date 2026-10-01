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
        self.input_keys = list(self.INPUT_KEYS)
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

        def forward(self, pose, intr, intr_inv, idx, ray_idx, image_sampled,
                    mask_sampled=None):
            d = {"pose": pose, "intr": intr, "intr_inv": intr_inv, "idx": idx,
                 "ray_idx": ray_idx, "image_sampled": image_sampled}
            if mask_sampled is not None:
                d["mask_sampled"] = mask_sampled
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
            # dists/weights are constants w.r.t. parameters; give them the
            # same zero-gradient path so MGC accepts them as outputs.
            flat = d["rgb"].sum() * 0.0
            dists_d = d["dists"].to(d["rgb"].dtype) + flat
            weights_d = d["weights"].to(d["rgb"].dtype) + flat
            # Sample positions (for weak-region masking); zero-gradient path.
            if "hier_points" in d:
                pts_d = d["hier_points"].to(d["rgb"].dtype) + flat
            else:
                pts_d = flat  # placeholder, unused
            return (d["rgb"], d["gradients"], d["hessians"], outside_d, d["sdfs"],
                    dists_d, weights_d, pts_d)

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

        keys = list(self.INPUT_KEYS)
        if "mask_sampled" in data:
            keys.append("mask_sampled")
        self.input_keys = keys
        self.static_in = {k: data[k].clone() for k in keys}
        args = tuple(self.static_in[k] for k in keys)
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
                # Diagnose which parameters are disconnected from the graph
                # (a common post-refine/prune failure: a level's values not
                # touched by the capture batch).
                try:
                    model = trainer.model_module
                    diag = self._StepModule(model)
                    outs = diag(*tuple(self.static_in[k] for k in self.input_keys))
                    total = sum(o.float().sum() for o in outs if o is not None)
                    params = [p for p in diag.parameters() if p.requires_grad]
                    grads = torch.autograd.grad(total, params, allow_unused=True)
                    unused = [n for n, g in zip([n for n, p in diag.named_parameters() if p.requires_grad], grads) if g is None]
                    print(f"[graph] capture diagnostic: unused params: {unused}")
                except Exception as diag_exc:  # noqa: BLE001
                    print(f"[graph] capture diagnostic failed: {diag_exc}")
                print(f"[graph] capture FAILED ({type(exc).__name__}: {exc}); "
                      "falling back to eager training permanently")
                self.failed = True
                self.graphed = None
                torch.cuda.synchronize()
                return False

        # Load this iteration's batch into the static input buffers.
        for k in self.input_keys:
            self.static_in[k].copy_(data[k], non_blocking=True)
        rgb, gradients, hessians, outside_d, sdfs, dists_d, weights_d, pts_d = self.graphed(
            *tuple(self.static_in[k] for k in self.input_keys))

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
                gradients, hessians, sdfs, outside=outside,
                points=pts_d if pts_d.dim() >= 2 else None)
            total = total + losses["mean_curvature"] * trainer.weights["mean_curvature"]
        if "thin_shell" in trainer.weights:
            n_obj = gradients.shape[2]
            losses["thin_shell"] = trainer._thin_shell_loss(
                sdfs, dists_d[:, :, :n_obj], weights_d[:, :, :n_obj])
            total = total + losses["thin_shell"] * trainer.weights["thin_shell"]
        if "depth" in trainer.weights and "depth_sampled" in data:
            # GT depth supervision (Replica): L1 between the composited ray
            # distance and the dataset's GT ray distance (normalized units).
            n_obj = gradients.shape[2]
            depth_pred = (weights_d[:, :, :n_obj, 0] * dists_d[:, :, :n_obj, 0]).sum(dim=-1)  # (B,R)
            depth_target = data["depth_sampled"].to(depth_pred.dtype)
            valid = depth_target > 0
            if valid.any():
                losses["depth"] = torch_F.l1_loss(depth_pred[valid], depth_target[valid])
                total = total + losses["depth"] * trainer.weights["depth"]
        if "mask" in trainer.weights and "mask_sampled" in self.static_in:
            # Mask supervision: background rays (mask=0) must carry zero
            # opacity inside the object volume — no free floating shells on
            # black ground-truth backgrounds.
            n_obj = gradients.shape[2]
            opacity_obj = weights_d[:, :, :n_obj, 0].sum(dim=-1)  # (B,R)
            bg = (self.static_in["mask_sampled"] < 0.5).to(opacity_obj.dtype)
            losses["mask"] = (opacity_obj.pow(2) * bg).sum() / (bg.sum() + 1e-8)
            total = total + losses["mask"] * trainer.weights["mask"]
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
