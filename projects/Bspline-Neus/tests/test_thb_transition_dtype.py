"""Accuracy test: half-precision THB transition (fp16/bf16) vs fp32.

Run from the repository root:

    python projects/Bspline-Neus/tests/test_thb_transition_dtype.py

Builds one random multi-level THB field, clones it with
``transition_compute_dtype`` in {"fp32", "fp16", "bf16"}, and compares
``evaluate`` / ``evaluate_with_deriv`` outputs at identical random points.
Also reports torch's reduced-precision-reduction flags, which control
whether fp16/bf16 GEMMs accumulate in fp32.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

hier_mod = importlib.import_module("projects.Bspline-Neus.bspline_field.hierarchical")

HierarchicalBSplineField = hier_mod.HierarchicalBSplineField

BOUNDS = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]


def _random_points(n, gen, device):
    return (torch.rand(n, 3, generator=gen) * 2.0 - 1.0).to(device) * 0.98


def _box_mark_fn(cc, round_idx):
    lo = cc // 4 + round_idx
    hi = 3 * cc // 4

    def mark_fn(level_index, corner_points):
        m = torch.zeros(cc, cc, cc, dtype=torch.bool)
        m[lo:hi, lo:hi, lo:hi] = True
        return m

    return mark_fn


def _make_field(degree, rounds, gen, device, transition_compute_dtype="fp32"):
    G = 16
    field = HierarchicalBSplineField(
        G, degree, BOUNDS, channels=1, max_levels=4, transfer_mode="thb",
        transition_compute_dtype=transition_compute_dtype, device=device,
    )
    field.levels[0].values.data = torch.randn(G, G, G, generator=gen).reshape(-1, 1).to(device)
    for round_idx in range(rounds):
        cc = field.levels[-1].cell_count
        info = field.refine(_box_mark_fn(cc, round_idx))
        assert info["refined"], info
        lv = field.levels[-1]
        lv.values.data = torch.randn(lv.values.shape, generator=gen).to(device)
    return field


def _clone_with_dtype(field, transition_compute_dtype):
    clone = HierarchicalBSplineField(
        field.levels[0].grid_size, field.spline_degree, BOUNDS,
        channels=field.channels, max_levels=field.max_levels,
        transfer_mode=field.transfer_mode,
        transition_compute_dtype=transition_compute_dtype,
        device=field.levels[0].values.device,
    )
    clone.restore_levels_from_state_dict(field.state_dict(), prefix="")
    return clone


def _report(name, a, b):
    diff = (a - b).abs()
    abs_err = diff.max().item()
    rel_err = (diff / b.abs().clamp_min(1e-6)).max().item()
    denom = b.abs().max().item()
    print(f"    {name}: max abs err = {abs_err:.3e} (|ref| max {denom:.3e}), "
          f"max rel err = {rel_err:.3e}")
    return abs_err, rel_err


def test_half_transition_accuracy():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[test] half-precision transition accuracy (device={device})")
    print(f"  allow_fp16_reduced_precision_reduction = "
          f"{torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}")
    print(f"  allow_bf16_reduced_precision_reduction = "
          f"{torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}")
    print("  (PyTorch issues fp16/bf16 GEMMs to cuBLAS with fp32 compute type;")
    print("   with the flags above at their default True, split-k reductions")
    print("   may still accumulate partial sums in fp16/bf16.)")

    gen = torch.Generator().manual_seed(31)
    for degree in (2, 3):
        ref = _make_field(degree, rounds=2, gen=gen, device=device,
                          transition_compute_dtype="fp32")
        pts = _random_points(8192, gen, device)
        v_ref = ref.evaluate(pts)
        val_ref, grad_ref, hess_ref = ref.evaluate_with_deriv(pts)
        for half in ("fp16", "bf16"):
            fld = _clone_with_dtype(ref, half)
            assert fld._transitions_half.dtype == getattr(torch, half.replace("fp", "float").replace("bf", "bfloat"))
            v = fld.evaluate(pts)
            val, grad, hess = fld.evaluate_with_deriv(pts)
            print(f"  degree={degree} levels={ref.num_levels} {half} vs fp32:")
            _report("evaluate", v, v_ref)
            _report("deriv.values", val, val_ref)
            _report("deriv.grads", grad, grad_ref)
            _report("deriv.hess", hess, hess_ref)
            # Sanity thresholds: fp16 ~1e-3 relative, bf16 ~1e-2 relative.
            lim = 5e-3 if half == "fp16" else 5e-2
            assert (v - v_ref).abs().max().item() < lim * max(1.0, v_ref.abs().max().item())
            assert (val - val_ref).abs().max().item() < lim * max(1.0, val_ref.abs().max().item())
            assert (grad - grad_ref).abs().max().item() < lim * max(1.0, grad_ref.abs().max().item())
            assert (hess - hess_ref).abs().max().item() < lim * max(1.0, hess_ref.abs().max().item())
    print("  ok")


def test_fp32_default_unchanged():
    print("[test] transition_compute_dtype='fp32' is the default and bit-exact")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator().manual_seed(32)
    a = _make_field(2, rounds=2, gen=gen, device=device)
    assert a.transition_compute_dtype == "fp32"
    assert not hasattr(a, "_transitions_half")
    b = _clone_with_dtype(a, "fp32")
    assert not hasattr(b, "_transitions_half")
    pts = _random_points(4096, gen, device)
    assert torch.equal(a.evaluate(pts), b.evaluate(pts))
    va, ga, ha = a.evaluate_with_deriv(pts)
    vb, gb, hb = b.evaluate_with_deriv(pts)
    assert torch.equal(va, vb) and torch.equal(ga, gb) and torch.equal(ha, hb)
    # Invalid value rejected.
    try:
        HierarchicalBSplineField(8, 2, BOUNDS, channels=1, transition_compute_dtype="fp64")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid transition_compute_dtype not rejected")
    print("  ok")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_fp32_default_unchanged()
    test_half_transition_accuracy()
    print("ALL THB TRANSITION-DTYPE TESTS PASSED")
