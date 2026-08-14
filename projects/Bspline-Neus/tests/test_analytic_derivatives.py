"""解析导数（evaluate_with_deriv）与 autograd 参照的数值对齐测试。

Run from the repository root:

    python projects/Bspline-Neus/tests/test_analytic_derivatives.py
    python -m pytest projects/Bspline-Neus/tests/test_analytic_derivatives.py -q

Checks:
1. Dense BSplineField (deg 3/2/1): analytic gradient and diagonal Hessian
   match nested-autograd derivatives of ``evaluate``.
2. HB HierarchicalBSplineField (2 levels): same comparison, ground truth
   from autograd on ``evaluate``.
3. THB HierarchicalBSplineField (2 levels): same comparison (truncation
   recursion commutes with differentiation).
4. Backward-graph: a scalar loss on the analytic gradient/hessian produces
   non-zero parameter gradients (first-order graph into control points).
5. (GPU only) the analytic path is CUDA-graph capture safe.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

field_mod = importlib.import_module("projects.Bspline-Neus.bspline_field.field")
hier_mod = importlib.import_module("projects.Bspline-Neus.bspline_field.hierarchical")

BSplineField = field_mod.BSplineField
HierarchicalBSplineField = hier_mod.HierarchicalBSplineField

BOUNDS = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]

GRAD_TOL = 1e-4
HESS_TOL = 1e-3


def _rand_pts(n, gen):
    # 远离外边界（两级实现都会在边界处 clamp，导数语义不同）。
    return (torch.rand(n, 3, generator=gen) * 2.0 - 1.0) * 0.9


def _autograd_reference(eval_fn, pts, channels):
    """Ground truth: per-channel gradient and diagonal Hessian via nested autograd.

    ``eval_fn`` maps (N, 3) -> (N, C).  Returns (grads (N,C,3), hess_diag (N,C,3)).
    """
    p = pts.detach().clone().requires_grad_(True)
    v = eval_fn(p)
    g_list, h_list = [], []
    for c in range(channels):
        gc = torch.autograd.grad(v[:, c].sum(), p, create_graph=True)[0]  # (N, 3)
        hc = []
        for d in range(3):
            hd = torch.autograd.grad(gc[:, d].sum(), p, retain_graph=True)[0][:, d]
            hc.append(hd)
        g_list.append(gc)
        h_list.append(torch.stack(hc, dim=-1))
    return torch.stack(g_list, dim=1), torch.stack(h_list, dim=1)


def _compare(name, values_a, grads_a, hess_a, values_ref, grads_ref, hess_ref):
    verr = (values_a - values_ref).abs().max().item()
    gerr = (grads_a - grads_ref).abs().max().item()
    herr = (hess_a - hess_ref).abs().max().item()
    print(f"  {name}: value err = {verr:.3e}, grad err = {gerr:.3e}, hess err = {herr:.3e}")
    assert verr < GRAD_TOL, verr
    assert gerr < GRAD_TOL, gerr
    assert herr < HESS_TOL, herr
    return gerr, herr


def test_dense_analytic_derivatives():
    print("[test 1] dense BSplineField analytic derivatives vs autograd")
    gen = torch.Generator().manual_seed(0)
    for degree, grid_size in ((3, 10), (2, 8), (1, 6)):
        coeffs = torch.randn(grid_size, grid_size, grid_size, generator=gen)
        field = BSplineField(
            grid_size, coeffs, bounds=BOUNDS, spline_degree=degree, trainable=True
        )
        pts = _rand_pts(4096, gen)

        values, grads, hess = field.evaluate_with_deriv(pts)
        assert values.shape == (4096,)
        assert grads.shape == (4096, 3) and hess.shape == (4096, 3)
        values_ref = field.evaluate(pts)
        grads_ref, hess_ref = _autograd_reference(
            lambda p: field.evaluate(p).unsqueeze(-1), pts, channels=1
        )
        _compare(
            f"deg{degree}",
            values, grads, hess,
            values_ref, grads_ref.squeeze(1), hess_ref.squeeze(1),
        )

        # Scalar (3,) input shape handling.
        v1, g1, h1 = field.evaluate_with_deriv(pts[0])
        assert v1.ndim == 0 and g1.shape == (3,) and h1.shape == (3,)
    print("  ok")


def _make_hier(gen, transfer_mode, degree=2, base_grid=8, channels=2):
    hier = HierarchicalBSplineField(
        base_grid, degree, BOUNDS, channels=channels, max_levels=2,
        transfer_mode=transfer_mode,
    )
    hier.levels[0].values.data = torch.randn(
        hier.levels[0].values.shape, generator=gen
    )
    cc = hier.levels[0].cell_count

    def mark_fn(level_index, corner_points):
        m = torch.zeros(cc, cc, cc, dtype=torch.bool)
        lo, hi = cc // 4, 3 * cc // 4
        m[lo:hi, lo:hi, lo:hi] = True
        return m

    info = hier.refine(mark_fn)
    assert info["refined"], info
    return hier


def test_hb_analytic_derivatives():
    print("[test 2] HB hierarchical analytic derivatives vs autograd")
    gen = torch.Generator().manual_seed(1)
    hier = _make_hier(gen, "hb")
    assert hier.num_levels == 2
    pts = _rand_pts(4096, gen)

    values, grads, hess = hier.evaluate_with_deriv(pts)
    assert values.shape == (4096, 2)
    assert grads.shape == (4096, 2, 3) and hess.shape == (4096, 2, 3)
    values_ref = hier.evaluate(pts)
    grads_ref, hess_ref = _autograd_reference(hier.evaluate, pts, channels=2)
    _compare("hb-2level", values, grads, hess, values_ref, grads_ref, hess_ref)
    print("  ok")


def test_thb_analytic_derivatives():
    print("[test 3] THB hierarchical analytic derivatives vs autograd")
    gen = torch.Generator().manual_seed(2)
    hier = _make_hier(gen, "thb")
    assert hier.num_levels == 2 and hier.transfer_mode == "thb"
    pts = _rand_pts(4096, gen)

    values, grads, hess = hier.evaluate_with_deriv(pts)
    values_ref = hier.evaluate(pts)  # dispatches to evaluate_thb
    grads_ref, hess_ref = _autograd_reference(hier.evaluate, pts, channels=2)
    _compare("thb-2level", values, grads, hess, values_ref, grads_ref, hess_ref)
    print("  ok")


def test_backward_graph_reaches_control_points():
    print("[test 4] scalar loss on analytic grad/hess backprops into control points")
    gen = torch.Generator().manual_seed(3)
    pts = _rand_pts(512, gen)

    # Dense field.
    coeffs = torch.randn(8, 8, 8, generator=gen)
    field = BSplineField(8, coeffs, bounds=BOUNDS, spline_degree=3, trainable=True)
    _, grads, hess = field.evaluate_with_deriv(pts)
    loss = (grads ** 2).sum() + (hess ** 2).sum()
    loss.backward()
    assert field.control_grid.grad is not None
    assert field.control_grid.grad.abs().sum() > 0
    print(f"  dense: |control_grid.grad| sum = {field.control_grid.grad.abs().sum():.3e}")

    # HB hierarchy (2 levels): every level's values must receive gradient.
    hier = _make_hier(gen, "hb", channels=1)
    _, grads, hess = hier.evaluate_with_deriv(pts)
    loss = ((grads.norm(dim=-1) - 1.0) ** 2).mean() + (hess ** 2).mean()
    loss.backward()
    for idx, lv in enumerate(hier.levels):
        assert lv.values.grad is not None and lv.values.grad.abs().sum() > 0, idx
    print("  hb: gradient flows into every level")

    # THB hierarchy: same through the truncation recursion.
    hier = _make_hier(gen, "thb", channels=1)
    _, grads, hess = hier.evaluate_with_deriv(pts)
    loss = ((grads.norm(dim=-1) - 1.0) ** 2).mean() + (hess ** 2).mean()
    loss.backward()
    for idx, lv in enumerate(hier.levels):
        assert lv.values.grad is not None and lv.values.grad.abs().sum() > 0, idx
    print("  thb: gradient flows into every level")
    print("  ok")


def test_cuda_graph_capture_safe():
    print("[test 5] analytic path is CUDA-graph capture safe (GPU only)")
    if not torch.cuda.is_available():
        print("  skipped (no CUDA)")
        return
    gen = torch.Generator().manual_seed(4)
    coeffs = torch.randn(10, 10, 10, generator=gen)
    field = BSplineField(
        10, coeffs, bounds=BOUNDS, spline_degree=3, trainable=True
    ).cuda()
    pts = _rand_pts(4096, gen).cuda()

    ref = [t.detach().clone() for t in field.evaluate_with_deriv(pts)]

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            field.evaluate_with_deriv(pts)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        v_s, g_s, h_s = field.evaluate_with_deriv(pts)
    graph.replay()
    torch.cuda.synchronize()
    for name, captured, expected in zip(("v", "g", "h"), (v_s, g_s, h_s), ref):
        err = (captured - expected).abs().max().item()
        assert err == 0.0, (name, err)
    print("  capture + replay matches eager exactly")
    print("  ok")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_dense_analytic_derivatives()
    test_hb_analytic_derivatives()
    test_thb_analytic_derivatives()
    test_backward_graph_reaches_control_points()
    test_cuda_graph_capture_safe()
    print("ALL ANALYTIC DERIVATIVE TESTS PASSED")
