"""SH coefficients and learned MLP features must use different gradient rules."""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
modules = importlib.import_module("projects.Bspline-Neus.utils.modules")


def make_wrapper(mode):
    cfg = SimpleNamespace(
        grid_size=8, sdf_spline_degree=2, color_spline_degree=2,
        color_sh_degree=2, bounds=[[-1., 1.]]*3, sdf_init=0.1,
        sdf_inv_s_init=1., sdf_init_mode="sphere", sdf_init_sphere_radius=0.5,
        sdf_init_sphere_center=[0., 0., 0.], color_init=[0.5]*3,
        color_lr=0.04, color_sh_rest_lr_multiplier=0.025,
        color_mode=mode, color_feature_dim=16, color_mlp_hidden_dims=[8],
        hierarchical=dict(enabled=True, color_enabled=True, base_grid_size=8,
            max_levels=2, refine_iters=[1], refine_sdf_band=10.,
            transfer_mode="hb", color_transfer_mode="hb", share_structure=True),
    )
    sdf = modules.BSplineSDFWrapper(cfg)
    return sdf, modules.BSplineRGBWrapper(cfg, sdf_wrapper=sdf)


def assert_gradient_rule(rgb, mode):
    for level in rgb.hier_field.levels:
        level.values.grad = None
        level.values.sum().backward()
        expected = torch.ones_like(level.values)
        if mode == "sh":
            expected[:, torch.arange(expected.shape[1]) % 9 != 0] = 0.025
        torch.testing.assert_close(level.values.grad, expected)


def test_mlp_features_after_restore_hook_registration():
    _, rgb = make_wrapper("mlp")
    rgb._register_sh_hooks()  # trainer's post-checkpoint hook
    rgb._register_sh_hooks()  # repeated registration must remain harmless
    assert_gradient_rule(rgb, "mlp")


def test_mlp_features_after_refinement():
    sdf, rgb = make_wrapper("mlp")
    info = sdf.maybe_refine(1)
    rgb.maybe_refine(1, sdf_wrapper=sdf, sdf_refine_info=info)
    assert len(rgb.hier_field.levels) == 2
    assert_gradient_rule(rgb, "mlp")


def test_sh_coefficients_keep_band_scaling():
    _, rgb = make_wrapper("sh")
    rgb._register_sh_hooks()
    rgb._register_sh_hooks()
    assert_gradient_rule(rgb, "sh")


if __name__ == "__main__":
    failures = 0
    for test in [test_mlp_features_after_restore_hook_registration,
                 test_mlp_features_after_refinement, test_sh_coefficients_keep_band_scaling]:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
    raise SystemExit(bool(failures))
