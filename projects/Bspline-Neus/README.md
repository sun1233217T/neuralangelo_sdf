# Bspline-Neus

A lightweight B-spline NeuS implementation for `neuralangelo_sdf`.  This project
replaces the MLP-based `NeuralSDF` and `NeuralRGB` of Neuralangelo with
tensor-product B-spline fields (SDF grid + SH color grid) while reusing the
original training loop, background NeRF, losses, logging, and checkpointing.

## Approach

This is the **point-sampling adapter** implementation:

- Keeps Neuralangelo's ray sampling and NeuS volume rendering.
- Replaces only the SDF/RGB representation with B-spline fields.
- Does **not** require the `bspline_field_cuda` extension; it runs with pure
  PyTorch B-spline evaluation.

## File layout

```
projects/Bspline-Neus/
├── bspline_field/          # pure-PyTorch B-spline field (from bijectiveImplicitShell-nerf)
│   ├── field.py            #   dense BSplineField
│   └── hierarchical.py     #   sparse hierarchical (HB/THB) field
├── utils/
│   └── modules.py          # BSplineSDFWrapper / BSplineRGBWrapper
├── model.py                # inherits projects.neuralangelo.model.Model
├── trainer.py              # inherits projects.neuralangelo.trainer.Trainer
├── data.py                 # reuses projects.neuralangelo.data.Dataset
├── configs/
│   ├── base.yaml
│   ├── dtu_scan24_bspline_neus.yaml
│   └── dtu_scan24_thb.yaml
├── tests/                  # standalone test scripts (run from repo root)
└── README.md
```

## Usage

Activate the `(neuralangelo)` environment and run from the repository root:

```bash
python train.py --config projects/Bspline-Neus/configs/dtu_scan24_bspline_neus.yaml \
                --single_gpu --wandb
```

For a quick smoke test without real data, a synthetic dataset is provided under
`datasets/synthetic_bspline_neus/` together with
`projects/Bspline-Neus/configs/synthetic_bspline_neus.yaml`.

```bash
python train.py --config projects/Bspline-Neus/configs/synthetic_bspline_neus.yaml \
                --single_gpu --max_iter 5
```

## Tight scene bounds (auto_bounds)

To reduce the effective world extent and save VRAM, enable `data.auto_bounds`.
It computes a quantile-based AABB from `points.ply` and derives the
`readjust.center/scale` and `model.object.bspline.bounds` automatically:

```yaml
data:
    auto_bounds:
        enabled: True
        target_half_bound: 2.0
        ply_filename: points.ply
        quantile_low: 0.03
        quantile_high: 0.99
        padding_ratio: 0.15
        force_cube: True
```

Two standalone smoke-test scripts are also left in the repository root:

```bash
python test_bspline_neus.py          # module/model forward & inference
python test_bspline_neus_trainer.py  # trainer construction + one train step
```

## Key configuration

All B-spline hyperparameters live under `model.object.bspline`:

| YAML key | Meaning | Default |
|---|---|---|
| `grid_size` | Control grid resolution `G` (total `G³` SDF params) | 128 |
| `sdf_spline_degree` | B-spline degree for SDF | 2 |
| `color_spline_degree` | B-spline degree for color | 2 |
| `color_sh_degree` | Spherical-harmonics degree for view-dependent color | 2 |
| `bounds` | Axis-aligned bounding box `[[xmin,xmax],[ymin,ymax],[zmin,zmax]]` | `[-2,2]³` |
| `sdf_init` | Constant SDF initialization value | 0.1 |
| `sdf_inv_s_init` | Initial NeuS inverse std `s` | 1.0 |
| `sdf_init_mode` | `"constant"` or `"sphere"` | `"sphere"` |
| `color_init` | Initial RGB | `[0.5,0.5,0.5]` |
| `color_lr` | Learning rate for the color grid | 0.08 |
| `color_sh_rest_lr_multiplier` | LR multiplier for SH bands > 0 | 0.025 |

Loss weights are controlled via `trainer.loss_weight`:

```yaml
trainer:
  loss_weight:
    render: 1.0
    eikonal: 5e-4
    curvature: 5e-4
```

## Learning rates

`model.get_param_groups()` provides three groups:

1. SDF grid + inverse std → `optim.params.lr` (default 3e-3).
2. Color grid → `model.object.bspline.color_lr` (default 0.08).
3. Background NeRF / appearance embeddings → `optim.params.lr`.

## Hierarchical (HB / THB) B-splines

Setting `model.object.bspline.hierarchical.enabled: True` replaces the dense
grids with a sparse hierarchical B-spline field
(`bspline_field/hierarchical.py`).  Instead of one dense `G³` grid, the field
is a sum of dyadic levels where each level stores control values only where
they are needed (sparse `values` + dense `index_grid` over a boolean cell
`region`).  This directly targets VRAM: fine resolution is allocated only in
a band around the surface.

```yaml
model:
  object:
    bspline:
      hierarchical:
        enabled: True
        color_enabled: True        # color also uses the hierarchical backend
        base_grid_size: 34         # level 0 (cells = 34 - degree)
        max_levels: 4              # 34 -> 66 -> 130 -> 258 (cells x2 each level)
        refine_iters: [20000, 60000, 120000]
        refine_sdf_band: 0.05      # refine cells whose corner |sdf| < band
        transfer_mode: thb         # "hb" or "thb" (SDF)
        color_transfer_mode: hb    # optional override for the color field
```

Semantics:

- **Refinement is exact**: at each `refine_iters` step, cells of the finest
  level whose corners come within `refine_sdf_band` of the surface move to a
  new finer level, and coefficients are transferred so the represented field
  (and its gradient) is unchanged.  The optimizer is rebuilt automatically
  after each refinement.
- **`hb`**: coarse basis functions straddling the refined region stay active
  untouched; only fully-refined ones are removed and prolonged into the fine
  level.  Cheap and exact.
- **`thb`**: coarse basis functions are *truncated* at evaluation time
  (`HierarchicalBSplineField.evaluate_thb`), removing coarse support inside
  finer levels.  The transfer must then equal the exact fine-level
  coefficients of the whole field, which is computed by the "shadow chain"
  (`_thb_full_transfer`): a per-level dense prolongation masked to the
  complement of each level's responsibility set.  For 27-channel color this
  chain needs full dense prolongation grids at refine time, so
  `color_transfer_mode: hb` is the recommended pairing with a THB SDF.

Ready-made config: `projects/Bspline-Neus/configs/dtu_scan24_thb.yaml`
(500k schedule) and `projects/Bspline-Neus/configs/dtu_scan24_thb_short.yaml`
(20k validation schedule with refinement at 2k/6k/12k).

**Checkpoint resume**: refined levels only exist in the checkpoint, not in a
freshly built model.  The model implements `prepare_load_state_dict`, which
the checkpointer calls *before* its key/shape filtering to grow the
hierarchy to the checkpoint's depth (`restore_levels_from_state_dict`).
Resume and mesh extraction across refinement boundaries therefore just work;
optimizer state is only meaningful when the loaded structure matches (it
does, via the hook).

**Short-run benchmark (DTU scan24, 20k iters, final cell resolution 256)**:

| | dense 256 baseline | THB (4 levels) |
|---|---|---|
| VRAM mean / peak | 12.6 / 16.0 GB | 5.6 / 8.7 GB |
| active SDF control points | 258³ = 17.2 M | ≈ 1.53 M (≈ 9%) |
| iteration speed | ≈ 5.1 it/s | ≈ 5.0 it/s |

Refinements triggered at 2k/6k/12k (9.5k → 50.7k → 287.8k marked cells),
training stayed stable, and mesh extraction from the final checkpoint works
via `projects/Bspline-Neus/scripts/extract_mesh.py`.

**Full-run benchmark (DTU scan24, 500k iters, final cell resolution 256)**:

| | dense 256 baseline | THB (4 levels, refine at 20k/60k/120k) |
|---|---|---|
| val PSNR (4-sample val split) | 24.42 | 22.92 |
| Chamfer (d2s / s2d / overall, raw eval units) | 7.95 / 8.13 / 8.04 | 9.64 / 13.85 / 11.74 |
| VRAM mean / peak (nvidia-smi, whole run) | not measured (20k short run: 12.6 / 16.0 GB) | 3.5 / 8.7 GB |
| SDF control points | 258³ = 17.2 M | 787.5 k (≈ 4.6 %) |
| iteration speed | ≈ 5 it/s | ≈ 5 it/s daytime (≈ 2.8 it/s with desktop GPU contention) |
| mesh faces @512³ marching cubes | 602 k | 433 k |

THB run: `logs/2026_0727_1635_40_dtu_scan24_thb/` (config
`projects/Bspline-Neus/configs/dtu_scan24_thb.yaml`); dense baseline:
`logs/Bneus_test256/`.  PSNR is computed by
`projects/Bspline-Neus/scripts/eval_psnr.py`, Chamfer by
`projects/sdf_angelo/scripts/eval_DTU_mesh.py` with
`--apply_scale_mat` (json dumps in `meshout/eval_thb/` and
`meshout/eval_dense/`).

Takeaway: THB reaches the surface hierarchy it was designed for (SDF
coefficients cut by ≈ 22×, steady-state VRAM ≈ 3.5 GB vs ≈ 12.6 GB) at the
same iteration speed, but on scan24 it gives back roughly 1.5 dB of val PSNR
and a third of the Chamfer accuracy compared to the dense 256 grid.  The gap
is concentrated in s2d (completeness), suggesting the refine band/schedule —
not the representation — is the current bottleneck.

Tests (run from the repository root):

```bash
python projects/Bspline-Neus/tests/test_thb_consistency.py   # subdivision/HB exactness
python projects/Bspline-Neus/tests/test_thb_truncation.py    # THB truncation exactness
python projects/Bspline-Neus/tests/test_thb_wrapper.py       # SDF wrapper + eikonal + THB e2e
python projects/Bspline-Neus/tests/test_thb_color.py         # hierarchical color field
python projects/Bspline-Neus/tests/test_thb_trainer.py       # trainer refine + optimizer rebuild
python projects/Bspline-Neus/tests/check_mesh_extract_thb.py # mesh extraction compatibility
```

## Differences from the source project

- Uses YAML configs and Neuralangelo's data pipeline instead of the original
  TOML/DTU loader.
- No CUDA DDA renderer; sampling follows Neuralangelo's fixed + hierarchical
  sampling strategy.
- No occupancy brick hierarchy or degree curriculum in this lightweight version.
- `s_var` is replaced by `neural_sdf.raw_sdf_inv_std` (log-space, softplus).

## Notes

- The project folder name contains a hyphen (`Bspline-Neus`).  Python imports use
  `importlib.import_module("projects.Bspline-Neus.model")` (done automatically by
  the config system).  Internal code uses relative imports to avoid syntax issues.
- Set `checkpoint.strict_resume: False` when loading weights, because the B-spline
  state-dict keys differ from the original Neuralangelo MLP keys.
