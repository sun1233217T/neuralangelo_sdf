# UV Viewer Light

Local UV mesh playback with `pygame + PyOpenGL`.

This viewer is intended for local machines with a display. It avoids the HTTP/web frontend in `uv_viewer` and renders the mesh directly in an OpenGL window.

## Dependencies

Install these on the playback machine:

```bash
pip install pygame PyOpenGL PyOpenGL_accelerate
```

The existing project dependencies are still required for model loading.

## Controls

- Left drag: orbit
- Right or middle drag: pan
- Mouse wheel: zoom
- `R`: reset camera
- `Tab`: toggle wireframe
- `Esc` / `Q`: quit

## Original Runtime

Use a precomputed UV bundle when possible:

```bash
python projects/sdf_angelo/uv_viewer_light/app.py original \
  --uv_bundle path/to/uv_bundle.pt \
  --mesh path/to/mesh.obj
```

Or build the UV cache on the fly:

```bash
python projects/sdf_angelo/uv_viewer_light/app.py original \
  --config projects/sdf_angelo/configs/dtu-win.yaml \
  --checkpoint path/to/checkpoint.pt \
  --mesh path/to/mesh.obj \
  --texture_size 4096 \
  --uv_project_to_surface
```

## Student Runtime

```bash
python projects/sdf_angelo/uv_viewer_light/app.py student \
  --mesh path/to/mesh.obj \
  --student_ckpt path/to/student_best.pt \
  --dataset_dir path/to/uv_teacher_dataset
```

If the distilled assets are stored separately, pass `--geometry` and `--base_texture` explicitly.
The viewer also accepts the existing median base output at `base/base_median.png`.
