import io
import json
import os
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
import trimesh
from PIL import Image

from imaginaire.config import Config, parse_cmdline_arguments, recursive_update_strict
from imaginaire.trainers.utils.get_trainer import get_trainer
from projects.sdf_angelo.uv_distill.common import load_geometry_cache
from projects.sdf_angelo.uv_distill.render import StudentTextureRenderer, load_student_checkpoint
from projects.sdf_angelo.uv_viewer.texture_generator import TextureGenerator
from projects.sdf_angelo.uv_viewer.uv_cache import build_uv_cache, load_uv_bundle


def _as_float3(value):
    return np.asarray(value, dtype=np.float32).reshape(3)


def _initial_camera_from_bounds(bounds):
    bounds = np.asarray(bounds, dtype=np.float32)
    center = bounds.mean(axis=0)
    size = bounds[1] - bounds[0]
    radius = np.linalg.norm(size) * 0.6
    return center + np.array([0.0, 0.0, max(radius, 1.0) * 2.5], dtype=np.float32)


def load_mesh(mesh_path=None, mesh_obj_text=None):
    if mesh_path:
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")
        mesh_loaded = trimesh.load(mesh_path, process=False)
    elif mesh_obj_text is not None:
        if isinstance(mesh_obj_text, bytes):
            mesh_obj_text = mesh_obj_text.decode("utf-8", errors="ignore")
        mesh_loaded = trimesh.load(io.StringIO(mesh_obj_text), file_type="obj", process=False)
    else:
        raise ValueError("Either mesh_path or mesh_obj_text is required.")
    if isinstance(mesh_loaded, trimesh.Scene):
        if not mesh_loaded.geometry:
            raise ValueError("Mesh scene is empty.")
        mesh = trimesh.util.concatenate(tuple(mesh_loaded.geometry.values()))
    else:
        mesh = mesh_loaded
    if not hasattr(mesh, "visual") or mesh.visual is None or getattr(mesh.visual, "uv", None) is None:
        raise ValueError("Input mesh does not contain UV coordinates.")
    return mesh


@dataclass
class RuntimeBundle:
    mode: str
    generator: object
    mesh: trimesh.Trimesh
    texture_size: int
    initial_camera: np.ndarray


class GeneratorAdapter:

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def update_camera(self, cam_pos_world):
        raise NotImplementedError

    def get_texture_bytes(self):
        raise NotImplementedError

    def get_stats(self):
        raise NotImplementedError


class OriginalGeneratorAdapter(GeneratorAdapter):

    def __init__(self, generator):
        self.generator = generator

    def start(self):
        self.generator.start()

    def stop(self):
        self.generator.stop()

    def update_camera(self, cam_pos_world):
        self.generator.update_camera(cam_pos_world)

    def get_texture_bytes(self):
        return self.generator.get_texture_bytes()

    def get_stats(self):
        return self.generator.get_stats()


class StudentTextureGenerator(GeneratorAdapter):

    def __init__(self, renderer, sphere_center, sphere_radius, update_fps, batch_size, pad_iters):
        self.renderer = renderer
        self.sphere_center = torch.as_tensor(_as_float3(sphere_center), device=renderer.device, dtype=torch.float32)
        self.sphere_radius = float(sphere_radius)
        self.batch_size = int(batch_size)
        self.pad_iters = int(max(pad_iters, 0))
        update_fps = float(update_fps)
        self.update_interval = 0.0 if update_fps < 0 else 1.0 / max(update_fps, 1e-6)

        self._lock = threading.Lock()
        self._texture_bytes = None
        self._texture_ready = threading.Event()
        self._stop = threading.Event()
        self._camera_lock = threading.Lock()
        self._camera_pos_world = None
        self._camera_dirty = False
        self._camera_event = threading.Event()
        self._thread = None
        self._stats_lock = threading.Lock()
        self._update_count = 0
        self._last_update_time = None
        self._last_update_fps = 0.0
        self._last_render_ms = 0.0
        self._last_total_ms = 0.0

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._camera_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def update_camera(self, cam_pos_world):
        with self._camera_lock:
            self._camera_pos_world = _as_float3(cam_pos_world)
            self._camera_dirty = True
        self._camera_event.set()

    def get_texture_bytes(self):
        if not self._texture_ready.wait(timeout=0.1):
            return None
        with self._lock:
            return self._texture_bytes

    def get_stats(self):
        with self._stats_lock:
            return {
                "updates": self._update_count,
                "update_fps": self._last_update_fps,
                "render_ms": self._last_render_ms,
                "encode_ms": 0.0,
                "total_ms": self._last_total_ms,
            }

    def _run(self):
        next_time = time.time()
        while not self._stop.is_set():
            if not self._camera_event.wait(timeout=0.1):
                continue
            if self._stop.is_set():
                break
            now = time.time()
            if self.update_interval > 0 and now < next_time:
                time.sleep(next_time - now)
            with self._camera_lock:
                if not self._camera_dirty or self._camera_pos_world is None:
                    self._camera_event.clear()
                    continue
                cam_pos = self._camera_pos_world.copy()
                self._camera_dirty = False
            self._camera_event.clear()
            render_start = time.time()
            texture_bytes = self._render_texture_bytes(cam_pos)
            render_ms = (time.time() - render_start) * 1000.0
            now = time.time()
            with self._lock:
                self._texture_bytes = texture_bytes
                self._texture_ready.set()
            with self._stats_lock:
                self._update_count += 1
                if self._last_update_time is not None:
                    dt = max(now - self._last_update_time, 1e-6)
                    inst_fps = 1.0 / dt
                    if self._last_update_fps == 0.0:
                        self._last_update_fps = inst_fps
                    else:
                        self._last_update_fps = 0.8 * self._last_update_fps + 0.2 * inst_fps
                self._last_update_time = now
                self._last_render_ms = render_ms
                self._last_total_ms = render_ms
            next_time = time.time() + self.update_interval

    def _render_texture_bytes(self, cam_pos_world):
        cam_norm = (
            torch.as_tensor(cam_pos_world, device=self.renderer.device, dtype=torch.float32)
            - self.sphere_center
        ) / self.sphere_radius
        texture = self.renderer.render_texture(
            camera_center_norm=cam_norm,
            batch_size=self.batch_size,
            pad_iters=self.pad_iters,
        )
        texture_u8 = (texture.clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
        return _pack_texture_rgba(texture_u8)


def _pack_texture_rgba(texture):
    height, width, channels = texture.shape
    if channels != 3:
        raise ValueError("Expected RGB texture.")
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = texture
    rgba[..., 3] = 255
    return rgba.tobytes()


def _load_scene_normalization_from_cfg(cfg):
    meta_fname = os.path.join(cfg.data.root, "transforms.json")
    with open(meta_fname, "r", encoding="utf-8") as file:
        meta = json.load(file)
    sphere_center = np.array(meta.get("sphere_center", [0.0, 0.0, 0.0]), dtype=np.float32)
    sphere_radius = float(meta.get("sphere_radius", 1.0))
    readjust = getattr(cfg.data, "readjust", None)
    if readjust is not None:
        sphere_center += np.array(getattr(readjust, "center", [0.0, 0.0, 0.0]), dtype=np.float32)
        sphere_radius *= float(getattr(readjust, "scale", 1.0))
    return sphere_center, sphere_radius


def build_original_runtime(args, cfg_cmd):
    device = torch.device(f"cuda:{int(getattr(args, 'local_rank', 0))}" if torch.cuda.is_available() else "cpu")
    if args.uv_bundle:
        uv_cache, neural_rgb, appear_embed, meta = load_uv_bundle(args.uv_bundle, device=device)
        sphere_center = np.asarray(meta.get("sphere_center", [0.0, 0.0, 0.0]), dtype=np.float32)
        sphere_radius = float(meta.get("sphere_radius", 1.0))
        if args.mesh:
            mesh = load_mesh(mesh_path=args.mesh)
            initial_camera = _initial_camera_from_bounds(mesh.bounds)
        else:
            mesh = load_mesh(mesh_obj_text=meta.get("mesh_obj"))
            init_cam = meta.get("init_camera")
            initial_camera = (
                np.asarray(init_cam, dtype=np.float32)
                if init_cam is not None
                else _initial_camera_from_bounds(mesh.bounds)
            )
        generator = TextureGenerator(
            uv_cache=uv_cache,
            neural_rgb=neural_rgb,
            appear_embed=appear_embed,
            sphere_center=sphere_center,
            sphere_radius=sphere_radius,
            update_fps=args.update_fps,
            batch_size=args.batch_size,
            appear_idx=args.appear_idx,
            encode_base64=False,
            include_raw=True,
            async_encode=True,
            pad_iters=args.uv_padding,
        )
        return RuntimeBundle(
            mode="original",
            generator=OriginalGeneratorAdapter(generator),
            mesh=mesh,
            texture_size=int(uv_cache.tex_size),
            initial_camera=initial_camera,
        )

    if not args.config or not args.checkpoint or not args.mesh:
        raise ValueError("original mode requires --uv_bundle or config/checkpoint/mesh.")

    cfg = Config(args.config)
    cfg_cmd = parse_cmdline_arguments(cfg_cmd)
    recursive_update_strict(cfg, cfg_cmd)
    cfg.logdir = ""

    trainer = get_trainer(cfg, is_inference=True, seed=0)
    trainer.checkpointer.load(args.checkpoint, load_opt=False, load_sch=False)
    trainer.model.eval()
    trainer.current_iteration = trainer.checkpointer.eval_iteration
    if cfg.model.object.sdf.encoding.coarse2fine.enabled:
        trainer.model_module.neural_sdf.set_active_levels(trainer.current_iteration)
        if cfg.model.object.sdf.gradient.mode == "numerical":
            trainer.model_module.neural_sdf.set_normal_epsilon()

    sphere_center, sphere_radius = _load_scene_normalization_from_cfg(cfg)
    mesh = load_mesh(mesh_path=args.mesh)
    uv_cache = build_uv_cache(
        mesh,
        trainer.model_module.neural_sdf,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
        texture_size=args.texture_size,
        raster_mode=args.uv_raster,
        batch_size=args.batch_size,
        project_to_surface=args.uv_project_to_surface,
        project_iters=args.uv_project_iters,
        project_step=args.uv_project_step,
        project_max_step=args.uv_project_max_step,
    )
    del trainer.model_module.neural_sdf
    generator = TextureGenerator(
        uv_cache=uv_cache,
        neural_rgb=trainer.model_module.neural_rgb,
        appear_embed=trainer.model_module.appear_embed,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
        update_fps=args.update_fps,
        batch_size=args.batch_size,
        appear_idx=args.appear_idx,
        encode_base64=False,
        include_raw=True,
        async_encode=True,
        pad_iters=args.uv_padding,
    )
    return RuntimeBundle(
        mode="original",
        generator=OriginalGeneratorAdapter(generator),
        mesh=mesh,
        texture_size=int(uv_cache.tex_size),
        initial_camera=_initial_camera_from_bounds(mesh.bounds),
    )


def _resolve_student_paths(args):
    geometry_path = args.geometry
    base_texture_path = args.base_texture
    if args.dataset_dir:
        if not geometry_path:
            geometry_path = os.path.join(args.dataset_dir, "geometry.pt")
        if not base_texture_path:
            candidates = [
                os.path.join(args.dataset_dir, "base_texture.png"),
                os.path.join(args.dataset_dir, "base", "base_median.png"),
                os.path.join(args.dataset_dir, "base", "base_median_raw.png"),
            ]
            for candidate in candidates:
                if os.path.isfile(candidate):
                    base_texture_path = candidate
                    break
    if not geometry_path:
        raise ValueError("student mode requires --geometry or --dataset_dir.")
    if not base_texture_path:
        raise ValueError("student mode requires --base_texture or --dataset_dir with base_texture.png.")
    return geometry_path, base_texture_path


def build_student_runtime(args):
    if not args.mesh:
        raise ValueError("student mode requires --mesh.")
    if not args.student_ckpt:
        raise ValueError("student mode requires --student_ckpt.")
    geometry_path, base_texture_path = _resolve_student_paths(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geometry = load_geometry_cache(geometry_path, device=device)
    model, _ = load_student_checkpoint(args.student_ckpt, texture_size=geometry["tex_size"], device=device)
    base_image = Image.open(base_texture_path).convert("RGB")
    base_rgb = torch.from_numpy(np.asarray(base_image, dtype=np.uint8))
    if base_rgb.shape[0] != geometry["tex_size"] or base_rgb.shape[1] != geometry["tex_size"]:
        raise ValueError(
            f"Base texture shape {tuple(base_rgb.shape[:2])} does not match geometry texture size {geometry['tex_size']}."
        )
    renderer = StudentTextureRenderer(model, geometry, base_rgb)
    generator = StudentTextureGenerator(
        renderer=renderer,
        sphere_center=geometry["sphere_center"],
        sphere_radius=geometry["sphere_radius"],
        update_fps=args.update_fps,
        batch_size=args.batch_size,
        pad_iters=args.uv_padding,
    )
    mesh = load_mesh(mesh_path=args.mesh)
    return RuntimeBundle(
        mode="student",
        generator=generator,
        mesh=mesh,
        texture_size=int(geometry["tex_size"]),
        initial_camera=_initial_camera_from_bounds(mesh.bounds),
    )
