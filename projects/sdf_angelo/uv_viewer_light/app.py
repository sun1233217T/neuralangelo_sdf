import argparse
import os
import sys
import time

import numpy as np

sys.path.append(os.getcwd())

from projects.sdf_angelo.uv_viewer_light.camera import OrbitCamera
from projects.sdf_angelo.uv_viewer_light.generator import build_original_runtime, build_student_runtime
from projects.sdf_angelo.uv_viewer_light.renderer import GLMeshRenderer


def _build_parser():
    parser = argparse.ArgumentParser(description="Local UV viewer using pygame + PyOpenGL")
    parser.add_argument("--window_width", default=1600, type=int)
    parser.add_argument("--window_height", default=900, type=int)
    parser.add_argument("--update_fps", default=-1.0, type=float,
                        help="Texture update rate. Use -1 for no limit.")
    parser.add_argument("--batch_size", default=262144, type=int)
    parser.add_argument("--uv_padding", default=2, type=int)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--vsync", action="store_true")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")))
    parser.add_argument("--single_gpu", action="store_true")

    subparsers = parser.add_subparsers(dest="runtime", required=True)

    original = subparsers.add_parser("original", help="Use the original UV runtime.")
    original.add_argument("--mesh", default="", type=str,
                          help="Path to UV mesh. Optional if --uv_bundle contains mesh_obj.")
    original.add_argument("--uv_bundle", default="", type=str,
                          help="Path to a saved UV bundle.")
    original.add_argument("--config", default="", type=str)
    original.add_argument("--checkpoint", default="", type=str)
    original.add_argument("--texture_size", default=4096, type=int)
    original.add_argument("--uv_raster", choices=["gpu", "cpu"], default="gpu")
    original.add_argument("--uv_project_to_surface", action="store_true")
    original.add_argument("--uv_project_iters", default=1, type=int)
    original.add_argument("--uv_project_step", default=1.0, type=float)
    original.add_argument("--uv_project_max_step", default=0.0, type=float)
    original.add_argument("--appear_idx", default=None, type=int)

    student = subparsers.add_parser("student", help="Use the distilled student UV runtime.")
    student.add_argument("--mesh", required=True, type=str)
    student.add_argument("--student_ckpt", required=True, type=str)
    student.add_argument("--dataset_dir", default="", type=str,
                         help="Directory containing geometry.pt and base_texture.png.")
    student.add_argument("--geometry", default="", type=str)
    student.add_argument("--base_texture", default="", type=str)

    return parser


def _import_pygame():
    try:
        import pygame
    except Exception as exc:
        raise RuntimeError(
            "pygame is required for uv_viewer_light. Install pygame on the local playback machine."
        ) from exc
    return pygame


def _set_caption(pygame, mode, draw_fps, generator_stats):
    tex_fps = generator_stats.get("update_fps", 0.0) or 0.0
    render_ms = generator_stats.get("render_ms", 0.0) or 0.0
    total_ms = generator_stats.get("total_ms", 0.0) or 0.0
    updates = int(generator_stats.get("updates", 0) or 0)
    title = (
        f"UV Viewer Light [{mode}] | draw {draw_fps:.1f} fps | "
        f"tex {tex_fps:.1f} fps | render {render_ms:.1f} ms | total {total_ms:.1f} ms | updates {updates}"
    )
    pygame.display.set_caption(title)


def _handle_camera_event(event, pygame, camera, viewport_size, drag_state):
    if event.type == pygame.MOUSEBUTTONDOWN:
        if event.button == 1:
            drag_state["left"] = True
            drag_state["trackball"] = camera.project_trackball(event.pos, viewport_size)
        elif event.button in (2, 3):
            drag_state["right"] = True
        elif event.button == 4:
            camera.zoom(+1.0)
            return True
        elif event.button == 5:
            camera.zoom(-1.0)
            return True
        drag_state["last_pos"] = np.array(event.pos, dtype=np.float32)
        return False
    if event.type == pygame.MOUSEBUTTONUP:
        if event.button == 1:
            drag_state["left"] = False
            drag_state["trackball"] = None
        elif event.button in (2, 3):
            drag_state["right"] = False
        drag_state["last_pos"] = np.array(event.pos, dtype=np.float32)
        return False
    if event.type == pygame.MOUSEMOTION:
        last_pos = drag_state["last_pos"]
        curr_pos = np.array(event.pos, dtype=np.float32)
        if last_pos is None:
            drag_state["last_pos"] = curr_pos
            return False
        delta = curr_pos - last_pos
        drag_state["last_pos"] = curr_pos
        changed = False
        if drag_state["left"]:
            start_trackball = drag_state.get("trackball")
            curr_trackball = camera.project_trackball(curr_pos, viewport_size)
            if start_trackball is None:
                drag_state["trackball"] = curr_trackball
            else:
                camera.trackball_rotate(start_trackball, curr_trackball)
                drag_state["trackball"] = curr_trackball
            changed = True
        if drag_state["right"]:
            camera.pan(delta[0], delta[1], viewport_size)
            changed = True
        return changed
    if event.type == pygame.MOUSEWHEEL:
        camera.zoom(float(event.y))
        return True
    return False


def _print_controls():
    print("Controls:")
    print("  Left drag: trackball rotate")
    print("  Right or middle drag: pan")
    print("  Mouse wheel: zoom")
    print("  R: reset camera")
    print("  Tab: toggle wireframe")
    print("  FPS/stats: window title")
    print("  Esc / Q: quit")


def _apply_initial_camera(camera, initial_camera):
    camera.set_from_position(initial_camera)


def main():
    parser = _build_parser()
    args, cfg_cmd = parser.parse_known_args()
    pygame = _import_pygame()

    if args.runtime == "original":
        runtime = build_original_runtime(args, cfg_cmd)
    else:
        runtime = build_student_runtime(args)

    mesh = runtime.mesh
    generator = runtime.generator
    camera = OrbitCamera.from_bounds(mesh.bounds)
    _apply_initial_camera(camera, runtime.initial_camera)

    flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    if args.fullscreen:
        flags |= pygame.FULLSCREEN
    try:
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        window = pygame.display.set_mode(
            (args.window_width, args.window_height),
            flags,
            vsync=1 if args.vsync else 0,
        )
    except TypeError:
        window = pygame.display.set_mode((args.window_width, args.window_height), flags)
    except Exception as exc:
        raise RuntimeError("Failed to create pygame OpenGL window.") from exc

    viewer = GLMeshRenderer(mesh, texture_size=runtime.texture_size)
    viewer.resize(window.get_width(), window.get_height())
    _print_controls()

    generator.update_camera(camera.get_position())
    generator.start()

    clock = pygame.time.Clock()
    drag_state = {"left": False, "right": False, "last_pos": None, "trackball": None}
    last_caption = 0.0
    last_update_seen = -1
    running = True
    reset_camera = np.asarray(runtime.initial_camera, dtype=np.float32)
    try:
        while running:
            camera_changed = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    continue
                if event.type == pygame.VIDEORESIZE:
                    window = pygame.display.set_mode((event.w, event.h), flags)
                    viewer.resize(event.w, event.h)
                    continue
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                        continue
                    if event.key == pygame.K_r:
                        camera.reset_from_bounds(mesh.bounds)
                        _apply_initial_camera(camera, reset_camera)
                        camera_changed = True
                        continue
                    if event.key == pygame.K_TAB:
                        viewer.toggle_wireframe()
                        continue
                if _handle_camera_event(
                    event,
                    pygame=pygame,
                    camera=camera,
                    viewport_size=(window.get_width(), window.get_height()),
                    drag_state=drag_state,
                ):
                    camera_changed = True

            if camera_changed:
                generator.update_camera(camera.get_position())

            stats = generator.get_stats()
            update_count = int(stats.get("updates", 0) or 0)
            if update_count != last_update_seen:
                texture_bytes = generator.get_texture_bytes()
                if texture_bytes is not None:
                    viewer.upload_texture(texture_bytes)
                    last_update_seen = update_count

            viewer.draw(camera, (window.get_width(), window.get_height()))
            pygame.display.flip()

            now = time.time()
            if now - last_caption > 0.25:
                draw_fps = float(clock.get_fps())
                _set_caption(pygame, runtime.mode, draw_fps, stats)
                last_caption = now
            clock.tick(120)
    finally:
        generator.stop()
        viewer.shutdown()
        pygame.quit()


if __name__ == "__main__":
    main()
