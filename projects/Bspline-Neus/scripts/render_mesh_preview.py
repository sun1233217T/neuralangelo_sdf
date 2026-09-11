"""Render a mesh PLY to a few preview images using Open3D offscreen rendering."""
import sys
import numpy as np
import open3d as o3d

path = sys.argv[1] if len(sys.argv) > 1 else "meshout/scan24_E22b_500k.ply"
out_prefix = sys.argv[2] if len(sys.argv) > 2 else path.replace(".ply", "_preview")

mesh = o3d.io.read_triangle_mesh(path)
mesh.compute_vertex_normals()
print(f"Loaded: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

# Normalize to unit sphere for consistent framing
center = mesh.get_center()
scale = 1.0 / np.max(np.linalg.norm(np.asarray(mesh.vertices) - center, axis=1))
mesh.translate(-center)
mesh.scale(scale, [0, 0, 0])

vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=800, visible=False)
vis.add_geometry(mesh)

# Set material and lighting
opt = vis.get_render_option()
opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
opt.background_color = np.array([0.1, 0.1, 0.1])

angles = [(0, 0), (90, 0), (180, 0), (0, 90)]
for i, (azim, elev) in enumerate(angles):
    ctr = vis.get_view_control()
    # Reset to default view
    vis.reset_view_point(True)
    ctr.set_front([np.cos(np.radians(azim)), np.sin(np.radians(elev)), -np.sin(np.radians(azim))])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.8)
    vis.poll_events()
    vis.update_renderer()
    out = f"{out_prefix}_{i}.png"
    vis.capture_screen_image(out, do_render=True)
    print(f"Saved {out}")

vis.destroy_window()
