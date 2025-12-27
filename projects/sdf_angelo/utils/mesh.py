'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import numpy as np
import trimesh
import mcubes
import torch
import torch.distributed as dist
import torch.nn.functional as torch_F
from tqdm import tqdm
from trimesh.visual.texture import TextureVisuals
from PIL import Image

from imaginaire.utils.distributed import get_world_size, is_master


@torch.no_grad()
def extract_mesh(sdf_func, bounds, intv, block_res=64, texture_func=None, filter_lcc=False):
    lattice_grid = LatticeGrid(bounds, intv=intv, block_res=block_res)
    data_loader = get_lattice_grid_loader(lattice_grid)
    mesh_blocks = []
    if is_master():
        data_loader = tqdm(data_loader, leave=False)
    for it, data in enumerate(data_loader):
        xyz = data["xyz"][0]
        xyz_cuda = xyz.cuda()
        sdf_cuda = sdf_func(xyz_cuda)[..., 0]
        sdf = sdf_cuda.cpu()
        mesh = marching_cubes(sdf.numpy(), xyz.numpy(), intv, texture_func, filter_lcc)
        mesh_blocks.append(mesh)
    mesh_blocks_gather = [None] * get_world_size()
    if dist.is_initialized():
        dist.all_gather_object(mesh_blocks_gather, mesh_blocks)
    else:
        mesh_blocks_gather = [mesh_blocks]
    if is_master():
        mesh_blocks_all = [mesh for mesh_blocks in mesh_blocks_gather for mesh in mesh_blocks
                           if mesh.vertices.shape[0] > 0]
        mesh = trimesh.util.concatenate(mesh_blocks_all)
        return mesh
    else:
        return None


@torch.no_grad()
def extract_texture(xyz, neural_rgb, neural_sdf, appear_embed):
    num_samples, _ = xyz.shape
    xyz_cuda = torch.from_numpy(xyz).float().cuda()[None, None]  # [N,3] -> [1,1,N,3]
    sdfs, feats = neural_sdf(xyz_cuda)
    gradients, _ = neural_sdf.compute_gradients(xyz_cuda, training=False, sdf=sdfs)
    normals = torch_F.normalize(gradients, dim=-1)
    if appear_embed is not None:
        feat_dim = appear_embed.embedding_dim  # [1,1,N,C]
        app = torch.zeros([1, 1, num_samples, feat_dim], device=sdfs.device)  # TODO: hard-coded to zero. better way?
    else:
        app = None
    rgbs = neural_rgb.forward(xyz_cuda, normals, -normals, feats, app=app)  # [1,1,N,3]
    return (rgbs.squeeze().cpu().numpy() * 255).astype(np.uint8)


def _unpack_xatlas_result(result):
    if isinstance(result, (tuple, list)):
        if len(result) != 3:
            raise ValueError("Unexpected xatlas result tuple length.")
        return result
    if isinstance(result, dict):
        vmapping = result.get("vmapping") or result.get("vertex_mapping") or result.get("vertex_map")
        indices = result.get("indices") or result.get("index")
        uvs = result.get("uvs") or result.get("uv")
        if vmapping is not None and indices is not None and uvs is not None:
            return vmapping, indices, uvs
    if hasattr(result, "vmapping") and hasattr(result, "indices") and hasattr(result, "uvs"):
        return result.vmapping, result.indices, result.uvs
    if hasattr(result, "vertex_mapping") and hasattr(result, "index") and hasattr(result, "uv"):
        return result.vertex_mapping, result.index, result.uv
    raise ValueError("Unsupported xatlas result format.")


def _unwrap_with_xatlas_module(xatlas, vertices, faces):
    if hasattr(xatlas, "parametrize"):
        vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    else:
        atlas = xatlas.Atlas()
        atlas.add_mesh(vertices, faces)
        atlas.generate()
        if hasattr(atlas, "__getitem__"):
            result = atlas[0]
        elif hasattr(atlas, "meshes"):
            result = atlas.meshes[0]
        else:
            raise ValueError("Unsupported xatlas Atlas API.")
        vmapping, indices, uvs = _unpack_xatlas_result(result)
    return vmapping, indices, uvs


def unwrap_uv_xatlas(vertices, faces):
    try:
        import xatlas  # type: ignore
        vmapping, indices, uvs = _unwrap_with_xatlas_module(xatlas, vertices, faces)
    except Exception as exc:
        try:
            import pyxatlas  # type: ignore
            vmapping, indices, uvs = _unwrap_with_xatlas_module(pyxatlas, vertices, faces)
        except Exception as exc2:
            raise ImportError(
                "xatlas is required for UV unwrapping. Install with: pip install xatlas"
            ) from exc2
        else:
            if exc is not None:
                pass
    vmapping = np.asarray(vmapping, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim == 1:
        indices = indices.reshape(-1, 3)
    uvs = np.asarray(uvs, dtype=np.float32)
    uvs = np.clip(uvs[..., :2], 0.0, 1.0)
    vertices_uv = vertices[vmapping]
    return vertices_uv, indices, uvs


def _rasterize_uv(uvs, faces, texture_size):
    height = width = int(texture_size)
    face_idx = np.full((height, width), -1, dtype=np.int64)
    bary = np.zeros((height, width, 3), dtype=np.float32)
    for fi in range(faces.shape[0]):
        uv = uvs[faces[fi]].copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        if not np.isfinite(uv).all():
            continue
        uv_px = uv * (width - 1)
        min_x = int(np.floor(uv_px[:, 0].min()))
        max_x = int(np.ceil(uv_px[:, 0].max()))
        min_y = int(np.floor(uv_px[:, 1].min()))
        max_y = int(np.ceil(uv_px[:, 1].max()))
        if max_x < 0 or max_y < 0 or min_x >= width or min_y >= height:
            continue
        min_x = max(min_x, 0)
        min_y = max(min_y, 0)
        max_x = min(max_x, width - 1)
        max_y = min(max_y, height - 1)
        if min_x > max_x or min_y > max_y:
            continue
        xs = np.arange(min_x, max_x + 1)
        ys = np.arange(min_y, max_y + 1)
        xx, yy = np.meshgrid(xs, ys)
        p = np.stack([xx + 0.5, yy + 0.5], axis=-1)
        a, b, c = uv_px
        v0 = b - a
        v1 = c - a
        denom = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(denom) < 1e-12:
            continue
        v2 = p - a
        w1 = (v2[..., 0] * v1[1] - v1[0] * v2[..., 1]) / denom
        w2 = (v0[0] * v2[..., 1] - v2[..., 0] * v0[1]) / denom
        w0 = 1.0 - w1 - w2
        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not mask.any():
            continue
        yy_i = yy[mask]
        xx_i = xx[mask]
        empty = face_idx[yy_i, xx_i] == -1
        if not empty.any():
            continue
        yy_i = yy_i[empty]
        xx_i = xx_i[empty]
        face_idx[yy_i, xx_i] = fi
        bary[yy_i, xx_i, 0] = w0[mask][empty]
        bary[yy_i, xx_i, 1] = w1[mask][empty]
        bary[yy_i, xx_i, 2] = w2[mask][empty]
    return face_idx, bary


def _query_neural_rgb(points, neural_rgb, neural_sdf, appear_embed):
    device = next(neural_sdf.parameters()).device
    pts = torch.from_numpy(points).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        sdfs, feats = neural_sdf(pts)
        grads, _ = neural_sdf.compute_gradients(pts, training=False, sdf=sdfs)
        normals = torch_F.normalize(grads, dim=-1)
        rays_unit = -normals
        if appear_embed is not None:
            app = torch.zeros((pts.shape[0], appear_embed.embedding_dim), device=device)
        else:
            app = None
        rgbs = neural_rgb.forward(pts, normals, rays_unit, feats, app=app).clamp(0.0, 1.0)
    return rgbs.cpu().numpy()


def _query_neural_rgb_torch(pts, neural_rgb, neural_sdf, appear_embed):
    with torch.no_grad():
        sdfs, feats = neural_sdf(pts)
        grads, _ = neural_sdf.compute_gradients(pts, training=False, sdf=sdfs)
        normals = torch_F.normalize(grads, dim=-1)
        rays_unit = -normals
        if appear_embed is not None:
            app = torch.zeros((pts.shape[0], appear_embed.embedding_dim), device=pts.device)
        else:
            app = None
        rgbs = neural_rgb.forward(pts, normals, rays_unit, feats, app=app).clamp(0.0, 1.0)
    return rgbs


def _get_nvdiffrast():
    try:
        import nvdiffrast.torch as dr  # type: ignore
    except Exception as exc:
        raise ImportError(
            "nvdiffrast is required for GPU rasterization. Install with: pip install nvdiffrast"
        ) from exc
    return dr


def bake_texture_neural(vertices, faces, uvs, neural_rgb, neural_sdf, appear_embed,
                        texture_size=2048, batch_size=65536):
    height = width = int(texture_size)
    face_idx, bary = _rasterize_uv(uvs, faces, texture_size)
    texture = np.zeros((height, width, 3), dtype=np.uint8)
    valid = face_idx >= 0
    if not valid.any():
        return texture
    ys, xs = np.where(valid)
    face_ids = face_idx[ys, xs]
    bary_vals = bary[ys, xs]
    total = face_ids.shape[0]
    iterator = range(0, total, batch_size)
    if total > batch_size:
        iterator = tqdm(iterator, desc="Neural bake (CPU)", leave=False)
    for start in iterator:
        end = min(start + batch_size, total)
        batch_faces = faces[face_ids[start:end]]
        tri = vertices[batch_faces]
        w = bary_vals[start:end]
        points = (tri * w[:, :, None]).sum(axis=1)
        rgbs = _query_neural_rgb(points, neural_rgb, neural_sdf, appear_embed)
        colors = (rgbs * 255.0).astype(np.uint8)
        texture[ys[start:end], xs[start:end]] = colors
    return texture


def bake_texture_neural_gpu(vertices, faces, uvs, neural_rgb, neural_sdf, appear_embed,
                            texture_size=2048, batch_size=65536):
    dr = _get_nvdiffrast()
    device = next(neural_sdf.parameters()).device
    height = width = int(texture_size)
    uv = torch.from_numpy(uvs).to(device=device, dtype=torch.float32).clone()
    uv[:, 1] = 1.0 - uv[:, 1]
    pos = torch.zeros((1, uv.shape[0], 4), device=device, dtype=torch.float32)
    pos[0, :, 0:2] = uv * 2.0 - 1.0
    pos[0, :, 2] = 0.0
    pos[0, :, 3] = 1.0
    tri = torch.from_numpy(faces).to(device=device, dtype=torch.int32)
    print(f"GPU rasterizing UVs at {height}x{width}...")
    ctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(ctx, pos, tri, resolution=[height, width])
    tri_id = rast[..., 3]
    if tri_id.min().item() < 0:
        mask = tri_id >= 0
    else:
        mask = tri_id > 0
    if not mask.any():
        return np.zeros((height, width, 3), dtype=np.uint8)
    verts = torch.from_numpy(vertices).to(device=device, dtype=torch.float32).unsqueeze(0)
    attr, _ = dr.interpolate(verts, rast, tri)
    mask_2d = mask[0]
    idx = torch.nonzero(mask_2d, as_tuple=False)
    points = attr[0][mask_2d]
    total = points.shape[0]
    print(f"Valid texels: {total}")
    texture = torch.zeros((height, width, 3), device=device, dtype=torch.float32)
    iterator = range(0, total, batch_size)
    if total > batch_size:
        iterator = tqdm(iterator, desc="Neural bake (GPU)", leave=False)
    for start in iterator:
        end = min(start + batch_size, total)
        colors = _query_neural_rgb_torch(points[start:end], neural_rgb, neural_sdf, appear_embed)
        coord = idx[start:end]
        texture[coord[:, 0], coord[:, 1]] = colors
    texture = (texture.clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
    return texture


def _simplify_mesh_for_uv(mesh, target_faces, max_faces):
    face_count = len(mesh.faces)
    target = None
    if target_faces is not None and target_faces > 0:
        target = target_faces
    elif max_faces is not None and max_faces > 0 and face_count > max_faces:
        target = max_faces
    if target is None or face_count <= target:
        return mesh
    simplify = getattr(mesh, "simplify_quadratic_decimation", None)
    if simplify is None:
        simplified = _simplify_with_open3d(mesh, target)
        if simplified is not None:
            return simplified
        print("Mesh simplification not available; skipping decimation.")
        return mesh
    try:
        print(f"Decimating mesh from {face_count} to {target} faces for UV bake.")
        simplified = simplify(int(target))
    except Exception as exc:
        print(f"Mesh simplification failed: {exc}")
        simplified = _simplify_with_open3d(mesh, target)
        if simplified is None:
            return mesh
    if simplified is None or len(simplified.faces) == 0:
        print("Mesh simplification produced empty mesh; keeping original.")
        return mesh
    print(f"Simplified mesh faces: {len(simplified.faces)}")
    return simplified


def _simplify_with_open3d(mesh, target_faces):
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:
        print(f"Open3D not available for decimation: {exc}")
        return None
    if target_faces <= 0:
        return None
    try:
        tri = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices),
            o3d.utility.Vector3iVector(mesh.faces),
        )
        tri.remove_degenerate_triangles()
        tri.remove_duplicated_triangles()
        tri.remove_duplicated_vertices()
        tri.remove_non_manifold_edges()
        print(f"Decimating mesh with Open3D to {target_faces} faces.")
        simplified = tri.simplify_quadric_decimation(int(target_faces))
        simplified.remove_degenerate_triangles()
        simplified.remove_duplicated_triangles()
        simplified.remove_duplicated_vertices()
        simplified.remove_non_manifold_edges()
        vertices = np.asarray(simplified.vertices)
        faces = np.asarray(simplified.triangles)
        if faces.size == 0 or vertices.size == 0:
            return None
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    except Exception as exc:
        print(f"Open3D decimation failed: {exc}")
        return None


def _estimate_remesh_target_len(mesh, target_faces=None):
    if target_faces is not None and target_faces > 0:
        area = float(getattr(mesh, "area", 0.0))
        if area > 0.0:
            return float(np.sqrt(4.0 * area / (np.sqrt(3.0) * float(target_faces))))
    lengths = getattr(mesh, "edges_unique_length", None)
    if lengths is not None and len(lengths) > 0:
        return float(np.mean(lengths))
    return 0.0


def _isotropic_remesh_pymeshlab(mesh, target_len, iterations=10):
    try:
        import pymeshlab as ml  # type: ignore
    except Exception as exc:
        print(f"pymeshlab not available for isotropic remesh: {exc}")
        return mesh
    if target_len <= 0:
        print("Invalid remesh target length; skipping isotropic remesh.")
        return mesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if vertices.size == 0 or faces.size == 0:
        return mesh
    try:
        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(vertices, faces), "mesh")
        ms.meshing_isotropic_explicit_remeshing(
            targetlen=float(target_len),
            iterations=int(iterations),
            adaptive=True,
            features=True,
        )
        remeshed = ms.current_mesh()
        vertices = remeshed.vertex_matrix()
        faces = remeshed.face_matrix()
    except Exception as exc:
        print(f"Isotropic remesh failed: {exc}")
        return mesh
    if vertices.size == 0 or faces.size == 0:
        print("Isotropic remesh produced empty mesh; keeping original.")
        return mesh
    print(f"Isotropic remesh done. Faces: {len(faces)}")
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def prepare_mesh_for_uv(mesh, target_faces=None, max_faces=None, keep_lcc=True,
                        remesh=False, remesh_target_len=0.0, remesh_iters=10,
                        remesh_position="after"):
    face_count = len(mesh.faces)
    if max_faces is not None and max_faces > 0 and face_count > max_faces:
        print(f"Pre-simplifying mesh to {max_faces} faces.")
        mesh = _simplify_mesh_for_uv(mesh, target_faces=max_faces, max_faces=None)
    if keep_lcc:
        print("Keeping largest connected component after pre-simplification.")
        mesh = filter_largest_cc(mesh)
    if remesh and remesh_position == "before":
        target_len = remesh_target_len
        if target_len <= 0:
            target_len = _estimate_remesh_target_len(mesh, target_faces=target_faces)
        print(f"Isotropic remesh before final simplify. target_len={target_len:.6f}")
        mesh = _isotropic_remesh_pymeshlab(mesh, target_len=target_len, iterations=remesh_iters)
    if target_faces is not None and target_faces > 0 and len(mesh.faces) > target_faces:
        print(f"Final simplifying mesh to {target_faces} faces.")
        mesh = _simplify_mesh_for_uv(mesh, target_faces=target_faces, max_faces=None)
    if remesh and remesh_position == "after":
        target_len = remesh_target_len
        if target_len <= 0:
            target_len = _estimate_remesh_target_len(mesh, target_faces=target_faces)
        print(f"Isotropic remesh after final simplify. target_len={target_len:.6f}")
        mesh = _isotropic_remesh_pymeshlab(mesh, target_len=target_len, iterations=remesh_iters)
    return mesh


def bake_uv_texture(mesh, neural_rgb, neural_sdf, appear_embed, texture_size=2048, batch_size=65536,
                    target_faces=None, max_faces=None, keep_lcc=True, raster_mode="gpu", preprocess=True,
                    remesh=False, remesh_target_len=0.0, remesh_iters=10, remesh_position="after"):
    if preprocess:
        mesh = prepare_mesh_for_uv(
            mesh,
            target_faces=target_faces,
            max_faces=max_faces,
            keep_lcc=keep_lcc,
            remesh=remesh,
            remesh_target_len=remesh_target_len,
            remesh_iters=remesh_iters,
            remesh_position=remesh_position,
        )
    else:
        print("Skipping UV preprocess; using mesh as-is.")
    face_count = len(mesh.faces)
    print(f"UV bake input faces: {face_count}")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    print("Running xatlas unwrap...")
    vertices_uv, faces_uv, uvs = unwrap_uv_xatlas(vertices, faces)
    print(f"UV unwrap done. UV verts: {len(uvs)} faces: {len(faces_uv)}")
    if raster_mode == "gpu":
        texture = bake_texture_neural_gpu(
            vertices_uv, faces_uv, uvs, neural_rgb, neural_sdf, appear_embed,
            texture_size=texture_size, batch_size=batch_size
        )
    elif raster_mode == "cpu":
        texture = bake_texture_neural(
            vertices_uv, faces_uv, uvs, neural_rgb, neural_sdf, appear_embed,
            texture_size=texture_size, batch_size=batch_size
        )
    else:
        raise ValueError(f"Unknown raster_mode: {raster_mode}")
    mesh_uv = trimesh.Trimesh(vertices=vertices_uv, faces=faces_uv, process=False)
    if not isinstance(texture, Image.Image):
        texture = Image.fromarray(texture)
    mesh_uv.visual = TextureVisuals(uv=uvs, image=texture)
    return mesh_uv


class LatticeGrid(torch.utils.data.Dataset):

    def __init__(self, bounds, intv, block_res=64):
        super().__init__()
        self.block_res = block_res
        ((x_min, x_max), (y_min, y_max), (z_min, z_max)) = bounds
        self.x_grid = torch.arange(x_min, x_max, intv)
        self.y_grid = torch.arange(y_min, y_max, intv)
        self.z_grid = torch.arange(z_min, z_max, intv)
        res_x, res_y, res_z = len(self.x_grid), len(self.y_grid), len(self.z_grid)
        print("Extracting surface at resolution", res_x, res_y, res_z)
        self.num_blocks_x = int(np.ceil(res_x / block_res))
        self.num_blocks_y = int(np.ceil(res_y / block_res))
        self.num_blocks_z = int(np.ceil(res_z / block_res))

    def __getitem__(self, idx):
        # Keep track of sample index for convenience.
        sample = dict(idx=idx)
        block_idx_x = idx // (self.num_blocks_y * self.num_blocks_z)
        block_idx_y = (idx // self.num_blocks_z) % self.num_blocks_y
        block_idx_z = idx % self.num_blocks_z
        xi = block_idx_x * self.block_res
        yi = block_idx_y * self.block_res
        zi = block_idx_z * self.block_res
        x, y, z = torch.meshgrid(self.x_grid[xi:xi+self.block_res+1],
                                 self.y_grid[yi:yi+self.block_res+1],
                                 self.z_grid[zi:zi+self.block_res+1], indexing="ij")
        xyz = torch.stack([x, y, z], dim=-1)
        sample.update(xyz=xyz)
        return sample

    def __len__(self):
        return self.num_blocks_x * self.num_blocks_y * self.num_blocks_z


def get_lattice_grid_loader(dataset, num_workers=8):
    if dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        sampler = None
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False
    )


def marching_cubes(sdf, xyz, intv, texture_func, filter_lcc):
    # marching cubes
    V, F = mcubes.marching_cubes(sdf, 0.)
    if V.shape[0] > 0:
        V = V * intv + xyz[0, 0, 0]
        if texture_func is not None:
            C = texture_func(V)
            mesh = trimesh.Trimesh(V, F, vertex_colors=C)
        else:
            mesh = trimesh.Trimesh(V, F)
        mesh = filter_points_outside_bounding_sphere(mesh)
        mesh = filter_largest_cc(mesh) if filter_lcc else mesh
    else:
        mesh = trimesh.Trimesh()
    return mesh


def filter_points_outside_bounding_sphere(old_mesh):
    mask = np.linalg.norm(old_mesh.vertices, axis=-1) < 1.0
    if np.any(mask):
        indices = np.ones(len(old_mesh.vertices), dtype=int) * -1
        indices[mask] = np.arange(mask.sum())
        faces_mask = mask[old_mesh.faces[:, 0]] & mask[old_mesh.faces[:, 1]] & mask[old_mesh.faces[:, 2]]
        new_faces = indices[old_mesh.faces[faces_mask]]
        new_vertices = old_mesh.vertices[mask]
        new_colors = old_mesh.visual.vertex_colors[mask]
        new_mesh = trimesh.Trimesh(new_vertices, new_faces, vertex_colors=new_colors)
    else:
        new_mesh = trimesh.Trimesh()
    return new_mesh


def filter_largest_cc(mesh):
    components = mesh.split(only_watertight=False)
    areas = np.array([c.area for c in components], dtype=float)
    if len(areas) > 0 and mesh.vertices.shape[0] > 0:
        new_mesh = components[areas.argmax()]
    else:
        new_mesh = trimesh.Trimesh()
    return new_mesh
