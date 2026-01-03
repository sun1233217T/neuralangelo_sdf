# adapted from https://github.com/jzhangbs/DTUeval-python
import os
import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import argparse

def _resolve_scale_mat_path(dataset_dir, scan, scale_mat_path):
    candidates = []
    if scale_mat_path:
        if os.path.isdir(scale_mat_path):
            candidates.extend([
                os.path.join(scale_mat_path, "cameras_sphere.npz"),
                os.path.join(scale_mat_path, f"scan{scan}", "cameras_sphere.npz"),
                os.path.join(scale_mat_path, f"scan{scan:03}", "cameras_sphere.npz"),
            ])
        else:
            candidates.append(scale_mat_path)
    else:
        candidates.extend([
            os.path.join(dataset_dir, "cameras_sphere.npz"),
            os.path.join(dataset_dir, f"scan{scan}", "cameras_sphere.npz"),
            os.path.join(dataset_dir, f"scan{scan:03}", "cameras_sphere.npz"),
        ])
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "cameras_sphere.npz not found. Provide --scale_mat_path or set --dataset_dir "
        "to the DTU root containing scan folders."
    )

def _load_scale_mat(scale_mat_path):
    scale_data = np.load(scale_mat_path)
    if "scale_mat_0" in scale_data:
        scale_mat = scale_data["scale_mat_0"]
    elif "scale_mat" in scale_data:
        scale_mat = scale_data["scale_mat"]
    else:
        scale_keys = sorted([key for key in scale_data.keys() if key.startswith("scale_mat")])
        if not scale_keys:
            raise KeyError(f"scale_mat not found in {scale_mat_path}")
        scale_mat = scale_data[scale_keys[0]]
    return scale_mat.reshape(4, 4)

def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1+1, :n2+1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1,2,0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:,:1] + v2 * k[:,1:] + tri_vert
    return q

def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)

if __name__ == '__main__':
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data_in.ply')
    parser.add_argument('--scan', type=int, default=1)
    parser.add_argument('--mode', type=str, default='mesh', choices=['mesh', 'pcd'])
    parser.add_argument('--dataset_dir', type=str, default='.')
    parser.add_argument('--vis_out_dir', type=str, default='.')
    parser.add_argument('--downsample_density', type=float, default=0.2)
    parser.add_argument('--downsample_method', type=str, default='radius',
                        choices=['radius', 'voxel'])
    parser.add_argument('--apply_scale_mat', action='store_true',
                        help='Apply DTU scale_mat (normalized -> world) before evaluation')
    parser.add_argument('--scale_mat_path', type=str, default='',
                        help='Path to cameras_sphere.npz or DTU root/scan folder')
    parser.add_argument('--patch_size', type=float, default=60)
    parser.add_argument('--max_dist', type=float, default=20)
    parser.add_argument('--visualize_threshold', type=float, default=10)
    args = parser.parse_args()

    thresh = args.downsample_density
    if args.mode == 'mesh':
        pbar = tqdm(total=9)
        pbar.set_description('read data mesh')
        data_mesh = o3d.io.read_triangle_mesh(args.data)

        vertices = np.asarray(data_mesh.vertices)
        triangles = np.asarray(data_mesh.triangles)
        tri_vert = vertices[triangles]

        pbar.update(1)
        pbar.set_description('sample pcd from mesh')
        v1 = tri_vert[:,1] - tri_vert[:,0]
        v2 = tri_vert[:,2] - tri_vert[:,0]
        l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
        l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
        area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
        non_zero_area = (area2 > 0)[:,0]
        l1, l2, area2, v1, v2, tri_vert = [
            arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
        ]
        thr = thresh * np.sqrt(l1 * l2 / area2)
        n1 = np.floor(l1 / thr)
        n2 = np.floor(l2 / thr)

        with mp.Pool() as mp_pool:
            new_pts = mp_pool.map(sample_single_tri, ((n1[i,0], n2[i,0], v1[i:i+1], v2[i:i+1], tri_vert[i:i+1,0]) for i in range(len(n1))), chunksize=1024)

        new_pts = np.concatenate(new_pts, axis=0)
        data_pcd = np.concatenate([vertices, new_pts], axis=0)
    
    elif args.mode == 'pcd':
        pbar = tqdm(total=8)
        pbar.set_description('read data pcd')
        data_pcd_o3d = o3d.io.read_point_cloud(args.data)
        data_pcd = np.asarray(data_pcd_o3d.points)

    if args.apply_scale_mat:
        scale_mat_path = _resolve_scale_mat_path(args.dataset_dir, args.scan, args.scale_mat_path)
        scale_mat = _load_scale_mat(scale_mat_path)
        data_pcd = data_pcd @ scale_mat[:3, :3].T + scale_mat[:3, 3]
        print(f'apply scale_mat from: {scale_mat_path}')

    pbar.update(1)
    pbar.set_description('random shuffle pcd index')
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    pbar.update(1)
    pbar.set_description('downsample pcd')
    if args.downsample_method == 'radius':
        nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
        nn_engine.fit(data_pcd)
        rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
        mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
        for curr, idxs in enumerate(rnn_idxs):
            if mask[curr]:
                mask[idxs] = 0
                mask[curr] = 1
        data_down = data_pcd[mask]
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data_pcd)
        data_down = np.asarray(pcd.voxel_down_sample(voxel_size=thresh).points)

    pbar.update(1)
    pbar.set_description('masking data pcd')
    obs_mask_file = loadmat(f'{args.dataset_dir}/ObsMask/ObsMask{args.scan}_10.mat')
    ObsMask, BB, Res = [obs_mask_file[attr] for attr in ['ObsMask', 'BB', 'Res']]
    BB = BB.astype(np.float32)

    patch = args.patch_size
    inbound = ((data_down >= BB[:1]-patch) & (data_down < BB[1:]+patch*2)).sum(axis=-1) ==3
    data_in = data_down[inbound]

    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(axis=-1) ==3
    data_grid_in = data_grid[grid_inbound]
    in_obs = ObsMask[data_grid_in[:,0], data_grid_in[:,1], data_grid_in[:,2]].astype(np.bool_)
    data_in_obs = data_in[grid_inbound][in_obs]

    pbar.update(1)
    pbar.set_description('read STL pcd')
    stl_pcd = o3d.io.read_point_cloud(f'{args.dataset_dir}/Points/stl/stl{args.scan:03}_total.ply')
    stl = np.asarray(stl_pcd.points)

    pbar.update(1)
    pbar.set_description('compute data2stl')
    max_dist = args.max_dist
    if data_in_obs.shape[0] == 0 or stl.shape[0] == 0:
        dist_d2s = np.empty((0, 1), dtype=np.float32)
        mean_d2s = np.nan
        print(f'warning: skip data2stl (data_in_obs={data_in_obs.shape[0]}, stl={stl.shape[0]})')
    else:
        nn_engine = skln.NearestNeighbors(n_neighbors=1, algorithm='kd_tree', n_jobs=-1)
        nn_engine.fit(stl)
        dist_d2s, idx_d2s = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)
        mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    pbar.update(1)
    pbar.set_description('compute stl2data')
    ground_plane = loadmat(f'{args.dataset_dir}/ObsMask/Plane{args.scan}.mat')['P']

    stl_hom = np.concatenate([stl, np.ones_like(stl[:,:1])], -1)
    above = (ground_plane.reshape((1,4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]

    if data_in.shape[0] == 0 or stl_above.shape[0] == 0:
        dist_s2d = np.empty((0, 1), dtype=np.float32)
        mean_s2d = np.nan
        print(f'warning: skip stl2data (data_in={data_in.shape[0]}, stl_above={stl_above.shape[0]})')
    else:
        nn_engine = skln.NearestNeighbors(n_neighbors=1, algorithm='kd_tree', n_jobs=-1)
        nn_engine.fit(data_in)
        dist_s2d, idx_s2d = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
        mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    pbar.update(1)
    pbar.set_description('visualize error')
    vis_dist = args.visualize_threshold
    R = np.array([[1,0,0]], dtype=np.float64)
    G = np.array([[0,1,0]], dtype=np.float64)
    B = np.array([[0,0,1]], dtype=np.float64)
    W = np.array([[1,1,1]], dtype=np.float64)
    data_color = np.tile(B, (data_down.shape[0], 1))
    if dist_d2s.size:
        data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
        data_color[ np.where(inbound)[0][grid_inbound][in_obs] ] = R * data_alpha + W * (1-data_alpha)
        data_color[ np.where(inbound)[0][grid_inbound][in_obs][dist_d2s[:,0] >= max_dist] ] = G
    write_vis_pcd(f'{args.vis_out_dir}/vis_{args.scan:03}_d2s.ply', data_down, data_color)
    stl_color = np.tile(B, (stl.shape[0], 1))
    if dist_s2d.size:
        stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
        stl_color[ np.where(above)[0] ] = R * stl_alpha + W * (1-stl_alpha)
        stl_color[ np.where(above)[0][dist_s2d[:,0] >= max_dist] ] = G
    write_vis_pcd(f'{args.vis_out_dir}/vis_{args.scan:03}_s2d.ply', stl, stl_color)

    pbar.update(1)
    pbar.set_description('done')
    pbar.close()
    over_all = (mean_d2s + mean_s2d) / 2
    print(mean_d2s, mean_s2d, over_all)
    
    import json
    with open(f'{args.vis_out_dir}/results.json', 'w') as fp:
        json.dump({
            'mean_d2s': mean_d2s,
            'mean_s2d': mean_s2d,
            'overall': over_all,
        }, fp, indent=True)
