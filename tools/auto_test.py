'''
原始指令：
计算CD?:
CHECKPOINT=logs/${GROUP}/${NAME}/epoch_04081_iteration_000200000_checkpoint.pt
OUTPUT_MESH=meshout/utlfast_eval.ply
RESOLUTION=2048
BLOCK_RES=128
# 第一次：CPU 可见性过滤（不走 nvdiffrast）
python projects/sdf_angelo/scripts/extract_mesh.py --single_gpu     --config=${CONFIG}     --checkpoint=${CHECKPOINT}      --resolution=${RESOLUTION}     --block_res=${BLOCK_RES} --visible_only --output_file meshout/mesh_vis.obj
# 第二次：对过滤后的 mesh 做 depth 可见性
python projects/sdf_angelo/scripts/extract_mesh.py --single_gpu     --config=${CONFIG}     --checkpoint=${CHECKPOINT}     --output_file=${OUTPUT_MESH} --input_mesh_space world  --input_mesh meshout/mesh_vis.obj --depth_visible

最终计算CD：
python projects/sdf_angelo/scripts/eval_DTU_mesh.py   --data ${OUTPUT_MESH}   --scan 24   --mode mesh   --dataset_dir /home/sunny/data/DTU/EVAL   --vis_out_dir meshout/   --downsample_method voxel   --downsample_density 0.4   --patch_size 60   --max_dist 20   --visualize_threshold 10 --apply_scale_mat --scale_mat_path ~/data/DTU/scan24/cameras.npz
'''

import argparse
import os
import shlex
from pathlib import Path

DEFAULT_RESOLUTION = 2048
DEFAULT_BLOCK_RES = 128
DEFAULT_UV_TEXTURE_SIZE = 4096
DEFAULT_UV_TARGET_FACES = 50000
DEFAULT_UV_WELD_TOL = 1e-4
EXTRACT_SCRIPT = "projects/sdf_angelo/scripts/extract_mesh.py"
EVAL_SCRIPT = "projects/sdf_angelo/scripts/eval_DTU_mesh.py"
UV_EVAL_SCRIPT = "projects/sdf_angelo/scripts/uv_mesh_psnr.py"

def _quote(value):
    return shlex.quote(str(value))


def _trim_group_prefix(name, group_name):
    if not name.startswith(group_name):
        return name
    return name[len(group_name):].lstrip("-_")


def generate_cmds(
    log_dir,
    group_name,
    config_path,
    experiment,
    resolution,
    block_res,
    dataset_dir,
    scale_mat_root,
    transforms_root,
    data_root,
    output_dir,
    gpu_id,
    align_cameras,
):
    log_path = Path(log_dir)
    if not log_path.is_dir():
        raise FileNotFoundError(f"log_dir not found: {log_path}")
    all_file_dirs = sorted(
        p for p in log_path.iterdir() if p.is_dir() and p.name.startswith(group_name)
    )

    cmds = []
    for file_dir in all_file_dirs:
        ckpts = sorted(file_dir.glob("*.pt"))
        if not ckpts:
            print(f"[warn] No checkpoints found in {file_dir}")
            continue
        last_ckpt = ckpts[-1]
        tmp_name = _trim_group_prefix(file_dir.name, group_name)
        if experiment == "dtu":
            if not tmp_name.startswith("scan"):
                raise ValueError(f"Unexpected DTU scan name: {tmp_name}")
            scanid = tmp_name[4:]
        else:
            raise NotImplementedError
        # if scanid!="110":
        #     continue
        output_dir_path = Path(output_dir)
        visible_mesh = output_dir_path / f"{tmp_name}_vis.obj"
        output_mesh = output_dir_path / f"{tmp_name}_eval.obj"
        uv_mesh = output_dir_path / f"{tmp_name}_eval_uv.obj"
        uv_debug_dir = output_dir_path / f"{tmp_name}_uv_debug"
        scale_mat_path = Path(scale_mat_root) / f"scan{scanid}/cameras.npz"
        transforms_json = Path(transforms_root) / f"scan{scanid}/transforms.json"

        gpu_prefix = f"CUDA_VISIBLE_DEVICES={gpu_id}" if gpu_id is not None else ""
        cmd1_parts = [
            gpu_prefix,
            "python",
            EXTRACT_SCRIPT,
            "--single_gpu",
            f"--config={_quote(config_path)}",
            f"--checkpoint={_quote(last_ckpt)}",
            f"--resolution={resolution}",
            f"--block_res={block_res}",
            "--visible_only",
            f"--output_file={_quote(visible_mesh)}",
        ]
        if data_root:
            cmd1_parts.append(f"--data.root={_quote(Path(data_root) / f'scan{scanid}')}")  # override config
        cmd2_parts = [
            gpu_prefix,
            "python",
            EXTRACT_SCRIPT,
            "--single_gpu",
            f"--config={_quote(config_path)}",
            f"--checkpoint={_quote(last_ckpt)}",
            f"--output_file={_quote(output_mesh)}",
            "--input_mesh_space=world",
            f"--input_mesh={_quote(visible_mesh)}",
            "--depth_visible",
            "--depth_percentile=95",
            "--depth_trim_low=1",
            "--depth_trim_high=99",
            "--depth_smooth_kernel=3",
            "--depth_margin_stat=median",
            "--depth_margin_ratio=0.01",
        ]
        if data_root:
            cmd2_parts.append(f"--data.root={_quote(Path(data_root) / f'scan{scanid}')}")  # override config
        cmd3_parts = [
            gpu_prefix,
            "python",
            EVAL_SCRIPT,
            f"--data={_quote(output_mesh)}",
            f"--scan={scanid}",
            "--mode=mesh",
            f"--dataset_dir={_quote(dataset_dir)}",
            f"--vis_out_dir={_quote(output_dir_path)}",
            "--downsample_method=voxel",
            "--downsample_density=0.2",
            "--patch_size=60",
            "--max_dist=20",
            "--visualize_threshold=10",
        ]
        if align_cameras:
            cmd3_parts.extend([
                "--align_cameras",
                f"--transforms_json={_quote(transforms_json)}",
                f"--dtu_camera_npz={_quote(scale_mat_path)}",
            ])
        else:
            cmd3_parts.extend([
                "--apply_scale_mat",
                f"--scale_mat_path={_quote(scale_mat_path)}",
            ])
        cmd4_parts = [
            gpu_prefix,
            "python",
            EXTRACT_SCRIPT,
            "--single_gpu",
            f"--config={_quote(config_path)}",
            f"--checkpoint={_quote(last_ckpt)}",
            f"--input_mesh={_quote(output_mesh)}",
            "--input_mesh_space=world",
            f"--output_file={_quote(uv_mesh)}",
            "--uv_textured",
            "--uv_raster=gpu",
            f"--texture_size={DEFAULT_UV_TEXTURE_SIZE}",
            f"--uv_target_faces={DEFAULT_UV_TARGET_FACES}",
            "--uv_remesh",
            "--uv_remesh_position=after",
            "--uv_no_lcc",
            f"--uv_weld_tol={DEFAULT_UV_WELD_TOL}",
        ]
        if data_root:
            cmd4_parts.append(f"--data.root={_quote(Path(data_root) / f'scan{scanid}')}")  # override config
        cmd5_parts = [
            gpu_prefix,
            "python",
            UV_EVAL_SCRIPT,
            "--single_gpu",
            f"--config={_quote(config_path)}",
            f"--checkpoint={_quote(last_ckpt)}",
            f"--mesh={_quote(uv_mesh)}",
            "--color_mode=uv",
            "--no_flip_y",
            "--split train",
            "--ignore_alpha",
            "--uv_project_to_surface",
            "--uv_project_iters 1",
            "--uv_project_step 1.0",
            "--uv_project_max_step 0.0",
            "--texture_size 2048",
            "--mesh_space=world",
            f"--texture_size={DEFAULT_UV_TEXTURE_SIZE}",
            "--uv_raster=gpu",
            f"--debug_dir={_quote(uv_debug_dir)}",
            "--debug_every=10",
            "--debug_max_views=20",
        ]

        if data_root:
            cmd5_parts.append(f"--data.root={_quote(Path(data_root) / f'scan{scanid}')}")  # override config
        cmds.append(
            [
                " ".join(part for part in cmd1_parts if part),
                " ".join(part for part in cmd2_parts if part),
                " ".join(part for part in cmd3_parts if part),
                " ".join(part for part in cmd4_parts if part),
                " ".join(part for part in cmd5_parts if part),
            ]
        )

    return cmds
    
def run_cmds(cmds,todo_id):
    for cmd in cmds:
        id=0
        for c in cmd:
            id += 1
            if id == todo_id or todo_id == 0:
                print(f'Running command of id: {id} {c}')
                os.system(c)


def print_cmds(cmds):
    for cmd in cmds:
        for c in cmd:
            print(c,end='\n\n')
        print('--------------------------------------------------------------------')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('-l','--log_dir', type=str, default='logs/all_test', help='Path to the log directory.')
    parser.add_argument('-e','--experiment', type=str, default='dtu', help='Path to the log directory.')
    parser.add_argument('-g','--group', type=str, required=True, help='Base name for the log files.')
    parser.add_argument('--resolution', type=int, default=DEFAULT_RESOLUTION, help='Resolution for mesh extraction.')
    parser.add_argument('--block_res', type=int, default=DEFAULT_BLOCK_RES, help='Block resolution for mesh extraction.')
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='/home/haochen/data/DTU/Offical_DTU_Dataset',
        help='Path to the DTU dataset directory.',
    )
    parser.add_argument(
        '--scale_mat_root',
        type=str,
        default='/home/haochen/data/DTU',
        help='Root directory that contains scan{ID}/cameras.npz.',
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='/home/haochen/data/DTU',
        help='Root directory that contains scan{ID} folders for data.root override.',
    )
    parser.add_argument(
        '--transforms_root',
        type=str,
        default='',
        help='Root directory that contains scan{ID}/transforms.json. Defaults to scale_mat_root.',
    )
    parser.add_argument(
        '--align_cameras',
        action='store_true',
        help='Align COLMAP mesh to DTU world using camera centers.',
    )
    parser.add_argument('--output_dir', type=str, default='meshout', help='Output directory for meshes.')
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU id for CUDA_VISIBLE_DEVICES (e.g., 1). Omit to use default CUDA behavior.',
    )
    parser.add_argument('--run', action='store_true', help='Actually execute the generated commands.')
    parser.add_argument('--run_the_cmd_id', type=int, default=0, help='Actually execute the generated commands of id (1-4).')
    args = parser.parse_args()

    transforms_root = args.transforms_root or args.scale_mat_root
    cmds = generate_cmds(
        args.log_dir,
        args.group,
        args.config,
        args.experiment,
        args.resolution,
        args.block_res,
        args.dataset_dir,
        args.scale_mat_root,
        transforms_root,
        args.data_root,
        args.output_dir,
        args.gpu,
        args.align_cameras,
    )
    if args.run:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        run_cmds(cmds,args.run_the_cmd_id)
    else:
        print_cmds(cmds)

if __name__ == '__main__':
    main()

# python tools/auto_test.py --gpu 1 -l logs/all_test_DTUv2_high -g sdf_high_test2 --config projects/sdf_angelo/configs/dtu-win-high.yaml  --run
# python tools/auto_test.py --gpu 1 -l logs/base_neuralangelo -g base_neuralangelo --config projects/neuralangelo/configs/dtu.yaml  --run