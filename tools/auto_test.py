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
EXTRACT_SCRIPT = "projects/sdf_angelo/scripts/extract_mesh.py"
EVAL_SCRIPT = "projects/sdf_angelo/scripts/eval_DTU_mesh.py"

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
        output_mesh = output_dir_path / f"{tmp_name}_eval.ply"
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
        cmds.append(
            [
                " ".join(part for part in cmd1_parts if part),
                " ".join(part for part in cmd2_parts if part),
                " ".join(part for part in cmd3_parts if part),
            ]
        )

    return cmds
    
def run_cmds(cmds):
    for cmd in cmds:
        for c in cmd:
            print(f'Running command: {c}')
            os.system(c)


def print_cmds(cmds):
    for cmd in cmds:
        for c in cmd:
            print(c)


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
        run_cmds(cmds)
    else:
        print_cmds(cmds)

if __name__ == '__main__':
    main()
