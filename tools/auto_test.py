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

import os
import argparse

RESOLUTION=2048
BLOCK_RES=128

def gengerate_cmd(log_path,group_name,log_base_name,config_path,experiment):
    all_file_dirs = os.listdir(log_path)
    all_file_dirs = [f for f in all_file_dirs if f.startswith(log_base_name+'_'+group_name)]

    cmds = []
    for i,file_dir in all_file_dirs:
        ckpts = os.listdir(os.path.join(log_path,file_dir))
        ckpts = [f for f in ckpts if f.endswith('.pt')]
        ckpts = sorted(ckpts)
        last_ckpt = ckpts[-1]
        tmp_name = file_dir[len(log_base_name+'_'+group_name)+1:]
        if experiment == 'dtu':
            scanid = tmp_name[4:]
        else:
            raise NotImplementedError
        cmd = []
        
        CHECKPOINT={os.path.join(log_path,file_dir,last_ckpt)}
        OUTPUT_MESH=f'meshout/{tmp_name}_eval.ply'
        cmd1 = 'python projects/sdf_angelo/scripts/extract_mesh.py --single_gpu '
        cmd1 += f'--config={config_path} '
        cmd1 += '--resolution={RESOLUTION} '
        cmd1 += '--block_res={BLOCK_RES} '
        cmd1 += f'--checkpoint={CHECKPOINT} '
        cmd1 += f'--visible_only --output_file {OUTPUT_MESH} '
        cmd.append(cmd1)

        cmd2 = 'python projects/sdf_angelo/scripts/extract_mesh.py --single_gpu '
        cmd2 += f'--config={config_path} '
        cmd2 += f'--checkpoint={CHECKPOINT} '
        cmd2 += f'--output_file={OUTPUT_MESH} --input_mesh_space world '
        cmd2 += f'--input_mesh {OUTPUT_MESH} --depth_visible'
        cmd.append(cmd2)
    
        cmd3 = 'python projects/sdf_angelo/scripts/eval_DTU_mesh.py '
        cmd3 += f'--data {OUTPUT_MESH} --scan {scanid} --mode mesh '
        cmd3 += '--dataset_dir /home/haochen/data/DTU/Offical_DTU_Dataset --vis_out_dir meshout/ '
        cmd3 += '--apply_scale_mat --downsample_method voxel --downsample_density 0.4 '
        cmd3 += '--patch_size 60 --max_dist 20 --visualize_threshold 10 '
        cmd3 += f'--scale_mat_path /home/haochen/data/DTU/scan{scanid}/cameras.npz '
        cmd.append(cmd3)
        cmds.append(cmd)

    return cmds
    
def run_cmds(cmds):
    for cmd in cmds:
        for c in cmd:
            print(f'Running command: {c}')
            os.system(c)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('-l','--log_dir', type=str, default='logs/all_test', help='Path to the log directory.')
    parser.add_argument('-e','--experiment', type=str, default='dtu', help='Path to the log directory.')
    parser.add_argument('-g','--group', type=str, required=True, help='Base name for the log files.')
    parser.add_argument('--resolution', type=int, default=2048, help='Resolution for mesh extraction.')
    parser.add_argument('--block_res', type=int, default=128, help='Block resolution for mesh extraction.')
    args = parser.parse_args()

    cmds = gengerate_cmd(args.config, args.checkpoint, args.output_mesh, args.resolution, args.block_res)
    print(cmds)

    # run_cmds(cmds)

if __name__ == '__main__':
    main()