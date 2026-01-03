import os
import shlex
import time
from concurrent.futures import ThreadPoolExecutor

import GPUtil
from mtools import logger_system as logger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PY = os.path.join(PROJECT_ROOT, "train.py")

# User config.
DATA_ROOT = "/home/haochen/data/mip360"
EXPERIMENT = "mip"

DATA_ROOT = "/home/haochen/data/nerf_DATA/DTU"
EXPERIMENT = "dtu"

CONFIG = os.path.join(
    PROJECT_ROOT, f"projects/sdf_angelo/configs/{EXPERIMENT}.yaml"
)
GROUP = "all_test"
NAME_PREFIX = "sdf_train_speed2"
SHOW_PBAR = True
USE_WANDB = True
WANDB_NAME = "test"
RESUME = True
SINGLE_GPU = True

SCENES = []  # e.g. ["scan36"] or ["garden"]; empty means auto-detect
EXCLUDE_SCENES = {"Points"}

EXCLUDED_GPUS = {6,7}  # GPUs to exclude from use


def collect_scenes():
    if SCENES:
        return [(scene, os.path.join(DATA_ROOT, scene)) for scene in SCENES]

    root_meta = os.path.join(DATA_ROOT, "transforms.json")
    if os.path.isfile(root_meta):
        scene_name = os.path.basename(DATA_ROOT.rstrip("/"))
        return [(scene_name, DATA_ROOT)]

    scenes = []
    for entry in sorted(os.listdir(DATA_ROOT)):
        if entry in EXCLUDE_SCENES:
            continue
        scene_root = os.path.join(DATA_ROOT, entry)
        if not os.path.isdir(scene_root):
            continue
        if not os.path.isfile(os.path.join(scene_root, "transforms.json")):
            continue
        scenes.append((entry, scene_root))
    return scenes


def build_cmd(gpu_id, scene_name, scene_root):
    name = f"{NAME_PREFIX}_{scene_name}" if NAME_PREFIX else scene_name
    logdir = (
        os.path.join(PROJECT_ROOT, "logs", GROUP, name)
        if GROUP
        else os.path.join(PROJECT_ROOT, "logs", name)
    )
    parts = [
        f"CUDA_VISIBLE_DEVICES={gpu_id}",
        "python",
        shlex.quote(TRAIN_PY),
        "--single_gpu" if SINGLE_GPU else "",
        f"--logdir={shlex.quote(logdir)}",
        f"--config={shlex.quote(CONFIG)}",
        "--show_pbar" if SHOW_PBAR else "",
        "--wandb" if USE_WANDB else "",
        f"--wandb_name={shlex.quote(WANDB_NAME)}" if USE_WANDB else "",
        "--resume" if RESUME else "",
        f"--data.root={shlex.quote(scene_root)}",
        f"--checkpoint.save_iter=100000"
    ]
    return " ".join(part for part in parts if part)


def run_one_scene(gpu_id, scene_name, scene_root):
    cmd = build_cmd(gpu_id, scene_name, scene_root)
    print(cmd)
    os.system(cmd)


def worker(gpu, scene):
    scene_name, scene_root = scene
    logger.info(f"Starting job on GPU {gpu} with scene: {scene_name}\n")
    start_time = time.time()
    run_one_scene(gpu, scene_name, scene_root)
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Finished job on GPU {gpu} with scene: {scene_name}\n")
    return duration


def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()
    time_records = 0

    while jobs or future_to_job:
        all_available_gpus = set(
            GPUtil.getAvailable(order="first", limit=10, excludeID=[], maxMemory=0.02)
        )
        available_gpus = list(all_available_gpus - reserved_gpus - EXCLUDED_GPUS)

        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, job)
            future_to_job[future] = (gpu, job)
            reserved_gpus.add(gpu)

        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)
            gpu = job[0]
            duration = future.result()
            time_records += duration
            reserved_gpus.discard(gpu)
            logger.info(
                f"Job {job} has finished, duration: {duration:.2f} seconds, "
                f"rellasing GPU {gpu}"
            )
        time.sleep(len_scenes // 2)

    logger.info("All jobs have been processed.")


scenes = collect_scenes()
len_scenes = len(scenes)
logger.info(f"Total {len_scenes} scenes")

jobs = list(scenes)

with ThreadPoolExecutor(max_workers=8) as executor:
    logger.info("Starting dispatche, jobs: %s", jobs)
    dispatch_jobs(jobs, executor)
