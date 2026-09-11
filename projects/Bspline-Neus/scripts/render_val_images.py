"""Render validation images from a trained checkpoint."""
import sys, os
sys.path.insert(0, ".")
import torch
from importlib import import_module
from imaginaire.config import Config
import torchvision.utils as tvu

Dataset = import_module("projects.Bspline-Neus.data").Dataset
Model = import_module("projects.Bspline-Neus.model").Model

cfg_path = sys.argv[1] if len(sys.argv) > 1 else "projects/Bspline-Neus/configs/stage_search/D1_anchor_r1_500k.yaml"
ckpt_path = sys.argv[2] if len(sys.argv) > 2 else "logs/D1_anchor_r1_500k/epoch_20833_iteration_000500000_checkpoint.pt"
out_dir = sys.argv[3] if len(sys.argv) > 3 else "logs/D1_anchor_r1_500k/renders"

cfg = Config(cfg_path)
cfg.data.val.subset = None
ds = Dataset(cfg, is_inference=True)
model = Model(cfg.model, cfg.data).cuda()
model.progress = 1.0
ckpt = torch.load(ckpt_path, map_location=lambda s, l: s)
sd = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt["model"].items()}
model.load_state_dict(sd, strict=False)
model.eval()

os.makedirs(out_dir, exist_ok=True)
indices = [0, 12, 24, 36, 48]  # val + extra views
with torch.no_grad():
    for i in indices:
        data = ds[i]
        data = {k: (v[None].cuda() if torch.is_tensor(v) else v) for k, v in data.items()}
        out = model.inference(data)
        rgb = out["rgb_map"][0]
        gt = data["image"][0]
        err = (rgb - gt).abs()
        tvu.save_image(rgb, f"{out_dir}/render_{i:03d}.png")
        tvu.save_image(gt, f"{out_dir}/gt_{i:03d}.png")
        tvu.save_image(err, f"{out_dir}/err_{i:03d}.png")
        mse = (rgb - gt).pow(2).mean().item()
        psnr = -10 * torch.log10(torch.tensor(mse)).item()
        print(f"Image {i}: PSNR = {psnr:.2f}")
print("Done")
