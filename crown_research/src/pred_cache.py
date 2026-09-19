"""Cache CanopyWatershedNet head outputs for the eval crops.

Seed-selection experiments re-decode the same network outputs hundreds of times.
Running the forward pass once and caching surface/boundary/canopy + the GT instance
map turns a sweep from GPU-minutes into CPU-seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

_conda_p = '/home/ubuntu/anaconda3/envs/cs2/lib/python3.11/site-packages'
_local_p = '/home/chung/.local/lib/python3.11/site-packages'
sys.path = [p for p in sys.path if p != _local_p and p != _conda_p]
sys.path.insert(0, _conda_p)
sys.path.append(_local_p)
sys.path.insert(0, "/home/chung/RS/Image Segmentation")

import numpy as np
import torch

CACHE_DIR = Path("crown_research/data/cache")
CKPT = "DeadTrees/experiments/canopy_watershed/best_sharp_watershed.pth"


def build_cache(num_crops: int = 30, split: str = "eval", crop_size: int = 1024,
                ckpt: str = CKPT, tag: str = "eval30") -> Path:
    from crown_segmentation_research.methods.canopy_watershed.dataset import BAMCanopyWatershedDataset
    from crown_segmentation_research.methods.canopy_watershed.model import CanopyWatershedNet

    out = CACHE_DIR / tag
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CanopyWatershedNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model_state_dict"])
    model.eval()

    ds = BAMCanopyWatershedDataset(split=split, crop_size=crop_size, augment=False, compute_targets=False)
    n = min(num_crops, len(ds))
    kept = 0
    for idx in range(n):
        f = out / f"crop_{idx:04d}.npz"
        if f.exists():
            kept += 1
            continue
        item = ds[idx]
        raw_gt = item["instance_label"].squeeze().numpy().astype(np.int32)
        gt_map = np.zeros_like(raw_gt)
        gt_map[raw_gt > 0] = raw_gt[raw_gt > 0] + 1
        if len(np.unique(raw_gt[raw_gt > 0])) == 0:
            continue
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            p = model(item["image"].unsqueeze(0).to(device))
        np.savez_compressed(
            f,
            surface=p["surface"].squeeze().float().cpu().numpy().astype(np.float16),
            boundary=p["boundary"].squeeze().float().cpu().numpy().astype(np.float16),
            canopy=p["canopy"].squeeze().float().cpu().numpy().astype(np.float16),
            gt_map=gt_map.astype(np.int16),
            rgb=(item["image"].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8),
        )
        kept += 1
        print(f"  cached {idx+1}/{n}", flush=True)
    print(f"[cache] {kept} crops at {out}")
    return out


def load_cache(tag: str = "eval30"):
    """Yields (surface, boundary, canopy, gt_map) as float32/int32 per crop."""
    out = CACHE_DIR / tag
    for f in sorted(out.glob("crop_*.npz")):
        z = np.load(f)
        yield (z["surface"].astype(np.float32), z["boundary"].astype(np.float32),
               z["canopy"].astype(np.float32), z["gt_map"].astype(np.int32))


def load_cache_rgb(tag: str = "eval30"):
    out = CACHE_DIR / tag
    for f in sorted(out.glob("crop_*.npz")):
        z = np.load(f)
        yield (z["surface"].astype(np.float32), z["boundary"].astype(np.float32),
               z["canopy"].astype(np.float32), z["gt_map"].astype(np.int32), z["rgb"])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_crops", type=int, default=30)
    ap.add_argument("--tag", type=str, default="eval30")
    ap.add_argument("--split", type=str, default="eval")
    build_cache(num_crops=ap.parse_args().num_crops, tag=ap.parse_args().tag, split=ap.parse_args().split)
