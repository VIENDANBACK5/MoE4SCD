import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet
from crown_segmentation_research.methods.tree_flow.decode import decode_organic

device = "cuda" if torch.cuda.is_available() else "cpu"

data = np.load("data/itc_benchmarks/extracted/quebec_cached_npz/quebec_z3_patch_0015.npz")
img_rgb = data["image"]
label_map = data["instance_label"]

ckpt_path = "crown_segmentation_research/experiments/tree_flow_exhaustive_v1/best_model.pth"
model = TreeFlowNet().to(device)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt.get("model_state_dict", ckpt))
model.eval()

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
norm = ((img_rgb.astype(np.float32) / 255.0) - mean) / std
t_img = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

with torch.no_grad():
    preds = model(t_img)
    preds_sq = {k: v[0] for k, v in preds.items()}

# H20 Baseline
insts_h20 = decode_organic(
    preds_sq, img_rgb=img_rgb,
    min_sink_dist=44, canopy_thresh=0.30, sdt_thresh=-0.15, min_area=150,
    gradient_mode="sobel", refine_mode="none", device=str(device),
)

# Find the brown crown instance: check which instance has highest mean Red/Green ratio
r_ratios = []
for idx, inst in enumerate(insts_h20):
    m = inst["mask"]
    r = img_rgb[:, :, 0][m].mean()
    g = img_rgb[:, :, 1][m].mean()
    r_ratios.append((r / (g + 1e-6), idx))
r_ratios.sort(reverse=True)
best_ratio, target_idx = r_ratios[0]
target_inst = insts_h20[target_idx]
m_target = target_inst["mask"]
ys, xs = np.where(m_target)
print(f"Target brown tree: idx={target_idx}, R/G={best_ratio:.2f}, area={m_target.sum()}, Y=[{ys.min()}:{ys.max()}], X=[{xs.min()}:{xs.max()}]")
