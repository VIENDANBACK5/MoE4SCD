import sys
import torch
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet
from crown_segmentation_research.methods.tree_flow.decode import decode_organic

device = "cuda" if torch.cuda.is_available() else "cpu"

data = np.load("data/itc_benchmarks/extracted/quebec_cached_npz/quebec_z3_patch_0015.npz")
img = data["image"]
label_map = data["instance_label"]

ckpt_path = "crown_segmentation_research/experiments/tree_flow_exhaustive_v1/best_model.pth"
model = TreeFlowNet().to(device)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt.get("model_state_dict", ckpt))
model.eval()

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
norm = ((img.astype(np.float32) / 255.0) - mean) / std
t_img = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

with torch.no_grad():
    preds = model(t_img)
    preds_sq = {k: v[0] for k, v in preds.items()}

# 1. H20 Baseline
i_base = decode_organic(preds_sq, img_rgb=img, gradient_mode="sobel", refine_mode="none", device=str(device))
print("1. H20 Baseline:", len(i_base), "instances")

# 2. H21 Di Zenzo
i_diz = decode_organic(preds_sq, img_rgb=img, gradient_mode="dizenzo", refine_mode="none", device=str(device))
print("2. H21 Di Zenzo:", len(i_diz), "instances")

# 3. H21 Guided Filter
i_guide = decode_organic(preds_sq, img_rgb=img, gradient_mode="sobel", refine_mode="guided_filter", device=str(device))
print("3. H21 Guided Filter:", len(i_guide), "instances")

# 4. H21 MorphSnakes
i_snake = decode_organic(preds_sq, img_rgb=img, gradient_mode="sobel", refine_mode="morph_snake", device=str(device))
print("4. H21 MorphSnakes:", len(i_snake), "instances")

# 5. H21 Combined
i_comb = decode_organic(preds_sq, img_rgb=img, gradient_mode="dizenzo", refine_mode="guided_filter", device=str(device))
print("5. H21 Combined (Di Zenzo + Guided Filter):", len(i_comb), "instances")

print("ALL REFINEMENT MODES EXECUTED SUCCESSFULLY!")
