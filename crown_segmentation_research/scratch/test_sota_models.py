import sys
import torch
import numpy as np
from pathlib import Path
from experiments.g1b_baselines.training.train_learned_baselines import build_maskrcnn_model, build_detectree2_model

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Mask R-CNN
m_path = "experiments/g1b_baselines/checkpoints/maskrcnn_seed42_best.pth"
print("Loading Mask R-CNN from", m_path)
maskrcnn = build_maskrcnn_model()
state = torch.load(m_path, map_location="cpu", weights_only=True)
maskrcnn.load_state_dict(state)
maskrcnn.to(device).eval()

# Load Detectree2
d_path = "experiments/g1b_baselines/checkpoints/detectree2_seed42_best.pth"
print("Loading Detectree2 from", d_path)
detectree = build_detectree2_model()
state_d = torch.load(d_path, map_location="cpu", weights_only=True)
detectree.load_state_dict(state_d)
detectree.to(device).eval()

# Load a test patch
data = np.load("data/itc_benchmarks/extracted/quebec_cached_npz/quebec_z3_patch_0001.npz")
img = data["image"]  # [512, 512, 3] uint8
t_img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
t_img = t_img.unsqueeze(0).to(device)

with torch.no_grad():
    out_m = maskrcnn(t_img)[0]
    out_d = detectree(t_img)[0]

print("Mask R-CNN predictions count:", len(out_m["boxes"]))
print("Detectree2 predictions count:", len(out_d["boxes"]))
scores_m = out_m["scores"].cpu().numpy()
scores_d = out_d["scores"].cpu().numpy()
print("Mask R-CNN scores > 0.3:", (scores_m > 0.3).sum())
print("Detectree2 scores > 0.3:", (scores_d > 0.3).sum())
print("SUCCESS!")
