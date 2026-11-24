import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch
from scipy import ndimage
from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TreeFlowNet().to(device)
ckpt = torch.load('crown_segmentation_research/experiments/tree_flow_exhaustive_v1/best_model.pth', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt.get('model_state_dict', ckpt))
model.eval()

d = np.load('data/itc_benchmarks/extracted/quebec_cached_npz/quebec_z3_patch_0012.npz')
img = d['image']
lbl = d['instance_label']
H, W = img.shape[:2]

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
norm = ((img.astype(np.float32) / 255.0) - mean) / std
t_img = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

with torch.no_grad():
    preds = model(t_img)
    preds_sq = {k: v[0] for k, v in preds.items()}

flow = preds_sq['flow'].to(device)
sdt = preds_sq['sdt'].to(device)
canopy = preds_sq['canopy'].to(device)
cent = preds_sq['centroid'].to(device)

sdt_np = sdt[0].cpu().numpy()
canopy_np = canopy[0].cpu().numpy()
cent_np = cent[0].cpu().numpy()

fg = (canopy_np > 0.30) & (sdt_np > -0.15)
mag = torch.sqrt((flow ** 2).sum(0, keepdim=True) + 1e-6)
flow_norm = flow / mag
fy, fx = torch.from_numpy(fg).nonzero(as_tuple=True)
pos = torch.stack([fy.float().to(device), fx.float().to(device)], dim=0)

for _ in range(30):
    py = pos[0].clamp(0, H - 1)
    px = pos[1].clamp(0, W - 1)
    grid_y = (py / (H - 1)) * 2 - 1
    grid_x = (px / (W - 1)) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1)[None, None, :, :]
    v = torch.nn.functional.grid_sample(flow_norm[None], grid, mode='bilinear', align_corners=True)[0, :, 0, :]
    pos = pos + 1.0 * v

final_y = pos[0].round().long().clamp(0, H - 1).cpu().numpy()
final_x = pos[1].round().long().clamp(0, W - 1).cpu().numpy()

sink_density = np.zeros((H, W), dtype=np.float32)
np.add.at(sink_density, (final_y, final_x), 1.0)
smoothed_sink = ndimage.gaussian_filter(sink_density, sigma=2.5)
apex_score = smoothed_sink * (1.0 + 2.0 * cent_np + np.clip(sdt_np, 0, 1))

peak_candidates = (apex_score == ndimage.maximum_filter(apex_score, size=9)) & (smoothed_sink >= 1.5)
py, px = np.where(peak_candidates)
scores = apex_score[py, px]
order = np.argsort(-scores)
py, px = py[order], px[order]

print(f"Total candidate peaks: {len(py)}")

# 1. Fixed min_sink_dist = 44
keep_fixed_y, keep_fixed_x = [], []
for y, x in zip(py, px):
    too_close = False
    for ky, kx in zip(keep_fixed_y, keep_fixed_x):
        if np.sqrt((y - ky)**2 + (x - kx)**2) < 44:
            too_close = True
            break
    if not too_close:
        keep_fixed_y.append(y)
        keep_fixed_x.append(x)

print(f"Fixed (44px) kept: {len(keep_fixed_y)} sinks")

# 2. Scale-Adaptive: estimate local radius from SDT and density
# At candidate (y, x), local expected crown radius:
# If sdt is high (0.7), crown is huge -> exclusion radius should be larger (45-55px)
# If sdt is moderate/low (0.1 - 0.3), crown is small/tight -> exclusion radius can be 18-28px
keep_adapt_y, keep_adapt_x = [], []
for y, x in zip(py, px):
    # Dynamic exclusion radius based on SDT and Centroid peak
    local_sdt = max(0.0, float(sdt_np[y, x]))
    local_cent = float(cent_np[y, x])
    
    # Scale formula: r_excl = 16 + 36 * (0.6 * local_sdt + 0.4 * local_cent)
    r_excl = 16.0 + 36.0 * (0.6 * local_sdt + 0.4 * min(1.0, 2.0 * local_cent))
    r_excl = np.clip(r_excl, 16.0, 52.0)
    
    too_close = False
    for ky, kx in zip(keep_adapt_y, keep_adapt_x):
        dist = np.sqrt((y - ky)**2 + (x - kx)**2)
        # Check against the average or min of their exclusion radii
        if dist < r_excl:
            too_close = True
            break
    if not too_close:
        keep_adapt_y.append(y)
        keep_adapt_x.append(x)

print(f"Scale-Adaptive kept: {len(keep_adapt_y)} sinks (GT trees = 20)")
