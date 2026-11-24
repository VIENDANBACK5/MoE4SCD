import sys
sys.path = [p for p in sys.path if not p.startswith('/home/chung/.local')]
import torch
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from crown_segmentation_research.methods.tree_flow.model_hr import TreeFlowNetHR
from crown_segmentation_research.methods.tree_flow.loss_hr import TreeFlowLossHR
from crown_segmentation_research.methods.tree_flow.train_hr import QuebecHRDataset

print("Imports succeeded!", flush=True)

ds = QuebecHRDataset("train")
print(f"Dataset size: {len(ds)}", flush=True)
item = ds[0]
print("Item keys:", list(item.keys()), flush=True)

model = TreeFlowNetHR()
model.load_from_v3("crown_segmentation_research/experiments/tree_flow_v3/best_model_v3.pth")
crit = TreeFlowLossHR()

imgs = item["image"].unsqueeze(0)
targets = {k: item[k].unsqueeze(0) for k in ("flow", "sdt", "centroid", "canopy", "fg_mask", "dist_map")}

out = model(imgs)
print("Forward passed!", flush=True)
loss = crit(out, targets, dist_map=targets["dist_map"])
print("Loss computed:", {k: v.item() for k, v in loss.items()}, flush=True)
loss["total"].backward()
print("Backward passed!", flush=True)
