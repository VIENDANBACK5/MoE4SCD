import sys
sys.path.insert(0, ".")
import torch
from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet

ckpt_path = "DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth"
sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state = sd.get("model_state_dict", sd.get("state_dict", sd))

model = TreeFlowNet()
missing, unexpected = model.load_state_dict(state, strict=True)
print("load_state_dict strict=True succeeded!")
print("Missing:", missing)
print("Unexpected:", unexpected)

model.eval()
x = torch.randn(1, 3, 1024, 1024)
with torch.no_grad():
    out = model(x)
    for k, v in out.items():
        print(f"  {k:10s}: shape={v.shape}, min={v.min():.3f}, max={v.max():.3f}")
print("TreeFlowNet forward pass 100% SUCCESS!")
