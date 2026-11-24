import sys
sys.path.insert(0, ".")
import json
import zipfile
import io
import torch
import numpy as np
from PIL import Image
from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet
from crown_segmentation_research.methods.tree_flow.decode import decode

ckpt_path = "DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth"
model = TreeFlowNet().cuda().eval()
sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(sd.get("model_state_dict", sd.get("state_dict", sd)))

with open("data/itc_benchmarks/extracted/bam_coco2048/zip_manifest.json") as f:
    manifest = json.load(f)

with open("data/itc_benchmarks/extracted/bam_coco2048/instances_tree_eval2023.json") as f:
    coco = json.load(f)

img_info = coco["images"][0]
print("Testing image:", img_info["file_name"])

with zipfile.ZipFile("data/itc_benchmarks/raw_archives/Bamberg_coco2048.zip", "r") as zf:
    data = zf.read(manifest[img_info["file_name"]])
    img_pil = Image.open(io.BytesIO(data)).convert("RGB").resize((1024, 1024), Image.BILINEAR)
    img_np = np.array(img_pil)

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
img_t = (torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0 - mean) / std
img_t = img_t.unsqueeze(0).cuda()

with torch.no_grad():
    out = model(img_t)
    for k, v in out.items():
        v0 = v[0].cpu().numpy()
        print(f"  {k:10s}: min={v0.min():.3f}, mean={v0.mean():.3f}, max={v0.max():.3f}")

# Check decode with various thresholds
for c_th in [0.1, 0.2, 0.3, 0.4]:
    for s_th in [-0.9, -0.7, -0.5]:
        insts = decode({k: v[0] for k, v in out.items()}, canopy_thresh=c_th, sdt_thresh=s_th, min_area=20, device="cuda")
        print(f"  c_th={c_th:.2f}, s_th={s_th:.2f} -> {len(insts)} instances decoded")
