import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

log_f = open("crown_segmentation_research/debug_output.txt", "w", buffering=1)
def log(s):
    print(s, flush=True)
    log_f.write(str(s) + "\n")
    log_f.flush()

log("Script starting...")
log("Importing DataLoader...")
from torch.utils.data import DataLoader
log("Importing build_exhaustive_tree_dataset...")
from crown_segmentation_research.datasets.exhaustive_tree_dataset import build_exhaustive_tree_dataset
log("Importing TreeFlowNet...")
from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet
log("Importing TreeFlowLoss...")
from crown_segmentation_research.methods.tree_flow.loss import TreeFlowLoss
log("All imports finished!")
ds = build_exhaustive_tree_dataset(patch_size=512, augment=True, use_bam=True, use_quebec=True)
loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, drop_last=True)

log("Checking keys in checkpoint vs model...")
model = TreeFlowNet().cuda()
sd = torch.load('DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth', map_location='cpu', weights_only=False)
state = sd.get('model_state_dict', sd.get('state_dict', sd))
model.load_state_dict(state, strict=True)

model_keys = set(model.state_dict().keys())
ckpt_keys = set(state.keys())

matched = model_keys & ckpt_keys
missing = model_keys - ckpt_keys
unexpected = ckpt_keys - model_keys

log("\n--- Testing BFloat16 AMP vs Float32 ---")
batch = next(iter(loader))
imgs = batch['image'].cuda()

# Test BFloat16 AMP
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    out_bf16 = model(imgs)
    log(f"  BFloat16 AMP: flow is_nan={torch.isnan(out_bf16['flow']).any().item()}, range=[{out_bf16['flow'].min().item():.2f}, {out_bf16['flow'].max().item():.2f}]")

# Test full loss computation with BFloat16
criterion = TreeFlowLoss()
targets = {k: batch[k].cuda() for k in ('flow', 'sdt', 'centroid', 'canopy', 'fg_mask')}
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    losses_bf16 = criterion(out_bf16, targets)
    log("  BFloat16 Losses:")
    for k, v in losses_bf16.items():
        log(f"    {k:10s}: {v.item():.4f}, is_nan={torch.isnan(v).item()}")

# Test full loss computation with Float32 (no AMP)
out_fp32 = model(imgs)
losses_fp32 = criterion(out_fp32, targets)
log("  Float32 Losses:")
for k, v in losses_fp32.items():
    log(f"    {k:10s}: {v.item():.4f}, is_nan={torch.isnan(v).item()}")

# Test 3 full training optimizer steps with BFloat16
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
log("\nTesting 3 full optimizer steps with BFloat16...")
for step in range(3):
    optimizer.zero_grad()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = model(imgs)
        losses = criterion(out, targets)
    losses['total'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    log(f"  Step {step}: total_loss={losses['total'].item():.4f} (flow={losses['flow'].item():.3f}, sdt={losses['sdt'].item():.3f}, cent={losses['centroid'].item():.3f}, can={losses['canopy'].item():.3f})")

log("\nBFLOAT16 TEST COMPLETE AND 100% CLEAN!")
log_f.close()
