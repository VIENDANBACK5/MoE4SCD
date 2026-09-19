"""Training Pipeline for TopoMetricFlowNet on BAMFORESTS (92k+ Tree Crowns).

Trains a 1-stage end-to-end multi-task neural network with:
- Normalized Centripetal Flow Loss (Cosine Similarity)
- Saddle Barrier Energy Loss (Weighted BCE / Focal)
- Potential Surface Regression (MSE Loss)
- Canopy Support Gate Loss (BCE)

100% Native PyTorch, Zero External Foundation Model Dependencies.
"""
from __future__ import annotations


# Ensure workspace root is in sys.path
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "crown_segmentation_research").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from crown_segmentation_research.datasets.bam_topometric_dataset import BAMTopoMetricDataset, collate_topometric_batch
from crown_segmentation_research.methods.topometric_flow.model import TopoMetricFlowNet
from crown_segmentation_research.methods.topometric_flow.decode import decode_topometric_instances

EXP_DIR = Path("DeadTrees/experiments/topometric_flow")
EXP_DIR.mkdir(parents=True, exist_ok=True)


def compute_topometric_loss(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_flow = preds["flow"]
    pred_saddle = preds["saddle"].float()
    pred_surface = preds["surface"].float()
    pred_canopy = preds["canopy"].float()

    tgt_flow = targets["flows"]
    tgt_saddle = targets["saddles"].float()
    tgt_surface = targets["surfaces"].float()
    tgt_canopy = targets["canopies"].float()

    # 1. Flow Cosine Loss (masked inside canopy)
    canopy_fg = (tgt_canopy > 0.5).float()
    cos_sim = (pred_flow * tgt_flow).sum(dim=1, keepdim=True)  # (B, 1, H, W)
    flow_loss_map = (1.0 - cos_sim) * canopy_fg
    flow_loss = flow_loss_map.sum() / (canopy_fg.sum() + 1e-6)

    # 2. Saddle Barrier Loss (Weighted BCE to handle sparsity)
    saddle_weights = 1.0 + 3.0 * tgt_saddle
    saddle_loss = F.binary_cross_entropy(pred_saddle, tgt_saddle, weight=saddle_weights)

    # 3. Potential Surface Loss
    surface_loss = F.mse_loss(pred_surface, tgt_surface)

    # 4. Canopy Gate Loss
    canopy_loss = F.binary_cross_entropy(pred_canopy, tgt_canopy)

    total_loss = 2.0 * flow_loss + 3.0 * saddle_loss + 2.0 * surface_loss + 1.0 * canopy_loss

    loss_dict = {
        "flow": flow_loss.item(),
        "saddle": saddle_loss.item(),
        "surface": surface_loss.item(),
        "canopy": canopy_loss.item(),
        "total": total_loss.item(),
    }
    return total_loss, loss_dict


def train_epoch(
    model: TopoMetricFlowNet,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss = 0.0
    accum_losses = {"flow": 0.0, "saddle": 0.0, "surface": 0.0, "canopy": 0.0}
    num_batches = 0

    for step, batch in enumerate(dataloader):
        images = batch["images"].to(device, non_blocking=True)
        targets = {
            "flows": batch["flows"].to(device, non_blocking=True),
            "saddles": batch["saddles"].to(device, non_blocking=True),
            "surfaces": batch["surfaces"].to(device, non_blocking=True),
            "canopies": batch["canopies"].to(device, non_blocking=True),
        }

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            preds = model(images)

        loss, loss_dict = compute_topometric_loss(preds, targets)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k in accum_losses:
            accum_losses[k] += loss_dict[k]
        num_batches += 1

        if (step + 1) % 25 == 0 or (step + 1) == len(dataloader):
            print(
                f"Epoch [{epoch:02d}] Step [{step+1:03d}/{len(dataloader):03d}] | "
                f"Loss: {loss.item():.4f} (Flow: {loss_dict['flow']:.4f}, "
                f"Saddle: {loss_dict['saddle']:.4f}, Surface: {loss_dict['surface']:.4f})",
                flush=True,
            )

    avg_loss = total_loss / num_batches
    avg_dict = {k: v / num_batches for k, v in accum_losses.items()}
    return avg_loss, avg_dict


@torch.no_grad()
def evaluate_epoch(
    model: TopoMetricFlowNet,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0

    for batch in dataloader:
        images = batch["images"].to(device, non_blocking=True)
        targets = {
            "flows": batch["flows"].to(device, non_blocking=True),
            "saddles": batch["saddles"].to(device, non_blocking=True),
            "surfaces": batch["surfaces"].to(device, non_blocking=True),
            "canopies": batch["canopies"].to(device, non_blocking=True),
        }

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            preds = model(images)

        loss, _ = compute_topometric_loss(preds, targets)
        total_loss += loss.item()

        # GPU Tensor IoU for canopy support
        pred_canopy_bin = (preds["canopy"] > 0.40)
        tgt_canopy_bin = (targets["canopies"] > 0.5)
        inter = (pred_canopy_bin & tgt_canopy_bin).sum(dim=(-2, -1)).float()
        union = (pred_canopy_bin | tgt_canopy_bin).sum(dim=(-2, -1)).float()
        iou = (inter + 1e-6) / (union + 1e-6)
        total_iou += iou.mean().item()
        num_batches += 1

    val_loss = total_loss / max(num_batches, 1)
    val_iou = total_iou / max(num_batches, 1)
    return val_loss, val_iou


def main():
    parser = argparse.ArgumentParser(description="Train TopoMetricFlowNet on BAMFORESTS Benchmark")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--crop_size", type=int, default=1024, help="Crop size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    print("\nInitializing BAMFORESTS TopoMetric Datasets...", flush=True)
    train_ds = BAMTopoMetricDataset(split="train", crop_size=args.crop_size, augment=True)
    val_ds = BAMTopoMetricDataset(split="eval", crop_size=args.crop_size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_topometric_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_topometric_batch
    )

    print("\nInstantiating TopoMetricFlowNet...", flush=True)
    model = TopoMetricFlowNet(pretrained_backbone=True).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    best_val_iou = 0.0
    ckpt_path = EXP_DIR / "best_topometric_flow.pth"

    print("=" * 75, flush=True)
    print(f"STARTING TOPOMETRIC FLOW TRAINING ({len(train_ds)} train imgs, {len(val_ds)} val imgs)", flush=True)
    print("=" * 75, flush=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, loss_dict = train_epoch(model, train_loader, optimizer, scaler, device, epoch)
        scheduler.step()

        val_iou, n_preds = evaluate_epoch(model, val_loader, device)
        elapsed = time.time() - t0

        print(
            f"--> Epoch [{epoch:02d}/{args.epochs:02d}] Done ({elapsed:.1f}s) | "
            f"Loss: {loss:.4f} (Flow: {loss_dict['flow']:.4f}, Saddle: {loss_dict['saddle']:.4f}) | "
            f"Val Canopy IoU: {val_iou:.4f} ({n_preds} crowns detected)",
            flush=True,
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), ckpt_path)
            print(f"*** New Best TopoMetric Checkpoint Saved! Val IoU: {best_val_iou:.4f} -> {ckpt_path} ***\n", flush=True)
        else:
            print("", flush=True)

    print(f"Training Complete! Best Validation IoU: {best_val_iou:.4f}", flush=True)
    print(f"Model saved to: {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
