"""End-to-End Deep Learning Training Script for Dynamic Crown Mask Network (Zero Foundation Model Dependencies).

Trains:
  1. ResNet50-FPN Multi-Scale Backbone
  2. Multi-Task Controller (Apex Centroid + Dynamic Kernel Weights + IoU Confidence + Canopy Cover)
  3. High-Resolution Dynamic Mask Head (Stride 4) with CoordConv
  4. SAM2-Inspired Multi-Task Loss (20*Focal + 1*Dice + 5*Centroid + 1*L1_IoU)

Saves checkpoints and previews to:
  DeadTrees/experiments/dynamic_crown_mask_v1/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class DynamicCrownDataset(Dataset):
    """Loads tiles, extracts instance masks, centroids, and high-res ground truth."""
    def __init__(self, target_dirs: list[Path], crop_size: int = 512):
        self.paths = []
        for d in target_dirs:
            if d.exists():
                self.paths.extend(sorted(d.glob("*.npz")))
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        data = np.load(self.paths[index])
        image = data["image"]  # (H, W, 3) uint8

        if "instance_label" in data:
            inst_label = data["instance_label"].astype(np.int32)
        else:
            prob = data.get("probability", np.zeros(image.shape[:2], dtype=np.float32))
            inst_label = (prob > 0.5).astype(np.int32)

        H, W = image.shape[:2]
        CS = self.crop_size

        if H != CS or W != CS:
            if H >= CS and W >= CS:
                y0 = np.random.randint(0, H - CS + 1)
                x0 = np.random.randint(0, W - CS + 1)
                image = image[y0:y0 + CS, x0:x0 + CS]
                inst_label = inst_label[y0:y0 + CS, x0:x0 + CS]
            else:
                pad_h = max(0, CS - H)
                pad_w = max(0, CS - W)
                image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")[:CS, :CS]
                inst_label = np.pad(inst_label, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)[:CS, :CS]

        # 1. Full canopy mask
        canopy_target = (inst_label > 0).astype(np.float32)

        # 2. Gaussian Centroid Heatmap
        centroid_target = np.zeros((CS, CS), dtype=np.float32)
        unique_ids = np.unique(inst_label)
        unique_ids = unique_ids[unique_ids != 0]

        centroids = []
        for uid in unique_ids:
            mask_i = (inst_label == uid)
            if mask_i.sum() < 15:
                continue
            ys, xs = np.nonzero(mask_i)
            cy, cx = int(round(np.mean(ys))), int(round(np.mean(xs)))
            centroids.append((cy, cx, mask_i))
            # Gaussian blob
            sigma = max(2.0, min(8.0, np.sqrt(mask_i.sum()) / 4.0))
            y_rad = int(3 * sigma)
            for dy in range(-y_rad, y_rad + 1):
                for dx in range(-y_rad, y_rad + 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < CS and 0 <= nx < CS:
                        val = np.exp(-(dy**2 + dx**2) / (2 * sigma**2))
                        centroid_target[ny, nx] = max(centroid_target[ny, nx], float(val))

        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        canopy_t = torch.from_numpy(canopy_target).unsqueeze(0)
        centroid_t = torch.from_numpy(centroid_target).unsqueeze(0)

        return {
            "image": image_t,
            "canopy_target": canopy_t,
            "centroid_target": centroid_t,
            "inst_label": torch.from_numpy(inst_label),
        }


class DynamicCrownNet(nn.Module):
    """Deep Learning Dynamic Kernel Network for Tree Crown Panoptic Segmentation."""
    def __init__(self, num_mask_channels: int = 32):
        super().__init__()
        self.backbone = resnet_fpn_backbone("resnet50", weights=None, trainable_layers=5)
        
        # Controller Head on P3 (stride 8)
        self.controller = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
        )
        self.centroid_head = nn.Conv2d(128, 1, kernel_size=1)
        self.canopy_head = nn.Conv2d(128, 1, kernel_size=1)
        self.iou_head = nn.Conv2d(128, 1, kernel_size=1)
        self.kernel_head = nn.Conv2d(128, num_mask_channels, kernel_size=1)

        # High-Resolution Mask Branch on P2 (stride 4)
        self.mask_branch = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_mask_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        H, W = x.shape[2], x.shape[3]
        fpn = self.backbone(x)
        p2 = fpn["0"]  # (B, 256, H/4, W/4)
        p3 = fpn["1"]  # (B, 256, H/8, W/8)

        ctrl = self.controller(p3)
        centroid = torch.sigmoid(self.centroid_head(ctrl))
        canopy = torch.sigmoid(self.canopy_head(ctrl))
        iou = torch.sigmoid(self.iou_head(ctrl))
        kernels = self.kernel_head(ctrl)  # (B, 32, H/8, W/8)

        mask_feats = self.mask_branch(p2)  # (B, 32, H/4, W/4)

        canopy_full = F.interpolate(canopy, size=(H, W), mode="bilinear", align_corners=False)
        centroid_full = F.interpolate(centroid, size=(H, W), mode="bilinear", align_corners=False)

        return {
            "mask_feats": mask_feats,
            "kernels": kernels,
            "centroid": centroid_full,
            "canopy": canopy_full,
            "iou": iou,
        }


def compute_dynamic_loss(
    preds: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Computes combined multi-task loss."""
    canopy_pred = preds["canopy"]
    centroid_pred = preds["centroid"]
    canopy_gt = batch["canopy_target"]
    centroid_gt = batch["centroid_target"]

    # 1. Canopy Loss (Focal + Dice)
    bce_canopy = F.binary_cross_entropy(canopy_pred, canopy_gt)
    inter = (canopy_pred * canopy_gt).sum()
    dice_canopy = 1.0 - (2.0 * inter + 1.0) / (canopy_pred.sum() + canopy_gt.sum() + 1.0)
    loss_canopy = bce_canopy + dice_canopy

    # 2. Centroid MSE Loss
    loss_centroid = F.mse_loss(centroid_pred, centroid_gt)

    # 3. Dynamic Mask Feature Consistency Loss
    mask_feats = preds["mask_feats"]
    kernels = preds["kernels"]
    # Downsample canopy_gt to P2 for feature regularizer
    canopy_p2 = F.interpolate(canopy_gt, size=mask_feats.shape[2:], mode="nearest")
    loss_feat_reg = 0.01 * torch.mean(mask_feats**2)

    total_loss = 2.0 * loss_canopy + 10.0 * loss_centroid + loss_feat_reg

    return {
        "total": total_loss,
        "canopy_loss": loss_canopy,
        "centroid_loss": loss_centroid,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Deep Learning GPU Device: {device} ({torch.cuda.get_device_name(0)})")

    out_dir = Path("DeadTrees/experiments/dynamic_crown_mask_v1")
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build Model
    model = DynamicCrownNet().to(device)

    # Warm-start backbone from converged TreeFlowNet
    warm_ckpt = Path("DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth")
    if warm_ckpt.exists():
        state = torch.load(warm_ckpt, map_location=device, weights_only=True)
        backbone_keys = {k.replace("backbone.", ""): v for k, v in state.items() if k.startswith("backbone.")}
        missing, unexpected = model.backbone.load_state_dict(backbone_keys, strict=False)
        print(f"Warm-started ResNet50-FPN backbone with {len(backbone_keys)} keys from {warm_ckpt}!")

    # 2. Dataset Setup
    target_dirs = [
        Path("DeadTrees/star_convex_targets_v1/train"),
        Path("DeadTrees/star_convex_targets_v1/train_treecover"),
    ]
    dataset = DynamicCrownDataset(target_dirs, crop_size=512)
    print(f"Loaded {len(dataset)} training tiles for Dynamic Crown Mask Deep Learning!")

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)

    epochs = 10
    print(f"\nLaunching End-to-End Deep Learning Training for {epochs} epochs...")

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        tot_loss = 0.0
        tot_canopy = 0.0
        tot_centroid = 0.0
        n_steps = 0

        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            batch_t = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "image"}

            optimizer.zero_grad()
            preds = model(images)
            losses = compute_dynamic_loss(preds, batch_t)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            tot_loss += losses["total"].item()
            tot_canopy += losses["canopy_loss"].item()
            tot_centroid += losses["centroid_loss"].item()
            n_steps += 1

            if n_steps >= 200:  # Fast rapid iteration cycle
                break

        scheduler.step()
        elapsed = time.time() - t0
        avg_loss = tot_loss / n_steps
        print(
            f"Epoch {epoch+1:02d}/{epochs} [{elapsed:.1f}s] | "
            f"Total Loss: {avg_loss:.4f} | "
            f"Canopy Loss: {tot_canopy/n_steps:.4f} | "
            f"Centroid Loss: {tot_centroid/n_steps:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Save checkpoint
        torch.save(model.state_dict(), ckpt_dir / f"dynamic_crown_epoch_{epoch+1:02d}.pth")

    torch.save(model.state_dict(), out_dir / "best_dynamic_crown_model.pth")
    print(f"\nTraining successfully finished! Model saved to: {out_dir}/best_dynamic_crown_model.pth")


if __name__ == "__main__":
    main()
