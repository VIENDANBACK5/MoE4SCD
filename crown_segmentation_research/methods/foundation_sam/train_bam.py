"""Train CrownTransformerSAM 2.0 on Exhaustive BAMFORESTS Benchmark (92k+ Tree Crowns).

100% Pure PyTorch on NVIDIA GPU (0% External Foundation Model Binaries).
Uses Ambiguity-Aware Multi-Mask Loss (min_i [20 * Focal + Dice]) + IoU Token Ranking MSE.
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

from crown_segmentation_research.datasets.bam_coco_dataset import BAMForestsDataset, collate_bam_batch
from crown_segmentation_research.methods.foundation_sam.model import CrownTransformerSAM

EXP_DIR = Path("DeadTrees/experiments/crown_transformer_bam")
EXP_DIR.mkdir(parents=True, exist_ok=True)


def sigmoid_focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Numerically robust Sigmoid Focal Loss matching Meta SAM formulation."""
    inputs = torch.clamp(inputs.float(), min=-15.0, max=15.0)
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    return loss.mean(dim=(-2, -1))


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice Loss with smooth Laplace-style epsilon for boundary stability."""
    inputs = torch.clamp(inputs.float(), min=-15.0, max=15.0)
    p = torch.sigmoid(inputs)
    targets = targets.float()
    numerator = 2.0 * (p * targets).sum(dim=(-2, -1)) + eps
    denominator = p.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1)) + eps
    return 1.0 - (numerator / denominator)


def calculate_iou_tensor(pred_logits: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
    """Computes binary IoU between predicted logits and target binary masks."""
    p_bin = (torch.sigmoid(pred_logits.float()) >= 0.50).float()
    target_masks = target_masks.float()
    intersection = (p_bin * target_masks).sum(dim=(-2, -1))
    union = p_bin.sum(dim=(-2, -1)) + target_masks.sum(dim=(-2, -1)) - intersection
    return (intersection + 1e-4) / (union + 1e-4)


def train_epoch(
    model: CrownTransformerSAM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_mask_loss = 0.0
    total_iou_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(dataloader):
        images = batch["images"].to(device, non_blocking=True)      # (B, 3, H, W)
        masks_list = batch["masks_list"]                             # List of (K_i, H, W)
        points_list = batch["points_list"]                           # List of (K_i, 2)
        labels_list = batch.get("labels_list", None)                 # List of (K_i,)
        B, _, H, W = images.shape

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Backbone visual feature extraction
            p4, _ = model.extract_features(images)

            batch_mask_loss = 0.0
            batch_iou_loss = 0.0
            valid_prompts_count = 0

            for b in range(B):
                gt_masks_b = masks_list[b].to(device, non_blocking=True)   # (K_i, H, W)
                pts_b = points_list[b].to(device, non_blocking=True)       # (K_i, 2)
                labels_b = labels_list[b].to(device, non_blocking=True) if labels_list is not None else None
                K_i = pts_b.shape[0]

                if K_i == 0:
                    continue

                # Forward Transformer Decoder
                pred_logits_h4, pred_ious = model.forward_decoder(p4[b : b + 1], pts_b, (H, W), point_labels=labels_b)
                pred_logits_h4 = pred_logits_h4.squeeze(0)  # (K_i, 3, H4, W4)
                pred_ious = pred_ious.squeeze(0)            # (K_i, 3)

                # Downsample target masks to feature map resolution (H4, W4)
                H4, W4 = pred_logits_h4.shape[-2:]
                gt_masks_h4 = F.interpolate(
                    gt_masks_b.unsqueeze(1), size=(H4, W4), mode="bilinear", align_corners=False
                ).squeeze(1)  # (K_i, H4, W4)

                # Expand GT masks for all 3 tokens: (K_i, 3, H4, W4)
                gt_expanded = gt_masks_h4.unsqueeze(1).expand(-1, 3, -1, -1)

                # Compute Focal + Dice loss for all 3 mask tokens
                focal = sigmoid_focal_loss(pred_logits_h4, gt_expanded)  # (K_i, 3)
                dice = dice_loss(pred_logits_h4, gt_expanded)            # (K_i, 3)
                token_losses = 20.0 * focal + dice                       # (K_i, 3)

                # Ambiguity-Aware Selection: min over the 3 tokens
                min_mask_loss, min_indices = torch.min(token_losses, dim=1)  # (K_i,)

                # True IoU for IoU ranking head supervision
                true_ious = calculate_iou_tensor(pred_logits_h4, gt_expanded)  # (K_i, 3)
                iou_mse = F.mse_loss(pred_ious.float(), true_ious.float().detach())

                batch_mask_loss += min_mask_loss.mean()
                batch_iou_loss += iou_mse
                valid_prompts_count += 1

            if valid_prompts_count > 0:
                loss = (batch_mask_loss / valid_prompts_count) + 0.5 * (batch_iou_loss / valid_prompts_count)
            else:
                continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mask_loss += (batch_mask_loss / valid_prompts_count).item()
        total_iou_loss += (batch_iou_loss / valid_prompts_count).item()
        num_batches += 1

        if (step + 1) % 25 == 0 or (step + 1) == len(dataloader):
            print(
                f"Epoch [{epoch:02d}] Step [{step+1:03d}/{len(dataloader):03d}] | "
                f"Loss: {loss.item():.4f} (Mask: {total_mask_loss/num_batches:.4f}, IoU: {total_iou_loss/num_batches:.4f})",
                flush=True,
            )

    return total_loss / num_batches, total_mask_loss / num_batches, total_iou_loss / num_batches


@torch.no_grad()
def evaluate_epoch(
    model: CrownTransformerSAM,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_iou = 0.0
    total_prompts = 0

    for batch in dataloader:
        images = batch["images"].to(device, non_blocking=True)
        masks_list = batch["masks_list"]
        points_list = batch["points_list"]
        labels_list = batch.get("labels_list", None)
        B, _, H, W = images.shape

        p4, _ = model.extract_features(images)

        for b in range(B):
            gt_masks_b = masks_list[b].to(device, non_blocking=True)
            pts_b = points_list[b].to(device, non_blocking=True)
            labels_b = labels_list[b].to(device, non_blocking=True) if labels_list is not None else None
            K_i = pts_b.shape[0]

            if K_i == 0:
                continue

            pred_logits_h4, pred_ious = model.forward_decoder(p4[b : b + 1], pts_b, (H, W), point_labels=labels_b)
            pred_logits_h4 = pred_logits_h4.squeeze(0)  # (K_i, 3, H4, W4)
            pred_ious = pred_ious.squeeze(0)            # (K_i, 3)

            # Pick best predicted mask according to IoU token
            best_token_idx = torch.argmax(pred_ious, dim=1)  # (K_i,)
            best_logits = pred_logits_h4[torch.arange(K_i, device=device), best_token_idx]  # (K_i, H4, W4)

            # Upscale to full image resolution
            best_logits_full = F.interpolate(
                best_logits.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(1)

            pred_bin = (torch.sigmoid(best_logits_full) >= 0.35).float()
            inter = (pred_bin * gt_masks_b).sum(dim=(-2, -1))
            union = pred_bin.sum(dim=(-2, -1)) + gt_masks_b.sum(dim=(-2, -1)) - inter
            iou_per_crown = (inter + 1e-6) / (union + 1e-6)

            total_iou += iou_per_crown.sum().item()
            total_prompts += K_i

    mean_iou = total_iou / max(total_prompts, 1)
    return mean_iou, total_prompts


def main():
    parser = argparse.ArgumentParser(description="Train CrownTransformerSAM on BAMFORESTS COCO Benchmark")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (images per batch)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--crop_size", type=int, default=1024, help="Training crop size")
    parser.add_argument("--prompts_per_img", type=int, default=16, help="Max tree prompts per tile")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    print("\nInitializing BAMFORESTS Datasets...", flush=True)
    train_ds = BAMForestsDataset(
        split="train", crop_size=args.crop_size, max_prompts_per_sample=args.prompts_per_img, augment=True
    )
    val_ds = BAMForestsDataset(
        split="eval", crop_size=args.crop_size, max_prompts_per_sample=args.prompts_per_img, augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_bam_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_bam_batch
    )

    print("\nInstantiating CrownTransformerSAM 2.0...", flush=True)
    model = CrownTransformerSAM().to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_iou = 0.0
    ckpt_path = EXP_DIR / "best_crown_transformer_bam.pth"

    print("=" * 75, flush=True)
    print(f"STARTING TRAINING ON BAMFORESTS ({len(train_ds)} train imgs, {len(val_ds)} val imgs)", flush=True)
    print("=" * 75, flush=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, mask_loss, iou_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()

        val_iou, n_eval_prompts = evaluate_epoch(model, val_loader, device)
        elapsed = time.time() - t0

        print(
            f"--> Epoch [{epoch:02d}/{args.epochs:02d}] Done ({elapsed:.1f}s) | "
            f"Train Loss: {loss:.4f} (Mask: {mask_loss:.4f}, IoU MSE: {iou_loss:.4f}) | "
            f"Val Mean IoU: {val_iou:.4f} ({n_eval_prompts} crowns evaluated)",
            flush=True,
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), ckpt_path)
            print(f"*** New Best Checkpoint Saved! Val IoU: {best_val_iou:.4f} -> {ckpt_path} ***\n", flush=True)
        else:
            print("", flush=True)

    print(f"Training Complete! Best Validation Mean IoU: {best_val_iou:.4f}", flush=True)
    print(f"Model saved to: {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
