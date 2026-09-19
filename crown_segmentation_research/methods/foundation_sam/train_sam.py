"""Training Script for CrownTransformerSAM 2.0 (Pure Standalone PyTorch SAM & SAM 2 Architecture).

Implements:
  1. Multi-Scale Skip Connections (Stride 4 and Stride 8) from ResNet50-FPN
  2. 3-Mask Ambiguity-Aware Resolution with min_i (20 * Focal + 1 * Dice) Loss
  3. Supervised IoU Ranking Head across all 3 mask scales
  4. Unified Multi-Biome Panoptic Training across 4,387 DeadTrees Tiles (Living Canopies + Snags)

Saves checkpoints to:
  DeadTrees/experiments/crown_transformer_sam_v2/
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
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from crown_segmentation_research.methods.foundation_sam.model import CrownTransformerSAM

OUT_DIR = Path("DeadTrees/experiments/crown_transformer_sam_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class PanopticPromptableTreeDataset(Dataset):
    """Loads tiles and unifies living canopy + deadwood snags into complete instance masks."""
    def __init__(
        self,
        dead_dir: Path,
        treecover_dir: Path,
        crop_size: int = 512,
        num_prompts_per_image: int = 16,
    ):
        self.dead_dir = dead_dir
        self.treecover_dir = treecover_dir
        self.crop_size = crop_size
        self.num_prompts = num_prompts_per_image

        dead_files = {p.stem: p for p in dead_dir.glob("*.npz")} if dead_dir.exists() else {}
        tree_files = {p.stem: p for p in treecover_dir.glob("*.npz")} if treecover_dir.exists() else {}

        all_stems = sorted(set(dead_files.keys()) | set(tree_files.keys()))
        self.samples = []
        for s in all_stems:
            self.samples.append({
                "stem": s,
                "dead_path": dead_files.get(s, None),
                "tree_path": tree_files.get(s, None),
            })
        print(f"[Dataset] Indexed {len(self.samples)} unique tiles ({len(dead_files)} deadwood, {len(tree_files)} living canopy).")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        info = self.samples[index]
        image = None
        inst_label = None

        if info["dead_path"] is not None and info["tree_path"] is not None:
            d_dead = np.load(info["dead_path"])
            d_tree = np.load(info["tree_path"])
            image = d_dead["image"]
            l_dead = d_dead["instance_label"].astype(np.int32)
            l_tree = d_tree["instance_label"].astype(np.int32)

            max_tree = int(l_tree.max()) if l_tree.max() > 0 else 0
            inst_label = l_tree.copy()
            dead_mask = (l_dead > 0)
            inst_label[dead_mask] = l_dead[dead_mask] + max_tree
        elif info["dead_path"] is not None:
            d_dead = np.load(info["dead_path"])
            image = d_dead["image"]
            inst_label = d_dead["instance_label"].astype(np.int32)
        else:
            d_tree = np.load(info["tree_path"])
            image = d_tree["image"]
            inst_label = d_tree["instance_label"].astype(np.int32)

        H, W = image.shape[:2]
        CS = self.crop_size

        # Random cropping / padding for scale invariance
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

        unique_ids = np.unique(inst_label)
        unique_ids = unique_ids[unique_ids != 0]

        point_coords = []
        point_labels = []
        gt_masks = []

        # 1. Sample positive prompts from real tree crown instances
        if len(unique_ids) > 0:
            sampled_ids = np.random.choice(unique_ids, size=min(self.num_prompts - 4, len(unique_ids)), replace=False)
            for uid in sampled_ids:
                m = (inst_label == uid)
                ys, xs = np.nonzero(m)
                if len(ys) > 0:
                    if np.random.rand() > 0.5:
                        cy, cx = int(np.mean(ys)), int(np.mean(xs))
                        if m[cy, cx]:
                            point_coords.append([cy, cx])
                        else:
                            idx = np.random.randint(0, len(ys))
                            point_coords.append([ys[idx], xs[idx]])
                    else:
                        idx = np.random.randint(0, len(ys))
                        point_coords.append([ys[idx], xs[idx]])
                    point_labels.append(1)
                    gt_masks.append(m.astype(np.float32))

        # 2. Add negative background / ground clutter points
        while len(point_coords) < self.num_prompts:
            ry = np.random.randint(0, CS)
            rx = np.random.randint(0, CS)
            uid = inst_label[ry, rx]
            m = (inst_label == uid) if uid > 0 else np.zeros((CS, CS), dtype=np.float32)
            point_coords.append([ry, rx])
            point_labels.append(1 if uid > 0 else 0)
            gt_masks.append(m.astype(np.float32))

        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        coords_t = torch.tensor(point_coords, dtype=torch.float32)
        labels_t = torch.tensor(point_labels, dtype=torch.long)
        masks_t = torch.from_numpy(np.stack(gt_masks, axis=0))  # (K, CS, CS)

        return {
            "image": image_t,
            "point_coords": coords_t,
            "point_labels": labels_t,
            "gt_masks": masks_t,
        }


def sam2_ambiguity_loss(
    mask_logits: torch.Tensor,
    iou_preds: torch.Tensor,
    gt_masks: torch.Tensor,
    point_labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Computes Ambiguity-Aware Loss (min_i [20 * Focal + Dice]) + MSE IoU Loss.
    
    Args:
        mask_logits: (B, K, 3, H4, W4)
        iou_preds: (B, K, 3)
        gt_masks: (B, K, H, W)
        point_labels: (B, K)
    """
    B, K, num_masks, H4, W4 = mask_logits.shape
    gt_h4 = F.interpolate(gt_masks, size=(H4, W4), mode="nearest")  # (B, K, H4, W4)
    gt_h4_3 = gt_h4.unsqueeze(2).repeat(1, 1, num_masks, 1, 1)     # (B, K, 3, H4, W4)

    probs = torch.sigmoid(mask_logits)  # (B, K, 3, H4, W4)

    # 1. Focal Loss (gamma=2.0)
    gamma = 2.0
    pt = probs * gt_h4_3 + (1.0 - probs) * (1.0 - gt_h4_3)
    focal_weight = (1.0 - pt) ** gamma
    bce = F.binary_cross_entropy_with_logits(mask_logits, gt_h4_3, reduction="none")
    focal_loss = (focal_weight * bce).mean(dim=(-2, -1))  # (B, K, 3)

    # 2. Dice Loss
    inter = (probs * gt_h4_3).sum(dim=(-2, -1))
    union = probs.sum(dim=(-2, -1)) + gt_h4_3.sum(dim=(-2, -1))
    dice_loss = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)  # (B, K, 3)

    # For positive prompts: 20 * Focal + 1 * Dice. For negative background: 20 * Focal
    fg_mask = (point_labels == 1).unsqueeze(-1)  # (B, K, 1)
    seg_loss_per_mask = torch.where(
        fg_mask,
        20.0 * focal_loss + 1.0 * dice_loss,
        20.0 * focal_loss,
    )  # (B, K, 3)

    # Take minimum loss across the 3 ambiguity tokens (SAM Ambiguity Resolution)
    min_seg_loss, best_idx = torch.min(seg_loss_per_mask, dim=2)  # (B, K)
    l_mask = min_seg_loss.mean()

    # 3. Supervised IoU Ranking Loss (Supervise all 3 heads to learn good vs bad masks)
    with torch.no_grad():
        pred_bin = (probs > 0.50).float()
        inter_iou = (pred_bin * gt_h4_3).sum(dim=(-2, -1))
        union_iou = pred_bin.sum(dim=(-2, -1)) + gt_h4_3.sum(dim=(-2, -1)) - inter_iou
        real_iou = (inter_iou / (union_iou + 1e-6)).clamp(0, 1)  # (B, K, 3)

    l_iou = F.mse_loss(iou_preds, real_iou)

    total_loss = l_mask + 1.0 * l_iou
    return {
        "total": total_loss,
        "mask": l_mask,
        "focal": focal_loss.mean(),
        "dice": dice_loss.mean(),
        "iou": l_iou,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training CrownTransformerSAM 2.0 on {device}...")

    model = CrownTransformerSAM().to(device)

    # Warm-start backbone from previous checkpoint
    warm_ckpt = Path("DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth")
    if warm_ckpt.exists():
        state = torch.load(warm_ckpt, map_location=device, weights_only=True)
        backbone_keys = {k.replace("backbone.", ""): v for k, v in state.items() if k.startswith("backbone.")}
        model.backbone.load_state_dict(backbone_keys, strict=False)
        print("Warm-started ResNet50-FPN backbone from TreeFlowNet!")

    dead_dir = Path("DeadTrees/star_convex_targets_v1/train")
    treecover_dir = Path("DeadTrees/star_convex_targets_v1/train_treecover")

    dataset = PanopticPromptableTreeDataset(dead_dir, treecover_dir, crop_size=512, num_prompts_per_image=16)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    epochs = 12
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print(f"\nLaunching Panoptic All-Crown SAM 2.0 Training for {epochs} epochs...")

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        tot_loss = 0.0
        tot_mask = 0.0
        tot_focal = 0.0
        tot_dice = 0.0
        tot_iou = 0.0
        n_steps = 0

        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            point_coords = batch["point_coords"].to(device, non_blocking=True)  # (B, K, 2)
            point_labels = batch["point_labels"].to(device, non_blocking=True)  # (B, K)
            gt_masks = batch["gt_masks"].to(device, non_blocking=True)          # (B, K, H, W)

            optimizer.zero_grad()
            p4, _ = model.extract_features(images)

            mask_logits, iou_preds = model.forward_decoder(
                p4, point_coords, (512, 512), point_labels=point_labels
            )

            losses = sam2_ambiguity_loss(mask_logits, iou_preds, gt_masks, point_labels)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            tot_loss += losses["total"].item()
            tot_mask += losses["mask"].item()
            tot_focal += losses["focal"].item()
            tot_dice += losses["dice"].item()
            tot_iou += losses["iou"].item()
            n_steps += 1

            if n_steps >= 200:
                break

        scheduler.step()
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1:02d}/{epochs} [{elapsed:.1f}s] | "
            f"Total: {tot_loss/n_steps:.4f} | "
            f"MinMaskLoss: {tot_mask/n_steps:.4f} | "
            f"Focal: {tot_focal/n_steps:.4f} | "
            f"Dice: {tot_dice/n_steps:.4f} | "
            f"IoU_MSE: {tot_iou/n_steps:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}",
            flush=True,
        )

        torch.save(model.state_dict(), OUT_DIR / f"crown_transformer_sam2_epoch_{epoch+1:02d}.pth")

    torch.save(model.state_dict(), OUT_DIR / "best_crown_transformer_sam.pth")
    print(f"\nTraining successfully finished! Model saved to: {OUT_DIR}/best_crown_transformer_sam.pth", flush=True)


if __name__ == "__main__":
    main()
