"""H17 — Unified Multi-Task Training (CanopyWatershedNet4Head).

Synthesizes:
1. Photometric & Shadow Simulation (H14)
2. Calibrated Boundary Loss with Sharp Contact Seams (H6)
3. Scale-Adaptive Gaussian Apex Head (H17)
4. Full Multi-Task Backprop into Shared ResNet-50 FPN Trunk (16 epochs, Cosine Annealing)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_conda_p = '/home/ubuntu/anaconda3/envs/cs2/lib/python3.11/site-packages'
_local_p = '/home/chung/.local/lib/python3.11/site-packages'
sys.path = [p for p in sys.path if p != _local_p and p != _conda_p]
sys.path.insert(0, _conda_p)
sys.path.append(_local_p)
sys.path.insert(0, "/home/chung/RS/Image Segmentation")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from dataset_h17 import H17MultiTaskDataset
from model_h17 import CanopyWatershedNet4Head


def tversky_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    tp = (pred * target).sum()
    fp = (pred * (1.0 - target)).sum()
    fn = ((1.0 - pred) * target).sum()
    return 1.0 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)


def balanced_bce_logits(logits: torch.Tensor, target: torch.Tensor, max_weight: float = 20.0) -> torch.Tensor:
    logits = logits.float()
    target = target.float()
    pos = target.sum()
    neg = target.numel() - pos
    w_pos = torch.clamp(neg / (pos + 1.0), max=max_weight)
    weight = 1.0 + (w_pos - 1.0) * target
    return F.binary_cross_entropy_with_logits(logits, target, weight=weight)


class UnifiedMultiTaskLoss(nn.Module):
    """Multi-task loss for Surface, Boundary, Canopy, and Scale-Adaptive Apex."""

    def __init__(self, bound_weight: float = 1.5, apex_weight: float = 1.0):
        super().__init__()
        self.bound_weight = bound_weight
        self.apex_weight = apex_weight

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        surf_tgt: torch.Tensor,
        bound_tgt: torch.Tensor,
        can_tgt: torch.Tensor,
        apex_tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # 1. Surface Loss (Smooth L1)
        l_surf = F.smooth_l1_loss(preds["surface"].float(), surf_tgt.float(), beta=0.05)

        # 2. Canopy Loss (BCE with logits)
        l_can = F.binary_cross_entropy_with_logits(preds["canopy_logits"].float(), can_tgt.float())

        # 3. Calibrated Boundary Loss (Balanced BCE on logits + Tversky on probs)
        l_bound_bce = balanced_bce_logits(preds["boundary_logits"], bound_tgt, max_weight=20.0)
        l_bound_tv = tversky_loss(preds["boundary"], bound_tgt, alpha=0.3, beta=0.7)
        l_bound = l_bound_bce + l_bound_tv

        # 4. Scale-Adaptive Apex Loss (Smooth L1 + Balanced BCE on centroid support)
        l_apex_reg = F.smooth_l1_loss(preds["adaptive_apex"].float(), apex_tgt.float(), beta=0.05)
        l_apex_bce = balanced_bce_logits(preds["apex_logits"], (apex_tgt > 0.1).float(), max_weight=15.0)
        l_apex = l_apex_reg + 0.5 * l_apex_bce

        total_loss = l_surf + l_can + self.bound_weight * l_bound + self.apex_weight * l_apex

        metrics = {
            "loss_surf": l_surf.item(),
            "loss_can": l_can.item(),
            "loss_bound": l_bound.item(),
            "loss_apex": l_apex.item(),
            "total_loss": total_loss.item(),
        }
        return total_loss, metrics


def load_warmstart_weights(model: CanopyWatershedNet4Head, ckpt_path: str, device: torch.device):
    """Warm-starts 4-head model from 3-head checkpoint."""
    print(f"Loading warm-start checkpoint from {ckpt_path}...", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    src_sd = ckpt["model_state_dict"]
    tgt_sd = model.state_dict()

    for k, v in src_sd.items():
        if k in tgt_sd:
            if tgt_sd[k].shape == v.shape:
                tgt_sd[k].copy_(v)
            elif k == "head.1.weight":  # Shape [4, 32, 1, 1] vs [3, 32, 1, 1]
                tgt_sd[k][:3].copy_(v)
                # Initialize 4th head from 1st head (surface) representation
                tgt_sd[k][3].copy_(v[0])
                print(f"  --> Adapted {k}: initialized channel 3 from surface head.")
            elif k == "head.1.bias":  # Shape [4] vs [3]
                tgt_sd[k][:3].copy_(v)
                tgt_sd[k][3] = -2.0  # Background prior
                print(f"  --> Adapted {k}: initialized channel 3 bias to -2.0.")

    model.load_state_dict(tgt_sd)
    print("Warm-start loading completed successfully!", flush=True)


def train_h17():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[H17] Training on device: {device}")

    # Datasets & Loaders
    train_ds = H17MultiTaskDataset(split="train", crop_size=1024, augment=True)
    eval_ds = H17MultiTaskDataset(split="eval", crop_size=1024, augment=False)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=2, shuffle=False, num_workers=2)

    # Model & Loss
    model = CanopyWatershedNet4Head(pretrained=False).to(device)
    warmstart_ckpt = "DeadTrees/experiments/canopy_watershed/best_shadow_watershed.pth"
    if Path(warmstart_ckpt).exists():
        load_warmstart_weights(model, warmstart_ckpt, device)

    criterion = UnifiedMultiTaskLoss(bound_weight=1.5, apex_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    epochs = 16
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    out_dir = Path("DeadTrees/experiments/canopy_watershed")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_ckpt_path = out_dir / "best_unified_v2_watershed.pth"

    print("\n" + "=" * 90)
    print(f"Beginning H17 Unified Multi-Task Retraining ({epochs} epochs)...")
    print("=" * 90)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            surf_tgt = batch["surface"].to(device)
            bound_tgt = batch["boundary"].to(device)
            can_tgt = batch["canopy"].to(device)
            apex_tgt = batch["adaptive_apex"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(images)
                loss, _ = criterion(preds, surf_tgt, bound_tgt, can_tgt, apex_tgt)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)
        dt = time.time() - t0

        # Quick validation evaluation
        model.eval()
        val_loss = 0.0
        val_surf, val_bound, val_apex = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in eval_loader:
                images = batch["image"].to(device)
                surf_tgt = batch["surface"].to(device)
                bound_tgt = batch["boundary"].to(device)
                can_tgt = batch["canopy"].to(device)
                apex_tgt = batch["adaptive_apex"].to(device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    preds = model(images)
                    vloss, m = criterion(preds, surf_tgt, bound_tgt, can_tgt, apex_tgt)
                val_loss += vloss.item()
                val_surf += m["loss_surf"]
                val_bound += m["loss_bound"]
                val_apex += m["loss_apex"]

        val_loss /= len(eval_loader)
        val_surf /= len(eval_loader)
        val_bound /= len(eval_loader)
        val_apex /= len(eval_loader)

        print(
            f"Epoch {epoch:2d}/{epochs:2d} [{dt:.1f}s] - Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} (Surf: {val_surf:.4f}, Bound: {val_bound:.4f}, Apex: {val_apex:.4f})",
            flush=True,
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                best_ckpt_path,
            )
            print(f"  --> Saved new best checkpoint to {best_ckpt_path}", flush=True)

    print("\n[H17] Training completed successfully!", flush=True)


if __name__ == "__main__":
    train_h17()
