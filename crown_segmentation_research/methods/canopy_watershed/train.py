"""Training script for Option A: Neural Learned Potential Surface + Persistence Watershed.

Trains CanopyWatershedNet on BAMFORESTS (1,438 training crops, 382 validation crops)
using mixed precision (bfloat16) and multi-task topological supervision:
- Surface Huber Loss (monotonic distance potential)
- Boundary Focal Loss (inter-crown touching ridges)
- Canopy BCE + Soft Dice Loss (semantic forest gate)

100% Native PyTorch, Zero external foundation model weights.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure root workspace is in sys.path
# Ensure workspace root is in sys.path
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "crown_segmentation_research").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crown_segmentation_research.methods.canopy_watershed.dataset import BAMCanopyWatershedDataset
from crown_segmentation_research.methods.canopy_watershed.model import CanopyWatershedNet


def compute_multitask_loss(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Calculates weighted multi-task loss."""
    pred_surf = preds["surface"].float()
    pred_bound_logits = preds["boundary_logits"].float()
    pred_bound = preds["boundary"].float()
    pred_canopy_logits = preds["canopy_logits"].float()
    pred_canopy = preds["canopy"].float()

    tgt_surf = targets["surface"].float()
    tgt_bound = targets["boundary"].float()
    tgt_canopy = targets["canopy"].float()

    # 1. Surface Huber Loss (weighted higher within canopy)
    canopy_weight = 1.0 + 2.0 * tgt_canopy
    loss_surf = (F.smooth_l1_loss(pred_surf, tgt_surf, reduction="none", beta=0.05) * canopy_weight).mean()

    # 2. Boundary Focal Loss with logits
    bce_bound = F.binary_cross_entropy_with_logits(pred_bound_logits, tgt_bound, reduction="none")
    pt_bound = torch.where(tgt_bound > 0.5, pred_bound, 1.0 - pred_bound)
    focal_bound = ((1.0 - pt_bound) ** 2.0) * bce_bound
    weight_bound = torch.where(tgt_bound > 0.5, 4.0, 1.0)
    loss_bound = (focal_bound * weight_bound).mean()

    # 3. Canopy Support Gate (BCE with logits + Soft Dice)
    loss_canopy_bce = F.binary_cross_entropy_with_logits(pred_canopy_logits, tgt_canopy)
    intersection = (pred_canopy * tgt_canopy).sum(dim=(2, 3))
    union = pred_canopy.sum(dim=(2, 3)) + tgt_canopy.sum(dim=(2, 3))
    dice = 1.0 - (2.0 * intersection + 1e-4) / (union + 1e-4)
    loss_canopy = loss_canopy_bce + dice.mean()

    # Total loss
    total_loss = 4.0 * loss_surf + 3.0 * loss_bound + 1.5 * loss_canopy

    loss_dict = {
        "loss_total": float(total_loss.item()),
        "loss_surface": float(loss_surf.item()),
        "loss_boundary": float(loss_bound.item()),
        "loss_canopy": float(loss_canopy.item()),
    }
    return total_loss, loss_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CanopyWatershedNet on BAMFORESTS.")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU step")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--crop_size", type=int, default=1024, help="Crop size for training")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--out_dir", type=str, default="DeadTrees/experiments/canopy_watershed", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CanopyWatershed] Target Device: {device}", flush=True)

    # Initialize Datasets
    print("[CanopyWatershed] Preparing BAMFORESTS Train & Eval Datasets...", flush=True)
    train_dataset = BAMCanopyWatershedDataset(split="train", crop_size=args.crop_size, augment=True)
    val_dataset = BAMCanopyWatershedDataset(split="eval", crop_size=args.crop_size, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Initialize Model
    print("[CanopyWatershed] Initializing CanopyWatershedNet with ResNet50-FPN backbone...", flush=True)
    model = CanopyWatershedNet(pretrained=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    best_val_loss = float("inf")
    history: list[dict] = []
    start_epoch = 1

    if args.resume and Path(args.resume).exists():
        print(f"[CanopyWatershed] Resuming from checkpoint: {args.resume}", flush=True)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "history" in ckpt:
            history = ckpt["history"]
            for h in history:
                if h.get("val_loss", float("inf")) < best_val_loss:
                    best_val_loss = h["val_loss"]
        saved_epoch = ckpt.get("epoch", 0)
        start_epoch = saved_epoch + 1
        # Step scheduler up to start_epoch
        for _ in range(saved_epoch):
            scheduler.step()
        print(f"[CanopyWatershed] Resumed from Epoch {saved_epoch}. Best Val Loss so far: {best_val_loss:.4f}", flush=True)

    print(f"[CanopyWatershed] Commencing Training Epochs {start_epoch}->{args.epochs} on {len(train_dataset)} crops...", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        running_losses = {"loss_total": 0.0, "loss_surface": 0.0, "loss_boundary": 0.0, "loss_canopy": 0.0}
        num_train_batches = len(train_loader)
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader, 1):
            images = batch["image"].to(device, non_blocking=True)
            targets = {
                "surface": batch["surface"].to(device, non_blocking=True),
                "boundary": batch["boundary"].to(device, non_blocking=True),
                "canopy": batch["canopy"].to(device, non_blocking=True),
            }

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(images)
                loss, loss_dict = compute_multitask_loss(preds, targets)
                loss = loss / args.grad_accum

            loss.backward()

            if batch_idx % args.grad_accum == 0 or batch_idx == num_train_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            for k in running_losses:
                running_losses[k] += loss_dict[k]

            if batch_idx % 25 == 0 or batch_idx == num_train_batches:
                avg_tot = running_losses["loss_total"] / batch_idx
                avg_surf = running_losses["loss_surface"] / batch_idx
                avg_bnd = running_losses["loss_boundary"] / batch_idx
                avg_can = running_losses["loss_canopy"] / batch_idx
                print(
                    f"Epoch [{epoch:02d}/{args.epochs:02d}] Step [{batch_idx:03d}/{num_train_batches:03d}] "
                    f"| Total: {avg_tot:.4f} | Surf: {avg_surf:.4f} | Bound: {avg_bnd:.4f} | Canopy: {avg_can:.4f}",
                    flush=True,
                )

        scheduler.step()
        train_time = time.perf_counter() - epoch_start

        # Validation Loop
        model.eval()
        val_start = time.perf_counter()
        val_losses = {"loss_total": 0.0, "loss_surface": 0.0, "loss_boundary": 0.0, "loss_canopy": 0.0}
        surface_errors = []
        canopy_ious = []
        num_val_batches = len(val_loader)

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                targets = {
                    "surface": batch["surface"].to(device, non_blocking=True),
                    "boundary": batch["boundary"].to(device, non_blocking=True),
                    "canopy": batch["canopy"].to(device, non_blocking=True),
                }

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    preds = model(images)
                    _, loss_dict = compute_multitask_loss(preds, targets)

                for k in val_losses:
                    val_losses[k] += loss_dict[k]

                # Metric tracking
                surf_err = (preds["surface"] - targets["surface"]).abs()
                surface_errors.append(float(surf_err.mean().item()))

                pred_c_bin = preds["canopy"] > 0.5
                tgt_c_bin = targets["canopy"] > 0.5
                inter = (pred_c_bin & tgt_c_bin).sum(dim=(2, 3)).float()
                un = (pred_c_bin | tgt_c_bin).sum(dim=(2, 3)).float()
                iou = ((inter + 1e-4) / (un + 1e-4)).mean().item()
                canopy_ious.append(float(iou))

        val_time = time.perf_counter() - val_start
        avg_val_tot = val_losses["loss_total"] / num_val_batches
        avg_surf_mae = float(np.mean(surface_errors))
        avg_canopy_iou = float(np.mean(canopy_ious))

        print(
            f"--> Epoch [{epoch:02d}/{args.epochs:02d}] Finished! "
            f"Train: {train_time:.1f}s, Val: {val_time:.1f}s | "
            f"Val Loss: {avg_val_tot:.4f} | Surface MAE: {avg_surf_mae:.4f} | Canopy IoU: {avg_canopy_iou:.4f}",
            flush=True,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": running_losses["loss_total"] / num_train_batches,
            "val_loss": avg_val_tot,
            "val_surface_mae": avg_surf_mae,
            "val_canopy_iou": avg_canopy_iou,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(epoch_record)

        # Save latest checkpoint
        latest_ckpt = out_path / "latest_canopy_watershed.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            },
            latest_ckpt,
        )

        # Save best checkpoint
        if avg_val_tot < best_val_loss:
            best_val_loss = avg_val_tot
            best_ckpt = out_path / "best_canopy_watershed.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": avg_val_tot,
                    "val_surface_mae": avg_surf_mae,
                    "val_canopy_iou": avg_canopy_iou,
                },
                best_ckpt,
            )
            print(f"*** New Best Model Saved at Epoch {epoch} with Val Loss: {avg_val_tot:.4f} ***", flush=True)

    with open(out_path / "training_metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    print("[CanopyWatershed] Training Completed Successfully!", flush=True)


if __name__ == "__main__":
    main()
