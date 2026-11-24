"""H7 variant — adds a scale-invariant apex head on top of H6's calibrated boundary.

Differences from train_calibrated.py: a fourth output channel supervised by the
previously-discarded batch["apex"] target, using the same class-balanced objective H6
validated. train_sharp.py and train_calibrated.py are left untouched so both earlier
baselines stay reproducible. See crown_research/experiments/h7-apex-head/protocol.md.

Original docstring follows.

Stage 2: Sharp-Ridge Retraining for CanopyWatershedNet.

Enforces deep zero-energy chasms at inter-crown touching interfaces (contact seams)
and sharp apex centers to eliminate saddle-point collapse.
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

for _p in Path(__file__).resolve().parents:
    if (_p / "crown_segmentation_research").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crown_segmentation_research.methods.canopy_watershed.dataset import BAMCanopyWatershedDataset
from crown_segmentation_research.methods.canopy_watershed.decode import decode_watershed_fast
from crown_segmentation_research.methods.canopy_watershed.model_apex import CanopyWatershedNet


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Computes continuous Dice loss on probabilities."""
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1.0 - (2.0 * intersection + eps) / (pred_flat.sum() + target_flat.sum() + eps)


def tversky_loss(pred: torch.Tensor, target: torch.Tensor,
                 alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6) -> torch.Tensor:
    """Asymmetric Dice (Salehi et al., 2017). beta > alpha penalises false negatives.

    Dice is the alpha = beta = 0.5 special case; it has already been shown insufficient
    to keep this head's magnitude off zero at a ~6% positive rate.
    """
    tp = (pred * target).sum()
    fp = (pred * (1.0 - target)).sum()
    fn = ((1.0 - pred) * target).sum()
    return 1.0 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)


def balanced_bce(pred: torch.Tensor, target: torch.Tensor, max_weight: float = 20.0) -> torch.Tensor:
    """BCE with the positive class up-weighted by the batch's own imbalance ratio.

    The ratio is measured, not tuned: it is a property of the data. Clamped so a crop
    that happens to be nearly seam-free cannot blow up the gradient scale.
    """
    pos = target.sum()
    neg = target.numel() - pos
    w_pos = torch.clamp(neg / (pos + 1.0), max=max_weight)
    weight = 1.0 + (w_pos - 1.0) * target
    return F.binary_cross_entropy(pred, target, weight=weight)


class SharpRidgeCanopyLoss(nn.Module):
    """Multi-task loss with explicit boundary chasm penalty."""

    def __init__(self, seam_weight: float = 4.0, bound_weight: float = 1.5, apex_weight: float = 1.5):
        super().__init__()
        self.seam_weight = seam_weight
        self.bound_weight = bound_weight
        self.apex_weight = apex_weight
        self.l1_loss = nn.SmoothL1Loss(beta=0.05)

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        surf_tgt: torch.Tensor,
        bound_tgt: torch.Tensor,
        canopy_tgt: torch.Tensor,
        apex_tgt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        with torch.amp.autocast("cuda", enabled=False):
            pred_surf = preds["surface"].float()
            pred_bound = preds["boundary"].float()
            pred_canopy = preds["canopy"].float()
            surf_tgt = surf_tgt.float()
            bound_tgt = bound_tgt.float()
            canopy_tgt = canopy_tgt.float()

            # 1. Surface Loss (Smooth L1 over canopy area)
            canopy_fg = (canopy_tgt > 0.5).float()
            l_surf_base = self.l1_loss(pred_surf * canopy_fg, surf_tgt * canopy_fg)

            # 2. Seam Chasm Penalty: Force surface to 0.0 at touching boundary seams
            seam_mask = (bound_tgt > 0.5).float()
            if seam_mask.sum() > 0:
                l_seam = torch.sum((pred_surf * seam_mask) ** 2) / (seam_mask.sum() + 1e-6)
            else:
                l_seam = torch.tensor(0.0, device=pred_surf.device)

            l_surf_total = l_surf_base + self.seam_weight * l_seam

            # 3. Boundary Barrier Loss: class-balanced BCE + asymmetric Tversky (H6)
            l_bound_bce = balanced_bce(pred_bound, bound_tgt)
            l_bound_dice = tversky_loss(pred_bound, bound_tgt, alpha=0.3, beta=0.7)
            l_bound_total = l_bound_bce + l_bound_dice

            # 4. Canopy Gate Loss: BCE + Dice
            l_canopy_bce = F.binary_cross_entropy(pred_canopy, canopy_tgt)
            l_canopy_dice = dice_loss(pred_canopy, canopy_tgt)
            l_canopy_total = l_canopy_bce + l_canopy_dice

            # 5. Apex Loss (H7): same class-balanced objective as the boundary head.
            # The apex target is even sparser than the seam band, so an unweighted loss
            # would collapse it exactly as the old boundary loss collapsed.
            if apex_tgt is not None:
                pred_apex = preds["apex"].float()
                l_apex = balanced_bce(pred_apex, apex_tgt) + tversky_loss(pred_apex, apex_tgt, 0.3, 0.7)
            else:
                l_apex = torch.zeros((), device=pred_surf.device)

            total_loss = (l_surf_total + self.bound_weight * l_bound_total
                          + l_canopy_total + self.apex_weight * l_apex)

            loss_dict = {
                "loss_total": float(total_loss.item()),
                "loss_surf": float(l_surf_base.item()),
            "loss_seam": float(l_seam.item()),
            "loss_bound": float(l_bound_total.item()),
            "loss_canopy": float(l_canopy_total.item()),
            "loss_apex": float(l_apex.item()),
        }
        return total_loss, loss_dict


def evaluate_val_quick(
    model: nn.Module,
    val_dataset: BAMCanopyWatershedDataset,
    device: torch.device,
    num_samples: int = 15,
) -> float:
    """Quick validation metric calculation."""
    model.eval()
    total_trees = 0
    N = min(num_samples, len(val_dataset))

    with torch.no_grad():
        for i in range(N):
            item = val_dataset[i]
            img = item["image"].unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(img)
            surf_p = preds["surface"].squeeze().float().cpu().numpy()
            bound_p = preds["boundary"].squeeze().float().cpu().numpy()
            canopy_p = preds["canopy"].squeeze().float().cpu().numpy()

            _, instances = decode_watershed_fast(
                surface=surf_p,
                boundary=bound_p,
                canopy=canopy_p,
                kernel_size=5,
                min_apex_val=0.20,
                min_canopy_val=0.30,
                pers_thresh=0.03,
                min_distance=5.0,
                bound_weight=1.0,
                min_area=25,
            )
            total_trees += len(instances)

    model.train()
    return total_trees / N


def main():
    import random
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, default="DeadTrees/experiments/canopy_watershed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / "best_apex_watershed.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage 2 Training] Initializing on {device}...", flush=True)

    train_dataset = BAMCanopyWatershedDataset(
        split="train",
        crop_size=1024,
        augment=True,
        compute_targets=True,
        sharp_targets=True,
    )
    val_dataset = BAMCanopyWatershedDataset(
        split="eval",
        crop_size=1024,
        augment=False,
        compute_targets=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    model = CanopyWatershedNet(pretrained=True).to(device)

    # Initialize from previous best watershed weights if available
    init_ckpt = out_dir / "best_calibrated_watershed.pth"
    if init_ckpt.exists():
        print(f"[Stage 2 Training] Warm-starting weights from {init_ckpt}...", flush=True)
        ckpt = torch.load(init_ckpt, map_location=device, weights_only=False)
        sd = ckpt["model_state_dict"]
        own = model.state_dict()
        # The final 1x1 conv grew from 3 to 4 output channels; copy the three trained
        # channels and leave the new apex channel at its initialisation.
        kept = 0
        for k, v in sd.items():
            if k not in own:
                continue
            if own[k].shape == v.shape:
                own[k] = v
                kept += 1
            elif own[k].shape[0] == v.shape[0] + 1 and own[k].shape[1:] == v.shape[1:]:
                own[k][: v.shape[0]] = v
                kept += 1
                print(f"[H7] partially loaded {k}: {tuple(v.shape)} -> {tuple(own[k].shape)}", flush=True)
        model.load_state_dict(own)
        print(f"[H7] warm-started {kept}/{len(own)} tensors from {init_ckpt}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = SharpRidgeCanopyLoss(seam_weight=4.0, bound_weight=1.5)
    scaler = torch.amp.GradScaler("cuda")

    best_loss = float("inf")
    print(f"[Stage 2 Training] Starting training for {args.epochs} epochs...", flush=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_losses = []

        for step, batch in enumerate(train_loader, 1):
            images = batch["image"].to(device, non_blocking=True)
            surf_tgt = batch["surface"].to(device, non_blocking=True)
            bound_tgt = batch["boundary"].to(device, non_blocking=True)
            canopy_tgt = batch["canopy"].to(device, non_blocking=True)
            apex_tgt = batch["apex"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(images)
                loss, loss_dict = criterion(preds, surf_tgt, bound_tgt, canopy_tgt, apex_tgt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss_dict["loss_total"])

            if step % 20 == 0 or step == len(train_loader):
                print(
                    f"  Epoch [{epoch:02d}/{args.epochs:02d}] Step [{step:03d}/{len(train_loader):03d}] | "
                    f"Loss: {loss_dict['loss_total']:.4f} (Surf: {loss_dict['loss_surf']:.4f}, "
                    f"Bound: {loss_dict['loss_bound']:.4f}, Apex: {loss_dict['loss_apex']:.4f})",
                    flush=True,
                )

        scheduler.step()
        avg_loss = float(np.mean(epoch_losses))
        avg_trees = evaluate_val_quick(model, val_dataset, device, num_samples=15)
        elapsed = time.time() - t0

        print(
            f"--> Epoch {epoch:02d} Finished in {elapsed:.1f}s | Avg Loss: {avg_loss:.4f} | "
            f"Val Tree Density: {avg_trees:.1f} trees/crop | LR: {scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                },
                best_ckpt_path,
            )
            print(f"    *** Checkpoint saved to {best_ckpt_path} ***", flush=True)

    print(f"\n[Stage 2 Training] Completed successfully! Best checkpoint at {best_ckpt_path}", flush=True)


if __name__ == "__main__":
    from crown_segmentation_research.research_integrity.a06_hardening_v2.runtime import reject_legacy_entrypoint
    reject_legacy_entrypoint(__file__)
    main()
