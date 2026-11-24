"""
TreeFlowNet-HR (Frontier 2026) Training Pipeline.

Integrates:
1. TreeFlowNetHR model architecture with FeatUp Continuous JBU Neck and Foliage Subpixel Head.
2. TreeFlowLossHR with Boundary-IoU, Kervadec Distance Penalty, and Modified Hausdorff SDT.
3. Fast warm-start from TreeFlowNetV3 (`best_model_v3.pth`).
4. Multi-scale target generation with Euclidean Distance Transforms for Kervadec loss.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Clean sys.path to avoid local user-site package conflicts
sys.path = [p for p in sys.path if not p.startswith('/home/chung/.local')]

import random
import cv2
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from crown_segmentation_research.methods.tree_flow.model_hr import TreeFlowNetHR
from crown_segmentation_research.methods.tree_flow.loss_hr import TreeFlowLossHR
from crown_segmentation_research.methods.tree_flow.targets import compute_tree_flow_targets
from crown_segmentation_research.evaluation.eval_exhaustive_quebec import get_patch_files
from crown_segmentation_research.datasets.exhaustive_tree_dataset import QuebecWindowSampler
from crown_segmentation_research.lineage import check_clean_init

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_kervadec_dist_map(canopy_mask: np.ndarray, max_dist: float = 25.0) -> np.ndarray:
    """
    Computes normalized signed distance map for Kervadec loss:
    Positive (> 0) in ground/shadow background.
    Negative (< 0) inside true tree crown.
    """
    fg = canopy_mask.astype(bool)
    bg = ~fg

    if fg.sum() == 0:
        return np.ones_like(canopy_mask, dtype=np.float32)
    if bg.sum() == 0:
        return -np.ones_like(canopy_mask, dtype=np.float32)

    dist_outside = ndimage.distance_transform_edt(bg)
    dist_inside = ndimage.distance_transform_edt(fg)

    # Normalize to [-1, 1] range bounded by max_dist
    norm_outside = np.clip(dist_outside / max_dist, 0.0, 1.0)
    norm_inside = np.clip(dist_inside / max_dist, 0.0, 1.0)

    signed_dist = norm_outside - norm_inside
    return signed_dist.astype(np.float32)[np.newaxis, ...]


class QuebecHRDataset(torch.utils.data.Dataset):
    def __init__(self, split: str = "train", augment: bool = True, sampler=None, n_samples: int = 200):
        # sampler: fresh random Z1 window per sample with labels over a grown grid (full-crown targets);
        # None = the fixed pre-cut `quebec_cached_npz` chips with clipped crowns (--legacy-cache, old runs only)
        self.sampler, self.n_samples = sampler, n_samples
        self.files = get_patch_files(split) if sampler is None else []
        self.augment = augment
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return self.n_samples if self.sampler is not None else len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.sampler is not None:
            img, label = self.sampler.sample(random)  # label grid is larger than img by a margin on every side
        else:
            data = np.load(self.files[idx])
            img = data["image"]  # [H, W, 3] uint8
            label = data["instance_label"]  # [H, W] int32

        if self.augment:
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                label = np.fliplr(label).copy()
            if random.random() > 0.5:
                img = np.flipud(img).copy()
                label = np.flipud(label).copy()
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                img = np.rot90(img, k).copy()
                label = np.rot90(label, k).copy()

        flow, sdt, cent, can = compute_tree_flow_targets(label)
        fg_mask = (label > 0).astype(np.float32)[np.newaxis, ...]
        dist_map = compute_kervadec_dist_map(can[0], max_dist=25.0)
        m, P = (label.shape[0] - img.shape[0]) // 2, img.shape[0]  # targets on the grown grid, keep the centre
        flow, sdt, cent, can, fg_mask, dist_map = (np.ascontiguousarray(t[..., m:m + P, m:m + P]) for t in (flow, sdt, cent, can, fg_mask, dist_map))

        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        img_norm = (img_t - self.mean) / self.std

        return {
            "image": img_norm,
            "raw_rgb": img_t,
            "flow": torch.from_numpy(flow),
            "sdt": torch.from_numpy(sdt),
            "centroid": torch.from_numpy(cent),
            "canopy": torch.from_numpy(can),
            "fg_mask": torch.from_numpy(fg_mask),
            "dist_map": torch.from_numpy(dist_map),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Train TreeFlowNet-HR (Frontier 2026)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument(
        "--warm-start-v3",
        type=str,
        default=None,  # was a checkpoint of the leaked V1 chain (lineage.py); None = ImageNet
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="crown_segmentation_research/experiments/tree_flow_hr",
    )
    parser.add_argument("--allow-leaked-init", action="store_true", help="reproduce an old run from a leaked checkpoint")
    parser.add_argument("--legacy-cache", action="store_true",
                        help="train on the fixed cached chips with clipped crowns (how V2/V3/HR were trained)")
    parser.add_argument("--patch-size", type=int, default=1024, help="512-px chips cover only ~9.6 m at 1.87 cm")
    parser.add_argument("--samples-per-epoch", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    check_clean_init(args.warm_start_v3, args.allow_leaked_init)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STARTING TREEFLOWNET-HR TRAINING (FEATUP JBU + FOLIAGE SUB-PIXEL HEAD)")
    print(f"Device: {DEVICE} | Epochs: {args.epochs} | Batch: {args.batch} | LR: {args.lr}")
    print("=" * 80)

    # 1. Dataset & DataLoader (Quebec Zone 1 Train split)
    print("\n[1/4] Building Training Dataset with Kervadec Euclidean Distance Maps...")
    sampler = None if args.legacy_cache else QuebecWindowSampler(
        REPO / "data/itc_benchmarks/extracted/quebec_2021_09_02", zones=("Z1",), patch_size=args.patch_size)
    train_dataset = QuebecHRDataset(split="train", augment=True, sampler=sampler, n_samples=args.samples_per_epoch)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=(DEVICE == "cuda"),
        drop_last=True,
    )

    # 2. Model Initialization
    print("\n[2/4] Initializing TreeFlowNet-HR Architecture...")
    model = TreeFlowNetHR(fpn_out_ch=256, hr_mid_ch=128, pretrained_backbone=not args.warm_start_v3).to(DEVICE)

    if args.warm_start_v3 and Path(args.warm_start_v3).exists():
        print(f"Warm-starting backbone and macro heads from V3 checkpoint: {args.warm_start_v3}...")
        model.load_from_v3(args.warm_start_v3, device=DEVICE)

    # 3. Loss & Optimizer
    print("\n[3/4] Initializing TreeFlowLossHR & AdamW Optimizer...")
    criterion = TreeFlowLossHR(
        w_flow=1.0,
        w_sdt=1.0,
        w_cent=1.0,
        w_lovasz=0.4,
        w_boundary_iou=0.6,
        w_kervadec=0.3,
        w_boundary_bce=0.3,
    ).to(DEVICE)

    # Discriminative learning rates: lower for pretrained backbone, higher for new HR head
    backbone_params = list(model.backbone.parameters()) + list(model.pyramid_neck.parameters())
    hr_params = (
        list(model.jbu_neck.parameters())
        + list(model.foliage_head.parameters())
        + list(model.flow_head.parameters())
        + list(model.centroid_head.parameters())
        + list(model.sdt_macro_head.parameters())
    )

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.5},
        {'params': hr_params, 'lr': args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 4. Training Loop
    print(f"\n[4/4] Beginning Training Loop ({args.epochs} Epochs)...")
    best_loss = float("inf")
    history = []

    # Fixed Sobel filters for optical edge extraction
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=DEVICE).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=DEVICE).view(1, 1, 3, 3)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_epoch_loss = 0.0
        comp_losses = {
            "flow": 0.0, "sdt": 0.0, "centroid": 0.0,
            "lovasz": 0.0, "b_iou": 0.0, "kervadec": 0.0, "b_bce": 0.0,
        }
        num_batches = 0

        for batch in train_loader:
            imgs = batch["image"].to(DEVICE)
            targets = {
                k: batch[k].to(DEVICE)
                for k in ("flow", "sdt", "centroid", "canopy", "fg_mask", "dist_map")
            }

            # Optical edge gradient for flow modulation
            with torch.no_grad():
                gray = imgs.mean(dim=1, keepdim=True)
                gx = F.conv2d(gray, kx, padding=1)
                gy = F.conv2d(gray, ky, padding=1)
                opt_grad = torch.sqrt(gx ** 2 + gy ** 2)
                opt_grad = opt_grad / (opt_grad.amax(dim=(-2, -1), keepdim=True) + 1e-6)

            optimizer.zero_grad()
            outputs = model(imgs)
            losses = criterion(outputs, targets, opt_grad=opt_grad, dist_map=targets["dist_map"])

            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_epoch_loss += losses["total"].item()
            comp_losses["flow"] += losses["flow"].item()
            comp_losses["sdt"] += losses["sdt"].item()
            comp_losses["centroid"] += losses["centroid"].item()
            comp_losses["lovasz"] += losses["lovasz"].item()
            comp_losses["b_iou"] += losses["boundary_iou"].item()
            comp_losses["kervadec"] += losses["kervadec"].item()
            comp_losses["b_bce"] += losses["boundary_bce"].item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_epoch_loss / max(1, num_batches)
        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "total_loss": avg_loss,
            "flow_loss": comp_losses["flow"] / max(1, num_batches),
            "sdt_loss": comp_losses["sdt"] / max(1, num_batches),
            "cent_loss": comp_losses["centroid"] / max(1, num_batches),
            "lovasz_loss": comp_losses["lovasz"] / max(1, num_batches),
            "b_iou_loss": comp_losses["b_iou"] / max(1, num_batches),
            "kervadec_loss": comp_losses["kervadec"] / max(1, num_batches),
            "time_s": elapsed,
        }
        history.append(row)

        print(
            f"Epoch [{epoch:2d}/{args.epochs:2d}] | Total: {avg_loss:.4f} "
            f"(Flow: {row['flow_loss']:.3f}, SDT: {row['sdt_loss']:.3f}, B-IoU: {row['b_iou_loss']:.3f}, Kerv: {row['kervadec_loss']:.3f}) | "
            f"Time: {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if epoch % 5 == 0 or epoch == args.epochs:  # choose among these on Z2 validation AP, not training loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, out_dir / f"epoch_{epoch:02d}.pth")
        if avg_loss < best_loss:  # kept for compatibility: lowest TRAINING loss, not a model-selection criterion
            best_loss = avg_loss
            ckpt_path = out_dir / "best_model_hr.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, ckpt_path)

    # Save training log
    with open(out_dir / "train_history_hr.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTreeFlowNet-HR Training Complete! Best checkpoint saved to {out_dir / 'best_model_hr.pth'}")


if __name__ == "__main__":
    from crown_segmentation_research.research_integrity.a06_hardening_v2.runtime import reject_legacy_entrypoint
    reject_legacy_entrypoint(__file__)
    main()
