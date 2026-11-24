"""
TreeFlowNetV3 Training Pipeline with Attention-Gated Multi-Scale Feature Fusion Neck (H28).

Features:
1. TreeFlowNetV3 model architecture with Attention-Gated Pyramid Neck.
2. Sobolev edge-alignment + Boundary-weighted SDT + Lovász canopy loss.
3. Warm-starts from TreeFlowNetV2 best model (`best_model_v2.pth`) to inherit deep representation
   while learning the spatial scale gating weights for small vs mature trees.
4. Auto-saves best model to `crown_segmentation_research/experiments/tree_flow_v3/best_model_v3.pth`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import random
from crown_segmentation_research.methods.tree_flow.model_v3 import TreeFlowNetV3
from crown_segmentation_research.methods.tree_flow.loss_v2 import TreeFlowLossV2
from crown_segmentation_research.methods.tree_flow.targets import compute_tree_flow_targets
from crown_segmentation_research.evaluation.eval_exhaustive_quebec import get_patch_files
from crown_segmentation_research.datasets.exhaustive_tree_dataset import QuebecWindowSampler
from crown_segmentation_research.lineage import check_clean_init

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class QuebecNpzDataset(torch.utils.data.Dataset):
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
        m, P = (label.shape[0] - img.shape[0]) // 2, img.shape[0]  # targets on the grown grid, keep the centre
        flow, sdt, cent, can, fg_mask = (np.ascontiguousarray(t[..., m:m + P, m:m + P]) for t in (flow, sdt, cent, can, fg_mask))

        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        img_t = (img_t - self.mean) / self.std

        return {
            "image": img_t,
            "flow": torch.from_numpy(flow),
            "sdt": torch.from_numpy(sdt),
            "centroid": torch.from_numpy(cent),
            "canopy": torch.from_numpy(can),
            "fg_mask": torch.from_numpy(fg_mask),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Train TreeFlowNetV3 (H28)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument(
        "--warm-start-v2",
        type=str,
        default=None,  # was a checkpoint of the leaked V1 chain (lineage.py); None = ImageNet
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="crown_segmentation_research/experiments/tree_flow_v3",
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
    check_clean_init(args.warm_start_v2, args.allow_leaked_init)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STARTING TREEFLOWNET V3 TRAINING (ATTENTION-GATED MULTI-SCALE NECK)")
    print(f"Device: {DEVICE} | Epochs: {args.epochs} | Batch: {args.batch} | LR: {args.lr}")
    print("=" * 80)

    # 1. Dataset & DataLoader (Quebec Zone 1 Train split)
    print("\n[1/4] Building Training Datasets (Quebec Zone 1)...")
    sampler = None if args.legacy_cache else QuebecWindowSampler(
        REPO / "data/itc_benchmarks/extracted/quebec_2021_09_02", zones=("Z1",), patch_size=args.patch_size)
    train_dataset = QuebecNpzDataset(split="train", augment=True, sampler=sampler, n_samples=args.samples_per_epoch)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=(DEVICE == "cuda"),
        drop_last=True,
    )

    # 2. Model Initialization
    print("\n[2/4] Initializing TreeFlowNetV3 Architecture...")
    model = TreeFlowNetV3(fpn_out_ch=256, head_mid_ch=128, pretrained_backbone=not args.warm_start_v2).to(DEVICE)

    if args.warm_start_v2 and Path(args.warm_start_v2).exists():
        print(f"Warm-starting from V2 checkpoint: {args.warm_start_v2}...")
        model.load_from_v2(args.warm_start_v2, device=DEVICE)

    # 3. Loss & Optimizer
    print("\n[3/4] Initializing Boundary-Aware Loss & AdamW Optimizer...")
    criterion = TreeFlowLossV2().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 4. Training Loop
    print(f"\n[4/4] Beginning Training Loop ({args.epochs} Epochs)...")
    best_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_epoch_loss = 0.0
        comp_losses = {"flow": 0.0, "sdt": 0.0, "centroid": 0.0, "canopy": 0.0}
        num_batches = 0

        for batch in train_loader:
            imgs = batch["image"].to(DEVICE)
            targets = {
                k: batch[k].to(DEVICE)
                for k in ("flow", "sdt", "centroid", "canopy", "fg_mask")
            }

            # Extract optical edge gradient for Sobolev alignment
            with torch.no_grad():
                gray = imgs.mean(dim=1, keepdim=True)
                kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=DEVICE).view(1, 1, 3, 3)
                ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=DEVICE).view(1, 1, 3, 3)
                gx = F.conv2d(gray, kx, padding=1)
                gy = F.conv2d(gray, ky, padding=1)
                opt_grad = torch.sqrt(gx ** 2 + gy ** 2)
                opt_grad = opt_grad / (opt_grad.amax(dim=(-2, -1), keepdim=True) + 1e-6)

            optimizer.zero_grad()
            outputs = model(imgs)
            losses = criterion(outputs, targets, opt_grad=opt_grad)

            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_epoch_loss += losses["total"].item()
            comp_losses["flow"] += losses["flow"].item()
            comp_losses["sdt"] += losses["sdt"].item()
            comp_losses["centroid"] += losses["centroid"].item()
            comp_losses["canopy"] += losses["lovasz"].item()
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
            "canopy_loss": comp_losses["canopy"] / max(1, num_batches),
            "time_s": elapsed,
        }
        history.append(row)

        print(
            f"Epoch [{epoch:2d}/{args.epochs:2d}] | Total: {avg_loss:.4f} "
            f"(Flow: {row['flow_loss']:.3f}, SDT: {row['sdt_loss']:.3f}, Cent: {row['cent_loss']:.3f}, Lov: {row['canopy_loss']:.3f}) | "
            f"Time: {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if epoch % 5 == 0 or epoch == args.epochs:  # choose among these on Z2 validation AP, not training loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, out_dir / f"epoch_{epoch:02d}.pth")
        if avg_loss < best_loss:  # kept for compatibility: lowest TRAINING loss, not a model-selection criterion
            best_loss = avg_loss
            ckpt_path = out_dir / "best_model_v3.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, ckpt_path)

    # Save training log
    with open(out_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTreeFlowNetV3 Training Complete! Best checkpoint saved to {out_dir / 'best_model_v3.pth'}")


if __name__ == "__main__":
    from crown_segmentation_research.research_integrity.a06_hardening_v2.runtime import reject_legacy_entrypoint
    reject_legacy_entrypoint(__file__)
    main()
