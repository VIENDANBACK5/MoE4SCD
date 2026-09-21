"""H14 — Shadow & Photometric Augmentation for Domain Adaptation.

Warm-starts from best_calibrated_watershed.pth (H6) and trains 12 epochs with
photometric shadow, HSV jitter, and gamma variation to make heads robust against
lighting and sensor shifts (TestSet1).
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

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from crown_segmentation_research.methods.canopy_watershed.dataset import BAMCanopyWatershedDataset
from crown_segmentation_research.methods.canopy_watershed.model import CanopyWatershedNet
from crown_segmentation_research.methods.canopy_watershed.train_calibrated import SharpRidgeCanopyLoss
from crown_research.src.augment import PhotometricAugmentor


class ShadowAugmentedDataset(BAMCanopyWatershedDataset):
    """Wraps BAMCanopyWatershedDataset with Photometric & Shadow Augmentor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.photo_augmentor = PhotometricAugmentor(
            shadow_prob=0.60,
            hsv_prob=0.70,
            gamma_prob=0.60,
            contrast_prob=0.50
        )

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        if self.augment:
            img_np = (item['image'].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            img_aug = self.photo_augmentor(img_np)
            item['image'] = torch.from_numpy(img_aug.astype(np.float32) / 255.0).permute(2, 0, 1)
        return item


def train_h14():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[H14] Using device: {device}")

    # Dataset & Loader
    train_ds = ShadowAugmentedDataset(
        split="train",
        crop_size=1024,
        augment=True,
        compute_targets=True,
        sharp_targets=True
    )
    eval_ds = BAMCanopyWatershedDataset(
        split="eval",
        crop_size=1024,
        augment=False,
        compute_targets=True,
        sharp_targets=True
    )

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=2, shuffle=False, num_workers=2)

    # Model: warm-start from H6 calibrated checkpoint
    model = CanopyWatershedNet(pretrained=False).to(device)
    ckpt_path = "DeadTrees/experiments/canopy_watershed/best_calibrated_watershed.pth"
    print(f"Loading warm-start checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    criterion = SharpRidgeCanopyLoss(seam_weight=4.0, bound_weight=1.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    epochs = 12
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    out_dir = Path("DeadTrees/experiments/canopy_watershed")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    best_ckpt_path = out_dir / "best_shadow_watershed.pth"

    print("\n" + "="*80)
    print("Beginning H14 Shadow & Photometric Retraining (12 epochs)...")
    print("="*80)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            surf_tgt = batch['surface'].to(device)
            bound_tgt = batch['boundary'].to(device)
            canopy_tgt = batch['canopy'].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                preds = model(images)
            total_loss, ldict = criterion(preds, surf_tgt, bound_tgt, canopy_tgt)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += total_loss.item()

        scheduler.step()
        train_loss /= len(train_loader)
        dt = time.time() - t0

        # Quick validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in eval_loader:
                images = batch['image'].to(device)
                surf_tgt = batch['surface'].to(device)
                bound_tgt = batch['boundary'].to(device)
                canopy_tgt = batch['canopy'].to(device)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    preds = model(images)
                vloss, _ = criterion(preds, surf_tgt, bound_tgt, canopy_tgt)
                val_loss += vloss.item()

        val_loss /= len(eval_loader)
        print(f"Epoch {epoch:2d}/{epochs:2d} [{dt:.1f}s] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, best_ckpt_path)
            print(f"  --> Saved new best checkpoint to {best_ckpt_path}")

    print("\n[H14] Training completed successfully!")


if __name__ == "__main__":
    train_h14()
