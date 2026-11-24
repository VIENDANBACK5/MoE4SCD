"""
Self-Supervised Masked Image Modeling (SimMIM) for Remote Sensing Canopy Generalization.
Trains an in-domain foundation representation from SCRATCH on aerial ortho-patches
WITHOUT relying on external pre-trained models or third-party weights.

Key Components:
1. Self-supervised pixel reconstruction of 50% masked canopy patches.
2. Photometric domain randomization:
   - Random Solar Elevation & Azimuth lighting drift.
   - Seasonal Foliage Spectral Shift (Summer Green <-> Autumn Senescent).
   - Multi-GSD Atmospheric Sensor Blur (10cm to 25cm simulation).
"""
from __future__ import annotations

import sys
import time
import glob
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


class AerialSSLDataset(Dataset):
    """Unsupervised remote sensing dataset loading raw optical RGB patches."""

    def __init__(self, npz_files: list[Path], img_size: int = 512, mask_patch_size: int = 32, mask_ratio: float = 0.5):
        self.files = npz_files
        self.img_size = img_size
        self.mask_patch_size = mask_patch_size
        self.mask_ratio = mask_ratio
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.files)

    def _apply_domain_randomization(self, img: np.ndarray) -> np.ndarray:
        """Simulates solar angles, seasonal foliage shifts, and camera blur."""
        img = img.astype(np.float32)
        H, W = img.shape[:2]

        # 1. Solar Azimuth & Elevation Gradient
        if np.random.rand() > 0.3:
            angle = np.random.uniform(0, 2 * np.pi)
            intensity = np.random.uniform(0.1, 0.3)
            gx, gy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
            solar_field = (gx * np.cos(angle) + gy * np.sin(angle)) * intensity
            img = img * (1.0 + solar_field[:, :, None])

        # 2. Seasonal Foliage Spectral Shift (Green <-> Senescent Bronze/Yellow)
        if np.random.rand() > 0.4:
            # Shift R and G channels to simulate autumn chlorosis
            r_gain = np.random.uniform(0.9, 1.25)
            g_gain = np.random.uniform(0.85, 1.15)
            b_gain = np.random.uniform(0.85, 1.1)
            img[:, :, 0] *= r_gain
            img[:, :, 1] *= g_gain
            img[:, :, 2] *= b_gain

        # 3. Atmospheric Haze & Multi-GSD Blur (10cm - 25cm GSD)
        if np.random.rand() > 0.5:
            k = np.random.choice([3, 5])
            sigma = np.random.uniform(0.6, 1.8)
            img = cv2.GaussianBlur(img, (k, k), sigma)

        return np.clip(img, 0, 255).astype(np.uint8)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = np.load(self.files[idx])
        img = d["image"]  # [H, W, 3] uint8

        # Random Flips
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
        if np.random.rand() > 0.5:
            img = np.flipud(img)

        # Domain Randomization
        aug_img = self._apply_domain_randomization(img)

        # Generate Random Binary Patch Mask (1 = masked, 0 = keep)
        H, W = self.img_size, self.img_size
        num_patches_h = H // self.mask_patch_size
        num_patches_w = W // self.mask_patch_size
        num_total_patches = num_patches_h * num_patches_w
        num_masked = int(num_total_patches * self.mask_ratio)

        patch_mask = np.zeros(num_total_patches, dtype=np.float32)
        masked_indices = np.random.choice(num_total_patches, num_masked, replace=False)
        patch_mask[masked_indices] = 1.0
        patch_mask = patch_mask.reshape(num_patches_h, num_patches_w)

        # Upsample patch mask to pixel resolution
        pixel_mask = cv2.resize(patch_mask, (W, H), interpolation=cv2.INTER_NEAREST)  # [H, W] float32

        # Normalize target image and masked input
        norm_img = ((aug_img.astype(np.float32) / 255.0) - self.mean) / self.std
        masked_img = norm_img.copy()
        # Set masked pixels to 0 in normalized space
        masked_img[pixel_mask > 0.5] = 0.0

        t_input  = torch.from_numpy(masked_img.transpose(2, 0, 1)).float()  # [3, H, W]
        t_target = torch.from_numpy(norm_img.transpose(2, 0, 1)).float()    # [3, H, W]
        t_mask   = torch.from_numpy(pixel_mask).unsqueeze(0).float()        # [1, H, W]

        return t_input, t_target, t_mask


class SimMIMAerialModel(nn.Module):
    """
    Self-supervised Masked Autoencoder tailored for ResNet-50 FPN aerial backbone.
    Reconstructs raw foliar pixel values at P2 (stride 4) resolution.
    """

    def __init__(self):
        super().__init__()
        # Same backbone as TreeFlowNet, initialized without pretrained weights
        self.encoder = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=None,
            trainable_layers=5,
        )
        # Lightweight pixel reconstruction decoder
        self.recon_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 48, kernel_size=1),  # 3 channels x 4x4 subpixels = 48
            nn.PixelShuffle(4),                # Stride-4 upsample to native resolution HxW x 3
        )

    def forward(self, masked_images: torch.Tensor) -> torch.Tensor:
        # Backbone extracts multi-scale pyramid
        fpn_features = self.encoder(masked_images)
        p2 = fpn_features['0']  # [B, 256, H/4, W/4]
        recon_rgb = self.recon_head(p2)  # [B, 3, H, W]
        return recon_rgb


def train_ssl(epochs: int = 15, batch_size: int = 8, lr: float = 3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Z1 (training zone) only: self-supervised pretraining on Z2/Z3 imagery also leaks the val/test zones (lineage.py)
    all_files = sorted(f for f in Path("data/itc_benchmarks/extracted/quebec_cached_npz").glob("*.npz") if "_z1_" in f.name)
    print(f"Starting Self-Supervised In-Domain Pretraining on {len(all_files)} Aerial Patches...")

    dataset = AerialSSLDataset(all_files, img_size=512, mask_patch_size=32, mask_ratio=0.5)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = SimMIMAerialModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    out_dir = Path("crown_segmentation_research/experiments/ssl_in_domain_weights")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0

        for inputs, targets, masks in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)  # [B, 1, H, W]

            optimizer.zero_grad()
            pred_rgb = model(inputs)

            # L1 Reconstruction loss strictly on masked pixels
            diff = torch.abs(pred_rgb - targets) * masks
            loss = diff.sum() / (masks.sum() * 3.0 + 1e-6)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, num_batches)
        elapsed = time.time() - t0
        print(f"Epoch [{epoch:2d}/{epochs:2d}] | Mask Recon Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = out_dir / "simmim_best_encoder.pth"
            # Save only the encoder (backbone) weights for TreeFlowNet integration
            torch.save(model.encoder.state_dict(), ckpt_path)

    print(f"\nPretraining Complete! Saved in-domain backbone weights to {out_dir / 'simmim_best_encoder.pth'}")


if __name__ == "__main__":
    train_ssl(epochs=15, batch_size=8, lr=3e-4)
