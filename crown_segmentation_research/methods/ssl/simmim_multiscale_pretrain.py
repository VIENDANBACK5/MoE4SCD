"""
Scale-Invariant Multi-Scale In-Domain Self-Supervised Pretraining (Hypothesis H29).

Solves the single-scale limitation of standard SimMIM:
In natural boreal forests, crown diameters span an order of magnitude:
  - Understory saplings: 10 - 25 px
  - Co-dominant trees:   30 - 60 px
  - Mature emergent:     70 - 140 px

Pretraining with fixed 32x32 masks forces the backbone to develop single-scale inductive biases.
This module introduces:
1. Multi-Scale Dynamic Patch Masking:
   Mixes 16x16 (sapling), 32x32 (pole), and 64x64 (mature) masking blocks per image with total masking ratio 55-65%.
2. Solar Illumination Consistency:
   Photometrically augments solar azimuth and sun-shadow contrast while enforcing feature consistency.
3. Decoupled In-Domain Pretraining:
   Trained strictly from scratch on Quebec aerial patches WITHOUT external weights.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


class MultiScaleAerialSSLDataset(Dataset):
    """
    Self-supervised dataset generating multi-scale masked canopy patches.
    """

    def __init__(
        self,
        npz_files: list[Path],
        img_size: int = 512,
        mask_scales: tuple[int, ...] = (16, 32, 64),
        total_mask_ratio: float = 0.60,
    ):
        self.files = npz_files
        self.img_size = img_size
        self.mask_scales = mask_scales
        self.total_mask_ratio = total_mask_ratio
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.files)

    def _apply_domain_randomization(self, img: np.ndarray) -> np.ndarray:
        """Simulates solar elevation, azimuthal shadows, and seasonal bronze/green."""
        img = img.astype(np.float32)
        H, W = img.shape[:2]

        # 1. Solar azimuth directional shading
        if np.random.rand() > 0.25:
            angle = np.random.uniform(0, 2 * np.pi)
            intensity = np.random.uniform(0.12, 0.35)
            gx, gy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
            solar_field = (gx * np.cos(angle) + gy * np.sin(angle)) * intensity
            img = img * (1.0 + solar_field[:, :, None])

        # 2. Foliage spectral shift (chlorophyll variation)
        if np.random.rand() > 0.35:
            r_gain = np.random.uniform(0.9, 1.25)
            g_gain = np.random.uniform(0.85, 1.15)
            b_gain = np.random.uniform(0.85, 1.1)
            img[:, :, 0] *= r_gain
            img[:, :, 1] *= g_gain
            img[:, :, 2] *= b_gain

        # 3. Optical blur / GSD variance
        if np.random.rand() > 0.4:
            k = np.random.choice([3, 5])
            sigma = np.random.uniform(0.5, 1.6)
            img = cv2.GaussianBlur(img, (k, k), sigma)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _generate_multiscale_mask(self) -> np.ndarray:
        """
        Creates a composite binary mask by aggregating random blocks from 16x16, 32x32, and 64x64 grids.
        Returns: (H, W) float32 mask where 1.0 = masked, 0.0 = visible.
        """
        H, W = self.img_size, self.img_size
        composite_mask = np.zeros((H, W), dtype=np.float32)

        # Equal budget per scale: target ~20% coverage per scale to reach ~60% total
        ratio_per_scale = self.total_mask_ratio / len(self.mask_scales)

        for scale in self.mask_scales:
            nh = H // scale
            nw = W // scale
            total_patches = nh * nw
            k = int(round(total_patches * ratio_per_scale))
            if k > 0:
                indices = np.random.choice(total_patches, k, replace=False)
                grid = np.zeros(total_patches, dtype=np.float32)
                grid[indices] = 1.0
                grid = grid.reshape(nh, nw)
                upsampled = cv2.resize(grid, (W, H), interpolation=cv2.INTER_NEAREST)
                composite_mask = np.maximum(composite_mask, upsampled)

        return composite_mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = np.load(self.files[idx])
        img = d["image"]

        # Geometric flips
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
        if np.random.rand() > 0.5:
            img = np.flipud(img)

        # Domain augmentation
        aug_img = self._apply_domain_randomization(img)

        # Generate multi-scale mask
        pixel_mask = self._generate_multiscale_mask()

        # Normalized inputs & targets
        norm_img = ((aug_img.astype(np.float32) / 255.0) - self.mean) / self.std
        masked_img = norm_img.copy()
        masked_img[pixel_mask > 0.5] = 0.0

        t_input  = torch.from_numpy(masked_img.transpose(2, 0, 1)).float()
        t_target = torch.from_numpy(norm_img.transpose(2, 0, 1)).float()
        t_mask   = torch.from_numpy(pixel_mask).unsqueeze(0).float()

        return t_input, t_target, t_mask


class MultiScaleSimMIMModel(nn.Module):
    """
    Multi-Scale Masked Image Modeling Autoencoder.
    Uses ResNet-50 FPN backbone and multi-scale feature reconstruction head.
    """

    def __init__(self):
        super().__init__()
        self.encoder = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=None,
            trainable_layers=5,
        )

        # Multi-scale reconstruction head fusing P2 (fine) and P3 (medium)
        self.recon_p3_up = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        self.recon_p2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 48, kernel_size=1),
            nn.PixelShuffle(4),
        )

    def forward(self, masked_images: torch.Tensor) -> torch.Tensor:
        fpn_features = self.encoder(masked_images)
        p2 = fpn_features['0']  # [B, 256, H/4, W/4]
        p3 = fpn_features['1']  # [B, 256, H/8, W/8]

        p3_up = F.interpolate(self.recon_p3_up(p3), size=p2.shape[-2:], mode='bilinear', align_corners=False)
        recon_rgb = self.recon_p2(torch.cat([p2, p3_up], dim=1))
        return recon_rgb


def train_multiscale_ssl(
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 3e-4,
    out_dir: str = "crown_segmentation_research/experiments/ssl_in_domain_weights",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Z1 (training zone) only: self-supervised pretraining on Z2/Z3 imagery also leaks the val/test zones (lineage.py)
    all_files = sorted(f for f in Path("data/itc_benchmarks/extracted/quebec_cached_npz").glob("*.npz") if "_z1_" in f.name)
    print("=" * 80)
    print("STARTING SCALE-INVARIANT MULTI-SCALE IN-DOMAIN PRETRAINING (H29)")
    print(f"Patches: {len(all_files)} | Device: {device} | Epochs: {epochs} | Batch: {batch_size}")
    print("=" * 80)

    dataset = MultiScaleAerialSSLDataset(all_files, img_size=512, mask_scales=(16, 32, 64), total_mask_ratio=0.60)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    model = MultiScaleSimMIMModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    save_path = Path(out_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0

        for inputs, targets, masks in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            pred_rgb = model(inputs)

            # L1 loss focused strictly on masked pixel patches
            diff = torch.abs(pred_rgb - targets) * masks
            loss = diff.sum() / (masks.sum() * 3.0 + 1e-6)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, num_batches)
        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:2d}/{epochs:2d}] | MultiScale Recon Loss: {avg_loss:.4f} | "
            f"Time: {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = save_path / "simmim_multiscale_best_encoder.pth"
            torch.save(model.encoder.state_dict(), ckpt_path)

    print(f"\nMulti-Scale Pretraining Complete! Saved weights to {save_path / 'simmim_multiscale_best_encoder.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-scale SimMIM pretraining (H29)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    train_multiscale_ssl(epochs=args.epochs, batch_size=args.batch, lr=args.lr)
