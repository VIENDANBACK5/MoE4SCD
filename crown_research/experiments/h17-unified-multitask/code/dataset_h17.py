"""BAMFORESTS Dataset for H17 Unified Multi-Task Architecture.

Loads 1024x1024 crops, applies Photometric & Shadow Simulation, and computes
the 4 multi-task supervision targets.
"""
from __future__ import annotations

import sys
_conda_p = '/home/ubuntu/anaconda3/envs/cs2/lib/python3.11/site-packages'
_local_p = '/home/chung/.local/lib/python3.11/site-packages'
sys.path = [p for p in sys.path if p != _local_p and p != _conda_p]
sys.path.insert(0, _conda_p)
sys.path.append(_local_p)
sys.path.insert(0, "/home/chung/RS/Image Segmentation")

import numpy as np
import torch
from torch.utils.data import Dataset

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from crown_segmentation_research.methods.canopy_watershed.dataset import BAMCanopyWatershedDataset
from crown_research.src.augment import PhotometricAugmentor
from targets import compute_h17_targets


class H17MultiTaskDataset(Dataset):
    """BAMFORESTS Dataset with Photometric Augmentation and 4 H17 Targets."""

    def __init__(
        self,
        split: str = "train",
        crop_size: int = 1024,
        augment: bool = True,
        seam_dilate_r: int = 3,
    ) -> None:
        super().__init__()
        self.split = split
        self.augment = augment
        self.seam_dilate_r = seam_dilate_r

        # Base dataset loads imagery and polygon instance annotations
        self.base_ds = BAMCanopyWatershedDataset(
            split=split,
            crop_size=crop_size,
            augment=False,  # We handle photometric augmentation explicitly
            compute_targets=False,
        )

        self.photo_augmentor = PhotometricAugmentor(
            shadow_prob=0.60,
            hsv_prob=0.70,
            gamma_prob=0.60,
            contrast_prob=0.50,
        ) if augment else None

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.base_ds[idx]
        image_t = item["image"]  # FloatTensor [3, H, W] in [0, 1]
        instance_label_t = item["instance_label"].squeeze()  # [H, W] int32

        # 1. Photometric & Shadow Simulation on Image
        if self.augment and self.photo_augmentor is not None:
            img_np = (image_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            img_aug = self.photo_augmentor(img_np)
            image_t = torch.from_numpy(img_aug.astype(np.float32) / 255.0).permute(2, 0, 1)

        # 2. Compute 4 Multi-Task Supervision Targets
        inst_np = instance_label_t.numpy().astype(np.int32)
        targets = compute_h17_targets(inst_np, seam_dilate_r=self.seam_dilate_r)

        return {
            "image": image_t,
            "surface": targets["surface"],
            "boundary": targets["boundary"],
            "canopy": targets["canopy"],
            "adaptive_apex": targets["adaptive_apex"],
            "instance_label": instance_label_t,
        }
