"""CanopyWatershedNet: Neural Learned Potential Surface & Boundary Ridge Network.

Architecture:
- Backbone: ResNet-50 FPN with multi-scale feature pyramid fusion (P2, P3, P4, P5).
- High-Resolution Feature Decoder: Progressive upsampling with residual GroupNorm convolutions
  recovering fine 1024x1024 spatial resolution (stride 4 -> stride 2 -> stride 1).
- Multi-Task Prediction Heads:
  1. surface_head: Virtual unimodal canopy potential U(y, x) in [0, 1].
  2. boundary_head: Inter-crown saddle boundary barrier B(y, x) in [0, 1].
  3. canopy_head: Binary semantic canopy support gate M(y, x) in [0, 1].

100% Native PyTorch, Zero external foundation model dependencies.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class ConvBlock(nn.Module):
    """Conv2d + GroupNorm + GELU building block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        num_groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CanopyWatershedNet(nn.Module):
    """Multi-task neural surface and boundary estimator."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights_name = "IMAGENET1K_V1" if pretrained else None
        self.backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=weights_name,
            trainable_layers=5,
        )
        fpn_dim = self.backbone.out_channels  # 256

        # Multi-scale FPN Fusion: fuse P3, P4, P5 into P2 resolution (stride 4)
        self.fuse_p3 = nn.Conv2d(fpn_dim, 64, kernel_size=1)
        self.fuse_p4 = nn.Conv2d(fpn_dim, 64, kernel_size=1)
        self.fuse_p5 = nn.Conv2d(fpn_dim, 64, kernel_size=1)
        self.fuse_p2 = nn.Conv2d(fpn_dim, 128, kernel_size=1)

        # Fused feature processing at stride 4 (320 channels -> 128 channels)
        self.fused_conv = nn.Sequential(
            ConvBlock(128 + 64 * 3, 128),
            ConvBlock(128, 128),
        )

        # Progressive high-resolution decoder:
        # Stride 4 -> Stride 2
        self.up_stride2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(128, 64),
        )
        # Stride 2 -> Stride 1 (full 1024x1024 resolution)
        self.up_stride1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(64, 32),
        )

        # Shared multi-task high-resolution prediction head
        self.head = nn.Sequential(
            ConvBlock(32, 32),
            nn.Conv2d(32, 3, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            images: (B, 3, H, W) normalized float32 tensor in [0, 1].

        Returns:
            Dictionary containing:
            - "surface": (B, 1, H, W) continuous unimodal potential in [0, 1].
            - "boundary": (B, 1, H, W) saddle barrier energy in [0, 1].
            - "canopy": (B, 1, H, W) binary canopy probability in [0, 1].
        """
        features = self.backbone(images)
        p2 = features["0"]  # Stride 4, (B, 256, H/4, W/4)
        p3 = features["1"]  # Stride 8
        p4 = features["2"]  # Stride 16
        p5 = features["3"]  # Stride 32

        target_size_s4 = p2.shape[-2:]

        # Project and upsample higher-level FPN features to stride 4
        p2_proj = self.fuse_p2(p2)
        p3_up = F.interpolate(self.fuse_p3(p3), size=target_size_s4, mode="bilinear", align_corners=False)
        p4_up = F.interpolate(self.fuse_p4(p4), size=target_size_s4, mode="bilinear", align_corners=False)
        p5_up = F.interpolate(self.fuse_p5(p5), size=target_size_s4, mode="bilinear", align_corners=False)

        # Multi-scale aggregation
        fused = torch.cat([p2_proj, p3_up, p4_up, p5_up], dim=1)
        feat_s4 = self.fused_conv(fused)

        # Progressive upsampling to stride 1
        feat_s2 = self.up_stride2(feat_s4)
        feat_s1 = self.up_stride1(feat_s2)

        # Output logits (B, 3, H, W)
        logits = self.head(feat_s1)
        surface_logits = logits[:, 0:1]
        boundary_logits = logits[:, 1:2]
        canopy_logits = logits[:, 2:3]

        # Constrain to [0, 1] via Sigmoid
        surface = torch.sigmoid(surface_logits)
        boundary = torch.sigmoid(boundary_logits)
        canopy = torch.sigmoid(canopy_logits)

        return {
            "surface": surface,
            "boundary": boundary,
            "canopy": canopy,
            "surface_logits": surface_logits,
            "boundary_logits": boundary_logits,
            "canopy_logits": canopy_logits,
        }
