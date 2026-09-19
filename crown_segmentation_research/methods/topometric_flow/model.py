"""TopoMetricFlowNet: 1-Stage End-to-End Individual Tree Crown Instance Segmentation Network.

Predicts 4 multi-task dense maps directly in a single forward pass:
1. Centripetal Flow Field V(y, x): 2-channel normalized unit vector pointing to tree apex.
2. Saddle Barrier Energy S(y, x): 1-channel repulsion energy at inter-crown contact interfaces.
3. Learned Potential Surface U(y, x): 1-channel unimodal distance potential.
4. Canopy Support Gate C(y, x): 1-channel binary canopy cover gate.

100% Native PyTorch, Zero external foundation model dependencies.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class TopoMetricDenseHead(nn.Module):
    """Refined convolutional decoder head with GroupNorm & GELU."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, hidden_channels)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, hidden_channels)
        self.act2 = nn.GELU()
        
        self.conv_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.gn1(self.conv1(x)))
        x = self.act2(self.gn2(self.conv2(x)))
        return self.conv_out(x)


class TopoMetricFlowNet(nn.Module):
    """Full 1-Stage TopoMetric Flow Network."""

    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()
        weights_name = "IMAGENET1K_V1" if pretrained_backbone else None
        self.backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=weights_name, trainable_layers=5)
        fpn_channels = self.backbone.out_channels  # 256

        self.flow_head = TopoMetricDenseHead(fpn_channels, 2, hidden_channels=128)
        self.saddle_head = TopoMetricDenseHead(fpn_channels, 1, hidden_channels=64)
        self.surface_head = TopoMetricDenseHead(fpn_channels, 1, hidden_channels=64)
        self.canopy_head = TopoMetricDenseHead(fpn_channels, 1, hidden_channels=64)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """images: (B, 3, H, W) float32 in [0, 1].

        Returns:
            Dictionary with full-resolution (B, C, H, W) maps:
            - "flow": (B, 2, H, W) normalized unit vectors
            - "saddle": (B, 1, H, W) in [0, 1]
            - "surface": (B, 1, H, W) in [0, 1]
            - "canopy": (B, 1, H, W) in [0, 1]
        """
        features = self.backbone(images)
        p2 = features["0"]  # Highest resolution FPN map (stride 4, 256 channels)
        target_size = images.shape[-2:]

        # Raw head logits at stride 4
        flow_raw = self.flow_head(p2)
        saddle_raw = self.saddle_head(p2)
        surface_raw = self.surface_head(p2)
        canopy_raw = self.canopy_head(p2)

        # Bilinear upsample back to original image resolution (stride 4 -> stride 1)
        flow_up = F.interpolate(flow_raw, size=target_size, mode="bilinear", align_corners=False)
        saddle_up = F.interpolate(saddle_raw, size=target_size, mode="bilinear", align_corners=False)
        surface_up = F.interpolate(surface_raw, size=target_size, mode="bilinear", align_corners=False)
        canopy_up = F.interpolate(canopy_raw, size=target_size, mode="bilinear", align_corners=False)

        # Activations
        # 1. Flow field: unit-normalize non-zero vectors
        flow_norm = torch.norm(flow_up, dim=1, keepdim=True) + 1e-6
        flow_unit = flow_up / flow_norm

        # 2. Saddle barrier in [0, 1]
        saddle = torch.sigmoid(saddle_up)

        # 3. Potential surface in [0, 1]
        surface = torch.sigmoid(surface_up)

        # 4. Canopy gate in [0, 1]
        canopy = torch.sigmoid(canopy_up)

        return {
            "flow": flow_unit,
            "saddle": saddle,
            "surface": surface,
            "canopy": canopy,
        }
