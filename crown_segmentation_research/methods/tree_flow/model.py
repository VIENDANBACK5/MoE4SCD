"""TreeFlowNet / OmniCrown architecture.

Multi-task network for Individual Tree Crown (ITC) instance segmentation:
1. Centripetal Flow Field Head: 2 channels (vy, vx) predicting unit direction vectors
   pointing to topological instance centers.
2. Boundary / Signed Distance Transform (SDT) Head: 1 channel in [-1, 1].
3. Centroid Heatmap Head: 1 channel in [0, 1].
4. Canopy Foreground Gate Head: 1 channel in [0, 1].

Uses ResNet50-FPN backbone (warm-startable from Mask R-CNN G1B backbone).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class FlowDenseHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class TreeFlowNet(nn.Module):
    """Dense centripetal flow and boundary prediction network."""

    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()
        weights_name = "IMAGENET1K_V1" if pretrained_backbone else None
        self.backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=weights_name, trainable_layers=5)
        fpn_out_channels = self.backbone.out_channels

        self.flow_head = FlowDenseHead(fpn_out_channels, 2)
        self.sdt_head = FlowDenseHead(fpn_out_channels, 1)
        self.centroid_head = FlowDenseHead(fpn_out_channels, 1)
        self.canopy_head = FlowDenseHead(fpn_out_channels, 1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """images: (N, 3, H, W) float in [0, 1]. Returns full-resolution maps."""
        features = self.backbone(images)
        finest = features["0"]  # stride 4
        target_size = images.shape[-2:]

        # Raw head predictions
        flow_raw = self.flow_head(finest)
        sdt_raw = self.sdt_head(finest)
        centroid_raw = self.centroid_head(finest)
        canopy_raw = self.canopy_head(finest)

        # Activations
        # Flow field: normalize non-zero vectors to unit length
        flow_norm = torch.norm(flow_raw, dim=1, keepdim=True) + 1e-6
        flow_unit = flow_raw / flow_norm

        sdt = torch.tanh(sdt_raw)
        centroid = torch.sigmoid(centroid_raw)
        canopy = torch.sigmoid(canopy_raw)

        # Upsample back to original resolution (stride 4 -> 1)
        flow_full = F.interpolate(flow_unit, size=target_size, mode="bilinear", align_corners=False)
        # Re-normalize after bilinear interpolation
        flow_full_norm = torch.norm(flow_full, dim=1, keepdim=True) + 1e-6
        flow_full = flow_full / flow_full_norm

        sdt_full = F.interpolate(sdt, size=target_size, mode="bilinear", align_corners=False)
        centroid_full = F.interpolate(centroid, size=target_size, mode="bilinear", align_corners=False)
        canopy_full = F.interpolate(canopy, size=target_size, mode="bilinear", align_corners=False)

        return {
            "flow": flow_full,
            "sdt": sdt_full,
            "centroid": centroid_full,
            "canopy": canopy_full,
        }
