"""
TreeFlowNetV2 — Full-Pyramid Multi-Scale Feature Fusion Neck.

Innovations over V1:
1. Re-integrates FPN levels P3 (stride 8), P4 (stride 16), and P5 (stride 32)
   which were previously discarded in V1.
2. Cross-scale pyramid fusion via 1x1 + 3x3 grouped convolutions with residual gating.
3. Expanded effective receptive field from ~60px to >250px to resolve large, spreading crowns.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from crown_segmentation_research.methods.tree_flow.model import DenseHead


class PyramidFusionNeck(nn.Module):
    """
    Fuses multi-scale FPN levels P2, P3, P4, P5 into a unified scale-aware representation.
    
    P5 (stride 32, RF > 256px): high-level semantic context for giant crowns.
    P4 (stride 16): intermediate crown clustering.
    P3 (stride 8): medium crown scale context.
    P2 (stride 4): fine spatial detail for leaf boundaries and close apices.
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 256):
        super().__init__()
        # Cross-scale fusion: 256 * 4 = 1024 -> 256
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )
        # Learnable residual gate initialized near 0.1 for stable warm-start
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, fpn_features: dict[str, torch.Tensor]) -> torch.Tensor:
        p2 = fpn_features['0']  # [B, 256, H/4, W/4]
        target_size = p2.shape[-2:]

        p3 = F.interpolate(fpn_features['1'], size=target_size, mode='bilinear', align_corners=False)
        p4 = F.interpolate(fpn_features['2'], size=target_size, mode='bilinear', align_corners=False)
        p5 = F.interpolate(fpn_features['3'], size=target_size, mode='bilinear', align_corners=False)

        concat = torch.cat([p2, p3, p4, p5], dim=1)  # [B, 1024, H/4, W/4]
        fused = self.fusion_conv(concat)

        return p2 + self.gate * fused


class TreeFlowNetV2(nn.Module):
    """
    Full-Pyramid Centripetal Flow Network with Cross-Scale Semantic Fusion.
    """

    def __init__(self, fpn_out_ch: int = 256, head_mid_ch: int = 128, pretrained_backbone: bool = False):
        super().__init__()
        weights_name = "IMAGENET1K_V1" if pretrained_backbone else None
        self.backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=weights_name,
            trainable_layers=5,
        )
        
        # Multi-scale feature fusion neck
        self.pyramid_neck = PyramidFusionNeck(in_channels=fpn_out_ch, out_channels=fpn_out_ch)

        # Four decoupled heads
        self.flow_head     = DenseHead(fpn_out_ch, 2, head_mid_ch)
        self.sdt_head      = DenseHead(fpn_out_ch, 1, head_mid_ch)
        self.centroid_head = DenseHead(fpn_out_ch, 1, head_mid_ch)
        self.canopy_head   = DenseHead(fpn_out_ch, 1, head_mid_ch)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        B, C, H, W = images.shape

        # Backbone generates pyramid P2, P3, P4, P5
        fpn_features = self.backbone(images)

        # Fuse multi-scale features to stride-4 resolution
        fused_features = self.pyramid_neck(fpn_features)  # [B, 256, H/4, W/4]

        # Heads
        flow_raw  = self.flow_head(fused_features)
        sdt_raw   = self.sdt_head(fused_features)
        cent_raw  = self.centroid_head(fused_features)
        can_raw   = self.canopy_head(fused_features)

        # Bilinear upsample to native image resolution H x W
        kw = dict(size=(H, W), mode='bilinear', align_corners=False)
        flow_up  = F.interpolate(flow_raw,  **kw)
        sdt_up   = F.interpolate(sdt_raw,   **kw)
        cent_up  = F.interpolate(cent_raw,  **kw)
        can_up   = F.interpolate(can_raw,   **kw)

        return {
            'flow':         flow_up,
            'sdt':          torch.tanh(sdt_up),
            'sdt_raw':      sdt_up,
            'centroid':     torch.sigmoid(cent_up),
            'centroid_raw': cent_up,
            'canopy':       torch.sigmoid(can_up),
            'canopy_raw':   can_up,
        }

    def load_from_v1(self, ckpt_path: str, device: str = 'cpu'):
        """Warm-start backbone and heads from a trained V1 checkpoint."""
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        own = self.state_dict()
        loaded, skipped = 0, 0
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v)
                loaded += 1
            else:
                skipped += 1
        self.load_state_dict(own)
        print(f"TreeFlowNetV2 warm-start: loaded {loaded} tensors, skipped {skipped} (neck freshly initialized).")
