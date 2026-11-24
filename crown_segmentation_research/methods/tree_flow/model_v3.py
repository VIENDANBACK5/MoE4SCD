"""
TreeFlowNetV3 — Attention-Gated Multi-Scale Feature Fusion Neck.

Innovation over V2 (Hypothesis H28):
1. In V2, linear pyramid neck addition caused coarse P5 features (stride 32, RF > 256px)
   to drown out high-frequency P2 edge features of small understory trees (APs = 0.11%).
2. AttentionGatedPyramidNeck computes a spatial-scale attention mask:
      Gate_mature = sigmoid(Conv_3x3(Up(P5)))
      Gate_sapling = 1 - Gate_mature
3. Selective feature gating:
      Fused = P2 * (1 + alpha * Gate_sapling) + Gate_mature * SemanticFusion(P3, P4, P5)
   Protecting small saplings and boundary fine details while providing deep context
   for large spreading crowns.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from crown_segmentation_research.methods.tree_flow.model import DenseHead


class AttentionGatedPyramidNeck(nn.Module):
    """
    Attention-Gated Multi-Scale Feature Fusion Neck.
    
    Dynamically balances fine spatial details (P2) for saplings vs
    deep contextual receptive field (P3-P5) for mature spreading crowns.
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 256):
        super().__init__()
        # Semantic context fusion for macro scales P3, P4, P5
        self.semantic_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

        # Spatial scale attention gate predicting mature canopy probability
        self.scale_gate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels // 4),
            nn.GELU(),
            nn.Conv2d(out_channels // 4, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

        # Detail enhancer for small understory saplings
        self.sapling_enhancer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

        # Learnable gating multipliers
        self.gamma_semantic = nn.Parameter(torch.tensor(0.1))
        self.alpha_sapling = nn.Parameter(torch.tensor(0.1))

    def forward(self, fpn_features: dict[str, torch.Tensor]) -> torch.Tensor:
        p2 = fpn_features['0']  # [B, 256, H/4, W/4]
        target_size = p2.shape[-2:]

        p3 = F.interpolate(fpn_features['1'], size=target_size, mode='bilinear', align_corners=False)
        p4 = F.interpolate(fpn_features['2'], size=target_size, mode='bilinear', align_corners=False)
        p5 = F.interpolate(fpn_features['3'], size=target_size, mode='bilinear', align_corners=False)

        # Macro semantic representation from P3 + P4 + P5
        macro_semantic = self.semantic_fusion(torch.cat([p3, p4, p5], dim=1))

        # Dynamic scale gating: where mature crowns exist (Gate ~ 1) vs understory (Gate ~ 0)
        gate_mature = self.scale_gate(p5)  # [B, 1, H/4, W/4]
        gate_sapling = 1.0 - gate_mature

        # Enhance P2 with sapling-aware local edges
        p2_enhanced = p2 + self.alpha_sapling * (gate_sapling * self.sapling_enhancer(p2))

        # Seamless fusion: mature canopy gets macro semantics, saplings preserve sharp P2
        fused = p2_enhanced + self.gamma_semantic * (gate_mature * macro_semantic)

        return fused


class TreeFlowNetV3(nn.Module):
    """
    TreeFlowNetV3 with Attention-Gated Multi-Scale Feature Fusion Neck.
    """

    def __init__(self, fpn_out_ch: int = 256, head_mid_ch: int = 128, pretrained_backbone: bool = False):
        super().__init__()
        weights_name = "IMAGENET1K_V1" if pretrained_backbone else None
        self.backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=weights_name,
            trainable_layers=5,
        )

        # Attention-Gated Multi-Scale Feature Fusion Neck
        self.pyramid_neck = AttentionGatedPyramidNeck(in_channels=fpn_out_ch, out_channels=fpn_out_ch)

        # Four decoupled heads
        self.flow_head     = DenseHead(fpn_out_ch, 2, head_mid_ch)
        self.sdt_head      = DenseHead(fpn_out_ch, 1, head_mid_ch)
        self.centroid_head = DenseHead(fpn_out_ch, 1, head_mid_ch)
        self.canopy_head   = DenseHead(fpn_out_ch, 1, head_mid_ch)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        B, C, H, W = images.shape

        # Backbone generates pyramid P2, P3, P4, P5
        fpn_features = self.backbone(images)

        # Attention-gated multi-scale feature fusion
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

    def load_from_v2(self, ckpt_path: str, device: str = 'cpu'):
        """Warm-start backbone and heads from a trained V2 checkpoint."""
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
        print(f"TreeFlowNetV3 warm-start: loaded {loaded} tensors, skipped {skipped}.")
