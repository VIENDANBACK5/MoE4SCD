"""
TreeFlowNet-HR (Frontier 2026 Edition) — High-Resolution Foliage Boundary Refinement Network.

Interdisciplinary Synthesis of:
1. FeatUp (ICLR 2024): Continuous Joint Bilateral Upsampling (JBU) Neck guided by native RGB.
2. Exemplar (arXiv 09/2026): Native-Resolution Filter Banks for high-frequency edge preservation.
3. BAISeg (IEEE ISBI 2026): Continuous Multi-Instance Signed Distance Functions (SDFs).
4. PointRend (CVPR Oral) & RefineMask (CVPR): Dual-stream sub-pixel boundary refinement.
5. Omnipose & HoVer-Net: Eikonal divergence-aware flow field separation.

Architecture:
- Asymmetric Dual-Stream:
  * Macro Stream (Stride 4, 256x256): Large receptive field for canopy topology, flow vectors V, and tree apex centroids.
  * Micro Stream (Stride 1, 1024x1024): FeatUp Continuous JBU Neck + FoliageSubpixelHead for 5cm needle foliage adherence.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from crown_segmentation_research.methods.tree_flow.model import DenseHead
from crown_segmentation_research.methods.tree_flow.model_v3 import AttentionGatedPyramidNeck


class NativeFilterBank(nn.Module):
    """
    Exemplar (09/2026) inspired Native-Resolution Filter Bank.
    Extracts fixed optical gradient and high-frequency texture features directly at Stride 1.
    """

    def __init__(self):
        super().__init__()
        # Fixed Sobel and Laplacian kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        laplace = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('laplace', laplace)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        rgb: [B, 3, H, W] normalized image
        Returns: [B, 6, H, W] edge texture tensor (RGB + GradX + GradY + Laplace)
        """
        gray = rgb.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        lap = F.conv2d(gray, self.laplace, padding=1)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return torch.cat([rgb, gx, gy, grad_mag], dim=1)  # 6 channels


class FeatUpJBUContinuousNeck(nn.Module):
    """
    FeatUp (ICLR 2024) inspired Joint Bilateral Upsampling (JBU) Neck.
    Dynamically transfers macro semantic features (Stride 4) to native resolution (Stride 1)
    using bilateral affinities guided by high-resolution optical gradients and textures.
    """

    def __init__(self, in_channels: int = 256, guide_channels: int = 6, out_channels: int = 128):
        super().__init__()
        self.filter_bank = NativeFilterBank()

        # Guidance projection from native RGB + filter bank
        self.guidance_encoder = nn.Sequential(
            nn.Conv2d(guide_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )

        # Macro feature projection
        self.macro_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

        # Bilateral kernel generator: computes dynamic spatial & range weights
        self.bilateral_gate = nn.Sequential(
            nn.Conv2d(out_channels + 64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # High-resolution refinement block
        self.hr_refiner = nn.Sequential(
            nn.Conv2d(out_channels + 64, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, macro_feat: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """
        macro_feat: [B, 256, H/4, W/4]
        rgb: [B, 3, H, W]
        Returns: [B, 128, H, W] sharp native resolution features
        """
        H, W = rgb.shape[-2:]

        # 1. Extract native filter bank guidance
        guide_t = self.filter_bank(rgb)  # [B, 6, H, W]
        guide_feat = self.guidance_encoder(guide_t)  # [B, 64, H, W]

        # 2. Project and upsample macro features
        macro_proj = self.macro_proj(macro_feat)  # [B, 128, H/4, W/4]
        macro_up = F.interpolate(macro_proj, size=(H, W), mode='bilinear', align_corners=False)  # [B, 128, H, W]

        # 3. Bilateral dynamic gating (FeatUp range kernel)
        fused_init = torch.cat([macro_up, guide_feat], dim=1)  # [B, 128 + 64, H, W]
        gate = self.bilateral_gate(fused_init)  # [B, 128, H, W]

        # 4. Modulated residual feature construction
        modulated = macro_up * (1.0 + gate)
        hr_features = self.hr_refiner(torch.cat([modulated, guide_feat], dim=1))  # [B, 128, H, W]

        return hr_features


class FoliageSubpixelHead(nn.Module):
    """
    Sub-Pixel Foliage Boundary Head (PointRend & BAISeg style).
    Predicts native resolution canopy coverage, sub-pixel residual boundary mask,
    and continuous Signed Distance Transform (SDF).
    """

    def __init__(self, in_channels: int = 128, mid_channels: int = 64):
        super().__init__()
        # Canopy coverage head (Sub-pixel Stride 1)
        self.canopy_head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True),
        )

        # High-resolution boundary band detector (3-5px perimeter)
        self.boundary_head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True),
        )

        # Native Continuous Signed Distance Transform (SDF) Head
        self.sdf_hr_head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, hr_features: torch.Tensor) -> dict[str, torch.Tensor]:
        canopy_raw = self.canopy_head(hr_features)
        boundary_raw = self.boundary_head(hr_features)
        sdf_raw = self.sdf_hr_head(hr_features)

        return {
            'canopy_raw': canopy_raw,
            'canopy': torch.sigmoid(canopy_raw),
            'boundary_raw': boundary_raw,
            'boundary': torch.sigmoid(boundary_raw),
            'sdt_raw': sdf_raw,
            'sdt': torch.tanh(sdf_raw),
        }


class TreeFlowNetHR(nn.Module):
    """
    TreeFlowNet-HR (Frontier 2026 Edition)
    Asymmetric Dual-Stream Network with FeatUp JBU Neck and Foliage Subpixel Head.
    """

    def __init__(self, fpn_out_ch: int = 256, hr_mid_ch: int = 128, pretrained_backbone: bool = False):
        super().__init__()
        weights_name = "IMAGENET1K_V1" if pretrained_backbone else None
        self.backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=weights_name,
            trainable_layers=5,
        )

        # 1. Macro Stream Neck (Stride 4)
        self.pyramid_neck = AttentionGatedPyramidNeck(in_channels=fpn_out_ch, out_channels=fpn_out_ch)

        # 2. Topological Heads (Stride 4)
        self.flow_head = DenseHead(fpn_out_ch, 2, hr_mid_ch)
        self.centroid_head = DenseHead(fpn_out_ch, 1, hr_mid_ch)
        self.sdt_macro_head = DenseHead(fpn_out_ch, 1, hr_mid_ch)

        # 3. Micro Stream: FeatUp Continuous JBU Neck (Stride 1)
        self.jbu_neck = FeatUpJBUContinuousNeck(
            in_channels=fpn_out_ch,
            guide_channels=6,
            out_channels=hr_mid_ch,
        )

        # 4. Foliage Sub-Pixel Head (Stride 1)
        self.foliage_head = FoliageSubpixelHead(in_channels=hr_mid_ch, mid_channels=64)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        B, C, H, W = images.shape

        # Backbone generates pyramid P2, P3, P4, P5
        fpn_features = self.backbone(images)

        # Macro Attention-Gated Fused Features [B, 256, H/4, W/4]
        macro_features = self.pyramid_neck(fpn_features)

        # Topological Core Heads (Stride 4)
        flow_raw = self.flow_head(macro_features)
        cent_raw = self.centroid_head(macro_features)
        sdt_macro_raw = self.sdt_macro_head(macro_features)

        # Upsample Topological Heads to native H x W
        kw = dict(size=(H, W), mode='bilinear', align_corners=False)
        flow_up = F.interpolate(flow_raw, **kw)
        cent_up = F.interpolate(cent_raw, **kw)
        sdt_macro_up = F.interpolate(sdt_macro_raw, **kw)

        # Micro Stream: Continuous FeatUp JBU Features [B, 128, H, W]
        hr_features = self.jbu_neck(macro_features, images)

        # High-Resolution Foliage Predictions [B, 1, H, W]
        foliage_out = self.foliage_head(hr_features)

        # Multi-scale SDF fusion: macro distance + subpixel detail
        fused_sdt_raw = 0.5 * sdt_macro_up + 0.5 * foliage_out['sdt_raw']

        return {
            'flow': flow_up,
            'centroid': torch.sigmoid(cent_up),
            'centroid_raw': cent_up,
            'canopy': foliage_out['canopy'],
            'canopy_raw': foliage_out['canopy_raw'],
            'boundary': foliage_out['boundary'],
            'boundary_raw': foliage_out['boundary_raw'],
            'sdt': torch.tanh(fused_sdt_raw),
            'sdt_raw': fused_sdt_raw,
            'sdt_macro_raw': sdt_macro_up,
            'sdt_hr_raw': foliage_out['sdt_raw'],
        }

    def load_from_v3(self, ckpt_path: str, device: str = 'cpu'):
        """Warm-start backbone, neck, and topological heads from TreeFlowNetV3 checkpoint."""
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        own = self.state_dict()
        loaded, skipped = 0, 0
        for k, v in state.items():
            # Map old sdt_head to sdt_macro_head if needed
            target_k = k
            if k.startswith("sdt_head."):
                target_k = k.replace("sdt_head.", "sdt_macro_head.")
            
            if target_k in own and own[target_k].shape == v.shape:
                own[target_k].copy_(v)
                loaded += 1
            else:
                skipped += 1
        self.load_state_dict(own)
        print(f"TreeFlowNet-HR warm-start from V3: loaded {loaded} tensors, skipped {skipped} new HR tensors.")
