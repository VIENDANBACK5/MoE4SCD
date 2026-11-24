"""
TreeFlowNet — Centripetal Gradient Flow + SDT + Centroid + Canopy gating.

Architecture (from research.md Addendum 12 + reverse-engineered from checkpoint):
  Backbone : ResNet-50 FPN (torchvision, pretrained=False / warm-start from G1B)
  Neck     : FPN P2 (stride 4, 256ch) → lateral conv
  Heads (decoupled, no shared bottleneck):
    flow_head    : 2ch unit vector field (vy, vx) — centripetal, diverges at boundaries
    sdt_head     : 1ch signed distance transform in [-1, 1]
    centroid_head: 1ch Gaussian heatmap for instance seeds
    canopy_head  : 1ch binary tree cover probability

Output spatial resolution: H/4 × W/4 (stride-4, then bilinear-upsample to H×W at decode time).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class DenseHead(nn.Module):
    """Refined convolutional decoder head with GroupNorm & GELU matching checkpoint."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 128, num_groups: int = 8):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(num_groups, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(num_groups, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def _modern_fpn_backbone(name: str, pretrained: bool, out_ch: int):
    """ConvNeXt-T/S or Swin-T/S (torchvision, ImageNet-1k) + FPN, returning the same dict as resnet_fpn_backbone
    (keys '0'..'3' at strides 4/8/16/32; the model reads '0'). Why: on a Mask2Former task in broadleaf crowns
    (Nakada et al. 2026, Table 3) Swin-S beat ResNet-50 by 5.3 mAP; C0-C5 all use ResNet-50."""
    from collections import OrderedDict
    from torchvision.ops import FeaturePyramidNetwork
    from torchvision.models import convnext_tiny, convnext_small, swin_t, swin_s
    from torchvision.models import ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, Swin_T_Weights, Swin_S_Weights
    table = {"convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights, [96, 192, 384, 768], "convnext"),
             "convnext_small": (convnext_small, ConvNeXt_Small_Weights, [96, 192, 384, 768], "convnext"),
             "swin_t": (swin_t, Swin_T_Weights, [96, 192, 384, 768], "swin"),
             "swin_s": (swin_s, Swin_S_Weights, [96, 192, 384, 768], "swin")}
    fn, W, chans, kind = table[name]
    body = fn(weights=W.IMAGENET1K_V1 if pretrained else None).features
    # feature stages end at these indices of `features` (ConvNeXt and Swin share the 8-block layout: stem, s1, down, s2, ...)
    stage_ends = [1, 3, 5, 7]

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.body, self.kind = body, kind
            self.fpn = FeaturePyramidNetwork(chans, out_ch)
            self.out_channels = out_ch

        def forward(self, x):
            feats, h = OrderedDict(), x
            for i, blk in enumerate(self.body):
                h = blk(h)
                if i in stage_ends:
                    feats[str(stage_ends.index(i))] = h.permute(0, 3, 1, 2) if self.kind == "swin" else h
            return self.fpn(feats)

    return Wrapper()


class TreeFlowNet(nn.Module):
    """
    Centripetal-flow instance segmentation network for exhaustive tree crown detection.

    Inputs : RGB image  [B, 3, H, W]  (H, W must be divisible by 32 for FPN)
    Outputs: dict with keys:
        'flow'     [B, 2, H, W]  — raw logits; normalize to unit vector at decode
        'sdt'      [B, 1, H, W]  — tanh-activated SDT in [-1,1]
        'centroid' [B, 1, H, W]  — sigmoid-activated Gaussian heatmap
        'canopy'   [B, 1, H, W]  — sigmoid-activated foreground prob

    Note: outputs are at the ORIGINAL image resolution (bilinear upsampled from P2).
    """

    def __init__(self, fpn_out_ch: int = 256, head_mid_ch: int = 128, pretrained_backbone: bool = False,
                 backbone: str = "resnet50"):
        super().__init__()
        weights_name = "IMAGENET1K_V1" if pretrained_backbone else None
        if backbone == "resnet50":  # default: C0-C5 and every earlier checkpoint
            self.backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=weights_name, trainable_layers=5)
        else:
            self.backbone = _modern_fpn_backbone(backbone, pretrained_backbone, fpn_out_ch)
        # Use only P2 (stride-4, highest resolution FPN feature)
        fpn_ch = fpn_out_ch  # 256 by default from torchvision FPN

        # Four independent head branches — no shared bottleneck (Finding 26 fix)
        self.flow_head     = DenseHead(fpn_ch, 2, head_mid_ch)
        self.sdt_head      = DenseHead(fpn_ch, 1, head_mid_ch)
        self.centroid_head = DenseHead(fpn_ch, 1, head_mid_ch)
        self.canopy_head   = DenseHead(fpn_ch, 1, head_mid_ch)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        B, C, H, W = images.shape

        # Backbone → FPN feature pyramid
        fpn_features = self.backbone(images)  # dict: '0','1','2','3','pool'
        p2 = fpn_features['0']  # [B, 256, H/4, W/4]

        # Heads
        flow_raw  = self.flow_head(p2)      # [B, 2, H/4, W/4]
        sdt_raw   = self.sdt_head(p2)       # [B, 1, H/4, W/4]
        cent_raw  = self.centroid_head(p2)  # [B, 1, H/4, W/4]
        can_raw   = self.canopy_head(p2)    # [B, 1, H/4, W/4]

        # Upsample all outputs to original resolution
        kw = dict(size=(H, W), mode='bilinear', align_corners=False)
        flow_up  = F.interpolate(flow_raw,  **kw)
        sdt_up   = F.interpolate(sdt_raw,   **kw)
        cent_up  = F.interpolate(cent_raw,  **kw)
        can_up   = F.interpolate(can_raw,   **kw)

        return {
            'flow':         flow_up,
            'sdt':          torch.tanh(sdt_up),        # [-1, 1]
            'sdt_raw':      sdt_up,
            'centroid':     torch.sigmoid(cent_up),    # [0, 1]
            'centroid_raw': cent_up,
            'canopy':       torch.sigmoid(can_up),     # [0, 1]
            'canopy_raw':   can_up,
        }

    def load_warm_start(self, ckpt_path: str, device: str = 'cpu'):
        """Load backbone weights from a G1B Mask-RCNN checkpoint (partial load)."""
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'model' in sd:
            sd = sd['model']
        own = self.state_dict()
        loaded, skipped = 0, 0
        for k, v in sd.items():
            # Try both with and without 'backbone.' prefix
            for candidate in (k, 'backbone.' + k, k.replace('backbone.', '')):
                if candidate in own and own[candidate].shape == v.shape:
                    own[candidate].copy_(v)
                    loaded += 1
                    break
            else:
                skipped += 1
        self.load_state_dict(own)
        print(f'  Warm-start: loaded {loaded} tensors, skipped {skipped}')


if __name__ == '__main__':
    print("Testing TreeFlowNet...")
    m = TreeFlowNet()
    x = torch.randn(1, 3, 256, 256)
    out = m(x)
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
    print("TreeFlowNet forward pass successful!")

