"""
Multi-task loss for TreeFlowNet.

Loss terms:
  L_flow     : angle loss on unit-vector field (cosine similarity)
  L_sdt      : smooth-L1 on signed distance transform
  L_centroid : focal MSE on sparse Gaussian heatmap
  L_canopy   : BCE with logits on binary tree cover
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Individual loss functions
# ─────────────────────────────────────────────────────────────────────────────

def flow_loss(
    pred_flow: torch.Tensor,  # [B, 2, H, W]  raw logits
    gt_flow:   torch.Tensor,  # [B, 2, H, W]  unit vectors
    fg_mask:   torch.Tensor,  # [B, 1, H, W]  foreground mask (float 0/1)
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Angular loss = 1 - cosine_similarity between predicted and GT flow vectors.
    Only computed on foreground pixels (inside a crown).
    Safe norm with eps placed inside sqrt to avoid NaN backward gradients at 0.
    """
    pred_sq   = (pred_flow ** 2).sum(dim=1, keepdim=True)
    gt_sq     = (gt_flow ** 2).sum(dim=1, keepdim=True)
    pred_norm = pred_flow / torch.sqrt(pred_sq + eps)
    gt_norm   = gt_flow   / torch.sqrt(gt_sq + eps)

    cos_sim  = (pred_norm * gt_norm).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    loss_map = 1.0 - cos_sim  # [B, 1, H, W]

    # Only penalize foreground pixels; safe mean
    masked = loss_map * fg_mask
    denom = fg_mask.sum().clamp(min=1.0)
    return masked.sum() / denom


def sdt_loss(
    pred_sdt: torch.Tensor,  # [B, 1, H, W]  raw (before tanh)
    gt_sdt:   torch.Tensor,  # [B, 1, H, W]  in [-1, 1]
) -> torch.Tensor:
    """Smooth-L1 on tanh-activated SDT."""
    return F.smooth_l1_loss(torch.tanh(pred_sdt), gt_sdt)


def centroid_focal_loss(
    pred:    torch.Tensor,  # [B, 1, H, W]  sigmoid or raw
    gt_heat: torch.Tensor,  # [B, 1, H, W]  Gaussian heatmap in [0,1]
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Standard CenterNet Modified Focal Loss for Gaussian heatmaps.
    Clamped probabilities prevent log(0) and log(1) infinities.
    """
    p = pred if (pred.min() >= 0.0 and pred.max() <= 1.0) else torch.sigmoid(pred)
    p = p.clamp(min=eps, max=1.0 - eps)

    pos_mask = (gt_heat >= 0.99).float()
    neg_mask = 1.0 - pos_mask

    pos_loss = torch.log(p) * torch.pow(1.0 - p, alpha) * pos_mask
    neg_loss = torch.log(1.0 - p) * torch.pow(p, alpha) * torch.pow(1.0 - gt_heat, beta) * neg_mask

    num_pos = pos_mask.sum().clamp(min=1.0)
    loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
    return loss


def canopy_loss(
    pred:    torch.Tensor,  # [B, 1, H, W]  raw logits
    gt_mask: torch.Tensor,  # [B, 1, H, W]  binary {0,1}
    pos_weight: float = 2.0,
) -> torch.Tensor:
    pw = torch.tensor([pos_weight], device=pred.device)
    return F.binary_cross_entropy_with_logits(pred, gt_mask, pos_weight=pw)


# ─────────────────────────────────────────────────────────────────────────────
# Combined loss
# ─────────────────────────────────────────────────────────────────────────────

class TreeFlowLoss(nn.Module):
    """
    Balanced multi-task loss for TreeFlowNet across 4 decoupled prediction heads:
      1. L_flow (unit vector field)
      2. L_sdt (signed distance transform)
      3. L_centroid (modified CenterNet focal heatmap)
      4. L_canopy (tree cover binary cross-entropy)
    """

    def __init__(
        self,
        w_flow:     float = 1.0,
        w_sdt:      float = 1.0,
        w_centroid: float = 1.0,
        w_canopy:   float = 1.0,
    ):
        super().__init__()
        self.w_flow     = w_flow
        self.w_sdt      = w_sdt
        self.w_centroid = w_centroid
        self.w_canopy   = w_canopy

    def forward(
        self,
        outputs: dict,   # from TreeFlowNet.forward()
        targets: dict,   # from dataset
    ) -> dict[str, torch.Tensor]:
        """
        targets keys:
            'flow'      [B, 2, H, W]  unit vector field
            'sdt'       [B, 1, H, W]  signed distance transform
            'centroid'  [B, 1, H, W]  Gaussian centroid heatmap
            'canopy'    [B, 1, H, W]  binary canopy mask
            'fg_mask'   [B, 1, H, W]  foreground (crown interior) pixels

        Returns dict with 'total' and per-task losses.
        """
        # Cast targets to float32
        gt_flow    = targets['flow'].float()
        gt_sdt     = targets['sdt'].float()
        gt_centroid= targets['centroid'].float()
        gt_canopy  = targets['canopy'].float()
        fg_mask    = targets.get('fg_mask', (gt_sdt > -0.5).float())
        canopy_raw = outputs.get('canopy_raw', outputs['canopy'])
        sdt_pred   = outputs['sdt']
        cent_pred  = outputs['centroid']

        l_flow = flow_loss(outputs['flow'], gt_flow, fg_mask)
        l_sdt  = F.smooth_l1_loss(sdt_pred, gt_sdt)
        l_cent = centroid_focal_loss(cent_pred, gt_centroid)
        l_can  = canopy_loss(canopy_raw, gt_canopy)

        total = (
            self.w_flow     * l_flow +
            self.w_sdt      * l_sdt  +
            self.w_centroid * l_cent +
            self.w_canopy   * l_can
        )

        return {
            'total':    total,
            'flow':     l_flow,
            'sdt':      l_sdt,
            'centroid': l_cent,
            'canopy':   l_can,
        }
