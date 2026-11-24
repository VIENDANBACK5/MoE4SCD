"""
H24 Boundary-Aware Differentiable Multi-Task Loss for TreeFlowNetV2.

Loss Innovations over V1:
1. Boundary-Weighted SDT Loss: 4x penalty on errors near the zero level-set (|SDT| < 0.2),
   forcing extreme precision directly at the crown perimeter.
2. Edge-Modulated Flow Alignment Loss: flow cosine similarity weighted by optical
   gradient magnitude (Sobolev-inspired), penalizing vectors that cross optical leaf edges.
3. Lovász-Softmax Loss on canopy coverage: optimizes Jaccard IoU directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def flow_loss_edge_aware(
    pred_flow: torch.Tensor,   # [B, 2, H, W] raw logits
    gt_flow:   torch.Tensor,   # [B, 2, H, W] unit vectors
    fg_mask:   torch.Tensor,   # [B, 1, H, W] float 0/1
    opt_grad:  torch.Tensor | None = None,  # [B, 1, H, W] normalized optical gradient [0, 1]
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Angular flow loss modulated by optical leaf edge gradients.
    """
    pred_sq   = (pred_flow ** 2).sum(dim=1, keepdim=True)
    gt_sq     = (gt_flow ** 2).sum(dim=1, keepdim=True)
    pred_norm = pred_flow / torch.sqrt(pred_sq + eps)
    gt_norm   = gt_flow   / torch.sqrt(gt_sq + eps)

    cos_sim  = (pred_norm * gt_norm).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    loss_map = 1.0 - cos_sim  # [B, 1, H, W]

    # Modulate by optical edge magnitude if provided
    if opt_grad is not None:
        edge_weight = 1.0 + 2.0 * opt_grad.clamp(0.0, 1.0)
        loss_map = loss_map * edge_weight

    masked = loss_map * fg_mask
    denom = fg_mask.sum().clamp(min=1.0)
    return masked.sum() / denom


def boundary_weighted_sdt_loss(
    pred_sdt: torch.Tensor,  # [B, 1, H, W] raw (before tanh)
    gt_sdt:   torch.Tensor,  # [B, 1, H, W] in [-1, 1]
    sigma_b: float = 0.20,
    boundary_multiplier: float = 3.0,
) -> torch.Tensor:
    """
    Smooth-L1 loss weighted heavily at the crown boundary zero-crossing (|SDT| < 0.2).
    """
    pred_act = torch.tanh(pred_sdt)
    raw_l1 = F.smooth_l1_loss(pred_act, gt_sdt, reduction='none')

    # Gaussian weighting centered on the zero level-set
    boundary_weight = 1.0 + boundary_multiplier * torch.exp(-((gt_sdt / sigma_b) ** 2))
    weighted_loss = raw_l1 * boundary_weight
    return weighted_loss.mean()


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Computes gradient of the Lovász extension for Jaccard loss."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary Lovász hinge loss for flattened predictions."""
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def lovasz_canopy_loss(pred_logits: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """Lovász hinge loss on canopy prediction [B, 1, H, W]."""
    B = pred_logits.shape[0]
    total_loss = 0.0
    for b in range(B):
        p_flat = pred_logits[b, 0].view(-1)
        l_flat = gt_mask[b, 0].view(-1)
        total_loss += lovasz_hinge_flat(p_flat, l_flat)
    return total_loss / max(1, B)


class TreeFlowLossV2(nn.Module):
    """
    H24 Boundary-Aware Multi-Task Loss:
    1. Edge-aware flow angular loss (Sobolev gradient alignment)
    2. Boundary-weighted SDT loss (focused on perimeter zero level-set)
    3. CenterNet focal centroid loss
    4. Combined BCE + Lovász canopy coverage loss
    """

    def __init__(
        self,
        w_flow: float = 1.0,
        w_sdt: float = 1.0,
        w_cent: float = 1.0,
        w_canopy_bce: float = 0.8,
        w_lovasz: float = 0.4,
    ):
        super().__init__()
        self.w_flow = w_flow
        self.w_sdt = w_sdt
        self.w_cent = w_cent
        self.w_canopy_bce = w_canopy_bce
        self.w_lovasz = w_lovasz

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        opt_grad: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        l_flow = flow_loss_edge_aware(
            outputs['flow'],
            targets['flow'],
            targets['fg_mask'],
            opt_grad=opt_grad,
        )

        l_sdt = boundary_weighted_sdt_loss(
            outputs['sdt_raw'],
            targets['sdt'],
        )

        from crown_segmentation_research.methods.tree_flow.loss import centroid_focal_loss, canopy_loss
        l_cent = centroid_focal_loss(
            outputs['centroid_raw'],
            targets['centroid'],
        )

        l_bce = canopy_loss(
            outputs['canopy_raw'],
            targets['canopy'],
        )

        l_lov = lovasz_canopy_loss(
            outputs['canopy_raw'],
            targets['canopy'],
        )

        total = (
            self.w_flow * l_flow
            + self.w_sdt * l_sdt
            + self.w_cent * l_cent
            + self.w_canopy_bce * l_bce
            + self.w_lovasz * l_lov
        )

        return {
            'total': total,
            'flow': l_flow,
            'sdt': l_sdt,
            'centroid': l_cent,
            'canopy_bce': l_bce,
            'lovasz': l_lov,
        }
