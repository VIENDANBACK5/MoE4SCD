"""
TreeFlowNet-HR Unified Boundary-Aware Multi-Task Loss (Frontier 2026 Edition).

Interdisciplinary Integration of:
1. Boundary-IoU Loss (CVPR 2021): Focuses gradients strictly on the 3-5px outer boundary band.
2. Kervadec Distance Penalty (MedIA / MICCAI): Penalizes false positives in ground shadow proportional
   to physical Euclidean distance from true leaf perimeter, creating an elastic pulling force.
3. Modified Hausdorff Distance (MHD) Loss (BAISeg ISBI 2026): Differentiable surface alignment for continuous SDFs.
4. Sobolev Edge-Aware Flow Loss (Omnipose & HoVer-Net): Flow vector cosine alignment weighted by optical gradients.
5. CenterNet Focal Apex Loss & Lovász-Softmax Canopy Coverage.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def differentiable_boundary_band(mask: torch.Tensor, radius: int = 3) -> torch.Tensor:
    """
    Computes a differentiable boundary band using soft max/min morphological pooling.
    mask: [B, 1, H, W] in [0, 1]
    radius: half-width of boundary band in pixels
    """
    ksize = 2 * radius + 1
    # Dilation: max_pool2d
    dilated = F.max_pool2d(mask, kernel_size=ksize, stride=1, padding=radius)
    # Erosion: -max_pool2d(-mask) = 1 - max_pool2d(1 - mask)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=ksize, stride=1, padding=radius)
    boundary = (dilated - eroded).clamp(min=0.0, max=1.0)
    return boundary


def boundary_iou_loss(
    pred_prob: torch.Tensor,   # [B, 1, H, W] in [0, 1]
    gt_mask: torch.Tensor,     # [B, 1, H, W] float 0/1
    radius: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Boundary-IoU Loss (Cheng et al., CVPR 2021).
    Computes IoU strictly over the thin boundary band d = radius.
    """
    pred_band = differentiable_boundary_band(pred_prob, radius=radius)
    gt_band = differentiable_boundary_band(gt_mask, radius=radius)

    intersection = (pred_band * gt_band).sum(dim=(-2, -1))
    union = pred_band.sum(dim=(-2, -1)) + gt_band.sum(dim=(-2, -1)) - intersection

    boundary_iou = (intersection + eps) / (union + eps)
    return (1.0 - boundary_iou).mean()


def kervadec_distance_loss(
    pred_prob: torch.Tensor,     # [B, 1, H, W] in [0, 1]
    dist_map: torch.Tensor,      # [B, 1, H, W] signed Euclidean distance to GT boundary (normalized)
) -> torch.Tensor:
    """
    Kervadec Boundary Loss (Kervadec et al., MedIA).
    Penalizes predictions based on signed/unsigned distance to true boundary:
    L_dist = (1 / |Omega|) * sum( dist_map * pred_prob )
    Positive outside GT canopy -> penalizes shadow over-segmentation.
    """
    loss = (dist_map * pred_prob).mean()
    return loss


def modified_hausdorff_sdt_loss(
    pred_sdt_raw: torch.Tensor,  # [B, 1, H, W] raw logits
    gt_sdt: torch.Tensor,        # [B, 1, H, W] in [-1, 1]
    sigma: float = 0.15,
) -> torch.Tensor:
    """
    Modified Hausdorff Distance (MHD) inspired Continuous SDF loss (BAISeg ISBI 2026).
    Combines Smooth-L1 with exponential surface zero-crossing penalty.
    """
    pred_sdt = torch.tanh(pred_sdt_raw)
    l1_diff = F.smooth_l1_loss(pred_sdt, gt_sdt, reduction='none')

    # Heavily penalize zero-level set transition (boundary where SDF ~ 0)
    boundary_weight = 1.0 + 4.0 * torch.exp(-((gt_sdt / sigma) ** 2))
    weighted = l1_diff * boundary_weight
    return weighted.mean()


def flow_loss_edge_aware(
    pred_flow: torch.Tensor,   # [B, 2, H, W]
    gt_flow: torch.Tensor,     # [B, 2, H, W]
    fg_mask: torch.Tensor,     # [B, 1, H, W]
    opt_grad: torch.Tensor | None = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Angular cosine flow loss modulated by native optical leaf edges.
    """
    pred_sq = (pred_flow ** 2).sum(dim=1, keepdim=True)
    gt_sq = (gt_flow ** 2).sum(dim=1, keepdim=True)
    pred_norm = pred_flow / torch.sqrt(pred_sq + eps)
    gt_norm = gt_flow / torch.sqrt(gt_sq + eps)

    cos_sim = (pred_norm * gt_norm).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    loss_map = 1.0 - cos_sim

    if opt_grad is not None:
        loss_map = loss_map * (1.0 + 2.0 * opt_grad.clamp(0.0, 1.0))

    masked = loss_map * fg_mask
    denom = fg_mask.sum().clamp(min=1.0)
    return masked.sum() / denom


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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
    B = pred_logits.shape[0]
    total_loss = 0.0
    for b in range(B):
        p_flat = pred_logits[b, 0].view(-1)
        l_flat = gt_mask[b, 0].view(-1)
        total_loss += lovasz_hinge_flat(p_flat, l_flat)
    return total_loss / max(1, B)


def centroid_focal_loss(pred_logits: torch.Tensor, gt_centroid: torch.Tensor) -> torch.Tensor:
    pred_prob = torch.sigmoid(pred_logits).clamp(1e-6, 1.0 - 1e-6)
    pos_mask = (gt_centroid == 1.0).float()
    neg_mask = (gt_centroid < 1.0).float()
    neg_weights = torch.pow(1.0 - gt_centroid, 4.0)

    pos_loss = torch.log(pred_prob) * torch.pow(1.0 - pred_prob, 2.0) * pos_mask
    neg_loss = torch.log(1.0 - pred_prob) * torch.pow(pred_prob, 2.0) * neg_weights * neg_mask

    num_pos = pos_mask.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / num_pos


class TreeFlowLossHR(nn.Module):
    """
    Unified High-Resolution Loss for TreeFlowNet-HR:
    - Topological separation (Flow + Centroid + Continuous SDF)
    - High-Resolution Boundary Refinement (Boundary-IoU + Kervadec Distance + Boundary BCE)
    - Global coverage (Lovász + BCE)
    """

    def __init__(
        self,
        w_flow: float = 1.0,
        w_sdt: float = 1.0,
        w_cent: float = 1.0,
        w_lovasz: float = 0.4,
        w_boundary_iou: float = 0.5,
        w_kervadec: float = 0.3,
        w_boundary_bce: float = 0.3,
    ):
        super().__init__()
        self.w_flow = w_flow
        self.w_sdt = w_sdt
        self.w_cent = w_cent
        self.w_lovasz = w_lovasz
        self.w_boundary_iou = w_boundary_iou
        self.w_kervadec = w_kervadec
        self.w_boundary_bce = w_boundary_bce

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        opt_grad: torch.Tensor | None = None,
        dist_map: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # 1. Flow angular loss
        l_flow = flow_loss_edge_aware(
            outputs['flow'],
            targets['flow'],
            targets['fg_mask'],
            opt_grad=opt_grad,
        )

        # 2. Continuous SDF modified Hausdorff loss
        l_sdt = modified_hausdorff_sdt_loss(
            outputs['sdt_raw'],
            targets['sdt'],
        )

        # 3. Centroid focal loss
        l_cent = centroid_focal_loss(
            outputs['centroid_raw'],
            targets['centroid'],
        )

        # 4. Global canopy Lovász loss
        l_lov = lovasz_canopy_loss(
            outputs['canopy_raw'],
            targets['canopy'],
        )

        # 5. High-Resolution Boundary-IoU Loss
        l_b_iou = boundary_iou_loss(
            outputs['canopy'],
            targets['canopy'],
            radius=3,
        )

        # 6. Kervadec Distance Penalty (elastic pull from shadows)
        if dist_map is not None:
            l_kervadec = kervadec_distance_loss(outputs['canopy'], dist_map)
        else:
            l_kervadec = torch.tensor(0.0, device=outputs['canopy'].device)

        # 7. Sub-pixel Boundary Band Classification
        gt_boundary = differentiable_boundary_band(targets['canopy'], radius=2)
        l_b_bce = F.binary_cross_entropy_with_logits(
            outputs['boundary_raw'],
            gt_boundary,
        )

        total = (
            self.w_flow * l_flow
            + self.w_sdt * l_sdt
            + self.w_cent * l_cent
            + self.w_lovasz * l_lov
            + self.w_boundary_iou * l_b_iou
            + self.w_kervadec * l_kervadec
            + self.w_boundary_bce * l_b_bce
        )

        return {
            'total': total,
            'flow': l_flow,
            'sdt': l_sdt,
            'centroid': l_cent,
            'lovasz': l_lov,
            'boundary_iou': l_b_iou,
            'kervadec': l_kervadec,
            'boundary_bce': l_b_bce,
        }
