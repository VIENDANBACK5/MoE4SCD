"""Loss Functions for Neuro-Algorithmic Deadwood Segmentation.

Combines:
  1. Focal Loss for extreme deadwood foreground class imbalance (<2% of pixels).
  2. Canopy Segmentation BCE loss.
  3. Directional eps-Kernel Support Extent Loss (Smooth L1 along canonical directions).
  4. Spectral Modularity Maximization Loss.
  5. Bellman-Ford Sparsity Regularization Loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


from torchvision.ops import sigmoid_focal_loss


class NeuroDeadwoodLoss(nn.Module):
    """Composite Neuro-Algorithmic Loss Function."""
    def __init__(
        self,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        weight_canopy: float = 0.5,
        weight_eps_kernel: float = 1.0,
        weight_modularity: float = 0.2,
        weight_sparse: float = 0.05,
    ):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.weight_canopy = weight_canopy
        self.weight_eps_kernel = weight_eps_kernel
        self.weight_modularity = weight_modularity
        self.weight_sparse = weight_sparse

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        outputs: model prediction dictionary
        targets: target dictionary with 'probability', 'canopy', 'rays' (or gt support)
        """
        loss_dict = {}
        
        # 1. Focal Loss on Deadwood Probability (using raw logits for 100% numerical stability)
        gt_prob = targets["probability"].float() # (B, 1, H, W)
        if "prob_logits" in outputs:
            prob_logits = outputs["prob_logits"].float()
            l_focal = sigmoid_focal_loss(prob_logits, gt_prob, alpha=self.focal_alpha, gamma=self.focal_gamma, reduction="mean")
        else:
            pred_prob = outputs["probability"].float().clamp(min=1e-6, max=1.0 - 1e-6)
            ce = -(gt_prob * torch.log(pred_prob) + (1.0 - gt_prob) * torch.log(1.0 - pred_prob))
            p_t = pred_prob * gt_prob + (1.0 - pred_prob) * (1.0 - gt_prob)
            alpha_t = self.focal_alpha * gt_prob + (1.0 - self.focal_alpha) * (1.0 - gt_prob)
            l_focal = (alpha_t * ((1.0 - p_t) ** self.focal_gamma) * ce).mean()

        l_focal = torch.nan_to_num(l_focal, nan=0.0)
        loss_dict["loss_focal"] = l_focal
        total_loss = l_focal
        
        # 2. Canopy Loss (using raw logits for numerical stability)
        if "canopy" in targets:
            gt_canopy = targets["canopy"].float()
            if "canopy_logits" in outputs:
                l_canopy = F.binary_cross_entropy_with_logits(outputs["canopy_logits"].float(), gt_canopy)
            elif "canopy" in outputs:
                pred_canopy = outputs["canopy"].float().clamp(min=1e-6, max=1.0 - 1e-6)
                l_canopy = -(gt_canopy * torch.log(pred_canopy) + (1.0 - gt_canopy) * torch.log(1.0 - pred_canopy)).mean()
            else:
                l_canopy = torch.tensor(0.0, device=gt_prob.device)
                
            l_canopy = torch.nan_to_num(l_canopy, nan=0.0)
            loss_dict["loss_canopy"] = l_canopy
            total_loss = total_loss + self.weight_canopy * l_canopy
            
        # 3. Directional eps-Kernel Extent Loss
        if "extent_radii" in outputs and "rays" in targets:
            pred_radii = outputs["extent_radii"].float() # (B, K) in [-1, 1] scale
            gt_rays = targets["rays"].float() # (B, K, H, W) or (B, K) in pixel scale
            
            # If gt_rays is dense (B, K, H, W), sample average along foreground mask
            if gt_rays.ndim == 4:
                fg_mask = (gt_prob > 0.05).float()
                sum_fg = fg_mask.sum(dim=[2, 3]).clamp(min=1.0) # (B, 1)
                # Normalize pixel radius (0-256 px) to coordinate space scale (divided by 256.0)
                gt_extent = (gt_rays * fg_mask).sum(dim=[2, 3]) / (sum_fg * 256.0) # (B, K)
                has_fg = (fg_mask.sum(dim=[1, 2, 3]) > 0).float() # (B,)
                if has_fg.sum() > 0:
                    per_sample_loss = F.smooth_l1_loss(pred_radii, gt_extent, reduction="none").mean(dim=-1) # (B,)
                    l_extent = (per_sample_loss * has_fg).sum() / has_fg.sum().clamp(min=1.0)
                else:
                    l_extent = torch.tensor(0.0, device=pred_radii.device, requires_grad=True)
            else:
                gt_extent = gt_rays / 256.0
                l_extent = F.smooth_l1_loss(pred_radii, gt_extent)
                
            loss_dict["loss_extent"] = torch.nan_to_num(l_extent, nan=0.0)
            total_loss = total_loss + self.weight_eps_kernel * loss_dict["loss_extent"]
            
        # 4. Spectral Modularity Loss
        if "loss_modularity" in outputs:
            l_mod = outputs["loss_modularity"]
            loss_dict["loss_modularity"] = l_mod
            total_loss = total_loss + self.weight_modularity * l_mod
            
        # 5. Bellman-Ford Sparsity Regularization Loss
        if "loss_sparse" in outputs:
            l_sp = outputs["loss_sparse"]
            loss_dict["loss_sparse"] = l_sp
            total_loss = total_loss + self.weight_sparse * l_sp
            
        loss_dict["total_loss"] = total_loss
        return loss_dict
