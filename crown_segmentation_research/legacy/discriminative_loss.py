"""Discriminative instance-embedding loss (De Brabandere, Neven, Van Gool
2017, arXiv:1708.02551): pull each instance's pixel embeddings toward their
own mean (variance term), and push different instances' means apart
(distance term), so that nearby, visually-similar instances end up
separated in embedding space -- the explicit inter-instance repulsion
signal star_convex_v6_boundary_weight_result.md found missing from a plain
(or reweighted) probability map.

Standalone module (not folded into train_star_convex.py) since the loss
itself is a nontrivial, independently-testable piece of logic operating on
a different kind of target (per-pixel instance identity) than the other
losses in this project.

Fully vectorized (scatter-based cluster means, no per-instance Python loop):
a first version looped over instances in Python, which measured at ~0.56s
per call for a dense 85-instance image and left the GPU at ~3% utilization
during training (confirmed directly) -- the loop's per-instance kernel
launches dominated wall time far more than the actual math. Rewritten here
to use index_add_ scatter operations, which cost a small constant number of
GPU kernel launches regardless of instance count.
"""

from __future__ import annotations

import torch


def discriminative_loss(
    embedding: torch.Tensor,
    instance_label: torch.Tensor,
    delta_v: float = 0.5,
    delta_d: float = 1.5,
    var_weight: float = 1.0,
    dist_weight: float = 1.0,
    reg_weight: float = 0.001,
) -> dict[str, torch.Tensor]:
    """embedding: (D, H, W) float. instance_label: (H, W) long, 0=background,
    1..C=instance ids (not required to be contiguous). Background pixels
    never participate (embeddings are only meaningful where there is an
    actual instance to identify).

    delta_v=0.5, delta_d=1.5 are the original paper's own defaults (chosen
    there so that, once both hinge terms are satisfied, instance clusters
    of radius delta_v cannot overlap since their centers are >= delta_d
    apart) -- kept as-is for this first screen rather than re-tuned.
    """
    device = embedding.device
    dim = embedding.shape[0]
    foreground = instance_label > 0
    if not foreground.any():
        zero = torch.tensor(0.0, device=device)
        return {"total": zero, "var_loss": zero, "dist_loss": zero, "reg_loss": zero}

    labels_present, remapped = torch.unique(instance_label[foreground], return_inverse=True)
    n_instances = len(labels_present)
    embedding_fg = embedding.permute(1, 2, 0)[foreground]  # (N_fg, D)

    counts = torch.zeros(n_instances, device=device).index_add_(0, remapped, torch.ones_like(remapped, dtype=torch.float32))
    sums = torch.zeros(n_instances, dim, device=device).index_add_(0, remapped, embedding_fg)
    means = sums / counts.unsqueeze(1)  # (C, D)

    distance_to_own_mean = torch.norm(embedding_fg - means[remapped], dim=1)  # (N_fg,)
    per_pixel_var = torch.clamp(distance_to_own_mean - delta_v, min=0.0).pow(2)
    var_sum_per_instance = torch.zeros(n_instances, device=device).index_add_(0, remapped, per_pixel_var)
    var_loss = (var_sum_per_instance / counts).mean()

    reg_loss = torch.norm(means, dim=1).mean()

    if n_instances > 1:
        diff = means.unsqueeze(0) - means.unsqueeze(1)  # (C, C, D)
        pairwise_dist = torch.norm(diff, dim=2)  # (C, C)
        off_diagonal = ~torch.eye(n_instances, dtype=torch.bool, device=device)
        dist_loss = torch.clamp(delta_d - pairwise_dist[off_diagonal], min=0.0).pow(2).mean()
    else:
        dist_loss = torch.tensor(0.0, device=device)

    total = var_weight * var_loss + dist_weight * dist_loss + reg_weight * reg_loss
    return {"total": total, "var_loss": var_loss, "dist_loss": dist_loss, "reg_loss": reg_loss}
