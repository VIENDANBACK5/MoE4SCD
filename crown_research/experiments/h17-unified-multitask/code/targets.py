"""Target generation for H17: Unified Multi-Task Architecture.

Generates 4 high-fidelity supervision targets:
1. surface: Continuous EDT potential U in [0, 1] per instance.
2. boundary: Calibrated sharp 3.5px contact seam barrier B in [0, 1].
3. canopy: Binary semantic canopy support gate C in {0, 1}.
4. adaptive_apex: Scale-adaptive Gaussian centroid peak with unit height (1.0)
   and sigma_i = max(3.5, 0.18 * sqrt(Area_i / pi)).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch


def compute_h17_targets(
    instance_label: np.ndarray,
    seam_dilate_r: int = 3,
) -> dict[str, torch.Tensor]:
    """Computes 4 targets from instance label map.

    Args:
        instance_label: (H, W) int32 array where 0 is background, 1..N are instances.
        seam_dilate_r: Radius for boundary dilation (3px -> ~7px contact band).

    Returns:
        Dictionary of FloatTensors with shape (1, H, W).
    """
    H, W = instance_label.shape
    surface = np.zeros((H, W), dtype=np.float32)
    boundary = np.zeros((H, W), dtype=np.float32)
    canopy = (instance_label > 0).astype(np.float32)
    adaptive_apex = np.zeros((H, W), dtype=np.float32)

    inst_ids = np.unique(instance_label)
    inst_ids = inst_ids[inst_ids > 0]

    if len(inst_ids) == 0:
        return {
            "surface": torch.from_numpy(surface).unsqueeze(0),
            "boundary": torch.from_numpy(boundary).unsqueeze(0),
            "canopy": torch.from_numpy(canopy).unsqueeze(0),
            "adaptive_apex": torch.from_numpy(adaptive_apex).unsqueeze(0),
        }

    # 1. Morphological boundary gradient for contact seams
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(instance_label.astype(np.float32), cv2.MORPH_GRADIENT, k3)
    raw_seams = (grad > 0) & (instance_label > 0)

    # Dilate boundary band slightly for clean supervision
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * seam_dilate_r + 1, 2 * seam_dilate_r + 1))
    dilated_seams = cv2.dilate(raw_seams.astype(np.uint8), kd)
    # Gaussian blur for soft probability target
    boundary = cv2.GaussianBlur(dilated_seams.astype(np.float32), (5, 5), 1.0)
    boundary = np.clip(boundary, 0.0, 1.0)

    # Grid coordinates for fast Gaussian apex generation
    y_coords, x_coords = np.mgrid[0:H, 0:W]

    # 2. Surface potential & Scale-Adaptive Gaussian Apex
    for inst_id in inst_ids:
        mask = (instance_label == inst_id).astype(np.uint8)
        area = float(np.sum(mask))
        if area < 4:
            continue

        # Distance transform
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        max_d = float(dist.max())
        if max_d > 0:
            surface += (dist / max_d)

        # Centroid location
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            py, px = np.where(mask)
            cy, cx = float(np.mean(py)), float(np.mean(px))

        # Scale-adaptive sigma: sigma >= 3.5px, scales smoothly with crown diameter
        eq_radius = np.sqrt(area / np.pi)
        sigma = max(3.5, 0.18 * eq_radius)
        two_sigma_sq = 2.0 * (sigma ** 2)

        # Local bounding box around centroid to speed up computation
        box_r = int(np.ceil(3.5 * sigma))
        y0, y1 = max(0, int(cy - box_r)), min(H, int(cy + box_r + 1))
        x0, x1 = max(0, int(cx - box_r)), min(W, int(cx + box_r + 1))

        if y1 > y0 and x1 > x0:
            sub_y = y_coords[y0:y1, x0:x1]
            sub_x = x_coords[y0:y1, x0:x1]
            g = np.exp(-((sub_y - cy) ** 2 + (sub_x - cx) ** 2) / two_sigma_sq)
            adaptive_apex[y0:y1, x0:x1] = np.maximum(adaptive_apex[y0:y1, x0:x1], g)

    surface = np.clip(surface, 0.0, 1.0)
    adaptive_apex = np.clip(adaptive_apex, 0.0, 1.0)

    return {
        "surface": torch.from_numpy(surface).unsqueeze(0),
        "boundary": torch.from_numpy(boundary).unsqueeze(0),
        "canopy": torch.from_numpy(canopy).unsqueeze(0),
        "adaptive_apex": torch.from_numpy(adaptive_apex).unsqueeze(0),
    }
