"""Sharp-Ridge Target Generation for CanopyWatershedNet (Stage 2).

Generates:
1. Sharp Unimodal Potential Surface U*(y, x) in [0, 1] with zeroed contact seams.
2. Inter-Crown Seam Boundary Barrier B*(y, x) in [0, 1] marking touching interfaces.
3. Sharp Apex Centroid Heatmap A*(y, x) in [0, 1] (sigma = 1.5px).
4. Binary Canopy Mask M*(y, x) in {0, 1}.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def compute_sharp_canopy_targets(
    instance_label: np.ndarray,
    boundary_dilation: int = 1,
    apex_sigma: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes sharp targets from an (H, W) integer instance label map.

    Returns:
        surface_target: (1, H, W) float32 in [0, 1] with sharp valleys at seams.
        boundary_target: (1, H, W) float32 in [0, 1].
        apex_target: (1, H, W) float32 in [0, 1].
        canopy_target: (1, H, W) float32 in {0, 1}.
    """
    height, width = instance_label.shape
    surface_target = np.zeros((height, width), dtype=np.float32)
    apex_target = np.zeros((height, width), dtype=np.float32)
    canopy_target = (instance_label > 0).astype(np.float32)

    unique_labels = np.unique(instance_label)
    unique_labels = unique_labels[unique_labels != 0]

    if len(unique_labels) == 0:
        boundary_target = np.zeros((height, width), dtype=np.float32)
        return (
            surface_target[np.newaxis, ...],
            boundary_target[np.newaxis, ...],
            apex_target[np.newaxis, ...],
            canopy_target[np.newaxis, ...],
        )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (boundary_dilation * 2 + 1, boundary_dilation * 2 + 1),
    )
    dilated_accumulator = np.zeros((height, width), dtype=np.uint8)

    for label in unique_labels:
        mask = (instance_label == label)
        mask_u8 = mask.astype(np.uint8)

        dilated = cv2.dilate(mask_u8, kernel, iterations=1)
        dilated_accumulator += dilated

        mask_edt = ndimage.distance_transform_edt(mask)
        max_dist = float(mask_edt.max())
        if max_dist > 0:
            u_inst = (mask_edt / max_dist).astype(np.float32)
            np.maximum(surface_target, u_inst, out=surface_target)

            # Apex centroid
            cy, cx = ndimage.center_of_mass(mask)
            if np.isfinite(cy) and np.isfinite(cx):
                cy_i, cx_i = int(round(cy)), int(round(cx))
                cy_i = np.clip(cy_i, 0, height - 1)
                cx_i = np.clip(cx_i, 0, width - 1)
                
                # Draw Gaussian peak around centroid
                y_rad = int(3 * apex_sigma + 1)
                y0, y1 = max(0, cy_i - y_rad), min(height, cy_i + y_rad + 1)
                x0, x1 = max(0, cx_i - y_rad), min(width, cx_i + y_rad + 1)
                
                ys = np.arange(y0, y1)[:, None]
                xs = np.arange(x0, x1)[None, :]
                g = np.exp(-((ys - cy)**2 + (xs - cx)**2) / (2 * apex_sigma**2))
                apex_target[y0:y1, x0:x1] = np.maximum(apex_target[y0:y1, x0:x1], g)

    # Touching seam boundary where dilated regions overlap
    touching_barrier = (dilated_accumulator >= 2) & (instance_label > 0)

    # Outer perimeter of tree canopy
    fg_eroded = cv2.erode(
        (instance_label > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    outer_boundary = ((instance_label > 0) & (fg_eroded == 0))
    combined_barrier = (touching_barrier | outer_boundary).astype(np.float32)

    # ZERO OUT surface potential directly on touching seams to form sharp valleys
    surface_target[touching_barrier] = 0.0

    boundary_target = cv2.GaussianBlur(combined_barrier, (3, 3), 0.8)
    boundary_target = np.clip(boundary_target, 0.0, 1.0)

    return (
        surface_target[np.newaxis, ...],
        boundary_target[np.newaxis, ...],
        apex_target[np.newaxis, ...],
        canopy_target[np.newaxis, ...],
    )
