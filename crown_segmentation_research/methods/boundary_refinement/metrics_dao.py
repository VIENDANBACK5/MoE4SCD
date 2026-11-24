"""Geometric Boundary Quality Metrics based on Dao et al. (ISPRS 2021) and SOTA Boundary-IoU.

Scientific References:
1. Phuong D. Dao, Yuhong He, et al., "Improving hyperspectral image segmentation by applying
   inverse noise weighting and outlier removal for optimal scale selection",
   ISPRS Journal of Photogrammetry and Remote Sensing, 171 (2021) 348-366.
2. Cheng et al., "Boundary IoU: Improving Object-Centric Image Segmentation Evaluation", CVPR 2021.
3. ISPRS Contest on Individual Tree Crown (ITC) Segmentation (2024-2026).
"""
from __future__ import annotations

import cv2
import numpy as np


def compute_dao_segmentation_indices(
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    min_overlap: float = 0.05,
) -> dict[str, float]:
    """Computes Over-Segmentation (OS), Under-Segmentation (US), and Euclidean Distance (ED).

    Formulation from Dao et al. (ISPRS 2021):
        OS_ij = Area(R_i ∩ S_j) / Area(R_i)
        US_ij = Area(R_i ∩ S_j) / Area(S_j)
        ED_ij = sqrt((1 - OS_ij)^2 + (1 - US_ij)^2)

    A lower ED (closer to 0.0) indicates a geometrically superior segmentation
    where both over-segmentation and under-segmentation errors are minimized.

    Args:
        pred_map: (H, W) integer array of predicted instance IDs (0/1=bg, >=2=instances).
        gt_map: (H, W) integer array of ground truth instance IDs (0/1=bg, >=2=instances).
        min_overlap: minimum intersection fraction to consider a valid match.

    Returns:
        Dict with keys: 'mean_os', 'mean_us', 'mean_ed', 'best_ed', 'matched_ratio'.
    """
    gt_ids = np.unique(gt_map)
    gt_ids = gt_ids[gt_ids >= 2]
    pred_ids = np.unique(pred_map)
    pred_ids = pred_ids[pred_ids >= 2]

    if len(gt_ids) == 0:
        return {"mean_os": 0.0, "mean_us": 0.0, "mean_ed": 0.0, "best_ed": 0.0, "matched_ratio": 1.0}

    if len(pred_ids) == 0:
        return {"mean_os": 0.0, "mean_us": 0.0, "mean_ed": np.sqrt(2.0), "best_ed": np.sqrt(2.0), "matched_ratio": 0.0}

    os_list = []
    us_list = []
    ed_list = []
    matched_gt = 0

    for gid in gt_ids:
        g_mask = gt_map == gid
        g_area = float(g_mask.sum())
        if g_area == 0:
            continue

        overlapping_preds = pred_map[g_mask]
        overlapping_preds = overlapping_preds[overlapping_preds >= 2]

        if len(overlapping_preds) == 0:
            # Entirely missed GT crown
            os_list.append(0.0)
            us_list.append(0.0)
            ed_list.append(float(np.sqrt(2.0)))
            continue

        unq_p, counts = np.unique(overlapping_preds, return_counts=True)
        best_p_idx = np.argmax(counts)
        best_pid = unq_p[best_p_idx]
        best_inter = float(counts[best_p_idx])

        p_mask = pred_map == best_pid
        p_area = float(p_mask.sum())

        os_val = best_inter / g_area
        us_val = best_inter / p_area if p_area > 0 else 0.0

        if os_val >= min_overlap:
            matched_gt += 1
            ed_val = float(np.sqrt((1.0 - os_val) ** 2 + (1.0 - us_val) ** 2))
            os_list.append(os_val)
            us_list.append(us_val)
            ed_list.append(ed_val)
        else:
            os_list.append(0.0)
            us_list.append(0.0)
            ed_list.append(float(np.sqrt(2.0)))

    mean_os = float(np.mean(os_list)) if os_list else 0.0
    mean_us = float(np.mean(us_list)) if us_list else 0.0
    mean_ed = float(np.mean(ed_list)) if ed_list else np.sqrt(2.0)
    best_ed = float(np.min(ed_list)) if ed_list else np.sqrt(2.0)
    matched_ratio = matched_gt / len(gt_ids)

    return {
        "mean_os": mean_os,
        "mean_us": mean_us,
        "mean_ed": mean_ed,
        "best_ed": best_ed,
        "matched_ratio": matched_ratio,
    }


def compute_boundary_iou(
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    dilation_radius: int = 4,
) -> float:
    """Computes Boundary-Intersection over Union (B-IoU) within a d-pixel perimeter band.

    Formulation:
        B-IoU = |(M_pred ∩ Ω_d) ∩ (M_gt ∩ Ω_d)| / |(M_pred ∩ Ω_d) ∪ (M_gt ∩ Ω_d)|

    Measures sub-pixel contour adherence along the outer boundaries of crowns.
    """
    pred_fg = (pred_map >= 2).astype(np.uint8)
    gt_fg = (gt_map >= 2).astype(np.uint8)

    if pred_fg.sum() == 0 and gt_fg.sum() == 0:
        return 1.0
    if pred_fg.sum() == 0 or gt_fg.sum() == 0:
        return 0.0

    k = max(3, dilation_radius * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # Extract perimeter band for prediction
    pred_eroded = cv2.erode(pred_fg, kernel, iterations=1)
    pred_band = (pred_fg - pred_eroded) > 0

    # Extract perimeter band for GT
    gt_eroded = cv2.erode(gt_fg, kernel, iterations=1)
    gt_band = (gt_fg - gt_eroded) > 0

    # Also compute inter-instance boundaries inside multi-instance maps
    pred_grad = cv2.morphologyEx(pred_map.astype(np.float32), cv2.MORPH_GRADIENT, kernel) > 0
    gt_grad = cv2.morphologyEx(gt_map.astype(np.float32), cv2.MORPH_GRADIENT, kernel) > 0

    pred_boundary = (pred_band | pred_grad) & (pred_fg > 0)
    gt_boundary = (gt_band | gt_grad) & (gt_fg > 0)

    # Dilate boundary bands by d pixels
    pred_dilated = cv2.dilate(pred_boundary.astype(np.uint8), kernel, iterations=1) > 0
    gt_dilated = cv2.dilate(gt_boundary.astype(np.uint8), kernel, iterations=1) > 0

    inter = float(np.logical_and(pred_dilated, gt_dilated).sum())
    union = float(np.logical_or(pred_dilated, gt_dilated).sum())

    return inter / union if union > 0 else 0.0


def compute_contour_curvature_regularity(contour: np.ndarray) -> float:
    """Measures curvature regularity to detect unnatural pixel-staircase artifacts.

    Returns:
        Roughness index (lower is smoother, higher is more irregular/staircased).
    """
    if len(contour) < 5:
        return 0.0

    pts = contour.squeeze(1).astype(np.float32)
    diff1 = np.diff(pts, axis=0, append=pts[:1])
    lengths = np.linalg.norm(diff1, axis=1) + 1e-6
    tangents = diff1 / lengths[:, None]

    diff2 = np.diff(tangents, axis=0, append=tangents[:1])
    curvatures = np.linalg.norm(diff2, axis=1)

    return float(np.mean(curvatures))
