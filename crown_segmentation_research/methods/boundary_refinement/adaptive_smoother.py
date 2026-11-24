"""Adaptive Douglas-Peucker Spline Simplification for Benchmark Alignment (Mode A).

Aligns detailed raster contours with human-annotated ground truth polygons by
pruning high-frequency raster pixelization noise while preserving true macro-canopy curvature.
"""
from __future__ import annotations

import cv2
import numpy as np


def smooth_instance_polygon(
    mask: np.ndarray,
    epsilon_ratio: float = 0.008,
    min_vertices: int = 12,
    max_vertices: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Simplifies and smooths a binary instance mask into an envelope polygon.

    Args:
        mask: (H, W) binary uint8 mask of a single tree crown.
        epsilon_ratio: fraction of contour arc length used as Douglas-Peucker tolerance.
        min_vertices: minimum vertices to retain smooth organic curvature.
        max_vertices: maximum vertices to prevent jagged staircase noise.

    Returns:
        smoothed_mask: (H, W) rasterized binary mask of the smoothed polygon.
        polygon: (N, 2) array of (x, y) vertices.
    """
    H, W = mask.shape
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return mask, np.zeros((0, 2), dtype=np.int32)

    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    if peri < 10:
        return mask, c.squeeze(1)

    # Adaptive epsilon search to target optimal vertex density (matching GT ~30-50 vertices)
    eps = epsilon_ratio * peri
    approx = cv2.approxPolyDP(c, eps, True)

    # Fine-tune epsilon if too coarse or too dense
    if len(approx) < min_vertices and eps > 0.5:
        eps = 0.003 * peri
        approx = cv2.approxPolyDP(c, eps, True)
    elif len(approx) > max_vertices:
        eps = 0.015 * peri
        approx = cv2.approxPolyDP(c, eps, True)

    poly = approx.squeeze(1)
    smoothed_mask = np.zeros((H, W), dtype=np.uint8)
    if len(poly) >= 3:
        cv2.fillPoly(smoothed_mask, [poly], 1)
    else:
        smoothed_mask = mask

    return smoothed_mask, poly


def apply_benchmark_smoothing(
    instance_map: np.ndarray,
    epsilon_ratio: float = 0.008,
    min_area: int = 25,
) -> tuple[np.ndarray, list[dict]]:
    """Applies adaptive polygon smoothing to all instances in an instance map.

    Args:
        instance_map: (H, W) int32 instance map (0/1=bg, >=2=instances).
        epsilon_ratio: polygon simplification tolerance.
        min_area: minimum area threshold.

    Returns:
        smoothed_map: (H, W) int32 instance map.
        smoothed_instances: list of instance dicts.
    """
    H, W = instance_map.shape
    smoothed_map = np.zeros((H, W), dtype=np.int32)
    inst_ids = np.unique(instance_map)
    inst_ids = inst_ids[inst_ids >= 2]

    smoothed_instances = []

    for iid in inst_ids:
        raw_mask = (instance_map == iid).astype(np.uint8)
        if raw_mask.sum() < min_area:
            continue

        s_mask, poly = smooth_instance_polygon(raw_mask, epsilon_ratio=epsilon_ratio)
        area = int(s_mask.sum())

        if area >= min_area:
            # Assign non-overlapping pixels
            target_mask = (smoothed_map == 0) & (s_mask > 0)
            smoothed_map[target_mask] = iid

            ys, xs = np.where(s_mask > 0)
            cy, cx = int(np.mean(ys)), int(np.mean(xs))

            smoothed_instances.append({
                "id": iid,
                "apex": (cy, cx),
                "score": float(raw_mask.sum() / max(1, area)),
                "area": area,
                "polygon": poly,
            })

    return smoothed_map, smoothed_instances
