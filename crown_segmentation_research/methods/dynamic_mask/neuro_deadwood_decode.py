"""Polytope & Instance Decoding for Neuro-Algorithmic Deadwood Segmentation.

Transforms network outputs:
  - Object probability map
  - Orientation field phi
  - Anisotropic eps-kernel extent radii / coreset points
  - Spectral Modularity cluster assignments
Into clean, validated Shapely polygons for evaluation and visualization.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


def find_peaks(probability: np.ndarray, prob_threshold: float, min_distance: int) -> list[tuple[int, int]]:
    """Local maxima of the probability map above threshold."""
    footprint = np.ones((2 * min_distance + 1, 2 * min_distance + 1), dtype=bool)
    local_max = ndimage.maximum_filter(probability, footprint=footprint) == probability
    candidates = local_max & (probability >= prob_threshold)
    ys, xs = np.where(candidates)
    if len(ys) == 0:
        return []
    scores = probability[ys, xs]
    order = np.argsort(-scores)
    return [(int(ys[i]), int(xs[i])) for i in order]


def eps_kernel_to_polygon(
    cy: int,
    cx: int,
    radii: np.ndarray,
    phi: float,
    n_directions: int = 16,
    aspect_ratio_scale: float = 1.0,
    coordinate_scale: float = 256.0,
) -> Polygon | None:
    """Constructs oriented anisotropic polytope polygon around (cx, cy)."""
    canonical_angles = np.linspace(0, 2.0 * np.pi, n_directions, endpoint=False)
    # Rotate by principal orientation phi
    angles = canonical_angles + phi
    
    # Scale radii: if values are normalized (magnitude <= 5.0), multiply by coordinate_scale
    raw_radii = np.asarray(radii, dtype=np.float32)
    if np.max(np.abs(raw_radii)) <= 5.0:
        scaled_radii = np.abs(raw_radii) * coordinate_scale
    else:
        scaled_radii = np.abs(raw_radii)
        
    scaled_radii = np.clip(scaled_radii * aspect_ratio_scale, a_min=3.0, a_max=400.0)
    
    points = [
        (float(cx + scaled_radii[k] * np.cos(angles[k])), float(cy + scaled_radii[k] * np.sin(angles[k])))
        for k in range(n_directions)
    ]
    try:
        poly = Polygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if not poly.is_empty and poly.area >= 10.0 else None
    except Exception:
        return None


def polygon_nms(
    polygons: list[Polygon],
    scores: list[float],
    orientations: list[float],
    iou_threshold: float = 0.20,
    angle_threshold: float = 0.5, # radians (~30 deg)
) -> list[int]:
    """Anisotropic IoU-NMS: two high-IoU instances are kept if their orientations differ (crossed logs)."""
    kept: list[int] = []
    for i, poly in enumerate(polygons):
        suppress = False
        for k in kept:
            other = polygons[k]
            # Fast bbox overlap check
            if not poly.envelope.intersects(other.envelope):
                continue
            try:
                inter = poly.intersection(other).area
                union = poly.union(other).area
                iou = inter / union if union > 0 else 0.0
            except Exception:
                iou = 0.0
                
            if iou > iou_threshold:
                # If orientations are significantly different, they are crisscrossing logs!
                delta_angle = abs(orientations[i] - orientations[k])
                if delta_angle > np.pi / 2:
                    delta_angle = np.pi - delta_angle
                    
                if delta_angle > angle_threshold:
                    # Keep both crossed logs
                    continue
                else:
                    suppress = True
                    break
        if not suppress:
            kept.append(i)
    return kept


def decode_neuro_deadwood(
    probability: np.ndarray,
    orientation: np.ndarray | None = None,
    extent_radii: np.ndarray | None = None,
    canopy: np.ndarray | None = None,
    cluster_assignments: np.ndarray | None = None,
    prob_threshold: float = 0.40,
    min_peak_distance: int = 4,
    nms_iou_threshold: float = 0.20,
    min_area: float = 20.0,
    n_directions: int = 16,
) -> list[Polygon]:
    """Decodes full network output into instance polygons."""
    h, w = probability.shape
    peaks = find_peaks(probability, prob_threshold=prob_threshold, min_distance=min_peak_distance)
    
    if not peaks:
        # Fallback to connected components thresholding
        mask = (probability >= prob_threshold).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for c in contours:
            if len(c) >= 3:
                pts = c.squeeze()
                if pts.ndim == 2 and len(pts) >= 3:
                    p = Polygon(pts)
                    if p.is_valid and p.area >= min_area:
                        polys.append(p)
        return polys

    # Default orientation and radii if not provided
    if orientation is None:
        orientation = np.zeros((h, w), dtype=np.float32)
    if extent_radii is None:
        extent_radii = np.full((n_directions,), 15.0, dtype=np.float32)

    polygons: list[Polygon] = []
    scores: list[float] = []
    angles: list[float] = []

    for cy, cx in peaks:
        score = float(probability[cy, cx])
        phi = float(orientation[cy, cx])
        
        # If extent radii is 1D (per-image) or 2D
        if extent_radii.ndim == 1:
            r = extent_radii
        elif extent_radii.ndim == 3: # (K, H, W)
            r = extent_radii[:, cy, cx]
        else:
            r = extent_radii.flatten()[:n_directions]

        poly = eps_kernel_to_polygon(cy, cx, r, phi, n_directions=n_directions)
        if poly is not None and poly.area >= min_area:
            polygons.append(poly)
            scores.append(score)
            angles.append(phi)

    if not polygons:
        return []

    # Run Anisotropic IoU-NMS
    kept_indices = polygon_nms(polygons, scores, angles, iou_threshold=nms_iou_threshold)
    return [polygons[idx] for idx in kept_indices]
