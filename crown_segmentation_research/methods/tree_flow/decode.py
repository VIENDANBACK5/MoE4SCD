"""Euler Flow Decoding and Sink Clustering for TreeFlowNet.

Decodes predicted flow fields, boundary maps, and centroid heatmaps into
individual instance polygons and masks.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon, MultiPolygon


def decode_flow_to_instances(
    flow: np.ndarray,
    canopy: np.ndarray,
    sdt: np.ndarray,
    centroid: np.ndarray,
    canopy_threshold: float = 0.5,
    sdt_threshold: float = -0.2,
    centroid_threshold: float = 0.3,
    min_peak_distance: int = 5,
    n_steps: int = 20,
    step_size: float = 1.5,
    min_instance_area: int = 30,
) -> tuple[np.ndarray, list[Polygon]]:
    """Decode TreeFlowNet outputs into instance label map and shapely Polygons.

    Args:
        flow: (2, H, W) float32 array of (vy, vx) unit vectors.
        canopy: (H, W) or (1, H, W) canopy probability.
        sdt: (H, W) or (1, H, W) signed distance transform in [-1, 1].
        centroid: (H, W) or (1, H, W) centroid heatmap.
        canopy_threshold: threshold for foreground canopy mask.
        sdt_threshold: threshold for boundary SDT filtering.
        centroid_threshold: threshold for identifying instance seed peaks.
        min_peak_distance: minimum distance between distinct centroid seeds.
        n_steps: number of Euler integration steps.
        step_size: step size (eta) for Euler flow integration.
        min_instance_area: minimum area in pixels to keep an instance.

    Returns:
        instance_map: (H, W) int32 array with 0=bg, 1..N = instance IDs.
        polygons: list of shapely Polygon / MultiPolygon objects.
    """
    if canopy.ndim == 3:
        canopy = canopy.squeeze(0)
    if sdt.ndim == 3:
        sdt = sdt.squeeze(0)
    if centroid.ndim == 3:
        centroid = centroid.squeeze(0)

    height, width = canopy.shape
    fg_mask = (canopy >= canopy_threshold) & (sdt >= sdt_threshold)
    if not np.any(fg_mask):
        return np.zeros((height, width), dtype=np.int32), []

    # Step 1: Find instance seeds from centroid heatmap local peaks
    peak_filter = ndimage.maximum_filter(centroid, size=min_peak_distance * 2 + 1)
    peaks = (centroid == peak_filter) & (centroid >= centroid_threshold) & fg_mask
    seed_ys, seed_xs = np.nonzero(peaks)

    # Step 2: Vectorized Euler flow integration on foreground pixels
    fg_ys, fg_xs = np.nonzero(fg_mask)
    cur_ys = fg_ys.astype(np.float32)
    cur_xs = fg_xs.astype(np.float32)

    vy_map = flow[0]
    vx_map = flow[1]

    for _ in range(n_steps):
        iy = np.clip(np.round(cur_ys).astype(np.int32), 0, height - 1)
        ix = np.clip(np.round(cur_xs).astype(np.int32), 0, width - 1)
        cur_ys += step_size * vy_map[iy, ix]
        cur_xs += step_size * vx_map[iy, ix]
        cur_ys = np.clip(cur_ys, 0, height - 1)
        cur_xs = np.clip(cur_xs, 0, width - 1)

    end_ys = cur_ys
    end_xs = cur_xs

    # Step 3: Assign foreground pixels to seeds or cluster endpoints
    instance_map = np.zeros((height, width), dtype=np.int32)

    if len(seed_ys) > 0:
        # Assign each pixel to the closest seed to its integrated endpoint
        seeds = np.stack([seed_ys, seed_xs], axis=1).astype(np.float32)  # (K, 2)
        endpoints = np.stack([end_ys, end_xs], axis=1).astype(np.float32)  # (M, 2)

        # Distance matrix (M, K)
        # For memory efficiency, compute in chunks if M is large
        chunk_size = 50000
        best_seeds = np.zeros(len(endpoints), dtype=np.int32)
        for start in range(0, len(endpoints), chunk_size):
            end = min(start + chunk_size, len(endpoints))
            dists = np.sum((endpoints[start:end, np.newaxis, :] - seeds[np.newaxis, :, :]) ** 2, axis=-1)
            best_seeds[start:end] = np.argmin(dists, axis=-1) + 1  # 1-indexed

        instance_map[fg_ys, fg_xs] = best_seeds
    else:
        # Fallback: connected components on foreground if no peaks found
        num_labels, labels = cv2.connectedComponents(fg_mask.astype(np.uint8))
        instance_map = labels

    # Step 4: Filter small instances & extract shapely polygons
    polygons = []
    clean_instance_map = np.zeros((height, width), dtype=np.int32)
    next_id = 1

    unique_ids = np.unique(instance_map)
    unique_ids = unique_ids[unique_ids != 0]

    for inst_id in unique_ids:
        inst_mask = (instance_map == inst_id).astype(np.uint8)
        if inst_mask.sum() < min_instance_area:
            continue

        contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) >= 3:
                pts = cnt.squeeze(1)
                poly = Polygon(pts)
                if poly.is_valid and poly.area >= min_instance_area:
                    polygons.append(poly)
                    clean_instance_map[inst_mask > 0] = next_id
                    next_id += 1

    return clean_instance_map, polygons
