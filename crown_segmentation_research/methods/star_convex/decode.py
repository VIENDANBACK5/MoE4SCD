"""Decode dense (object_probability, ray_distances) maps into instance
polygons: peak-finding for centers + per-pixel star-convex polygon
construction + polygon-IoU NMS.

Works on either GT-derived targets (oracle mode, no network involved) or a
trained network's predicted maps -- same decode either way, matching this
project's convention of testing decode logic against oracle targets first.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon


def find_peaks(probability: np.ndarray, prob_threshold: float, min_distance: int) -> list[tuple[int, int]]:
    """Local maxima of the probability map above threshold, at least
    `min_distance` apart (simple non-max suppression on the probability map
    itself, before any polygon is built).
    """
    footprint = np.ones((2 * min_distance + 1, 2 * min_distance + 1), dtype=bool)
    local_max = ndimage.maximum_filter(probability, footprint=footprint) == probability
    candidates = local_max & (probability >= prob_threshold)
    ys, xs = np.where(candidates)
    scores = probability[ys, xs]
    order = np.argsort(-scores)
    return [(int(ys[i]), int(xs[i])) for i in order]


def ray_to_polygon(cy: int, cx: int, ray_distances: np.ndarray, n_rays: int) -> Polygon | None:
    angles = 2.0 * np.pi * np.arange(n_rays) / n_rays
    points = [
        (cx + ray_distances[k] * np.cos(angles[k]), cy + ray_distances[k] * np.sin(angles[k]))
        for k in range(n_rays)
    ]
    try:
        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        return polygon if not polygon.is_empty else None
    except Exception:
        return None


def polygon_nms(
    polygons: list[Polygon],
    scores: list[float],
    iou_threshold: float,
    peak_embeddings: list[np.ndarray] | None = None,
    embedding_delta_d: float = 1.5,
) -> list[int]:
    """Standard greedy IoU-NMS on shapely polygons. Returns kept indices.

    If `peak_embeddings` is given (StarConvexNet's optional discriminative
    embedding, code/discriminative_loss.py), a high-IoU pair is only
    suppressed if their embeddings are also close (< embedding_delta_d) --
    i.e. the network's own explicit inter-instance signal can override a
    pure-geometry NMS decision and keep two overlapping-but-different
    touching crowns that plain IoU-NMS would otherwise merge into one.
    embedding_delta_d matches the delta_d margin used to train that
    embedding (discriminative_loss.py), since it is only meaningful in the
    units that loss shaped the embedding space in.
    """
    order = list(np.argsort(-np.asarray(scores)))
    keep: list[int] = []
    while order:
        current = order.pop(0)
        keep.append(current)
        remaining = []
        for index in order:
            inter = polygons[current].intersection(polygons[index]).area
            union = polygons[current].union(polygons[index]).area
            iou = inter / union if union > 0 else 0.0
            if iou < iou_threshold:
                remaining.append(index)
                continue
            if peak_embeddings is not None:
                embedding_distance = float(np.linalg.norm(peak_embeddings[current] - peak_embeddings[index]))
                if embedding_distance >= embedding_delta_d:
                    remaining.append(index)  # high IoU, but embedding says: different instances -- keep both
        order = remaining
    return keep


def decode(
    probability: np.ndarray,
    rays: np.ndarray,
    n_rays: int,
    prob_threshold: float = 0.5,
    min_peak_distance: int = 3,
    nms_iou_threshold: float = 0.3,
    canopy: np.ndarray | None = None,
    canopy_threshold: float = 0.5,
    embedding: np.ndarray | None = None,
    embedding_delta_d: float = 1.5,
) -> list[Polygon]:
    """`canopy`, if given (StarConvexNet's optional canopy head output),
    gates the probability map before peak-finding: pixels the network
    thinks are not crown material at all are zeroed out first, addressing
    the 70%-of-false-positives-are-background-clutter failure mode
    (star_convex_v3_failure_diagnosis.md) directly rather than relying on
    object_probability thresholding alone to reject non-crown texture.

    `embedding`, if given (StarConvexNet's optional discriminative-loss
    embedding, (D, H, W)), is sampled at each peak and passed to
    polygon_nms so it can keep two high-IoU, embedding-distant peaks
    instead of collapsing them -- see polygon_nms's docstring.
    """
    if canopy is not None:
        probability = np.where(canopy >= canopy_threshold, probability, 0.0)
    peaks = find_peaks(probability, prob_threshold, min_peak_distance)
    polygons: list[Polygon] = []
    scores: list[float] = []
    peak_embeddings: list[np.ndarray] = []
    for cy, cx in peaks:
        polygon = ray_to_polygon(cy, cx, rays[:, cy, cx], n_rays)
        if polygon is None or polygon.area < 1.0:
            continue
        polygons.append(polygon)
        scores.append(float(probability[cy, cx]))
        if embedding is not None:
            peak_embeddings.append(embedding[:, cy, cx])
    if not polygons:
        return []
    keep = polygon_nms(
        polygons, scores, nms_iou_threshold,
        peak_embeddings=peak_embeddings if embedding is not None else None,
        embedding_delta_d=embedding_delta_d,
    )
    return [polygons[i] for i in keep]
