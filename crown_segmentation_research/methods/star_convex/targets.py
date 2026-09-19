"""Per-pixel star-convex (StarDist-style) training targets.

Unlike the centroid-only oracle test in oracle_test_stardist_encoding.md,
real StarDist supervises *every foreground pixel* with its own K
ray-distances to that instance's boundary (not only the centroid), and a
separate object-probability target so that, at inference, the network's own
highest-probability pixel is used as the center -- no ground-truth centroid
is available at inference time.

Ray-distance computation uses vectorized array marching (repeated integer
pixel steps along each of K directions, checking when the mask is exited)
rather than per-pixel shapely queries, since a shapely query per foreground
pixel per ray would be far too slow to run over a whole training image.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def ray_distance_maps(mask: np.ndarray, n_rays: int, max_radius: int | None = None) -> np.ndarray:
    """For every foreground pixel, distance (in pixels) to the mask boundary
    along each of `n_rays` evenly spaced directions.

    Returns an (n_rays, H, W) float32 array, 0 outside the mask.
    Implemented by marching all foreground pixels simultaneously one unit
    step at a time along each direction (numpy roll/shift, no per-pixel
    Python loop) until they exit the mask.
    """
    height, width = mask.shape
    if max_radius is None:
        max_radius = int(np.ceil(np.hypot(height, width)))
    angles = 2.0 * np.pi * np.arange(n_rays) / n_rays
    out = np.zeros((n_rays, height, width), dtype=np.float32)

    for ray_index, angle in enumerate(angles):
        step_x, step_y = np.cos(angle), np.sin(angle)
        distance = np.zeros((height, width), dtype=np.float32)
        still_inside = mask.copy()
        for radius in range(1, max_radius + 1):
            if not still_inside.any():
                break
            sample_x = np.clip(np.round(np.arange(width) + radius * step_x).astype(int), 0, width - 1)
            sample_y = np.clip(np.round(np.arange(height) + radius * step_y).astype(int), 0, height - 1)
            sampled = mask[np.ix_(sample_y, sample_x)]
            still_inside &= sampled
            distance[still_inside] = radius
        out[ray_index] = distance
    out[:, ~mask] = 0.0
    return out


def object_probability_map(mask: np.ndarray) -> np.ndarray:
    """StarDist-style object probability: normalized distance-to-background,
    peaking at 1.0 at the most-interior pixel of each instance, 0 at the
    boundary and outside.
    """
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float32)
    distance = ndimage.distance_transform_edt(mask)
    peak = distance.max()
    if peak <= 0:
        return mask.astype(np.float32)
    return (distance / peak).astype(np.float32)


def boundary_weight_map(
    instance_masks: list[np.ndarray], w0: float = 10.0, sigma: float = 10.0, padding: int = 60,
) -> np.ndarray:
    """Ronneberger et al. (2015) U-Net weighted loss, adapted to crowns:
    w(x) = w0 * exp(-(d1(x) + d2(x))^2 / (2*sigma^2)) on background pixels,
    where d1/d2 are distances to the two nearest instances. Up-weights the
    thin background ridge between two touching, separate instances --
    exactly the signal star_convex_v3_failure_diagnosis.md found missing
    (plain/focal BCE has no term telling the network where one instance
    ends and its visually-similar neighbor begins).

    sigma=10px matches the scale of touching-crown gaps observed directly
    in this project's own failure-diagnosis preview images (a handful of
    pixels at ~1.7cm/px BAM resolution), not an arbitrary default; w0=10
    is the original paper's own constant, kept as a defensible starting
    point rather than re-tuned here (this is a Stage-1 cheap-fix screen,
    not a hyperparameter search).

    Zero on foreground pixels: object_probability_map already encodes each
    instance's own boundary falloff there, so adding this term would double
    up on existing structure instead of adding new information.
    """
    if len(instance_masks) < 2:
        return np.zeros_like(instance_masks[0], dtype=np.float32) if instance_masks else np.zeros((0, 0), np.float32)

    shape = instance_masks[0].shape
    d1 = np.full(shape, np.inf, dtype=np.float32)
    d2 = np.full(shape, np.inf, dtype=np.float32)
    any_foreground = np.zeros(shape, dtype=bool)

    for mask in instance_masks:
        any_foreground |= mask
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        y0, y1 = max(0, ys.min() - padding), min(shape[0], ys.max() + padding + 1)
        x0, x1 = max(0, xs.min() - padding), min(shape[1], xs.max() + padding + 1)
        crop_mask = mask[y0:y1, x0:x1]
        distance = ndimage.distance_transform_edt(~crop_mask).astype(np.float32)

        region_d1 = d1[y0:y1, x0:x1]
        region_d2 = d2[y0:y1, x0:x1]
        closer_than_d1 = distance < region_d1
        new_d2 = np.where(closer_than_d1, region_d1, np.minimum(region_d2, distance))
        new_d1 = np.where(closer_than_d1, distance, region_d1)
        d1[y0:y1, x0:x1] = new_d1
        d2[y0:y1, x0:x1] = new_d2

    finite_pair = np.isfinite(d1) & np.isfinite(d2)
    d1 = np.where(finite_pair, d1, 0.0)
    d2 = np.where(finite_pair, d2, 0.0)
    weight = w0 * np.exp(-((d1 + d2) ** 2) / (2.0 * sigma**2))
    weight = np.where(finite_pair & ~any_foreground, weight, 0.0)
    return weight.astype(np.float32)


def build_targets(
    instance_masks: list[np.ndarray], n_rays: int, padding: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Combine per-instance targets into one dense (probability, rays) pair
    for a whole image. Later instances win on overlap (rare in BAM).

    Each instance is processed on a crop of its own bounding box (+padding),
    not the full image: ray_distance_maps' cost scales with max_radius (the
    marching distance) times image area, and running it at full-image scale
    for a small instance is enormously wasteful -- ~100s for a single
    125,629-px instance at 2048x2048 scale versus a fraction of a second
    cropped to its own ~450x450 bounding box. This was measured directly
    before this fix, not assumed.
    """
    if not instance_masks:
        return np.zeros((0, 0), np.float32), np.zeros((n_rays, 0, 0), np.float32)
    shape = instance_masks[0].shape
    probability = np.zeros(shape, dtype=np.float32)
    rays = np.zeros((n_rays, *shape), dtype=np.float32)
    for mask in instance_masks:
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        y0, y1 = max(0, ys.min() - padding), min(shape[0], ys.max() + padding + 1)
        x0, x1 = max(0, xs.min() - padding), min(shape[1], xs.max() + padding + 1)
        crop = mask[y0:y1, x0:x1]
        crop_probability = object_probability_map(crop)
        crop_rays = ray_distance_maps(crop, n_rays)
        probability[y0:y1, x0:x1][crop] = crop_probability[crop]
        rays[:, y0:y1, x0:x1][:, crop] = crop_rays[:, crop]
    return probability, rays
