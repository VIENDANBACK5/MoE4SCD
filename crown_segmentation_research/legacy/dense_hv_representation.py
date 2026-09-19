"""Dense P_fg/H/V crown representation and marker-controlled-watershed decode.

Implements Section 3 of design_docs/method_design_dense_crown_separation_v1.md
(supervision targets + decode only -- the prediction network itself is a
separate, later step). H/V follow HoVer-Net's per-instance normalization
(each side of the centroid normalized independently to [-1, 1] by that
side's own extent, so the field is well-defined regardless of shape
asymmetry) rather than a plain bbox-relative normalization.

decode_instances() is deliberately usable on GT-derived targets, not only
network predictions, so the representation+decode mechanism itself can be
oracle-tested before any model is trained -- if it cannot recover instances
from noise-free targets, no amount of training will fix that.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed


def compute_hv_targets(
    instance_masks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense (P_fg, H, V) targets from a list of non-overlapping instance masks.

    Each mask is (H, W) bool. Returns three (H, W) float32 arrays. Where
    instance masks overlap (rare edge case in BAM), later instances in the
    list win -- not resolved more carefully here since BAM crowns are
    documented as largely non-overlapping polygons.
    """
    if not instance_masks:
        shape = (0, 0)
        return np.zeros(shape, np.float32), np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    shape = instance_masks[0].shape
    p_fg = np.zeros(shape, dtype=bool)
    h_map = np.zeros(shape, dtype=np.float32)
    v_map = np.zeros(shape, dtype=np.float32)

    for mask in instance_masks:
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        cx, cy = xs.mean(), ys.mean()
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        dx = xs.astype(np.float32) - cx
        dy = ys.astype(np.float32) - cy
        left_extent = max(cx - x0, 1e-6)
        right_extent = max(x1 - cx, 1e-6)
        top_extent = max(cy - y0, 1e-6)
        bottom_extent = max(y1 - cy, 1e-6)
        h_values = np.where(dx < 0, dx / left_extent, dx / right_extent)
        v_values = np.where(dy < 0, dy / top_extent, dy / bottom_extent)
        p_fg[mask] = True
        h_map[ys, xs] = np.clip(h_values, -1.0, 1.0)
        v_map[ys, xs] = np.clip(v_values, -1.0, 1.0)

    return p_fg.astype(np.float32), h_map, v_map


def decode_instances(
    p_fg: np.ndarray,
    h_map: np.ndarray,
    v_map: np.ndarray,
    fg_threshold: float = 0.5,
    marker_percentile: float = 30.0,
    min_marker_area: int = 10,
) -> list[np.ndarray]:
    """Marker-controlled watershed decode. Returns a list of boolean instance masks.

    S(x,y) = max(|d H/dx|, |d V/dy|) -- Sobel-gradient separation energy,
    per Section 3 (matches HoVer-Net's Eq. 6 construction).

    Marker threshold is *relative*, not a fixed absolute value: within each
    connected foreground blob (which may contain several touching, not-yet-
    separated instances), markers are the lowest `marker_percentile`% of S
    values in that blob. A fixed absolute threshold does not generalize
    across crown sizes -- H's interior gradient scales as ~1/instance_extent
    by construction (see compute_hv_targets), so a small crown and a large
    crown have different "flat interior" gradient magnitudes even though
    both are equally flat relative to their own boundary spike. This was
    found by testing on synthetic rectangles, where a fixed threshold failed
    to separate two touching instances entirely (see
    tests/test_dense_hv_representation.py).
    """
    foreground = p_fg >= fg_threshold
    if not foreground.any():
        return []

    grad_h_x = ndimage.sobel(h_map, axis=1)
    grad_v_y = ndimage.sobel(v_map, axis=0)
    separation_energy = np.maximum(np.abs(grad_h_x), np.abs(grad_v_y))
    # compute_hv_targets() produces an exactly piecewise-linear field, so a
    # large fraction of interior pixels can share the *exact* same gradient
    # value. Percentile-thresholding on massively-tied data is numerically
    # fragile (ties split arbitrarily across the cutoff by floating-point
    # noise, fragmenting a single flat interior into several spurious
    # pieces -- found by testing on a plain square, see
    # tests/test_dense_hv_representation.py). A light Gaussian blur breaks
    # exact ties; it also makes this synthetic-target behavior closer to
    # what a trained network's predicted (naturally smooth, never exactly
    # piecewise-linear) H/V field would look like.
    separation_energy = ndimage.gaussian_filter(separation_energy, sigma=1.0)

    blob_labels, n_blobs = ndimage.label(foreground)
    marker_candidates = np.zeros_like(foreground, dtype=bool)
    for blob_id in range(1, n_blobs + 1):
        blob_mask = blob_labels == blob_id
        threshold = np.percentile(separation_energy[blob_mask], marker_percentile)
        marker_candidates |= blob_mask & (separation_energy <= threshold)

    labeled_markers, n_markers = ndimage.label(marker_candidates)
    if n_markers == 0:
        return [foreground]

    sizes = ndimage.sum(marker_candidates, labeled_markers, range(1, n_markers + 1))
    keep_labels = {index + 1 for index, size in enumerate(sizes) if size >= min_marker_area}
    if not keep_labels:
        return [foreground]
    cleaned_markers = np.where(
        np.isin(labeled_markers, list(keep_labels)), labeled_markers, 0
    )

    labels = watershed(separation_energy, markers=cleaned_markers, mask=foreground)
    instance_masks = [labels == label for label in np.unique(labels) if label != 0]
    return instance_masks
