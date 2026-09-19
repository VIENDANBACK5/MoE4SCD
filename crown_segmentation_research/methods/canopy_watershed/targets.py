"""Target generation for Option A: Neural Learned Canopy Potential Surface + Persistence Watershed.

Computes 3 mathematically rigorous continuous target fields:
1. Unimodal Potential Surface U*(y, x) in [0, 1]: Normalized distance-to-boundary transform
   taking 1.0 at instance apex (medial centroid) and decreasing monotonically to 0.0 at perimeter.
2. Saddle Boundary Ridge B*(y, x) in [0, 1]: Energy barrier along touching interfaces between
   neighboring crowns and crown perimeters.
3. Canopy Support Gate M*(y, x) in {0, 1}: Binary foreground canopy mask.

100% Vectorized with OpenCV & SciPy. Zero external foundation dependencies.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def compute_canopy_watershed_targets(
    instance_label: np.ndarray,
    boundary_dilation: int = 2,
    boundary_sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes targets from an (H, W) integer instance label map (0=bg, 1..N=tree IDs).

    Args:
        instance_label: (H, W) integer array.
        boundary_dilation: dilation radius in pixels to locate inter-crown contact seams.
        boundary_sigma: Gaussian blur sigma for boundary ridge field.

    Returns:
        surface_target: (1, H, W) float32 in [0, 1].
        boundary_target: (1, H, W) float32 in [0, 1].
        canopy_target: (1, H, W) float32 in {0, 1}.
    """
    height, width = instance_label.shape
    surface_target = np.zeros((height, width), dtype=np.float32)
    canopy_target = (instance_label > 0).astype(np.float32)

    unique_labels = np.unique(instance_label)
    unique_labels = unique_labels[unique_labels != 0]

    if len(unique_labels) == 0:
        boundary_target = np.zeros((height, width), dtype=np.float32)
        return (
            surface_target[np.newaxis, ...],
            boundary_target[np.newaxis, ...],
            canopy_target[np.newaxis, ...],
        )

    # Structuring element for dilation to detect touching borders
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (boundary_dilation * 2 + 1, boundary_dilation * 2 + 1),
    )
    dilated_accumulator = np.zeros((height, width), dtype=np.uint8)

    for label in unique_labels:
        mask = (instance_label == label)
        mask_u8 = mask.astype(np.uint8)

        # Accumulate dilated footprints
        dilated = cv2.dilate(mask_u8, kernel, iterations=1)
        dilated_accumulator += dilated

        # Exact distance transform inside instance
        mask_edt = ndimage.distance_transform_edt(mask)
        max_dist = float(mask_edt.max())
        if max_dist > 0:
            # Monotonic normalized potential surface: 1 at apex, 0 at border
            u_inst = mask_edt / max_dist
            np.maximum(surface_target, u_inst, out=surface_target)

    # Touching boundary occurs where 2 or more dilated instances overlap
    touching_barrier = (dilated_accumulator >= 2) & (instance_label > 0)

    # Outer perimeter of tree canopy
    fg_eroded = cv2.erode(
        (instance_label > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    outer_boundary = ((instance_label > 0) & (fg_eroded == 0))

    combined_barrier = (touching_barrier | outer_boundary).astype(np.float32)

    if boundary_sigma > 0:
        blurred = cv2.GaussianBlur(combined_barrier, (5, 5), boundary_sigma)
        boundary_target = np.clip(blurred, 0.0, 1.0)
    else:
        boundary_target = combined_barrier

    return (
        surface_target[np.newaxis, ...],
        boundary_target[np.newaxis, ...],
        canopy_target[np.newaxis, ...],
    )
