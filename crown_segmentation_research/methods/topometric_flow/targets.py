"""Target generation for TopoMetric Flow 1-Stage ITC Instance Segmentation.

Computes 4 high-precision geometric & topological target maps:
1. Centripetal Vector Flow Field V*(2, H, W): Unit direction vector pointing to instance apex.
2. Saddle Barrier Energy S*(1, H, W): Repulsion barrier at the contact boundary between touching crowns.
3. Learned Potential Surface U*(1, H, W): Unimodal normalized distance potential (1 at apex, 0 at boundary).
4. Canopy Support Gate C*(1, H, W): Binary foreground canopy mask.

100% Fully vectorized in NumPy / SciPy / OpenCV.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def compute_topometric_targets(
    instance_label: np.ndarray,
    saddle_dilation: int = 2,
    saddle_sigma: float = 1.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes target maps from an (H, W) integer instance label map (0=bg).

    Args:
        instance_label: (H, W) integer array (0 = background, 1..N = instance IDs).
        saddle_dilation: radius in pixels for detecting touching interfaces.
        saddle_sigma: Gaussian smoothing sigma for the saddle barrier field.

    Returns:
        flow_target: (2, H, W) float32 array with (vy, vx) unit vectors.
        saddle_target: (1, H, W) float32 array in [0, 1].
        surface_target: (1, H, W) float32 array in [0, 1].
        canopy_target: (1, H, W) float32 array in {0, 1}.
    """
    height, width = instance_label.shape
    flow_target = np.zeros((2, height, width), dtype=np.float32)
    surface_target = np.zeros((1, height, width), dtype=np.float32)
    canopy_target = (instance_label > 0).astype(np.float32)[np.newaxis, ...]

    unique_labels = np.unique(instance_label)
    unique_labels = unique_labels[unique_labels != 0]

    if len(unique_labels) == 0:
        saddle_target = np.zeros((1, height, width), dtype=np.float32)
        return flow_target, saddle_target, surface_target, canopy_target

    yy, xx = np.mgrid[0:height, 0:width]

    # Structuring element for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (saddle_dilation * 2 + 1, saddle_dilation * 2 + 1))
    dilated_accumulator = np.zeros((height, width), dtype=np.uint8)

    for label in unique_labels:
        mask = (instance_label == label)
        mask_u8 = mask.astype(np.uint8)
        
        # Dilate to find contact interfaces
        dilated = cv2.dilate(mask_u8, kernel, iterations=1)
        dilated_accumulator += dilated

        # Distance transform to find topological apex
        mask_edt = ndimage.distance_transform_edt(mask)
        max_dist = float(mask_edt.max())
        if max_dist > 0:
            surface_k = mask_edt / max_dist
            np.maximum(surface_target[0], surface_k, out=surface_target[0])

        # Topological apex = deepest interior point
        max_idx = np.argmax(mask_edt)
        cy, cx = np.unravel_index(max_idx, (height, width))

        # Centripetal flow unit vectors
        dy = cy - yy[mask]
        dx = cx - xx[mask]
        norm = np.sqrt(dy * dy + dx * dx) + 1e-6
        flow_target[0, mask] = dy / norm
        flow_target[1, mask] = dx / norm

    # Saddle barrier is formed where 2 or more dilated instances overlap
    touching_barrier = (dilated_accumulator >= 2) & (instance_label > 0)
    
    # Also add outer boundaries touching background
    fg_eroded = cv2.erode((instance_label > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    outer_boundary = ((instance_label > 0) & (fg_eroded == 0))
    
    combined_barrier = (touching_barrier | outer_boundary).astype(np.float32)
    
    # Smooth saddle barrier with Gaussian blur
    if saddle_sigma > 0:
        saddle_blurred = cv2.GaussianBlur(combined_barrier, (5, 5), saddle_sigma)
        saddle_target = np.clip(saddle_blurred, 0.0, 1.0)[np.newaxis, ...].astype(np.float32)
    else:
        saddle_target = combined_barrier[np.newaxis, ...]

    return flow_target, saddle_target, surface_target, canopy_target
