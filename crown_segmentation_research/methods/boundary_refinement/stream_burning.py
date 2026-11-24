"""
Hydrological Stream-Burning Topographic Gorge Conditioning (H27).

Bisociation from computational geomorphology & hydrology:
When elevation surfaces have flat saddles or ambiguous divides, Stream Burning carves
deep synthetic trenches into the DEM along verified optical channels to guarantee
watertight drainage boundaries.

Applied to individual tree crown instance segmentation:
Extracts high-curvature optical valleys and chromatic boundaries between touching crowns,
injecting an impenetrable ridge barrier into the topographic cost surface to prevent
watershed spillover in dense forest stands (especially Zone 2).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from crown_segmentation_research.methods.boundary_refinement.dizenzo_gradient import compute_dizenzo_gradient


def compute_stream_burning_gorge(
    img_rgb: np.ndarray,
    sigma: float = 1.2,
    ridge_boost: float = 2.0,
) -> np.ndarray:
    """Computes a synthetic topographic gorge/wall map from multi-spectral transitions.

    Args:
        img_rgb: (H, W, 3) optical RGB patch.
        sigma: scale for morphological/Laplacian ridge extraction.
        ridge_boost: contrast scaling factor for the gorge barrier.

    Returns:
        gorge_norm: (H, W) float32 array in [0, 1] with peak values along touching crown boundaries.
    """
    if img_rgb.dtype == np.uint8:
        img_f = img_rgb.astype(np.float32) / 255.0
    else:
        img_f = img_rgb.astype(np.float32)
        if img_f.max() > 1.0:
            img_f = img_f / 255.0

    # 1. Multi-spectral Di Zenzo gradient magnitude
    grad_dizenzo = compute_dizenzo_gradient(img_rgb, sigma=sigma)

    # 2. Green-Red contrast chasm (dark shadows / branch gaps between crowns)
    r = img_f[:, :, 0]
    g = img_f[:, :, 1]
    b = img_f[:, :, 2]
    # Total luminance
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    # Inverted valley: touching boundaries are local luminance and canopy minima
    # Black top-hat transform (bottom-hat) extracts narrow dark valleys
    ksize = int(round(sigma * 4)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    valleys = cv2.morphologyEx(luminance, cv2.MORPH_BLACKHAT, kernel)

    # 3. Morphological Ridge / Laplacian of gradient
    # Positive Laplacian of edge magnitude identifies the crisp crest of the boundary
    grad_smooth = cv2.GaussianBlur(grad_dizenzo, (ksize, ksize), sigma)
    laplacian = cv2.Laplacian(grad_smooth, cv2.CV_32F, ksize=3)
    ridge = np.clip(-laplacian, 0, None)  # Negative laplacian of smooth gradient has peaks at edges
    ridge = ridge / (ridge.max() + 1e-6)

    # 4. Composite gorge surface: sharp color jump + dark shadow valley + boundary ridge
    gorge = grad_dizenzo * 0.5 + valleys * 1.5 + ridge * 0.5
    gorge = np.clip(gorge * ridge_boost, 0.0, 1.0).astype(np.float32)

    return gorge
