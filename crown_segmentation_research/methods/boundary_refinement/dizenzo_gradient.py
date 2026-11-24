"""
Di Zenzo Multi-Spectral Color Tensor Gradient.

Reference:
    Silvano Di Zenzo, "A Note on the Gradient of a Multi-Image",
    Computer Vision, Graphics, and Image Processing, 33(1):116-125, 1986.

Computes the Riemannian structure tensor across multiple optical channels (e.g. R, G, B)
to capture true chromatic transitions (e.g. reddish-brown senescent foliage vs dark-green canopy)
that are invisible to luminance-only grayscale Sobel filters.
"""
from __future__ import annotations

import cv2
import numpy as np


def compute_dizenzo_gradient(
    img_rgb: np.ndarray,
    sigma: float = 1.0,
    ksize: int = 3,
) -> np.ndarray:
    """Computes the Di Zenzo color tensor gradient magnitude.

    Args:
        img_rgb: (H, W, 3) image in uint8 [0, 255] or float32 [0, 1].
        sigma: Gaussian smoothing scale for pre-filtering high-frequency noise.
        ksize: Sobel kernel size (default 3).

    Returns:
        grad_norm: (H, W) float32 array in [0, 1] representing multi-spectral edge strength.
    """
    if img_rgb.dtype == np.uint8:
        img_f = img_rgb.astype(np.float32) / 255.0
    else:
        img_f = img_rgb.astype(np.float32)
        if img_f.max() > 1.0:
            img_f = img_f / 255.0

    if sigma > 0:
        # Pre-smooth slightly to avoid single-pixel leaf noise while keeping macro-crown edges
        img_f = cv2.GaussianBlur(img_f, (0, 0), sigma)

    # Compute spatial derivatives per channel
    # R channel
    rx = cv2.Sobel(img_f[:, :, 0], cv2.CV_32F, 1, 0, ksize=ksize)
    ry = cv2.Sobel(img_f[:, :, 0], cv2.CV_32F, 0, 1, ksize=ksize)
    # G channel
    gx = cv2.Sobel(img_f[:, :, 1], cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img_f[:, :, 1], cv2.CV_32F, 0, 1, ksize=ksize)
    # B channel
    bx = cv2.Sobel(img_f[:, :, 2], cv2.CV_32F, 1, 0, ksize=ksize)
    by = cv2.Sobel(img_f[:, :, 2], cv2.CV_32F, 0, 1, ksize=ksize)

    # Structure tensor components:
    # J = [[g_xx, g_xy], [g_xy, g_yy]]
    g_xx = rx**2 + gx**2 + bx**2
    g_yy = ry**2 + gy**2 + by**2
    g_xy = rx * ry + gx * gy + bx * by

    # Maximum eigenvalue of 2x2 symmetric matrix:
    # lambda_max = 0.5 * ( (g_xx + g_yy) + sqrt( (g_xx - g_yy)^2 + 4 * g_xy^2 ) )
    delta = np.sqrt(np.maximum(0.0, (g_xx - g_yy)**2 + 4.0 * (g_xy**2)))
    lambda_max = 0.5 * ((g_xx + g_yy) + delta)

    # Gradient magnitude is square root of principal eigenvalue
    grad_mag = np.sqrt(np.maximum(0.0, lambda_max))

    # Normalize to [0, 1] with robust clipping (p99.5) to avoid sensor glare artifacts
    p99 = np.percentile(grad_mag, 99.5)
    if p99 > 1e-6:
        grad_norm = np.clip(grad_mag / p99, 0.0, 1.0)
    else:
        grad_norm = grad_mag / (grad_mag.max() + 1e-6)

    return grad_norm.astype(np.float32)
