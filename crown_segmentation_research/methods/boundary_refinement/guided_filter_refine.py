"""
High-Resolution Guided Filter for Organic Crown Boundary Refinement.

Reference:
    Kaiming He, Jian Sun, Xiaoou Tang,
    "Guided Image Filtering", IEEE TPAMI 35(6):1397-1409, 2013.

Uses high-resolution (512x512) optical RGB canopy imagery as guidance to transfer
crisp, edge-aligned natural foliage boundaries onto coarse/smooth instance masks.
"""
from __future__ import annotations

import cv2
import numpy as np


def guided_filter_color(
    guide_rgb: np.ndarray,
    src_mask: np.ndarray,
    radius: int = 5,
    eps: float = 1e-3,
) -> np.ndarray:
    """Applies multi-channel color guided filtering to an instance mask.

    Args:
        guide_rgb: (H, W, 3) float32 in [0, 1].
        src_mask: (H, W) float32 in [0, 1] or uint8.
        radius: window filter radius in pixels.
        eps: regularization parameter controlling edge sharpness vs smoothness.

    Returns:
        refined: (H, W) float32 array in [0, 1].
    """
    I = guide_rgb.astype(np.float32)
    p = src_mask.astype(np.float32)
    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_p = cv2.boxFilter(p, -1, ksize)

    # Multi-channel color guidance
    # Compute for each channel R, G, B
    q = np.zeros_like(p)
    for c in range(3):
        Ic = I[:, :, c]
        mean_I = cv2.boxFilter(Ic, -1, ksize)
        mean_Ip = cv2.boxFilter(Ic * p, -1, ksize)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(Ic * Ic, -1, ksize)
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, -1, ksize)
        mean_b = cv2.boxFilter(b, -1, ksize)

        q += (mean_a * Ic + mean_b) / 3.0

    return np.clip(q, 0.0, 1.0)


def refine_mask_guided_filter(
    mask: np.ndarray,
    img_rgb: np.ndarray,
    radius: int = 4,
    eps: float = 1e-3,
    threshold: float = 0.5,
    pad: int = 10,
) -> np.ndarray:
    """Refines a single binary instance mask using local bounding-box guided filtering.

    Args:
        mask: (H, W) boolean or uint8 binary mask.
        img_rgb: (H, W, 3) uint8 optical image.
        radius: guided filter window radius.
        eps: regularization.
        threshold: binarization threshold on refined soft output.
        pad: bounding box margin.

    Returns:
        refined_mask: (H, W) boolean mask tightly hugging the high-resolution RGB boundaries.
    """
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask.copy()

    # Crop to bounding box for 10x-50x speedup
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(H, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(W, int(xs.max()) + pad + 1)

    sub_mask = mask[y0:y1, x0:x1].astype(np.float32)
    sub_rgb = img_rgb[y0:y1, x0:x1].astype(np.float32) / 255.0

    refined_sub = guided_filter_color(sub_rgb, sub_mask, radius=radius, eps=eps)
    binary_sub = (refined_sub >= threshold)

    # Ensure mask remains non-empty
    if binary_sub.sum() < 10:
        binary_sub = (refined_sub >= 0.3)

    out_mask = np.zeros((H, W), dtype=bool)
    out_mask[y0:y1, x0:x1] = binary_sub
    return out_mask
