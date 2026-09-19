"""Target generation for TreeFlowNet / OmniCrown instance segmentation.

Computes:
1. Centripetal Flow Field (2, H, W): unit vector field (vy, vx) pointing toward
   the topological center (maximum distance transform point) of each instance.
   Diverges sharply at boundaries between touching instances.
2. Centroid Heatmap (1, H, W): Gaussian peaks centered at each instance's core.
3. Multi-instance Signed Distance Transform (1, H, W): Smooth distance map dipping
   to zero at instance boundaries and negative outside.
4. Canopy Mask (1, H, W): Binary tree cover foreground mask.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
import torch


def compute_tree_flow_targets(
    instance_label: np.ndarray,
    sdt_max_distance: float = 15.0,
    centroid_sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute target maps from an (H, W) instance label map (0=background).

    Args:
        instance_label: (H, W) array with integer instance IDs (0=bg).
        sdt_max_distance: maximum distance cutoff for SDT in pixels.
        centroid_sigma: Gaussian standard deviation for centroid heatmap.

    Returns:
        flow_target: (2, H, W) float32 array with (vy, vx) unit vectors.
        sdt_target: (1, H, W) float32 array in range [-1, 1].
        centroid_target: (1, H, W) float32 array in range [0, 1].
        canopy_target: (1, H, W) float32 array with binary {0, 1}.
    """
    height, width = instance_label.shape
    flow_target = np.zeros((2, height, width), dtype=np.float32)
    centroid_target = np.zeros((1, height, width), dtype=np.float32)
    inside_dist = np.zeros((height, width), dtype=np.float32)

    unique_labels = np.unique(instance_label)
    unique_labels = unique_labels[unique_labels != 0]

    yy, xx = np.mgrid[0:height, 0:width]

    for label in unique_labels:
        mask = (instance_label == label)
        mask_edt = ndimage.distance_transform_edt(mask)
        inside_dist[mask] = mask_edt[mask]

        # Topological center = point deepest inside the crown
        max_idx = np.argmax(mask_edt)
        cy, cx = np.unravel_index(max_idx, (height, width))

        # Vector field pointing toward (cy, cx) for pixels in this instance
        dy = cy - yy[mask]
        dx = cx - xx[mask]
        norm = np.sqrt(dy * dy + dx * dx) + 1e-6
        flow_target[0, mask] = dy / norm
        flow_target[1, mask] = dx / norm

        # Gaussian centroid peak
        gaussian = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * centroid_sigma ** 2))
        np.maximum(centroid_target[0], gaussian, out=centroid_target[0])

    outside_dist = ndimage.distance_transform_edt(instance_label == 0)
    sdt = inside_dist - outside_dist
    sdt_target = np.clip(sdt, -sdt_max_distance, sdt_max_distance) / sdt_max_distance
    sdt_target = sdt_target[np.newaxis, ...].astype(np.float32)

    canopy_target = (instance_label > 0).astype(np.float32)[np.newaxis, ...]

    return flow_target, sdt_target, centroid_target, canopy_target
