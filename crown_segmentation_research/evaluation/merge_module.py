"""Post-hoc merge of over-segmented instance-mask fragments.

Implements Section 4 of reports/method_design_crown_confidence_v2.md: after
any instance segmenter (Mask R-CNN here) produces a set of masks for one
image, merge adjacent fragments that plausibly belong to the same crown --
targeted at the large/multi-lobed-crown over-segmentation failure mode
independently confirmed by Freudenberg et al. (2022) and consistent with
BAM's own measured G1B split_rate (0.1676 on BAM_test2).

E(R) here is a simple mean+std RGB appearance descriptor over each mask's
pixels (not a trained embedding branch) -- deliberately the simplest correct
version per this project's "simplicity first" convention; a learned decoder
embedding is a possible upgrade, not a prerequisite to test the mechanism.

Merge distance is expressed directly in physical metres (not multiplied by a
second GSD factor), simplifying the design doc's
`d_phys(c_i, c_j) < delta_d * GSD` to `d_phys(c_i, c_j) < delta_d_m`, since
d_phys is already computed in physical units via GSD -- multiplying by GSD a
second time would not be dimensionally meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Fragment:
    mask: np.ndarray  # (H, W) bool
    score: float


def _pairwise_adjacent(masks: list[np.ndarray], dilation_px: int = 2) -> set[tuple[int, int]]:
    """Fragment pairs whose dilated masks overlap -- i.e. touch or nearly touch."""
    dilated = [ndimage.binary_dilation(m, iterations=dilation_px) for m in masks]
    pairs = set()
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if np.any(dilated[i] & dilated[j]):
                pairs.add((i, j))
    return pairs


def _appearance_embedding(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pixels = image_rgb[mask].astype(np.float64) / 255.0
    if len(pixels) == 0:
        return np.zeros(6, dtype=np.float64)
    return np.concatenate([pixels.mean(axis=0), pixels.std(axis=0)])


def _centroid_m(mask: np.ndarray, gsd_cm: float) -> np.ndarray:
    ys, xs = np.where(mask)
    centroid_px = np.array([xs.mean(), ys.mean()])
    return centroid_px * (gsd_cm / 100.0)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def merge_oversegmented_instances(
    masks: list[np.ndarray],
    scores: list[float],
    image_rgb: np.ndarray,
    gsd_cm: float,
    delta_appearance: float,
    delta_distance_m: float,
    dilation_px: int = 2,
) -> tuple[list[np.ndarray], list[float]]:
    """Merge adjacent fragments whose appearance is similar and centroids close.

    Returns (merged_masks, merged_scores) -- one entry per merged group,
    score = max of the group's member scores (standard convention for
    combining redundant/fragment detections of the same object).
    """
    n = len(masks)
    if n <= 1:
        return list(masks), list(scores)

    embeddings = [_appearance_embedding(image_rgb, m) for m in masks]
    centroids = [_centroid_m(m, gsd_cm) for m in masks]

    union_find = _UnionFind(n)
    for i, j in _pairwise_adjacent(masks, dilation_px=dilation_px):
        appearance_distance = float(np.linalg.norm(embeddings[i] - embeddings[j]))
        physical_distance = float(np.linalg.norm(centroids[i] - centroids[j]))
        if appearance_distance < delta_appearance and physical_distance < delta_distance_m:
            union_find.union(i, j)

    groups: dict[int, list[int]] = {}
    for index in range(n):
        groups.setdefault(union_find.find(index), []).append(index)

    merged_masks: list[np.ndarray] = []
    merged_scores: list[float] = []
    for members in groups.values():
        combined = masks[members[0]].copy()
        for member in members[1:]:
            combined = combined | masks[member]
        merged_masks.append(combined)
        merged_scores.append(max(scores[member] for member in members))

    return merged_masks, merged_scores
