"""Seed-selection and flat-basin flooding primitives shared across experiments."""
from __future__ import annotations

import cv2
import numpy as np

MIN_CANOPY, MIN_AREA = 0.25, 25


def _raw_peaks(surf, canopy, footprint, min_apex, min_canopy):
    k = max(3, int(footprint) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    peaks = (surf == cv2.dilate(surf, kernel)) & (surf >= min_apex) & (canopy >= min_canopy)
    ys, xs = np.where(peaks)
    if len(ys) == 0:
        return np.empty(0, int), np.empty(0, int), np.empty(0, np.float32)
    scores = surf[ys, xs]
    order = np.argsort(-scores)
    return ys[order], xs[order], scores[order]


def global_nms_seeds(surf, canopy, min_dist=24.0, min_apex=0.25, min_canopy=MIN_CANOPY):
    """Baseline: one global suppression radius for every peak."""
    ys, xs, ss = _raw_peaks(surf, canopy, min_dist, min_apex, min_canopy)
    kept_y, kept_x, kept_s = [], [], []
    d2 = min_dist ** 2
    for y, x, s in zip(ys, xs, ss):
        if not any((y - ky) ** 2 + (x - kx) ** 2 < d2 for ky, kx in zip(kept_y, kept_x)):
            kept_y.append(y); kept_x.append(x); kept_s.append(s)
    return [(int(y), int(x), float(s)) for y, x, s in zip(kept_y, kept_x, kept_s)]


def superlevel_radii(surf, canopy, ys, xs, t=0.5, min_canopy=MIN_CANOPY, r_floor=6.0):
    """Per-peak crown radius from the absolute pixel extent of {U >= t}.

    Per-instance normalisation of the surface target rescales values but not support,
    so the superlevel-set component around a peak still carries absolute scale.
    """
    comp = ((surf >= t) & (canopy >= min_canopy)).astype(np.uint8)
    n_lab, lab = cv2.connectedComponents(comp, connectivity=8)
    areas = np.bincount(lab.ravel(), minlength=n_lab).astype(np.float32)
    lids = lab[ys, xs]
    a = np.where(lids > 0, areas[lids], 0.0)
    return np.maximum(np.sqrt(a / np.pi), r_floor)


def adaptive_seeds(surf, canopy, t=0.5, gamma=1.0, min_apex=0.25, min_canopy=MIN_CANOPY,
                   footprint=9, r_floor=6.0):
    """Scale-adaptive NMS: suppression radius read off each peak's own basin extent."""
    ys, xs, ss = _raw_peaks(surf, canopy, footprint, min_apex, min_canopy)
    if len(ys) == 0:
        return []
    radii = superlevel_radii(surf, canopy, ys, xs, t=t, min_canopy=min_canopy, r_floor=r_floor)

    kept = []  # (y, x, score, radius)
    for y, x, s, r in zip(ys, xs, ss, radii):
        ok = True
        for ky, kx, _, kr in kept:
            thr = gamma * max(r, kr)
            if (y - ky) ** 2 + (x - kx) ** 2 < thr * thr:
                ok = False
                break
        if ok:
            kept.append((int(y), int(x), float(s), float(r)))
    return [(y, x, s) for y, x, s, _ in kept]


def flood(seeds, energy, canopy, min_canopy=MIN_CANOPY, min_area=MIN_AREA):
    """Flat-basin watershed: relief is the ridge field alone, basins are flat."""
    H, W = energy.shape
    if not seeds:
        return np.zeros((H, W), np.int32), []
    markers = np.zeros((H, W), np.int32)
    markers[canopy < min_canopy] = 1
    for i, (y, x, _) in enumerate(seeds):
        markers[max(0, y - 1):min(H, y + 2), max(0, x - 1):min(W, x + 2)] = i + 2
    relief = (energy * 255.0).clip(0, 255).astype(np.uint8)
    cv2.watershed(cv2.merge([relief] * 3), markers)

    areas = np.bincount(markers[markers >= 2].ravel(), minlength=len(seeds) + 2)
    insts = []
    for i, (y, x, s) in enumerate(seeds):
        iid = i + 2
        area = int(areas[iid]) if iid < len(areas) else 0
        if area >= min_area:
            insts.append({"id": iid, "apex": (y, x), "score": s, "area": area})
    return markers, insts
