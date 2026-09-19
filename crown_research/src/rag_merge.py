"""Region-adjacency merging of over-segmented watershed basins.

Once basins exist, the predicted boundary field can be read *along the actual shared
seam* between two of them — information that peak-spacing NMS never has access to.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def fill_watershed_lines(markers: np.ndarray) -> np.ndarray:
    """Assign cv2.watershed's -1 ridge pixels to their nearest labelled region."""
    unknown = markers <= 0
    if not unknown.any():
        return markers.copy()
    _, (iy, ix) = ndimage.distance_transform_edt(unknown, return_indices=True)
    filled = markers.copy()
    filled[unknown] = markers[iy[unknown], ix[unknown]]
    return filled


def seam_statistics(filled: np.ndarray, boundary: np.ndarray, max_label: int):
    """Mean boundary evidence and pixel length for every adjacent basin pair.

    Returns (sum_B, count) as dense (max_label+1, max_label+1) upper-triangular arrays.
    """
    n = max_label + 1
    s = np.zeros((n, n), np.float64)
    c = np.zeros((n, n), np.int64)

    for a_sl, b_sl in (((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
                       ((slice(0, -1), slice(None)), (slice(1, None), slice(None)))):
        la, lb = filled[a_sl], filled[b_sl]
        m = (la >= 2) & (lb >= 2) & (la != lb)
        if not m.any():
            continue
        u = np.minimum(la[m], lb[m]).astype(np.int64)
        v = np.maximum(la[m], lb[m]).astype(np.int64)
        keep = v < n
        u, v = u[keep], v[keep]
        # boundary evidence at the seam = stronger of the two straddling pixels
        ev = np.maximum(boundary[a_sl][m], boundary[b_sl][m])[keep].astype(np.float64)
        flat = u * n + v
        np.add.at(s.reshape(-1), flat, ev)
        np.add.at(c.reshape(-1), flat, 1)
    return s, c


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def merge_basins(markers: np.ndarray, boundary: np.ndarray, tau: float = 0.35,
                 l_min: int = 15, min_area: int = 25, gt_map: np.ndarray | None = None):
    """Merge adjacent basins whose shared seam carries weak boundary evidence.

    If `gt_map` is given, the seam criterion is replaced by an ORACLE rule (merge iff
    both basins have the same dominant GT label) — used to measure the ceiling of the
    merge framework itself, independent of the criterion.
    """
    filled = fill_watershed_lines(markers)
    labels = np.unique(filled)
    labels = labels[labels >= 2]
    if len(labels) <= 1:
        return _relabel(filled, min_area)

    max_label = int(labels.max())
    uf = _UF(max_label + 1)

    if gt_map is None:
        s, c = seam_statistics(filled, boundary, max_label)
        au, av = np.nonzero(c >= l_min)
        if len(au):
            mean_ev = s[au, av] / c[au, av]
            for a, b in zip(au[mean_ev < tau], av[mean_ev < tau]):
                uf.union(int(a), int(b))
    else:
        dom = {}
        for lab in labels:
            g = gt_map[filled == lab]
            g = g[g >= 2]
            dom[int(lab)] = int(np.bincount(g).argmax()) if len(g) else -1
        _, c = seam_statistics(filled, np.zeros_like(boundary), max_label)
        au, av = np.nonzero(c >= l_min)
        for a, b in zip(au, av):
            a, b = int(a), int(b)
            if dom[a] >= 2 and dom[a] == dom[b]:
                uf.union(a, b)

    lut = np.arange(max_label + 1)
    for lab in labels:
        lut[lab] = uf.find(int(lab))
    merged = np.where(filled >= 2, lut[np.clip(filled, 0, max_label)], filled)
    return _relabel(merged, min_area)


def _relabel(m: np.ndarray, min_area: int):
    """Compact labels to 2..K+1 and emit instance records, dropping small fragments."""
    out = np.zeros_like(m)
    insts = []
    nid = 2
    for lab in np.unique(m):
        if lab < 2:
            continue
        mask = m == lab
        area = int(mask.sum())
        if area < min_area:
            continue
        out[mask] = nid
        ys, xs = np.nonzero(mask)
        insts.append({"id": nid, "apex": (int(ys.mean()), int(xs.mean())),
                      "score": 1.0, "area": area})
        nid += 1
    return out, insts
