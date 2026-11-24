"""Phase 0 model-agnostic absorption measure (PHASE0_LANDSCAPE_AND_BENCHMARK_20261124.md). Dataset-independent core.

Per image:
- gt: int label raster (0 = background);
- excl: bool[n+1], instances to exclude (cut by the image border / invalid area);
- dets: list of (rle, score), already score-filtered.

Returns one row per non-excluded GT instance:
- area: pixel area;
- hit: matched one-to-one (COCO-style greedy by score, IoU >= 0.5);
- r[d]: max neighbour-area / own-area for touch distances d (0 if no neighbour);
- M[d]: absorption event (not a hit, and some prediction covers >= 50% of the instance and >= 50% of a d-neighbour);
- density: number of other GT centroids within 3 own diameters.
"""
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage as ndi

D_TOUCH = (1, 2, 5)   # primary 2 px; 1 and 5 px for sensitivity (protocol)


def neighbours(gt, d):
    """{id: set(neighbour ids)} for instances within d px (square neighbourhood)."""
    H, W = gt.shape; nb = {}
    for dy in range(-d, d + 1):
        for dx in range(-d, d + 1):
            if dy == 0 and dx == 0: continue
            a = gt[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)]
            b = gt[max(0, -dy):H + min(0, -dy), max(0, -dx):W + min(0, -dx)]
            m = (a > 0) & (b > 0) & (a != b)
            if not m.any(): continue
            for u, v in np.unique(np.stack([a[m], b[m]], 1), axis=0):
                nb.setdefault(int(u), set()).add(int(v)); nb.setdefault(int(v), set()).add(int(u))
    return nb


def image_rows(gt, excl, dets):
    ids = [g for g in range(1, int(gt.max()) + 1)]
    area = np.bincount(gt.ravel(), minlength=len(ids) + 1).astype(np.int64)
    ids = [g for g in ids if area[g] > 0]
    if not ids: return []
    grle = [mu.encode(np.asfortranarray((gt == g).astype(np.uint8))) for g in ids]
    pos = {g: k for k, g in enumerate(ids)}
    dets = sorted(dets, key=lambda x: -x[1]); prle = [d[0] for d in dets]
    if prle:
        iou = mu.iou(prle, grle, [0] * len(grle))                 # (np, ng)
        cov = mu.iou(grle, prle, [1] * len(prle))                 # (ng, np) = |g ∩ P| / |g|
    else:
        iou = np.zeros((0, len(ids))); cov = np.zeros((len(ids), 0))
    hit = np.zeros(len(ids), bool)
    for p in range(len(prle)):                                    # one-to-one greedy by score
        cand = np.where(~hit & (iou[p] >= .5))[0]
        if len(cand): hit[cand[np.argmax(iou[p, cand])]] = True
    nbs = {d: neighbours(gt, d) for d in D_TOUCH}
    com = ndi.center_of_mass(np.ones_like(gt), gt, ids); com = np.array(com)
    diam = 2 * np.sqrt(area[ids] / np.pi)
    slices = ndi.find_objects(gt)
    rows = []
    for k, g in enumerate(ids):
        if excl[g]: continue
        r, M = {}, {}
        for d in D_TOUCH:
            nb = [b for b in nbs[d].get(g, ()) if b in pos]
            r[d] = float(max(area[b] for b in nb) / area[g]) if nb else 0.0
            ab = False
            if not hit[k] and nb and cov.shape[1]:
                own = cov[k] >= .5
                if own.any():
                    for b in nb:
                        if (own & (cov[pos[b]] >= .5)).any(): ab = True; break
            M[d] = ab
        dens = int((np.hypot(*(com - com[k]).T) <= 3 * diam[k]).sum() - 1)
        sl = slices[g - 1]; thick = float(ndi.distance_transform_edt(np.pad(gt[sl] == g, 1)).max())   # max inscribed radius (px); added post-hoc for the sliver check
        rows.append({"id": g, "area": int(area[g]), "hit": bool(hit[k]), "r": r, "M": M, "density": dens, "thick": thick,
                     "cy": float(com[k][0]), "cx": float(com[k][1])})
    return rows


def stratum(r):
    if r == 0: return "non_touching"
    if r < 2: return "balanced"
    if r < 4: return "moderate"
    return "strong"


def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; dd = 1 + z * z / n; c = (p + z * z / (2 * n)) / dd; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    return [round(c - h, 4), round(c + h, 4)]


def summarise(rows, group_key, size_edges, dens_edges, d=2, n_boot=1000, seed=0):
    """Per-stratum absorption rates (Wilson CIs); the strong-vs-balanced difference standardised to the strong stratum's
    size × density distribution, with a group-bootstrap CI (group = image/tile); consequence measures."""
    for x in rows:
        x["_s"] = stratum(x["r"][d]); x["_sz"] = int(np.searchsorted(size_edges, x["area"], side="right"))
        x["_dn"] = int(np.searchsorted(dens_edges, x["density"], side="right"))
    out = {"d": d, "strata": {}}
    for s in ("non_touching", "balanced", "moderate", "strong"):
        rr = [x for x in rows if x["_s"] == s]; k = sum(x["M"][d] for x in rr); h = sum(x["hit"] for x in rr)
        out["strata"][s] = {"n": len(rr), "absorbed": int(k), "rate": k / len(rr) if rr else None, "ci": wilson(k, len(rr)),
                            "hit_rate": h / len(rr) if rr else None}

    def std_diff(rr):
        st = [x for x in rr if x["_s"] == "strong"]; bl = [x for x in rr if x["_s"] == "balanced"]
        cells = {}   # (size bin, density bin) -> [n_strong, absorbed_strong, n_balanced, absorbed_balanced]
        for x in st:
            c = cells.setdefault((x["_sz"], x["_dn"]), [0, 0, 0, 0]); c[0] += 1; c[1] += x["M"][d]
        for x in bl:
            c = cells.setdefault((x["_sz"], x["_dn"]), [0, 0, 0, 0]); c[2] += 1; c[3] += x["M"][d]
        use = {c: v for c, v in cells.items() if v[0] > 0 and v[2] >= 10}
        ns = sum(v[0] for v in use.values())
        if ns == 0: return None, None, 0
        rs = sum(v[1] for v in use.values()) / ns
        rb = sum(v[0] / ns * v[3] / v[2] for v in use.values())
        return rs, rb, ns

    rs, rb, ns = std_diff(rows)
    groups = sorted({x[group_key] for x in rows}); by = {g: [x for x in rows if x[group_key] == g] for g in groups}
    rng = np.random.default_rng(seed); boots = []
    for _ in range(n_boot):
        samp = [x for g in rng.choice(groups, len(groups), replace=True) for x in by[g]]
        a, b, _ = std_diff(samp)
        if a is not None: boots.append(a - b)
    out["standardised"] = {"strong_rate_in_cells": rs, "balanced_rate_std_to_strong": rb, "n_strong_in_cells": ns,
                           "diff": (rs - rb) if rs is not None else None,
                           "diff_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None,
                           "ratio": (rs / rb) if rb else None}
    miss = [x for x in rows if not x["hit"]]; n = len(rows)
    out["consequence"] = {"n_instances": n, "recall": sum(x["hit"] for x in rows) / n if n else None,
                          "share_of_misses_that_are_absorption": sum(x["M"][d] for x in miss) / len(miss) if miss else None,
                          "recall_gain_if_strong_absorptions_were_hits": sum(x["M"][d] for x in rows if x["_s"] == "strong") / n if n else None,
                          "recall_gain_if_all_absorptions_were_hits": sum(x["M"][d] for x in rows) / n if n else None}
    for x in rows: del x["_s"], x["_sz"], x["_dn"]
    return out
