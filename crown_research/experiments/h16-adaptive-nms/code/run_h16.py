"""H16 — Dynamic Local Scale-Adaptive NMS for Marker Extraction.

Tests whether estimating local crown radius from the surface potential field
and using dynamic suppression R_supp(y, x) = clamp(gamma * R_est, R_min, R_max)
breaks past the invariant Global NMS precision/recall frontier (Finding 22).
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

_c = '/home/ubuntu/anaconda3/envs/cs2/lib/python3.11/site-packages'
_l = '/home/chung/.local/lib/python3.11/site-packages'
sys.path = [p for p in sys.path if p not in (_l, _c)]
sys.path.insert(0, _c); sys.path.append(_l)
sys.path.insert(0, "/home/chung/RS/Image Segmentation")

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crown_research.src.coco_ap import APAccumulator, instance_scores_from_surface
from crown_research.src.pred_cache import load_cache
from crown_research.src.seeding import flood, global_nms_seeds
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import evaluate_metrics

OUT = Path("crown_research/experiments/h16-adaptive-nms/results")
FIG = Path("crown_research/to_human/figures")

MIN_CANOPY, MIN_AREA = 0.25, 25


def _raw_candidate_peaks(surf, canopy, footprint=7, min_apex=0.35, min_canopy=MIN_CANOPY):
    k = max(3, int(footprint) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    peaks = (surf == cv2.dilate(surf, kernel)) & (surf >= min_apex) & (canopy >= min_canopy)
    ys, xs = np.where(peaks)
    if len(ys) == 0:
        return np.empty(0, int), np.empty(0, int), np.empty(0, np.float32)
    scores = surf[ys, xs]
    order = np.argsort(-scores)
    return ys[order], xs[order], scores[order]


def estimate_peak_radii(surf, canopy, ys, xs, scores, rel_thresh=0.45, min_r=25.0, max_r=220.0):
    """Estimates local crown radius around each peak by superlevel support."""
    radii = []
    H, W = surf.shape
    for y, x, s in zip(ys, xs, scores):
        # Local bounding box
        box_r = int(max_r)
        y0, y1 = max(0, y - box_r), min(H, y + box_r + 1)
        x0, x1 = max(0, x - box_r), min(W, x + box_r + 1)

        sub_surf = surf[y0:y1, x0:x1]
        sub_can = canopy[y0:y1, x0:x1]
        py, px = y - y0, x - x0

        # Connected component of {surf >= rel_thresh * s} containing (py, px)
        comp = ((sub_surf >= rel_thresh * s) & (sub_can >= MIN_CANOPY)).astype(np.uint8)
        n_lab, lab = cv2.connectedComponents(comp, connectivity=8)
        target_lab = lab[py, px]
        if target_lab > 0:
            area = float(np.sum(lab == target_lab))
            r_est = np.sqrt(area / np.pi)
        else:
            r_est = min_r
        radii.append(float(np.clip(r_est, min_r, max_r)))
    return np.array(radii, dtype=np.float32)


def dynamic_scale_nms_seeds(surf, canopy, gamma=1.0, rel_thresh=0.45, min_r=30.0, max_r=200.0, min_apex=0.40):
    """Adaptive suppression with radius scaled to individual crown support."""
    ys, xs, scores = _raw_candidate_peaks(surf, canopy, footprint=7, min_apex=min_apex)
    if len(ys) == 0:
        return []

    radii = estimate_peak_radii(surf, canopy, ys, xs, scores, rel_thresh=rel_thresh, min_r=min_r, max_r=max_r)

    kept_y, kept_x, kept_s, kept_r = [], [], [], []
    for y, x, s, r in zip(ys, xs, scores, radii):
        ok = True
        for ky, kx, _, kr in zip(kept_y, kept_x, kept_s, kept_r):
            # Mutual suppression distance: scaled to either the current or existing crown
            d_thr = gamma * max(r, kr)
            if (y - ky) ** 2 + (x - kx) ** 2 < d_thr ** 2:
                ok = False
                break
        if ok:
            kept_y.append(y); kept_x.append(x); kept_s.append(s); kept_r.append(r)

    return [(int(y), int(x), float(s)) for y, x, s in zip(kept_y, kept_x, kept_s)]


def audit_seeding_performance(crops, seed_fn):
    acc = APAccumulator()
    f1_list = []
    tot_gt, tot_seeds, tot_hit, tot_dup, tot_orphan = 0, 0, 0, 0, 0

    for surf, bound, canopy, gt in crops:
        seeds = seed_fn(surf, canopy)
        m, insts = flood(seeds, bound, canopy)
        insts = instance_scores_from_surface(m, insts, surf)
        acc.add(m, insts, gt)
        f1_list.append(evaluate_metrics(m, insts, gt)["ap50"])

        gids = np.unique(gt)
        gids = gids[gids >= 2]
        tot_gt += len(gids)
        tot_seeds += len(seeds)

        seen = {}
        for y, x, _ in seeds:
            g = int(gt[y, x])
            if g < 2:
                tot_orphan += 1
            else:
                seen[g] = seen.get(g, 0) + 1
        tot_hit += len(seen)
        tot_dup += sum(v - 1 for v in seen.values())

    r = acc.ap()
    return {
        "COCO AP": r["AP"],
        "AP50": r["AP50"],
        "AP75": r["AP75"],
        "F1@50": float(np.mean(f1_list)),
        "seed_ratio": round(tot_seeds / max(1, tot_gt), 3),
        "hit_rate_%": round(100.0 * tot_hit / max(1, tot_gt), 2),
        "dup_per_gt": round(tot_dup / max(1, tot_gt), 3),
        "orphan_per_gt": round(tot_orphan / max(1, tot_gt), 3),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    crops_eval = list(load_cache("evalhold60_shadow"))
    print(f"[H16] Loaded {len(crops_eval)} crops for evalhold60_shadow", flush=True)

    # 1. Baseline Global NMS at optimal point (md=160, ma=0.50)
    base_res = audit_seeding_performance(crops_eval, lambda s, c: global_nms_seeds(s, c, min_dist=160, min_apex=0.50))
    print(f"Global NMS Baseline: COCO AP = {base_res['COCO AP']:.2f}, AP50 = {base_res['AP50']:.2f}, F1@50 = {base_res['F1@50']:.2f}%, Seed Ratio = {base_res['seed_ratio']}")

    # 2. Sweep Adaptive NMS parameters (focused fast grid)
    results = []
    gammas = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    rel_threshs = [0.40, 0.50]
    min_rs = [25.0, 40.0]
    min_apexs = [0.40, 0.50]

    for g, rt, mr, ma in product(gammas, rel_threshs, min_rs, min_apexs):
        def _fn(surf, canopy, _g=g, _rt=rt, _mr=mr, _ma=ma):
            return dynamic_scale_nms_seeds(surf, canopy, gamma=_g, rel_thresh=_rt, min_r=_mr, min_apex=_ma)

        res = audit_seeding_performance(crops_eval, _fn)
        results.append({
            "gamma": g,
            "rel_thresh": rt,
            "min_r": mr,
            "min_apex": ma,
            **res
        })
        print(f"gamma={g:.1f}, rt={rt:.2f}, mr={mr:.0f}, ma={ma:.2f} -> AP={res['COCO AP']:.2f}, F1={res['F1@50']:.2f}%, Ratio={res['seed_ratio']}", flush=True)

    df = pd.DataFrame(results).round(4)
    df.to_csv(OUT / "h16_adaptive_nms_sweep.csv", index=False)

    top5 = df.sort_values("COCO AP", ascending=False).head(5)
    print("\n" + "=" * 90)
    print("Top 5 Dynamic Scale-Adaptive NMS Configurations (evalhold60):")
    print(top5.to_string(index=False))
    print("=" * 90)

    best = top5.iloc[0]
    print(f"\nBest Adaptive NMS: AP = {best['COCO AP']} (vs Baseline {base_res['COCO AP']}), F1@50 = {best['F1@50']}% (vs Baseline {base_res['F1@50']}%)")


if __name__ == "__main__":
    main()
