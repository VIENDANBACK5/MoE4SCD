"""H15 — Topometric Relief Flooding for Watershed Instance Segmentation.

Tests whether replacing flat-basin relief (energy = B) with inverted potential
relief E = (1 - U)^gamma + alpha * B improves boundary placement, saddle point
separation, and higher-IoU mask quality (AP75 / COCO AP).
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
from crown_research.src.seeding import global_nms_seeds
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import evaluate_metrics

OUT = Path("crown_research/experiments/h15-topometric-relief/results")
FIG = Path("crown_research/to_human/figures")

MIN_CANOPY, MIN_AREA = 0.25, 25


def flood_topometric(seeds, surf, bound, canopy, mode="composite", gamma=1.0, alpha=1.0, min_canopy=MIN_CANOPY, min_area=MIN_AREA):
    """Floods with generalized topometric potential energy."""
    H, W = bound.shape
    if not seeds:
        return np.zeros((H, W), np.int32), []

    markers = np.zeros((H, W), np.int32)
    markers[canopy < min_canopy] = 1
    for i, (y, x, _) in enumerate(seeds):
        markers[max(0, y - 1):min(H, y + 2), max(0, x - 1):min(W, x + 2)] = i + 2

    # Construct relief surface
    if mode == "flat_b":
        # Baseline: flat basin, ridge alone
        relief_norm = bound
    elif mode == "inv_surf":
        # Pure inverted potential
        relief_norm = np.clip(1.0 - (surf ** gamma), 0.0, 1.0)
    elif mode == "composite":
        # Additive composite
        pot = np.clip(1.0 - (surf ** gamma), 0.0, 1.0)
        relief_norm = np.clip(pot + alpha * bound, 0.0, 1.0 + alpha) / (1.0 + alpha)
    elif mode == "mult_composite":
        # Multiplicative composite: saddle ridges scale with boundary probability
        pot = np.clip(1.0 - (surf ** gamma), 0.0, 1.0)
        relief_norm = np.clip(pot * (1.0 + alpha * bound), 0.0, 1.0 + alpha) / (1.0 + alpha)
    elif mode == "grad_composite":
        # Gradient of surface + boundary
        gx = cv2.Sobel(surf, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(surf, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        relief_norm = np.clip(grad_mag + alpha * bound, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    relief = (relief_norm * 255.0).clip(0, 255).astype(np.uint8)
    cv2.watershed(cv2.merge([relief] * 3), markers)

    areas = np.bincount(markers[markers >= 2].ravel(), minlength=len(seeds) + 2)
    insts = []
    for i, (y, x, s) in enumerate(seeds):
        iid = i + 2
        area = int(areas[iid]) if iid < len(areas) else 0
        if area >= min_area:
            insts.append({"id": iid, "apex": (y, x), "score": s, "area": area})
    return markers, insts


def evaluate_relief_config(crops, md=160, ma=0.50, mode="composite", gamma=1.0, alpha=1.0):
    acc = APAccumulator()
    f1_list = []
    for surf, bound, canopy, gt in crops:
        seeds = global_nms_seeds(surf, canopy, float(md), ma)
        m, insts = flood_topometric(seeds, surf, bound, canopy, mode=mode, gamma=gamma, alpha=alpha)
        insts = instance_scores_from_surface(m, insts, surf)
        acc.add(m, insts, gt)
        f1_list.append(evaluate_metrics(m, insts, gt)["ap50"])

    r = acc.ap()
    return {
        "COCO AP": r["AP"],
        "AP50": r["AP50"],
        "AP75": r["AP75"],
        "F1@50": float(np.mean(f1_list)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    splits = {
        "evalhold60 (Shadow H14)": "evalhold60_shadow",
        "test2 In-Domain (Shadow H14)": "test2_shadow",
        "test1 OOD (Shadow H14)": "test1_shadow",
        "evalhold60 (H6 Calibrated)": "h6_final",
    }

    print("=" * 90)
    print("H15: Sweeping Topometric Relief Formulations on Cached Predictions")
    print("=" * 90)

    # 1. Sweep modes on evalhold60_shadow
    crops_eval = list(load_cache("evalhold60_shadow"))
    print(f"Loaded {len(crops_eval)} crops for evalhold60_shadow")

    rows = []
    # Baseline: Flat B
    b_res = evaluate_relief_config(crops_eval, mode="flat_b")
    rows.append({"mode": "flat_b (Baseline)", "gamma": 1.0, "alpha": 1.0, **b_res})
    print(f"Baseline (flat_b): COCO AP = {b_res['COCO AP']:.2f}, AP50 = {b_res['AP50']:.2f}, AP75 = {b_res['AP75']:.2f}")

    # Pure inverted surface: inv_surf
    for g in [0.5, 0.75, 1.0, 1.5, 2.0]:
        r = evaluate_relief_config(crops_eval, mode="inv_surf", gamma=g)
        rows.append({"mode": "inv_surf", "gamma": g, "alpha": 0.0, **r})

    # Additive composite: (1 - U^gamma) + alpha * B
    for g in [0.5, 1.0, 1.5]:
        for a in [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
            r = evaluate_relief_config(crops_eval, mode="composite", gamma=g, alpha=a)
            rows.append({"mode": "composite", "gamma": g, "alpha": a, **r})

    # Multiplicative composite: (1 - U^gamma) * (1 + alpha * B)
    for g in [0.5, 1.0, 1.5]:
        for a in [0.5, 1.0, 2.0, 4.0, 8.0]:
            r = evaluate_relief_config(crops_eval, mode="mult_composite", gamma=g, alpha=a)
            rows.append({"mode": "mult_composite", "gamma": g, "alpha": a, **r})

    df_eval = pd.DataFrame(rows).round(4)
    df_eval.to_csv(OUT / "h15_eval_sweep.csv", index=False)

    top5 = df_eval.sort_values("COCO AP", ascending=False).head(10)
    print("\nTop 10 Relief Formulations on evalhold60_shadow:")
    print(top5.to_string(index=False))

    best_mode = top5.iloc[0]["mode"]
    best_gamma = top5.iloc[0]["gamma"]
    best_alpha = top5.iloc[0]["alpha"]
    print(f"\nBest configuration: {best_mode} (gamma={best_gamma}, alpha={best_alpha})")

    # 2. Test best configuration across all splits vs Baseline
    split_comparison = []
    for split_label, tag in splits.items():
        crops = list(load_cache(tag))
        if not crops:
            continue
        base_r = evaluate_relief_config(crops, mode="flat_b")
        best_r = evaluate_relief_config(crops, mode=best_mode, gamma=best_gamma, alpha=best_alpha)
        split_comparison.append({
            "split": split_label,
            "Baseline_AP": base_r["COCO AP"],
            "Baseline_AP50": base_r["AP50"],
            "Baseline_AP75": base_r["AP75"],
            "Topometric_AP": best_r["COCO AP"],
            "Topometric_AP50": best_r["AP50"],
            "Topometric_AP75": best_r["AP75"],
            "AP_Gain": round(best_r["COCO AP"] - base_r["COCO AP"], 2),
            "AP75_Gain": round(best_r["AP75"] - base_r["AP75"], 2),
        })

    df_comp = pd.DataFrame(split_comparison)
    df_comp.to_csv(OUT / "h15_split_comparison.csv", index=False)
    print("\n" + "=" * 90)
    print("Cross-Split Comparison: Baseline Flat-B vs Best Topometric Relief")
    print("=" * 90)
    print(df_comp.to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()
