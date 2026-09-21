"""H13 — Re-sweep the NMS seeding frontier across models:
Baseline (original), H6 (calibrated B), H11 (aux control), H12 (wide seam).
Tests whether boundary calibration shifts the precision/recall frontier or if the
curve is invariant with respect to marker selection.
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

OUT = Path("crown_research/experiments/h13-frontier-resweep/results")
FIG = Path("crown_research/to_human/figures")

MODELS = {
    "Original Baseline": "evalhold60",
    "H6 Calibrated B": "h6_final",
    "H11 Matched Epochs": "h11_x2",
    "H12 Wide Seam": "h12_wide",
}

# Grid of min_dist and min_apex
DIST_GRID = [24, 40, 60, 80, 100, 120, 140, 160, 180, 200, 240]
APEX_GRID = [0.25, 0.35, 0.45, 0.55, 0.65]


def audit_and_score(crops, md, ma):
    acc = APAccumulator()
    f1_list = []
    tot_gt = 0
    tot_seeds = 0
    tot_hit = 0
    tot_dup = 0
    tot_orphan = 0

    for surf, bound, canopy, gt in crops:
        seeds = global_nms_seeds(surf, canopy, float(md), ma)
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
    ratio = tot_seeds / max(1, tot_gt)
    hit_rate = 100.0 * tot_hit / max(1, tot_gt)
    dup_per_gt = tot_dup / max(1, tot_gt)
    orphan_per_gt = tot_orphan / max(1, tot_gt)

    return {
        "COCO AP": r["AP"],
        "AP50": r["AP50"],
        "AP75": r["AP75"],
        "F1@50": float(np.mean(f1_list)),
        "seed_ratio": ratio,
        "hit_rate_%": hit_rate,
        "dup_per_gt": dup_per_gt,
        "orphan_per_gt": orphan_per_gt,
        "n_seeds": tot_seeds,
        "n_gt": tot_gt,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for model_name, cache_tag in MODELS.items():
        crops = list(load_cache(cache_tag))
        print(f"Loaded {len(crops)} crops for {model_name} ({cache_tag})...", flush=True)
        if not crops:
            continue

        for md, ma in product(DIST_GRID, APEX_GRID):
            res = audit_and_score(crops, md, ma)
            all_rows.append({
                "model": model_name,
                "min_dist": md,
                "min_apex": ma,
                **res
            })

    df = pd.DataFrame(all_rows).round(4)
    df.to_csv(OUT / "h13_frontier_sweep.csv", index=False)
    print(f"Saved {len(df)} sweep points to {OUT / 'h13_frontier_sweep.csv'}")

    best_rows = []
    for model_name in MODELS:
        sub = df[df.model == model_name]
        if sub.empty:
            continue
        best_ap = sub.sort_values("COCO AP", ascending=False).iloc[0]
        best_f1 = sub.sort_values("F1@50", ascending=False).iloc[0]
        best_rows.append({
            "model": model_name,
            "best_COCO_AP": best_ap["COCO AP"],
            "best_COCO_AP_cell": f"md={int(best_ap.min_dist)},ma={best_ap.min_apex}",
            "best_COCO_AP_ratio": round(best_ap["seed_ratio"], 3),
            "best_F1@50": best_f1["F1@50"],
            "best_F1@50_cell": f"md={int(best_f1.min_dist)},ma={best_f1.min_apex}",
            "best_F1@50_ratio": round(best_f1["seed_ratio"], 3),
        })
    df_best = pd.DataFrame(best_rows)
    df_best.to_csv(OUT / "h13_best_summary.csv", index=False)
    print("\n" + "="*95)
    print(df_best.to_string(index=False))
    print("="*95)

    # Plot frontier (F1@50 vs seed ratio)
    plt.figure(figsize=(10, 6))
    colors = {
        "Original Baseline": "#7f7f7f",
        "H6 Calibrated B": "#1f77b4",
        "H11 Matched Epochs": "#2ca02c",
        "H12 Wide Seam": "#d62728",
    }
    markers = {
        "Original Baseline": "o",
        "H6 Calibrated B": "s",
        "H11 Matched Epochs": "^",
        "H12 Wide Seam": "D",
    }

    for model_name in MODELS:
        sub = df[df.model == model_name]
        if sub.empty:
            continue
        plt.scatter(
            sub["seed_ratio"], sub["F1@50"],
            c=colors.get(model_name, "blue"),
            marker=markers.get(model_name, "o"),
            alpha=0.65,
            s=40,
            label=f"{model_name} (max: {sub['F1@50'].max():.1f}%)"
        )

    plt.axvline(1.0, color="black", linestyle=":", alpha=0.5, label="1.0 seed/crown")
    plt.axhline(69.03, color="#c98a2a", linestyle="--", alpha=0.7, label="Oracle seeds ceiling (69.0%)")
    plt.xlabel("Seeds per ground-truth crown (seed_ratio)", fontsize=11)
    plt.ylabel("F1@50 (%) — held-out crops", fontsize=11)
    plt.title("H13: NMS Seeding Frontier Across Calibrated Model Generations", fontweight="bold", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, facecolor="white", loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG / "h13_seeding_frontier_comparison.png", dpi=120)
    plt.savefig(OUT / "h13_frontier_plot.png", dpi=120)
    plt.close()

    # Plot frontier (COCO AP vs seed ratio)
    plt.figure(figsize=(10, 6))
    for model_name in MODELS:
        sub = df[df.model == model_name]
        if sub.empty:
            continue
        plt.scatter(
            sub["seed_ratio"], sub["COCO AP"],
            c=colors.get(model_name, "blue"),
            marker=markers.get(model_name, "o"),
            alpha=0.65,
            s=40,
            label=f"{model_name} (max: {sub['COCO AP'].max():.2f})"
        )

    plt.axvline(1.0, color="black", linestyle=":", alpha=0.5, label="1.0 seed/crown")
    plt.xlabel("Seeds per ground-truth crown (seed_ratio)", fontsize=11)
    plt.ylabel("COCO AP — held-out crops", fontsize=11)
    plt.title("H13: COCO AP vs Seeding Ratio Across Model Generations", fontweight="bold", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, facecolor="white", loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG / "h13_coco_ap_frontier_comparison.png", dpi=120)
    plt.savefig(OUT / "h13_coco_ap_frontier_plot.png", dpi=120)
    plt.close()
    print("Saved H13 comparison figures.")

if __name__ == "__main__":
    main()
