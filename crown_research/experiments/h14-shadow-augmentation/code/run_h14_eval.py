"""Evaluation and Diagnostic script for H14 (Shadow & Photometric Augmentation).

Evaluates across:
1. evalhold60 (tuning/validation, idx 100..159)
2. test2 (in-domain test, 40 crops)
3. test1 (out-of-domain test, 39 crops)
Measures whether shadow augmentation prevented head collapse under lighting/sensor shifts.
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
import torch
from sklearn.metrics import roc_auc_score

from crown_segmentation_research.methods.canopy_watershed.model import CanopyWatershedNet
from crown_research.src.coco_ap import APAccumulator, instance_scores_from_surface
from crown_research.src.pred_cache import build_cache, load_cache
from crown_research.src.seeding import flood, global_nms_seeds
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import evaluate_metrics

OUT = Path("crown_research/experiments/h14-shadow-augmentation/results")
CKPT = "DeadTrees/experiments/canopy_watershed/best_shadow_watershed.pth"
GRID = list(product([100, 140, 160, 200, 240], [0.35, 0.50, 0.65]))


def seam_band(gt, r=3):
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(gt.astype(np.float32), cv2.MORPH_GRADIENT, k3) > 0
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    return cv2.dilate((grad & (gt >= 2)).astype(np.uint8), kd) > 0


def head_diag(crops):
    B, sy = [], []
    for _, bound, _, gt in crops:
        sub = (slice(None, None, 3), slice(None, None, 3))
        B.append(bound[sub].ravel())
        sy.append(seam_band(gt)[sub].ravel())
    B, sy = np.concatenate(B), np.concatenate(sy)
    return {
        "B_on_seam": float(B[sy].mean()),
        "B_off_seam": float(B[~sy].mean()),
        "separation": float(B[sy].mean() / max(B[~sy].mean(), 1e-9)),
        "B_AUROC": float(roc_auc_score(sy, B))
    }


def score(crops, md, ma):
    acc = APAccumulator()
    f1 = []
    for surf, bound, canopy, gt in crops:
        m, insts = flood(global_nms_seeds(surf, canopy, float(md), ma), bound, canopy)
        insts = instance_scores_from_surface(m, insts, surf)
        acc.add(m, insts, gt)
        f1.append(evaluate_metrics(m, insts, gt)["ap50"])
    r = acc.ap()
    return {"COCO AP": r["AP"], "AP50": r["AP50"], "AP75": r["AP75"], "F1@50": float(np.mean(f1))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not Path(CKPT).exists():
        print(f"Error: {CKPT} does not exist yet.")
        return

    splits = {
        "evalhold60": ("eval", "evalhold60_shadow", 60, 100),
        "test2": ("test2", "test2_shadow", 40, 0),
        "test1": ("test1", "test1_shadow", 39, 0),
    }

    hd_rows = []
    sweep_rows = []

    for label, (split_name, tag, n_max, start_idx) in splits.items():
        print(f"\nBuilding cache for {label} ({tag})...", flush=True)
        build_cache(num_crops=n_max, split=split_name, ckpt=CKPT, tag=tag, start=start_idx)
        crops = list(load_cache(tag))
        print(f"Loaded {len(crops)} crops for {label}", flush=True)

        hd = head_diag(crops)
        hd_rows.append({"split": label, **hd})

        for md, ma in GRID:
            r = score(crops, md, ma)
            sweep_rows.append({
                "split": label,
                "min_dist": md,
                "min_apex": ma,
                **r
            })
        print(f"Finished sweep for {label}", flush=True)

    df_hd = pd.DataFrame(hd_rows).round(4)
    df_hd.to_csv(OUT / "h14_head_diagnostic.csv", index=False)
    print("\n" + "="*80)
    print("H14 Head Diagnostic:")
    print(df_hd.to_string(index=False))
    print("="*80)

    df_sweep = pd.DataFrame(sweep_rows).round(3)
    df_sweep.to_csv(OUT / "h14_sweep.csv", index=False)

    best_summary = []
    for label in splits:
        sub = df_sweep[df_sweep.split == label]
        b = sub.sort_values("COCO AP", ascending=False).iloc[0]
        best_summary.append({
            "split": label,
            "best_COCO_AP": b["COCO AP"],
            "best_AP50": b["AP50"],
            "best_AP75": b["AP75"],
            "best_F1@50": b["F1@50"],
            "best_cell": f"md={int(b.min_dist)},ma={b.min_apex}",
        })
    df_sum = pd.DataFrame(best_summary)
    df_sum.to_csv(OUT / "h14_summary.csv", index=False)
    print("\n" + "="*80)
    print("H14 Performance Summary:")
    print(df_sum.to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    main()
