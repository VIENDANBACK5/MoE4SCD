"""Mechanism check for H10: how does each head degrade under the domain shift?

EXPLORATORY. H10 found boundary calibration helps out-of-domain 4.5x more in relative
terms than in-domain. The candidate explanation is that a head collapsed toward zero has
no headroom left to lose under shift, so its signal disappears entirely, whereas a
calibrated head degrades gracefully. This measures it directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

_c = '/home/ubuntu/anaconda3/envs/cs2/lib/python3.11/site-packages'
_l = '/home/chung/.local/lib/python3.11/site-packages'
sys.path = [p for p in sys.path if p not in (_l, _c)]
sys.path.insert(0, _c); sys.path.append(_l)
sys.path.insert(0, "/home/chung/RS/Image Segmentation")

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from crown_research.src.pred_cache import load_cache

OUT = Path("crown_research/experiments/h10-ood-robustness/results")
PAIRS = [("evalhold60", "evalhold60", "h6_final"),
         ("test2", "test2", "test2_h6"),
         ("test1", "test1", "test1_h6")]


def seam_band(gt, r=3):
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(gt.astype(np.float32), cv2.MORPH_GRADIENT, k3) > 0
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    return cv2.dilate((grad & (gt >= 2)).astype(np.uint8), kd) > 0


def diag(tag):
    B, C, sy, fy = [], [], [], []
    for surf, bound, canopy, gt in load_cache(tag):
        sub = (slice(None, None, 3), slice(None, None, 3))
        B.append(bound[sub].ravel()); C.append(canopy[sub].ravel())
        sy.append(seam_band(gt)[sub].ravel()); fy.append((gt >= 2)[sub].ravel())
    B, C = np.concatenate(B), np.concatenate(C)
    sy, fy = np.concatenate(sy), np.concatenate(fy)
    return {"B_on_seam": float(B[sy].mean()), "B_off_seam": float(B[~sy].mean()),
            "B_separation": float(B[sy].mean() / max(B[~sy].mean(), 1e-9)),
            "B_AUROC": float(roc_auc_score(sy, B)),
            "C_AUROC": float(roc_auc_score(fy, C)),
            "C_on_fg": float(C[fy].mean())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, old_tag, new_tag in PAIRS:
        for ck, tag in (("old (best_sharp)", old_tag), ("H6 (calibrated)", new_tag)):
            rows.append({"split": split, "checkpoint": ck, **diag(tag)})
            print(f"  {split:<11} {ck}", flush=True)
    df = pd.DataFrame(rows).round(4)
    df.to_csv(OUT / "h10_head_shift.csv", index=False)
    print("\n" + "=" * 118)
    print(df.to_string(index=False))
    print("=" * 118)

    print("\nDegradation from in-domain (test2) to out-of-domain (test1):")
    for ck in ("old (best_sharp)", "H6 (calibrated)"):
        s = df[df.checkpoint == ck].set_index("split")
        print(f"  {ck:<18} B_on_seam {s.loc['test2','B_on_seam']:.4f} -> "
              f"{s.loc['test1','B_on_seam']:.4f} "
              f"({100*(s.loc['test1','B_on_seam']/s.loc['test2','B_on_seam']-1):+.0f}%)   "
              f"B_AUROC {s.loc['test2','B_AUROC']:.3f} -> {s.loc['test1','B_AUROC']:.3f}   "
              f"C_AUROC {s.loc['test2','C_AUROC']:.3f} -> {s.loc['test1','C_AUROC']:.3f}")


if __name__ == "__main__":
    main()
