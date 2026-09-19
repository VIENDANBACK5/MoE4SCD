"""Head-quality diagnostic (EXPLORATORY, triggered by H4's flat lambda sweep).

The lambda sweep moved mAP@50 by < 0.2 across lambda in [0, 2], which can only happen
if `canopy - lambda*B` is essentially `canopy` — i.e. B is near zero almost everywhere.
This measures each head against its own target to find out which ones actually carry
signal.
"""
from __future__ import annotations

import sys
from pathlib import Path

_conda_p = '/home/ubuntu/anaconda3/envs/cs2/lib/python3.11/site-packages'
_local_p = '/home/chung/.local/lib/python3.11/site-packages'
sys.path = [p for p in sys.path if p not in (_local_p, _conda_p)]
sys.path.insert(0, _conda_p)
sys.path.append(_local_p)
sys.path.insert(0, "/home/chung/RS/Image Segmentation")

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from crown_research.src.pred_cache import load_cache

OUT = Path("crown_research/experiments/h4-coupled-gate/results")


def gt_seam_band(gt, radius=3):
    """Inter-crown seams only (crown|crown contacts), dilated to a band of `radius`."""
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(gt.astype(np.float32), cv2.MORPH_GRADIENT, k3) > 0
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    return cv2.dilate((grad & (gt >= 2)).astype(np.uint8), kd) > 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag in ("eval30", "evalhold60"):
        Bs, Cs, Ss, seam_y, fg_y = [], [], [], [], []
        for surf, bound, canopy, gt in load_cache(tag):
            sub = (slice(None, None, 3), slice(None, None, 3))  # subsample for tractable AUC
            Bs.append(bound[sub].ravel()); Cs.append(canopy[sub].ravel()); Ss.append(surf[sub].ravel())
            seam_y.append(gt_seam_band(gt)[sub].ravel())
            fg_y.append((gt >= 2)[sub].ravel())
        B, C, S = np.concatenate(Bs), np.concatenate(Cs), np.concatenate(Ss)
        sy, fy = np.concatenate(seam_y), np.concatenate(fg_y)

        rows.append({
            "split": tag, "head": "boundary B", "target": "GT seam band (r=3)",
            "pos_rate_%": round(100.0 * sy.mean(), 3),
            "pred_mean": round(float(B.mean()), 4), "pred_p50": round(float(np.percentile(B, 50)), 4),
            "pred_p99": round(float(np.percentile(B, 99)), 4), "pred_max": round(float(B.max()), 4),
            "mean_on_pos": round(float(B[sy].mean()), 4), "mean_on_neg": round(float(B[~sy].mean()), 4),
            "AUROC": round(float(roc_auc_score(sy, B)), 4),
            "AP": round(float(average_precision_score(sy, B)), 4),
            "AP_over_chance": round(float(average_precision_score(sy, B)) / float(sy.mean()), 2),
        })
        rows.append({
            "split": tag, "head": "canopy C", "target": "GT foreground",
            "pos_rate_%": round(100.0 * fy.mean(), 3),
            "pred_mean": round(float(C.mean()), 4), "pred_p50": round(float(np.percentile(C, 50)), 4),
            "pred_p99": round(float(np.percentile(C, 99)), 4), "pred_max": round(float(C.max()), 4),
            "mean_on_pos": round(float(C[fy].mean()), 4), "mean_on_neg": round(float(C[~fy].mean()), 4),
            "AUROC": round(float(roc_auc_score(fy, C)), 4),
            "AP": round(float(average_precision_score(fy, C)), 4),
            "AP_over_chance": round(float(average_precision_score(fy, C)) / float(fy.mean()), 2),
        })
        # Does the canopy head itself detect seams? (H3's claim that it is the real seam detector)
        rows.append({
            "split": tag, "head": "1 - canopy C", "target": "GT seam band (r=3)",
            "pos_rate_%": round(100.0 * sy.mean(), 3),
            "pred_mean": round(float(1 - C.mean()), 4), "pred_p50": np.nan,
            "pred_p99": np.nan, "pred_max": np.nan,
            "mean_on_pos": round(float((1 - C)[sy].mean()), 4),
            "mean_on_neg": round(float((1 - C)[~sy].mean()), 4),
            "AUROC": round(float(roc_auc_score(sy, 1 - C)), 4),
            "AP": round(float(average_precision_score(sy, 1 - C)), 4),
            "AP_over_chance": round(float(average_precision_score(sy, 1 - C)) / float(sy.mean()), 2),
        })
        rows.append({
            "split": tag, "head": "surface U", "target": "GT foreground",
            "pos_rate_%": round(100.0 * fy.mean(), 3),
            "pred_mean": round(float(S.mean()), 4), "pred_p50": round(float(np.percentile(S, 50)), 4),
            "pred_p99": round(float(np.percentile(S, 99)), 4), "pred_max": round(float(S.max()), 4),
            "mean_on_pos": round(float(S[fy].mean()), 4), "mean_on_neg": round(float(S[~fy].mean()), 4),
            "AUROC": round(float(roc_auc_score(fy, S)), 4),
            "AP": round(float(average_precision_score(fy, S)), 4),
            "AP_over_chance": round(float(average_precision_score(fy, S)) / float(fy.mean()), 2),
        })
        print(f"  done {tag}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "head_diagnostic.csv", index=False)
    print("\n" + "=" * 150)
    print(df.to_string(index=False))
    print("=" * 150)


if __name__ == "__main__":
    main()
