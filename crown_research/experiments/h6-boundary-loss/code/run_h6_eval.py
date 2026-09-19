"""H6 evaluation — old vs calibrated boundary head, head-level and end-to-end.

Usage: run_h6_eval.py --ckpt <path> --tag <cache tag> [--quick]
Builds a prediction cache for the given checkpoint, then scores it against the frozen
baseline settings. See ../protocol.md.
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
sys.path.insert(0, "/home/chung/RS/Image Segmentation/crown_research/experiments/h5-boundary-calibration/code")

import argparse
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from crown_research.src.pred_cache import build_cache, load_cache
from crown_research.src.rag_merge import merge_basins
from crown_research.src.seeding import flood, global_nms_seeds
from run_h5 import merge_percentile
from crown_segmentation_research.methods.boundary_refinement.metrics_dao import (
    compute_boundary_iou, compute_dao_segmentation_indices)
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import evaluate_metrics

OUT = Path("crown_research/experiments/h6-boundary-loss/results")
MD, MA = 160.0, 0.50


def seam_band(gt, radius=3):
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(gt.astype(np.float32), cv2.MORPH_GRADIENT, k3) > 0
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    return cv2.dilate((grad & (gt >= 2)).astype(np.uint8), kd) > 0


def head_diag(tag, label):
    Bs, Cs, Ss, sy, fy = [], [], [], [], []
    for surf, bound, canopy, gt in load_cache(tag):
        sub = (slice(None, None, 3), slice(None, None, 3))
        Bs.append(bound[sub].ravel()); Cs.append(canopy[sub].ravel()); Ss.append(surf[sub].ravel())
        sy.append(seam_band(gt)[sub].ravel()); fy.append((gt >= 2)[sub].ravel())
    B, C, S = map(np.concatenate, (Bs, Cs, Ss))
    sy, fy = np.concatenate(sy), np.concatenate(fy)
    rows = []
    for name, pred, y, tgt in (("boundary B", B, sy, "GT seam band"),
                               ("canopy C", C, fy, "GT foreground"),
                               ("surface U", S, fy, "GT foreground")):
        rows.append({"checkpoint": label, "head": name, "target": tgt,
                     "mean_on_pos": round(float(pred[y].mean()), 4),
                     "mean_on_neg": round(float(pred[~y].mean()), 4),
                     "median": round(float(np.median(pred)), 4),
                     "p99": round(float(np.percentile(pred, 99)), 4),
                     "AUROC": round(float(roc_auc_score(y, pred)), 4),
                     "AP": round(float(average_precision_score(y, pred)), 4)})
    return pd.DataFrame(rows)


def system(tag, label):
    crops = list(load_cache(tag))
    rows = []
    variants = [("no merge", None, None)] + \
               [(f"absolute merge tau={t}", t, None) for t in (0.15, 0.25, 0.35, 0.45, 0.55)] + \
               [("percentile merge q=50", None, 50)]
    for name, tau, q in variants:
        acc = []
        for surf, bound, canopy, gt in crops:
            seeds = global_nms_seeds(surf, canopy, MD, MA)
            m, insts = flood(seeds, bound, canopy)
            if tau is not None:
                m, insts = merge_basins(m, bound, tau=tau, l_min=30)
            elif q is not None:
                m, insts = merge_percentile(m, bound, q=q, l_min=30)
            ev = evaluate_metrics(m, insts, gt)
            da = compute_dao_segmentation_indices(m, gt)
            n_gt = int((np.unique(gt) >= 2).sum())
            acc.append({**{k: ev[k] for k in ("ap50", "ap75", "pq", "sq", "rq")},
                        "biou": compute_boundary_iou(m, gt) * 100.0,
                        "ed": da["mean_ed"], "ratio": len(insts) / max(1, n_gt)})
        df = pd.DataFrame(acc)
        rows.append({"checkpoint": label, "variant": name, **df.mean().to_dict(),
                     "ap50_sem": df["ap50"].sem()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    build_cache(num_crops=60, split="eval", start=100, tag=a.tag, ckpt=a.ckpt)

    hd = pd.concat([head_diag("evalhold60", "old (best_sharp)"),
                    head_diag(a.tag, a.label)], ignore_index=True)
    hd.to_csv(OUT / "h6_head_diagnostic.csv", index=False)
    print("\n" + "=" * 118)
    print(hd.to_string(index=False))

    sy = pd.concat([system("evalhold60", "old (best_sharp)"),
                    system(a.tag, a.label)], ignore_index=True).round(3)
    sy.to_csv(OUT / "h6_system.csv", index=False)
    cols = ["checkpoint", "variant", "ap50", "ap50_sem", "ap75", "pq", "sq", "biou", "ed", "ratio"]
    print("\n" + "=" * 118)
    print(sy[cols].to_string(index=False))
    print("=" * 118)

    b_new = hd[(hd.checkpoint == a.label) & (hd["head"] == "boundary B")].iloc[0]
    b_old = hd[(hd.checkpoint == "old (best_sharp)") & (hd["head"] == "boundary B")].iloc[0]
    new = sy[sy.checkpoint == a.label]
    nm = new[new.variant == "no merge"].ap50.iloc[0]
    bm = new[new.variant.str.startswith("absolute")].ap50.max()
    print(f"\nP1 mean B on true seams > 0.40 : {b_old.mean_on_pos:.4f} -> {b_new.mean_on_pos:.4f} -> "
          f"{'PASS' if b_new.mean_on_pos > 0.40 else 'FAIL'}")
    print(f"P2 AUROC >= 0.85               : {b_old.AUROC:.4f} -> {b_new.AUROC:.4f} -> "
          f"{'PASS' if b_new.AUROC >= 0.85 else 'FAIL'}")
    print(f"P3 absolute merge >= no-merge+2: {bm:.2f} vs {nm:.2f} -> "
          f"{'PASS' if bm >= nm + 2 else 'FAIL'}")
    print(f"P4 best config >= 70 mAP@50    : {new.ap50.max():.2f} -> "
          f"{'PASS' if new.ap50.max() >= 70 else 'FAIL'}")


if __name__ == "__main__":
    main()
