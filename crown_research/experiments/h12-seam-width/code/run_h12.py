"""H12 — does a widened seam target raise magnitude, and does it pay more out-of-domain?

Compares the wide-target checkpoint against H11 (matched epochs, narrow target) so the
only difference is target width. See ../protocol.md.
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
from sklearn.metrics import roc_auc_score

from crown_research.src.coco_ap import APAccumulator, instance_scores_from_surface
from crown_research.src.pred_cache import load_cache
from crown_research.src.seeding import flood, global_nms_seeds
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import evaluate_metrics

OUT = Path("crown_research/experiments/h12-seam-width/results")
# split -> (H11 control cache, H12 wide cache)
SPLITS = {"evalhold60": ("h11_x2", "h12_wide"),
          "test2": ("test2_h11", "test2_h12"),
          "test1": ("test1_h11", "test1_h12")}
GRID = list(product([100, 140, 160, 200, 240], [0.35, 0.50, 0.65]))


def seam_band(gt, r=3):
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(gt.astype(np.float32), cv2.MORPH_GRADIENT, k3) > 0
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    return cv2.dilate((grad & (gt >= 2)).astype(np.uint8), kd) > 0


def head_diag(tag):
    B, sy = [], []
    for _, bound, _, gt in load_cache(tag):
        sub = (slice(None, None, 3), slice(None, None, 3))
        B.append(bound[sub].ravel()); sy.append(seam_band(gt)[sub].ravel())
    B, sy = np.concatenate(B), np.concatenate(sy)
    return {"B_on_seam": float(B[sy].mean()), "B_off_seam": float(B[~sy].mean()),
            "separation": float(B[sy].mean() / max(B[~sy].mean(), 1e-9)),
            "B_AUROC": float(roc_auc_score(sy, B))}


def score(crops, md, ma):
    acc = APAccumulator(); f1 = []
    for surf, bound, canopy, gt in crops:
        m, insts = flood(global_nms_seeds(surf, canopy, float(md), ma), bound, canopy)
        insts = instance_scores_from_surface(m, insts, surf)
        acc.add(m, insts, gt)
        f1.append(evaluate_metrics(m, insts, gt)["ap50"])
    r = acc.ap()
    return {"COCO AP": r["AP"], "AP50": r["AP50"], "AP75": r["AP75"], "F1@50": float(np.mean(f1))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    hd = []
    for split, (ctrl, wide) in SPLITS.items():
        for label, tag in (("H11 narrow target", ctrl), ("H12 wide target", wide)):
            try:
                hd.append({"split": split, "checkpoint": label, **head_diag(tag)})
            except Exception as e:
                print(f"  !! head_diag {tag}: {e}", flush=True)
    dh = pd.DataFrame(hd).round(4)
    dh.to_csv(OUT / "h12_head_diagnostic.csv", index=False)
    print("=" * 96); print(dh.to_string(index=False)); print("=" * 96)

    rows = []
    for split, (ctrl, wide) in SPLITS.items():
        for label, tag in (("H11 narrow target", ctrl), ("H12 wide target", wide)):
            crops = list(load_cache(tag))
            if not crops:
                print(f"  !! no cache {tag}", flush=True); continue
            for md, ma in GRID:
                rows.append({"split": split, "checkpoint": label, "min_dist": md,
                             "min_apex": ma, **score(crops, md, ma)})
            print(f"  swept {split} / {label} ({len(crops)} crops)", flush=True)
    df = pd.DataFrame(rows).round(3)
    df.to_csv(OUT / "h12_sweep.csv", index=False)

    summ = []
    for split in SPLITS:
        sub = df[df.split == split]
        if sub.empty:
            continue
        row = {"split": split}
        for label, key in (("H11 narrow target", "narrow"), ("H12 wide target", "wide")):
            s = sub[sub.checkpoint == label]
            if s.empty:
                continue
            b = s.sort_values("COCO AP", ascending=False).iloc[0]
            row[f"{key}_AP"] = b["COCO AP"]
            row[f"{key}_cell"] = f"md={int(b.min_dist)},ma={b.min_apex}"
        if "wide_AP" in row and "narrow_AP" in row:
            row["dAP"] = round(row["wide_AP"] - row["narrow_AP"], 3)
            row["dAP_rel_%"] = round(100.0 * (row["wide_AP"] - row["narrow_AP"]) /
                                     max(row["narrow_AP"], 1e-9), 1)
        summ.append(row)
    ds = pd.DataFrame(summ)
    ds.to_csv(OUT / "h12_summary.csv", index=False)
    print("\n" + "=" * 110); print(ds.to_string(index=False)); print("=" * 110)

    try:
        w = dh[(dh.split == "evalhold60") & (dh.checkpoint == "H12 wide target")].iloc[0]
        n = dh[(dh.split == "evalhold60") & (dh.checkpoint == "H11 narrow target")].iloc[0]
        e = ds[ds.split == "evalhold60"].iloc[0]
        t1 = ds[ds.split == "test1"].iloc[0]
        t2 = ds[ds.split == "test2"].iloc[0]
        print(f"\nP1 mean B on seams > 0.30 : {n.B_on_seam:.4f} -> {w.B_on_seam:.4f} -> "
              f"{'PASS' if w.B_on_seam > 0.30 else 'FAIL'}")
        print(f"P2 in-domain dAP in [-1,+2]: {e['dAP']:+.2f} -> "
              f"{'PASS' if -1.0 <= e['dAP'] <= 2.0 else 'FAIL'}")
        print(f"P3 OOD rel > in-domain rel : test1 {t1['dAP_rel_%']:+.1f}% vs "
              f"test2 {t2['dAP_rel_%']:+.1f}% -> "
              f"{'PASS' if t1['dAP_rel_%'] > t2['dAP_rel_%'] else 'FAIL'}")
    except Exception as ex:
        print(f"\n(prediction check incomplete: {ex})")


if __name__ == "__main__":
    main()
