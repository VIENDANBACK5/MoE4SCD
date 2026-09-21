"""EXPLORATORY follow-up to H12's negative result.

A 7 px training target produces a thick predicted ridge; the flat-basin watershed then
erodes every crown by the ridge half-width at each seam, which punishes AP@[.5:.95]. If
that is the mechanism, thinning or sharpening B at decode time — with no retraining —
should recover much of the loss while keeping the magnitude H12 bought.
"""
from __future__ import annotations
import sys
from itertools import product
from pathlib import Path
_c='/home/ubuntu/anaconda3/envs/cs2/lib/python3.11/site-packages'; _l='/home/chung/.local/lib/python3.11/site-packages'
sys.path=[p for p in sys.path if p not in (_l,_c)]; sys.path.insert(0,_c); sys.path.append(_l)
sys.path.insert(0,"/home/chung/RS/Image Segmentation")
import cv2, numpy as np, pandas as pd
from crown_research.src.coco_ap import APAccumulator, instance_scores_from_surface
from crown_research.src.pred_cache import load_cache
from crown_research.src.seeding import flood, global_nms_seeds
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import evaluate_metrics

OUT = Path("crown_research/experiments/h12-seam-width/results")
K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

TRANSFORMS = {
    "identity":          lambda b: b,
    "erode 3x3":         lambda b: cv2.erode(b, K3),
    "erode 3x3 x2":      lambda b: cv2.erode(cv2.erode(b, K3), K3),
    "gamma 2":           lambda b: b ** 2,
    "gamma 3":           lambda b: b ** 3,
    "subtract bg 0.10":  lambda b: np.clip(b - 0.10, 0, 1),
    "erode + gamma 2":   lambda b: cv2.erode(b, K3) ** 2,
}


def score(crops, tf, md, ma):
    acc = APAccumulator(); f1 = []
    for surf, bound, canopy, gt in crops:
        b2 = tf(bound).astype(np.float32)
        m, insts = flood(global_nms_seeds(surf, canopy, float(md), ma), b2, canopy)
        insts = instance_scores_from_surface(m, insts, surf); acc.add(m, insts, gt)
        f1.append(evaluate_metrics(m, insts, gt)["ap50"])
    r = acc.ap()
    return {"COCO AP": r["AP"], "AP50": r["AP50"], "AP75": r["AP75"], "F1@50": float(np.mean(f1))}


def main():
    rows = []
    for tag, ck in (("h11_x2", "H11 narrow"), ("h12_wide", "H12 wide")):
        crops = list(load_cache(tag))
        for name, tf in TRANSFORMS.items():
            best = None
            for md, ma in product([140, 160, 200], [0.35, 0.50]):
                r = score(crops, tf, md, ma)
                if best is None or r["COCO AP"] > best["COCO AP"]:
                    best = {**r, "min_dist": md, "min_apex": ma}
            rows.append({"checkpoint": ck, "B transform": name, **best})
            print(f"  {ck:<11} {name:<18} AP {best['COCO AP']:5.2f}  AP75 {best['AP75']:5.2f}", flush=True)
    df = pd.DataFrame(rows).round(3)
    df.to_csv(OUT / "h12_thin_probe.csv", index=False)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
