"""Re-score the key configurations with true COCO AP alongside the repo's F1 metric.

EXPLORATORY (metric-integrity follow-up, not a hypothesis test).
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

import cv2
import numpy as np
import pandas as pd

from crown_research.src.coco_ap import APAccumulator, instance_scores_from_surface
from crown_research.src.pred_cache import load_cache
from crown_research.src.seeding import flood, global_nms_seeds
from run_h5 import merge_percentile
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import evaluate_metrics

OUT = Path("crown_research/experiments/h5-boundary-calibration/results")


def oracle_all(gt, _bound):
    seeds = []
    for gid in np.unique(gt):
        if gid < 2:
            continue
        dt = cv2.distanceTransform((gt == gid).astype(np.uint8), cv2.DIST_L2, 3)
        y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
        seeds.append((int(y), int(x), 1.0))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seam = (cv2.morphologyEx(gt.astype(np.float32), cv2.MORPH_GRADIENT, k) > 0).astype(np.float32)
    energy = np.clip(cv2.GaussianBlur(seam, (0, 0), 1.0) * 3.0, 0, 1)
    return flood(seeds, energy, (gt >= 2).astype(np.float32))


CONFIGS = {
    "published (md=24, ma=0.25)": lambda s, b, c, g: flood(global_nms_seeds(s, c, 24.0, 0.25), b, c),
    "H1 tuned (md=160, ma=0.50)": lambda s, b, c, g: flood(global_nms_seeds(s, c, 160.0, 0.50), b, c),
    "H5 best (tuned + percentile merge)": lambda s, b, c, g: merge_percentile(
        flood(global_nms_seeds(s, c, 160.0, 0.50), b, c)[0], b, q=50, l_min=30),
    "H3 oracle heads": lambda s, b, c, g: oracle_all(g, b),
}


def main():
    rows = []
    for tag in ("eval30", "evalhold60"):
        crops = list(load_cache(tag))
        for name, fn in CONFIGS.items():
            acc = APAccumulator()
            f1 = []
            for surf, bound, canopy, gt in crops:
                m, insts = fn(surf, bound, canopy, gt)
                insts = instance_scores_from_surface(m, insts, surf)
                acc.add(m, insts, gt)
                f1.append(evaluate_metrics(m, insts, gt)["ap50"])
            ap = acc.ap()
            rows.append({"config": name, "split": tag,
                         "COCO AP": round(ap["AP"], 2), "COCO AP50": round(ap["AP50"], 2),
                         "COCO AP75": round(ap["AP75"], 2),
                         "repo F1@50 (labelled ap50)": round(float(np.mean(f1)), 2)})
            print(f"  [{tag}] {name:<38} AP {ap['AP']:5.2f} | AP50 {ap['AP50']:5.2f} | "
                  f"F1@50 {np.mean(f1):5.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "coco_ap_rescore.csv", index=False)
    print("\n" + "=" * 110)
    print(df.to_string(index=False))
    print("=" * 110)


if __name__ == "__main__":
    main()
