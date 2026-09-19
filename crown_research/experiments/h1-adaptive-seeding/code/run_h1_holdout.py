"""H1 hold-out validation.

`min_dist`/`min_apex` were selected on the same 30 eval crops used to report the
result, so the 66.15 figure is a selection-on-test number and cannot be trusted as
stated. This re-scores the frozen settings on 60 *disjoint* eval crops (indices
100-159) that took no part in any sweep.
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

import numpy as np
import pandas as pd

from crown_research.src.pred_cache import load_cache
from crown_research.src.seeding import adaptive_seeds, flood, global_nms_seeds
from crown_segmentation_research.methods.boundary_refinement.metrics_dao import (
    compute_boundary_iou, compute_dao_segmentation_indices)
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import evaluate_metrics

OUT = Path("crown_research/experiments/h1-adaptive-seeding/results")

# Frozen settings. Nothing below is re-tuned on the hold-out.
ARMS = {
    "baseline (md=24, ma=0.25)":      lambda s, c: global_nms_seeds(s, c, 24.0, 0.25),
    "H1-a locked-grid (md=56, ma=0.45)": lambda s, c: global_nms_seeds(s, c, 56.0, 0.45),
    "H1-a extended  (md=144, ma=0.50)": lambda s, c: global_nms_seeds(s, c, 144.0, 0.50),
    "H1-a extended  (md=160, ma=0.50)": lambda s, c: global_nms_seeds(s, c, 160.0, 0.50),
    "H1-b adaptive  (t=0.3, g=1.2, ma=0.35)": lambda s, c: adaptive_seeds(s, c, t=0.3, gamma=1.2, min_apex=0.35),
}


def score_split(tag):
    crops = list(load_cache(tag))
    rows = []
    for name, fn in ARMS.items():
        acc = []
        for surf, bound, canopy, gt in crops:
            seeds = fn(surf, canopy)
            m, insts = flood(seeds, bound, canopy)
            ev = evaluate_metrics(m, insts, gt)
            da = compute_dao_segmentation_indices(m, gt)
            n_gt = int((np.unique(gt) >= 2).sum())
            acc.append({**{k: ev[k] for k in ("ap50", "ap75", "pq", "sq", "rq")},
                        "biou": compute_boundary_iou(m, gt) * 100.0,
                        "os": da["mean_os"], "us": da["mean_us"], "ed": da["mean_ed"],
                        "ratio": len(insts) / max(1, n_gt)})
        df = pd.DataFrame(acc)
        rows.append({"arm": name, "split": tag, "n_crops": len(crops),
                     **df.mean().to_dict(), "ap50_sem": df["ap50"].sem()})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.concat([score_split("eval30"), score_split("evalhold60")], ignore_index=True).round(3)
    df.to_csv(OUT / "h1_holdout.csv", index=False)
    cols = ["arm", "split", "n_crops", "ap50", "ap50_sem", "ap75", "pq", "sq", "rq", "biou", "ed", "ratio"]
    print("=" * 130)
    print(df[cols].to_string(index=False))
    print("=" * 130)
    piv = df.pivot(index="arm", columns="split", values="ap50")
    piv["delta (hold - tune)"] = (piv["evalhold60"] - piv["eval30"]).round(3)
    print("\nGeneralisation check (mAP@50):\n", piv.to_string())


if __name__ == "__main__":
    main()
