"""H4 — decode-time consistency coupling: gate = canopy - lambda * boundary. See ../protocol.md."""
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
from crown_research.src.rag_merge import merge_basins
from crown_research.src.seeding import flood, global_nms_seeds
from crown_segmentation_research.methods.boundary_refinement.metrics_dao import (
    compute_boundary_iou, compute_dao_segmentation_indices)
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import evaluate_metrics

OUT = Path("crown_research/experiments/h4-coupled-gate/results")
MD, MA = 160.0, 0.50
MERGE = dict(tau=0.15, l_min=30)


def evaluate(crops, lam, merge=False):
    acc = []
    for surf, bound, canopy, gt in crops:
        gate = np.clip(canopy - lam * bound, 0.0, 1.0)
        seeds = global_nms_seeds(surf, gate, MD, MA)
        m, insts = flood(seeds, bound, gate)
        if merge:
            m, insts = merge_basins(m, bound, **MERGE)
        ev = evaluate_metrics(m, insts, gt)
        da = compute_dao_segmentation_indices(m, gt)
        n_gt = int((np.unique(gt) >= 2).sum())
        acc.append({**{k: ev[k] for k in ("ap50", "ap75", "pq", "sq", "rq")},
                    "biou": compute_boundary_iou(m, gt) * 100.0,
                    "os": da["mean_os"], "us": da["mean_us"], "ed": da["mean_ed"],
                    "ratio": len(insts) / max(1, n_gt)})
    df = pd.DataFrame(acc)
    out = df.mean().to_dict()
    out["ap50_sem"] = df["ap50"].sem()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tune = list(load_cache("eval30"))
    hold = list(load_cache("evalhold60"))
    lams = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

    rows = []
    for lam in lams:
        for mg in (False, True):
            d = evaluate(tune, lam, mg)
            rows.append({"lambda": lam, "merge": mg, "split": "eval30", **d})
            print(f"  lam={lam:.2f} merge={int(mg)} -> ap50={d['ap50']:.2f} sq={d['sq']:.2f} "
                  f"pq={d['pq']:.2f} ratio={d['ratio']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "h4_lambda_sweep.csv", index=False)

    best = df.sort_values(["ap50", "pq"], ascending=False).iloc[0]
    print(f"\n  best on tune: lambda={best['lambda']} merge={bool(best['merge'])}\n", flush=True)

    summary = []
    for lam, mg, name in [(0.0, False, "control lambda=0, no merge"),
                          (0.0, True, "control lambda=0 + merge"),
                          (float(best["lambda"]), bool(best["merge"]),
                           f"H4 best lambda={best['lambda']}, merge={bool(best['merge'])}")]:
        for split, crops in (("eval30 (tune)", tune), ("evalhold60 (hold-out)", hold)):
            summary.append({"arm": name, "split": split, **evaluate(crops, lam, mg)})
    ds = pd.DataFrame(summary).round(3)
    ds.to_csv(OUT / "h4_summary.csv", index=False)
    cols = ["arm", "split", "ap50", "ap50_sem", "ap75", "pq", "sq", "rq", "biou", "ed", "ratio"]
    print("=" * 130)
    print(ds[cols].to_string(index=False))
    print("=" * 130)

    h = ds[ds.split == "evalhold60 (hold-out)"]
    ctrl = h[h.arm.str.startswith("control lambda=0, no")].iloc[0]
    bst = h[h.arm.str.startswith("H4 best")].iloc[0]
    print(f"\nPrediction 1  best - control >= +2 ap50 : {bst.ap50 - ctrl.ap50:+.2f} -> "
          f"{'PASS' if bst.ap50 - ctrl.ap50 >= 2 else 'FAIL'}")
    print(f"Prediction 2  SQ gain >= +1.5           : {bst.sq - ctrl.sq:+.2f} -> "
          f"{'PASS' if bst.sq - ctrl.sq >= 1.5 else 'FAIL'}")
    print(f"Prediction 3  best lambda <= 1.0        : {best['lambda']} -> "
          f"{'PASS' if float(best['lambda']) <= 1.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
