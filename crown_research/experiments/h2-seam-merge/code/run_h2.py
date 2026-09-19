"""H2 — over-seed then merge on seam evidence. See ../protocol.md."""
from __future__ import annotations

import sys
from itertools import product
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

OUT = Path("crown_research/experiments/h2-seam-merge/results")
LOOSE = dict(min_dist=24.0, min_apex=0.25)
TUNED = dict(min_dist=160.0, min_apex=0.50)


def evaluate(crops, seed_kw, merge_kw=None, oracle=False, full=False):
    acc = []
    for surf, bound, canopy, gt in crops:
        seeds = global_nms_seeds(surf, canopy, seed_kw["min_dist"], seed_kw["min_apex"])
        m, insts = flood(seeds, bound, canopy)
        if merge_kw is not None:
            m, insts = merge_basins(m, bound, gt_map=(gt if oracle else None), **merge_kw)
        ev = evaluate_metrics(m, insts, gt)
        n_gt = int((np.unique(gt) >= 2).sum())
        rec = {**{k: ev[k] for k in ("ap50", "ap75", "pq", "sq", "rq")},
               "ratio": len(insts) / max(1, n_gt)}
        if full:
            da = compute_dao_segmentation_indices(m, gt)
            rec.update({"biou": compute_boundary_iou(m, gt) * 100.0,
                        "os": da["mean_os"], "us": da["mean_us"], "ed": da["mean_ed"]})
        acc.append(rec)
    df = pd.DataFrame(acc)
    out = df.mean().to_dict()
    out["ap50_sem"] = df["ap50"].sem()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tune = list(load_cache("eval30"))
    hold = list(load_cache("evalhold60"))
    print(f"[H2] tune={len(tune)} hold={len(hold)}", flush=True)

    # --- Arm C sweep on the tuning split ---
    rows = []
    for tau, l_min in product([0.15, 0.25, 0.35, 0.45, 0.55, 0.65], [5, 15, 30]):
        d = evaluate(tune, LOOSE, dict(tau=tau, l_min=l_min))
        rows.append({"tau": tau, "l_min": l_min, **d})
        print(f"  C tau={tau:.2f} L={l_min:>2} -> ap50={d['ap50']:.2f} pq={d['pq']:.2f} ratio={d['ratio']:.2f}", flush=True)
    df_c = pd.DataFrame(rows).sort_values(["ap50", "pq"], ascending=False)
    df_c.to_csv(OUT / "h2_merge_sweep.csv", index=False)
    best = df_c.iloc[0]
    print(f"\n  best C: tau={best.tau} l_min={int(best.l_min)} (tune ap50={best.ap50:.2f})\n", flush=True)

    bt, bl = float(best.tau), int(best.l_min)
    arms = [
        ("A  loose seeds, no merge", LOOSE, None, False),
        ("B  tuned NMS, no merge", TUNED, None, False),
        (f"C  loose + seam merge (tau={bt}, L={bl})", LOOSE, dict(tau=bt, l_min=bl), False),
        ("D  loose + ORACLE merge (ceiling)", LOOSE, dict(tau=0.0, l_min=bl), True),
        (f"E  tuned + seam merge (tau={bt}, L={bl})", TUNED, dict(tau=bt, l_min=bl), False),
    ]

    summary = []
    for split_name, crops in (("eval30 (tune)", tune), ("evalhold60 (hold-out)", hold)):
        for name, sk, mk, orc in arms:
            d = evaluate(crops, sk, mk, oracle=orc, full=True)
            summary.append({"arm": name, "split": split_name, **d})
            print(f"  [{split_name:<22}] {name:<44} ap50={d['ap50']:.2f} pq={d['pq']:.2f}", flush=True)

    df_s = pd.DataFrame(summary).round(3)
    df_s.to_csv(OUT / "h2_summary.csv", index=False)
    cols = ["arm", "split", "ap50", "ap50_sem", "ap75", "pq", "sq", "rq", "biou", "ed", "ratio"]
    print("\n" + "=" * 135)
    print(df_s[cols].to_string(index=False))
    print("=" * 135)

    h = df_s[df_s.split == "evalhold60 (hold-out)"].set_index("arm")["ap50"]
    B, C, D = h.iloc[1], h.iloc[2], h.iloc[3]
    print(f"\nPrediction 1  C >= B+3 : C={C:.2f} B={B:.2f} -> {'PASS' if C >= B + 3 else 'FAIL'}")
    print(f"Prediction 2  D >= 80   : D={D:.2f} -> {'PASS' if D >= 80 else 'FAIL'}")
    frac = (C - B) / (D - B) if D > B else float('nan')
    print(f"Prediction 3  (C-B)/(D-B) >= 0.40 : {frac:.3f} -> {'PASS' if frac >= 0.40 else 'FAIL'}")


if __name__ == "__main__":
    main()
