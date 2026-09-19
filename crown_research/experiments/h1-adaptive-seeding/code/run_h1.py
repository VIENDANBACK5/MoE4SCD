"""H1 — scale-blind vs scale-adaptive seed suppression. See ../protocol.md."""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

_conda_p = '/home/ubuntu/anaconda3/envs/cs2/lib/python3.11/site-packages'
_local_p = '/home/chung/.local/lib/python3.11/site-packages'
sys.path = [p for p in sys.path if p != _local_p and p != _conda_p]
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


def score(crops, seed_fn):
    acc = []
    for surf, bound, canopy, gt in crops:
        seeds = seed_fn(surf, canopy)
        m, insts = flood(seeds, bound, canopy)
        ev = evaluate_metrics(m, insts, gt)
        da = compute_dao_segmentation_indices(m, gt)
        n_gt = int((np.unique(gt) >= 2).sum())
        acc.append({**{k: ev[k] for k in ("ap50", "ap75", "pq", "sq", "rq")},
                    "os": da["mean_os"], "us": da["mean_us"], "ed": da["mean_ed"],
                    "ratio": len(insts) / max(1, n_gt), "n_pred": len(insts)})
    d = pd.DataFrame(acc).mean()
    return d


def full_metrics(crops, seed_fn):
    acc = []
    for surf, bound, canopy, gt in crops:
        seeds = seed_fn(surf, canopy)
        m, insts = flood(seeds, bound, canopy)
        ev = evaluate_metrics(m, insts, gt)
        da = compute_dao_segmentation_indices(m, gt)
        n_gt = int((np.unique(gt) >= 2).sum())
        acc.append({**{k: ev[k] for k in ("ap50", "ap75", "pq", "sq", "rq")},
                    "biou": compute_boundary_iou(m, gt) * 100.0,
                    "os": da["mean_os"], "us": da["mean_us"], "ed": da["mean_ed"],
                    "ratio": len(insts) / max(1, n_gt)})
    return pd.DataFrame(acc).mean()


def main():
    crops = list(load_cache("eval30"))
    print(f"[H1] {len(crops)} cached crops", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    # --- Arm A: global radius ---
    rows_a = []
    for md, ma in product([16, 20, 24, 28, 32, 40, 48, 56], [0.20, 0.25, 0.35, 0.45]):
        d = score(crops, lambda s, c, md=md, ma=ma: global_nms_seeds(s, c, min_dist=float(md), min_apex=ma))
        rows_a.append({"min_dist": md, "min_apex": ma, **d.to_dict()})
        print(f"  A md={md:>2} ma={ma:.2f} -> ap50={d['ap50']:.2f} ratio={d['ratio']:.2f}", flush=True)
    df_a = pd.DataFrame(rows_a).sort_values(["ap50", "pq"], ascending=False)
    df_a.to_csv(OUT / "h1a_global_sweep.csv", index=False)

    # --- Arm B: scale-adaptive radius ---
    rows_b = []
    for t, g, ma in product([0.3, 0.5, 0.7], [0.6, 0.8, 1.0, 1.2], [0.25, 0.35]):
        d = score(crops, lambda s, c, t=t, g=g, ma=ma: adaptive_seeds(s, c, t=t, gamma=g, min_apex=ma))
        rows_b.append({"t": t, "gamma": g, "min_apex": ma, **d.to_dict()})
        print(f"  B t={t} g={g} ma={ma:.2f} -> ap50={d['ap50']:.2f} ratio={d['ratio']:.2f}", flush=True)
    df_b = pd.DataFrame(rows_b).sort_values(["ap50", "pq"], ascending=False)
    df_b.to_csv(OUT / "h1b_adaptive_sweep.csv", index=False)

    # --- Summary at locked selection rule (best mean ap50, tie-break PQ) ---
    ba, bb = df_a.iloc[0], df_b.iloc[0]
    summary = []
    summary.append({"arm": "baseline (md=24, ma=0.25)", **full_metrics(
        crops, lambda s, c: global_nms_seeds(s, c, 24.0, 0.25)).to_dict()})
    summary.append({"arm": f"H1-a best (md={int(ba.min_dist)}, ma={ba.min_apex})", **full_metrics(
        crops, lambda s, c: global_nms_seeds(s, c, float(ba.min_dist), float(ba.min_apex))).to_dict()})
    summary.append({"arm": f"H1-b best (t={bb.t}, g={bb.gamma}, ma={bb.min_apex})", **full_metrics(
        crops, lambda s, c: adaptive_seeds(s, c, t=float(bb.t), gamma=float(bb.gamma), min_apex=float(bb.min_apex))).to_dict()})
    df_s = pd.DataFrame(summary).round(3)
    df_s.to_csv(OUT / "h1_summary.csv", index=False)

    print("\n" + "=" * 120)
    print(df_s.to_string(index=False))
    print("=" * 120)
    print("\nTop-5 Arm A:\n", df_a.head(5).round(3).to_string(index=False))
    print("\nTop-5 Arm B:\n", df_b.head(5).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
