"""H11 — matched-epoch control separating auxiliary-task effect from extra epochs.

Three checkpoints over one decode grid: H6 (12 ep), H11 (+12 ep, no apex head),
H7 (+12 ep, with apex head). Surface seeding, no merge. See ../protocol.md.
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

import numpy as np
import pandas as pd

from crown_research.src.coco_ap import APAccumulator, instance_scores_from_surface
from crown_research.src.pred_cache import load_cache
from crown_research.src.seeding import flood, global_nms_seeds
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import evaluate_metrics

OUT = Path("crown_research/experiments/h11-aux-task/results")
H7_CACHE = Path("crown_research/data/cache/h7_apex")
CKPTS = [("H6 calibrated (12 ep)", "h6_final"),
         ("H11 +12 ep, no apex", "h11_x2"),
         ("H7 +12 ep, with apex", "__h7__")]
GRID = list(product([100, 140, 160, 200], [0.35, 0.50, 0.65]))


def load_any(tag):
    if tag != "__h7__":
        yield from load_cache(tag)
        return
    for f in sorted(H7_CACHE.glob("*.npz")):  # H7 cache stores 4 heads
        z = np.load(f)
        yield (z["surface"].astype(np.float32), z["boundary"].astype(np.float32),
               z["canopy"].astype(np.float32), z["gt_map"].astype(np.int32))


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
    rows = []
    for label, tag in CKPTS:
        crops = list(load_any(tag))
        if not crops:
            print(f"  !! no cache for {label} ({tag}) — skipping", flush=True)
            continue
        for md, ma in GRID:
            rows.append({"checkpoint": label, "min_dist": md, "min_apex": ma,
                         **score(crops, md, ma)})
        print(f"  swept {label} ({len(crops)} crops)", flush=True)
    df = pd.DataFrame(rows).round(3)
    df.to_csv(OUT / "h11_sweep.csv", index=False)

    summ = []
    for label, _ in CKPTS:
        sub = df[df.checkpoint == label]
        if sub.empty:
            continue
        b = sub.sort_values("COCO AP", ascending=False).iloc[0]
        c = sub[(sub.min_dist == 160) & (sub.min_apex == 0.35)].iloc[0]
        summ.append({"checkpoint": label, "best_AP": b["COCO AP"],
                     "best_cell": f"md={int(b.min_dist)},ma={b.min_apex}",
                     "best_AP50": b["AP50"], "best_AP75": b["AP75"], "best_F1@50": b["F1@50"],
                     "common_AP (md=160,ma=0.35)": c["COCO AP"]})
    ds = pd.DataFrame(summ).round(3)
    ds.to_csv(OUT / "h11_summary.csv", index=False)
    print("\n" + "=" * 118)
    print(ds.to_string(index=False))
    print("=" * 118)

    g = {r["checkpoint"]: r for r in summ}
    h6 = g["H6 calibrated (12 ep)"]["best_AP"]
    h11 = g.get("H11 +12 ep, no apex", {}).get("best_AP")
    h7 = g.get("H7 +12 ep, with apex", {}).get("best_AP")
    if h11 is None or h7 is None:
        print("\n(missing a checkpoint — predictions not evaluated)")
        return
    frac = (h11 - h6) / (h7 - h6) if abs(h7 - h6) > 1e-9 else float("nan")
    print(f"\nP1 H11 >= 28.5 (>=70% of H7's gain) : {h11:.2f}, recovers {100*frac:.0f}% -> "
          f"{'PASS' if h11 >= 28.5 else 'FAIL'}")
    print(f"P2 |H7 - H11| < 1.0                  : {abs(h7 - h11):.2f} -> "
          f"{'PASS' if abs(h7 - h11) < 1.0 else 'FAIL'}")
    print(f"P3 H11 >= H6 ({h6:.2f})                 : {h11:.2f} -> "
          f"{'PASS' if h11 >= h6 else 'FAIL'}")
if __name__ == "__main__": main()
