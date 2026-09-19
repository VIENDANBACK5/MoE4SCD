"""H8 — does the tuned suppression radius transfer, and is it a crown-diameter rule?
See ../protocol.md."""
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

from crown_research.src.coco_ap import APAccumulator, instance_scores_from_surface
from crown_research.src.pred_cache import load_cache
from crown_research.src.seeding import flood, global_nms_seeds
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import evaluate_metrics

OUT = Path("crown_research/experiments/h8-cross-site/results")
SPLITS = ["evalhold60", "test1", "test2"]
FROZEN = (160.0, 0.50)
PUBLISHED = (24.0, 0.25)


def crown_stats(crops):
    areas = []
    for _, _, _, gt in crops:
        for g in np.unique(gt):
            if g >= 2:
                areas.append(int((gt == g).sum()))
    areas = np.array(areas, float)
    return {"n_crowns": len(areas), "mean_area": areas.mean(),
            "median_area": np.median(areas),
            "D_mean": 2.0 * np.sqrt(areas.mean() / np.pi),
            "D_median": 2.0 * np.sqrt(np.median(areas) / np.pi)}


def score(crops, md, ma):
    acc = APAccumulator()
    f1, ratios = [], []
    for surf, bound, canopy, gt in crops:
        seeds = global_nms_seeds(surf, canopy, md, ma)
        m, insts = flood(seeds, bound, canopy)
        insts = instance_scores_from_surface(m, insts, surf)
        acc.add(m, insts, gt)
        ev = evaluate_metrics(m, insts, gt)
        f1.append(ev["ap50"])
        ratios.append(len(insts) / max(1, int((np.unique(gt) >= 2).sum())))
    ap = acc.ap()
    return {"F1@50": float(np.mean(f1)), "COCO AP": ap["AP"], "AP50": ap["AP50"],
            "AP75": ap["AP75"], "ratio": float(np.mean(ratios))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cached = {s: list(load_cache(s)) for s in SPLITS}

    stats = pd.DataFrame([{"split": s, **crown_stats(c)} for s, c in cached.items()]).round(1)
    stats.to_csv(OUT / "h8_split_stats.csv", index=False)
    print(stats.to_string(index=False), flush=True)

    grid = [40, 60, 80, 100, 120, 140, 160, 180, 200, 240]
    sweep = []
    for s, c in cached.items():
        for md in grid:
            r = score(c, float(md), 0.50)
            sweep.append({"split": s, "min_dist": md, **r})
            print(f"  {s:<11} md={md:>3} -> F1@50 {r['F1@50']:5.2f} | AP {r['COCO AP']:5.2f} "
                  f"| ratio {r['ratio']:.2f}", flush=True)
    dfs = pd.DataFrame(sweep)
    dfs.to_csv(OUT / "h8_sweep.csv", index=False)

    rows = []
    for s, c in cached.items():
        D = float(stats[stats.split == s].D_mean.iloc[0])
        best = dfs[dfs.split == s].sort_values("F1@50", ascending=False).iloc[0]
        for name, md, ma in [("published (24, 0.25)", *PUBLISHED),
                             ("frozen eval-tuned (160, 0.50)", *FROZEN),
                             (f"per-split optimum (md={int(best.min_dist)})", float(best.min_dist), 0.50),
                             (f"D-rule (md=D={D:.0f})", D, 0.50)]:
            rows.append({"split": s, "config": name, "D_mean": round(D, 1), **score(c, md, ma)})
    df = pd.DataFrame(rows).round(3)
    df.to_csv(OUT / "h8_transfer.csv", index=False)
    print("\n" + "=" * 118)
    print(df.to_string(index=False))
    print("=" * 118)

    print("\nPredictions:")
    for s in ("test1", "test2"):
        sub = df[df.split == s].set_index("config")
        pub = sub.iloc[0]["F1@50"]
        frz = sub.iloc[1]["F1@50"]
        opt_row = dfs[dfs.split == s].sort_values("F1@50", ascending=False).iloc[0]
        D = float(stats[stats.split == s].D_mean.iloc[0])
        rel = abs(opt_row.min_dist - D) / D
        print(f"  P1 {s}: frozen - published = {frz - pub:+.2f} F1@50 -> "
              f"{'PASS' if frz - pub >= 10 else 'FAIL'}")
        print(f"  P2 {s}: optimum md={int(opt_row.min_dist)} vs D={D:.0f} "
              f"(off by {100*rel:.0f}%) -> {'PASS' if rel <= 0.30 else 'FAIL'}")
        print(f"  P3 {s}: frozen within 2 of optimum? {opt_row['F1@50'] - frz:+.2f} -> "
              f"{'PASS' if opt_row['F1@50'] - frz <= 2 else 'FAIL'}")


if __name__ == "__main__":
    main()
