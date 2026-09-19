"""H10 — paired checkpoint comparison at matched decode. See ../protocol.md."""
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

OUT = Path("crown_research/experiments/h10-ood-robustness/results")
# (split label, old-checkpoint cache tag, calibrated-checkpoint cache tag, in/out of domain)
PAIRS = [("evalhold60", "evalhold60", "h6_final", "val (in-domain)"),
         ("test2", "test2", "test2_h6", "in-domain"),
         ("test1", "test1", "test1_h6", "OUT-of-domain")]
GRID = list(product([160, 200, 240, 280, 320], [0.35, 0.50, 0.65]))
COMMON = (200, 0.50)


def score(crops, md, ma):
    acc = APAccumulator(); f1 = []
    for surf, bound, canopy, gt in crops:
        m, insts = flood(global_nms_seeds(surf, canopy, float(md), ma), bound, canopy)
        insts = instance_scores_from_surface(m, insts, surf)
        acc.add(m, insts, gt)
        f1.append(evaluate_metrics(m, insts, gt)["ap50"])
    r = acc.ap()
    return {"COCO AP": r["AP"], "AP50": r["AP50"], "F1@50": float(np.mean(f1))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, old_tag, new_tag, domain in PAIRS:
        for ck, tag in (("old (best_sharp)", old_tag), ("H6 (calibrated)", new_tag)):
            crops = list(load_cache(tag))
            for md, ma in GRID:
                rows.append({"split": split, "domain": domain, "checkpoint": ck,
                             "min_dist": md, "min_apex": ma, **score(crops, md, ma)})
            print(f"  swept {split} / {ck} ({len(crops)} crops)", flush=True)
    df = pd.DataFrame(rows).round(3)
    df.to_csv(OUT / "h10_paired_sweep.csv", index=False)

    summ = []
    for split, _, _, domain in PAIRS:
        sub = df[df.split == split]
        row = {"split": split, "domain": domain}
        for ck in ("old (best_sharp)", "H6 (calibrated)"):
            b = sub[sub.checkpoint == ck].sort_values("COCO AP", ascending=False).iloc[0]
            key = "old" if ck.startswith("old") else "new"
            row[f"{key}_best_AP"] = b["COCO AP"]
            row[f"{key}_best_cell"] = f"md={int(b.min_dist)},ma={b.min_apex}"
            c = sub[(sub.checkpoint == ck) & (sub.min_dist == COMMON[0]) & (sub.min_apex == COMMON[1])].iloc[0]
            row[f"{key}_common_AP"] = c["COCO AP"]
        row["dAP_best"] = round(row["new_best_AP"] - row["old_best_AP"], 3)
        row["dAP_rel_best_%"] = round(100.0 * (row["new_best_AP"] - row["old_best_AP"]) / max(row["old_best_AP"], 1e-9), 1)
        row["dAP_common"] = round(row["new_common_AP"] - row["old_common_AP"], 3)
        row["dAP_rel_common_%"] = round(100.0 * (row["new_common_AP"] - row["old_common_AP"]) / max(row["old_common_AP"], 1e-9), 1)
        summ.append(row)
    ds = pd.DataFrame(summ)
    ds.to_csv(OUT / "h10_summary.csv", index=False)
    print("\n" + "=" * 132)
    print(ds.to_string(index=False))
    print("=" * 132)

    t1 = ds[ds.split == "test1"].iloc[0]
    t2 = ds[ds.split == "test2"].iloc[0]
    print(f"\nP1 rel gain OOD > in-domain : test1 {t1['dAP_rel_best_%']:.1f}% vs "
          f"test2 {t2['dAP_rel_best_%']:.1f}% -> "
          f"{'PASS' if t1['dAP_rel_best_%'] > t2['dAP_rel_best_%'] else 'FAIL'}")
    print(f"P2 OOD absolute gain >= +1.0: {t1['dAP_best']:+.2f} -> "
          f"{'PASS' if t1['dAP_best'] >= 1.0 else 'FAIL'}")
    print(f"P3 controlled gain < 83%    : {t1['dAP_rel_best_%']:.1f}% -> "
          f"{'PASS' if t1['dAP_rel_best_%'] < 83 else 'FAIL'}")


if __name__ == "__main__":
    main()
