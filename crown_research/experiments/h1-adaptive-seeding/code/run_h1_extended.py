"""H1 follow-up: the pre-registered Arm-A grid hit its upper boundary at
(min_dist=56, min_apex=0.45), so the optimum was outside the locked grid.

EXPLORATORY (not part of the H1 pre-registration): extend the global-radius grid and,
separately, measure *where* the residual gap to the oracle lives — is it seed count,
or seed placement?
"""
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

import cv2
import numpy as np
import pandas as pd

from crown_research.src.pred_cache import load_cache
from crown_research.src.seeding import flood, global_nms_seeds
from crown_segmentation_research.methods.boundary_refinement.metrics_dao import compute_dao_segmentation_indices
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import evaluate_metrics

OUT = Path("crown_research/experiments/h1-adaptive-seeding/results")


def seed_placement_audit(crops, seed_fn):
    """Decompose seeds against GT: how many crowns get exactly one seed?

    hit      = GT crowns containing >= 1 seed
    dup      = seeds beyond the first inside an already-seeded crown (splits a crown)
    orphan   = seeds landing on background / no GT crown (spurious instance)
    missed   = GT crowns with no seed at all (unrecoverable miss)
    """
    tot = {"n_gt": 0, "n_seed": 0, "hit": 0, "dup": 0, "orphan": 0, "missed": 0}
    for surf, bound, canopy, gt in crops:
        seeds = seed_fn(surf, canopy)
        gids = np.unique(gt)
        gids = gids[gids >= 2]
        tot["n_gt"] += len(gids)
        tot["n_seed"] += len(seeds)
        seen = {}
        for y, x, _ in seeds:
            g = int(gt[y, x])
            if g < 2:
                tot["orphan"] += 1
            else:
                seen[g] = seen.get(g, 0) + 1
        tot["hit"] += len(seen)
        tot["dup"] += sum(v - 1 for v in seen.values())
        tot["missed"] += len(gids) - len(seen)
    return tot


def main():
    crops = list(load_cache("eval30"))
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[H1-ext] {len(crops)} crops", flush=True)

    rows = []
    for md, ma in product([40, 48, 56, 64, 72, 80, 96, 112], [0.35, 0.45, 0.55, 0.65, 0.75]):
        acc = []
        for surf, bound, canopy, gt in crops:
            seeds = global_nms_seeds(surf, canopy, float(md), ma)
            m, insts = flood(seeds, bound, canopy)
            ev = evaluate_metrics(m, insts, gt)
            da = compute_dao_segmentation_indices(m, gt)
            n_gt = int((np.unique(gt) >= 2).sum())
            acc.append({**{k: ev[k] for k in ("ap50", "ap75", "pq", "sq", "rq")},
                        "os": da["mean_os"], "us": da["mean_us"], "ed": da["mean_ed"],
                        "ratio": len(insts) / max(1, n_gt)})
        d = pd.DataFrame(acc).mean()
        rows.append({"min_dist": md, "min_apex": ma, **d.to_dict()})
        print(f"  md={md:>3} ma={ma:.2f} -> ap50={d['ap50']:.2f} pq={d['pq']:.2f} ratio={d['ratio']:.2f}", flush=True)

    df = pd.DataFrame(rows).sort_values(["ap50", "pq"], ascending=False)
    df.to_csv(OUT / "h1a_extended_sweep.csv", index=False)
    print("\nTop-8 extended Arm A:\n", df.head(8).round(3).to_string(index=False))

    best = df.iloc[0]
    print("\n=== Seed placement audit ===")
    audits = []
    for name, fn in [
        ("baseline md=24 ma=0.25", lambda s, c: global_nms_seeds(s, c, 24.0, 0.25)),
        (f"tuned md={int(best.min_dist)} ma={best.min_apex}",
         lambda s, c: global_nms_seeds(s, c, float(best.min_dist), float(best.min_apex))),
    ]:
        a = seed_placement_audit(crops, fn)
        a["arm"] = name
        a["hit_rate_%"] = round(100.0 * a["hit"] / a["n_gt"], 2)
        a["dup_per_gt"] = round(a["dup"] / a["n_gt"], 3)
        a["orphan_per_gt"] = round(a["orphan"] / a["n_gt"], 3)
        a["missed_rate_%"] = round(100.0 * a["missed"] / a["n_gt"], 2)
        audits.append(a)
    df_aud = pd.DataFrame(audits)[["arm", "n_gt", "n_seed", "hit_rate_%", "dup_per_gt",
                                   "orphan_per_gt", "missed_rate_%"]]
    df_aud.to_csv(OUT / "h1_seed_audit.csv", index=False)
    print(df_aud.to_string(index=False))


if __name__ == "__main__":
    main()
