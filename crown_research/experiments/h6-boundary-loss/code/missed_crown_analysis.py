"""Open question 3 — what kind of crown never gets a seed?

EXPLORATORY. Under loose suppression 16.9% of GT crowns receive no seed at all, which
bounds recall before any decoding happens. This classifies each missed crown by the
most likely cause, using only cached head outputs.

Causes distinguished:
  never_lit    - the surface head peaks below min_apex anywhere inside the crown
  gated_out    - surface peaks fine but the canopy head is below 0.25 there
  suppressed   - a valid peak exists and passes the gate, but NMS removed it in favour
                 of a stronger nearby peak (i.e. the crown was outranked, not unseen)
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

import cv2
import numpy as np
import pandas as pd

from crown_research.src.pred_cache import load_cache
from crown_research.src.seeding import _raw_peaks, global_nms_seeds

OUT = Path("crown_research/experiments/h6-boundary-loss/results")
SETTINGS = [("loose md=24 ma=0.25", 24.0, 0.25), ("tuned md=160 ma=0.50", 160.0, 0.50)]


def analyse(tag):
    rows = []
    for surf, bound, canopy, gt in load_cache(tag):
        gids = np.unique(gt)
        gids = gids[gids >= 2]
        areas = {int(g): int((gt == g).sum()) for g in gids}

        for name, md, ma in SETTINGS:
            seeded = {int(gt[y, x]) for y, x, _ in global_nms_seeds(surf, canopy, md, ma)}
            # candidate peaks BEFORE the NMS pass, at the same thresholds
            py, px, _ = _raw_peaks(surf, canopy, md, ma, 0.25)
            cand = {int(gt[y, x]) for y, x in zip(py, px)}

            for g in gids:
                g = int(g)
                if g in seeded:
                    continue
                m = gt == g
                if surf[m].max() < ma:
                    cause = "never_lit"
                elif canopy[m].max() < 0.25:
                    cause = "gated_out"
                elif g in cand:
                    cause = "suppressed"
                else:
                    cause = "no_local_max"  # lit and gated in, but never a local maximum
                rows.append({"split": tag, "setting": name, "gt_id": g,
                             "area": areas[g], "cause": cause,
                             "surf_max": float(surf[m].max()),
                             "canopy_max": float(canopy[m].max())})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.concat([analyse(t) for t in ("eval30", "evalhold60")], ignore_index=True)
    df.to_csv(OUT / "missed_crowns.csv", index=False)

    # totals per setting need the GT counts, recomputed cheaply
    totals = {}
    for tag in ("eval30", "evalhold60"):
        n = sum(int(((np.unique(gt) >= 2).sum())) for _, _, _, gt in load_cache(tag))
        totals[tag] = n

    print("=" * 100)
    for tag in ("eval30", "evalhold60"):
        for name, _, _ in SETTINGS:
            sub = df[(df.split == tag) & (df.setting == name)]
            n = totals[tag]
            print(f"\n{tag} · {name} · {len(sub)}/{n} crowns missed ({100*len(sub)/n:.1f}%)")
            vc = sub.cause.value_counts()
            for c, k in vc.items():
                med = sub[sub.cause == c].area.median()
                print(f"    {c:<14} {k:>4}  ({100*k/n:>5.1f}% of all GT)   median area {med:>8.0f} px")
    print("\n" + "=" * 100)
    hit = df[(df.split == "evalhold60") & (df.setting == "loose md=24 ma=0.25")]
    all_areas = []
    for _, _, _, gt in load_cache("evalhold60"):
        for g in np.unique(gt):
            if g >= 2:
                all_areas.append(int((gt == g).sum()))
    print(f"\nHold-out crown area: median {np.median(all_areas):.0f} px, "
          f"p10 {np.percentile(all_areas,10):.0f}, p90 {np.percentile(all_areas,90):.0f}")
    print(f"Missed (loose) crown area: median {hit.area.median():.0f} px")


if __name__ == "__main__":
    main()
