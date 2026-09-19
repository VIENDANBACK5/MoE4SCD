"""H3 — full 2^3 error budget over {seeds} x {energy} x {canopy gate}. See ../protocol.md."""
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

import cv2
import numpy as np
import pandas as pd

from crown_research.src.pred_cache import load_cache
from crown_research.src.seeding import flood, global_nms_seeds
from crown_segmentation_research.methods.boundary_refinement.metrics_dao import (
    compute_boundary_iou, compute_dao_segmentation_indices)
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import evaluate_metrics

OUT = Path("crown_research/experiments/h3-canopy-gate/results")
MD, MA = 160.0, 0.50  # H1-tuned


def oracle_seeds(gt):
    seeds = []
    for gid in np.unique(gt):
        if gid < 2:
            continue
        dt = cv2.distanceTransform((gt == gid).astype(np.uint8), cv2.DIST_L2, 3)
        y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
        seeds.append((int(y), int(x), 1.0))
    return seeds


def oracle_energy(gt):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seam = (cv2.morphologyEx(gt.astype(np.float32), cv2.MORPH_GRADIENT, k) > 0).astype(np.float32)
    return np.clip(cv2.GaussianBlur(seam, (0, 0), 1.0) * 3.0, 0.0, 1.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, tag in (("eval30 (tune)", "eval30"), ("evalhold60 (hold-out)", "evalhold60")):
        crops = list(load_cache(tag))
        print(f"\n[H3] {split}: {len(crops)} crops", flush=True)
        for s_o, e_o, c_o in product([False, True], repeat=3):
            acc = []
            for surf, bound, canopy, gt in crops:
                # oracle canopy is a hard 0/1 mask; the 0.25 gate then selects gt >= 2
                cn = (gt >= 2).astype(np.float32) if c_o else canopy
                seeds = oracle_seeds(gt) if s_o else global_nms_seeds(surf, cn, MD, MA)
                energy = oracle_energy(gt) if e_o else bound
                m, insts = flood(seeds, energy, cn)
                ev = evaluate_metrics(m, insts, gt)
                da = compute_dao_segmentation_indices(m, gt)
                n_gt = int((np.unique(gt) >= 2).sum())
                acc.append({**{k: ev[k] for k in ("ap50", "ap75", "pq", "sq", "rq")},
                            "biou": compute_boundary_iou(m, gt) * 100.0,
                            "os": da["mean_os"], "us": da["mean_us"], "ed": da["mean_ed"],
                            "ratio": len(insts) / max(1, n_gt)})
            df = pd.DataFrame(acc)
            cell = f"{'O' if s_o else 'p'}seed/{'O' if e_o else 'p'}energy/{'O' if c_o else 'p'}canopy"
            rows.append({"cell": cell, "split": split, "seed_O": s_o, "energy_O": e_o,
                         "canopy_O": c_o, **df.mean().to_dict(), "ap50_sem": df["ap50"].sem()})
            print(f"  {cell:<34} ap50={df['ap50'].mean():6.2f}  ap75={df['ap75'].mean():6.2f}  "
                  f"pq={df['pq'].mean():6.2f}  sq={df['sq'].mean():6.2f}", flush=True)

    d = pd.DataFrame(rows).round(3)
    d.to_csv(OUT / "h3_factorial.csv", index=False)

    # Main effects: mean ap50 lift from switching each factor to oracle, averaged over the cube
    eff = []
    for split in d.split.unique():
        sub = d[d.split == split]
        for f in ("seed_O", "energy_O", "canopy_O"):
            eff.append({"split": split, "factor": f.replace("_O", ""),
                        "main_effect_ap50": round(sub[sub[f]].ap50.mean() - sub[~sub[f]].ap50.mean(), 3),
                        "main_effect_sq": round(sub[sub[f]].sq.mean() - sub[~sub[f]].sq.mean(), 3)})
        base = sub[(~sub.seed_O) & (~sub.energy_O) & (~sub.canopy_O)].ap50.iloc[0]
        s_only = sub[(sub.seed_O) & (~sub.energy_O) & (~sub.canopy_O)].ap50.iloc[0]
        c_only = sub[(~sub.seed_O) & (~sub.energy_O) & (sub.canopy_O)].ap50.iloc[0]
        sc = sub[(sub.seed_O) & (~sub.energy_O) & (sub.canopy_O)].ap50.iloc[0]
        eff.append({"split": split, "factor": "seed x canopy interaction",
                    "main_effect_ap50": round(sc - s_only - c_only + base, 3), "main_effect_sq": np.nan})
    de = pd.DataFrame(eff)
    de.to_csv(OUT / "h3_main_effects.csv", index=False)

    print("\n" + "=" * 128)
    print(d[["cell", "split", "ap50", "ap50_sem", "ap75", "pq", "sq", "rq", "biou", "ed", "ratio"]].to_string(index=False))
    print("=" * 128)
    print("\nMain effects (mean mAP@50 lift from making a factor oracle):\n", de.to_string(index=False))

    h = d[d.split == "evalhold60 (hold-out)"].set_index("cell")["ap50"]
    p1 = h["pseed/penergy/Ocanopy"] - h["pseed/penergy/pcanopy"]
    print(f"\nPrediction 1  canopy-only lift >= +8 : {p1:+.2f} -> {'PASS' if p1 >= 8 else 'FAIL'}")
    print(f"Prediction 2  Oseed+Ocanopy >= 85   : {h['Oseed/penergy/Ocanopy']:.2f} -> "
          f"{'PASS' if h['Oseed/penergy/Ocanopy'] >= 85 else 'FAIL'}")
    print(f"Prediction 3  full oracle >= 92     : {h['Oseed/Oenergy/Ocanopy']:.2f} -> "
          f"{'PASS' if h['Oseed/Oenergy/Ocanopy'] >= 92 else 'FAIL'}")


if __name__ == "__main__":
    main()
