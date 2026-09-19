"""H5 — rank-based use of a collapsed-magnitude boundary head. See ../protocol.md."""
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
from crown_research.src.rag_merge import (_relabel, fill_watershed_lines, merge_basins,
                                          seam_statistics, _UF)
from crown_research.src.seeding import flood, global_nms_seeds
from crown_segmentation_research.methods.boundary_refinement.metrics_dao import (
    compute_boundary_iou, compute_dao_segmentation_indices)
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import evaluate_metrics

OUT = Path("crown_research/experiments/h5-boundary-calibration/results")
MD, MA = 160.0, 0.50


def rank_transform(b, gamma=1.0):
    """Per-crop rank transform: maps B onto [0,1] by its own empirical CDF.

    Restores full dynamic range to a head whose magnitudes collapsed under class
    imbalance, while preserving exactly the ordering the head actually learned.
    """
    flat = b.ravel()
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(flat.size, np.float64)
    ranks[order] = np.arange(flat.size, dtype=np.float64)
    r = (ranks / max(1, flat.size - 1)).reshape(b.shape)
    return (r ** gamma).astype(np.float32)


def merge_percentile(markers, boundary, q=80, l_min=30, min_area=25):
    """H2's seam merge with a per-crop percentile threshold instead of an absolute one."""
    filled = fill_watershed_lines(markers)
    labels = np.unique(filled)
    labels = labels[labels >= 2]
    if len(labels) <= 1:
        return _relabel(filled, min_area)
    max_label = int(labels.max())
    s, c = seam_statistics(filled, boundary, max_label)
    au, av = np.nonzero(c >= l_min)
    uf = _UF(max_label + 1)
    if len(au):
        mean_ev = s[au, av] / c[au, av]
        thr = np.percentile(mean_ev, q)
        for a, b in zip(au[mean_ev < thr], av[mean_ev < thr]):
            uf.union(int(a), int(b))
    lut = np.arange(max_label + 1)
    for lab in labels:
        lut[lab] = uf.find(int(lab))
    merged = np.where(filled >= 2, lut[np.clip(filled, 0, max_label)], filled)
    return _relabel(merged, min_area)


def evaluate(crops, gamma=None, q=None, l_min=30):
    acc = []
    for surf, bound, canopy, gt in crops:
        relief = bound if gamma is None else rank_transform(bound, gamma)
        seeds = global_nms_seeds(surf, canopy, MD, MA)
        m, insts = flood(seeds, relief, canopy)
        if q is not None:
            m, insts = merge_percentile(m, bound, q=q, l_min=l_min)
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


def gap_audit(crops):
    """Arm D: does the canopy gate place hard barriers on GT crown-crown seams?"""
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    rows = []
    for surf, bound, canopy, gt in crops:
        grad = cv2.morphologyEx(gt.astype(np.float32), cv2.MORPH_GRADIENT, k3) > 0
        seam = grad & (gt >= 2)
        interior = (gt >= 2) & ~cv2.dilate(grad.astype(np.uint8), k3).astype(bool)
        bg = canopy < 0.25
        if seam.sum() and interior.sum():
            rows.append({"seam_in_bg_%": 100.0 * bg[seam].mean(),
                         "interior_in_bg_%": 100.0 * bg[interior].mean()})
    return pd.DataFrame(rows).mean()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tune = list(load_cache("eval30"))
    hold = list(load_cache("evalhold60"))

    rows = []
    for g in [0.5, 1.0, 2.0]:
        d = evaluate(tune, gamma=g)
        rows.append({"arm": "B rank relief", "gamma": g, "q": np.nan, **d})
        print(f"  B gamma={g} -> ap50={d['ap50']:.2f} sq={d['sq']:.2f} pq={d['pq']:.2f}", flush=True)
    for q in [50, 70, 80, 90, 95]:
        d = evaluate(tune, gamma=None, q=q)
        rows.append({"arm": "C percentile merge", "gamma": np.nan, "q": q, **d})
        print(f"  C q={q} -> ap50={d['ap50']:.2f} sq={d['sq']:.2f} pq={d['pq']:.2f} ratio={d['ratio']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "h5_sweep.csv", index=False)

    bg_ = df[df.arm == "B rank relief"].sort_values("ap50", ascending=False).iloc[0]
    bq = df[df.arm == "C percentile merge"].sort_values("ap50", ascending=False).iloc[0]
    print(f"\n  best rank gamma={bg_.gamma}, best percentile q={bq.q}\n", flush=True)

    arms = [("A control (raw B, no merge)", None, None),
            (f"B rank relief (gamma={bg_.gamma})", float(bg_.gamma), None),
            (f"C percentile merge (q={int(bq.q)})", None, int(bq.q)),
            (f"E rank relief + percentile merge", float(bg_.gamma), int(bq.q))]
    summary = []
    for split, crops in (("eval30 (tune)", tune), ("evalhold60 (hold-out)", hold)):
        for name, g, q in arms:
            summary.append({"arm": name, "split": split, **evaluate(crops, gamma=g, q=q)})
    ds = pd.DataFrame(summary).round(3)
    ds.to_csv(OUT / "h5_summary.csv", index=False)
    cols = ["arm", "split", "ap50", "ap50_sem", "ap75", "pq", "sq", "rq", "biou", "ed", "ratio"]
    print("=" * 132)
    print(ds[cols].to_string(index=False))
    print("=" * 132)

    ga = pd.DataFrame([{"split": t, **gap_audit(list(load_cache(t))).to_dict()}
                       for t in ("eval30", "evalhold60")]).round(2)
    ga.to_csv(OUT / "h5_gap_audit.csv", index=False)
    print("\nArm D — canopy gate as hard barrier:\n", ga.to_string(index=False))

    h = ds[ds.split == "evalhold60 (hold-out)"].set_index("arm")["ap50"]
    ctrl = h.iloc[0]
    print(f"\nPrediction 1  rank relief >= control+2 : {h.iloc[1]:.2f} vs {ctrl:.2f} -> "
          f"{'PASS' if h.iloc[1] >= ctrl + 2 else 'FAIL'}")
    print(f"Prediction 2  percentile merge >= 67.96 : {h.iloc[2]:.2f} -> "
          f"{'PASS' if h.iloc[2] >= 67.96 else 'FAIL'}")
    sm = ga[ga.split == "evalhold60"].iloc[0]["seam_in_bg_%"]
    print(f"Prediction 3  seam-in-background >= 30% : {sm:.2f}% -> {'PASS' if sm >= 30 else 'FAIL'}")


if __name__ == "__main__":
    main()
