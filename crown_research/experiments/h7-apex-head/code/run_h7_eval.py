"""H7 evaluation — does seeding from the scale-invariant apex head beat the surface head?

Caches the 4-head outputs, re-runs the missed-crown analysis for both heads, sweeps each
head's own (min_dist, min_apex), and scores end-to-end. See ../protocol.md.
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

import argparse
import numpy as np
import pandas as pd
import torch

from crown_research.src.coco_ap import APAccumulator, instance_scores_from_surface
from crown_research.src.seeding import _raw_peaks, flood, global_nms_seeds
from crown_segmentation_research.methods.canopy_watershed.dataset import BAMCanopyWatershedDataset
from crown_segmentation_research.methods.canopy_watershed.model_apex import CanopyWatershedNet
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import evaluate_metrics

OUT = Path("crown_research/experiments/h7-apex-head/results")
CACHE = Path("crown_research/data/cache/h7_apex")
CKPT = "DeadTrees/experiments/canopy_watershed/best_apex_watershed.pth"


def build(ckpt, n=60, start=100, split="eval"):
    CACHE.mkdir(parents=True, exist_ok=True)
    if len(list(CACHE.glob("*.npz"))) >= n:
        print(f"[H7] cache already has {len(list(CACHE.glob('*.npz')))} crops", flush=True)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CanopyWatershedNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model_state_dict"])
    model.eval()
    ds = BAMCanopyWatershedDataset(split=split, crop_size=1024, augment=False, compute_targets=False)
    for idx in range(start, min(start + n, len(ds))):
        f = CACHE / f"crop_{idx:04d}.npz"
        if f.exists():
            continue
        it = ds[idx]
        raw = it["instance_label"].squeeze().numpy().astype(np.int32)
        gt = np.zeros_like(raw); gt[raw > 0] = raw[raw > 0] + 1
        if len(np.unique(raw[raw > 0])) == 0:
            continue
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            p = model(it["image"].unsqueeze(0).to(device))
        np.savez_compressed(f, **{k: p[k].squeeze().float().cpu().numpy().astype(np.float16)
                                  for k in ("surface", "boundary", "canopy", "apex")},
                            gt_map=gt.astype(np.int16))
    print(f"[H7] cached {len(list(CACHE.glob('*.npz')))} crops", flush=True)


def load():
    for f in sorted(CACHE.glob("*.npz")):
        z = np.load(f)
        yield {k: z[k].astype(np.float32) for k in ("surface", "boundary", "canopy", "apex")} | \
              {"gt": z["gt_map"].astype(np.int32)}


def never_lit(crops, field, md, ma):
    """Fraction of GT crowns whose seeding field never exceeds `ma` inside them."""
    tot = miss = nl = 0
    for c in crops:
        gt, f = c["gt"], c[field]
        seeded = {int(gt[y, x]) for y, x, _ in global_nms_seeds(f, c["canopy"], md, ma)}
        for g in np.unique(gt):
            if g < 2:
                continue
            tot += 1
            if int(g) in seeded:
                continue
            miss += 1
            if f[gt == g].max() < ma:
                nl += 1
    return {"missed_%": 100.0 * miss / tot, "never_lit_%": 100.0 * nl / tot, "n_gt": tot}


def score(crops, field, md, ma):
    acc = APAccumulator(); f1 = []; ap75 = []; ratios = []
    for c in crops:
        seeds = global_nms_seeds(c[field], c["canopy"], md, ma)
        m, insts = flood(seeds, c["boundary"], c["canopy"])
        insts = instance_scores_from_surface(m, insts, c["surface"])
        acc.add(m, insts, c["gt"])
        ev = evaluate_metrics(m, insts, c["gt"])
        f1.append(ev["ap50"]); ap75.append(ev["ap75"])
        ratios.append(len(insts) / max(1, int((np.unique(c["gt"]) >= 2).sum())))
    r = acc.ap()
    return {"COCO AP": r["AP"], "AP50": r["AP50"], "AP75": r["AP75"],
            "F1@50": float(np.mean(f1)), "F1@75": float(np.mean(ap75)),
            "ratio": float(np.mean(ratios))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    build(a.ckpt)
    crops = list(load())
    print(f"[H7] {len(crops)} crops", flush=True)

    rows = []
    for field, grid_ma in (("surface", [0.35, 0.50, 0.65]), ("apex", [0.10, 0.20, 0.35, 0.50])):
        for md, ma in product([60, 100, 140, 160, 200], grid_ma):
            r = score(crops, field, float(md), ma)
            rows.append({"seed_field": field, "min_dist": md, "min_apex": ma, **r})
            print(f"  {field:<8} md={md:>3} ma={ma:.2f} -> AP {r['COCO AP']:5.2f} | "
                  f"F1@50 {r['F1@50']:5.2f} | ratio {r['ratio']:.2f}", flush=True)
    df = pd.DataFrame(rows).round(3)
    df.to_csv(OUT / "h7_seed_sweep.csv", index=False)

    best = {f: df[df.seed_field == f].sort_values("COCO AP", ascending=False).iloc[0]
            for f in ("surface", "apex")}
    nl = []
    for f, b in best.items():
        nl.append({"seed_field": f, "min_dist": b.min_dist, "min_apex": b.min_apex,
                   **never_lit(crops, f, float(b.min_dist), float(b.min_apex))})
    dnl = pd.DataFrame(nl).round(2)
    dnl.to_csv(OUT / "h7_missed_crowns.csv", index=False)

    summ = pd.DataFrame([{"seed_field": f, "min_dist": b.min_dist, "min_apex": b.min_apex,
                          **{k: b[k] for k in ("COCO AP", "AP50", "AP75", "F1@50", "F1@75", "ratio")}}
                         for f, b in best.items()]).round(3)
    summ.to_csv(OUT / "h7_system.csv", index=False)

    print("\n" + "=" * 104)
    print(summ.to_string(index=False)); print()
    print(dnl.to_string(index=False)); print("=" * 104)

    s_ap, a_ap = best["surface"]["COCO AP"], best["apex"]["COCO AP"]
    s_f1, a_f1 = best["surface"]["F1@50"], best["apex"]["F1@50"]
    apex_nl = dnl[dnl.seed_field == "apex"]["never_lit_%"].iloc[0]
    print(f"\nP1 apex never_lit < 15%   : {apex_nl:.2f}% -> {'PASS' if apex_nl < 15 else 'FAIL'}")
    print(f"P2 apex F1@50 >= surface+3: {a_f1:.2f} vs {s_f1:.2f} -> "
          f"{'PASS' if a_f1 >= s_f1 + 3 else 'FAIL'}")
    print(f"P3 best F1@50 >= 73       : {max(a_f1, s_f1):.2f} -> "
          f"{'PASS' if max(a_f1, s_f1) >= 73 else 'FAIL'}")
    print(f"   (COCO AP: surface {s_ap:.2f} vs apex {a_ap:.2f}; H6 reference 26.84)")


if __name__ == "__main__":
    main()
