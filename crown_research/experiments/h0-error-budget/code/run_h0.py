"""H0 — Error-budget decomposition of the ITC watershed via a 2x2 oracle ablation.

Factors: seeds in {predicted apices, oracle GT-EDT maxima}
         energy in {predicted boundary head B, oracle GT seam map}

Everything else (flooding algorithm, min_area, metric code) is held fixed, so each
cell isolates one source of error. See ../protocol.md for the pre-registered design.
"""
from __future__ import annotations

import sys
from pathlib import Path

_conda_p = '/home/ubuntu/anaconda3/envs/cs2/lib/python3.11/site-packages'
_local_p = '/home/chung/.local/lib/python3.11/site-packages'
sys.path = [p for p in sys.path if p != _local_p and p != _conda_p]
sys.path.insert(0, _conda_p)
sys.path.append(_local_p)
sys.path.insert(0, "/home/chung/RS/Image Segmentation")

import argparse
import cv2
import numpy as np
import pandas as pd
import torch

from crown_segmentation_research.methods.boundary_refinement.metrics_dao import (
    compute_boundary_iou,
    compute_dao_segmentation_indices,
)
from crown_segmentation_research.methods.canopy_watershed.dataset import BAMCanopyWatershedDataset
from crown_segmentation_research.methods.canopy_watershed.model import CanopyWatershedNet
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import evaluate_metrics

CKPT = "DeadTrees/experiments/canopy_watershed/best_sharp_watershed.pth"
MIN_DIST, MIN_APEX, MIN_CANOPY, MIN_AREA = 24.0, 0.25, 0.25, 25


def predicted_seeds(surf, canopy, min_dist=MIN_DIST, min_apex=MIN_APEX, min_canopy=MIN_CANOPY):
    """Apex extraction identical to the published flat-basin baseline."""
    k = max(3, int(min_dist) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    peaks = (surf == cv2.dilate(surf, kernel)) & (surf >= min_apex) & (canopy >= min_canopy)
    ys, xs = np.where(peaks)
    if len(ys) == 0:
        return []
    scores = surf[ys, xs]
    order = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]

    kept = []
    for y, x, s in zip(ys, xs, scores):
        if not any((y - ky) ** 2 + (x - kx) ** 2 < min_dist ** 2 for ky, kx, _ in kept):
            kept.append((int(y), int(x), float(s)))
    return kept


def oracle_seeds(gt_map):
    """One seed per GT crown, placed at the crown's distance-transform maximum.

    The EDT max is the most interior pixel, so the seed is guaranteed inside the
    crown even for strongly non-convex or crescent-shaped canopies.
    """
    seeds = []
    for gid in np.unique(gt_map):
        if gid < 2:
            continue
        m = (gt_map == gid).astype(np.uint8)
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
        seeds.append((int(y), int(x), 1.0))
    return seeds


def oracle_energy(gt_map):
    """GT inter-instance seams as a blurred ridge, matching the scale of B."""
    lab = gt_map.astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seam = (cv2.morphologyEx(lab, cv2.MORPH_GRADIENT, kernel) > 0).astype(np.float32)
    return np.clip(cv2.GaussianBlur(seam, (0, 0), 1.0) * 3.0, 0.0, 1.0)


def flood(seeds, energy, canopy, min_canopy=MIN_CANOPY, min_area=MIN_AREA):
    """Flat-basin watershed: relief is the ridge field alone, basins are flat."""
    H, W = energy.shape
    if not seeds:
        return np.zeros((H, W), np.int32), []
    markers = np.zeros((H, W), np.int32)
    markers[canopy < min_canopy] = 1
    for i, (y, x, _) in enumerate(seeds):
        markers[max(0, y - 1):min(H, y + 2), max(0, x - 1):min(W, x + 2)] = i + 2

    relief = (energy * 255.0).clip(0, 255).astype(np.uint8)
    cv2.watershed(cv2.merge([relief] * 3), markers)

    areas = np.bincount(markers[markers >= 2].ravel(), minlength=len(seeds) + 2)
    insts = []
    for i, (y, x, s) in enumerate(seeds):
        iid = i + 2
        area = int(areas[iid]) if iid < len(areas) else 0
        if area >= min_area:
            insts.append({"id": iid, "apex": (y, x), "score": s, "area": area})
    return markers, insts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_val", type=int, default=30)
    ap.add_argument("--crop_size", type=int, default=1024)
    ap.add_argument("--out_dir", type=str, default="crown_research/experiments/h0-error-budget/results")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CanopyWatershedNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model_state_dict"])
    model.eval()

    ds = BAMCanopyWatershedDataset(split="eval", crop_size=args.crop_size, augment=False, compute_targets=False)
    n = min(args.num_val, len(ds))

    cells = {
        "A_pred-seed_pred-energy": {},
        "B_oracle-seed_pred-energy": {},
        "C_pred-seed_oracle-energy": {},
        "D_oracle-seed_oracle-energy": {},
    }
    rows, seed_rows = [], []

    print(f"[H0] {n} crops on {device}", flush=True)
    for idx in range(n):
        item = ds[idx]
        raw_gt = item["instance_label"].squeeze().numpy().astype(np.int32)
        gt_map = np.zeros_like(raw_gt)
        gt_map[raw_gt > 0] = raw_gt[raw_gt > 0] + 1
        n_gt = len(np.unique(raw_gt[raw_gt > 0]))
        if n_gt == 0:
            continue

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            p = model(item["image"].unsqueeze(0).to(device))
        surf = p["surface"].squeeze().float().cpu().numpy()
        bound = p["boundary"].squeeze().float().cpu().numpy()
        canopy = p["canopy"].squeeze().float().cpu().numpy()

        s_pred = predicted_seeds(surf, canopy)
        s_orac = oracle_seeds(gt_map)
        e_pred = bound
        e_orac = oracle_energy(gt_map)

        seed_rows.append({"crop": idx, "n_gt": n_gt, "n_pred_seed": len(s_pred),
                          "ratio": len(s_pred) / max(1, n_gt)})

        for name, (seeds, energy) in {
            "A_pred-seed_pred-energy": (s_pred, e_pred),
            "B_oracle-seed_pred-energy": (s_orac, e_pred),
            "C_pred-seed_oracle-energy": (s_pred, e_orac),
            "D_oracle-seed_oracle-energy": (s_orac, e_orac),
        }.items():
            m, insts = flood(seeds, energy, canopy)
            ev = evaluate_metrics(m, insts, gt_map)
            da = compute_dao_segmentation_indices(m, gt_map)
            bi = compute_boundary_iou(m, gt_map)
            rec = {"cell": name, "crop": idx, "n_gt": n_gt, "n_pred": len(insts),
                   **{k: ev[k] for k in ("ap50", "ap75", "pq", "sq", "rq")},
                   "biou": bi * 100.0, "os": da["mean_os"], "us": da["mean_us"], "ed": da["mean_ed"]}
            rows.append(rec)
            cells[name].setdefault("_", None)

        if (idx + 1) % 5 == 0:
            print(f"  [{idx+1:02d}/{n}]", flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "h0_per_crop.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out / "h0_seed_counts.csv", index=False)

    agg = df.groupby("cell").agg(
        n_pred=("n_pred", "mean"), n_gt=("n_gt", "mean"),
        ap50=("ap50", "mean"), ap75=("ap75", "mean"), pq=("pq", "mean"),
        sq=("sq", "mean"), rq=("rq", "mean"), biou=("biou", "mean"),
        OS=("os", "mean"), US=("us", "mean"), ED=("ed", "mean"),
    ).round(3)
    agg.to_csv(out / "h0_error_budget.csv")

    print("\n" + "=" * 110)
    print(agg.to_string())
    print("=" * 110)
    sr = pd.DataFrame(seed_rows)
    print(f"\nSeed density: mean n_pred/n_gt = {sr['ratio'].mean():.3f} "
          f"(n_pred={sr['n_pred_seed'].mean():.1f}, n_gt={sr['n_gt'].mean():.1f})")


if __name__ == "__main__":
    main()
