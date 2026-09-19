"""Parameter sweep for NeuroDeadwoodNet decoding on DeadTrees val set."""
from __future__ import annotations


# Ensure workspace root is in sys.path
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "crown_segmentation_research").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from crown_segmentation_research.methods.dynamic_mask.neuro_deadwood_net import NeuroDeadwoodNet
from crown_segmentation_research.methods.dynamic_mask.neuro_deadwood_decode import decode_neuro_deadwood

VAL_DIR = Path("DeadTrees/star_convex_targets_v1/val")
MODEL_PATH = Path("DeadTrees/experiments/neuro_deadwood_unified_v2/best_model.pth")


def compute_iou(poly1: Polygon, poly2: Polygon) -> float:
    if not poly1.is_valid or not poly2.is_valid:
        poly1 = poly1.buffer(0)
        poly2 = poly2.buffer(0)
    if not poly1.intersects(poly2):
        return 0.0
    try:
        inter = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def evaluate_matching(pred_polys: list[Polygon], gt_polys: list[Polygon], iou_thresh: float = 0.25):
    n_pred = len(pred_polys)
    n_gt = len(gt_polys)
    if n_gt == 0:
        return {"tp": 0, "fp": n_pred, "fn": 0}
    if n_pred == 0:
        return {"tp": 0, "fp": 0, "fn": n_gt}

    cost_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)
    for i, p in enumerate(pred_polys):
        for j, g in enumerate(gt_polys):
            cost_matrix[i, j] = compute_iou(p, g)

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    tp = sum(1 for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] >= iou_thresh)
    return {"tp": tp, "fp": n_pred - tp, "fn": n_gt - tp}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuroDeadwoodNet(n_directions=16, pretrained_backbone=False).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    val_files = sorted(VAL_DIR.glob("*.npz"))
    cached_data = []

    print("Caching network predictions on val set...")
    for path in val_files:
        data = np.load(path)
        img = data["image"]
        prob_gt = data["probability"]
        gt_mask = (prob_gt > 0.05).astype(np.uint8)
        cnts, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gt_polys = [Polygon(c.squeeze(1)) for c in cnts if len(c) >= 3 and cv2.contourArea(c) >= 10]

        img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            out = model(img_t)
            prob = out["probability"].squeeze(0).squeeze(0).cpu().numpy()
            orient = out["orientation"].squeeze(0).squeeze(0).cpu().numpy()
            radii = out["extent_radii"].squeeze(0).cpu().numpy()
            canopy = out["canopy"].squeeze(0).squeeze(0).cpu().numpy()

        cached_data.append({
            "prob": prob,
            "orient": orient,
            "radii": radii,
            "canopy": canopy,
            "gt_polys": gt_polys,
        })

    prob_thresholds = [0.55, 0.58, 0.60, 0.62, 0.65, 0.70]
    min_peak_distances = [8, 12, 16, 20]
    min_areas = [20.0, 40.0, 80.0]

    best_f1 = 0.0
    best_params = {}
    results = []

    print(f"\nSweeping {len(prob_thresholds) * len(min_peak_distances) * len(min_areas)} configurations...")
    for p_thresh in prob_thresholds:
        for min_dist in min_peak_distances:
            for m_area in min_areas:
                tot_tp, tot_fp, tot_fn = 0, 0, 0
                for item in cached_data:
                    preds = decode_neuro_deadwood(
                        probability=item["prob"],
                        orientation=item["orient"],
                        extent_radii=item["radii"],
                        canopy=item["canopy"],
                        prob_threshold=p_thresh,
                        min_peak_distance=min_dist,
                        min_area=m_area,
                    )
                    res = evaluate_matching(preds, item["gt_polys"])
                    tot_tp += res["tp"]
                    tot_fp += res["fp"]
                    tot_fn += res["fn"]

                prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
                rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

                record = {
                    "p_thresh": p_thresh,
                    "min_dist": min_dist,
                    "min_area": m_area,
                    "tp": tot_tp,
                    "fp": tot_fp,
                    "fn": tot_fn,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                }
                results.append(record)
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = record

    print("\n" + "=" * 60)
    print("BEST DECODING PARAMETERS FOR NeuroDeadwoodNet:")
    print(f"  prob_threshold:     {best_params.get('p_thresh')}")
    print(f"  min_peak_distance:  {best_params.get('min_dist')}")
    print(f"  min_area:           {best_params.get('min_area')}")
    print(f"  Precision:          {best_params.get('precision', 0):.4f} ({best_params.get('precision', 0)*100:.2f}%)")
    print(f"  Recall:             {best_params.get('recall', 0):.4f} ({best_params.get('recall', 0)*100:.2f}%)")
    print(f"  F1-Score:           {best_params.get('f1', 0):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
