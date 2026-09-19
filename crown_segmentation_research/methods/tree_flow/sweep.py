"""Fast decode hyperparameter sweep for TreeFlowNet.

Caches raw predictions of best TreeFlowNet checkpoint on validation images once,
then rapidly sweeps decode hyperparameters (centroid_thresh, canopy_thresh, sdt_thresh, min_peak_dist, step_size, n_steps).
"""

from __future__ import annotations


# Ensure workspace root is in sys.path
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "crown_segmentation_research").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from shapely.geometry import Polygon, MultiPolygon

from crown_segmentation_research.methods.tree_flow.decode import decode_flow_to_instances
from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet
from deadtrees_pipeline.metrics import hungarian_match, overlap_matrices

VAL_DIR = Path("DeadTrees/star_convex_targets_v1/val")
MIN_GT_AREA = 20
MIN_PRED_AREA = 30


def polygons_to_masks(polygons: list[Polygon], shape: tuple[int, int]) -> np.ndarray:
    masks = []
    for polygon in polygons:
        mask = np.zeros(shape, dtype=np.uint8)
        subs = polygon.geoms if isinstance(polygon, MultiPolygon) else [polygon]
        for sub in subs:
            coords = np.array(sub.exterior.coords).round().astype(np.int32)
            cv2.fillPoly(mask, [coords], 1)
        masks.append(mask.astype(bool))
    return np.stack(masks) if masks else np.zeros((0, *shape), dtype=bool)


def load_val_items():
    items = []
    for npz_path in sorted(VAL_DIR.glob("*.npz")):
        data = np.load(npz_path)
        image = data["image"].astype(np.float32) / 255.0
        label_map = data["instance_label"]
        gt_masks = []
        for label in np.unique(label_map):
            if label == 0:
                continue
            mask = label_map == label
            if mask.sum() >= MIN_GT_AREA:
                gt_masks.append(mask)
        gt_masks = np.stack(gt_masks) if gt_masks else np.zeros((0, *image.shape[:2]), dtype=bool)
        items.append((npz_path.stem, image, gt_masks))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=Path("DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-csv", type=Path, default=Path("DeadTrees/experiments/tree_flow_decode_sweep.csv"))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    val_items = load_val_items()
    print(f"Loaded {len(val_items)} validation images", flush=True)

    model = TreeFlowNet(pretrained_backbone=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()

    # Cache forward passes
    cached = []
    with torch.no_grad():
        for stem, image, gt_masks in val_items:
            image_t = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device)
            out = model(image_t)
            flow = out["flow"].squeeze(0).cpu().numpy().astype(np.float32)
            sdt = out["sdt"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            centroid = out["centroid"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            canopy = out["canopy"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            cached.append((stem, image.shape[:2], gt_masks, flow, sdt, centroid, canopy))
    print(f"Cached forward outputs for {len(cached)} images. Starting hyperparameter grid sweep...", flush=True)

    canopy_thresholds = [0.3, 0.4, 0.5, 0.6]
    sdt_thresholds = [-0.5, -0.2, 0.0, 0.2]
    centroid_thresholds = [0.15, 0.20, 0.25, 0.30, 0.40]
    min_peak_distances = [3, 5, 8]
    step_sizes = [1.0, 1.5, 2.0]
    n_steps_list = [15, 25]

    grid = list(itertools.product(
        canopy_thresholds,
        sdt_thresholds,
        centroid_thresholds,
        min_peak_distances,
        step_sizes,
        n_steps_list,
    ))
    print(f"Total hyperparameter combinations: {len(grid)}", flush=True)

    rows = []
    for canopy_th, sdt_th, cent_th, min_dist, step_sz, n_st in grid:
        total_tp = total_fp = total_fn = 0
        for stem, shape, gt_masks, flow, sdt, centroid, canopy in cached:
            _, polygons = decode_flow_to_instances(
                flow, canopy, sdt, centroid,
                canopy_threshold=canopy_th,
                sdt_threshold=sdt_th,
                centroid_threshold=cent_th,
                min_peak_distance=min_dist,
                n_steps=n_st,
                step_size=step_sz,
                min_instance_area=MIN_PRED_AREA,
            )
            pred_masks = polygons_to_masks(polygons, shape)
            pred_areas = pred_masks.reshape(len(pred_masks), -1).sum(axis=1) if len(pred_masks) else np.zeros(0)
            pred_masks = pred_masks[pred_areas >= MIN_PRED_AREA]

            iou, _, _, _ = overlap_matrices(gt_masks, pred_masks)
            match = hungarian_match(iou, 0.50)
            total_tp += match.tp
            total_fp += match.fp
            total_fn += match.fn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        rows.append({
            "canopy_threshold": canopy_th,
            "sdt_threshold": sdt_th,
            "centroid_threshold": cent_th,
            "min_peak_distance": min_dist,
            "step_size": step_sz,
            "n_steps": n_st,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        })

    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved sweep results to {args.out_csv}", flush=True)
    print("\nTop 10 Hyperparameter Configurations:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
