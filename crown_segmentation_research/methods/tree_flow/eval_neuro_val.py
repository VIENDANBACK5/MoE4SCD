"""Comprehensive Validation Ablation for Neuro-Flow-Graph (NFG) vs TreeFlowNet Baseline.

Evaluates 4 configurations on held-out Site 5737 (20 tiles, 208 real instances):
  1. Config A: Vanilla TreeFlowNet (Euler flow integration only)
  2. Config B: TreeFlowNet + Medial Axis Bellman-Ford Bridging
  3. Config C: TreeFlowNet + Spectral Modularity Cut
  4. Config D: Full Neuro-Flow-Graph (NFG: Bridging + Modularity Cut)
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

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from shapely.geometry import MultiPolygon, Polygon

from crown_segmentation_research.methods.tree_flow.graph_decode import decode_neuro_flow_graph
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


def run_ablation_eval(
    model: TreeFlowNet,
    device: torch.device,
    val_items: list,
    use_bellman_ford: bool,
    use_spectral_cut: bool,
    canopy_threshold: float = 0.5,
    sdt_threshold: float = -0.2,
    centroid_threshold: float = 0.25,
    max_bridge_distance: float = 35.0,
    modularity_threshold: float = 0.08,
) -> dict:
    total_tp = total_fp = total_fn = 0
    total_splits = total_merges = 0
    n_gt_total = 0
    n_pred_total = 0

    with torch.no_grad():
        for stem, image, gt_masks in val_items:
            image_t = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device)
            out = model(image_t)
            flow = out["flow"].squeeze(0).cpu().numpy().astype(np.float32)
            sdt = out["sdt"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            centroid = out["centroid"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            canopy = out["canopy"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

            _, polygons = decode_neuro_flow_graph(
                flow, canopy, sdt, centroid,
                canopy_threshold=canopy_threshold,
                sdt_threshold=sdt_threshold,
                centroid_threshold=centroid_threshold,
                min_instance_area=MIN_PRED_AREA,
                use_bellman_ford=use_bellman_ford,
                use_spectral_cut=use_spectral_cut,
                max_bridge_distance=max_bridge_distance,
                modularity_threshold=modularity_threshold,
            )

            pred_masks = polygons_to_masks(polygons, image.shape[:2])
            pred_areas = pred_masks.reshape(len(pred_masks), -1).sum(axis=1) if len(pred_masks) else np.zeros(0)
            pred_masks = pred_masks[pred_areas >= MIN_PRED_AREA]

            n_gt_total += len(gt_masks)
            n_pred_total += len(pred_masks)

            if len(gt_masks) > 0 and len(pred_masks) > 0:
                iou, _, gt_cov, pred_cov = overlap_matrices(gt_masks, pred_masks)
                match = hungarian_match(iou, 0.50)
                total_tp += match.tp
                total_fp += match.fp
                total_fn += match.fn

                # Calculate splits and merges (gt_cov >= 0.3)
                # Split: 1 GT covered by >=2 preds
                splits = (gt_cov >= 0.3).sum(axis=1)
                total_splits += int((splits > 1).sum())
                # Merge: 1 Pred covering >=2 GTs
                merges = (pred_cov >= 0.3).sum(axis=0)
                total_merges += int((merges > 1).sum())
            elif len(gt_masks) > 0:
                total_fn += len(gt_masks)
            elif len(pred_masks) > 0:
                total_fp += len(pred_masks)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    split_rate = total_splits / n_gt_total if n_gt_total else 0.0
    merge_rate = total_merges / n_pred_total if n_pred_total else 0.0

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "n_gt": n_gt_total,
        "n_pred": n_pred_total,
        "split_rate": split_rate,
        "merge_rate": merge_rate,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    val_items = load_val_items()
    print(f"Loaded {len(val_items)} validation images (Site 5737)", flush=True)

    ckpt_path = Path("DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth")
    model = TreeFlowNet(pretrained_backbone=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    configs = [
        ("Config A: Vanilla TreeFlowNet (Euler Sinks)", False, False),
        ("Config B: TreeFlowNet + Medial Axis Bridging", True, False),
        ("Config C: TreeFlowNet + Spectral Modularity Cut", False, True),
        ("Config D: Full Neuro-Flow-Graph (NFG)", True, True),
    ]

    results = {}
    print("\n================= RUNNING VALIDATION ABLATION (SITE 5737) =================")
    for name, use_bf, use_cut in configs:
        res = run_ablation_eval(model, device, val_items, use_bellman_ford=use_bf, use_spectral_cut=use_cut)
        results[name] = res
        print(f"[{name}] F1: {res['f1']:.4f} | P: {res['precision']:.4f} | R: {res['recall']:.4f} | TP: {res['tp']}/{res['n_gt']} | FP: {res['fp']} | SplitRate: {res['split_rate']:.4f} | MergeRate: {res['merge_rate']:.4f}", flush=True)

    out_path = Path("DeadTrees/experiments/neuro_flow_val_ablation.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nAblation results saved to: {out_path}")


if __name__ == "__main__":
    main()
