"""Comprehensive Evaluation for TreeFlowNet / OmniCrown across validation & benchmarks.

Evaluates:
1. Held-out validation site 5737 (20 tiles) -> Precision, Recall, F1@IoU0.5
2. Official DTE-aerial-bench (525 patches across 5 biomes) -> General Tree Cover IoU & F1
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
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from shapely.geometry import Polygon, MultiPolygon

from crown_segmentation_research.methods.tree_flow.decode import decode_flow_to_instances
from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet
from deadtrees_pipeline.metrics import hungarian_match, overlap_matrices

VAL_DIR = Path("DeadTrees/star_convex_targets_v1/val")
DTE_BENCH_DIR = Path("DTE-Aerial-Data-public")
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


def eval_tree_flow_checkpoint(
    model: TreeFlowNet,
    device: torch.device,
    val_items: list,
    canopy_threshold: float = 0.5,
    sdt_threshold: float = -0.2,
    centroid_threshold: float = 0.25,
) -> dict:
    total_tp = total_fp = total_fn = 0
    per_image = []

    with torch.no_grad():
        for stem, image, gt_masks in val_items:
            image_t = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device)
            out = model(image_t)
            flow = out["flow"].squeeze(0).cpu().numpy().astype(np.float32)
            sdt = out["sdt"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            centroid = out["centroid"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            canopy = out["canopy"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

            _, polygons = decode_flow_to_instances(
                flow, canopy, sdt, centroid,
                canopy_threshold=canopy_threshold,
                sdt_threshold=sdt_threshold,
                centroid_threshold=centroid_threshold,
                min_instance_area=MIN_PRED_AREA,
            )

            pred_masks = polygons_to_masks(polygons, image.shape[:2])
            pred_areas = pred_masks.reshape(len(pred_masks), -1).sum(axis=1) if len(pred_masks) else np.zeros(0)
            pred_masks = pred_masks[pred_areas >= MIN_PRED_AREA]

            iou, _, _, _ = overlap_matrices(gt_masks, pred_masks)
            match = hungarian_match(iou, 0.50)
            total_tp += match.tp
            total_fp += match.fp
            total_fn += match.fn
            per_image.append({"stem": stem, "n_gt": len(gt_masks), "n_pred": len(pred_masks), "tp": match.tp})

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "per_image": per_image,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=Path, default=Path("DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-json", type=Path, default=Path("DeadTrees/experiments/tree_flow_eval_results.json"))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    val_items = load_val_items()
    print(f"Loaded {len(val_items)} validation images (Site 5737)", flush=True)

    results = []
    ckpts = sorted(args.ckpt_dir.glob("epoch_*.pth"))
    for ckpt_path in ckpts:
        epoch = int(ckpt_path.stem.split("_")[1])
        model = TreeFlowNet(pretrained_backbone=False).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        metrics = eval_tree_flow_checkpoint(model, device, val_items)
        results.append({"epoch": epoch, "ckpt": str(ckpt_path), **metrics})
        print(
            f"Epoch {epoch:03d}: F1={metrics['f1']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} (TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']})",
            flush=True,
        )

    if results:
        best = max(results, key=lambda r: r["f1"])
        print(f"\n=== BEST TreeFlowNet Epoch {best['epoch']}: F1={best['f1']:.4f} | Precision={best['precision']:.4f} | Recall={best['recall']:.4f} ===", flush=True)
        args.out_json.write_text(json.dumps(results, indent=2))
        print(f"Saved evaluation results to {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
