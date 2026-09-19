"""Comprehensive Evaluation for Unified StarConvex baseline across validation site 5737.
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
import torch
from shapely.geometry import Polygon, MultiPolygon

from crown_segmentation_research.methods.star_convex.decode import decode as star_convex_decode
from crown_segmentation_research.methods.star_convex.model import StarConvexNet
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


def eval_star_convex_checkpoint(
    model: StarConvexNet,
    device: torch.device,
    val_items: list,
    prob_threshold: float = 0.5,
    min_peak_distance: int = 5,
    nms_iou_threshold: float = 0.2,
    canopy_threshold: float = 0.7,
    embedding_delta_d: float = 3.0,
) -> dict:
    total_tp = total_fp = total_fn = 0
    per_image = []

    with torch.no_grad():
        for stem, image, gt_masks in val_items:
            image_t = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device)
            out = model(image_t)
            prob = out["probability"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            rays = out["rays"].squeeze(0).cpu().numpy().astype(np.float32)
            canopy = out["canopy"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32) if "canopy" in out else None
            embedding = out["embedding"].squeeze(0).cpu().numpy().astype(np.float32) if "embedding" in out else None

            polygons = star_convex_decode(
                prob, rays, n_rays=16,
                prob_threshold=prob_threshold,
                min_peak_distance=min_peak_distance,
                nms_iou_threshold=nms_iou_threshold,
                canopy=canopy,
                canopy_threshold=canopy_threshold,
                embedding=embedding,
                embedding_delta_d=embedding_delta_d,
            )
            polygons = [p for p in polygons if p.is_valid and not p.is_empty]

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
    parser.add_argument("--ckpt-dir", type=Path, default=Path("DeadTrees/experiments/star_convex_unified_v1/epoch_checkpoints"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-json", type=Path, default=Path("DeadTrees/experiments/star_convex_eval_results.json"))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    val_items = load_val_items()
    print(f"Loaded {len(val_items)} validation images (Site 5737)", flush=True)

    results = []
    ckpts = sorted(args.ckpt_dir.glob("epoch_*.pth"))
    for ckpt_path in ckpts:
        epoch = int(ckpt_path.stem.split("_")[1])
        model = StarConvexNet(
            n_rays=16,
            pretrained_backbone=False,
            use_canopy_head=True,
            use_embedding_head=True,
            embedding_dim=8,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        metrics = eval_star_convex_checkpoint(model, device, val_items)
        results.append({"epoch": epoch, "ckpt": str(ckpt_path), **metrics})
        print(
            f"Epoch {epoch:03d}: F1={metrics['f1']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} (TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']})",
            flush=True,
        )

    if results:
        best = max(results, key=lambda r: r["f1"])
        print(f"\n=== BEST StarConvex Epoch {best['epoch']}: F1={best['f1']:.4f} | Precision={best['precision']:.4f} | Recall={best['recall']:.4f} ===", flush=True)
        args.out_json.write_text(json.dumps(results, indent=2))
        print(f"Saved evaluation results to {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
