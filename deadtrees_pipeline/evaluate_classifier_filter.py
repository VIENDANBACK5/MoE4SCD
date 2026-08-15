"""Evaluate OOF proposal classifiers as end-to-end DeadTrees detectors."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from deadtrees_pipeline.evaluate_raw_sam2 import rasterize_instances
from deadtrees_pipeline.metrics import hungarian_match, overlap_matrices, structural_errors


DEFAULT_INSTANCES = Path("DeadTrees/instances_gt/instances.gpkg")
DEFAULT_IMAGE_ROOT = Path(
    "DeadTrees/raw/image-tiles-1024-global-aerial-sampled-20-random"
)
DEFAULT_SAM_DIR = Path("DeadTrees/sam2_masks")
DEFAULT_PREDICTIONS = Path(
    "DeadTrees/experiments/classification_v1/oof_predictions.csv"
)
DEFAULT_OUTPUT = Path(
    "DeadTrees/experiments/classification_v1/end_to_end_metrics.json"
)
IOU_THRESHOLDS = (0.25, 0.50, 0.75)


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_oof_filters(
    predictions_path: Path = DEFAULT_PREDICTIONS,
    instances_path: Path = DEFAULT_INSTANCES,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    sam_dir: Path = DEFAULT_SAM_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    min_gt_area: int = 20,
    min_pred_area: int = 50,
    structural_overlap: float = 0.10,
) -> dict:
    predictions = pd.read_csv(predictions_path)
    raw_experiments = sorted(
        predictions.loc[predictions["universe"] == "raw_sam2", "experiment"].unique()
    )
    selected_ids = {
        experiment: set(
            predictions.loc[
                (predictions["experiment"] == experiment)
                & (predictions["prediction"] == 1),
                "sample_id",
            ]
        )
        for experiment in raw_experiments
    }
    configurations = ["raw_unfiltered", *raw_experiments]

    instances = gpd.read_file(instances_path, layer="instances")
    grouped = {stem: group for stem, group in instances.groupby("stem", sort=True)}
    totals = {
        name: {
            "thresholds": {threshold: defaultdict(int) for threshold in IOU_THRESHOLDS},
            "proposal_covered": {threshold: 0 for threshold in IOU_THRESHOLDS},
            "structure": defaultdict(int),
            "best_ious": [],
        }
        for name in configurations
    }

    image_paths = sorted(image_root.glob("**/*.tif"))
    for image_path in tqdm(image_paths, desc="Evaluating OOF classifier filters"):
        stem = image_path.stem
        group = grouped.get(stem, instances.iloc[0:0])
        with rasterio.open(image_path) as src:
            gt_masks, _ = rasterize_instances(group, src)
        gt_areas = (
            gt_masks.reshape(len(gt_masks), -1).sum(axis=1)
            if len(gt_masks) else np.zeros(0, dtype=int)
        )
        keep_gt = gt_areas >= min_gt_area
        gt_masks, gt_areas = gt_masks[keep_gt], gt_areas[keep_gt]

        data = np.load(sam_dir / f"{stem}.npz")
        all_masks = data["masks"].astype(bool)
        all_areas = all_masks.reshape(len(all_masks), -1).sum(axis=1)
        original_indices = np.flatnonzero(all_areas >= min_pred_area)
        retained_masks = all_masks[original_indices]

        masks_by_configuration = {"raw_unfiltered": retained_masks}
        for experiment in raw_experiments:
            keep = np.asarray([
                f"sam2:{stem}:{int(index)}" in selected_ids[experiment]
                for index in original_indices
            ], dtype=bool)
            masks_by_configuration[experiment] = retained_masks[keep]

        for configuration, pred_masks in masks_by_configuration.items():
            record = totals[configuration]
            iou, intersections, _, _ = overlap_matrices(gt_masks, pred_masks)
            best_ious = (
                iou.max(axis=1)
                if len(gt_masks) and len(pred_masks)
                else np.zeros(len(gt_masks), dtype=float)
            )
            record["best_ious"].extend(best_ious.tolist())
            for threshold in IOU_THRESHOLDS:
                match = hungarian_match(iou, threshold)
                for metric in ("tp", "fp", "fn"):
                    record["thresholds"][threshold][metric] += getattr(match, metric)
                record["proposal_covered"][threshold] += int((best_ious >= threshold).sum())

            structure = structural_errors(intersections, gt_areas, structural_overlap)
            for key in (
                "split_gt_count",
                "merge_pred_count",
                "uncovered_gt_count",
                "unrelated_pred_count",
            ):
                record["structure"][key] += int(structure[key])
            record["structure"]["n_gt"] += len(gt_masks)
            record["structure"]["n_pred"] += len(pred_masks)

    result = {
        "protocol": {
            "prediction_source": "strict OOF probabilities; each site predicted by a model not trained on that site",
            "classifier_threshold": "selected by inner LOSO within each outer training fold",
            "same_object_universe": True,
            "n_images": len(image_paths),
            "min_gt_area_px": min_gt_area,
            "min_pred_area_px": min_pred_area,
        },
        "configurations": {},
    }
    for configuration, record in totals.items():
        n_gt = record["structure"]["n_gt"]
        n_pred = record["structure"]["n_pred"]
        result["configurations"][configuration] = {
            "n_gt": n_gt,
            "n_predictions_after_filter": n_pred,
            "detection": {
                f"iou_{threshold:.2f}": _prf(
                    record["thresholds"][threshold]["tp"],
                    record["thresholds"][threshold]["fp"],
                    record["thresholds"][threshold]["fn"],
                )
                for threshold in IOU_THRESHOLDS
            },
            "proposal_quality": {
                "mean_best_iou_per_gt": float(np.mean(record["best_ious"])),
                **{
                    f"recall_iou_{threshold:.2f}": (
                        record["proposal_covered"][threshold] / max(n_gt, 1)
                    )
                    for threshold in IOU_THRESHOLDS
                },
            },
            "structural": {
                "split_rate": record["structure"]["split_gt_count"] / max(n_gt, 1),
                "merge_rate": record["structure"]["merge_pred_count"] / max(n_pred, 1),
                "uncovered_gt_rate": record["structure"]["uncovered_gt_count"] / max(n_gt, 1),
                "unrelated_pred_rate": record["structure"]["unrelated_pred_count"] / max(n_pred, 1),
            },
        }

    baseline = result["configurations"]["raw_unfiltered"]["detection"]
    result["delta_f1_vs_raw"] = {
        experiment: {
            key: result["configurations"][experiment]["detection"][key]["f1"]
            - baseline[key]["f1"]
            for key in baseline
        }
        for experiment in raw_experiments
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--sam-dir", type=Path, default=DEFAULT_SAM_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_oof_filters(
        predictions_path=args.predictions,
        instances_path=args.instances,
        image_root=args.image_root,
        sam_dir=args.sam_dir,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
