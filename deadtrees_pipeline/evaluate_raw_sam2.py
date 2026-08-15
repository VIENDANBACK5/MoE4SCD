"""Evaluate raw SAM2 proposals against manual DeadTrees polygon instances."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from PIL import Image
from tqdm import tqdm

from deadtrees_pipeline.metrics import (
    average_precision,
    boundary_metrics,
    hungarian_match,
    overlap_matrices,
    structural_errors,
)
from deadtrees_pipeline.gt_instances import dataset_id_from_stem


DEFAULT_INSTANCES = Path("DeadTrees/instances_gt/instances.gpkg")
DEFAULT_IMAGE_ROOT = Path(
    "DeadTrees/raw/image-tiles-1024-global-aerial-sampled-20-random"
)
DEFAULT_SAM_DIR = Path("DeadTrees/sam2_masks")
DEFAULT_OUTPUT = Path("DeadTrees/experiments/raw_sam2_v1")
IOU_THRESHOLDS = (0.25, 0.50, 0.75)


def rasterize_instances(group: gpd.GeoDataFrame, src) -> tuple[np.ndarray, list[str]]:
    if group.empty:
        return np.zeros((0, src.height, src.width), dtype=bool), []
    native = group.to_crs(src.crs)
    masks, ids = [], []
    for _, row in native.iterrows():
        mask = rasterio.features.rasterize(
            [(row.geometry, 1)],
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype=np.uint8,
        ).astype(bool)
        masks.append(mask)
        ids.append(str(row["instance_id"]))
    return np.stack(masks) if masks else np.zeros((0, src.height, src.width), dtype=bool), ids


def _safe_mean(values: list[float]) -> float | None:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _micro_prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _summarize_gt_rows(rows: list[dict]) -> dict:
    """Summarize proposal failures for a homogeneous GT subset."""
    n_gt = len(rows)
    if not n_gt:
        return {
            "n_gt": 0,
            "mean_best_iou": 0.0,
            "recall_iou_0.25": 0.0,
            "recall_iou_0.50": 0.0,
            "recall_iou_0.75": 0.0,
            "split_rate": 0.0,
        }
    best = np.asarray([row["best_iou"] for row in rows], dtype=float)
    return {
        "n_gt": n_gt,
        "mean_best_iou": float(best.mean()),
        "recall_iou_0.25": float((best >= 0.25).mean()),
        "recall_iou_0.50": float((best >= 0.50).mean()),
        "recall_iou_0.75": float((best >= 0.75).mean()),
        "split_rate": sum(bool(row["is_split"]) for row in rows) / n_gt,
    }


def _image_to_uint8(src) -> np.ndarray:
    image = np.transpose(src.read([1, 2, 3]), (1, 2, 0)).astype(np.float32)
    out = np.zeros_like(image, dtype=np.uint8)
    for channel in range(3):
        band = image[..., channel]
        low, high = np.percentile(band, [2, 98])
        if high <= low:
            continue
        out[..., channel] = np.clip((band - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    return out


def save_overlay(image_path: Path, gt_masks: np.ndarray, pred_masks: np.ndarray, output_path: Path) -> None:
    from skimage.segmentation import find_boundaries

    with rasterio.open(image_path) as src:
        image = _image_to_uint8(src)
    gt_union = gt_masks.any(axis=0) if len(gt_masks) else np.zeros(image.shape[:2], dtype=bool)
    pred_union = pred_masks.any(axis=0) if len(pred_masks) else np.zeros(image.shape[:2], dtype=bool)
    gt_boundary = find_boundaries(gt_union, mode="inner")
    pred_boundary = find_boundaries(pred_union, mode="inner")

    gt_panel = image.copy()
    gt_panel[gt_boundary] = [255, 255, 0]
    pred_panel = image.copy()
    pred_panel[pred_boundary] = [255, 0, 0]
    combined = image.copy()
    combined[gt_boundary] = [255, 255, 0]
    combined[pred_boundary] = [255, 0, 0]
    combined[gt_boundary & pred_boundary] = [0, 255, 255]
    panel = np.concatenate([image, gt_panel, pred_panel, combined], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel).save(output_path)


def evaluate(
    instances_path: Path = DEFAULT_INSTANCES,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    sam_dir: Path = DEFAULT_SAM_DIR,
    output_dir: Path = DEFAULT_OUTPUT,
    min_gt_area: int = 20,
    min_pred_area: int = 50,
    boundary_tolerance_px: float = 3.0,
    structural_overlap: float = 0.10,
    n_visualizations: int = 10,
) -> dict:
    if not instances_path.exists():
        raise FileNotFoundError(f"Run gt_instances.py first: {instances_path}")
    instances = gpd.read_file(instances_path, layer="instances")
    grouped = {stem: group for stem, group in instances.groupby("stem", sort=True)}
    image_paths = sorted(image_root.glob("**/*.tif"))
    if not image_paths:
        raise FileNotFoundError(f"No GeoTIFF images found under {image_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_totals = {threshold: defaultdict(int) for threshold in IOU_THRESHOLDS}
    boundary_values = defaultdict(list)
    structural_totals = defaultdict(int)
    per_image: list[dict] = []
    per_gt: list[dict] = []
    ap_predictions: list[tuple[str, float, np.ndarray]] = []
    gt_counts: dict[str, int] = {}
    proposal_covered = {threshold: 0 for threshold in IOU_THRESHOLDS}
    best_ious_all: list[float] = []
    visualization_cache: dict[str, tuple[Path, np.ndarray, np.ndarray]] = {}

    for image_path in tqdm(image_paths, desc="Evaluating raw SAM2 proposals"):
        stem = image_path.stem
        group = grouped.get(stem, instances.iloc[0:0])
        sam_path = sam_dir / f"{stem}.npz"
        if not image_path.exists() or not sam_path.exists():
            continue

        with rasterio.open(image_path) as src:
            gt_masks, gt_ids = rasterize_instances(group, src)
        gt_areas = gt_masks.reshape(len(gt_masks), -1).sum(axis=1) if len(gt_masks) else np.zeros(0)
        keep_gt = gt_areas >= min_gt_area
        gt_masks = gt_masks[keep_gt]
        gt_areas = gt_areas[keep_gt]
        gt_ids = [instance_id for instance_id, keep in zip(gt_ids, keep_gt) if keep]

        sam_data = np.load(sam_path)
        pred_masks = sam_data["masks"].astype(bool)
        scores = sam_data["scores"].astype(float)
        pred_areas = pred_masks.reshape(len(pred_masks), -1).sum(axis=1)
        keep = pred_areas >= min_pred_area
        pred_masks, scores, pred_areas = pred_masks[keep], scores[keep], pred_areas[keep]

        iou, intersections, _, _ = overlap_matrices(gt_masks, pred_masks)
        gt_counts[stem] = len(gt_masks)
        best_ious = (
            iou.max(axis=1) if len(gt_masks) and len(pred_masks)
            else np.zeros(len(gt_masks), dtype=float)
        )
        best_ious_all.extend(best_ious.tolist())
        for threshold in IOU_THRESHOLDS:
            proposal_covered[threshold] += int((best_ious >= threshold).sum())
        for pred_index, score in enumerate(scores):
            ap_predictions.append((stem, float(score), iou[:, pred_index].copy()))

        image_row = {
            "stem": stem,
            "dataset_id": dataset_id_from_stem(stem),
            "n_gt": len(gt_masks),
            "n_pred": len(pred_masks),
            "mean_best_iou_per_gt": float(best_ious.mean()) if len(best_ious) else 0.0,
        }

        for threshold in IOU_THRESHOLDS:
            match = hungarian_match(iou, threshold)
            key = str(threshold).replace(".", "_")
            image_row[f"tp_{key}"] = match.tp
            image_row[f"fp_{key}"] = match.fp
            image_row[f"fn_{key}"] = match.fn
            image_row[f"f1_{key}"] = match.f1
            image_row[f"recall_{key}"] = match.recall
            image_row[f"proposal_recall_{key}"] = (
                float((best_ious >= threshold).mean()) if len(best_ious) else 0.0
            )
            for metric in ("tp", "fp", "fn"):
                threshold_totals[threshold][metric] += getattr(match, metric)

            if threshold == 0.50:
                for gt_index, pred_index, matched_iou in match.pairs:
                    boundary = boundary_metrics(
                        gt_masks[gt_index], pred_masks[pred_index], boundary_tolerance_px
                    )
                    boundary_values["matched_iou"].append(matched_iou)
                    for name, value in boundary.items():
                        boundary_values[name].append(value)

        structure = structural_errors(intersections, gt_areas, structural_overlap)
        significant_relations = intersections >= (gt_areas[:, None] * structural_overlap)
        significant_per_gt = (
            significant_relations.sum(axis=1)
            if significant_relations.size
            else np.zeros(len(gt_masks), dtype=int)
        )
        for gt_index, (instance_id, area, best_iou) in enumerate(
            zip(gt_ids, gt_areas, best_ious)
        ):
            n_significant = int(significant_per_gt[gt_index])
            if best_iou >= 0.50:
                failure_type = "good_iou_ge_0.50"
            elif best_iou >= 0.25:
                failure_type = "partial_iou_0.25_0.50"
            elif n_significant:
                failure_type = "weak_overlap_iou_lt_0.25"
            else:
                failure_type = "missed_no_significant_overlap"
            per_gt.append({
                "instance_id": instance_id,
                "stem": stem,
                "dataset_id": dataset_id_from_stem(stem),
                "raster_area_px": int(area),
                "best_iou": float(best_iou),
                "n_significant_proposals": n_significant,
                "is_split": n_significant >= 2,
                "failure_type": failure_type,
            })
        image_row.update(structure)
        for name in ("split_gt_count", "merge_pred_count", "uncovered_gt_count", "unrelated_pred_count"):
            structural_totals[name] += int(structure[name])
        structural_totals["n_gt"] += len(gt_masks)
        structural_totals["n_pred"] += len(pred_masks)
        per_image.append(image_row)
        visualization_cache[stem] = (image_path, gt_masks, pred_masks)

    if not per_image:
        raise RuntimeError("No image pairs were evaluated")
    if not per_gt:
        raise RuntimeError(
            "No GT instance survived rasterization and min_gt_area filtering"
        )

    detection = {
        f"iou_{threshold:.2f}": _micro_prf(
            threshold_totals[threshold]["tp"],
            threshold_totals[threshold]["fp"],
            threshold_totals[threshold]["fn"],
        )
        for threshold in IOU_THRESHOLDS
    }
    ap_thresholds = np.arange(0.50, 0.951, 0.05)
    ap_values = [average_precision(ap_predictions, gt_counts, float(t)) for t in ap_thresholds]

    # Global quartiles make the size diagnosis reproducible without hand-picked bins.
    area_quartiles = np.quantile(
        [row["raster_area_px"] for row in per_gt], [0.25, 0.50, 0.75]
    )
    size_labels = ("q1_small", "q2", "q3", "q4_large")
    for row in per_gt:
        row["size_bin"] = size_labels[
            int(np.searchsorted(area_quartiles, row["raster_area_px"], side="left"))
        ]

    failure_counts = {
        name: sum(row["failure_type"] == name for row in per_gt)
        for name in (
            "good_iou_ge_0.50",
            "partial_iou_0.25_0.50",
            "weak_overlap_iou_lt_0.25",
            "missed_no_significant_overlap",
        )
    }
    by_site = {}
    for dataset_id in sorted({row["dataset_id"] for row in per_gt}):
        site_rows = [row for row in per_gt if row["dataset_id"] == dataset_id]
        site_images = [row for row in per_image if row["dataset_id"] == dataset_id]
        site_summary = _summarize_gt_rows(site_rows)
        site_summary.update({
            "n_images": len(site_images),
            "n_predictions": sum(row["n_pred"] for row in site_images),
            "detection_iou_0.50": _micro_prf(
                sum(row["tp_0_5"] for row in site_images),
                sum(row["fp_0_5"] for row in site_images),
                sum(row["fn_0_5"] for row in site_images),
            ),
        })
        by_site[str(dataset_id)] = site_summary

    by_size = {
        label: _summarize_gt_rows([row for row in per_gt if row["size_bin"] == label])
        for label in size_labels
    }
    summary = {
        "protocol": "Raw SAM2 proposals; all retained proposals are evaluated as class-agnostic object predictions.",
        "n_images": len(per_image),
        "n_gt_instances": structural_totals["n_gt"],
        "n_predictions": structural_totals["n_pred"],
        "min_gt_area_px": min_gt_area,
        "min_pred_area_px": min_pred_area,
        "boundary_tolerance_px": boundary_tolerance_px,
        "structural_overlap_fraction_of_gt": structural_overlap,
        "detection": detection,
        "proposal_quality": {
            "mean_best_iou_per_gt": float(np.mean(best_ious_all)) if best_ious_all else 0.0,
            "proposal_recall_iou_0.25": proposal_covered[0.25] / max(structural_totals["n_gt"], 1),
            "proposal_recall_iou_0.50": proposal_covered[0.50] / max(structural_totals["n_gt"], 1),
            "proposal_recall_iou_0.75": proposal_covered[0.75] / max(structural_totals["n_gt"], 1),
        },
        "matched_boundary_at_iou_0.50": {
            name: _safe_mean(values) for name, values in boundary_values.items()
        },
        "structural": {
            "split_gt_count": structural_totals["split_gt_count"],
            "split_rate": structural_totals["split_gt_count"] / max(structural_totals["n_gt"], 1),
            "merge_pred_count": structural_totals["merge_pred_count"],
            "merge_rate": structural_totals["merge_pred_count"] / max(structural_totals["n_pred"], 1),
            "uncovered_gt_count": structural_totals["uncovered_gt_count"],
            "uncovered_gt_rate": structural_totals["uncovered_gt_count"] / max(structural_totals["n_gt"], 1),
            "unrelated_pred_count": structural_totals["unrelated_pred_count"],
            "unrelated_pred_rate": structural_totals["unrelated_pred_count"] / max(structural_totals["n_pred"], 1),
        },
        "failure_taxonomy": {
            "definitions": {
                "significant_overlap": f"intersection / GT area >= {structural_overlap}",
                "good_iou_ge_0.50": "at least one raw proposal has IoU >= 0.50",
                "partial_iou_0.25_0.50": "best proposal IoU is in [0.25, 0.50)",
                "weak_overlap_iou_lt_0.25": "a significant proposal exists but best IoU < 0.25",
                "missed_no_significant_overlap": "no proposal covers a significant part of the GT",
            },
            "counts": failure_counts,
            "rates": {
                name: count / max(len(per_gt), 1)
                for name, count in failure_counts.items()
            },
            "raster_area_quartile_edges_px": area_quartiles.tolist(),
            "by_size_quartile": by_size,
            "by_site": by_site,
        },
        "confidence_metrics": {
            "score_source": "SAM2 stability score",
            "ap_0.50": ap_values[0],
            "ap_0.50_0.95": float(np.mean(ap_values)),
        },
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(output_dir / "per_image.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)
    with open(output_dir / "per_gt.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_gt[0].keys()))
        writer.writeheader()
        writer.writerows(per_gt)

    if n_visualizations > 0:
        ranked = sorted(per_image, key=lambda row: row["mean_best_iou_per_gt"])
        n_worst = n_visualizations // 2
        selected = ranked[:n_worst] + ranked[-(n_visualizations - n_worst):]
        for rank, row in enumerate(selected):
            stem = row["stem"]
            image_path, gt_masks, pred_masks = visualization_cache[stem]
            label = "worst" if rank < n_worst else "best"
            save_overlay(
                image_path,
                gt_masks,
                pred_masks,
                output_dir / "overlays" / f"{label}_{stem}.png",
            )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate raw SAM2 on DeadTrees polygons")
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--sam-dir", type=Path, default=DEFAULT_SAM_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-gt-area", type=int, default=20)
    parser.add_argument("--min-pred-area", type=int, default=50)
    parser.add_argument("--boundary-tolerance", type=float, default=3.0)
    parser.add_argument("--structural-overlap", type=float, default=0.10)
    parser.add_argument("--visualizations", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(
        instances_path=args.instances,
        image_root=args.image_root,
        sam_dir=args.sam_dir,
        output_dir=args.output,
        min_gt_area=args.min_gt_area,
        min_pred_area=args.min_pred_area,
        boundary_tolerance_px=args.boundary_tolerance,
        structural_overlap=args.structural_overlap,
        n_visualizations=args.visualizations,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
