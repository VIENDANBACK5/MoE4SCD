"""Create per-site comparison tables and fixed representative failure overlays."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw
from skimage.segmentation import find_boundaries

from deadtrees_pipeline.evaluate_raw_sam2 import _image_to_uint8, rasterize_instances


CONFIGURATIONS = {
    "baseline": (
        Path("DeadTrees/sam2_masks"),
        Path("DeadTrees/experiments/raw_sam2_v1"),
    ),
    "high_recall": (
        Path("DeadTrees/sam2_masks_high_recall_v1"),
        Path("DeadTrees/experiments/high_recall_v1"),
    ),
    "suppressed": (
        Path("DeadTrees/sam2_masks_improved_v1"),
        Path("DeadTrees/experiments/improved_v1"),
    ),
}
DEFAULT_INSTANCES = Path("DeadTrees/instances_gt/instances.gpkg")
DEFAULT_OUTPUT = Path("DeadTrees/experiments/segmentation_comparison_v1")


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def build_per_site_table(output_dir: Path) -> pd.DataFrame:
    rows = []
    for configuration, (_, experiment_dir) in CONFIGURATIONS.items():
        per_image = pd.read_csv(experiment_dir / "per_image.csv")
        for site, group in per_image.groupby("dataset_id", sort=True):
            n_gt = int(group["n_gt"].sum())
            n_pred = int(group["n_pred"].sum())
            tp = int(group["tp_0_5"].sum())
            fp = int(group["fp_0_5"].sum())
            fn = int(group["fn_0_5"].sum())
            precision, recall, f1 = _prf(tp, fp, fn)
            rows.append({
                "configuration": configuration,
                "dataset_id": int(site),
                "n_images": len(group),
                "n_gt": n_gt,
                "n_predictions": n_pred,
                "mean_best_iou_per_gt": float(
                    np.average(group["mean_best_iou_per_gt"], weights=np.maximum(group["n_gt"], 1))
                ),
                "precision_iou_0.50": precision,
                "recall_iou_0.50": recall,
                "f1_iou_0.50": f1,
                "miss_rate": int(group["uncovered_gt_count"].sum()) / max(n_gt, 1),
                "split_rate": int(group["split_gt_count"].sum()) / max(n_gt, 1),
                "merge_rate": int(group["merge_pred_count"].sum()) / max(n_pred, 1),
                "unrelated_prediction_rate": int(group["unrelated_pred_count"].sum()) / max(n_pred, 1),
            })
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "per_site_metrics.csv", index=False)
    return frame


def select_failure_cases() -> pd.DataFrame:
    high = pd.read_csv(CONFIGURATIONS["high_recall"][1] / "per_image.csv")
    positive = high[high["n_gt"] > 0].copy()
    positive["miss_rate_local"] = positive["uncovered_gt_count"] / positive["n_gt"]
    split_cases = positive.nlargest(5, ["split_gt_count", "n_gt"]).copy()
    remaining = positive[~positive["stem"].isin(split_cases["stem"]) & (positive["n_gt"] >= 5)]
    miss_cases = remaining.nlargest(5, ["miss_rate_local", "n_gt"]).copy()
    selected = pd.concat([split_cases.assign(case_type="split"), miss_cases.assign(case_type="miss")])
    return selected[["stem", "dataset_id", "case_type", "n_gt", "n_pred", "split_gt_count", "uncovered_gt_count", "mean_best_iou_per_gt"]]


def _load_retained_masks(mask_dir: Path, stem: str, min_area: int = 50) -> np.ndarray:
    with np.load(mask_dir / f"{stem}.npz") as data:
        masks = data["masks"].astype(bool)
    areas = masks.reshape(len(masks), -1).sum(axis=1) if len(masks) else np.zeros(0)
    return masks[areas >= min_area]


def _boundary_panel(image: np.ndarray, masks: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    union = masks.any(axis=0) if len(masks) else np.zeros(image.shape[:2], dtype=bool)
    boundary = find_boundaries(union, mode="inner")
    panel = image.copy()
    panel[boundary] = color
    return panel


def _add_label(panel: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(panel)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(260, panel.shape[1]), 28), fill=(0, 0, 0))
    draw.text((7, 7), label, fill=(255, 255, 255))
    return np.asarray(image)


def create_failure_overlays(
    cases: pd.DataFrame,
    instances_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    instances = gpd.read_file(instances_path, layer="instances")
    grouped = {stem: group for stem, group in instances.groupby("stem", sort=True)}
    high_rows = pd.read_csv(CONFIGURATIONS["high_recall"][1] / "per_image.csv").set_index("stem")
    suppressed_rows = pd.read_csv(CONFIGURATIONS["suppressed"][1] / "per_image.csv").set_index("stem")
    baseline_rows = pd.read_csv(CONFIGURATIONS["baseline"][1] / "per_image.csv").set_index("stem")
    image_lookup = {
        path.stem: path
        for path in Path("DeadTrees/raw/image-tiles-1024-global-aerial-sampled-20-random").glob("**/*.tif")
    }
    manifest_rows = []

    for _, case in cases.iterrows():
        stem = case["stem"]
        image_path = image_lookup[stem]
        with rasterio.open(image_path) as src:
            image = _image_to_uint8(src)
            gt_masks, _ = rasterize_instances(grouped.get(stem, instances.iloc[0:0]), src)
        gt_areas = gt_masks.reshape(len(gt_masks), -1).sum(axis=1) if len(gt_masks) else np.zeros(0)
        gt_masks = gt_masks[gt_areas >= 20]
        baseline_masks = _load_retained_masks(CONFIGURATIONS["baseline"][0], stem)
        high_masks = _load_retained_masks(CONFIGURATIONS["high_recall"][0], stem)
        suppressed_masks = _load_retained_masks(CONFIGURATIONS["suppressed"][0], stem)

        panels = [
            _add_label(image, "RGB"),
            _add_label(_boundary_panel(image, gt_masks, (255, 255, 0)), f"GT ({len(gt_masks)})"),
            _add_label(_boundary_panel(image, baseline_masks, (255, 0, 0)), f"Baseline ({len(baseline_masks)})"),
            _add_label(_boundary_panel(image, high_masks, (255, 0, 0)), f"High recall ({len(high_masks)})"),
            _add_label(_boundary_panel(image, suppressed_masks, (255, 0, 0)), f"Suppressed ({len(suppressed_masks)})"),
        ]
        output_path = output_dir / "overlays" / f"{case['case_type']}_{stem}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.concatenate(panels, axis=1)).save(output_path)

        before_split = int(high_rows.loc[stem, "split_gt_count"])
        after_split = int(suppressed_rows.loc[stem, "split_gt_count"])
        diagnosis = (
            "duplicate_or_nested_component_present_but_fragments_remain"
            if before_split > after_split and after_split > 0
            else "mostly_duplicate_or_nested" if before_split > after_split
            else "fragment_or_nonduplicate_overlap"
        )
        for configuration, table in (
            ("baseline", baseline_rows),
            ("high_recall", high_rows),
            ("suppressed", suppressed_rows),
        ):
            row = table.loc[stem]
            manifest_rows.append({
                "stem": stem,
                "dataset_id": int(case["dataset_id"]),
                "case_type": case["case_type"],
                "diagnosis": diagnosis,
                "configuration": configuration,
                "n_gt": int(row["n_gt"]),
                "n_predictions": int(row["n_pred"]),
                "mean_best_iou_per_gt": float(row["mean_best_iou_per_gt"]),
                "f1_iou_0.50": float(row["f1_0_5"]),
                "split_gt_count": int(row["split_gt_count"]),
                "uncovered_gt_count": int(row["uncovered_gt_count"]),
                "overlay": str(output_path),
            })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "failure_cases.csv", index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    per_site = build_per_site_table(args.output)
    cases = select_failure_cases()
    failure_rows = create_failure_overlays(cases, args.instances, args.output)
    overall = {
        name: json.loads((experiment_dir / "summary.json").read_text())
        for name, (_, experiment_dir) in CONFIGURATIONS.items()
    }
    summary = {
        "configurations": list(CONFIGURATIONS),
        "n_per_site_rows": len(per_site),
        "n_failure_cases": len(cases),
        "n_failure_rows": len(failure_rows),
        "overall_summary_paths": {
            name: str(experiment_dir / "summary.json")
            for name, (_, experiment_dir) in CONFIGURATIONS.items()
        },
        "overall_key_metrics": {
            name: {
                "n_predictions": data["n_predictions"],
                "f1_iou_0.50": data["detection"]["iou_0.50"]["f1"],
                "recall_iou_0.50": data["detection"]["iou_0.50"]["recall"],
                "mean_best_iou_per_gt": data["proposal_quality"]["mean_best_iou_per_gt"],
                "boundary_f1": data["matched_boundary_at_iou_0.50"]["boundary_f1"],
                "miss_rate": data["structural"]["uncovered_gt_rate"],
                "split_rate": data["structural"]["split_rate"],
                "merge_rate": data["structural"]["merge_rate"],
            }
            for name, data in overall.items()
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
