"""Build auditable object features and run site-held-out DeadTrees ablations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from skimage.measure import perimeter, regionprops
from tqdm import tqdm

from deadtrees_pipeline.evaluate_raw_sam2 import rasterize_instances
from deadtrees_pipeline.gt_instances import dataset_id_from_stem
from deadtrees_pipeline.metrics import hungarian_match, overlap_matrices


DEFAULT_INSTANCES = Path("DeadTrees/instances_gt/instances.gpkg")
DEFAULT_IMAGE_ROOT = Path(
    "DeadTrees/raw/image-tiles-1024-global-aerial-sampled-20-random"
)
DEFAULT_SAM_DIR = Path("DeadTrees/sam2_masks")
DEFAULT_OUTPUT = Path("DeadTrees/experiments/classification_v1")
POSITIVE_IOU = 0.25
CLEAN_NEGATIVE_OVERLAP = 0.10

SHAPE_FEATURES = (
    "shape_log_area",
    "shape_log_perimeter",
    "shape_log_aspect_ratio",
    "shape_extent",
    "shape_compactness",
    "shape_solidity",
    "shape_eccentricity",
    "shape_log_major_axis",
    "shape_minor_major_ratio",
    "shape_log_components",
    "shape_hole_fraction",
    "shape_border_fraction",
)
RGB_FEATURES = tuple(
    f"rgb_{stat}_{channel}"
    for stat in ("mean", "std", "p10", "p50", "p90")
    for channel in ("r", "g", "b")
)


def extract_shape_features(mask: np.ndarray) -> dict[str, float]:
    area = int(mask.sum())
    if area == 0:
        return {name: 0.0 for name in SHAPE_FEATURES}
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = mask[y0:y1, x0:x1]
    height, width = crop.shape
    region = regionprops(crop.astype(np.uint8))[0]
    boundary_length = float(perimeter(crop, neighborhood=8))
    filled_area = int(ndimage.binary_fill_holes(crop).sum())
    _, n_components = ndimage.label(crop)
    border_pixels = int(mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum())
    major = float(region.axis_major_length)
    minor = float(region.axis_minor_length)
    return {
        "shape_log_area": float(np.log1p(area)),
        "shape_log_perimeter": float(np.log1p(boundary_length)),
        "shape_log_aspect_ratio": float(np.log((height + 1e-6) / (width + 1e-6))),
        "shape_extent": area / max(height * width, 1),
        "shape_compactness": float(4 * np.pi * area / max(boundary_length**2, 1.0)),
        "shape_solidity": float(region.solidity),
        "shape_eccentricity": float(region.eccentricity),
        "shape_log_major_axis": float(np.log1p(major)),
        "shape_minor_major_ratio": minor / max(major, 1e-6),
        "shape_log_components": float(np.log1p(n_components)),
        "shape_hole_fraction": (filled_area - area) / max(filled_area, 1),
        "shape_border_fraction": border_pixels / max(boundary_length, 1.0),
    }


def extract_rgb_features(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    pixels = image[mask].astype(np.float32) / 255.0
    if len(pixels) == 0:
        return {name: 0.0 for name in RGB_FEATURES}
    values = {
        "mean": pixels.mean(axis=0),
        "std": pixels.std(axis=0),
        "p10": np.percentile(pixels, 10, axis=0),
        "p50": np.percentile(pixels, 50, axis=0),
        "p90": np.percentile(pixels, 90, axis=0),
    }
    return {
        f"rgb_{stat}_{channel}": float(vector[index])
        for stat, vector in values.items()
        for index, channel in enumerate(("r", "g", "b"))
    }


def _feature_row(mask: np.ndarray, image: np.ndarray) -> dict[str, float]:
    return {**extract_shape_features(mask), **extract_rgb_features(image, mask)}


def build_object_table(
    instances_path: Path = DEFAULT_INSTANCES,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    sam_dir: Path = DEFAULT_SAM_DIR,
    output_csv: Path | None = None,
    min_gt_area: int = 20,
    min_pred_area: int = 50,
) -> pd.DataFrame:
    instances = gpd.read_file(instances_path, layer="instances")
    grouped = {stem: group for stem, group in instances.groupby("stem", sort=True)}
    image_paths = sorted(image_root.glob("**/*.tif"))
    rows: list[dict] = []

    for image_path in tqdm(image_paths, desc="Building object feature table"):
        stem = image_path.stem
        dataset_id = dataset_id_from_stem(stem)
        sam_path = sam_dir / f"{stem}.npz"
        if not sam_path.exists():
            raise FileNotFoundError(f"Missing SAM2 proposals: {sam_path}")
        group = grouped.get(stem, instances.iloc[0:0])

        with rasterio.open(image_path) as src:
            image = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
            if not np.isfinite(image).all():
                raise ValueError(f"RGB GeoTIFF contains NaN/Inf: {image_path}")
            # DeadTrees mixes uint8 and float32 containers, but both encode
            # the same documented 0..255 RGB scale. Preserve the values and
            # normalize only inside extract_rgb_features.
            if float(image.min()) < 0 or float(image.max()) > 255:
                raise ValueError(
                    f"Expected RGB values in [0, 255], got "
                    f"[{float(image.min())}, {float(image.max())}]: {image_path}"
                )
            gt_masks, gt_ids = rasterize_instances(group, src)

        gt_areas = (
            gt_masks.reshape(len(gt_masks), -1).sum(axis=1)
            if len(gt_masks) else np.zeros(0, dtype=int)
        )
        keep_gt = gt_areas >= min_gt_area
        gt_masks = gt_masks[keep_gt]
        gt_areas = gt_areas[keep_gt]
        gt_ids = [instance_id for instance_id, keep in zip(gt_ids, keep_gt) if keep]

        data = np.load(sam_path)
        pred_masks = data["masks"].astype(bool)
        pred_scores = data["scores"].astype(float)
        pred_areas = pred_masks.reshape(len(pred_masks), -1).sum(axis=1)
        keep_pred = pred_areas >= min_pred_area
        original_pred_indices = np.flatnonzero(keep_pred)
        pred_masks = pred_masks[keep_pred]
        pred_scores = pred_scores[keep_pred]
        pred_areas = pred_areas[keep_pred]

        iou, _, overlap_gt, overlap_pred = overlap_matrices(gt_masks, pred_masks)
        matched = hungarian_match(iou, POSITIVE_IOU)
        positive_by_pred = {
            pred_index: (gt_index, matched_iou)
            for gt_index, pred_index, matched_iou in matched.pairs
        }

        for pred_index, (source_index, mask, score, area) in enumerate(
            zip(original_pred_indices, pred_masks, pred_scores, pred_areas)
        ):
            best_iou = float(iou[:, pred_index].max()) if len(gt_masks) else 0.0
            max_overlap_gt = float(overlap_gt[:, pred_index].max()) if len(gt_masks) else 0.0
            max_overlap_pred = float(overlap_pred[:, pred_index].max()) if len(gt_masks) else 0.0
            if pred_index in positive_by_pred:
                gt_index, matched_iou = positive_by_pred[pred_index]
                label, label_name = 1, "positive"
                matched_gt_id = gt_ids[gt_index]
            elif max(max_overlap_gt, max_overlap_pred) < CLEAN_NEGATIVE_OVERLAP:
                label, label_name = 0, "clean_negative"
                matched_gt_id, matched_iou = "", 0.0
            else:
                label, label_name = None, "ambiguous"
                matched_gt_id, matched_iou = "", 0.0
            rows.append({
                "sample_id": f"sam2:{stem}:{int(source_index)}",
                "source": "sam2",
                "stem": stem,
                "dataset_id": dataset_id,
                "object_index": int(source_index),
                "label": label,
                "label_name": label_name,
                "area_px": int(area),
                "sam_stability_score": float(score),
                "best_iou": best_iou,
                "matched_iou": float(matched_iou),
                "matched_gt_instance_id": matched_gt_id,
                "max_overlap_gt": max_overlap_gt,
                "max_overlap_pred": max_overlap_pred,
                **_feature_row(mask, image),
            })

        for gt_index, (instance_id, mask, area) in enumerate(zip(gt_ids, gt_masks, gt_areas)):
            rows.append({
                "sample_id": f"gt:{instance_id}",
                "source": "gt",
                "stem": stem,
                "dataset_id": dataset_id,
                "object_index": gt_index,
                "label": 1,
                "label_name": "positive",
                "area_px": int(area),
                "sam_stability_score": "",
                "best_iou": 1.0,
                "matched_iou": 1.0,
                "matched_gt_instance_id": instance_id,
                "max_overlap_gt": 1.0,
                "max_overlap_pred": 1.0,
                **_feature_row(mask, image),
            })

    frame = pd.DataFrame(rows)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_csv, index=False)
    return frame


def _classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float | str,
    predictions: np.ndarray | None = None,
) -> dict:
    if predictions is None:
        if not isinstance(threshold, float):
            raise TypeError("A numeric threshold is required when predictions are omitted")
        predictions = (probabilities >= threshold).astype(int)
    result = {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((y_true == 0).sum()),
        "threshold": threshold,
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }
    result["roc_auc"] = (
        float(roc_auc_score(y_true, probabilities))
        if len(np.unique(y_true)) == 2 else None
    )
    return result


def _new_classifier(n_estimators: int, random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2,
    )


def _select_threshold_inner_loso(
    train: pd.DataFrame,
    feature_names: tuple[str, ...],
    n_estimators: int,
    random_state: int,
) -> tuple[float, dict]:
    """Select an operating threshold without seeing the outer held-out site."""
    inner_probabilities = np.zeros(len(train), dtype=float)
    inner_sites = sorted(int(site) for site in train["dataset_id"].unique())
    for inner_site in inner_sites:
        inner_train = train[train["dataset_id"] != inner_site]
        inner_valid_positions = np.flatnonzero(
            train["dataset_id"].to_numpy() == inner_site
        )
        inner_valid = train.iloc[inner_valid_positions]
        y_inner_train = inner_train["label"].astype(int).to_numpy()
        if len(np.unique(y_inner_train)) != 2:
            raise RuntimeError(f"Inner training split lacks a class for site {inner_site}")
        model = _new_classifier(n_estimators, random_state)
        model.fit(
            inner_train[list(feature_names)].to_numpy(float),
            y_inner_train,
        )
        inner_probabilities[inner_valid_positions] = model.predict_proba(
            inner_valid[list(feature_names)].to_numpy(float)
        )[:, 1]

    y_train = train["label"].astype(int).to_numpy()
    candidates = np.linspace(0.05, 0.95, 91)
    scored = []
    for threshold in candidates:
        predictions = (inner_probabilities >= threshold).astype(int)
        scored.append((
            float(f1_score(y_train, predictions, zero_division=0)),
            float(precision_score(y_train, predictions, zero_division=0)),
            -abs(float(threshold) - 0.5),
            float(threshold),
        ))
    selected = max(scored)[3]
    return selected, _classification_metrics(
        y_train, inner_probabilities, float(selected)
    )


def run_loso_ablation(
    frame: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT,
    n_estimators: int = 300,
    random_state: int = 42,
    experiment_names: tuple[str, ...] | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_experiments = (
        ("raw_shape", "raw_sam2", SHAPE_FEATURES),
        ("raw_rgb", "raw_sam2", RGB_FEATURES),
        ("raw_shape_rgb", "raw_sam2", SHAPE_FEATURES + RGB_FEATURES),
        ("gt_upper_shape_rgb", "gt_upper_bound", SHAPE_FEATURES + RGB_FEATURES),
    )
    experiments = tuple(
        experiment
        for experiment in all_experiments
        if experiment_names is None or experiment[0] in experiment_names
    )
    if not experiments:
        raise ValueError("No classification experiments selected")
    sites = sorted(int(site) for site in frame["dataset_id"].unique())
    fold_rows: list[dict] = []
    prediction_rows: list[dict] = []
    importances: dict[str, list[np.ndarray]] = defaultdict(list)

    raw = frame[frame["source"] == "sam2"].copy()
    raw_labeled = raw[raw["label_name"].isin(["positive", "clean_negative"])].copy()
    gt_upper = pd.concat([
        frame[frame["source"] == "gt"],
        raw[raw["label_name"] == "clean_negative"],
    ], ignore_index=True)

    for experiment, universe, feature_names in experiments:
        metric_frame = raw_labeled if universe == "raw_sam2" else gt_upper
        prediction_frame = raw if universe == "raw_sam2" else gt_upper
        for heldout_site in sites:
            train = metric_frame[metric_frame["dataset_id"] != heldout_site]
            test = metric_frame[metric_frame["dataset_id"] == heldout_site]
            predict_test = prediction_frame[prediction_frame["dataset_id"] == heldout_site]
            y_train = train["label"].astype(int).to_numpy()
            y_test = test["label"].astype(int).to_numpy()
            if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
                raise RuntimeError(
                    f"Both classes are required in train/test for {experiment}, site {heldout_site}"
                )

            threshold, inner_metrics = _select_threshold_inner_loso(
                train,
                feature_names,
                n_estimators,
                random_state,
            )
            model = _new_classifier(n_estimators, random_state)
            model.fit(train[list(feature_names)].to_numpy(float), y_train)
            test_prob = model.predict_proba(test[list(feature_names)].to_numpy(float))[:, 1]
            all_prob = model.predict_proba(
                predict_test[list(feature_names)].to_numpy(float)
            )[:, 1]
            metrics = _classification_metrics(y_test, test_prob, float(threshold))
            fold_rows.append({
                "experiment": experiment,
                "universe": universe,
                "heldout_site": heldout_site,
                "n_train": len(train),
                "train_positive": int(y_train.sum()),
                "train_negative": int((y_train == 0).sum()),
                "inner_oof_f1_at_selected_threshold": inner_metrics["f1"],
                "inner_oof_pr_auc": inner_metrics["pr_auc"],
                **metrics,
            })
            importances[experiment].append(model.feature_importances_)

            for (_, row), probability in zip(predict_test.iterrows(), all_prob):
                prediction_rows.append({
                    "experiment": experiment,
                    "universe": universe,
                    "heldout_site": heldout_site,
                    "sample_id": row["sample_id"],
                    "source": row["source"],
                    "stem": row["stem"],
                    "dataset_id": int(row["dataset_id"]),
                    "object_index": int(row["object_index"]),
                    "label_name": row["label_name"],
                    "true_label": "" if pd.isna(row["label"]) else int(row["label"]),
                    "used_in_metrics": row["label_name"] != "ambiguous",
                    "selected_threshold": float(threshold),
                    "probability": float(probability),
                    "prediction": int(probability >= threshold),
                    "best_iou": float(row["best_iou"]),
                })

    fold_frame = pd.DataFrame(fold_rows)
    prediction_frame = pd.DataFrame(prediction_rows)
    fold_frame.to_csv(output_dir / "fold_metrics.csv", index=False)
    prediction_frame.to_csv(output_dir / "oof_predictions.csv", index=False)

    summary = {
        "protocol": {
            "outer_split": "Leave-one-dataset_id-out; five fixed geographical sites",
            "classifier": "RandomForestClassifier",
            "n_estimators": n_estimators,
            "random_state": random_state,
            "class_weight": "balanced_subsample",
            "min_samples_leaf": 2,
            "decision_threshold": "selected independently inside each outer fold",
            "threshold_tuning": (
                "inner leave-one-training-site-out OOF F1; grid 0.05..0.95; "
                "outer test site is never used"
            ),
            "positive_rule": f"Hungarian match at IoU >= {POSITIVE_IOU}",
            "clean_negative_rule": (
                "max(intersection/GT, intersection/prediction) "
                f"< {CLEAN_NEGATIVE_OVERLAP} for every GT"
            ),
            "ambiguous_rule": "all remaining SAM2 proposals; excluded from classifier metrics",
            "gt_upper_bound": "GT positive masks plus the same clean-negative SAM2 universe",
        },
        "dataset": {
            "n_rows": int(len(frame)),
            "n_gt_positive": int((frame["source"] == "gt").sum()),
            "n_raw_positive": int((raw["label_name"] == "positive").sum()),
            "n_raw_clean_negative": int((raw["label_name"] == "clean_negative").sum()),
            "n_raw_ambiguous": int((raw["label_name"] == "ambiguous").sum()),
            "raw_label_distribution_by_site": {
                str(site): {
                    name: int(((raw["dataset_id"] == site) & (raw["label_name"] == name)).sum())
                    for name in ("positive", "clean_negative", "ambiguous")
                }
                for site in sites
            },
        },
        "experiments": {},
    }

    for experiment, universe, feature_names in experiments:
        folds = fold_frame[fold_frame["experiment"] == experiment]
        preds = prediction_frame[
            (prediction_frame["experiment"] == experiment)
            & prediction_frame["used_in_metrics"]
        ]
        y_true = preds["true_label"].astype(int).to_numpy()
        probabilities = preds["probability"].to_numpy(float)
        pooled_predictions = preds["prediction"].astype(int).to_numpy()
        summary["experiments"][experiment] = {
            "universe": universe,
            "features": list(feature_names),
            "macro_fold_mean": {
                metric: float(folds[metric].mean())
                for metric in ("precision", "recall", "f1", "pr_auc", "roc_auc")
            },
            "macro_fold_std": {
                metric: float(folds[metric].std(ddof=1))
                for metric in ("precision", "recall", "f1", "pr_auc", "roc_auc")
            },
            "pooled_oof": _classification_metrics(
                y_true,
                probabilities,
                "per-fold inner-LOSO threshold",
                predictions=pooled_predictions,
            ),
            "top_feature_importance": [
                {"feature": name, "importance": float(value)}
                for name, value in sorted(
                    zip(feature_names, np.mean(importances[experiment], axis=0)),
                    key=lambda item: item[1],
                    reverse=True,
                )[:10]
            ],
        }

    if {"raw_shape", "raw_shape_rgb"}.issubset(summary["experiments"]):
        raw_shape_f1 = summary["experiments"]["raw_shape"]["macro_fold_mean"]["f1"]
        combined_f1 = summary["experiments"]["raw_shape_rgb"]["macro_fold_mean"]["f1"]
        summary["ablation_deltas_macro_f1"] = {
            "shape_rgb_minus_shape": combined_f1 - raw_shape_f1,
        }
        if "raw_rgb" in summary["experiments"]:
            raw_rgb_f1 = summary["experiments"]["raw_rgb"]["macro_fold_mean"]["f1"]
            summary["ablation_deltas_macro_f1"].update({
                "rgb_minus_shape": raw_rgb_f1 - raw_shape_f1,
                "shape_rgb_minus_rgb": combined_f1 - raw_rgb_f1,
            })
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--sam-dir", type=Path, default=DEFAULT_SAM_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-gt-area", type=int, default=20)
    parser.add_argument("--min-pred-area", type=int, default=50)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Run only Shape and Shape+RGB on the SAM2 proposal universe",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    object_table = args.output / "object_features.csv"
    if args.reuse_features and object_table.exists():
        frame = pd.read_csv(object_table)
    else:
        frame = build_object_table(
            instances_path=args.instances,
            image_root=args.image_root,
            sam_dir=args.sam_dir,
            output_csv=object_table,
            min_gt_area=args.min_gt_area,
            min_pred_area=args.min_pred_area,
        )
    summary = run_loso_ablation(
        frame,
        output_dir=args.output,
        n_estimators=args.trees,
        experiment_names=("raw_shape", "raw_shape_rgb") if args.core_only else None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
