"""Oracle test: does compute_hv_targets + decode_instances recover real BAM
crowns from noise-free, GT-derived targets?

This deliberately skips training a network. If the representation+decode
mechanism cannot recover instances from perfect targets, no amount of model
training fixes that -- this is the cheap check to run before committing to
Section 3 of design_docs/method_design_dense_crown_separation_v1.md.

Reuses the same benchmark.evaluator machinery as the rest of this project so
matched_iou/split_rate/merge_rate/miss_rate are directly comparable to the
G1B baseline and the Section-4 merge-module numbers.
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
from pathlib import Path

import cv2
import pandas as pd
import rasterio
import rasterio.features
import shapely.wkb
from affine import Affine
from shapely.geometry import Polygon

from benchmark.evaluator.geometry import validate_and_repair_geometry
from benchmark.evaluator.metrics import EvaluatorMetrics
from benchmark.evaluator.policies import EvaluationPolicy
from benchmark.evaluator.schema import CanonicalInstance, PredictionInstance
from experiments.g1b_baselines.eval.evaluate_g1b_baselines import aggregate_eval_list
from crown_segmentation_research.legacy.dense_hv_representation import (
    compute_hv_targets,
    decode_instances,
)

MIN_AREA_PX = 100.0


def _masks_to_predictions(image_id: str, masks: list) -> list[PredictionInstance]:
    predictions = []
    for index, mask in enumerate(masks):
        contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) < 3:
                continue
            points = contour.squeeze()
            if points.ndim != 2 or len(points) < 3:
                continue
            polygon = Polygon(points)
            repaired, _, _ = validate_and_repair_geometry(polygon)
            if repaired is None or repaired.is_empty or repaired.area < MIN_AREA_PX:
                continue
            predictions.append(
                PredictionInstance(image_id=image_id, prediction_id=f"o_{index}", geometry=repaired, score=1.0)
            )
    return predictions


def run(args: argparse.Namespace) -> None:
    images_df = pd.read_csv("benchmark/manifests/bam_images.csv")
    images_df["image_id"] = images_df["image_id"].astype(str)
    split_df = images_df[images_df["split"] == args.split].reset_index(drop=True)
    if args.limit is not None:
        split_df = split_df.head(args.limit)
    instances_df = pd.read_parquet("benchmark/manifests/bam_instances.parquet")
    policy = EvaluationPolicy("benchmark/eval_config.yaml")
    engine = EvaluatorMetrics(policy)

    results = []
    for row in split_df.itertuples():
        image_id = str(row.image_id)
        image_instances = instances_df[instances_df["image_id"].astype(str) == image_id]
        if image_instances.empty:
            continue

        ground_truth = []
        instance_masks = []
        for instance_row in image_instances.itertuples():
            geometry = shapely.wkb.loads(instance_row.geometry_wkb)
            ground_truth.append(
                CanonicalInstance(
                    image_id=image_id,
                    instance_id=str(instance_row.instance_id),
                    dataset="bam",
                    geometry=geometry,
                    bbox=(instance_row.bbox_xmin, instance_row.bbox_ymin, instance_row.bbox_xmax, instance_row.bbox_ymax),
                    area_px=float(instance_row.area_px),
                    area_m2=float(instance_row.area_m2),
                    gsd_cm=getattr(instance_row, "gsd_cm", 1.70),
                    edge_flag=bool(instance_row.edge_flag),
                    ignore_flag=bool(instance_row.ignore_flag),
                )
            )
            mask = rasterio.features.rasterize(
                [(geometry, 1)], out_shape=(2048, 2048), transform=Affine.identity(), fill=0, dtype="uint8"
            ).astype(bool)
            instance_masks.append(mask)

        p_fg, h_map, v_map = compute_hv_targets(instance_masks)
        decoded_masks = decode_instances(
            p_fg, h_map, v_map,
            marker_percentile=args.marker_percentile,
            min_marker_area=args.min_marker_area,
        )
        predictions = _masks_to_predictions(image_id, decoded_masks)
        results.append(engine.evaluate_image(ground_truth, predictions, track="primary"))
        if len(results) % 10 == 0:
            print(f"  processed {len(results)} images", flush=True)

    aggregated = aggregate_eval_list(results)
    print(f"\n=== Oracle H/V decode on BAM_{args.split} (n={len(results)} images) ===")
    for key in ("matched_iou", "split_rate", "merge_rate", "miss_rate", "n_gt", "n_pred", "tp"):
        print(f"  {key}: {aggregated[key]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--marker-percentile", type=float, default=30.0)
    parser.add_argument("--min-marker-area", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
