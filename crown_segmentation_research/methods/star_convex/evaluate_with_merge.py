"""Integrate the Section-4 merge module into a trained StarConvexNet
checkpoint's decoded output, and evaluate against the un-merged decode and
the G1B Mask R-CNN baseline, all via the same `benchmark.evaluator` engine.

Uses cached raw (probability, rays) outputs (code/cache_star_convex_raw_outputs.py)
so this script is decode+merge-only -- no repeated forward pass. Reuses the
already-adopted merge delta (delta_appearance=0.15, delta_distance_m=2.0)

# Ensure workspace root is in sys.path
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "crown_segmentation_research").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

from design_docs/... merge module result rather than re-sweeping it, since
that parameter is about appearance/distance similarity between mask
fragments and is not specific to which segmenter produced the fragments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import shapely.wkb
from shapely.geometry import MultiPolygon, Polygon

from benchmark.evaluator.geometry import validate_and_repair_geometry
from benchmark.evaluator.metrics import EvaluatorMetrics
from benchmark.evaluator.policies import EvaluationPolicy
from benchmark.evaluator.schema import CanonicalInstance, PredictionInstance
from experiments.g1b_baselines.eval.evaluate_g1b_baselines import aggregate_eval_list
from crown_segmentation_research.methods.star_convex.decode import decode
from crown_segmentation_research.evaluation.merge_module import merge_oversegmented_instances

MASK_SIZE = 2048


def load_ground_truth(instances_df: pd.DataFrame, image_id: str) -> list[CanonicalInstance]:
    rows = instances_df[instances_df["image_id"].astype(str) == image_id]
    gts = []
    for row in rows.itertuples():
        geometry = shapely.wkb.loads(row.geometry_wkb)
        gts.append(
            CanonicalInstance(
                image_id=image_id, instance_id=str(row.instance_id), dataset="bam", geometry=geometry,
                bbox=(row.bbox_xmin, row.bbox_ymin, row.bbox_xmax, row.bbox_ymax),
                area_px=float(row.area_px), area_m2=float(row.area_m2),
                gsd_cm=getattr(row, "gsd_cm", 1.70), edge_flag=bool(row.edge_flag), ignore_flag=bool(row.ignore_flag),
            )
        )
    return gts


def polygons_to_masks(polygons: list[Polygon]) -> list[np.ndarray]:
    masks = []
    for polygon in polygons:
        mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.uint8)
        # ray_to_polygon can fall back to polygon.buffer(0) on a
        # self-intersecting ring, which may yield a MultiPolygon -- draw
        # every sub-polygon's exterior into the same mask.
        sub_polygons = polygon.geoms if isinstance(polygon, MultiPolygon) else [polygon]
        for sub_polygon in sub_polygons:
            coords = np.array(sub_polygon.exterior.coords).round().astype(np.int32)
            cv2.fillPoly(mask, [coords], 1)
        masks.append(mask.astype(bool))
    return masks


def masks_to_predictions(image_id: str, masks: list[np.ndarray], scores: list[float]) -> list[PredictionInstance]:
    predictions = []
    for index, (mask, score) in enumerate(zip(masks, scores)):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) < 3:
                continue
            points = contour.squeeze()
            if points.ndim != 2 or len(points) < 3:
                continue
            polygon = Polygon(points)
            repaired, _, _ = validate_and_repair_geometry(polygon)
            if repaired is None or repaired.is_empty:
                continue
            predictions.append(
                PredictionInstance(image_id=image_id, prediction_id=f"p_{index}", geometry=repaired, score=float(score))
            )
    return predictions


def run(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.val_manifest)
    instances_df = pd.read_parquet("benchmark/manifests/bam_instances.parquet")
    provenance = pd.read_csv("benchmark/manifests/bam_gsd_provenance.csv")
    gsd_lookup = dict(zip(provenance["image_id"].astype(str), provenance["gsd_cm"].astype(float)))
    policy = EvaluationPolicy("benchmark/eval_config.yaml")
    engine = EvaluatorMetrics(policy)

    no_merge_results = []
    merged_results = []
    for row in manifest.itertuples():
        image_id = str(row.image_id)
        raw_path = args.raw_output_dir / f"{image_id.replace(':', '__')}.npz"
        target_path = args.target_dir / f"{image_id.replace(':', '__')}.npz"
        if not raw_path.exists() or not target_path.exists():
            continue
        raw = np.load(raw_path)
        image_rgb = np.load(target_path)["image"]
        gts = load_ground_truth(instances_df, image_id)
        canopy = raw["canopy"].astype(np.float32) if "canopy" in raw else None

        polygons = decode(
            raw["probability"].astype(np.float32), raw["rays"].astype(np.float32), n_rays=args.n_rays,
            prob_threshold=args.prob_threshold, min_peak_distance=args.min_peak_distance,
            nms_iou_threshold=args.nms_iou, canopy=canopy, canopy_threshold=args.canopy_threshold,
        )
        polygons = [p for p in polygons if p.is_valid and not p.is_empty]
        scores = [1.0] * len(polygons)

        no_merge_results.append(engine.evaluate_image(gts, [
            PredictionInstance(image_id=image_id, prediction_id=f"p{i}", geometry=p, score=1.0)
            for i, p in enumerate(polygons)
        ], track="primary"))

        if len(polygons) > 1:
            masks = polygons_to_masks(polygons)
            gsd_cm = gsd_lookup.get(image_id, 1.70)
            merged_masks, merged_scores = merge_oversegmented_instances(
                masks, scores, image_rgb, gsd_cm,
                delta_appearance=args.delta_appearance, delta_distance_m=args.delta_distance_m,
            )
            merged_preds = masks_to_predictions(image_id, merged_masks, merged_scores)
        else:
            merged_preds = masks_to_predictions(image_id, polygons_to_masks(polygons), scores) if polygons else []
        merged_results.append(engine.evaluate_image(gts, merged_preds, track="primary"))

    no_merge_agg = aggregate_eval_list(no_merge_results)
    merged_agg = aggregate_eval_list(merged_results)

    print(f"{'':20s} {'matched_iou':>12s} {'precision':>10s} {'recall':>8s} {'f1':>8s} {'split_rate':>11s} {'merge_rate':>11s} {'miss_rate':>10s} {'n_pred':>7s}")
    for label, agg in [("star_convex (no merge)", no_merge_agg), ("star_convex + merge", merged_agg)]:
        print(f"{label:20s} {agg['matched_iou']:12.4f} {agg['precision']:10.4f} {agg['recall']:8.4f} {agg['f1']:8.4f} "
              f"{agg['split_rate']:11.4f} {agg['merge_rate']:11.4f} {agg['miss_rate']:10.4f} {agg['n_pred']:7d}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-output-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True, help="precomputed dir with the raw RGB images (has 'image' key in each npz)")
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--n-rays", type=int, default=16)
    parser.add_argument("--prob-threshold", type=float, default=0.4)
    parser.add_argument("--min-peak-distance", type=int, default=3)
    parser.add_argument("--nms-iou", type=float, default=0.2)
    parser.add_argument("--canopy-threshold", type=float, default=0.5, help="only used if the cached raw output has a 'canopy' array")
    parser.add_argument("--delta-appearance", type=float, default=0.15)
    parser.add_argument("--delta-distance-m", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
