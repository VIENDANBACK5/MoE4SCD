"""Qualitative + quantitative diagnosis of star-convex v3's failure modes:
WHICH ground-truth crowns get missed, and WHERE false positives come from.

Follows this project's diagnose-before-design convention (see
g4d_rpn_roi_instrumentation.md, oracle_test_hv_decode_result.md): before
designing any new architecture to close the recall/precision gap identified
in star_convex_v3_training_and_merge_result.md, characterize the actual
failure pattern instead of guessing.

Uses the exact same InstanceMatcher the canonical evaluator uses (not a
reimplementation), so "miss"/"false positive" here match the evaluator's own
definitions exactly.
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

import numpy as np
import pandas as pd
import shapely.wkb

from benchmark.evaluator.matching import InstanceMatcher
from benchmark.evaluator.policies import EvaluationPolicy
from benchmark.evaluator.schema import CanonicalInstance, PredictionInstance
from crown_segmentation_research.methods.star_convex.decode import decode

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


def edge_distance_px(geometry) -> float:
    cx, cy = geometry.centroid.x, geometry.centroid.y
    return float(min(cx, cy, MASK_SIZE - cx, MASK_SIZE - cy))


def run(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.val_manifest)
    instances_df = pd.read_parquet("benchmark/manifests/bam_instances.parquet")
    policy = EvaluationPolicy("benchmark/eval_config.yaml")
    matcher = InstanceMatcher(primary_iou_threshold=policy.primary_iou_threshold, split_merge_min_iou=policy.split_merge_min_iou)

    miss_rows = []
    fp_rows = []
    match_rows = []

    for row in manifest.itertuples():
        image_id = str(row.image_id)
        raw_path = args.raw_output_dir / f"{image_id.replace(':', '__')}.npz"
        if not raw_path.exists():
            continue
        raw = np.load(raw_path)
        probability = raw["probability"].astype(np.float32)
        rays = raw["rays"].astype(np.float32)
        canopy = raw["canopy"].astype(np.float32) if "canopy" in raw else None
        embedding = raw["embedding"].astype(np.float32) if "embedding" in raw else None

        gts = load_ground_truth(instances_df, image_id)
        gts = policy.filter_gt_instances(gts, track="primary")
        polygons = decode(
            probability, rays, n_rays=args.n_rays,
            prob_threshold=args.prob_threshold, min_peak_distance=args.min_peak_distance,
            nms_iou_threshold=args.nms_iou, canopy=canopy, canopy_threshold=args.canopy_threshold,
            embedding=embedding, embedding_delta_d=args.embedding_delta_d,
        )
        polygons = [p for p in polygons if p.is_valid and not p.is_empty]
        preds = [PredictionInstance(image_id=image_id, prediction_id=f"p{i}", geometry=p, score=1.0) for i, p in enumerate(polygons)]

        result = matcher.match_instances(gts, preds, iou_thresh=policy.primary_iou_threshold)

        for gt_idx in result["misses"]:
            gt = gts[gt_idx]
            cx, cy = int(gt.geometry.centroid.x), int(gt.geometry.centroid.y)
            cx, cy = np.clip(cx, 0, MASK_SIZE - 1), np.clip(cy, 0, MASK_SIZE - 1)
            miss_rows.append({
                "image_id": image_id, "instance_id": gt.instance_id, "area_px": gt.area_px, "edge_dist_px": edge_distance_px(gt.geometry),
                "prob_at_centroid": float(probability[cy, cx]), "prob_max_in_bbox": _max_prob_in_bbox(probability, gt.bbox),
            })

        for gt_idx, _pred_idx, iou in result["matches"]:
            gt = gts[gt_idx]
            match_rows.append({"image_id": image_id, "instance_id": gt.instance_id, "area_px": gt.area_px, "edge_dist_px": edge_distance_px(gt.geometry), "iou": iou})

        for pred_idx in result["unmatched_pred"]:
            pred = preds[pred_idx]
            overlaps_any_gt = any(result["iou_matrix"][:, pred_idx] > 0.0) if result["iou_matrix"].size else False
            fp_rows.append({
                "image_id": image_id, "area_px": pred.geometry.area, "edge_dist_px": edge_distance_px(pred.geometry),
                "overlaps_some_gt_below_thresh": bool(overlaps_any_gt),
            })

    miss_df = pd.DataFrame(miss_rows)
    match_df = pd.DataFrame(match_rows)
    fp_df = pd.DataFrame(fp_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    miss_df.to_csv(args.output_dir / "misses.csv", index=False)
    match_df.to_csv(args.output_dir / "matches.csv", index=False)
    fp_df.to_csv(args.output_dir / "false_positives.csv", index=False)

    print(f"n_misses={len(miss_df)}, n_matches={len(match_df)}, n_fp={len(fp_df)}\n")

    print("=== Area (px^2): missed vs matched GT crowns ===")
    print(f"  missed : median={miss_df['area_px'].median():.0f}  mean={miss_df['area_px'].mean():.0f}")
    print(f"  matched: median={match_df['area_px'].median():.0f}  mean={match_df['area_px'].mean():.0f}")

    print("\n=== Edge distance (px): missed vs matched GT crowns ===")
    print(f"  missed : median={miss_df['edge_dist_px'].median():.0f}  mean={miss_df['edge_dist_px'].mean():.0f}")
    print(f"  matched: median={match_df['edge_dist_px'].median():.0f}  mean={match_df['edge_dist_px'].mean():.0f}")

    print("\n=== Missed-crown network probability (was there ANY signal?) ===")
    print(f"  prob_at_centroid  : median={miss_df['prob_at_centroid'].median():.3f}  frac>0.1={(miss_df['prob_at_centroid'] > 0.1).mean():.2f}")
    print(f"  prob_max_in_bbox  : median={miss_df['prob_max_in_bbox'].median():.3f}  frac>=prob_threshold({args.prob_threshold})={(miss_df['prob_max_in_bbox'] >= args.prob_threshold).mean():.2f}")

    print("\n=== False positives: do they overlap a GT crown below IoU threshold, or are they pure clutter? ===")
    print(f"  overlaps some GT (below 0.5 IoU): {fp_df['overlaps_some_gt_below_thresh'].mean():.2f}")
    print(f"  pure clutter (no GT overlap at all): {(~fp_df['overlaps_some_gt_below_thresh']).mean():.2f}")
    print(f"  FP area: median={fp_df['area_px'].median():.0f}  matched GT area median={match_df['area_px'].median():.0f}")

    print(f"\nsaved per-instance CSVs to {args.output_dir}")


def _max_prob_in_bbox(probability: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
    xmin, ymin, xmax, ymax = bbox
    xmin, ymin = max(0, int(xmin)), max(0, int(ymin))
    xmax, ymax = min(MASK_SIZE, int(xmax) + 1), min(MASK_SIZE, int(ymax) + 1)
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    return float(probability[ymin:ymax, xmin:xmax].max())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-output-dir", type=Path, default=Path("crown_segmentation_research/experiments/star_convex_screen_v3/val_raw_outputs"))
    parser.add_argument("--val-manifest", type=Path, default=Path("crown_segmentation_research/experiments/star_convex_targets_v2/val/manifest.csv"))
    parser.add_argument("--n-rays", type=int, default=16)
    parser.add_argument("--prob-threshold", type=float, default=0.5)
    parser.add_argument("--min-peak-distance", type=int, default=8)
    parser.add_argument("--nms-iou", type=float, default=0.2)
    parser.add_argument("--canopy-threshold", type=float, default=0.5, help="only used if the cached raw output has a 'canopy' array")
    parser.add_argument("--embedding-delta-d", type=float, default=1.5, help="only used if the cached raw output has an 'embedding' array")
    parser.add_argument("--output-dir", type=Path, default=Path("crown_segmentation_research/experiments/star_convex_screen_v3/failure_diagnosis"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
