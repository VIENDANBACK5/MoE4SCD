"""Sweep decode hyperparameters (prob_threshold, min_peak_distance,
nms_iou_threshold) for the trained StarConvexNet, using cached raw network
outputs so the sweep is decode-only (no repeated forward passes).

Matches this project's established "cache once, sweep cheaply" pattern
(see code/evaluate_merge.py for the Section-4 merge module's parameter
sweep). Evaluated on BAM_val via the same benchmark.evaluator as everything
else in this track, so results are directly comparable to
design_docs/star_convex_v2_training_result.md.
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
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import shapely.wkb

from benchmark.evaluator.metrics import EvaluatorMetrics
from benchmark.evaluator.policies import EvaluationPolicy
from benchmark.evaluator.schema import CanonicalInstance, PredictionInstance
from experiments.g1b_baselines.eval.evaluate_g1b_baselines import aggregate_eval_list
from crown_segmentation_research.methods.star_convex.decode import decode


def load_ground_truth(instances_df: pd.DataFrame, image_id: str) -> list[CanonicalInstance]:
    image_instances = instances_df[instances_df["image_id"].astype(str) == image_id]
    gts = []
    for row in image_instances.itertuples():
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


def run(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.val_manifest)
    instances_df = pd.read_parquet("benchmark/manifests/bam_instances.parquet")
    policy = EvaluationPolicy("benchmark/eval_config.yaml")
    engine = EvaluatorMetrics(policy)

    cached = []
    for row in manifest.itertuples():
        image_id = str(row.image_id)
        path = args.raw_output_dir / f"{image_id.replace(':', '__')}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        gts = load_ground_truth(instances_df, image_id)
        canopy = data["canopy"].astype(np.float32) if "canopy" in data else None
        embedding = data["embedding"].astype(np.float32) if "embedding" in data else None
        cached.append((image_id, data["probability"].astype(np.float32), data["rays"].astype(np.float32), gts, canopy, embedding))
    print(f"Loaded {len(cached)} cached raw outputs" + (" (with canopy head)" if cached and cached[0][4] is not None else "") + (" (with embedding head)" if cached and cached[0][5] is not None else ""))

    has_canopy = bool(cached) and cached[0][4] is not None
    has_embedding = bool(cached) and cached[0][5] is not None
    canopy_thresholds = args.canopy_threshold if has_canopy else [args.canopy_threshold[0]]
    embedding_deltas = args.embedding_delta_d if has_embedding else [args.embedding_delta_d[0]]
    grid = list(product(args.prob_threshold, args.min_peak_distance, args.nms_iou, canopy_thresholds, embedding_deltas))
    rows = []
    for prob_threshold, min_peak_distance, nms_iou, canopy_threshold, embedding_delta_d in grid:
        results = []
        for image_id, probability, rays, gts, canopy, embedding in cached:
            polygons = decode(
                probability, rays, n_rays=args.n_rays,
                prob_threshold=prob_threshold, min_peak_distance=min_peak_distance, nms_iou_threshold=nms_iou,
                canopy=canopy, canopy_threshold=canopy_threshold,
                embedding=embedding, embedding_delta_d=embedding_delta_d,
            )
            preds = [
                PredictionInstance(image_id=image_id, prediction_id=f"p{i}", geometry=p, score=1.0)
                for i, p in enumerate(polygons) if p.is_valid and not p.is_empty
            ]
            results.append(engine.evaluate_image(gts, preds, track="primary"))
        agg = aggregate_eval_list(results)
        rows.append({
            "prob_threshold": prob_threshold, "min_peak_distance": min_peak_distance, "nms_iou": nms_iou,
            "canopy_threshold": canopy_threshold if has_canopy else None,
            "embedding_delta_d": embedding_delta_d if has_embedding else None,
            **{k: agg[k] for k in ("matched_iou", "precision", "recall", "f1", "split_rate", "merge_rate", "miss_rate", "n_pred")},
        })
        print(rows[-1], flush=True)

    frame = pd.DataFrame(rows).sort_values("f1", ascending=False)
    frame.to_csv(args.output_csv, index=False)
    print(f"\nsaved {len(frame)} configs to {args.output_csv}")
    print("\nTop 5 by F1:")
    print(frame.head(5).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-output-dir", type=Path, default=Path("crown_segmentation_research/experiments/star_convex_screen_v2/val_raw_outputs"))
    parser.add_argument("--val-manifest", type=Path, default=Path("crown_segmentation_research/experiments/star_convex_targets_v2/val/manifest.csv"))
    parser.add_argument("--n-rays", type=int, default=16)
    parser.add_argument("--prob-threshold", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--min-peak-distance", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--nms-iou", type=float, nargs="+", default=[0.2, 0.3, 0.5])
    parser.add_argument("--canopy-threshold", type=float, nargs="+", default=[0.5], help="only used if the cached raw output has a 'canopy' array")
    parser.add_argument("--embedding-delta-d", type=float, nargs="+", default=[1.5], help="only used if the cached raw output has an 'embedding' array; matches discriminative_loss.py training delta_d by default")
    parser.add_argument("--output-csv", type=Path, default=Path("crown_segmentation_research/experiments/star_convex_screen_v2/decode_sweep.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
