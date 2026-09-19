"""Evaluate the Section-4 merge module against the frozen G1B Mask R-CNN
baseline, on the same evaluator (`benchmark/evaluator`) G1B itself uses, so
numbers are directly comparable to `reports/g1b_baseline_report.md`.

Two modes:
  cache   -- run the frozen model once per image in a split, save raw
             (masks, scores) to disk. Expensive (GPU), run once.
  sweep   -- given cached masks, try a grid of (delta_appearance,
             delta_distance_m) and report split_rate/matched_iou deltas vs
             the no-merge baseline. Cheap (CPU-bound), run many times.
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
import io
import zipfile
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import PIL.Image
import shapely.wkb
import torch
from shapely.geometry import Polygon

from benchmark.evaluator.geometry import validate_and_repair_geometry
from benchmark.evaluator.metrics import EvaluatorMetrics
from benchmark.evaluator.policies import EvaluationPolicy
from benchmark.evaluator.schema import CanonicalInstance, PredictionInstance
from experiments.g1b_baselines.eval.evaluate_g1b_baselines import aggregate_eval_list
from experiments.g1b_baselines.training.train_learned_baselines import build_maskrcnn_model
from crown_segmentation_research.evaluation.merge_module import merge_oversegmented_instances

SCORE_THRESHOLD = 0.40
MIN_AREA_PX = 100.0


def load_ground_truth(instances_df: pd.DataFrame) -> dict[str, list[CanonicalInstance]]:
    gt: dict[str, list[CanonicalInstance]] = {}
    for row in instances_df.itertuples():
        image_id = str(row.image_id)
        geometry = shapely.wkb.loads(row.geometry_wkb)
        gt.setdefault(image_id, []).append(
            CanonicalInstance(
                image_id=image_id,
                instance_id=str(row.instance_id),
                dataset="bam",
                geometry=geometry,
                bbox=(row.bbox_xmin, row.bbox_ymin, row.bbox_xmax, row.bbox_ymax),
                area_px=float(row.area_px),
                area_m2=float(row.area_m2),
                gsd_cm=getattr(row, "gsd_cm", 1.70),
                edge_flag=bool(row.edge_flag),
                ignore_flag=bool(row.ignore_flag),
            )
        )
    return gt


def cmd_cache(args: argparse.Namespace) -> None:
    images_df = pd.read_csv("benchmark/manifests/bam_images.csv")
    split_df = images_df[images_df["split"] == args.split].reset_index(drop=True)
    if args.limit is not None:
        split_df = split_df.head(args.limit)
    print(f"Caching raw predictions for {len(split_df)} images ({args.split})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_maskrcnn_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.to(device).eval()

    zfile = zipfile.ZipFile(args.archive, "r")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for count, row in enumerate(split_df.itertuples(), start=1):
            image_id = str(row.image_id)
            cache_path = args.cache_dir / f"{image_id.replace(':', '__')}.npz"
            if cache_path.exists():
                continue
            buf = zfile.read(row.archive_member)
            image_rgb = np.array(PIL.Image.open(io.BytesIO(buf)))[:, :, :3]
            tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().to(device) / 255.0
            output = model([tensor])[0]
            scores = output["scores"].cpu().numpy()
            masks = output["masks"].cpu().numpy()[:, 0]  # (N, H, W) float
            keep = scores >= SCORE_THRESHOLD
            masks_bin = (masks[keep] > 0.5)
            areas = masks_bin.reshape(len(masks_bin), -1).sum(axis=1) if len(masks_bin) else np.zeros(0)
            area_keep = areas >= MIN_AREA_PX
            np.savez_compressed(
                cache_path,
                masks=np.packbits(masks_bin[area_keep], axis=None) if area_keep.any() else np.zeros(0, dtype=np.uint8),
                mask_shape=np.array(masks_bin.shape[1:]) if masks_bin.ndim == 3 else np.array(image_rgb.shape[:2]),
                n_masks=int(area_keep.sum()),
                scores=scores[keep][area_keep],
            )
            if count % 20 == 0:
                print(f"  cached {count}/{len(split_df)}", flush=True)
    zfile.close()
    print("done")


def _load_cached(cache_path: Path) -> tuple[list[np.ndarray], list[float]]:
    data = np.load(cache_path)
    n_masks = int(data["n_masks"])
    if n_masks == 0:
        return [], []
    h, w = data["mask_shape"]
    flat = np.unpackbits(data["masks"])[: n_masks * h * w]
    masks = flat.reshape(n_masks, h, w).astype(bool)
    return [masks[i] for i in range(n_masks)], list(data["scores"])


def _masks_to_predictions(image_id: str, masks: list[np.ndarray], scores: list[float]) -> list[PredictionInstance]:
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
            if repaired is None or repaired.is_empty or repaired.area < MIN_AREA_PX:
                continue
            predictions.append(
                PredictionInstance(image_id=image_id, prediction_id=f"p_{index}", geometry=repaired, score=float(score))
            )
    return predictions


def cmd_sweep(args: argparse.Namespace) -> None:
    images_df = pd.read_csv("benchmark/manifests/bam_images.csv")
    split_df = images_df[images_df["split"] == args.split].reset_index(drop=True)
    instances_df = pd.read_parquet("benchmark/manifests/bam_instances.parquet")
    gt_dict = load_ground_truth(instances_df)
    provenance = pd.read_csv("benchmark/manifests/bam_gsd_provenance.csv")
    gsd_lookup = dict(zip(provenance["image_id"].astype(str), provenance["gsd_cm"].astype(float)))

    policy = EvaluationPolicy("benchmark/eval_config.yaml")
    engine = EvaluatorMetrics(policy)
    zfile = zipfile.ZipFile(args.archive, "r")

    cached = []
    for row in split_df.itertuples():
        image_id = str(row.image_id)
        cache_path = args.cache_dir / f"{image_id.replace(':', '__')}.npz"
        if not cache_path.exists():
            continue
        masks, scores = _load_cached(cache_path)
        cached.append((image_id, row.archive_member, masks, scores))
    print(f"Loaded {len(cached)} cached prediction sets")

    grid = list(product(args.delta_appearance, args.delta_distance_m))
    results = {}
    baseline_results = []
    for image_id, _member, masks, scores in cached:
        preds = _masks_to_predictions(image_id, masks, scores)
        baseline_results.append(engine.evaluate_image(gt_dict.get(image_id, []), preds, track="primary"))
    results["no_merge"] = aggregate_eval_list(baseline_results)

    rgb_cache: dict[str, np.ndarray] = {}
    for delta_appearance, delta_distance_m in grid:
        merged_results = []
        for image_id, member, masks, scores in cached:
            if image_id not in rgb_cache:
                buf = zfile.read(member)
                rgb_cache[image_id] = np.array(PIL.Image.open(io.BytesIO(buf)))[:, :, :3]
            gsd_cm = gsd_lookup[image_id]
            merged_masks, merged_scores = merge_oversegmented_instances(
                masks, scores, rgb_cache[image_id], gsd_cm,
                delta_appearance=delta_appearance, delta_distance_m=delta_distance_m,
            )
            preds = _masks_to_predictions(image_id, merged_masks, merged_scores)
            merged_results.append(engine.evaluate_image(gt_dict.get(image_id, []), preds, track="primary"))
        key = f"merge_dA{delta_appearance}_dD{delta_distance_m}"
        results[key] = aggregate_eval_list(merged_results)
    zfile.close()

    print(f"\n{'config':35s} {'matched_iou':>12s} {'split_rate':>12s} {'merge_rate':>12s} {'miss_rate':>12s} {'n_pred':>8s}")
    for key, agg in results.items():
        print(f"{key:35s} {agg['matched_iou']:12.4f} {agg['split_rate']:12.4f} {agg['merge_rate']:12.4f} {agg['miss_rate']:12.4f} {agg['n_pred']:8d}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    cache = sub.add_parser("cache")
    cache.add_argument("--split", default="val")
    cache.add_argument("--checkpoint", type=Path, default=Path("experiments/g1b_baselines/checkpoints/maskrcnn_seed42_best.pth"))
    cache.add_argument("--archive", type=Path, default=Path("data/itc_benchmarks/raw_archives/Bamberg_coco2048.zip"))
    cache.add_argument("--cache-dir", type=Path, default=Path("crown_segmentation_research/experiments/merge_module_results/cache/val"))
    cache.add_argument("--limit", type=int)
    cache.set_defaults(func=cmd_cache)

    sweep = sub.add_parser("sweep")
    sweep.add_argument("--split", default="val")
    sweep.add_argument("--archive", type=Path, default=Path("data/itc_benchmarks/raw_archives/Bamberg_coco2048.zip"))
    sweep.add_argument("--cache-dir", type=Path, default=Path("crown_segmentation_research/experiments/merge_module_results/cache/val"))
    sweep.add_argument("--delta-appearance", type=float, nargs="+", default=[0.05, 0.10, 0.15])
    sweep.add_argument("--delta-distance-m", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    sweep.set_defaults(func=cmd_sweep)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
