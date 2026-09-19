"""Stage-wise recall instrumentation for the frozen G1B Mask R-CNN.

Implements the "Correct Next Gate" from
reports/g4c_missed_crown_characterization.md: measure, for the same crowns
already characterized in per_crown_transitions.csv, whether GT recall is lost
at the RPN (proposal never generated / dropped by NMS) or downstream at the
ROI/mask head, and whether that differs between strict-miss and
partial-overlap failure crowns.

Three stages, using the model's own submodules directly (no reimplementation
of torchvision's algorithms -- only invoking its public/semi-public methods):

  Stage A -- pre-NMS: RPN proposals after per-FPN-level top-k selection
             (RegionProposalNetwork._get_top_n_idx), before NMS.
  Stage B -- post-NMS/top-k: the exact proposals RPN hands to the ROI heads
             (RegionProposalNetwork.filter_proposals output).
  Stage C -- final detection survival: reuses the already-computed
             native_matched / target_10cm_matched columns in
             per_crown_transitions.csv (mask-IoU Hungarian match), so the
             existing, already-validated matching logic is not duplicated.

Stage A/B recall is computed as *box* IoU (proposals are boxes, not masks) in
the model's internally-resized coordinate space, since RPN anchors operate
there, not in the original 2048x2048 pixel grid.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image
import shapely.wkb
import torch
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.ops import box_iou

from benchmark.evaluator.policies import EvaluationPolicy
from benchmark.evaluator.schema import CanonicalInstance
from experiments.g1b_baselines.training.train_learned_baselines import build_maskrcnn_model
from experiments.g4b_scale_sensitivity.run_sensitivity import degrade_to_gsd

IOU_THRESHOLDS = (0.10, 0.25, 0.50)
CONDITIONS = (("native", None), ("10cm", 10.0))


def build_ground_truth_boxes(image_instances: pd.DataFrame, policy: EvaluationPolicy) -> tuple[list[str], np.ndarray]:
    """Same primary-track GT filter as run_characterization.py, boxes only."""
    ground_truth = []
    for row in image_instances.itertuples(index=False):
        geometry = shapely.wkb.loads(row.geometry_wkb)
        ground_truth.append(
            CanonicalInstance(
                image_id=str(row.image_id),
                instance_id=str(row.instance_id),
                dataset="bam",
                geometry=geometry,
                bbox=(row.bbox_xmin, row.bbox_ymin, row.bbox_xmax, row.bbox_ymax),
                area_px=float(geometry.area),
                area_m2=0.0,
                site=str(row.site),
                gsd_cm=0.0,
                health_status=str(row.health_status),
                edge_flag=bool(row.edge_flag),
                ignore_flag=bool(row.ignore_flag),
                source_annotation_id=str(row.source_annotation_id),
            )
        )
    filtered = policy.filter_gt_instances(ground_truth, track="primary")
    instance_ids = [instance.instance_id for instance in filtered]
    boxes = np.array([instance.bbox for instance in filtered], dtype=np.float64) if filtered else np.zeros((0, 4))
    return instance_ids, boxes


@torch.inference_mode()
def instrument_image(model, tensor: torch.Tensor, gt_boxes_original: np.ndarray) -> dict:
    """Run the RPN manually to expose Stage A / Stage B proposals for one image."""
    device = tensor.device
    images_transformed, _ = model.transform([tensor])
    original_h, original_w = tensor.shape[-2], tensor.shape[-1]
    transformed_h, transformed_w = images_transformed.image_sizes[0]
    scale = float(transformed_h) / float(original_h)

    features = model.backbone(images_transformed.tensors)
    if isinstance(features, torch.Tensor):
        features = {"0": features}
    feature_list = list(features.values())

    objectness, pred_bbox_deltas = model.rpn.head(feature_list)
    anchors = model.rpn.anchor_generator(images_transformed, feature_list)
    num_anchors_per_level = [o[0].numel() for o in objectness]
    objectness_flat, pred_bbox_deltas_flat = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    proposals = model.rpn.box_coder.decode(pred_bbox_deltas_flat.detach(), anchors)
    proposals = proposals.view(len(anchors), -1, 4)

    # Stage A: per-level top-k selection, before NMS. filter_proposals() does
    # this same reshape internally before calling _get_top_n_idx; replicate
    # it here since we call _get_top_n_idx directly.
    objectness_per_image = objectness_flat.detach().reshape(len(anchors), -1)
    top_n_idx = model.rpn._get_top_n_idx(objectness_per_image, num_anchors_per_level)
    stage_a_boxes = proposals[0, top_n_idx[0]]

    # Stage B: the exact proposals filter_proposals hands to the ROI heads.
    stage_b_boxes_list, _stage_b_scores = model.rpn.filter_proposals(
        proposals, objectness_flat, images_transformed.image_sizes, num_anchors_per_level
    )
    stage_b_boxes = stage_b_boxes_list[0]

    if gt_boxes_original.shape[0] == 0:
        return {"scale": scale, "stage_a_best_iou": np.zeros(0), "stage_b_best_iou": np.zeros(0)}

    gt_boxes_scaled = torch.as_tensor(gt_boxes_original * scale, dtype=torch.float32, device=device)
    stage_a_iou = box_iou(gt_boxes_scaled, stage_a_boxes)
    stage_b_iou = box_iou(gt_boxes_scaled, stage_b_boxes)
    return {
        "scale": scale,
        "n_stage_a_proposals": int(stage_a_boxes.shape[0]),
        "n_stage_b_proposals": int(stage_b_boxes.shape[0]),
        "stage_a_best_iou": stage_a_iou.max(dim=1).values.cpu().numpy(),
        "stage_b_best_iou": stage_b_iou.max(dim=1).values.cpu().numpy(),
    }


def run(args: argparse.Namespace) -> None:
    images = pd.read_csv(args.images_manifest)
    images = images[images["split"] == "val"].copy().sort_values("image_id")
    if args.max_images is not None:
        images = images.head(args.max_images)
    instances = pd.read_parquet(args.instances_manifest)
    instances = instances[instances["image_id"].astype(str).isin(images["image_id"].astype(str))].copy()
    by_image = {key: group for key, group in instances.groupby("image_id", sort=False)}

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_maskrcnn_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.to(device).eval()
    policy = EvaluationPolicy(str(args.eval_config))

    archive = zipfile.ZipFile(args.archive, "r")
    rows: list[dict] = []
    n_images = len(images)
    for count, image_row in enumerate(images.itertuples(index=False), start=1):
        image_id = str(image_row.image_id)
        instance_ids, gt_boxes = build_ground_truth_boxes(by_image.get(image_id, instances.iloc[0:0]), policy)
        if not instance_ids:
            continue
        buffer = archive.read(image_row.archive_member)
        image_rgb = np.array(PIL.Image.open(io.BytesIO(buffer)))[:, :, :3]

        native_gsd = _native_gsd_lookup(args, image_id)
        for condition, target_gsd in CONDITIONS:
            degraded, _low_shape = degrade_to_gsd(image_rgb, native_gsd_cm=native_gsd, target_gsd_cm=target_gsd)
            tensor = torch.from_numpy(degraded).permute(2, 0, 1).float().to(device) / 255.0
            result = instrument_image(model, tensor, gt_boxes)
            for index, instance_id in enumerate(instance_ids):
                rows.append({
                    "image_id": image_id,
                    "instance_id": instance_id,
                    "condition": condition,
                    "stage_a_best_iou": float(result["stage_a_best_iou"][index]),
                    "stage_b_best_iou": float(result["stage_b_best_iou"][index]),
                })
        if count % args.log_interval == 0:
            print(f"instrumented {count}/{n_images} images", flush=True)
    archive.close()

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"wrote {len(frame)} rows to {args.output}")


_GSD_CACHE: dict[str, float] | None = None


def _native_gsd_lookup(args: argparse.Namespace, image_id: str) -> float:
    global _GSD_CACHE
    if _GSD_CACHE is None:
        provenance = pd.read_csv(args.gsd_map)
        _GSD_CACHE = dict(zip(provenance["image_id"].astype(str), provenance["gsd_cm"].astype(float)))
    return _GSD_CACHE[image_id]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/g1b_baselines/checkpoints/maskrcnn_seed42_best.pth"))
    parser.add_argument("--images-manifest", type=Path, default=Path("benchmark/manifests/bam_images.csv"))
    parser.add_argument("--instances-manifest", type=Path, default=Path("benchmark/manifests/bam_instances.parquet"))
    parser.add_argument("--gsd-map", type=Path, default=Path("benchmark/manifests/bam_gsd_provenance.csv"))
    parser.add_argument("--archive", type=Path, default=Path("data/itc_benchmarks/raw_archives/Bamberg_coco2048.zip"))
    parser.add_argument("--eval-config", type=Path, default=Path("benchmark/eval_config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("crown_segmentation_research/experiments/instrumentation_results/rpn_stage_recall.csv"))
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
