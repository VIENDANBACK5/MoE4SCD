"""Split the 668 G4C-lost crowns (at 10cm) into: RPN-adjacent proposal exists,
then either (a) filtered by the classification score threshold, (b) survives
scoring but the decoded mask itself falls below IoU 0.50, or (c) neither --
proposal, score, and mask all look fine, so the failure must come from
elsewhere (e.g. cross-instance NMS suppression), a residual this script
surfaces but does not further attribute.

This is the open question left by reports/g4d_rpn_roi_instrumentation.md: G4D
showed proposals survive; this script asks *why the ROI/mask stage still
fails* for the same crowns, using the model's own box_predictor and
mask_predictor directly (no reimplementation of their internals) on exactly
the best-IoU Stage-B proposal already identified for each crown.

Simplification versus the full production path: this script scores and masks
the *original* Stage-B proposal box, without applying the box-regression
refinement RoIHeads normally applies before the final NMS/threshold step. A
refined box could shift the mask-IoU number; treat "mask_quality_fail" here as
an upper bound on how much the raw region already explains the failure, not
an exact reproduction of the production decision.
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
from affine import Affine
from rasterio.features import rasterize
from torchvision.models.detection.roi_heads import paste_masks_in_image
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.ops import box_iou

from benchmark.evaluator.policies import EvaluationPolicy
from benchmark.evaluator.schema import CanonicalInstance
from experiments.g1b_baselines.training.train_learned_baselines import build_maskrcnn_model
from experiments.g4b_scale_sensitivity.run_sensitivity import degrade_to_gsd

SCORE_THRESHOLD = 0.40  # same operating threshold used everywhere else in this track (run_characterization.py --score-threshold default)
MASK_IOU_THRESHOLD = 0.50  # same primary IoU threshold used by benchmark/eval_config.yaml


def load_target_instances(transitions_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(transitions_path)
    return frame[frame["transition"].isin(["lost_strict_miss", "lost_partial_overlap"])].copy()


def build_ground_truth(image_instances: pd.DataFrame, policy: EvaluationPolicy) -> dict[str, CanonicalInstance]:
    ground_truth = {}
    for row in image_instances.itertuples(index=False):
        geometry = shapely.wkb.loads(row.geometry_wkb)
        instance = CanonicalInstance(
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
        ground_truth[instance.instance_id] = instance
    filtered_ids = {instance.instance_id for instance in policy.filter_gt_instances(list(ground_truth.values()), track="primary")}
    return {instance_id: inst for instance_id, inst in ground_truth.items() if instance_id in filtered_ids}


@torch.inference_mode()
def stage_b_proposals(model, tensor: torch.Tensor):
    images_transformed, _ = model.transform([tensor])
    original_h = tensor.shape[-2]
    transformed_h = images_transformed.image_sizes[0][0]
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
    boxes_list, _scores = model.rpn.filter_proposals(
        proposals, objectness_flat, images_transformed.image_sizes, num_anchors_per_level
    )
    return features, boxes_list[0], scale, images_transformed.image_sizes[0]


@torch.inference_mode()
def score_and_mask(model, features, proposal_boxes: torch.Tensor, image_size: tuple[int, int]):
    box_features = model.roi_heads.box_roi_pool(features, [proposal_boxes], [image_size])
    box_features = model.roi_heads.box_head(box_features)
    class_logits, _box_regression = model.roi_heads.box_predictor(box_features)
    tree_scores = torch.softmax(class_logits, dim=1)[:, 1]

    mask_features = model.roi_heads.mask_roi_pool(features, [proposal_boxes], [image_size])
    mask_features = model.roi_heads.mask_head(mask_features)
    mask_logits = model.roi_heads.mask_predictor(mask_features)
    mask_probs = mask_logits[:, 1:2].sigmoid()
    return tree_scores, mask_probs


def run(args: argparse.Namespace) -> None:
    targets = load_target_instances(args.transitions)
    target_ids_by_image = {
        image_id: set(group["instance_id"])
        for image_id, group in targets.groupby("image_id")
    }
    instances = pd.read_parquet(args.instances_manifest)
    provenance = pd.read_csv(args.gsd_map)
    provenance["image_id"] = provenance["image_id"].astype(str)
    gsd_lookup = dict(zip(provenance["image_id"], provenance["gsd_cm"].astype(float)))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_maskrcnn_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.to(device).eval()
    policy = EvaluationPolicy(str(args.eval_config))

    archive = zipfile.ZipFile(args.archive, "r")
    images = pd.read_csv(args.images_manifest)
    images["image_id"] = images["image_id"].astype(str)
    rows: list[dict] = []
    processed = 0
    for image_row in images.itertuples(index=False):
        image_id = str(image_row.image_id)
        target_ids = target_ids_by_image.get(image_id)
        if not target_ids:
            continue
        image_instances = instances[instances["image_id"].astype(str) == image_id]
        ground_truth = build_ground_truth(image_instances, policy)
        ground_truth = {k: v for k, v in ground_truth.items() if k in target_ids}
        if not ground_truth:
            continue

        native_gsd = gsd_lookup[image_id]
        buffer = archive.read(image_row.archive_member)
        image_rgb = np.array(PIL.Image.open(io.BytesIO(buffer)))[:, :, :3]
        degraded, _low_shape = degrade_to_gsd(image_rgb, native_gsd_cm=native_gsd, target_gsd_cm=10.0)
        tensor = torch.from_numpy(degraded).permute(2, 0, 1).float().to(device) / 255.0

        features, stage_b_boxes, scale, image_size = stage_b_proposals(model, tensor)

        instance_ids = list(ground_truth.keys())
        gt_boxes_original = np.array([ground_truth[i].bbox for i in instance_ids], dtype=np.float64)
        gt_boxes_scaled = torch.as_tensor(gt_boxes_original * scale, dtype=torch.float32, device=device)
        iou = box_iou(gt_boxes_scaled, stage_b_boxes)
        best_iou, best_index = iou.max(dim=1)

        selected_boxes = stage_b_boxes[best_index]
        tree_scores, mask_probs = score_and_mask(model, features, selected_boxes, image_size)

        original_boxes = (selected_boxes / scale).cpu()
        pasted = paste_masks_in_image(mask_probs.cpu(), original_boxes, (image_rgb.shape[0], image_rgb.shape[1]))
        pasted_binary = (pasted[:, 0] > 0.5).numpy()

        for index, instance_id in enumerate(instance_ids):
            geometry = ground_truth[instance_id].geometry
            gt_mask = rasterize(
                [(geometry, 1)],
                out_shape=(image_rgb.shape[0], image_rgb.shape[1]),
                transform=Affine.identity(),
                fill=0,
                dtype=np.uint8,
            ).astype(bool)
            pred_mask = pasted_binary[index]
            intersection = float((gt_mask & pred_mask).sum())
            union = float((gt_mask | pred_mask).sum())
            mask_iou = intersection / union if union > 0 else 0.0
            score = float(tree_scores[index])
            if score < SCORE_THRESHOLD:
                cause = "score_filtered"
            elif mask_iou < MASK_IOU_THRESHOLD:
                cause = "mask_quality_fail"
            else:
                cause = "unexplained_by_score_or_mask"
            rows.append({
                "image_id": image_id,
                "instance_id": instance_id,
                "stage_b_box_iou": float(best_iou[index]),
                "tree_class_score": score,
                "mask_iou_at_best_proposal": mask_iou,
                "cause": cause,
            })
        processed += 1
        if processed % args.log_interval == 0:
            print(f"processed {processed} images with target crowns", flush=True)
    archive.close()

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"wrote {len(frame)} rows to {args.output}")
    print(frame["cause"].value_counts())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/g1b_baselines/checkpoints/maskrcnn_seed42_best.pth"))
    parser.add_argument("--images-manifest", type=Path, default=Path("benchmark/manifests/bam_images.csv"))
    parser.add_argument("--instances-manifest", type=Path, default=Path("benchmark/manifests/bam_instances.parquet"))
    parser.add_argument("--gsd-map", type=Path, default=Path("benchmark/manifests/bam_gsd_provenance.csv"))
    parser.add_argument("--archive", type=Path, default=Path("data/itc_benchmarks/raw_archives/Bamberg_coco2048.zip"))
    parser.add_argument("--eval-config", type=Path, default=Path("benchmark/eval_config.yaml"))
    parser.add_argument("--transitions", type=Path, default=Path("experiments/g4c_missed_crowns/results/seed42/per_crown_transitions.csv"))
    parser.add_argument("--output", type=Path, default=Path("crown_segmentation_research/experiments/instrumentation_results/roi_score_vs_mask_split.csv"))
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
