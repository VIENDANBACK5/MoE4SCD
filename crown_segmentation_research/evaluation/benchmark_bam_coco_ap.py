"""
Academic COCO AP Benchmark on BAMFORESTS Individual Tree Crown Dataset.

Evaluates trained models (StarConvex v8, v6, v7, TreeFlowNet, etc.) on the official
BAMFORESTS COCO benchmarks using standard pycocotools.cocoeval.COCOeval.

Metrics reported:
  - COCO AP@[0.50:0.95], AP50, AP75
  - APs (small), APm (medium), APl (large)
  - AR1, AR10, AR100, ARs, ARm, ARl
  - Precision, Recall, F1@IoU0.5 (Hungarian matched)
  - Panoptic Quality (PQ), SQ, RQ
"""
from __future__ import annotations

import os
import sys
# Remove user site-packages to avoid torch/torchvision ABI incompatibility
sys.path = [p for p in sys.path if not p.startswith('/home/chung/.local')]

import argparse
import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as coco_mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon, MultiPolygon

# Ensure workspace root is in sys.path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ── Paths & Config ─────────────────────────────────────────────────────────────
ZIP_ARCHIVE = Path("/home/chung/RS/data/itc_benchmarks/raw_archives/Bamberg_coco2048.zip")
BAM_COCO_DIR = Path("/home/chung/RS/data/itc_benchmarks/extracted/bam_coco2048")
MANIFEST_CACHE = REPO / "crown_segmentation_research/datasets/bam_zip_manifest.json"

SPLIT_MAP = {
    "eval": BAM_COCO_DIR / "instances_tree_eval2023.json",
    "test1": BAM_COCO_DIR / "instances_tree_TestSet12023.json",
    "test2": BAM_COCO_DIR / "instances_tree_TestSet22023.json",
    "train": BAM_COCO_DIR / "instances_tree_train2023.json",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_zip_manifest() -> dict[str, str]:
    if MANIFEST_CACHE.exists():
        with open(MANIFEST_CACHE, "r") as f:
            return json.load(f)
    print(f"Building zip manifest cache from {ZIP_ARCHIVE}...", flush=True)
    with zipfile.ZipFile(ZIP_ARCHIVE, "r") as zf:
        tifs = [n for n in zf.namelist() if n.endswith(".tif")]
        name_to_zip = {Path(n).name: n for n in tifs}
    MANIFEST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_CACHE, "w") as f:
        json.dump(name_to_zip, f)
    return name_to_zip


def load_image_from_zip(zf: zipfile.ZipFile, name_to_zip: dict[str, str], file_name: str) -> np.ndarray:
    if file_name in name_to_zip:
        data = zf.read(name_to_zip[file_name])
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img)
    for prefix in ("", "images/", "coco2048/images/", "coco2048/val2023/", "coco2048/test2023/Test-Set-1/", "coco2048/test2023/Test-Set-2/", "coco2048/train2023/"):
        try:
            data = zf.read(prefix + file_name)
            img = Image.open(io.BytesIO(data)).convert("RGB")
            return np.array(img)
        except KeyError:
            continue
    raise FileNotFoundError(f"Image {file_name} not found in zip archive.")


def polygons_to_binary_masks(polygons: list[Polygon], shape: tuple[int, int]) -> list[np.ndarray]:
    masks = []
    for polygon in polygons:
        mask = np.zeros(shape, dtype=np.uint8)
        subs = polygon.geoms if isinstance(polygon, MultiPolygon) else [polygon]
        for sub in subs:
            coords = np.array(sub.exterior.coords).round().astype(np.int32)
            cv2.fillPoly(mask, [coords], 1)
        if mask.sum() > 0:
            masks.append(mask)
    return masks


def mask_to_bbox(mask: np.ndarray) -> list[float]:
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()
    return [float(cmin), float(rmin), float(cmax - cmin + 1), float(rmax - rmin + 1)]


def rle_to_mask(rle_or_poly, H: int, W: int) -> np.ndarray:
    if isinstance(rle_or_poly, dict):
        return coco_mask_util.decode(rle_or_poly).astype(bool)
    else:
        rle = coco_mask_util.frPyObjects(rle_or_poly, H, W)
        rle = coco_mask_util.merge(rle)
        return coco_mask_util.decode(rle).astype(bool)


# ── Model Loaders ─────────────────────────────────────────────────────────────

def load_star_convex_model(ckpt_path: str, model_type: str = "v8"):
    from crown_segmentation_research.methods.star_convex.model import StarConvexNet

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    n_rays = state["ray_head.body.4.weight"].shape[0]
    use_canopy = "canopy_head.body.4.weight" in state
    use_embedding = "embedding_head.body.4.weight" in state
    use_centroid = "centroid_head.body.4.weight" in state
    use_sdt = "sdt_head.body.4.weight" in state

    model = StarConvexNet(
        n_rays=n_rays,
        pretrained_backbone=False,
        use_canopy_head=use_canopy,
        use_embedding_head=use_embedding,
        embedding_dim=state["embedding_head.body.4.weight"].shape[0] if use_embedding else 8,
        use_centroid_head=use_centroid,
        use_sdt_head=use_sdt,
    )
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    print(f"Loaded StarConvex ({model_type}) checkpoint: {ckpt_path} (n_rays={n_rays}, canopy={use_canopy}, embedding={use_embedding})", flush=True)
    return model


def load_tree_flow_model(ckpt_path: str):
    from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet

    model = TreeFlowNet()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    print(f"Loaded TreeFlowNet checkpoint: {ckpt_path}", flush=True)
    return model


# ── Inference Functions ────────────────────────────────────────────────────────

def infer_star_convex_tile(
    model,
    img_tile: np.ndarray,
    decode_cfg: dict[str, Any],
) -> tuple[list[np.ndarray], list[float]]:
    """Infers on a single tile (e.g. 1024x1024 or 2048x2048) with StarConvexNet."""
    from crown_segmentation_research.methods.star_convex.decode import decode as star_convex_decode

    H, W = img_tile.shape[:2]
    # StarConvexNet training normalization: / 255.0
    img_t = torch.from_numpy(img_tile).permute(2, 0, 1).float() / 255.0
    img_t = img_t.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(img_t)
        prob = out["probability"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        rays = out["rays"].squeeze(0).cpu().numpy().astype(np.float32)
        canopy = out["canopy"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32) if "canopy" in out else None
        embedding = out["embedding"].squeeze(0).cpu().numpy().astype(np.float32) if "embedding" in out else None

    polygons = star_convex_decode(
        probability=prob,
        rays=rays,
        n_rays=model.n_rays,
        prob_threshold=decode_cfg.get("prob_threshold", 0.5),
        min_peak_distance=decode_cfg.get("min_peak_distance", 5),
        nms_iou_threshold=decode_cfg.get("nms_iou_threshold", 0.2),
        canopy=canopy,
        canopy_threshold=decode_cfg.get("canopy_threshold", 0.7),
        embedding=embedding,
        embedding_delta_d=decode_cfg.get("embedding_delta_d", 3.0),
    )
    polygons = [p for p in polygons if p.is_valid and not p.is_empty]
    masks = polygons_to_binary_masks(polygons, (H, W))
    
    # Compute confidence score per mask based on peak probability in mask
    scores = []
    for m in masks:
        score = float(prob[m > 0].max()) if m.sum() > 0 else 0.5
        scores.append(score)

    return masks, scores


def infer_tree_flow_tile(
    model,
    img_tile: np.ndarray,
    decode_cfg: dict[str, Any],
) -> tuple[list[np.ndarray], list[float]]:
    """Infers on a single tile with TreeFlowNet."""
    from crown_segmentation_research.methods.tree_flow.decode import decode as tree_flow_decode

    H, W = img_tile.shape[:2]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_t = torch.from_numpy(img_tile).permute(2, 0, 1).float() / 255.0
    img_t = (img_t - mean) / std
    img_t = img_t.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(img_t)
        out_sq = {k: v[0] for k, v in out.items()}

    instances = tree_flow_decode(out_sq, **decode_cfg)
    masks = [inst["mask"].astype(np.uint8) for inst in instances]
    scores = [float(inst["score"]) for inst in instances]
    return masks, scores


# ── Full Split Evaluation ──────────────────────────────────────────────────────

def run_split_evaluation(
    model: Any,
    model_type: str,
    split_name: str,
    decode_cfg: dict[str, Any],
    max_images: int | None = None,
    eval_mode: str = "full_2048",  # "full_2048" or "crops_1024"
) -> dict[str, Any]:
    json_path = SPLIT_MAP[split_name]
    print(f"\n====================================================================", flush=True)
    print(f"RUNNING BENCHMARK: [{model_type.upper()}] on [{split_name.upper()}] split", flush=True)
    print(f"Annotation file: {json_path}", flush=True)
    print(f"Eval mode: {eval_mode}, Max images: {max_images or 'ALL'}", flush=True)
    print(f"Decode config: {decode_cfg}", flush=True)
    print(f"====================================================================", flush=True)

    coco_gt = COCO(str(json_path))
    name_to_zip = get_zip_manifest()
    img_ids = coco_gt.getImgIds()
    if max_images is not None:
        img_ids = img_ids[:max_images]

    coco_results = []
    t0 = time.time()

    with zipfile.ZipFile(ZIP_ARCHIVE, "r") as zf:
        for idx, img_id in enumerate(img_ids):
            img_info = coco_gt.loadImgs(img_id)[0]
            file_name = img_info["file_name"]
            try:
                img_np = load_image_from_zip(zf, name_to_zip, file_name)
            except FileNotFoundError as e:
                print(f"  [WARN] Skipping {file_name}: {e}", flush=True)
                continue

            H, W = img_np.shape[:2]

            if eval_mode == "full_2048":
                # Evaluate full resolution directly
                if "star_convex" in model_type:
                    masks, scores = infer_star_convex_tile(model, img_np, decode_cfg)
                elif "tree_flow" in model_type:
                    masks, scores = infer_tree_flow_tile(model, img_np, decode_cfg)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                for mask, score in zip(masks, scores):
                    if mask.sum() < 10:
                        continue
                    rle = coco_mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    bbox = mask_to_bbox(mask)
                    coco_results.append({
                        "image_id": img_id,
                        "category_id": 1,
                        "segmentation": rle,
                        "bbox": bbox,
                        "score": float(score),
                        "area": float(mask.sum()),
                    })

            elif eval_mode == "crops_1024":
                # 4 non-overlapping 1024x1024 crops
                crops_coords = [(0, 0), (0, 1024), (1024, 0), (1024, 1024)]
                for r0, c0 in crops_coords:
                    r1, c1 = min(r0 + 1024, H), min(c0 + 1024, W)
                    crop_img = img_np[r0:r1, c0:c1]

                    if "star_convex" in model_type:
                        c_masks, c_scores = infer_star_convex_tile(model, crop_img, decode_cfg)
                    elif "tree_flow" in model_type:
                        c_masks, c_scores = infer_tree_flow_tile(model, crop_img, decode_cfg)
                    else:
                        raise ValueError(f"Unknown model type: {model_type}")

                    for c_mask, score in zip(c_masks, c_scores):
                        if c_mask.sum() < 10:
                            continue
                        full_mask = np.zeros((H, W), dtype=np.uint8)
                        full_mask[r0:r1, c0:c1] = c_mask
                        rle = coco_mask_util.encode(np.asfortranarray(full_mask))
                        rle["counts"] = rle["counts"].decode("utf-8")
                        bbox = mask_to_bbox(full_mask)
                        coco_results.append({
                            "image_id": img_id,
                            "category_id": 1,
                            "segmentation": rle,
                            "bbox": bbox,
                            "score": float(score),
                            "area": float(full_mask.sum()),
                        })

            if (idx + 1) % 25 == 0 or (idx + 1) == len(img_ids):
                elapsed = time.time() - t0
                fps = (idx + 1) / elapsed
                print(f"  [{idx+1}/{len(img_ids)}] {elapsed:.1f}s ({fps:.2f} img/s) | {len(coco_results)} predictions so far...", flush=True)

    elapsed = time.time() - t0
    print(f"\nInference complete in {elapsed:.2f}s ({len(coco_results)} total predictions). Running COCOeval...", flush=True)

    if len(coco_results) == 0:
        print("ERROR: No predictions generated!", flush=True)
        return {"error": "no_predictions"}

    # Run standard COCOeval on segmentation
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval_segm = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval_segm.params.imgIds = img_ids
    coco_eval_segm.evaluate()
    coco_eval_segm.accumulate()
    coco_eval_segm.summarize()

    # Run standard COCOeval on bbox
    coco_eval_bbox = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval_bbox.params.imgIds = img_ids
    coco_eval_bbox.evaluate()
    coco_eval_bbox.accumulate()
    coco_eval_bbox.summarize()

    segm_names = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    bbox_names = ["bbox_AP", "bbox_AP50", "bbox_AP75", "bbox_APs", "bbox_APm", "bbox_APl", "bbox_AR1", "bbox_AR10", "bbox_AR100", "bbox_ARs", "bbox_ARm", "bbox_ARl"]

    segm_stats = {f"segm_{n}": float(v * 100) for n, v in zip(segm_names, coco_eval_segm.stats)}
    bbox_stats = {n: float(v * 100) for n, v in zip(bbox_names, coco_eval_bbox.stats)}

    # Hungarian F1@IoU0.5 computation
    f1_stats = compute_f1_hungarian(coco_gt, coco_results, img_ids)

    summary = {
        "model_type": model_type,
        "split": split_name,
        "eval_mode": eval_mode,
        "num_images": len(img_ids),
        "num_predictions": len(coco_results),
        "inference_time_sec": elapsed,
        **segm_stats,
        **bbox_stats,
        **f1_stats,
    }

    print("\n" + "="*50, flush=True)
    print(f"FINAL METRIC SUMMARY: {model_type} on {split_name}", flush=True)
    print("="*50, flush=True)
    print(f"  COCO segm AP [0.50:0.95] : {summary['segm_AP']:.2f}%", flush=True)
    print(f"  COCO segm AP50           : {summary['segm_AP50']:.2f}%", flush=True)
    print(f"  COCO segm AP75           : {summary['segm_AP75']:.2f}%", flush=True)
    print(f"  COCO segm APs (small)    : {summary['segm_APs']:.2f}%", flush=True)
    print(f"  COCO segm APm (medium)   : {summary['segm_APm']:.2f}%", flush=True)
    print(f"  COCO segm APl (large)    : {summary['segm_APl']:.2f}%", flush=True)
    print(f"  COCO segm AR100          : {summary['segm_AR100']:.2f}%", flush=True)
    print(f"  Hungarian F1@0.5         : {summary['Hungarian_F1_50']:.4f}", flush=True)
    print(f"  Hungarian Precision@0.5  : {summary['Hungarian_Prec_50']:.4f}", flush=True)
    print(f"  Hungarian Recall@0.5     : {summary['Hungarian_Rec_50']:.4f}", flush=True)
    print(f"  Panoptic Quality (PQ)    : {summary['PQ']:.4f}", flush=True)
    print("="*50 + "\n", flush=True)

    return summary


def compute_f1_hungarian(coco_gt: COCO, results: list[dict], img_ids: list[int]) -> dict[str, float]:
    pred_by_img: dict[int, list] = {}
    for r in results:
        pred_by_img.setdefault(r["image_id"], []).append(r)

    tp_50 = fp_50 = fn_50 = 0
    matched_ious = []

    for img_id in img_ids:
        gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        gt_anns = coco_gt.loadAnns(gt_ann_ids)
        preds = pred_by_img.get(img_id, [])

        if len(gt_anns) == 0 and len(preds) == 0:
            continue
        if len(gt_anns) == 0:
            fp_50 += len(preds)
            continue
        if len(preds) == 0:
            fn_50 += len(gt_anns)
            continue

        img_info = coco_gt.loadImgs(img_id)[0]
        H, W = img_info["height"], img_info["width"]

        gt_rles = []
        for ann in gt_anns:
            seg = ann["segmentation"]
            if isinstance(seg, dict):
                gt_rles.append(seg)
            else:
                rle = coco_mask_util.frPyObjects(seg, H, W)
                gt_rles.append(coco_mask_util.merge(rle))

        pr_rles = [p["segmentation"] for p in preds]

        # Fast C-level IoU matrix shape: (len(gt_rles), len(pr_rles))
        iou_mat = coco_mask_util.iou(gt_rles, pr_rles, [0] * len(pr_rles))

        row_ind, col_ind = linear_sum_assignment(-iou_mat)
        matched = 0
        for r, c in zip(row_ind, col_ind):
            iou = float(iou_mat[r, c])
            if iou >= 0.50:
                matched += 1
                matched_ious.append(iou)

        tp_50 += matched
        fp_50 += (len(preds) - matched)
        fn_50 += (len(gt_anns) - matched)

    prec = tp_50 / max(tp_50 + fp_50, 1)
    rec = tp_50 / max(tp_50 + fn_50, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)

    # PQ
    sq = float(np.mean(matched_ious)) if len(matched_ious) > 0 else 0.0
    rq = tp_50 / max(tp_50 + 0.5 * fp_50 + 0.5 * fn_50, 1)
    pq = sq * rq

    return {
        "Hungarian_F1_50": float(f1),
        "Hungarian_Prec_50": float(prec),
        "Hungarian_Rec_50": float(rec),
        "PQ": float(pq),
        "SQ": float(sq),
        "RQ": float(rq),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ITC Models on BAMFORESTS Official Benchmark")
    parser.add_argument("--model", type=str, choices=["star_convex_v8", "star_convex_v6", "star_convex_v7", "tree_flow"], default="star_convex_v8")
    parser.add_argument("--ckpt", type=str, default=None, help="Custom checkpoint path")
    parser.add_argument("--split", type=str, choices=["eval", "test1", "test2", "train"], default="eval")
    parser.add_argument("--eval_mode", type=str, choices=["full_2048", "crops_1024"], default="full_2048")
    parser.add_argument("--max_images", type=int, default=None, help="Limit number of images for quick testing")
    parser.add_argument("--out_json", type=str, default=None, help="Path to save summary JSON")
    args = parser.parse_args()

    # Default checkpoints
    default_ckpts = {
        "star_convex_v8": "crown_segmentation_research/experiments/star_convex_screen_v8/star_convex_screen.pth",
        "star_convex_v6": "crown_segmentation_research/experiments/star_convex_screen_v6/star_convex_screen.pth",
        "star_convex_v7": "crown_segmentation_research/experiments/star_convex_screen_v7/star_convex_screen.pth",
        "tree_flow": "crown_segmentation_research/experiments/tree_flow_exhaustive_v1/best_model.pth",
    }

    ckpt_path = args.ckpt or default_ckpts[args.model]

    # Decode configs
    decode_configs = {
        "star_convex_v8": {
            "prob_threshold": 0.5,
            "min_peak_distance": 5,
            "nms_iou_threshold": 0.2,
            "canopy_threshold": 0.7,
            "embedding_delta_d": 3.0,
        },
        "star_convex_v6": {
            "prob_threshold": 0.5,
            "min_peak_distance": 5,
            "nms_iou_threshold": 0.2,
            "canopy_threshold": 0.7,
        },
        "star_convex_v7": {
            "prob_threshold": 0.5,
            "min_peak_distance": 5,
            "nms_iou_threshold": 0.2,
            "canopy_threshold": 0.7,
            "embedding_delta_d": 1.5,
        },
        "tree_flow": {
            "sdt_threshold": 0.0,
            "centroid_threshold": 0.3,
            "flow_steps": 10,
            "step_size": 1.0,
            "min_crown_area": 30,
        },
    }

    if "star_convex" in args.model:
        model = load_star_convex_model(ckpt_path, model_type=args.model)
    elif "tree_flow" in args.model:
        model = load_tree_flow_model(ckpt_path)
    else:
        raise ValueError(f"Unknown model {args.model}")

    cfg = decode_configs[args.model]
    results = run_split_evaluation(
        model=model,
        model_type=args.model,
        split_name=args.split,
        decode_cfg=cfg,
        max_images=args.max_images,
        eval_mode=args.eval_mode,
    )

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
