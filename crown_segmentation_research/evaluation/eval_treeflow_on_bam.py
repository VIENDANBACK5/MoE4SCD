"""
Step 0: Evaluate existing TreeFlowNet ep39 checkpoint on BAM COCO AP metric.

This is the baseline measurement for exhaustive all-tree segmentation.
Runs on BAM evalhold60 (382 images, COCO JSON) and reports:
  - COCO AP [0.5:0.95]
  - AP50, AP75
  - F1@IoU0.5 (for direct comparison with DeadTrees results)
  - Recall by crown size (small / medium / large)

Usage:
    python crown_segmentation_research/evaluation/eval_treeflow_on_bam.py \
        --ckpt DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth \
        --split eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as coco_mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ── path setup ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet
from crown_segmentation_research.methods.tree_flow.decode import decode, DECODE_DEFAULTS

# ── constants ──────────────────────────────────────────────────────────────────
BAM_COCO_DIR  = REPO / "data/itc_benchmarks/extracted/bam_coco2048"
BAM_ARCHIVE   = REPO / "data/itc_benchmarks/raw_archives/Bamberg_coco2048.zip"
SPLIT_JSON = {
    "eval":  "instances_tree_eval2023.json",
    "test1": "instances_tree_TestSet12023.json",
    "test2": "instances_tree_TestSet22023.json",
    "train": "instances_tree_train2023.json",
}
CROP_SIZE = 1024  # evaluate on 1024×1024 crops (consistent with crown_research)
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


# ── image loading from zip ──────────────────────────────────────────────────────

def _load_image_from_zip(
    zf: zipfile.ZipFile,
    file_name: str,
    manifest: dict[str, str] | None = None,
) -> np.ndarray:
    """Load a TIFF from the BAM zip archive as uint8 RGB numpy array."""
    import io
    if manifest and file_name in manifest:
        try:
            data = zf.read(manifest[file_name])
            img = Image.open(io.BytesIO(data)).convert("RGB")
            return np.array(img)
        except KeyError:
            pass
    # Try a few common path layouts inside the zip
    for prefix in ("", "images/", "coco2048/images/", "coco2048/val2023/", "coco2048/test2023/Test-Set-1/", "coco2048/test2023/Test-Set-2/", "coco2048/train2023/"):
        try:
            data = zf.read(prefix + file_name)
            img = Image.open(io.BytesIO(data)).convert("RGB")
            return np.array(img)
        except KeyError:
            continue
    raise FileNotFoundError(f"{file_name} not found in archive")


# ── model loading ──────────────────────────────────────────────────────────────

def load_model(ckpt_path: str) -> TreeFlowNet:
    model = TreeFlowNet()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Handle various checkpoint formats
    state = sd.get("model_state_dict", sd.get("state_dict", sd))
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    print(f"  Loaded checkpoint: {ckpt_path}")
    return model


# ── target computation (flow, sdt, centroid, canopy from COCO segmentation) ──────

def _rle_to_mask(rle_or_poly, H: int, W: int) -> np.ndarray:
    """Convert COCO segmentation (RLE or polygon) to binary mask."""
    if isinstance(rle_or_poly, dict):
        # RLE
        return coco_mask_util.decode(rle_or_poly).astype(bool)
    else:
        # Polygon list
        rle = coco_mask_util.frPyObjects(rle_or_poly, H, W)
        rle = coco_mask_util.merge(rle)
        return coco_mask_util.decode(rle).astype(bool)


def compute_bam_targets(
    anns: list[dict],
    H: int,
    W: int,
    sigma_fraction: float = 0.08,   # Gaussian sigma = sigma_fraction * sqrt(area)
) -> dict[str, np.ndarray]:
    """
    Build flow/SDT/centroid/canopy targets from COCO instance annotations.
    Returns dict of float32 arrays shape [H, W] (single image).
    """
    from scipy.ndimage import distance_transform_edt

    canopy  = np.zeros((H, W), dtype=np.float32)
    sdt_out = np.full((H, W), -1.0, dtype=np.float32)
    centroid_heat = np.zeros((H, W), dtype=np.float32)
    flow_y  = np.zeros((H, W), dtype=np.float32)
    flow_x  = np.zeros((H, W), dtype=np.float32)

    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)

    for ann in anns:
        mask = _rle_to_mask(ann["segmentation"], H, W)
        if mask.sum() == 0:
            continue

        canopy[mask] = 1.0

        # Signed distance transform for this instance
        dist_in  = distance_transform_edt(mask)
        dist_out = distance_transform_edt(~mask)
        max_d = dist_in.max() + 1e-6
        sdt_inst = (dist_in - dist_out) / max_d  # [-1, 1]
        # Take max of existing SDT and this instance's (handles overlaps)
        sdt_out = np.maximum(sdt_out, sdt_inst)

        # Centroid heatmap
        cy, cx = ann.get("centroid", (None, None))
        if cy is None:
            # Compute from mask centroid
            rows, cols = np.where(mask)
            cy, cx = rows.mean(), cols.mean()
        area = mask.sum()
        sigma = max(sigma_fraction * np.sqrt(area), 3.0)
        dist_sq = (ys - cy) ** 2 + (xs - cx) ** 2
        gauss = np.exp(-dist_sq / (2 * sigma ** 2))
        centroid_heat = np.maximum(centroid_heat, gauss * mask)

        # Flow: unit vector toward centroid (only inside crown)
        dy = cy - ys
        dx = cx - xs
        mag = np.sqrt(dy ** 2 + dx ** 2 + 1e-6)
        flow_y[mask] = (dy / mag)[mask]
        flow_x[mask] = (dx / mag)[mask]

    flow = np.stack([flow_y, flow_x], axis=0)  # [2, H, W]
    fg_mask = canopy > 0.5

    return {
        "flow":     flow,                   # [2, H, W]
        "sdt":      sdt_out[None],          # [1, H, W]
        "centroid": centroid_heat[None],    # [1, H, W]
        "canopy":   canopy[None],           # [1, H, W]
        "fg_mask":  fg_mask[None].astype(np.float32),  # [1, H, W]
    }


# ── COCO AP evaluation ──────────────────────────────────────────────────────────

def masks_to_coco_results(
    instances: list[dict],
    image_id: int,
) -> list[dict]:
    """Convert decoded instances to COCO-format result dicts."""
    results = []
    for inst in instances:
        mask = inst["mask"].astype(np.uint8)
        rle  = coco_mask_util.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("utf-8")
        r, c, h, w = inst["bbox"]
        results.append({
            "image_id":    image_id,
            "category_id": 1,
            "segmentation": rle,
            "bbox":        [float(c), float(r), float(w), float(h)],
            "score":       float(inst["score"]) if inst["score"] > 0 else 0.5,
            "area":        float(inst["area"]),
        })
    return results


# ── main evaluation loop ────────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(
    model:       TreeFlowNet,
    coco_gt:     COCO,
    zf:          zipfile.ZipFile,
    decode_cfg:  dict,
    crop_size:   int = CROP_SIZE,
    manifest:    dict[str, str] | None = None,
    max_images:  int | None = None,
) -> list[dict]:
    """Run inference on all images in coco_gt and return COCO result list."""
    all_results = []
    img_ids = coco_gt.getImgIds()
    if max_images is not None:
        img_ids = img_ids[:max_images]

    t0 = time.time()
    for i, img_id in enumerate(img_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        file_name = img_info["file_name"]

        # Load image
        try:
            img_np = _load_image_from_zip(zf, file_name, manifest=manifest)
        except FileNotFoundError as e:
            print(f"  [WARN] {e}")
            continue

        H_orig, W_orig = img_np.shape[:2]

        # Resize to crop_size if needed (preserve aspect)
        if H_orig != crop_size or W_orig != crop_size:
            img_pil = Image.fromarray(img_np).resize((crop_size, crop_size), Image.BILINEAR)
            img_np = np.array(img_pil)

        # To tensor, normalize ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - mean) / std
        img_t = img_t.unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

        # Forward
        outputs = model(img_t)

        # Squeeze batch
        outputs_sq = {k: v[0] for k, v in outputs.items()}

        # Decode instances
        instances = decode(outputs_sq, **decode_cfg)

        # Scale bboxes/masks back to original resolution if we resized
        if H_orig != crop_size:
            scale_y = H_orig / crop_size
            scale_x = W_orig / crop_size
            for inst in instances:
                r, c, h, w = inst["bbox"]
                inst["bbox"] = (r * scale_y, c * scale_x, h * scale_y, w * scale_x)
                # Upsample mask
                m = torch.from_numpy(inst["mask"]).float().unsqueeze(0).unsqueeze(0)
                m = F.interpolate(m, size=(H_orig, W_orig), mode="nearest")
                inst["mask"] = m.squeeze().numpy().astype(bool)
                inst["area"] = int(inst["mask"].sum())

        # Convert to COCO format
        results = masks_to_coco_results(instances, img_id)
        all_results.extend(results)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(img_ids)}] {elapsed:.0f}s elapsed, {len(all_results)} total predictions")

    return all_results


def evaluate_coco_ap(coco_gt: COCO, results: list[dict]) -> dict[str, float]:
    """Compute COCO AP [0.5:0.95], AP50, AP75, APs, APm, APl."""
    if len(results) == 0:
        print("  [WARN] No predictions — returning zeros")
        return {"AP": 0, "AP50": 0, "AP75": 0, "APs": 0, "APm": 0, "APl": 0}

    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "segm")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    names = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    return {n: float(v * 100) for n, v in zip(names, ev.stats)}


def compute_f1_iou50(coco_gt: COCO, results: list[dict]) -> dict[str, float]:
    """Simple F1@IoU0.5 by Hungarian matching (for comparison with DeadTrees F1)."""
    from scipy.optimize import linear_sum_assignment

    img_ids = coco_gt.getImgIds()
    pred_by_img: dict[int, list] = {}
    for r in results:
        pred_by_img.setdefault(r["image_id"], []).append(r)

    tp = fp = fn = 0
    for img_id in img_ids:
        gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        gt_anns = coco_gt.loadAnns(gt_ann_ids)
        preds = pred_by_img.get(img_id, [])

        if len(gt_anns) == 0 and len(preds) == 0:
            continue
        fn += len(gt_anns)
        fp += len(preds)

        if len(gt_anns) == 0 or len(preds) == 0:
            continue

        img_info = coco_gt.loadImgs(img_id)[0]
        H, W = img_info["height"], img_info["width"]

        # Build GT masks
        gt_masks = []
        for ann in gt_anns:
            gt_masks.append(_rle_to_mask(ann["segmentation"], H, W))

        # Build pred masks
        pr_masks = []
        for p in preds:
            pr_masks.append(_rle_to_mask(p["segmentation"], H, W))

        # IoU matrix
        iou_mat = np.zeros((len(gt_masks), len(pr_masks)), dtype=np.float32)
        for gi, gm in enumerate(gt_masks):
            for pi, pm in enumerate(pr_masks):
                inter = (gm & pm).sum()
                union = (gm | pm).sum()
                iou_mat[gi, pi] = inter / (union + 1e-6)

        # Hungarian matching at IoU ≥ 0.5
        row_ind, col_ind = linear_sum_assignment(-iou_mat)
        matched = sum(1 for r, c in zip(row_ind, col_ind) if iou_mat[r, c] >= 0.5)

        tp += matched
        fp -= matched
        fn -= matched

    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    f1   = 2 * prec * rec / (prec + rec + 1e-6)
    return {"F1@0.5": f1, "Precision": prec, "Recall": rec, "TP": tp, "FP": fp, "FN": fn}


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to TreeFlowNet checkpoint (.pth)")
    p.add_argument("--split", default="eval",
                   choices=["eval", "test1", "test2", "train"],
                   help="Which BAM split to evaluate on")
    p.add_argument("--max-images", type=int, default=None,
                   help="Limit number of images (for quick sanity check)")
    # Decode params
    p.add_argument("--canopy-thresh", type=float, default=0.40)
    p.add_argument("--sdt-thresh",    type=float, default=-0.50)
    p.add_argument("--nms-iou",       type=float, default=0.30)
    p.add_argument("--min-area",      type=int,   default=30)
    p.add_argument("--n-steps",       type=int,   default=30)
    p.add_argument("--out", default=None, help="Save results JSON to this path")
    return p.parse_args()


def main():
    args = parse_args()

    decode_cfg = dict(
        canopy_thresh=args.canopy_thresh,
        sdt_thresh=args.sdt_thresh,
        nms_iou=args.nms_iou,
        min_area=args.min_area,
        n_steps=args.n_steps,
        eta=1.0,
        device=DEVICE,
    )

    print(f"\n{'='*70}")
    print(f"TreeFlowNet — BAM Evaluation (Step 0 Baseline)")
    print(f"  checkpoint : {args.ckpt}")
    print(f"  split      : {args.split}")
    print(f"  device     : {DEVICE}")
    print(f"  decode     : {decode_cfg}")
    print(f"{'='*70}\n")

    # Load model
    model = load_model(args.ckpt)

    # Load COCO GT
    json_path = BAM_COCO_DIR / SPLIT_JSON[args.split]
    if not json_path.exists():
        raise FileNotFoundError(f"BAM JSON not found: {json_path}")
    print(f"Loading GT from {json_path} ...")
    coco_gt = COCO(str(json_path))
    n_imgs = len(coco_gt.getImgIds())
    n_anns = len(coco_gt.getAnnIds())
    print(f"  {n_imgs} images, {n_anns} annotations")

    # Load zip manifest if available
    manifest_path = BAM_COCO_DIR / "zip_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest:
        print(f"  Loaded manifest with {len(manifest)} image paths")

    # Open BAM archive
    if not BAM_ARCHIVE.exists():
        raise FileNotFoundError(f"BAM archive not found: {BAM_ARCHIVE}")
    print(f"Opening archive {BAM_ARCHIVE.name} ...")
    with zipfile.ZipFile(BAM_ARCHIVE, "r") as zf:
        results = run_eval(
            model, coco_gt, zf, decode_cfg,
            manifest=manifest,
            max_images=args.max_images,
        )

    print(f"\nTotal predictions: {len(results)}")

    # COCO AP
    print("\n── COCO AP ─────────────────────────────────────────────────────────")
    ap_metrics = evaluate_coco_ap(coco_gt, results)
    for k, v in ap_metrics.items():
        print(f"  {k:6s}: {v:.2f}")

    # F1@0.5
    print("\n── F1@IoU0.5 ───────────────────────────────────────────────────────")
    f1_metrics = compute_f1_iou50(coco_gt, results)
    for k, v in f1_metrics.items():
        if isinstance(v, float):
            print(f"  {k:12s}: {v:.4f}")
        else:
            print(f"  {k:12s}: {v}")

    # Summary
    out = {
        "checkpoint":    args.ckpt,
        "split":         args.split,
        "n_images":      n_imgs,
        "n_gt_anns":     n_anns,
        "n_predictions": len(results),
        "decode_cfg":    decode_cfg,
        "coco_ap":       ap_metrics,
        "f1_metrics":    f1_metrics,
    }

    out_path = args.out or f"DeadTrees/experiments/treeflow_bam_{args.split}_eval.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")

    print(f"\n{'='*70}")
    print(f"SUMMARY [{args.split}]:")
    print(f"  COCO AP      = {ap_metrics['AP']:.2f}")
    print(f"  AP50         = {ap_metrics['AP50']:.2f}")
    print(f"  AP75         = {ap_metrics['AP75']:.2f}")
    print(f"  F1@IoU0.5    = {f1_metrics['F1@0.5']:.4f}")
    print(f"  Recall@0.5   = {f1_metrics['Recall']:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
