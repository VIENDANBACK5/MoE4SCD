"""
Fast Decoder Hyperparameter Sweep on BAM Benchmark for TreeFlowNet.

Caches raw predictions from a TreeFlowNet checkpoint on BAM eval images,
then performs a rapid grid sweep over:
  - centroid_thresh: peak detector sensitivity
  - canopy_thresh: binary canopy gate threshold
  - sdt_thresh: signed distance threshold for crown seeds
  - nms_iou: Non-Maximum Suppression IoU threshold
  - step_size (eta): flow gradient descent step size
  - n_steps: flow tracing iterations

Usage:
    python crown_segmentation_research/methods/tree_flow/sweep_bam.py \
        --ckpt crown_segmentation_research/experiments/tree_flow_exhaustive_v1/best_model.pth \
        --n-val 50 \
        --out-csv crown_segmentation_research/experiments/tree_flow_exhaustive_v1/sweep_results.csv
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as coco_mask_util
from pycocotools.coco import COCO

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet
from crown_segmentation_research.methods.tree_flow.decode import decode, DECODE_DEFAULTS

BAM_COCO_DIR  = REPO / "data/itc_benchmarks/extracted/bam_coco2048"
BAM_ARCHIVE   = REPO / "data/itc_benchmarks/raw_archives/Bamberg_coco2048.zip"
MANIFEST_PATH = BAM_COCO_DIR / "zip_manifest.json"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


def _load_manifest() -> dict[str, str]:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def _load_image_from_zip(zf: zipfile.ZipFile, file_name: str, manifest: dict[str, str]) -> np.ndarray:
    if file_name in manifest:
        try:
            data = zf.read(manifest[file_name])
            return np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        except KeyError:
            pass
    for prefix in ("", "images/", "coco2048/images/", "coco2048/val2023/", "coco2048/train2023/"):
        try:
            data = zf.read(prefix + file_name)
            return np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        except KeyError:
            continue
    raise FileNotFoundError(f"{file_name} not found in archive")


def compute_fast_f1(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray], iou_thresh: float = 0.5) -> dict[str, float]:
    n_p = len(pred_masks)
    n_g = len(gt_masks)
    if n_p == 0 or n_g == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": n_p, "fn": n_g}

    # Pre-flatten binary masks
    p_flat = np.array([m.flatten() for m in pred_masks], dtype=np.uint8)  # (P, HW)
    g_flat = np.array([m.flatten() for m in gt_masks], dtype=np.uint8)    # (G, HW)

    intersection = p_flat @ g_flat.T  # (P, G)
    p_area = p_flat.sum(axis=1, keepdims=True)  # (P, 1)
    g_area = g_flat.sum(axis=1, keepdims=True).T  # (1, G)
    union = p_area + g_area - intersection
    iou_mat = intersection / np.maximum(union, 1e-6)

    # Greedy match
    matched_gt = set()
    matched_pred = set()
    matches = 0

    # Sort candidates by IoU
    p_idx, g_idx = np.where(iou_mat >= iou_thresh)
    ious = iou_mat[p_idx, g_idx]
    sort_ord = np.argsort(-ious)

    for k in sort_ord:
        pi = p_idx[k]
        gi = g_idx[k]
        if pi not in matched_pred and gi not in matched_gt:
            matched_pred.add(pi)
            matched_gt.add(gi)
            matches += 1

    prec = matches / (n_p + 1e-6)
    rec  = matches / (n_g + 1e-6)
    f1   = 2 * prec * rec / (prec + rec + 1e-6)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1), "tp": matches, "fp": n_p - matches, "fn": n_g - matches}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="crown_segmentation_research/experiments/tree_flow_exhaustive_v1/best_model.pth")
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--out-csv", type=str, default="crown_segmentation_research/experiments/tree_flow_exhaustive_v1/sweep_results.csv")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint {ckpt_path} does not exist yet!")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    model = TreeFlowNet().to(DEVICE).eval()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd.get("model_state_dict", sd.get("state_dict", sd))
    model.load_state_dict(state, strict=True)

    json_path = BAM_COCO_DIR / "instances_tree_eval2023.json"
    print(f"Loading COCO val annotations: {json_path}")
    coco = COCO(str(json_path))
    manifest = _load_manifest()
    zf = zipfile.ZipFile(BAM_ARCHIVE, "r")

    img_ids = list(coco.imgs.keys())[:args.n_val]
    print(f"Caching forward outputs for {len(img_ids)} BAM validation images...")

    cached_samples = []
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for i, img_id in enumerate(img_ids):
        img_info = coco.imgs[img_id]
        fn = img_info["file_name"]
        img_rgb = _load_image_from_zip(zf, fn, manifest)
        H, W = img_rgb.shape[:2]

        if H > 1024 or W > 1024:
            ch, cw = min(H, 1024), min(W, 1024)
            sy = (H - ch) // 2
            sx = (W - cw) // 2
            crop_rgb = img_rgb[sy:sy+ch, sx:sx+cw]
        else:
            crop_rgb = img_rgb
            sx, sy = 0, 0
            ch, cw = H, W

        # Get GT masks
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        gt_masks = []
        for ann in anns:
            seg = ann.get("segmentation", [])
            if isinstance(seg, list) and len(seg) > 0:
                # Polygon
                m = np.zeros((H, W), dtype=np.uint8)
                for p_coords in seg:
                    coords = np.array(p_coords).reshape(-1, 2).round().astype(np.int32)
                    if len(coords) >= 3:
                        cv2.fillPoly(m, [coords], 1)
                crop_m = m[sy:sy+ch, sx:sx+cw]
                if crop_m.sum() >= 20:
                    gt_masks.append(crop_m.astype(bool))
            elif isinstance(seg, dict):
                m = coco_mask_util.decode(seg)
                crop_m = m[sy:sy+ch, sx:sx+cw]
                if crop_m.sum() >= 20:
                    gt_masks.append(crop_m.astype(bool))

        norm = ((crop_rgb.astype(np.float32) / 255.0) - mean) / std
        t_img = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
        pad_h = (32 - ch % 32) % 32
        pad_w = (32 - cw % 32) % 32
        if pad_h > 0 or pad_w > 0:
            t_img = F.pad(t_img, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            preds = model(t_img)

        flow_t = preds["flow"][:, :, :ch, :cw].cpu()
        sdt_t  = preds["sdt"][:, :, :ch, :cw].cpu()
        cent_t = preds["centroid"][:, :, :ch, :cw].cpu()
        can_t  = preds["canopy"][:, :, :ch, :cw].cpu()

        cached_samples.append({
            "img_id": img_id,
            "flow": flow_t,
            "sdt": sdt_t,
            "centroid": cent_t,
            "canopy": can_t,
            "gt_masks": gt_masks,
        })

    zf.close()
    print(f"Forward passes cached. Total validation GT crowns: {sum(len(s['gt_masks']) for s in cached_samples)}")

    # ── Grid definition ───────────────────────────────────────────────────────
    canopy_thresholds   = [0.25, 0.35, 0.50]
    sdt_thresholds      = [-0.6, -0.4, -0.2, 0.0]
    centroid_thresholds = [0.15, 0.25, 0.35]
    nms_ious            = [0.20, 0.30, 0.40]
    n_steps_list        = [20, 30]

    grid = list(itertools.product(canopy_thresholds, sdt_thresholds, centroid_thresholds, nms_ious, n_steps_list))
    print(f"Sweeping {len(grid)} parameter combinations...")

    results = []
    t_start = time.time()

    for idx, (can_th, sdt_th, cent_th, nms_th, n_steps) in enumerate(grid):
        total_tp, total_fp, total_fn = 0, 0, 0

        for sample in cached_samples:
            # Decode on GPU/CPU
            f_t = sample["flow"].to(DEVICE)
            s_t = sample["sdt"].to(DEVICE)
            c_t = sample["centroid"].to(DEVICE)
            k_t = sample["canopy"].to(DEVICE)

            insts = decode(
                f_t, s_t, c_t, k_t,
                centroid_thresh = cent_th,
                canopy_thresh   = can_th,
                sdt_thresh      = sdt_th,
                nms_iou         = nms_th,
                min_area        = 30,
                n_steps         = n_steps,
                eta             = 1.0,
            )

            p_masks = [inst["mask"] for inst in insts]
            g_masks = sample["gt_masks"]

            metrics = compute_fast_f1(p_masks, g_masks, iou_thresh=0.5)
            total_tp += metrics["tp"]
            total_fp += metrics["fp"]
            total_fn += metrics["fn"]

        prec = total_tp / (total_tp + total_fp + 1e-6)
        rec  = total_tp / (total_tp + total_fn + 1e-6)
        f1   = 2 * prec * rec / (prec + rec + 1e-6)

        results.append({
            "canopy_thresh": can_th,
            "sdt_thresh": sdt_th,
            "centroid_thresh": cent_th,
            "nms_iou": nms_th,
            "n_steps": n_steps,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        })

        if (idx + 1) % 25 == 0 or idx == len(grid) - 1:
            print(f"  [{idx+1:3d}/{len(grid)}] Best F1 so far: {max(r['f1'] for r in results):.4f} "
                  f"(current: F1={f1:.4f} P={prec:.3f} R={rec:.3f})")

    df = pd.DataFrame(results).sort_values("f1", ascending=False)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nTop 5 Decoder Configurations saved to {out_csv}:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
