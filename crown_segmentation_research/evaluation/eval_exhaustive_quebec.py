"""
Exhaustive Ground Truth Evaluation Benchmark on Quebec 2021 Individual Tree Crown Dataset.

Evaluates TreeFlowNet (and classical baselines) on 100% EXHAUSTIVELY ANNOTATED
boreal forest plots where every single canopy tree is delineated with zero omission.

Benchmarks:
  - Quebec Zone 3 (Test Split): 200 patches (512x512), 4,645 fully labeled crowns (100% exhaustive GT)
  - Quebec Zone 2 (Val Split):  200 patches (512x512)

Metrics:
  - COCO Mask AP [0.50:0.95], AP50, AP75, APs, APm, APl
  - COCO Mask AR [0.50:0.95], AR1, AR10, AR100, ARs, ARm, ARl
  - Hungarian Matched Precision, Recall, F1@IoU0.50, F1@IoU0.75
  - Panoptic Quality (PQ), Segmentation Quality (SQ), Recognition Quality (RQ)
  - All-Tree Counting Metrics: Mean Absolute Error (MAE), Omission Rate (FN/GT), Commission Rate (FP/Pred)

Usage:
  python crown_segmentation_research/evaluation/eval_exhaustive_quebec.py \
      --ckpt crown_segmentation_research/experiments/tree_flow_exhaustive_v1/best_model.pth \
      --split test \
      --out crown_segmentation_research/experiments/quebec_exhaustive_eval_results.json \
      --save-vis
"""
from __future__ import annotations

import os
import sys
# Remove user site-packages to avoid torch/torchvision ABI incompatibility
sys.path = [p for p in sys.path if not p.startswith('/home/chung/.local')]

import argparse
import glob
import json
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as coco_mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment
from scipy import ndimage
from skimage import morphology, feature, segmentation

# Ensure workspace root is in sys.path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet
from crown_segmentation_research.methods.tree_flow.decode import decode, decode_organic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUEBEC_NPZ_DIR = REPO / "data/itc_benchmarks/extracted/quebec_cached_npz"


def load_model(ckpt_path: str | Path) -> TreeFlowNet:
    model = TreeFlowNet().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def get_patch_files(split: str = "test") -> list[Path]:
    if not QUEBEC_NPZ_DIR.exists():
        raise FileNotFoundError(f"Quebec cached NPZ directory not found at {QUEBEC_NPZ_DIR}")
    
    all_files = sorted(QUEBEC_NPZ_DIR.glob("*.npz"))
    if split == "test":
        files = [f for f in all_files if "_z3_" in f.name]
    elif split == "val":
        files = [f for f in all_files if "_z2_" in f.name]
    elif split == "train":
        files = [f for f in all_files if "_z1_" in f.name]
    elif split == "all":
        files = all_files
    else:
        raise ValueError(f"Unknown split: {split}. Choose test, val, train, or all.")
    return files


def run_baseline_watershed(img_np: np.ndarray) -> list[dict[str, Any]]:
    """Classical Canopy Distance Transform Watershed Baseline."""
    # Simple excess green index / grayscale threshold for canopy
    r, g, b = img_np[:, :, 0].astype(float), img_np[:, :, 1].astype(float), img_np[:, :, 2].astype(float)
    exg = 2 * g - r - b
    canopy_mask = (exg > 0) & (g > 30)
    
    if canopy_mask.sum() < 20:
        return []
    
    distance = ndimage.distance_transform_edt(canopy_mask)
    coords = feature.peak_local_max(distance, min_distance=7, labels=canopy_mask)
    mask_peaks = np.zeros(distance.shape, dtype=bool)
    if len(coords) > 0:
        mask_peaks[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask_peaks)
    labels = segmentation.watershed(-distance, markers, mask=canopy_mask)
    
    instances = []
    for uid in np.unique(labels):
        if uid == 0:
            continue
        m = (labels == uid)
        if m.sum() < 15:
            continue
        instances.append({
            "mask": m,
            "score": float(distance[m].max() / (distance.max() + 1e-6)),
            "area": int(m.sum()),
        })
    return instances


def evaluate_dataset(
    model: TreeFlowNet | None,
    files: list[Path],
    baseline: bool = False,
    decoder_type: str = "organic",
    decode_cfg: dict | None = None,
) -> dict[str, Any]:
    if decode_cfg is None:
        decode_cfg = {
            "canopy_thresh": 0.30,
            "sdt_thresh": -0.15,
            "min_sink_dist": 44,
            "min_area": 150,
            "n_steps": 30,
            "eta": 1.0,
            "device": str(DEVICE),
        }

    coco_gt_dict: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "tree"}],
    }
    coco_results: list[dict[str, Any]] = []

    ann_id_counter = 1
    total_gt_crowns = 0
    total_pred_crowns = 0
    patch_gt_counts = []
    patch_pred_counts = []

    t0 = time.time()
    tag = "Baseline Watershed" if baseline else f"TreeFlowNet ({decoder_type.upper()})"
    print(f"\nProcessing {len(files)} patches ({tag})...", flush=True)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for img_idx, fpath in enumerate(files):
        data = np.load(fpath)
        img_np = data["image"]  # [H, W, 3] uint8
        label_map = data["instance_label"]  # [H, W] int32
        H, W = img_np.shape[:2]

        image_id = img_idx + 1
        coco_gt_dict["images"].append({
            "id": image_id,
            "file_name": fpath.name,
            "height": H,
            "width": W,
        })

        # Ground Truth Instances
        gt_uids = np.unique(label_map)
        gt_uids = gt_uids[gt_uids > 0]
        n_gt = 0
        for uid in gt_uids:
            m = (label_map == uid)
            if m.sum() < 5:
                continue
            rle = coco_mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            bbox = [
                float(np.where(m)[1].min()),
                float(np.where(m)[0].min()),
                float(np.where(m)[1].max() - np.where(m)[1].min() + 1),
                float(np.where(m)[0].max() - np.where(m)[0].min() + 1),
            ]
            coco_gt_dict["annotations"].append({
                "id": ann_id_counter,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": rle,
                "area": float(m.sum()),
                "bbox": bbox,
                "iscrowd": 0,
            })
            ann_id_counter += 1
            n_gt += 1

        total_gt_crowns += n_gt
        patch_gt_counts.append(n_gt)

        # Model Inference
        if baseline:
            instances = run_baseline_watershed(img_np)
        else:
            norm = ((img_np.astype(np.float32) / 255.0) - mean) / std
            t_img = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                preds = model(t_img)
                preds_sq = {k: v[0] for k, v in preds.items()}
                if decoder_type == "organic":
                    organic_kwargs = {
                        "min_sink_dist": 44,
                        "canopy_thresh": 0.30,
                        "sdt_thresh": -0.15,
                        "min_area": 150,
                        "device": str(DEVICE),
                    }
                    if decode_cfg:
                        for k in ["min_sink_dist", "canopy_thresh", "sdt_thresh", "min_area"]:
                            if k in decode_cfg:
                                organic_kwargs[k] = decode_cfg[k]
                    instances = decode_organic(
                        preds_sq,
                        img_rgb=img_np,
                        **organic_kwargs,
                    )
                else:
                    instances = decode(preds_sq, **decode_cfg)

        n_pred = len(instances)
        total_pred_crowns += n_pred
        patch_pred_counts.append(n_pred)

        for inst in instances:
            m = inst["mask"]
            if m.sum() < 5:
                continue
            rle = coco_mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            coco_results.append({
                "image_id": image_id,
                "category_id": 1,
                "segmentation": rle,
                "score": float(inst.get("score", 0.9)),
            })

        if (img_idx + 1) % 50 == 0 or (img_idx + 1) == len(files):
            print(f"  [{img_idx + 1:3d}/{len(files)}] GT crowns: {total_gt_crowns:5d} | Predicted crowns: {total_pred_crowns:5d} ({time.time() - t0:.1f}s)", flush=True)

    elapsed = time.time() - t0

    # Build pycocotools COCO objects
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json") as f_gt:
        json.dump(coco_gt_dict, f_gt)
        f_gt.flush()
        coco_gt = COCO(f_gt.name)

    if len(coco_results) == 0:
        print("WARNING: Zero predictions generated!")
        return {"error": "Zero predictions"}

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    segm_names = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    segm_stats = {f"segm_{n}": float(v * 100) for n, v in zip(segm_names, coco_eval.stats)}

    # Hungarian Matching & Panoptic Quality
    img_ids = [img["id"] for img in coco_gt_dict["images"]]
    f1_stats = compute_f1_hungarian(coco_gt, coco_results, img_ids)

    # Counting Metrics
    count_errors = np.abs(np.array(patch_pred_counts) - np.array(patch_gt_counts))
    mae_count = float(np.mean(count_errors))
    mean_gt_count = float(np.mean(patch_gt_counts))
    mean_pred_count = float(np.mean(patch_pred_counts))

    results_summary = {
        "method": "Baseline Watershed" if baseline else "TreeFlowNet (Ours)",
        "num_patches": len(files),
        "total_gt_crowns": total_gt_crowns,
        "total_pred_crowns": total_pred_crowns,
        "mean_gt_per_patch": mean_gt_count,
        "mean_pred_per_patch": mean_pred_count,
        "mae_count": mae_count,
        "inference_time_sec": elapsed,
        "latency_per_patch_ms": (elapsed / len(files)) * 1000,
        **segm_stats,
        **f1_stats,
    }
    return results_summary


def compute_f1_hungarian(coco_gt: COCO, results: list[dict], img_ids: list[int]) -> dict[str, float]:
    pred_by_img: dict[int, list] = {}
    for r in results:
        pred_by_img.setdefault(r["image_id"], []).append(r)

    tp_50 = fp_50 = fn_50 = 0
    tp_75 = fp_75 = fn_75 = 0
    matched_ious = []

    for img_id in img_ids:
        gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        gt_anns = coco_gt.loadAnns(gt_ann_ids)
        preds = pred_by_img.get(img_id, [])

        if len(gt_anns) == 0 and len(preds) == 0:
            continue
        if len(gt_anns) == 0:
            fp_50 += len(preds)
            fp_75 += len(preds)
            continue
        if len(preds) == 0:
            fn_50 += len(gt_anns)
            fn_75 += len(gt_anns)
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
        matched_50 = 0
        matched_75 = 0
        for r, c in zip(row_ind, col_ind):
            iou = float(iou_mat[r, c])
            if iou >= 0.50:
                matched_50 += 1
                matched_ious.append(iou)
            if iou >= 0.75:
                matched_75 += 1

        tp_50 += matched_50
        fp_50 += (len(preds) - matched_50)
        fn_50 += (len(gt_anns) - matched_50)

        tp_75 += matched_75
        fp_75 += (len(preds) - matched_75)
        fn_75 += (len(gt_anns) - matched_75)

    prec_50 = tp_50 / max(tp_50 + fp_50, 1)
    rec_50  = tp_50 / max(tp_50 + fn_50, 1)
    f1_50   = 2 * prec_50 * rec_50 / max(prec_50 + rec_50, 1e-6)

    prec_75 = tp_75 / max(tp_75 + fp_75, 1)
    rec_75  = tp_75 / max(tp_75 + fn_75, 1)
    f1_75   = 2 * prec_75 * rec_75 / max(prec_75 + rec_75, 1e-6)

    # Panoptic Quality
    sq = float(np.mean(matched_ious)) if len(matched_ious) > 0 else 0.0
    rq = float(tp_50 / max(tp_50 + 0.5 * fp_50 + 0.5 * fn_50, 1))
    pq = float(sq * rq)

    # Omission & Commission rates
    total_gt = tp_50 + fn_50
    total_pred = tp_50 + fp_50
    omission_rate = float(fn_50 / max(total_gt, 1))
    commission_rate = float(fp_50 / max(total_pred, 1))

    return {
        "Hungarian_F1_50": float(f1_50),
        "Hungarian_Prec_50": float(prec_50),
        "Hungarian_Rec_50": float(rec_50),
        "Hungarian_F1_75": float(f1_75),
        "Hungarian_Prec_75": float(prec_75),
        "Hungarian_Rec_75": float(rec_75),
        "PQ": pq,
        "SQ": sq,
        "RQ": rq,
        "omission_rate": omission_rate,
        "commission_rate": commission_rate,
    }


def generate_visual_comparisons(
    model: TreeFlowNet,
    files: list[Path],
    out_dir: Path,
    num_samples: int = 5,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Pick samples with high crown density
    sample_candidates = []
    for f in files:
        data = np.load(f)
        n = len(np.unique(data["instance_label"])) - 1
        sample_candidates.append((n, f))
    sample_candidates.sort(reverse=True)

    selected_files = [f for _, f in sample_candidates[:num_samples]]
    print(f"\nGenerating {len(selected_files)} high-resolution visual verification figures...", flush=True)

    import matplotlib
    try:
        cmap = matplotlib.colormaps["tab20"].resampled(50)
        colors = [cmap(i) for i in range(50)]
    except Exception:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(50)]

    for idx, fpath in enumerate(selected_files):
        data = np.load(fpath)
        img_np = data["image"]
        label_map = data["instance_label"]
        H, W = img_np.shape[:2]

        norm = ((img_np.astype(np.float32) / 255.0) - mean) / std
        t_img = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            preds = model(t_img)
            preds_sq = {k: v[0] for k, v in preds.items()}
            insts = decode_organic(
                preds_sq,
                img_rgb=img_np,
                min_sink_dist=44,
                canopy_thresh=0.30,
                sdt_thresh=-0.15,
                min_area=150,
                device=str(DEVICE),
            )

        # Render 6-panel figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
        axes = axes.flatten()

        # Panel 1: RGB
        axes[0].imshow(img_np)
        axes[0].set_title(f"A. RGB Orthomosaic ({fpath.name})", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: 100% Exhaustive Ground Truth
        gt_vis = img_np.copy()
        gt_uids = np.unique(label_map)
        gt_uids = gt_uids[gt_uids > 0]
        gt_overlay = np.zeros((H, W, 3), dtype=np.uint8)
        for i, uid in enumerate(gt_uids):
            m = (label_map == uid)
            c = (np.array(colors[i % len(colors)][:3]) * 255).astype(np.uint8)
            gt_overlay[m] = c
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(gt_vis, contours, -1, (255, 255, 255), 1)
        
        blended_gt = cv2.addWeighted(gt_vis, 0.7, gt_overlay, 0.3, 0)
        axes[1].imshow(blended_gt)
        axes[1].set_title(f"B. 100% Exhaustive GT ({len(gt_uids)} Crowns Annotated)", fontsize=12, fontweight="bold", color="darkgreen")
        axes[1].axis("off")

        # Panel 3: TreeFlowNet Prediction
        pred_vis = img_np.copy()
        pred_overlay = np.zeros((H, W, 3), dtype=np.uint8)
        for i, inst in enumerate(insts):
            m = inst["mask"]
            c = (np.array(colors[i % len(colors)][:3]) * 255).astype(np.uint8)
            pred_overlay[m] = c
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(pred_vis, contours, -1, (255, 255, 0), 1)

        blended_pred = cv2.addWeighted(pred_vis, 0.7, pred_overlay, 0.3, 0)
        axes[2].imshow(blended_pred)
        axes[2].set_title(f"C. TreeFlowNet Prediction ({len(insts)} Detected Crowns)", fontsize=12, fontweight="bold", color="blue")
        axes[2].axis("off")

        # Panel 4: Vector Flow Field Advection
        flow_np = preds_sq["flow"].cpu().numpy()  # [2, H, W]
        fy, fx = flow_np[0], flow_np[1]
        flow_mag = np.sqrt(fy ** 2 + fx ** 2)
        flow_ang = np.arctan2(fy, fx)
        hsv = np.zeros((H, W, 3), dtype=np.uint8)
        hsv[:, :, 0] = ((flow_ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = np.clip(flow_mag * 255, 0, 255).astype(np.uint8)
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Overlay quiver arrows
        step = 16
        y_grid, x_grid = np.mgrid[step//2:H:step, step//2:W:step]
        axes[3].imshow(flow_rgb)
        can_mask = preds_sq["canopy"].sigmoid().cpu().numpy()[0] > 0.4
        can_sub = can_mask[y_grid, x_grid]
        if can_sub.sum() > 0:
            axes[3].quiver(x_grid[can_sub], y_grid[can_sub], fx[y_grid, x_grid][can_sub], -fy[y_grid, x_grid][can_sub], color="white", scale=25, width=0.003)
        axes[3].set_title("D. Centripetal Vector Flow Field", fontsize=12, fontweight="bold")
        axes[3].axis("off")

        # Panel 5: SDT (Signed Distance Transform)
        sdt_map = preds_sq["sdt"].cpu().numpy()[0]
        im_sdt = axes[4].imshow(sdt_map, cmap="inferno", vmin=-1.0, vmax=1.0)
        plt.colorbar(im_sdt, ax=axes[4], fraction=0.046, pad=0.04)
        axes[4].set_title("E. Predicted Signed Distance Surface (SDT)", fontsize=12, fontweight="bold")
        axes[4].axis("off")

        # Panel 6: Centroid Density Heatmap
        cent_map = preds_sq["centroid"].sigmoid().cpu().numpy()[0]
        im_cent = axes[5].imshow(cent_map, cmap="magma", vmin=0.0, vmax=1.0)
        plt.colorbar(im_cent, ax=axes[5], fraction=0.046, pad=0.04)
        axes[5].set_title("F. Gaussian Crown Centroid Sinks", fontsize=12, fontweight="bold")
        axes[5].axis("off")

        plt.suptitle(f"100% Exhaustive Tree Crown Verification — {fpath.name} | GT: {len(gt_uids)} trees vs Pred: {len(insts)} trees", fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout()

        out_path = out_dir / f"quebec_z3_exhaustive_sample_{idx+1:02d}_{fpath.stem}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved visual comparison: {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on Exhaustive Quebec ITC Benchmark")
    parser.add_argument("--ckpt", type=str, default="crown_segmentation_research/experiments/tree_flow_exhaustive_v1/best_model.pth")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train", "all"])
    parser.add_argument("--out", type=str, default="crown_segmentation_research/experiments/quebec_exhaustive_eval_results.json")
    parser.add_argument("--decoder", type=str, default="organic", choices=["organic", "voronoi"])
    parser.add_argument("--save-vis", action="store_true")
    args = parser.parse_args()

    files = get_patch_files(args.split)
    print(f"\nLoaded {len(files)} patches from Quebec split '{args.split}'")

    print(f"Loading checkpoint: {args.ckpt}")
    model = load_model(args.ckpt)

    # 1. Evaluate TreeFlowNet
    treeflow_results = evaluate_dataset(model, files, baseline=False, decoder_type=args.decoder)

    # 2. Evaluate Baseline Watershed
    baseline_results = evaluate_dataset(None, files, baseline=True)

    final_results = {
        "split": args.split,
        "checkpoint": args.ckpt,
        "treeflow_net": treeflow_results,
        "baseline_watershed": baseline_results,
    }

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nSaved benchmark results to {out_file}")

    # Print ASCII Comparison Table
    print("\n" + "="*80)
    print(f"EXHAUSTIVE ALL-TREE BENCHMARK EVALUATION (Quebec Zone 3 Test Split)")
    print("="*80)
    print(f"{'Metric':<30} | {'Baseline Watershed':<20} | {'TreeFlowNet (Ours)':<20}")
    print("-" * 80)
    print(f"{'COCO Mask AP [0.50:0.95]':<30} | {baseline_results['segm_AP']:>19.2f}% | {treeflow_results['segm_AP']:>19.2f}%")
    print(f"{'COCO Mask AP50':<30} | {baseline_results['segm_AP50']:>19.2f}% | {treeflow_results['segm_AP50']:>19.2f}%")
    print(f"{'COCO Mask AP75':<30} | {baseline_results['segm_AP75']:>19.2f}% | {treeflow_results['segm_AP75']:>19.2f}%")
    print(f"{'COCO Mask AR100':<30} | {baseline_results['segm_AR100']:>19.2f}% | {treeflow_results['segm_AR100']:>19.2f}%")
    print(f"{'Hungarian F1 @ IoU 0.50':<30} | {baseline_results['Hungarian_F1_50']:>20.4f} | {treeflow_results['Hungarian_F1_50']:>20.4f}")
    print(f"{'Hungarian Precision @ 0.50':<30} | {baseline_results['Hungarian_Prec_50']:>20.4f} | {treeflow_results['Hungarian_Prec_50']:>20.4f}")
    print(f"{'Hungarian Recall @ 0.50':<30} | {baseline_results['Hungarian_Rec_50']:>20.4f} | {treeflow_results['Hungarian_Rec_50']:>20.4f}")
    print(f"{'Panoptic Quality (PQ)':<30} | {baseline_results['PQ']:>20.4f} | {treeflow_results['PQ']:>20.4f}")
    print(f"{'Segmentation Quality (SQ)':<30} | {baseline_results['SQ']:>20.4f} | {treeflow_results['SQ']:>20.4f}")
    print(f"{'Recognition Quality (RQ)':<30} | {baseline_results['RQ']:>20.4f} | {treeflow_results['RQ']:>20.4f}")
    print(f"{'Tree Count MAE per patch':<30} | {baseline_results['mae_count']:>20.2f} | {treeflow_results['mae_count']:>20.2f}")
    print(f"{'Omission Rate (FN/GT)':<30} | {baseline_results['omission_rate']:>19.2%} | {treeflow_results['omission_rate']:>19.2%}")
    print(f"{'Commission Rate (FP/Pred)':<30} | {baseline_results['commission_rate']:>19.2%} | {treeflow_results['commission_rate']:>19.2%}")
    print(f"{'Latency per 512x512 patch':<30} | {baseline_results['latency_per_patch_ms']:>18.1f}ms | {treeflow_results['latency_per_patch_ms']:>18.1f}ms")
    print("="*80 + "\n")

    if args.save_vis:
        vis_dir = REPO / "crown_segmentation_research/images"
        generate_visual_comparisons(model, files, vis_dir, num_samples=5)


if __name__ == "__main__":
    main()
