"""
Benchmark TreeFlowNet vs Deep Learning SOTAs (Mask R-CNN, Detectree2, StarConvexNet)
and Classical Watershed on the Exhaustive Quebec ITC Benchmark.

Evaluates:
  1. TreeFlowNet (Organic TopoFlow-Watershed, Ours)
  2. Detectree2 (Ball et al. 2023, Forestry Mask R-CNN)
  3. Mask R-CNN (He et al., ResNet50-FPN)
  4. StarConvexNet (StarDist-based dense polygon regression)
  5. Baseline Watershed (Marker-controlled watershed on canopy)

Reports full COCO Mask AP, AP50, Hungarian F1, Precision, Recall, PQ, Count MAE.
"""
from __future__ import annotations

import os
import sys
sys.path = [p for p in sys.path if not p.startswith('/home/chung/.local')]
import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from pycocotools import mask as coco_mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet
from crown_segmentation_research.methods.tree_flow.decode import decode_organic
from crown_segmentation_research.evaluation.eval_exhaustive_quebec import (
    run_baseline_watershed,
    get_patch_files,
    compute_f1_hungarian,
)
from crown_segmentation_research.evaluation.benchmark_bam_coco_ap import (
    load_star_convex_model,
    infer_star_convex_tile,
)
from experiments.g1b_baselines.training.train_learned_baselines import (
    build_maskrcnn_model,
    build_detectree2_model,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_all_models():
    models = {}

    # 1. TreeFlowNet
    tf_ckpt = REPO / "crown_segmentation_research/experiments/tree_flow_exhaustive_v1/best_model.pth"
    if tf_ckpt.exists():
        tf_model = TreeFlowNet()
        ckpt = torch.load(tf_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        tf_model.load_state_dict(state)
        tf_model.to(DEVICE).eval()
        models["TreeFlowNet (Ours)"] = ("treeflow", tf_model)
        print("Loaded TreeFlowNet (Ours)")

    # 2. Detectree2
    dt2_ckpt = REPO / "experiments/g1b_baselines/checkpoints/detectree2_seed42_best.pth"
    if dt2_ckpt.exists():
        dt2_model = build_detectree2_model()
        state = torch.load(dt2_ckpt, map_location="cpu", weights_only=True)
        dt2_model.load_state_dict(state)
        dt2_model.to(DEVICE).eval()
        models["Detectree2 (Ball et al.)"] = ("torchvision_rcnn", dt2_model)
        print("Loaded Detectree2 (Ball et al.)")

    # 3. Mask R-CNN
    mrcnn_ckpt = REPO / "experiments/g1b_baselines/checkpoints/maskrcnn_seed42_best.pth"
    if mrcnn_ckpt.exists():
        mrcnn_model = build_maskrcnn_model()
        state = torch.load(mrcnn_ckpt, map_location="cpu", weights_only=True)
        mrcnn_model.load_state_dict(state)
        mrcnn_model.to(DEVICE).eval()
        models["Mask R-CNN (ResNet-50)"] = ("torchvision_rcnn", mrcnn_model)
        print("Loaded Mask R-CNN (ResNet-50)")

    # 4. StarConvexNet (StarDist)
    sc_ckpt = REPO / "crown_segmentation_research/experiments/star_convex_screen_v8/star_convex_screen_latest.pth"
    if sc_ckpt.exists():
        sc_model = load_star_convex_model(str(sc_ckpt), "v8")
        models["StarConvex (StarDist v8)"] = ("star_convex", sc_model)
        print("Loaded StarConvex (StarDist v8)")

    # 5. Baseline Watershed
    models["Baseline Watershed"] = ("watershed", None)

    return models


def evaluate_single_method(
    name: str,
    model_info: tuple[str, Any],
    files: list[Path],
) -> dict[str, Any]:
    m_type, model = model_info

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
    print(f"\n--> Evaluating {name} on {len(files)} patches...", flush=True)

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

        # Ground truth
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

        # Inference
        instances = []
        if m_type == "watershed":
            instances = run_baseline_watershed(img_np)
        elif m_type == "treeflow":
            norm = ((img_np.astype(np.float32) / 255.0) - mean) / std
            t_img = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                preds = model(t_img)
                preds_sq = {k: v[0] for k, v in preds.items()}
                instances = decode_organic(
                    preds_sq,
                    img_rgb=img_np,
                    min_sink_dist=44,
                    canopy_thresh=0.30,
                    sdt_thresh=-0.15,
                    min_area=150,
                    device=str(DEVICE),
                )
        elif m_type == "torchvision_rcnn":
            t_img = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
            t_img = t_img.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                preds = model(t_img)[0]
            scores = preds["scores"].cpu().numpy()
            masks = preds["masks"].squeeze(1).cpu().numpy()
            for score, m_prob in zip(scores, masks):
                if score < 0.25:
                    continue
                m_bin = (m_prob > 0.5)
                if m_bin.sum() < 25:
                    continue
                instances.append({"mask": m_bin, "score": float(score), "area": int(m_bin.sum())})
        elif m_type == "star_convex":
            cfg = {"prob_threshold": 0.35, "min_peak_distance": 5, "nms_iou_threshold": 0.2}
            masks, scores = infer_star_convex_tile(model, img_np, cfg)
            for m, s in zip(masks, scores):
                if m.sum() < 25:
                    continue
                instances.append({"mask": m.astype(bool), "score": float(s), "area": int(m.sum())})

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

    elapsed = time.time() - t0
    latency_ms = (elapsed / len(files)) * 1000

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json") as f_gt:
        json.dump(coco_gt_dict, f_gt)
        f_gt.flush()
        coco_gt = COCO(f_gt.name)

    if len(coco_results) == 0:
        return {
            "method": name,
            "error": "No predictions",
            "total_gt": total_gt_crowns,
            "total_pred": 0,
        }

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    segm_names = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    segm_stats = {f"segm_{n}": float(v * 100) for n, v in zip(segm_names, coco_eval.stats)}

    img_ids = [img["id"] for img in coco_gt_dict["images"]]
    f1_stats = compute_f1_hungarian(coco_gt, coco_results, img_ids)

    count_errors = np.abs(np.array(patch_pred_counts) - np.array(patch_gt_counts))
    mae_count = float(np.mean(count_errors))

    return {
        "method": name,
        "num_patches": len(files),
        "total_gt": total_gt_crowns,
        "total_pred": total_pred_crowns,
        "mae_count": mae_count,
        "latency_ms": latency_ms,
        **segm_stats,
        **f1_stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train", "all"])
    parser.add_argument("--out", type=str, default="crown_segmentation_research/experiments/quebec_sota_benchmark.json")
    args = parser.parse_args()

    files = get_patch_files(args.split)
    print(f"\n================================================================================")
    print(f"BENCHMARKING TREEFLOWNET VS SOTA DEEP LEARNING ARCHITECTURES ON QUEBEC ({args.split.upper()})")
    print(f"Total patches: {len(files)}")
    print(f"================================================================================")

    models = load_all_models()
    results = {}

    for name, info in models.items():
        res = evaluate_single_method(name, info, files)
        results[name] = res

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved SOTA benchmark results to {out_file}")

    # Print Comprehensive Table
    print("\n" + "="*115)
    print(f"COMPREHENSIVE SOTA BENCHMARK ON QUEBEC ({args.split.upper()} SPLIT - {len(files)} PATCHES)")
    print("="*115)
    header = f"{'Method':<26} | {'AP (%)':<8} | {'AP50 (%)':<9} | {'AP75 (%)':<9} | {'Hung F1':<8} | {'Prec':<7} | {'Rec':<7} | {'PQ':<7} | {'Count MAE':<9} | {'Latency':<8}"
    print(header)
    print("-" * 115)
    for name, r in results.items():
        if "segm_AP" in r:
            row = (
                f"{name:<26} | "
                f"{r['segm_AP']:>7.2f}% | "
                f"{r['segm_AP50']:>8.2f}% | "
                f"{r['segm_AP75']:>8.2f}% | "
                f"{r['Hungarian_F1_50']:>8.4f} | "
                f"{r['Hungarian_Prec_50']:>7.2%} | "
                f"{r['Hungarian_Rec_50']:>7.2%} | "
                f"{r['PQ']:>7.4f} | "
                f"{r['mae_count']:>9.2f} | "
                f"{r['latency_ms']:>6.1f}ms"
            )
            print(row)
    print("="*115 + "\n")


if __name__ == "__main__":
    main()
