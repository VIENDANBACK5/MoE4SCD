"""Fine-Grained Multi-Scale Parameter Sweep for CanopyWatershedNet.

Tests decoupled kernel sizes and fine spatial distance / prominence thresholds
to systematically optimize mAP@50 and PQ against ground truth.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import os
import sys
import time
from pathlib import Path

_conda_p = '/home/ubuntu/anaconda3/envs/cs2/lib/python3.11/site-packages'
_local_p = '/home/chung/.local/lib/python3.11/site-packages'
sys.path = [p for p in sys.path if p != _local_p and p != _conda_p]
sys.path.insert(0, _conda_p)
sys.path.append(_local_p)

for _p in Path(__file__).resolve().parents:
    if (_p / "crown_segmentation_research").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

import numpy as np
import pandas as pd
import torch

from crown_segmentation_research.methods.canopy_watershed.dataset import BAMCanopyWatershedDataset
from crown_segmentation_research.methods.canopy_watershed.decode import decode_watershed_fast
from crown_segmentation_research.methods.canopy_watershed.model import CanopyWatershedNet

_CACHED_VAL_DATA: list[dict] = []


def _init_worker(cached_items: list[dict]) -> None:
    global _CACHED_VAL_DATA
    _CACHED_VAL_DATA = cached_items


def evaluate_metrics_single_crop(
    pred_markers: np.ndarray,
    pred_instances: list[dict],
    gt_map: np.ndarray,
    gt_ids: np.ndarray,
    gt_areas: np.ndarray,
    gt_lookup: np.ndarray,
) -> dict[str, float]:
    num_gt = len(gt_ids)
    num_pred = len(pred_instances)

    if num_gt == 0 and num_pred == 0:
        return {"ap50": 1.0, "ap75": 1.0, "pq": 1.0, "sq": 1.0, "rq": 1.0}
    if num_gt == 0 or num_pred == 0:
        return {"ap50": 0.0, "ap75": 0.0, "pq": 0.0, "sq": 0.0, "rq": 0.0}

    flat_pred = pred_markers.ravel()
    flat_gt = gt_map.ravel()

    max_pred = int(flat_pred.max())
    if max_pred < 2:
        return {"ap50": 0.0, "ap75": 0.0, "pq": 0.0, "sq": 0.0, "rq": 0.0}

    pred_lookup = np.full(max_pred + 1, -1, dtype=np.int64)
    for idx, p in enumerate(pred_instances):
        pred_lookup[p["id"]] = idx

    valid_mask = (flat_pred >= 2) & (flat_gt >= 2)
    p_vals = flat_pred[valid_mask]
    g_vals = flat_gt[valid_mask]

    p_indices = pred_lookup[p_vals]
    g_indices = gt_lookup[g_vals]

    valid_pair = (p_indices >= 0) & (g_indices >= 0)
    p_indices = p_indices[valid_pair]
    g_indices = g_indices[valid_pair]

    flat_pair_indices = p_indices * num_gt + g_indices
    intersection = np.bincount(flat_pair_indices, minlength=num_pred * num_gt).reshape(num_pred, num_gt).astype(np.float32)

    pred_areas = np.array([p["area"] for p in pred_instances], dtype=np.float32)[:, None]
    union = pred_areas + gt_areas - intersection
    iou_mat = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)

    # Calculate AP50, AP75, PQ
    # AP50 greedy matching
    iou_50 = (iou_mat >= 0.5)
    matched_gt50 = set()
    tp50 = 0
    scores = np.array([p["score"] for p in pred_instances])
    p_order = np.argsort(-scores)

    for p_i in p_order:
        gt_cands = np.where(iou_50[p_i])[0]
        for g_i in gt_cands:
            if g_i not in matched_gt50:
                matched_gt50.add(g_i)
                tp50 += 1
                break

    prec50 = tp50 / num_pred if num_pred > 0 else 0.0
    rec50 = tp50 / num_gt if num_gt > 0 else 0.0
    ap50 = 2 * (prec50 * rec50) / (prec50 + rec50) if (prec50 + rec50) > 0 else 0.0

    # AP75 greedy matching
    iou_75 = (iou_mat >= 0.75)
    matched_gt75 = set()
    tp75 = 0
    for p_i in p_order:
        gt_cands = np.where(iou_75[p_i])[0]
        for g_i in gt_cands:
            if g_i not in matched_gt75:
                matched_gt75.add(g_i)
                tp75 += 1
                break

    prec75 = tp75 / num_pred if num_pred > 0 else 0.0
    rec75 = tp75 / num_gt if num_gt > 0 else 0.0
    ap75 = 2 * (prec75 * rec75) / (prec75 + rec75) if (prec75 + rec75) > 0 else 0.0

    # Panoptic Quality (PQ)
    tp_mask = (iou_mat > 0.5)
    tp_pairs = np.argwhere(tp_mask)
    num_tp = len(tp_pairs)
    fp = num_pred - num_tp
    fn = num_gt - num_tp

    if num_tp > 0:
        iou_sum = sum(iou_mat[p, g] for p, g in tp_pairs)
        sq = iou_sum / num_tp
        rq = num_tp / (num_tp + 0.5 * fp + 0.5 * fn)
        pq = sq * rq
    else:
        sq, rq, pq = 0.0, 0.0, 0.0

    return {
        "ap50": float(ap50 * 100.0),
        "ap75": float(ap75 * 100.0),
        "pq": float(pq * 100.0),
        "sq": float(sq * 100.0),
        "rq": float(rq * 100.0),
    }


def _eval_single_config(params: dict) -> dict:
    total_ap50, total_ap75, total_pq, total_sq, total_rq = 0.0, 0.0, 0.0, 0.0, 0.0
    total_trees = 0
    N = len(_CACHED_VAL_DATA)

    for item in _CACHED_VAL_DATA:
        pred_markers, pred_instances = decode_watershed_fast(
            surface=item["surface"],
            boundary=item["boundary"],
            canopy=item["canopy"],
            kernel_size=params["kernel_size"],
            min_apex_val=params["min_apex_val"],
            min_canopy_val=params["min_canopy_val"],
            pers_thresh=params["pers_thresh"],
            min_distance=params["min_distance"],
            bound_weight=params["bound_weight"],
            min_area=params["min_area"],
        )
        total_trees += len(pred_instances)

        m = evaluate_metrics_single_crop(
            pred_markers=pred_markers,
            pred_instances=pred_instances,
            gt_map=item["gt_map"],
            gt_ids=item["gt_ids"],
            gt_areas=item["gt_areas"],
            gt_lookup=item["gt_lookup"],
        )
        total_ap50 += m["ap50"]
        total_ap75 += m["ap75"]
        total_pq += m["pq"]
        total_sq += m["sq"]
        total_rq += m["rq"]

    return {
        **params,
        "mAP50": total_ap50 / N,
        "mAP75": total_ap75 / N,
        "PQ": total_pq / N,
        "SQ": total_sq / N,
        "RQ": total_rq / N,
        "avg_trees": total_trees / N,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="DeadTrees/experiments/canopy_watershed/best_canopy_watershed.pth")
    parser.add_argument("--num_val", type=int, default=25)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage 1 Sweep] Loading model from {args.model_path} on {device}...", flush=True)

    model = CanopyWatershedNet(pretrained=False).to(device)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_dataset = BAMCanopyWatershedDataset(split="eval", crop_size=1024, augment=False, compute_targets=False)
    num_samples = min(args.num_val, len(val_dataset))
    print(f"[Stage 1 Sweep] Caching neural predictions for {num_samples} validation samples in RAM...", flush=True)

    cached_data = []
    with torch.no_grad():
        for idx in range(num_samples):
            item = val_dataset[idx]
            img = item["image"].unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(img)

            surf_p = preds["surface"].squeeze().float().cpu().numpy()
            bound_p = preds["boundary"].squeeze().float().cpu().numpy()
            canopy_p = preds["canopy"].squeeze().float().cpu().numpy()

            raw_inst = item["instance_label"].squeeze().numpy().astype(np.int32)
            gt_map = np.zeros_like(raw_inst)
            gt_map[raw_inst > 0] = raw_inst[raw_inst > 0] + 1

            gt_ids = np.unique(gt_map)
            gt_ids = gt_ids[gt_ids >= 2]
            max_gt = int(gt_map.max()) if len(gt_ids) > 0 else 0
            gt_lookup = np.full(max_gt + 1, -1, dtype=np.int64)
            for g_i, gid in enumerate(gt_ids):
                gt_lookup[gid] = g_i

            gt_valid = gt_map[gt_map >= 2]
            if len(gt_valid) > 0:
                g_idx_arr = gt_lookup[gt_valid]
                gt_areas = np.bincount(g_idx_arr, minlength=len(gt_ids))[None, :].astype(np.float32)
            else:
                gt_areas = np.zeros((1, 0), dtype=np.float32)

            cached_data.append({
                "surface": surf_p,
                "boundary": bound_p,
                "canopy": canopy_p,
                "gt_map": gt_map,
                "gt_ids": gt_ids,
                "gt_areas": gt_areas,
                "gt_lookup": gt_lookup,
            })

    # Fine-grained search grid tailored for high tree density (50-150 trees/crop)
    grid = {
        "kernel_size": [3, 5],
        "min_apex_val": [0.12, 0.18, 0.25],
        "min_canopy_val": [0.25, 0.35],
        "pers_thresh": [0.005, 0.015, 0.03, 0.06],
        "min_distance": [3.0, 5.0, 8.0],
        "bound_weight": [0.5, 1.0],
        "min_area": [15, 30],
    }

    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    num_configs = len(combinations)
    num_workers = min(args.workers, os.cpu_count() or 4)
    print(f"[Stage 1 Sweep] Commencing parallel search across {num_configs} configurations with {num_workers} CPU workers...", flush=True)

    results_table = []
    t_start = time.perf_counter()
    best_mAP = 0.0

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(cached_data,),
    ) as executor:
        futures = {executor.submit(_eval_single_config, params): params for params in combinations}
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            res = future.result()
            results_table.append(res)
            if res["mAP50"] > best_mAP:
                best_mAP = res["mAP50"]

            if count % 200 == 0 or count == num_configs:
                elapsed = time.perf_counter() - t_start
                rate = count / elapsed if elapsed > 0 else 0
                print(f"  [Progress {count:04d}/{num_configs}] Elapsed: {elapsed:.1f}s ({rate:.1f} cfg/s) | Best mAP50: {best_mAP:.2f}%", flush=True)

    df = pd.DataFrame(results_table)
    df_sorted = df.sort_values(by=["mAP50", "PQ"], ascending=False).reset_index(drop=True)

    out_csv = Path("crown_segmentation_research/sweep_watershed_fine_results.csv")
    df_sorted.to_csv(out_csv, index=False)
    print(f"\n[Stage 1 Sweep] Results saved to {out_csv}", flush=True)

    print("\n" + "=" * 95)
    print("                 TOP 10 OPTIMAL FINE-GRAINED CONFIGURATIONS           ")
    print("=" * 95)
    print(df_sorted.head(10).to_string(index=True))
    print("=" * 95)

    best_cfg = df_sorted.iloc[0].to_dict()
    print(f"\n[Stage 1 Sweep] Best Calibration Found:")
    print(f"  - Kernel Size     : {int(best_cfg['kernel_size'])}")
    print(f"  - Min Apex Value  : {best_cfg['min_apex_val']:.3f}")
    print(f"  - Min Canopy Gate : {best_cfg['min_canopy_val']:.3f}")
    print(f"  - Persistence Tau : {best_cfg['pers_thresh']:.3f}")
    print(f"  - Min Distance    : {best_cfg['min_distance']:.1f} px")
    print(f"  - Boundary Weight : {best_cfg['bound_weight']:.2f}")
    print(f"  - Min Area        : {int(best_cfg['min_area'])} px")
    print(f"  - Avg Detected Trees : {best_cfg['avg_trees']:.1f} trees/crop")
    print(f"  => Validation mAP@50 : {best_cfg['mAP50']:.2f}% | mAP@75 : {best_cfg['mAP75']:.2f}% | PQ : {best_cfg['PQ']:.2f}%\n")


if __name__ == "__main__":
    main()
