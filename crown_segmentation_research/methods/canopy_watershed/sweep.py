"""Systematic Hyperparameter Sweep for CanopyWatershedNet (Option A).

Caches neural model outputs (U, B, C) on BAMFORESTS validation set in memory,
then sweeps topological decoding parameters in parallel across CPU workers
to maximize mAP@50 and Panoptic Quality (PQ).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import os
import sys
import time
from pathlib import Path

# Ensure root workspace is in sys.path
# Ensure workspace root is in sys.path
import sys
from pathlib import Path
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


def evaluate_batch_metrics_fast(
    pred_markers: np.ndarray,
    pred_instances: list[dict],
    gt_map: np.ndarray,
    gt_ids: np.ndarray,
    gt_areas: np.ndarray,
    gt_lookup: np.ndarray,
    iou_thresholds: list[float] = [0.5, 0.75],
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
        if p["id"] <= max_pred:
            pred_lookup[p["id"]] = idx

    valid_mask = (flat_pred >= 2) & (flat_gt >= 2)
    iou_mat = np.zeros((num_pred, num_gt), dtype=np.float32)

    if np.any(valid_mask):
        p_sub = flat_pred[valid_mask]
        g_sub = flat_gt[valid_mask]

        p_idx_arr = pred_lookup[p_sub]
        g_sub_mask = g_sub < len(gt_lookup)
        p_idx_arr = p_idx_arr[g_sub_mask]
        g_idx_arr = gt_lookup[g_sub[g_sub_mask]]

        both_valid = (p_idx_arr >= 0) & (g_idx_arr >= 0)

        if np.any(both_valid):
            p_final = p_idx_arr[both_valid]
            g_final = g_idx_arr[both_valid]

            encoded = p_final * num_gt + g_final
            intersections = np.bincount(encoded, minlength=num_pred * num_gt).reshape(num_pred, num_gt)

            pred_areas = np.array([p["area"] for p in pred_instances], dtype=np.float32)[:, None]

            unions = pred_areas + gt_areas - intersections
            unions[unions <= 0] = 1.0
            iou_mat = intersections.astype(np.float32) / unions

    results: dict[str, float] = {}

    for thresh in iou_thresholds:
        matched_gt = set()
        tp = 0
        order = np.argsort([-p["score"] for p in pred_instances])
        for p_idx in order:
            best_g_idx = -1
            best_iou = thresh
            for g_idx in range(num_gt):
                if g_idx not in matched_gt and iou_mat[p_idx, g_idx] >= best_iou:
                    best_iou = iou_mat[p_idx, g_idx]
                    best_g_idx = g_idx
            if best_g_idx >= 0:
                tp += 1
                matched_gt.add(best_g_idx)

        rec = tp / num_gt if num_gt > 0 else 0.0
        prec = tp / num_pred if num_pred > 0 else 0.0
        ap = 2.0 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        key = "ap50" if abs(thresh - 0.5) < 1e-4 else "ap75"
        results[key] = ap

    matched_ious = []
    used_preds = set()
    used_gts = set()

    for p_idx in range(num_pred):
        for g_idx in range(num_gt):
            if iou_mat[p_idx, g_idx] > 0.5:
                if p_idx not in used_preds and g_idx not in used_gts:
                    used_preds.add(p_idx)
                    used_gts.add(g_idx)
                    matched_ious.append(iou_mat[p_idx, g_idx])

    tp = len(matched_ious)
    fp = num_pred - tp
    fn = num_gt - tp

    sq = float(np.mean(matched_ious)) if tp > 0 else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0.0
    pq = sq * rq

    results["pq"] = pq
    results["sq"] = sq
    results["rq"] = rq
    return results


# Global cache for worker processes
_WORKER_CACHED_DATA: list[dict] = []


def _init_worker(cached_data: list[dict]) -> None:
    global _WORKER_CACHED_DATA
    _WORKER_CACHED_DATA = cached_data


def _eval_single_config(params: dict) -> dict:
    global _WORKER_CACHED_DATA
    num_samples = len(_WORKER_CACHED_DATA)
    total_ap50 = 0.0
    total_ap75 = 0.0
    total_pq = 0.0
    total_sq = 0.0
    total_rq = 0.0
    total_trees = 0

    for item in _WORKER_CACHED_DATA:
        pred_markers, pred_insts = decode_watershed_fast(
            surface=item["surface"],
            boundary=item["boundary"],
            canopy=item["canopy"],
            kernel_size=params["kernel_size"],
            min_apex_val=params["min_apex_val"],
            min_canopy_val=params["min_canopy_val"],
            pers_thresh=params["pers_thresh"],
            min_distance=params["min_distance"],
            bound_weight=params["bound_weight"],
            min_area=params.get("min_area", 35),
        )

        metrics = evaluate_batch_metrics_fast(
            pred_markers=pred_markers,
            pred_instances=pred_insts,
            gt_map=item["gt_map"],
            gt_ids=item["gt_ids"],
            gt_areas=item["gt_areas"],
            gt_lookup=item["gt_lookup"],
        )
        total_ap50 += metrics["ap50"]
        total_ap75 += metrics["ap75"]
        total_pq += metrics["pq"]
        total_sq += metrics["sq"]
        total_rq += metrics["rq"]
        total_trees += len(pred_insts)

    mean_ap50 = (total_ap50 / num_samples) * 100.0
    mean_ap75 = (total_ap75 / num_samples) * 100.0
    mean_pq = (total_pq / num_samples) * 100.0
    mean_sq = (total_sq / num_samples) * 100.0
    mean_rq = (total_rq / num_samples) * 100.0
    avg_trees = total_trees / num_samples

    return {
        **params,
        "mAP50": mean_ap50,
        "mAP75": mean_ap75,
        "PQ": mean_pq,
        "SQ": mean_sq,
        "RQ": mean_rq,
        "avg_trees": avg_trees,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep hyperparameters for CanopyWatershedNet.")
    parser.add_argument("--model_path", type=str, default="DeadTrees/experiments/canopy_watershed/best_canopy_watershed.pth")
    parser.add_argument("--num_val", type=int, default=20, help="Number of validation crops to evaluate")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel CPU workers")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Sweep] Loading model from {args.model_path} on {device}...", flush=True)

    model = CanopyWatershedNet(pretrained=False).to(device)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_dataset = BAMCanopyWatershedDataset(split="eval", crop_size=1024, augment=False, compute_targets=False)
    num_samples = min(args.num_val, len(val_dataset))
    print(f"[Sweep] Caching neural predictions for {num_samples} validation samples in RAM...", flush=True)

    cached_data = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for idx in range(num_samples):
            item = val_dataset[idx]
            img = item["image"].unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(img)

            surf_p = preds["surface"].squeeze().float().cpu().numpy()
            bound_p = preds["boundary"].squeeze().float().cpu().numpy()
            canopy_p = preds["canopy"].squeeze().float().cpu().numpy()

            # Ground truth instance label directly from dataset (instances >= 2)
            raw_inst = item["instance_label"].squeeze().numpy().astype(np.int32)
            gt_map = np.zeros_like(raw_inst)
            gt_map[raw_inst > 0] = raw_inst[raw_inst > 0] + 1  # 0 is bg, 1 is unused, 2+ are instances

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

    cache_time = time.perf_counter() - t0
    print(f"[Sweep] Cached {num_samples} items and ground truth in {cache_time:.2f}s.", flush=True)

    # Expanded, Domain-Calibrated Search Grid for Tree Crowns
    grid = {
        "kernel_size": [7, 11, 15],
        "min_apex_val": [0.25, 0.35, 0.45, 0.55],
        "min_canopy_val": [0.30, 0.42, 0.55],
        "pers_thresh": [0.05, 0.10, 0.18, 0.26],
        "min_distance": [8.0, 13.0, 18.0, 24.0],
        "bound_weight": [0.6, 1.2, 2.0],
        "min_area": [35, 75],
    }

    # Generate parameter combinations
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    num_configs = len(combinations)
    num_workers = min(args.workers, os.cpu_count() or 4)
    print(f"[Sweep] Commencing parallel grid search across {num_configs} configurations using {num_workers} CPU workers...", flush=True)

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

            if count % 100 == 0 or count == num_configs:
                elapsed = time.perf_counter() - t_start
                rate = count / elapsed if elapsed > 0 else 0
                print(f"  [Progress {count:04d}/{num_configs}] Elapsed: {elapsed:.1f}s ({rate:.1f} cfg/s) | Best mAP50: {best_mAP:.2f}%", flush=True)

    # Sort results
    df = pd.DataFrame(results_table)
    df_sorted = df.sort_values(by=["mAP50", "PQ"], ascending=False).reset_index(drop=True)

    out_csv = Path("crown_segmentation_research/sweep_watershed_results.csv")
    df_sorted.to_csv(out_csv, index=False)
    print(f"\n[Sweep] Results saved to {out_csv}", flush=True)

    print("\n" + "=" * 95)
    print("                 TOP 10 OPTIMAL DECODING CONFIGURATIONS                ")
    print("=" * 95)
    print(df_sorted.head(10).to_string(index=True))
    print("=" * 95)

    best_cfg = df_sorted.iloc[0].to_dict()
    print(f"\n[Sweep] Best Configuration Found:")
    print(f"  - Kernel Size     : {int(best_cfg['kernel_size'])}")
    print(f"  - Min Apex Value  : {best_cfg['min_apex_val']:.3f}")
    print(f"  - Min Canopy Gate : {best_cfg['min_canopy_val']:.3f}")
    print(f"  - Persistence Tau : {best_cfg['pers_thresh']:.3f}")
    print(f"  - Min Distance    : {best_cfg['min_distance']:.1f} px")
    print(f"  - Boundary Weight : {best_cfg['bound_weight']:.2f}")
    print(f"  - Min Area        : {int(best_cfg['min_area'])} px")
    print(f"  => Validation mAP@50 : {best_cfg['mAP50']:.2f}% | mAP@75 : {best_cfg['mAP75']:.2f}% | PQ : {best_cfg['PQ']:.2f}%\n")


if __name__ == "__main__":
    main()
