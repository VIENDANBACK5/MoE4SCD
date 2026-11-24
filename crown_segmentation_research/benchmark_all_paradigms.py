"""Master Benchmark: All Paradigms and 3-Stage Improvements on BAMFORESTS ITC.

Evaluates:
1. Baseline CanopyWatershedNet (Old dilation coupling)
2. Stage 1 Calibrated CanopyWatershedNet (Decoupled sweep)
3. Stage 2 Sharp-Ridge CanopyWatershedNet (Contact seam barrier)
4. Stage 3 Hamiltonian TopoFlow-Watershed (Vector flow divergence + topological relief)
"""
from __future__ import annotations

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

import argparse
import json
import cv2
import numpy as np
import pandas as pd
import torch

from crown_segmentation_research.methods.canopy_watershed.dataset import BAMCanopyWatershedDataset
from crown_segmentation_research.methods.canopy_watershed.decode import decode_watershed_fast
from crown_segmentation_research.methods.canopy_watershed.model import CanopyWatershedNet
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import (
    decode_hamiltonian_topoflow,
    evaluate_metrics,
)
from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_val", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Master Benchmark] Initializing benchmark on {device} ({args.num_val} crops)...", flush=True)

    # 1. Baseline Model
    base_model = CanopyWatershedNet(pretrained=False).to(device)
    ckpt_base = torch.load("DeadTrees/experiments/canopy_watershed/best_canopy_watershed.pth", map_location=device, weights_only=False)
    base_model.load_state_dict(ckpt_base["model_state_dict"])
    base_model.eval()

    # 2. Stage 2 Sharp Model
    sharp_ckpt_path = Path("DeadTrees/experiments/canopy_watershed/best_sharp_watershed.pth")
    sharp_model = None
    if sharp_ckpt_path.exists():
        sharp_model = CanopyWatershedNet(pretrained=False).to(device)
        ckpt_sharp = torch.load(sharp_ckpt_path, map_location=device, weights_only=False)
        sharp_model.load_state_dict(ckpt_sharp["model_state_dict"])
        sharp_model.eval()

    # 3. Stage 3 Flow Model
    flow_model = TreeFlowNet(pretrained_backbone=False).to(device)
    flow_ckpt_path = Path("DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth")
    if flow_ckpt_path.exists():
        ckpt_flow = torch.load(flow_ckpt_path, map_location=device, weights_only=False)
        flow_sd = ckpt_flow["model_state_dict"] if isinstance(ckpt_flow, dict) and "model_state_dict" in ckpt_flow else ckpt_flow
        flow_model.load_state_dict(flow_sd)
        flow_model.eval()

    val_dataset = BAMCanopyWatershedDataset(split="eval", crop_size=1024, augment=False, compute_targets=False)
    N = min(args.num_val, len(val_dataset))

    records = {
        "Baseline CanopyWatershed": [],
        "Stage 1 Decoupled Sweep": [],
        "Stage 2 Sharp Retrained": [],
        "Stage 3 TopoFlow Watershed": [],
    }

    print(f"[Master Benchmark] Running evaluation across {N} validation crops...", flush=True)

    with torch.no_grad():
        for idx in range(N):
            item = val_dataset[idx]
            img = item["image"].unsqueeze(0).to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                base_preds = base_model(img)
                flow_preds = flow_model(img) if flow_model else None
                sharp_preds = sharp_model(img) if sharp_model else None

            raw_inst = item["instance_label"].squeeze().numpy().astype(np.int32)
            gt_map = np.zeros_like(raw_inst)
            gt_map[raw_inst > 0] = raw_inst[raw_inst > 0] + 1

            # 1. Baseline
            b_surf = base_preds["surface"].squeeze().float().cpu().numpy()
            b_bound = base_preds["boundary"].squeeze().float().cpu().numpy()
            b_canopy = base_preds["canopy"].squeeze().float().cpu().numpy()
            m_base, inst_base = decode_watershed_fast(
                b_surf, b_bound, b_canopy,
                kernel_size=9, min_apex_val=0.28, min_canopy_val=0.38,
                pers_thresh=0.10, min_distance=10.0, bound_weight=1.0, min_area=30,
            )
            records["Baseline CanopyWatershed"].append(evaluate_metrics(m_base, inst_base, gt_map))

            # 2. Stage 1 Decoupled Sweep
            m_st1, inst_st1 = decode_watershed_fast(
                b_surf, b_bound, b_canopy,
                kernel_size=5, min_apex_val=0.25, min_canopy_val=0.25,
                pers_thresh=0.03, min_distance=8.0, bound_weight=0.5, min_area=30,
            )
            records["Stage 1 Decoupled Sweep"].append(evaluate_metrics(m_st1, inst_st1, gt_map))

            # 3. Stage 2 Sharp Retrained
            if sharp_preds:
                s_surf = sharp_preds["surface"].squeeze().float().cpu().numpy()
                s_bound = sharp_preds["boundary"].squeeze().float().cpu().numpy()
                s_canopy = sharp_preds["canopy"].squeeze().float().cpu().numpy()
                m_st2, inst_st2 = decode_watershed_fast(
                    s_surf, s_bound, s_canopy,
                    kernel_size=5, min_apex_val=0.25, min_canopy_val=0.30,
                    pers_thresh=0.03, min_distance=8.0, bound_weight=1.0, min_area=30,
                )
                records["Stage 2 Sharp Retrained"].append(evaluate_metrics(m_st2, inst_st2, gt_map))

            # 4. Stage 3 TopoFlow Watershed
            if flow_preds:
                f_flow = flow_preds["flow"].squeeze().float().cpu().numpy()
                m_st3, inst_st3 = decode_hamiltonian_topoflow(
                    surface=b_surf, boundary=b_bound, canopy=b_canopy, flow=f_flow,
                    kernel_size=5, min_apex_val=0.20, min_canopy_val=0.30,
                    pers_thresh=0.03, min_distance=6.0, bound_weight=1.0, div_weight=2.0, min_area=25,
                )
                records["Stage 3 TopoFlow Watershed"].append(evaluate_metrics(m_st3, inst_st3, gt_map))

            if (idx + 1) % 10 == 0 or (idx + 1) == N:
                print(f"  Processed [{idx+1:02d}/{N:02d}] validation crops...", flush=True)

    summary_rows = []
    for name, r_list in records.items():
        if not r_list:
            continue
        summary_rows.append({
            "Method / Paradigm": name,
            "Detected Trees/Crop": float(np.mean([x["trees"] for x in r_list])),
            "mAP@50 (%)": float(np.mean([x["ap50"] for x in r_list])),
            "mAP@75 (%)": float(np.mean([x["ap75"] for x in r_list])),
            "PQ (%)": float(np.mean([x["pq"] for x in r_list])),
            "SQ (%)": float(np.mean([x["sq"] for x in r_list])),
            "RQ (%)": float(np.mean([x["rq"] for x in r_list])),
        })

    df_summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 95)
    print("           BAMFORESTS BENCHMARK: 3-STAGE EXPERIMENTAL CAMPAIGN RESULTS        ")
    print("=" * 95)
    print(df_summary.to_string(index=False))
    print("=" * 95 + "\n")

    out_json = Path("crown_segmentation_research/master_3stage_benchmark_results.json")
    out_json.write_text(df_summary.to_json(orient="records", indent=2))
    print(f"[Master Benchmark] Saved quantitative summary to {out_json}", flush=True)


if __name__ == "__main__":
    from crown_segmentation_research.research_integrity.a06_hardening_v2.runtime import reject_legacy_entrypoint
    reject_legacy_entrypoint(__file__)
    main()
