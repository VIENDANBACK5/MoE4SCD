"""Multi-Stage Master Benchmark & Diagnostic Visualizer.

Compares:
1. Ground Truth (ITC annotated polygons & centroids)
2. Baseline CanopyWatershedNet (Under-segmented / Over-merged failure)
3. Stage 1 Calibrated CanopyWatershedNet (Decoupled sweep)
4. Stage 2 Sharp-Ridge Retrained CanopyWatershedNet (Contact seam barrier)
5. Stage 3 Hamiltonian TopoFlow-Watershed (Vector divergence + topological relief)
"""
from __future__ import annotations

import os
import sys
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

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from crown_segmentation_research.methods.canopy_watershed.dataset import BAMCanopyWatershedDataset
from crown_segmentation_research.methods.canopy_watershed.decode import decode_watershed_fast
from crown_segmentation_research.methods.canopy_watershed.model import CanopyWatershedNet
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import (
    decode_hamiltonian_topoflow,
    evaluate_metrics,
)
from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet


def create_colored_mask(instance_map: np.ndarray) -> np.ndarray:
    """Generates bright distinct color overlay for instance label map."""
    H, W = instance_map.shape
    colored = np.zeros((H, W, 3), dtype=np.uint8)
    inst_ids = np.unique(instance_map)
    inst_ids = inst_ids[inst_ids > 0]

    np.random.seed(42)
    colors = np.random.randint(40, 255, size=(max(len(inst_ids) + 10, 500), 3), dtype=np.uint8)

    for i, iid in enumerate(inst_ids):
        colored[instance_map == iid] = colors[i % len(colors)]

    return colored


def render_overlay(rgb: np.ndarray, instance_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blends colored instance masks over RGB image with crisp instance boundaries."""
    colored = create_colored_mask(instance_map)
    has_mask = instance_map > 0

    blended = rgb.copy()
    blended[has_mask] = (
        (1 - alpha) * rgb[has_mask].astype(np.float32) + alpha * colored[has_mask].astype(np.float32)
    ).astype(np.uint8)

    # Add white boundary contours
    contours_img = np.zeros((instance_map.shape[0], instance_map.shape[1]), dtype=np.uint8)
    inst_ids = np.unique(instance_map)
    for iid in inst_ids[inst_ids > 0]:
        mask_i = (instance_map == iid).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, cnts, -1, (255, 255, 255), 1)

    return blended


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Master Visualizer] Loading models on {device}...", flush=True)

    val_dataset = BAMCanopyWatershedDataset(split="eval", crop_size=1024, augment=False, compute_targets=False)

    # 1. CanopyWatershed Baseline / Stage 1 Model
    ws_model = CanopyWatershedNet(pretrained=False).to(device)
    ckpt_ws = torch.load("DeadTrees/experiments/canopy_watershed/best_canopy_watershed.pth", map_location=device, weights_only=False)
    ws_model.load_state_dict(ckpt_ws["model_state_dict"])
    ws_model.eval()

    # 2. Stage 2 Sharp Model (if trained)
    sharp_ckpt_path = Path("DeadTrees/experiments/canopy_watershed/best_sharp_watershed.pth")
    sharp_model = None
    if sharp_ckpt_path.exists():
        sharp_model = CanopyWatershedNet(pretrained=False).to(device)
        ckpt_sharp = torch.load(sharp_ckpt_path, map_location=device, weights_only=False)
        sharp_model.load_state_dict(ckpt_sharp["model_state_dict"])
        sharp_model.eval()

    # 3. TreeFlowNet Model
    flow_model = TreeFlowNet(pretrained_backbone=False).to(device)
    flow_ckpt_path = Path("DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth")
    if flow_ckpt_path.exists():
        ckpt_flow = torch.load(flow_ckpt_path, map_location=device, weights_only=False)
        flow_sd = ckpt_flow["model_state_dict"] if isinstance(ckpt_flow, dict) and "model_state_dict" in ckpt_flow else ckpt_flow
        flow_model.load_state_dict(flow_sd)
        flow_model.eval()

    # Select representative complex test crops
    sample_indices = [0, 1, 2]
    out_dir = Path("crown_segmentation_research")
    out_dir.mkdir(parents=True, exist_ok=True)

    for s_idx in sample_indices:
        item = val_dataset[s_idx]
        img_tensor = item["image"].unsqueeze(0).to(device)
        rgb_raw = (item["image"].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        raw_gt = item["instance_label"].squeeze().numpy().astype(np.int32)

        gt_map = np.zeros_like(raw_gt)
        gt_map[raw_gt > 0] = raw_gt[raw_gt > 0] + 1
        num_gt_trees = len(np.unique(raw_gt[raw_gt > 0]))

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                ws_preds = ws_model(img_tensor)
                flow_preds = flow_model(img_tensor) if flow_model else None
                sharp_preds = sharp_model(img_tensor) if sharp_model else None

        surf_p = ws_preds["surface"].squeeze().float().cpu().numpy()
        bound_p = ws_preds["boundary"].squeeze().float().cpu().numpy()
        canopy_p = ws_preds["canopy"].squeeze().float().cpu().numpy()

        # Method A: Baseline (Old failure parameters)
        markers_base, insts_base = decode_watershed_fast(
            surface=surf_p,
            boundary=bound_p,
            canopy=canopy_p,
            kernel_size=9,
            min_apex_val=0.28,
            min_canopy_val=0.38,
            pers_thresh=0.10,
            min_distance=10.0,
            bound_weight=1.0,
            min_area=30,
        )
        m_base = evaluate_metrics(markers_base, insts_base, gt_map)

        # Method B: Stage 1 Calibrated (Decoupled fine parameters)
        markers_st1, insts_st1 = decode_watershed_fast(
            surface=surf_p,
            boundary=bound_p,
            canopy=canopy_p,
            kernel_size=5,
            min_apex_val=0.25,
            min_canopy_val=0.25,
            pers_thresh=0.03,
            min_distance=8.0,
            bound_weight=0.5,
            min_area=30,
        )
        m_st1 = evaluate_metrics(markers_st1, insts_st1, gt_map)

        # Method C: Stage 2 Sharp-Ridge Retrained
        if sharp_preds:
            s_surf = sharp_preds["surface"].squeeze().float().cpu().numpy()
            s_bound = sharp_preds["boundary"].squeeze().float().cpu().numpy()
            s_canopy = sharp_preds["canopy"].squeeze().float().cpu().numpy()
            markers_st2, insts_st2 = decode_watershed_fast(
                surface=s_surf,
                boundary=s_bound,
                canopy=s_canopy,
                kernel_size=5,
                min_apex_val=0.25,
                min_canopy_val=0.30,
                pers_thresh=0.03,
                min_distance=8.0,
                bound_weight=1.0,
                min_area=30,
            )
            m_st2 = evaluate_metrics(markers_st2, insts_st2, gt_map)
        else:
            markers_st2, insts_st2 = markers_st1, insts_st1
            m_st2 = m_st1

        # Method D: Stage 3 Hamiltonian TopoFlow-Watershed
        if flow_preds:
            flow_p = flow_preds["flow"].squeeze().float().cpu().numpy()
            markers_st3, insts_st3 = decode_hamiltonian_topoflow(
                surface=surf_p,
                boundary=bound_p,
                canopy=canopy_p,
                flow=flow_p,
                kernel_size=5,
                min_apex_val=0.20,
                min_canopy_val=0.30,
                pers_thresh=0.03,
                min_distance=6.0,
                bound_weight=1.0,
                div_weight=2.0,
                min_area=25,
            )
            m_st3 = evaluate_metrics(markers_st3, insts_st3, gt_map)
        else:
            markers_st3, insts_st3 = markers_st1, insts_st1
            m_st3 = m_st1

        # Plot 5-Panel Master Comparison
        fig, axes = plt.subplots(1, 5, figsize=(30, 6.5), dpi=200)

        # 1. Ground Truth
        overlay_gt = render_overlay(rgb_raw, raw_gt)
        axes[0].imshow(overlay_gt)
        axes[0].set_title(f"GROUND TRUTH\nTrees: {num_gt_trees} | Reference Polygons", fontsize=11, fontweight="bold", color="darkgreen")
        axes[0].axis("off")

        # 2. Baseline CanopyWatershed
        overlay_base = render_overlay(rgb_raw, markers_base)
        axes[1].imshow(overlay_base)
        axes[1].set_title(
            f"BASELINE (Dilation-Coupled)\nTrees: {len(insts_base)} | mAP50: {m_base['ap50']:.1f}% | PQ: {m_base['pq']:.1f}%\n(Severe Over-merging Failure)",
            fontsize=10, fontweight="bold", color="darkred",
        )
        axes[1].axis("off")

        # 3. Stage 1 Calibrated
        overlay_st1 = render_overlay(rgb_raw, markers_st1)
        axes[2].imshow(overlay_st1)
        axes[2].set_title(
            f"STAGE 1: DECOUPLED SWEEP\nTrees: {len(insts_st1)} | mAP50: {m_st1['ap50']:.1f}% | PQ: {m_st1['pq']:.1f}%\n(Decoupled Kernel Calibration)",
            fontsize=10, fontweight="bold", color="navy",
        )
        axes[2].axis("off")

        # 4. Stage 2 Sharp-Ridge Retrained
        overlay_st2 = render_overlay(rgb_raw, markers_st2)
        axes[3].imshow(overlay_st2)
        axes[3].set_title(
            f"STAGE 2: SHARP-RIDGE RETRAIN\nTrees: {len(insts_st2)} | mAP50: {m_st2['ap50']:.1f}% | PQ: {m_st2['pq']:.1f}%\n(Contact Seam Barrier Loss)",
            fontsize=10, fontweight="bold", color="darkorange",
        )
        axes[3].axis("off")

        # 5. Stage 3 Hamiltonian TopoFlow-Watershed
        overlay_st3 = render_overlay(rgb_raw, markers_st3)
        axes[4].imshow(overlay_st3)
        axes[4].set_title(
            f"STAGE 3: HAMILTONIAN TOPOFLOW\nTrees: {len(insts_st3)} | mAP50: {m_st3['ap50']:.1f}% | PQ: {m_st3['pq']:.1f}%\n(Vector Divergence Relief)",
            fontsize=10, fontweight="bold", color="purple",
        )
        axes[4].axis("off")

        plt.suptitle(
            f"BAMFORESTS ITC Benchmark — Complete 3-Stage Breakthrough Comparison (Sample {s_idx+1})",
            fontsize=15, fontweight="bold", y=0.98,
        )
        plt.tight_layout()

        out_img_path = out_dir / f"master_3stage_comparison_sample_{s_idx}.png"
        plt.savefig(out_img_path, bbox_inches="tight")
        plt.close()
        print(f"[Master Visualizer] Saved comparison to {out_img_path}", flush=True)


if __name__ == "__main__":
    main()
