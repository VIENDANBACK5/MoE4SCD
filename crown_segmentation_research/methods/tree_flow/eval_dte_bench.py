"""Evaluate TreeFlowNet against the OFFICIAL DTE-aerial-bench (525 patches across 5 global biomes).

DTE-aerial-bench contains PIXEL-WISE semantic masks (0=background, 1=live-tree-cover, 2=standing-deadwood/mortality).
TreeFlowNet predicts multi-task centripetal flow, SDT boundaries, centroid heatmaps, and canopy probability.
We evaluate both:
1. Canopy / Tree-Cover IoU and F1 (mask > 0 vs canopy_pred)
2. Deadwood / Instance polygon rasterized segmentation F1 (mask == 2 vs decoded polygons)
"""

from __future__ import annotations


# Ensure workspace root is in sys.path
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "crown_segmentation_research").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from shapely.geometry import MultiPolygon

from crown_segmentation_research.methods.tree_flow.decode import decode_flow_to_instances
from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet

BENCH_DIR = Path("DTE-Aerial-Data-public")
MIN_PRED_AREA = 30


def load_model(path: Path, device: torch.device) -> TreeFlowNet:
    model = TreeFlowNet(pretrained_backbone=False).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def polygons_to_binary_mask(polygons, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        subs = polygon.geoms if isinstance(polygon, MultiPolygon) else [polygon]
        for sub in subs:
            coords = np.array(sub.exterior.coords).round().astype(np.int32)
            cv2.fillPoly(mask, [coords], 1)
    return mask.astype(bool)


def pixel_confusion(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    tp = int(np.logical_and(pred_mask, gt_mask).sum())
    fp = int(np.logical_and(pred_mask, ~gt_mask).sum())
    fn = int(np.logical_and(~pred_mask, gt_mask).sum())
    return {"tp": tp, "fp": fp, "fn": fn}


def aggregate(rows: list[dict]) -> dict:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou, "tp": tp, "fp": fp, "fn": fn, "n": len(rows)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = pd.read_csv(BENCH_DIR / "DTE-aerial-bench-meta-public-assets.csv")
    print(f"Loaded {len(meta)} bench patches across {meta['biome'].nunique()} biomes, "
          f"resolutions {sorted(meta['resolution'].unique())}", flush=True)

    ckpt_path = Path("DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth")
    model = load_model(ckpt_path, device)

    canopy_rows = []
    instance_rows = []

    with torch.no_grad():
        for i, row in meta.iterrows():
            image = np.array(Image.open(BENCH_DIR / row["tile_path"]).convert("RGB"))
            mask = np.array(Image.open(BENCH_DIR / row["mask_path"]))
            gt_canopy = mask > 0
            gt_mortality = mask == 2

            image_f = image.astype(np.float32) / 255.0
            image_t = torch.from_numpy(image_f).permute(2, 0, 1).float().unsqueeze(0).to(device)

            out = model(image_t)
            flow = out["flow"].squeeze(0).cpu().numpy().astype(np.float32)
            sdt = out["sdt"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            centroid = out["centroid"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            canopy = out["canopy"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

            # 1. Canopy semantic evaluation
            pred_canopy = canopy >= 0.5
            conf_canopy = pixel_confusion(pred_canopy, gt_canopy)
            canopy_rows.append({
                "patch_stem": row["patch_stem"], "biome": row["biome"],
                "resolution": row["resolution"], **conf_canopy,
            })

            # 2. Instance flow decode evaluation
            _, polygons = decode_flow_to_instances(
                flow, canopy, sdt, centroid,
                canopy_threshold=0.5,
                sdt_threshold=-0.2,
                centroid_threshold=0.25,
                min_instance_area=MIN_PRED_AREA,
            )
            pred_inst_mask = polygons_to_binary_mask(polygons, image.shape[:2])
            conf_mortality = pixel_confusion(pred_inst_mask, gt_mortality)
            instance_rows.append({
                "patch_stem": row["patch_stem"], "biome": row["biome"],
                "resolution": row["resolution"], "n_pred_polygons": len(polygons),
                **conf_mortality,
            })

            if (i + 1) % 100 == 0 or (i + 1) == len(meta):
                print(f"[TreeFlowNet DTE Bench] {i+1}/{len(meta)} done", flush=True)

    df_canopy = pd.DataFrame(canopy_rows)
    df_inst = pd.DataFrame(instance_rows)

    canopy_overall = aggregate(canopy_rows)
    inst_overall = aggregate(instance_rows)

    canopy_by_biome = {biome: aggregate(g.to_dict("records")) for biome, g in df_canopy.groupby("biome")}
    canopy_by_resolution = {res: aggregate(g.to_dict("records")) for res, g in df_canopy.groupby("resolution")}

    inst_by_biome = {biome: aggregate(g.to_dict("records")) for biome, g in df_inst.groupby("biome")}
    inst_by_resolution = {res: aggregate(g.to_dict("records")) for res, g in df_inst.groupby("resolution")}

    results = {
        "tree_cover_overall": canopy_overall,
        "tree_cover_by_biome": canopy_by_biome,
        "tree_cover_by_resolution": canopy_by_resolution,
        "mortality_overall": inst_overall,
        "mortality_by_biome": inst_by_biome,
        "mortality_by_resolution": inst_by_resolution,
    }

    out_path = Path("DeadTrees/experiments/tree_flow_dte_bench_eval.json")
    out_path.write_text(json.dumps(results, indent=2))
    df_inst.to_csv(Path("DeadTrees/experiments/tree_flow_dte_bench_per_patch.csv"), index=False)

    print("\n================ DTE-AERIAL-BENCH RESULTS (TreeFlowNet) ================")
    print(f"Overall Canopy IoU: {canopy_overall['iou']:.4f} | Canopy F1: {canopy_overall['f1']:.4f}")
    print(f"Overall Mortality F1: {inst_overall['f1']:.4f} | Precision: {inst_overall['precision']:.4f} | Recall: {inst_overall['recall']:.4f}")
    print("========================================================================")


if __name__ == "__main__":
    main()
