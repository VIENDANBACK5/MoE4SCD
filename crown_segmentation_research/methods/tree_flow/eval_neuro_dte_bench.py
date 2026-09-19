"""Evaluate Neuro-Flow-Graph (NFG) against OFFICIAL DTE-aerial-bench (525 patches across 5 biomes).

Compares:
  1. Vanilla TreeFlowNet baseline
  2. Full Neuro-Flow-Graph (NFG: Euler Flow + Bellman-Ford Bridging + Spectral Modularity Cut)
Evaluates per-biome and per-resolution breakdown for:
  - Canopy / Tree Cover IoU and F1
  - Deadwood / Standing & Fallen Tree Instance F1, Precision, Recall
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

from crown_segmentation_research.methods.tree_flow.graph_decode import decode_neuro_flow_graph
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


def evaluate_bench_pipeline(
    model: TreeFlowNet,
    meta: pd.DataFrame,
    device: torch.device,
    use_bellman_ford: bool = True,
    use_spectral_cut: bool = True,
) -> dict:
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

            # 2. Neuro-Flow-Graph instance decode evaluation with GSD-Adaptive Scaling & Divergence
            _, polygons = decode_neuro_flow_graph(
                flow, canopy, sdt, centroid,
                resolution=row["resolution"],
                canopy_threshold=0.40,
                sdt_threshold=-0.25,
                centroid_threshold=0.15,
                use_divergence_seeds=True,
                use_bellman_ford=use_bellman_ford,
                use_spectral_cut=use_spectral_cut,
            )
            pred_inst_mask = polygons_to_binary_mask(polygons, image.shape[:2])
            conf_mortality = pixel_confusion(pred_inst_mask, gt_mortality)
            instance_rows.append({
                "patch_stem": row["patch_stem"], "biome": row["biome"],
                "resolution": row["resolution"], "n_pred_polygons": len(polygons),
                **conf_mortality,
            })

            if (i + 1) % 100 == 0 or (i + 1) == len(meta):
                print(f"[NFG DTE Bench] {i+1}/{len(meta)} done", flush=True)

    df_canopy = pd.DataFrame(canopy_rows)
    df_inst = pd.DataFrame(instance_rows)

    canopy_overall = aggregate(canopy_rows)
    inst_overall = aggregate(instance_rows)

    canopy_by_biome = {biome: aggregate(g.to_dict("records")) for biome, g in df_canopy.groupby("biome")}
    canopy_by_resolution = {res: aggregate(g.to_dict("records")) for res, g in df_canopy.groupby("resolution")}

    inst_by_biome = {biome: aggregate(g.to_dict("records")) for biome, g in df_inst.groupby("biome")}
    inst_by_resolution = {res: aggregate(g.to_dict("records")) for res, g in df_inst.groupby("resolution")}

    return {
        "tree_cover_overall": canopy_overall,
        "tree_cover_by_biome": canopy_by_biome,
        "tree_cover_by_resolution": canopy_by_resolution,
        "mortality_overall": inst_overall,
        "mortality_by_biome": inst_by_biome,
        "mortality_by_resolution": inst_by_resolution,
        "df_inst": df_inst,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = pd.read_csv(BENCH_DIR / "DTE-aerial-bench-meta-public-assets.csv")
    print(f"Loaded {len(meta)} bench patches across {meta['biome'].nunique()} biomes, "
          f"resolutions {sorted(meta['resolution'].unique())}", flush=True)

    ckpt_path = Path("DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth")
    model = load_model(ckpt_path, device)

    print("\n--- Running Full Neuro-Flow-Graph (NFG) on 525 DTE Benchmark Patches ---")
    nfg_res = evaluate_bench_pipeline(model, meta, device, use_bellman_ford=True, use_spectral_cut=True)

    out_json = {
        "tree_cover_overall": nfg_res["tree_cover_overall"],
        "tree_cover_by_biome": nfg_res["tree_cover_by_biome"],
        "tree_cover_by_resolution": nfg_res["tree_cover_by_resolution"],
        "mortality_overall": nfg_res["mortality_overall"],
        "mortality_by_biome": nfg_res["mortality_by_biome"],
        "mortality_by_resolution": nfg_res["mortality_by_resolution"],
    }

    out_path = Path("DeadTrees/experiments/neuro_flow_dte_bench_eval.json")
    out_path.write_text(json.dumps(out_json, indent=2))
    nfg_res["df_inst"].to_csv(Path("DeadTrees/experiments/neuro_flow_dte_bench_per_patch.csv"), index=False)

    print("\n================ OFFICIAL DTE-AERIAL-BENCH: NEURO-FLOW-GRAPH (NFG) ================")
    print(f"Overall Canopy IoU: {nfg_res['tree_cover_overall']['iou']:.4f} | Canopy F1: {nfg_res['tree_cover_overall']['f1']:.4f}")
    print(f"Overall Mortality F1: {nfg_res['mortality_overall']['f1']:.4f} | Precision: {nfg_res['mortality_overall']['precision']:.4f} | Recall: {nfg_res['mortality_overall']['recall']:.4f}")
    print("-----------------------------------------------------------------------------------")
    print("Breakdown by Biome:")
    for biome, b_res in nfg_res["mortality_by_biome"].items():
        print(f"  * {biome:30s} | F1: {b_res['f1']:.4f} | P: {b_res['precision']:.4f} | R: {b_res['recall']:.4f} | n={b_res['n']}")
    print("-----------------------------------------------------------------------------------")
    print("Breakdown by Resolution:")
    for res, r_res in nfg_res["mortality_by_resolution"].items():
        print(f"  * {res:30s} | F1: {r_res['f1']:.4f} | P: {r_res['precision']:.4f} | R: {r_res['recall']:.4f} | n={r_res['n']}")
    print("===================================================================================")


if __name__ == "__main__":
    main()
