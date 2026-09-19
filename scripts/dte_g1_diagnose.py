#!/usr/bin/env python3
"""DTE-aerial Gate-1 failure diagnosis.

Runs immediately after G0 PASS. Produces:
  - Per-tile per-class metrics stratified by resolution / biome / site
  - Connected-component diagnostics (miss/split/merge) for mortality
  - Physical-area bins: recall vs GSD vs component size
  - Visual figures: same-scene rows, error maps, recall-vs-area curve

Usage (background, after G0 PASS):
    nohup python scripts/dte_g1_diagnose.py \\
        --bench  datasets/DTE-aerial-bench \\
        --meta   datasets/DTE-aerial-bench/DTE-aerial-bench-meta.csv \\
        --preds  dte_runs/g0_parity/predictions \\
        --out    dte_runs/g1_diagnosis \\
        > dte_runs/logs/g1_diagnose.log 2>&1 &
    echo $! > dte_runs/logs/g1_diagnose.pid
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

warnings.filterwarnings("ignore")

RESOLUTIONS = ["5cm", "10cm", "20cm"]
CLASS_NAMES  = {0: "background", 1: "tree-cover", 2: "mortality"}
MORTALITY_IDX = 2
# Minimum overlap fraction to declare a GT–Pred component pair "connected"
OVERLAP_THRESH = 0.10
# Physical area bins (m²)
AREA_BINS = [0, 1, 4, 16, 64, 256, 1024, float("inf")]
AREA_LABELS = ["<1", "1-4", "4-16", "16-64", "64-256", "256-1k", ">1k"]


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─────────────────────────────────────────────
# Resolution → GSD (m/px)
# ─────────────────────────────────────────────
GSD_MAP = {"5cm": 0.05, "10cm": 0.10, "20cm": 0.20}


# ─────────────────────────────────────────────
# Component diagnostics
# ─────────────────────────────────────────────
def component_diagnostics(gt: np.ndarray, pred: np.ndarray, gsd: float) -> list[dict]:
    """Per-component analysis for mortality class.

    Returns one record per GT component with fields:
        area_px, area_m2, matched, tp_px, fp_px, fn_px,
        n_pred_partners (split), is_merged (bool, from pred side)
    """
    # Binary masks for mortality
    gt_bin   = (gt == MORTALITY_IDX).astype(np.uint8)
    pred_bin = (pred == MORTALITY_IDX).astype(np.uint8)

    gt_labeled,   n_gt   = ndimage.label(gt_bin)
    pred_labeled, n_pred = ndimage.label(pred_bin)

    records = []
    for gid in range(1, n_gt + 1):
        gt_comp = gt_labeled == gid
        area_px = int(gt_comp.sum())
        area_m2 = area_px * (gsd ** 2)

        # Find overlapping pred components
        pred_ids_in_comp = np.unique(pred_labeled[gt_comp])
        pred_ids_in_comp = pred_ids_in_comp[pred_ids_in_comp > 0]

        # Filter by overlap fraction threshold
        partners = []
        for pid in pred_ids_in_comp:
            pred_comp = pred_labeled == pid
            intersection = int((gt_comp & pred_comp).sum())
            smaller = min(area_px, int(pred_comp.sum()))
            if smaller > 0 and intersection / smaller >= OVERLAP_THRESH:
                partners.append(pid)

        matched  = len(partners) > 0
        tp_px    = int((gt_comp & pred_bin).sum())
        fn_px    = int(gt_comp.sum()) - tp_px

        # Split: GT component matched to >1 pred component
        n_split = len(partners)

        records.append({
            "gt_comp_id":    gid,
            "area_px":       area_px,
            "area_m2":       area_m2,
            "matched":       matched,
            "tp_px":         tp_px,
            "fn_px":         fn_px,
            "n_pred_partners": n_split,
            "is_split":      n_split > 1,
            "is_missed":     not matched,
        })

    # Merge: pred component overlaps with >1 GT component
    for pid in range(1, n_pred + 1):
        pred_comp = pred_labeled == pid
        gt_ids_in_pred = np.unique(gt_labeled[pred_comp])
        gt_ids_in_pred = gt_ids_in_pred[gt_ids_in_pred > 0]
        if len(gt_ids_in_pred) > 1:
            for gid_match in gt_ids_in_pred:
                for rec in records:
                    if rec["gt_comp_id"] == gid_match:
                        rec["is_merged"] = True

    # Ensure field exists
    for rec in records:
        rec.setdefault("is_merged", False)

    return records


# ─────────────────────────────────────────────
# Core per-tile analysis
# ─────────────────────────────────────────────
def analyse_tile(row: pd.Series, bench_dir: Path, pred_dir: Path) -> dict:
    tile_name = Path(row["tile_path"]).stem
    gt  = np.array(Image.open(bench_dir / row["mask_path"]))
    pred_path = pred_dir / f"{tile_name}_pred.npy"
    if not pred_path.exists():
        return {}
    pred = np.load(pred_path)

    gsd = GSD_MAP.get(str(row["resolution"]), 0.10)
    valid = gt != 255

    result = {
        "tile":       tile_name,
        "ortho_id":   str(row["site"]),
        "resolution": str(row["resolution"]),
        "biome":      str(row["biome"]),
        "gsd":        gsd,
    }

    # Per-class pixel stats
    K = 3
    gt_flat   = gt.ravel().astype(np.int64)[valid.ravel()]
    pred_flat = pred.ravel().astype(np.int64)[valid.ravel()]
    cm = np.zeros((K, K), dtype=np.int64)
    np.add.at(cm, (gt_flat, pred_flat), 1)
    for c in range(K):
        result[f"tp_{c}"] = int(cm[c, c])
        result[f"fp_{c}"] = int(cm[:, c].sum() - cm[c, c])
        result[f"fn_{c}"] = int(cm[c, :].sum() - cm[c, c])

    # Component diagnostics for mortality
    gt_valid  = gt.copy();  gt_valid[~valid] = 0
    pred_valid = pred.copy(); pred_valid[~valid] = 0
    comps = component_diagnostics(gt_valid, pred_valid, gsd)
    result["components"] = comps
    result["n_gt_components"]   = len(comps)
    result["n_missed"]          = sum(1 for c in comps if c["is_missed"])
    result["n_split"]           = sum(1 for c in comps if c["is_split"])
    result["n_merged"]          = sum(1 for c in comps if c["is_merged"])

    return result


# ─────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────
PALETTE = {0: [50, 50, 50], 1: [34, 139, 34], 2: [220, 50, 30]}


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in PALETTE.items():
        rgb[mask == label] = color
    rgb[mask == 255] = [128, 128, 128]
    return rgb


def error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """TP=green, FP=red, FN=blue, correct-background=black."""
    h, w = gt.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    valid = gt != 255
    tp = valid & (gt == MORTALITY_IDX) & (pred == MORTALITY_IDX)
    fp = valid & (gt != MORTALITY_IDX) & (pred == MORTALITY_IDX)
    fn = valid & (gt == MORTALITY_IDX) & (pred != MORTALITY_IDX)
    rgb[tp] = [34, 200, 34]    # green
    rgb[fp] = [220, 30, 30]    # red
    rgb[fn] = [30, 80, 220]    # blue
    return rgb


def figure_same_scene(site_rows: pd.DataFrame, bench_dir: Path, pred_dir: Path, out_dir: Path):
    """Figure A: RGB | GT | Pred for each resolution of the same site."""
    site = site_rows.iloc[0]["site"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f"Site {site} — same scene across GSD", fontsize=13)

    for row_i, res in enumerate(RESOLUTIONS):
        sub = site_rows[site_rows["resolution"] == res]
        if len(sub) == 0:
            continue
        row = sub.iloc[0]   # pick first tile
        tile_name = Path(row["tile_path"]).stem
        img  = np.array(Image.open(bench_dir / row["tile_path"]).convert("RGB"))
        gt   = np.array(Image.open(bench_dir / row["mask_path"]))
        pred_path = pred_dir / f"{tile_name}_pred.npy"
        pred = np.load(pred_path) if pred_path.exists() else np.zeros_like(gt)

        axes[row_i, 0].imshow(img);            axes[row_i, 0].set_title(f"RGB {res}");   axes[row_i, 0].axis("off")
        axes[row_i, 1].imshow(mask_to_rgb(gt)); axes[row_i, 1].set_title(f"GT {res}");    axes[row_i, 1].axis("off")
        axes[row_i, 2].imshow(mask_to_rgb(pred)); axes[row_i, 2].set_title(f"MiT-B3 {res}"); axes[row_i, 2].axis("off")

    plt.tight_layout()
    out = out_dir / f"figA_same_scene_{site}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_error_map(site_rows: pd.DataFrame, bench_dir: Path, pred_dir: Path, out_dir: Path):
    """Figure B: Error map (TP/FP/FN) per resolution for a site."""
    site = site_rows.iloc[0]["site"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f"Site {site} — Error Maps (green=TP, red=FP, blue=FN)", fontsize=13)

    for row_i, res in enumerate(RESOLUTIONS):
        sub = site_rows[site_rows["resolution"] == res]
        if len(sub) == 0:
            continue
        row = sub.iloc[0]
        tile_name = Path(row["tile_path"]).stem
        img  = np.array(Image.open(bench_dir / row["tile_path"]).convert("RGB"))
        gt   = np.array(Image.open(bench_dir / row["mask_path"]))
        pred_path = pred_dir / f"{tile_name}_pred.npy"
        pred = np.load(pred_path) if pred_path.exists() else np.zeros_like(gt)

        axes[row_i, 0].imshow(img);               axes[row_i, 0].set_title(f"RGB {res}");        axes[row_i, 0].axis("off")
        axes[row_i, 1].imshow(mask_to_rgb(gt));   axes[row_i, 1].set_title(f"GT {res}");         axes[row_i, 1].axis("off")
        axes[row_i, 2].imshow(mask_to_rgb(pred)); axes[row_i, 2].set_title(f"Pred {res}");       axes[row_i, 2].axis("off")
        axes[row_i, 3].imshow(error_map(gt, pred)); axes[row_i, 3].set_title(f"Error {res}");   axes[row_i, 3].axis("off")

    plt.tight_layout()
    out = out_dir / f"figB_error_map_{site}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_recall_vs_area(comp_df: pd.DataFrame, out_dir: Path):
    """Figure D: Mortality recall vs physical component area, per resolution."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for res, color in zip(RESOLUTIONS, ["steelblue", "darkorange", "crimson"]):
        sub = comp_df[comp_df["resolution"] == res]
        if len(sub) == 0:
            continue
        # Bin by area_m2
        bins = AREA_BINS
        bin_recalls = []
        bin_mids    = []
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            mask = (sub["area_m2"] >= lo) & (sub["area_m2"] < hi)
            total = mask.sum()
            if total == 0:
                continue
            matched = sub.loc[mask, "matched"].sum()
            bin_recalls.append(matched / total)
            mid = (lo + hi) / 2 if hi < 1e9 else lo * 2
            bin_mids.append(mid)
        if bin_mids:
            ax.plot(bin_mids, bin_recalls, "o-", label=res, color=color)

    ax.set_xscale("log")
    ax.set_xlabel("Physical component area (m²)", fontsize=12)
    ax.set_ylabel("Recall (% GT components matched)", fontsize=12)
    ax.set_title("Mortality recall vs. physical size — MiT-B3 baseline", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    out = out_dir / "figD_recall_vs_area.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ─────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────
def compute_f1(tp, fp, fn, eps=1e-8):
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    return 2 * p * r / (p + r + eps)


def write_diagnosis_report(df: pd.DataFrame, comp_df: pd.DataFrame, out_dir: Path):
    lines = ["# G1 Minimal Diagnosis Report", ""]

    lines += ["## Per-resolution metrics (site-macro F1)", "",
              "| Resolution | Mortality F1 | Tree-cover F1 | Mort Recall | Mort Precision |",
              "|---|---:|---:|---:|---:|"]
    for res in RESOLUTIONS:
        sub = df[df["resolution"] == res]
        if len(sub) == 0:
            lines.append(f"| {res} | — | — | — | — |"); continue
        mort_f1s = []
        for _, grp in sub.groupby("ortho_id"):
            tp2 = grp["tp_2"].sum(); fp2 = grp["fp_2"].sum(); fn2 = grp["fn_2"].sum()
            mort_f1s.append(compute_f1(tp2, fp2, fn2))
        mort_f1 = np.mean(mort_f1s)
        tp1 = sub["tp_1"].sum(); fp1 = sub["fp_1"].sum(); fn1 = sub["fn_1"].sum()
        tree_f1 = compute_f1(tp1, fp1, fn1)
        tp2 = sub["tp_2"].sum(); fp2 = sub["fp_2"].sum(); fn2 = sub["fn_2"].sum()
        prec = tp2/(tp2+fp2+1e-8); rec = tp2/(tp2+fn2+1e-8)
        lines.append(f"| {res} | {mort_f1:.4f} | {tree_f1:.4f} | {rec:.4f} | {prec:.4f} |")

    lines += ["", "## Component diagnostics (mortality)", "",
              "| Resolution | GT comps | Missed | Split | Merged | Miss% |",
              "|---|---:|---:|---:|---:|---:|"]
    for res in RESOLUTIONS:
        sub = comp_df[comp_df["resolution"] == res]
        if len(sub) == 0:
            lines.append(f"| {res} | — | — | — | — | — |"); continue
        total   = len(sub)
        missed  = sub["is_missed"].sum()
        split   = sub["is_split"].sum()
        merged  = sub["is_merged"].sum()
        miss_pct = 100 * missed / (total + 1e-8)
        lines.append(f"| {res} | {total} | {missed} | {split} | {merged} | {miss_pct:.1f}% |")

    lines += ["", "## Key failure signals", "", "*(Filled in after visual inspection)*", "",
              "- [ ] Small regions disappear at 20cm → Recall collapse",
              "- [ ] Boundary contraction",
              "- [ ] Mortality → tree-cover confusion",
              "- [ ] Merge into canopy",
              "- [ ] False positives increase",
              "",
              "## Proposed mechanism (to be confirmed)",
              "",
              "*(Select one after reviewing figures)*",
              "",
              "- [ ] A. Small mortality structures destroyed by resolution shift",
              "- [ ] B. GSD-dependent feature distribution shift",
              "- [ ] C. Boundary/structural collapse",
              "",
              "## Next step",
              "",
              "Based on dominant mechanism, proceed to G2 (method design). "
              "Run DeepLabV3+ sanity baseline to confirm task-level failure.",
              ]

    (out_dir / "minimal_diagnosis.md").write_text("\n".join(lines))
    log(f"  Diagnosis report → {out_dir/'minimal_diagnosis.md'}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("DTE G1 failure diagnosis")
    p.add_argument("--bench",   required=True)
    p.add_argument("--meta",    required=True)
    p.add_argument("--preds",   required=True, help="Directory with *_pred.npy files")
    p.add_argument("--out",     default="dte_runs/g1_diagnosis")
    p.add_argument("--n-visual-sites", type=int, default=3, help="Sites to visualize")
    return p.parse_args()


def main():
    args      = parse_args()
    bench_dir = Path(args.bench)
    meta_csv  = Path(args.meta)
    pred_dir  = Path(args.preds)
    out_dir   = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    log(f"DTE G1 diagnosis  |  preds={pred_dir}  |  out={out_dir}")

    df_meta = pd.read_csv(meta_csv)

    # Per-tile analysis
    log("Analysing tiles …")
    tile_records = []
    all_comps    = []

    for _, row in df_meta.iterrows():
        result = analyse_tile(row, bench_dir, pred_dir)
        if not result:
            log(f"  SKIP (no pred): {row['tile_path']}")
            continue
        comps = result.pop("components", [])
        for c in comps:
            c["tile"]       = result["tile"]
            c["ortho_id"]   = result["ortho_id"]
            c["resolution"] = result["resolution"]
            c["biome"]      = result["biome"]
            c["gsd"]        = result["gsd"]
        all_comps.extend(comps)
        tile_records.append(result)

    df_tiles = pd.DataFrame(tile_records)
    df_tiles.to_parquet(out_dir / "per_tile_diagnosis.parquet", index=False)
    log(f"  Saved per-tile → {out_dir/'per_tile_diagnosis.parquet'}")

    comp_df = pd.DataFrame(all_comps) if all_comps else pd.DataFrame()
    if not comp_df.empty:
        comp_df.to_parquet(out_dir / "components.parquet", index=False)
        log(f"  Saved components → {out_dir/'components.parquet'}")

    # Visualizations
    log("Generating figures …")
    fig_dir = out_dir / "figures"
    sites_to_viz = df_meta["site"].unique()[:args.n_visual_sites]

    for site in sites_to_viz:
        site_rows = df_meta[df_meta["site"] == site]
        try:
            figure_same_scene(site_rows, bench_dir, pred_dir, fig_dir)
            figure_error_map(site_rows, bench_dir, pred_dir, fig_dir)
        except Exception as e:
            log(f"  Figure error for site {site}: {e}")

    if not comp_df.empty:
        try:
            figure_recall_vs_area(comp_df, fig_dir)
        except Exception as e:
            log(f"  Recall-vs-area figure error: {e}")

    # Summary report
    write_diagnosis_report(df_tiles, comp_df, out_dir)

    log("G1 done. Review:")
    log(f"  Figures: {fig_dir}/")
    log(f"  Report:  {out_dir/'minimal_diagnosis.md'}")
    log("Fill in the failure mechanism, then proceed to G2 (method design).")


if __name__ == "__main__":
    main()
