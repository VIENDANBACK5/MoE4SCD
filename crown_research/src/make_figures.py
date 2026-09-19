"""Progress figures + GT-vs-prediction preview overlays."""
from __future__ import annotations

import sys
from pathlib import Path

_conda_p = '/home/ubuntu/anaconda3/envs/cs2/lib/python3.11/site-packages'
_local_p = '/home/chung/.local/lib/python3.11/site-packages'
sys.path = [p for p in sys.path if p not in (_local_p, _conda_p)]
sys.path.insert(0, _conda_p)
sys.path.append(_local_p)
sys.path.insert(0, "/home/chung/RS/Image Segmentation")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crown_research.src.pred_cache import load_cache_rgb
from crown_research.src.rag_merge import fill_watershed_lines
from crown_research.src.seeding import flood, global_nms_seeds

sys.path.insert(0, "/home/chung/RS/Image Segmentation/crown_research/experiments/h5-boundary-calibration/code")
from run_h5 import merge_percentile  # noqa: E402
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import evaluate_metrics  # noqa: E402

FIG = Path("crown_research/to_human/figures")
PALETTE = np.random.default_rng(0).integers(60, 255, size=(4096, 3), dtype=np.uint8)


def colorise(labels, rgb, alpha=0.55):
    """Overlay instance labels on RGB with per-instance colour and white seams."""
    out = rgb.copy().astype(np.float32)
    m = labels >= 2
    col = PALETTE[np.clip(labels, 0, 4095)].astype(np.float32)
    out[m] = (1 - alpha) * out[m] + alpha * col[m]
    filled = fill_watershed_lines(labels.astype(np.int32))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(filled.astype(np.float32), cv2.MORPH_GRADIENT, k) > 0
    out[edges & (filled >= 2)] = 255.0
    return out.astype(np.uint8)


def oracle_cell(gt, bound):
    seeds = []
    for gid in np.unique(gt):
        if gid < 2:
            continue
        dt = cv2.distanceTransform((gt == gid).astype(np.uint8), cv2.DIST_L2, 3)
        y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
        seeds.append((int(y), int(x), 1.0))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seam = (cv2.morphologyEx(gt.astype(np.float32), cv2.MORPH_GRADIENT, k) > 0).astype(np.float32)
    energy = np.clip(cv2.GaussianBlur(seam, (0, 0), 1.0) * 3.0, 0, 1)
    cn = (gt >= 2).astype(np.float32)
    return flood(seeds, energy, cn)


def previews(n=3):
    FIG.mkdir(parents=True, exist_ok=True)
    for i, (surf, bound, canopy, gt, rgb) in enumerate(load_cache_rgb("evalhold60")):
        if i >= n:
            break
        panels = [("RGB input", rgb, None), ("Ground truth", colorise(gt, rgb), None)]

        m0, in0 = flood(global_nms_seeds(surf, canopy, 24.0, 0.25), bound, canopy)
        e0 = evaluate_metrics(m0, in0, gt)
        panels.append((f"Published (md=24)\nmAP@50 {e0['ap50']:.1f} | {len(in0)} inst", colorise(m0, rgb), None))

        m1, in1 = flood(global_nms_seeds(surf, canopy, 160.0, 0.50), bound, canopy)
        m2, in2 = merge_percentile(m1, bound, q=50, l_min=30)
        e2 = evaluate_metrics(m2, in2, gt)
        panels.append((f"Best found (md=160 + merge)\nmAP@50 {e2['ap50']:.1f} | {len(in2)} inst",
                       colorise(m2, rgb), None))

        mo, ino = oracle_cell(gt, bound)
        eo = evaluate_metrics(mo, ino, gt)
        panels.append((f"Oracle heads, same decoder\nmAP@50 {eo['ap50']:.1f} | {len(ino)} inst",
                       colorise(mo, rgb), None))

        panels.append(("Boundary head B (raw)\nmedian 0.001 - collapsed", bound, "magma"))

        fig, axes = plt.subplots(2, 3, figsize=(19, 13))
        for ax, (title, img, cmap) in zip(axes.ravel(), panels):
            ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.axis("off")
        fig.suptitle(f"Held-out crop {i} - GT vs predictions across the research arc",
                     fontsize=16, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG / f"preview_crop_{i}.png", dpi=72, bbox_inches="tight")
        plt.close(fig)
        print(f"  preview_crop_{i}.png", flush=True)


def plots():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

    # 1. Research trajectory
    traj = [("Published\n(md=24)", 44.29), ("H1 tuned\n(md=160)", 65.54),
            ("H2 seam\nmerge", 66.57), ("H5 percentile\nmerge", 66.91),
            ("H0 oracle\nseeds", 69.03), ("H2 oracle\nmerge", 69.74),
            ("H3 oracle\nheads", 97.30)]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs = np.arange(len(traj))
    cols = ["#888"] + ["#2a7fbf"] * 3 + ["#c98a2a"] * 2 + ["#2a9d5c"]
    ax.bar(xs, [v for _, v in traj], color=cols)
    for x, (_, v) in zip(xs, traj):
        ax.text(x, v + 1.2, f"{v:.1f}", ha="center", fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([t for t, _ in traj])
    ax.set_ylabel("mAP@50 (%) — 60 held-out crops"); ax.set_ylim(0, 108)
    ax.axhspan(69, 70.5, color="#c98a2a", alpha=0.12)
    ax.text(3.6, 71.5, "three independent oracle routes converge here", color="#8a5c00", fontsize=10)
    ax.set_title("Research trajectory: what moved the number, and what the ceiling is",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "trajectory.png", dpi=110); plt.close(fig)

    # 2. H3 factorial cube
    d = pd.read_csv("crown_research/experiments/h3-canopy-gate/results/h3_factorial.csv")
    d = d[d.split == "evalhold60 (hold-out)"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    lab = d.cell.str.replace("/", "\n", regex=False)
    c = ["#2a9d5c" if v >= 75 else ("#c0392b" if v < 50 else "#2a7fbf") for v in d.ap50]
    ax.bar(np.arange(len(d)), d.ap50, color=c)
    for x, v in zip(np.arange(len(d)), d.ap50):
        ax.text(x, v + 1.2, f"{v:.1f}", ha="center", fontweight="bold", fontsize=9)
    ax.axhline(65.54, ls="--", c="k", lw=1)
    ax.text(0.05, 67, "all-predicted baseline", fontsize=9)
    ax.set_xticks(np.arange(len(d))); ax.set_xticklabels(lab, fontsize=8)
    ax.set_ylabel("mAP@50 (%)"); ax.set_ylim(0, 108)
    ax.set_title("H3: every single-factor oracle HURTS; only the energy+canopy pair helps\n"
                 "(p = predicted, O = oracle)", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "h3_cube.png", dpi=110); plt.close(fig)

    # 3. Seeding frontier
    e = pd.read_csv("crown_research/experiments/h1-adaptive-seeding/results/h1a_extended_sweep.csv")
    e2 = pd.read_csv("crown_research/experiments/h1-adaptive-seeding/results/h1a_extended2_sweep.csv")
    a = pd.read_csv("crown_research/experiments/h1-adaptive-seeding/results/h1a_global_sweep.csv")
    al = pd.concat([a, e, e2], ignore_index=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sc = ax.scatter(al.ratio, al.ap50, c=al.min_dist, cmap="viridis", s=42)
    ax.axhline(69.03, ls="--", c="#c98a2a"); ax.text(2.4, 69.8, "oracle seeds (69.0)", color="#8a5c00")
    ax.axvline(1.0, ls=":", c="k", alpha=0.5)
    ax.set_xlabel("seeds per ground-truth crown"); ax.set_ylabel("mAP@50 (%) — tuning split")
    ax.set_title("H1: global NMS rides a precision/recall frontier it cannot leave",
                 fontweight="bold")
    fig.colorbar(sc, label="min_dist (px)")
    fig.tight_layout(); fig.savefig(FIG / "seeding_frontier.png", dpi=110); plt.close(fig)

    # 4. Boundary head calibration
    hd = pd.read_csv("crown_research/experiments/h4-coupled-gate/results/head_diagnostic.csv")
    hd = hd[hd.split == "evalhold60"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    r = hd[hd["head"] == "boundary B"].iloc[0]
    ax1.bar(["on true\nseams", "elsewhere"], [r.mean_on_pos, r.mean_on_neg], color=["#c0392b", "#999"])
    ax1.set_ylabel("mean predicted boundary value"); ax1.set_ylim(0, 1)
    ax1.axhline(0.35, ls="--", c="k"); ax1.text(0.5, 0.37, "H2 merge threshold that never fired", fontsize=9)
    ax1.set_title(f"Boundary head is collapsed\n(AUROC {r.AUROC:.3f} — it ranks fine)", fontweight="bold")
    names = ["boundary B\nvs seams", "canopy C\nvs foreground", "1-canopy\nvs seams", "surface U\nvs foreground"]
    vals = [hd[hd["head"] == h].iloc[0].AUROC for h in ["boundary B", "canopy C", "1 - canopy C", "surface U"]]
    ax2.bar(names, vals, color=["#c0392b", "#2a9d5c", "#999", "#2a7fbf"])
    for x, v in enumerate(vals):
        ax2.text(x, v + 0.015, f"{v:.3f}", ha="center", fontweight="bold")
    ax2.axhline(0.5, ls="--", c="k"); ax2.set_ylim(0, 1.08); ax2.set_ylabel("AUROC")
    ax2.set_title("Which heads carry signal", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "head_calibration.png", dpi=110); plt.close(fig)
    print("  4 plots written", flush=True)


if __name__ == "__main__":
    plots()
    previews(3)
