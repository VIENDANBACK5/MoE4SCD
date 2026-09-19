"""Second-round figures: COCO AP trajectory, H6 effect, cross-site transfer."""
from __future__ import annotations

import sys
from pathlib import Path

_c = '/home/ubuntu/anaconda3/envs/cs2/lib/python3.11/site-packages'
_l = '/home/chung/.local/lib/python3.11/site-packages'
sys.path = [p for p in sys.path if p not in (_l, _c)]
sys.path.insert(0, _c); sys.path.append(_l)
sys.path.insert(0, "/home/chung/RS/Image Segmentation")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = Path("crown_research/to_human/figures")
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .3})


def trajectory_ap():
    steps = [("Published\ndecode consts", 15.81), ("H1 tuned\n(md=160)", 23.85),
             ("H5 percentile\nmerge", 24.32), ("H6 calibrated\nboundary head", 26.84),
             ("H3 oracle heads\n(same decoder)", 84.48)]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    x = np.arange(len(steps))
    cols = ["#888", "#2a7fbf", "#2a7fbf", "#1f6f4a", "#9a6a12"]
    ax.bar(x, [v for _, v in steps], color=cols)
    for xi, (_, v) in zip(x, steps):
        ax.text(xi, v + 1.5, f"{v:.2f}", ha="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([s for s, _ in steps])
    ax.set_ylabel("COCO AP @[.5:.95] — 60 held-out crops"); ax.set_ylim(0, 95)
    ax.set_title("Research trajectory in true COCO AP (not the repo's mislabelled F1)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "trajectory_ap.png", dpi=110); plt.close(fig)


def h6_effect():
    m = ["COCO AP", "AP50", "AP75", "F1@75", "PQ", "SQ", "B-IoU"]
    old = [24.32, 56.56, 16.48, 30.45, 48.82, 72.78, 12.84]
    new = [26.84, 58.06, 20.40, 37.06, 49.55, 73.79, 13.72]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8),
                                 gridspec_kw={"width_ratios": [1.7, 1]})
    x = np.arange(len(m)); w = .38
    a1.bar(x - w/2, old, w, label="old head", color="#9aa79a")
    a1.bar(x + w/2, new, w, label="H6 calibrated head", color="#1f6f4a")
    for xi, (o, n) in zip(x, zip(old, new)):
        a1.text(xi + w/2, n + .9, f"+{n-o:.1f}", ha="center", fontsize=9,
                fontweight="bold", color="#1f6f4a")
    a1.set_xticks(x); a1.set_xticklabels(m, fontsize=9); a1.legend(frameon=False)
    a1.set_title("H6: the gain is geometric, not recognitional\n"
                 "(F1@50 moved +0.07 and is omitted)", fontweight="bold")

    cats = ["mean B\non true seams", "mean B\nelsewhere"]
    a2.bar(np.arange(2) - .19, [0.105, 0.013], .38, label="old", color="#9aa79a")
    a2.bar(np.arange(2) + .19, [0.186, 0.041], .38, label="H6", color="#1f6f4a")
    a2.axhline(0.40, ls="--", c="#9e3b2f")
    a2.text(-.4, .42, "pre-registered criterion (missed)", color="#9e3b2f", fontsize=9)
    a2.set_xticks(np.arange(2)); a2.set_xticklabels(cats, fontsize=9)
    a2.set_ylim(0, .55); a2.legend(frameon=False)
    a2.set_title("Calibration only half-fixed:\nseparation fell 8.0x -> 4.5x", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "h6_effect.png", dpi=110); plt.close(fig)


def cross_site():
    sw = pd.read_csv("crown_research/experiments/h8-cross-site/results/h8_sweep.csv")
    ex = pd.read_csv("crown_research/experiments/h8-cross-site/results/h8_extended_sweep.csv")
    al = pd.concat([sw, ex], ignore_index=True).drop_duplicates(["split", "min_dist"])
    D = {"evalhold60": 210.6, "test1": 314.0, "test2": 211.6}
    cols = {"evalhold60": "#1f6f4a", "test2": "#2a7fbf", "test1": "#9e3b2f"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for s, g in al.groupby("split"):
        g = g.sort_values("min_dist")
        a1.plot(g.min_dist, g["COCO AP"], "o-", color=cols[s], label=f"{s} (D={D[s]:.0f}px)")
        b = g.sort_values("COCO AP", ascending=False).iloc[0]
        a1.scatter([b.min_dist], [b["COCO AP"]], s=150, facecolors="none",
                   edgecolors=cols[s], linewidths=2.2, zorder=5)
    a1.axvline(24, ls=":", c="k"); a1.text(28, 14, "published 24px", fontsize=9)
    a1.axvline(160, ls="--", c="#1f6f4a", alpha=.6); a1.text(166, 19, "frozen 160px", fontsize=9)
    a1.set_xlabel("min_dist (px)"); a1.set_ylabel("COCO AP")
    a1.legend(frameon=False, fontsize=9)
    a1.set_title("One setting transfers across splits...", fontweight="bold")

    sp = ["evalhold60", "test2", "test1"]
    a2.bar(np.arange(3) - .19, [15.81, 13.18, 0.86], .38, label="published (24px)", color="#9aa79a")
    a2.bar(np.arange(3) + .19, [23.85, 21.29, 2.98], .38, label="best per split", color="#1f6f4a")
    for i, v in enumerate([23.85, 21.29, 2.98]):
        a2.text(i + .19, v + .7, f"{v:.1f}", ha="center", fontweight="bold", fontsize=9)
    a2.set_xticks(np.arange(3))
    a2.set_xticklabels([f"{s}\nD={D[s]:.0f}px" for s in sp], fontsize=9)
    a2.set_ylabel("COCO AP"); a2.legend(frameon=False, fontsize=9)
    a2.set_title("...but the MODEL does not survive a scale shift", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "cross_site.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    trajectory_ap(); h6_effect(); cross_site()
    print("3 figures written")
