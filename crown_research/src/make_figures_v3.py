"""Third-round figures: the sparsity law, the auxiliary-task decomposition, the domain gap."""
from __future__ import annotations
import sys
from pathlib import Path
_c='/home/ubuntu/anaconda3/envs/cs2/lib/python3.11/site-packages'
_l='/home/chung/.local/lib/python3.11/site-packages'
sys.path=[p for p in sys.path if p not in (_l,_c)]; sys.path.insert(0,_c); sys.path.append(_l)
sys.path.insert(0,"/home/chung/RS/Image Segmentation")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path("crown_research/to_human/figures")
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .3})
GREEN, BLUE, RED, GREY, GOLD = "#1f6f4a", "#2a7fbf", "#9e3b2f", "#9aa79a", "#9a6a12"


def sparsity_law():
    heads = ["canopy", "boundary\n(post-H6)", "boundary\n(pre-H6)", "apex"]
    pos   = [60.0, 6.0, 6.0, 1.2]
    mean  = [0.850, 0.186, 0.105, 0.0041]
    auroc = [0.961, 0.872, 0.866, 0.841]
    cols  = [GREEN, GREEN, GOLD, RED]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.scatter(pos, mean, s=180, c=cols, zorder=5, edgecolors="white", linewidths=1.5)
    for p, m, h in zip(pos, mean, heads):
        a1.annotate(h, (p, m), textcoords="offset points", xytext=(10, 6), fontsize=9)
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("positive rate of the target (%, log)")
    a1.set_ylabel("mean predicted value on positives (log)")
    a1.axhspan(1e-3, 0.11, color=RED, alpha=.07)
    a1.text(1.4, 0.0065, "below the decoder's\nusability floor", color=RED, fontsize=9)
    a1.set_title("Magnitude collapses with target sparsity", fontweight="bold")

    x = np.arange(len(heads)); w = .38
    a2.bar(x - w/2, auroc, w, color=BLUE, label="AUROC (ranking)")
    a2.bar(x + w/2, mean, w, color=RED, label="mean output (magnitude)")
    for xi, (a, m) in zip(x, zip(auroc, mean)):
        a2.text(xi - w/2, a + .02, f"{a:.2f}", ha="center", fontsize=9, fontweight="bold")
        a2.text(xi + w/2, m + .02, f"{m:.3f}", ha="center", fontsize=8)
    a2.set_xticks(x); a2.set_xticklabels(heads, fontsize=9)
    a2.set_ylim(0, 1.12); a2.legend(frameon=False, fontsize=9)
    a2.set_title("Ranking survives. Magnitude does not.", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "sparsity_law.png", dpi=110); plt.close(fig)


def aux_task():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    names = ["H6\n(12 ep)", "H11\n+12 ep,\nno apex head", "H7\n+12 ep,\nwith apex head"]
    vals = [27.281, 27.852, 29.074]
    a1.bar(np.arange(3), vals, color=[GREY, BLUE, GREEN])
    for i, v in enumerate(vals):
        a1.text(i, v + .12, f"{v:.2f}", ha="center", fontweight="bold")
    a1.set_ylim(26.5, 29.8); a1.set_ylabel("COCO AP — 60 held-out crops")
    a1.set_title("Matched-epoch control", fontweight="bold")

    a2.barh([0, 1], [0.571, 1.222], color=[BLUE, GREEN])
    a2.text(0.571 + .03, 0, "  +0.57  (32%)\n  more epochs", va="center", fontsize=10)
    a2.text(1.222 + .03, 1, "  +1.22  (68%)\n  auxiliary task", va="center",
            fontsize=10, fontweight="bold")
    a2.set_yticks([]); a2.set_xlim(0, 2.1); a2.set_xlabel("contribution to H7's +1.79 AP")
    a2.set_title("Two thirds came from a head that is never read", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "aux_task.png", dpi=110); plt.close(fig)


def domain_gap():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(2); w = .35
    a1.bar(x - w/2, [21.32, 2.98], w, color=GREY, label="old head")
    a1.bar(x + w/2, [25.28, 5.46], w, color=GREEN, label="calibrated head (H6)")
    for xi, (o, n) in zip(x, [(21.32, 25.28), (2.98, 5.46)]):
        a1.text(xi - w/2, o + .5, f"{o:.1f}", ha="center", fontsize=9)
        a1.text(xi + w/2, n + .5, f"{n:.1f}", ha="center", fontsize=9, fontweight="bold")
    a1.set_xticks(x); a1.set_xticklabels(["test2\n(in-domain)", "test1\n(out-of-domain)"])
    a1.set_ylabel("COCO AP"); a1.legend(frameon=False, fontsize=9)
    a1.set_title("Gap narrows 7.15x -> 4.63x\n(+18.6% in-domain, +83.0% out)", fontweight="bold")

    labels = ["seam magnitude\nlost to the shift", "barrier height\nout-of-domain"]
    a2.bar([0 - .18, 0 + .18], [43, 40], .35, color=[GREY, GREEN])
    a2.text(-.18, 44.5, "-43%", ha="center", fontsize=10)
    a2.text(.18, 41.5, "-40%", ha="center", fontsize=10, fontweight="bold")
    a2.bar([1 - .18, 1 + .18], [14, 27], .35, color=[GREY, GREEN])
    a2.text(1 - .18, 15.5, "14/255", ha="center", fontsize=10)
    a2.text(1 + .18, 28.5, "27/255", ha="center", fontsize=10, fontweight="bold")
    a2.axhline(20, ls="--", c=RED)
    a2.text(0.55, 21.5, "usability floor (approx.)", color=RED, fontsize=9)
    a2.set_xticks([0, 1]); a2.set_xticklabels(labels, fontsize=9)
    a2.set_ylim(0, 52)
    a2.set_title("Both lose the same ~40%.\nOnly one can afford it.", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "domain_gap.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    sparsity_law(); aux_task(); domain_gap()
    print("3 figures written")
