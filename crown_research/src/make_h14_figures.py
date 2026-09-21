"""Generate publication-ready figures for H14 Domain Adaptation."""
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

FIG = Path("crown_research/to_human/figures")
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

# Figure 1: Out-of-domain AP trajectory
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

stages = ["Original\nBaseline", "H6 Calibrated\nBoundary", "H14 Photometric\n& Shadow Aug"]
ood_ap = [2.98, 5.46, 8.68]
indomain_ap = [21.29, 25.24, 24.66]
gaps = [indomain_ap[i] / ood_ap[i] for i in range(3)]

x = np.arange(3)
w = 0.35

ax1.bar(x - w/2, indomain_ap, w, label="In-Domain (TestSet2)", color="#2a7fbf")
ax1.bar(x + w/2, ood_ap, w, label="Out-of-Domain (TestSet1 City Park)", color="#1f6f4a")

for i in range(3):
    ax1.text(x[i] - w/2, indomain_ap[i] + 0.6, f"{indomain_ap[i]:.2f}", ha="center", fontweight="bold", fontsize=10)
    ax1.text(x[i] + w/2, ood_ap[i] + 0.6, f"{ood_ap[i]:.2f}", ha="center", fontweight="bold", fontsize=10, color="#1f6f4a")

ax1.set_xticks(x)
ax1.set_xticklabels(stages)
ax1.set_ylabel("COCO AP")
ax1.set_ylim(0, 30)
ax1.set_title("COCO AP: In-Domain vs Out-of-Domain", fontweight="bold")
ax1.legend(frameon=True, facecolor="white")

# Subplot 2: Domain Transfer Gap Ratio
ax2.plot(stages, gaps, marker="o", color="#9e3b2f", linewidth=2.5, markersize=8)
for i, g in enumerate(gaps):
    ax2.text(i, g + 0.25, f"{g:.2f}x\ngap", ha="center", fontweight="bold", color="#9e3b2f")

ax2.set_ylabel("Domain Gap Ratio (In-Domain / OOD)")
ax2.set_ylim(1.5, 8.2)
ax2.set_title("Generalisation Gap Shrinks from 7.14x → 2.84x", fontweight="bold")

plt.suptitle("H14: Photometric & Shadow Augmentation Closes the Out-of-Domain Gap", fontweight="bold", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIG / "h14_domain_gap_narrowing.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved h14_domain_gap_narrowing.png")
