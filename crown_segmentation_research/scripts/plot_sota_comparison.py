import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil

with open("crown_segmentation_research/experiments/quebec_sota_benchmark_test.json") as f:
    data = json.load(f)

methods = [
    "Baseline\nWatershed",
    "Mask R-CNN\n(ResNet-50)",
    "Detectree2\n(Ball et al.)",
    "StarConvex\n(StarDist v8)",
    "TreeFlowNet\n(Ours SOTA)",
]
keys = [
    "Baseline Watershed",
    "Mask R-CNN (ResNet-50)",
    "Detectree2 (Ball et al.)",
    "StarConvex (StarDist v8)",
    "TreeFlowNet (Ours)",
]

ap50 = [data[k]["segm_AP50"] for k in keys]
f1 = [data[k]["Hungarian_F1_50"] for k in keys]
pq = [data[k]["PQ"] for k in keys]
mae = [data[k]["mae_count"] for k in keys]

colors = ["#64748b", "#2563eb", "#f97316", "#8b5cf6", "#059669"]

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(1, 4, figsize=(20, 5.8), dpi=300)
plt.subplots_adjust(top=0.82, bottom=0.22, wspace=0.28, left=0.05, right=0.98)

# 1. AP50
bars1 = axes[0].bar(range(len(methods)), ap50, color=colors, edgecolor="black", linewidth=0.9, alpha=0.9)
axes[0].set_title("COCO Mask AP50 (%) [↑ Higher is Better]", fontsize=11.5, fontweight="bold", pad=10)
axes[0].grid(axis="y", linestyle="--", alpha=0.5)
axes[0].set_ylim(0, 50)
axes[0].set_xticks(range(len(methods)))
axes[0].set_xticklabels(methods, fontsize=9.5, fontweight="bold", rotation=20, ha="right", rotation_mode="anchor")
for bar in bars1:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, yval + 1.0, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

# 2. Hungarian F1
bars2 = axes[1].bar(range(len(methods)), f1, color=colors, edgecolor="black", linewidth=0.9, alpha=0.9)
axes[1].set_title("Hungarian F1 @ IoU 0.50 [↑ Higher is Better]", fontsize=11.5, fontweight="bold", pad=10)
axes[1].grid(axis="y", linestyle="--", alpha=0.5)
axes[1].set_ylim(0, 0.58)
axes[1].set_xticks(range(len(methods)))
axes[1].set_xticklabels(methods, fontsize=9.5, fontweight="bold", rotation=20, ha="right", rotation_mode="anchor")
for bar in bars2:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, yval + 0.012, f"{yval:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

# 3. Panoptic Quality
bars3 = axes[2].bar(range(len(methods)), pq, color=colors, edgecolor="black", linewidth=0.9, alpha=0.9)
axes[2].set_title("Panoptic Quality (PQ) [↑ Higher is Better]", fontsize=11.5, fontweight="bold", pad=10)
axes[2].grid(axis="y", linestyle="--", alpha=0.5)
axes[2].set_ylim(0, 0.42)
axes[2].set_xticks(range(len(methods)))
axes[2].set_xticklabels(methods, fontsize=9.5, fontweight="bold", rotation=20, ha="right", rotation_mode="anchor")
for bar in bars3:
    yval = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2, yval + 0.009, f"{yval:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

# 4. Count MAE
bars4 = axes[3].bar(range(len(methods)), mae, color=colors, edgecolor="black", linewidth=0.9, alpha=0.9)
axes[3].set_title("Tree Count MAE / Patch [↓ Lower is Better]", fontsize=11.5, fontweight="bold", pad=10)
axes[3].grid(axis="y", linestyle="--", alpha=0.5)
axes[3].set_ylim(0, 38)
axes[3].set_xticks(range(len(methods)))
axes[3].set_xticklabels(methods, fontsize=9.5, fontweight="bold", rotation=20, ha="right", rotation_mode="anchor")
for bar in bars4:
    yval = bar.get_height()
    axes[3].text(bar.get_x() + bar.get_width()/2, yval + 0.8, f"{yval:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

fig.suptitle(
    "SOTA Benchmark Comparison on Exhaustive Quebec Forest Dataset (Zone 3 Test Split, 2,212 Trees)",
    fontsize=14.5,
    fontweight="bold",
    y=0.96
)

out_path1 = Path("crown_segmentation_research/images/quebec_sota_comparison_benchmark.png")
out_path1.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path1, dpi=300, bbox_inches="tight")

out_path2 = Path("/home/chung/.gemini/antigravity-ide/brain/430ab9e9-db0b-4bb3-9c66-8def6a5fa89b/quebec_sota_comparison_benchmark.png")
plt.savefig(out_path2, dpi=300, bbox_inches="tight")

out_path3 = Path("meeting/week1/images/13_sota_comparison_benchmark.png")
plt.savefig(out_path3, dpi=300, bbox_inches="tight")

plt.close()
print("Saved SOTA comparison charts to:", out_path1, out_path2, out_path3)

