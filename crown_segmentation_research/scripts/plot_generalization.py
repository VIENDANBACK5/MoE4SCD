import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
data_dir = Path("crown_segmentation_research/experiments")
with open(data_dir / "quebec_train_exhaustive_results.json") as f:
    r_train = json.load(f)
with open(data_dir / "quebec_val_exhaustive_results.json") as f:
    r_val = json.load(f)
with open(data_dir / "quebec_test_exhaustive_results.json") as f:
    r_test = json.load(f)
with open(data_dir / "quebec_all600_exhaustive_results.json") as f:
    r_all = json.load(f)

splits = ["Zone 1 (Train)\n200 patches", "Zone 2 (Val)\n200 patches", "Zone 3 (Test)\n200 patches", "Full Dataset\n600 patches"]
ap50_tf = [r_train["treeflow_net"]["segm_AP50"], r_val["treeflow_net"]["segm_AP50"], r_test["treeflow_net"]["segm_AP50"], r_all["treeflow_net"]["segm_AP50"]]
ap50_ws = [r_train["baseline_watershed"]["segm_AP50"], r_val["baseline_watershed"]["segm_AP50"], r_test["baseline_watershed"]["segm_AP50"], r_all["baseline_watershed"]["segm_AP50"]]

f1_tf = [r_train["treeflow_net"]["Hungarian_F1_50"], r_val["treeflow_net"]["Hungarian_F1_50"], r_test["treeflow_net"]["Hungarian_F1_50"], r_all["treeflow_net"]["Hungarian_F1_50"]]
f1_ws = [r_train["baseline_watershed"]["Hungarian_F1_50"], r_val["baseline_watershed"]["Hungarian_F1_50"], r_test["baseline_watershed"]["Hungarian_F1_50"], r_all["baseline_watershed"]["Hungarian_F1_50"]]

mae_tf = [r_train["treeflow_net"]["mae_count"], r_val["treeflow_net"]["mae_count"], r_test["treeflow_net"]["mae_count"], r_all["treeflow_net"]["mae_count"]]
mae_ws = [r_train["baseline_watershed"]["mae_count"], r_val["baseline_watershed"]["mae_count"], r_test["baseline_watershed"]["mae_count"], r_all["baseline_watershed"]["mae_count"]]

pq_tf = [r_train["treeflow_net"]["PQ"], r_val["treeflow_net"]["PQ"], r_test["treeflow_net"]["PQ"], r_all["treeflow_net"]["PQ"]]
pq_ws = [r_train["baseline_watershed"]["PQ"], r_val["baseline_watershed"]["PQ"], r_test["baseline_watershed"]["PQ"], r_all["baseline_watershed"]["PQ"]]

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
x = np.arange(len(splits))
w = 0.35

# Color scheme
c_tf = "#2ca02c"
c_ws = "#d62728"

# 1. AP50
axes[0, 0].bar(x - w/2, ap50_tf, w, label="TreeFlowNet (Ours)", color=c_tf, edgecolor="black", alpha=0.9)
axes[0, 0].bar(x + w/2, ap50_ws, w, label="Baseline Watershed", color=c_ws, edgecolor="black", alpha=0.7)
axes[0, 0].set_title("COCO Mask AP50 (%) [Higher is Better]", fontsize=13, fontweight="bold")
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(splits, fontsize=10)
axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)
for i, v in enumerate(ap50_tf):
    axes[0, 0].text(i - w/2, v + 0.8, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=9)
for i, v in enumerate(ap50_ws):
    axes[0, 0].text(i + w/2, v + 0.8, f"{v:.1f}%", ha="center", fontsize=8)
axes[0, 0].legend(loc="upper right")

# 2. Hungarian F1
axes[0, 1].bar(x - w/2, f1_tf, w, label="TreeFlowNet (Ours)", color=c_tf, edgecolor="black", alpha=0.9)
axes[0, 1].bar(x + w/2, f1_ws, w, label="Baseline Watershed", color=c_ws, edgecolor="black", alpha=0.7)
axes[0, 1].set_title("Hungarian F1 Score @ IoU 0.50 [Higher is Better]", fontsize=13, fontweight="bold")
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(splits, fontsize=10)
axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)
for i, v in enumerate(f1_tf):
    axes[0, 1].text(i - w/2, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)
for i, v in enumerate(f1_ws):
    axes[0, 1].text(i + w/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

# 3. Panoptic Quality
axes[1, 0].bar(x - w/2, pq_tf, w, label="TreeFlowNet (Ours)", color=c_tf, edgecolor="black", alpha=0.9)
axes[1, 0].bar(x + w/2, pq_ws, w, label="Baseline Watershed", color=c_ws, edgecolor="black", alpha=0.7)
axes[1, 0].set_title("Panoptic Quality (PQ) [Higher is Better]", fontsize=13, fontweight="bold")
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(splits, fontsize=10)
axes[1, 0].grid(axis="y", linestyle="--", alpha=0.5)
for i, v in enumerate(pq_tf):
    axes[1, 0].text(i - w/2, v + 0.008, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)
for i, v in enumerate(pq_ws):
    axes[1, 0].text(i + w/2, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)

# 4. Count MAE
axes[1, 1].bar(x - w/2, mae_tf, w, label="TreeFlowNet (Ours)", color=c_tf, edgecolor="black", alpha=0.9)
axes[1, 1].bar(x + w/2, mae_ws, w, label="Baseline Watershed", color=c_ws, edgecolor="black", alpha=0.7)
axes[1, 1].set_title("Tree Count MAE per Patch [Lower is Better]", fontsize=13, fontweight="bold")
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(splits, fontsize=10)
axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)
for i, v in enumerate(mae_tf):
    axes[1, 1].text(i - w/2, v + 0.6, f"{v:.2f}", ha="center", fontweight="bold", fontsize=9)
for i, v in enumerate(mae_ws):
    axes[1, 1].text(i + w/2, v + 0.6, f"{v:.1f}", ha="center", fontsize=8)

plt.suptitle("Exhaustive All-Tree Benchmark Generalization Across All Splits (Quebec Dataset)", fontsize=16, fontweight="bold", y=0.99)
plt.tight_layout()

out_path1 = Path("crown_segmentation_research/images/quebec_all_splits_generalization.png")
out_path1.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path1, dpi=200, bbox_inches="tight")

out_path2 = Path("/home/chung/.gemini/antigravity-ide/brain/430ab9e9-db0b-4bb3-9c66-8def6a5fa89b/quebec_all_splits_generalization.png")
plt.savefig(out_path2, dpi=200, bbox_inches="tight")
plt.close()
print("Saved plots to", out_path1, "and", out_path2)
