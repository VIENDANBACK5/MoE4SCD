"""
Generate publication-quality diagram illustrating the Large-Scale Coverage (1x1 km^2)
Inference & Seamless Boundary Stitching Pipeline for TreeFlowNet.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

fig = plt.figure(figsize=(17, 6.8), dpi=300)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.25, 1.25], wspace=0.28, left=0.05, right=0.96, top=0.85, bottom=0.10)

# =========================================================================
# Panel 1: Tiled Sliding-Window Decomposition (1 x 1 km^2)
# =========================================================================
ax1 = fig.add_subplot(gs[0])
ax1.set_title("(A) Large-Area Orthomosaic Tiling\n1 × 1 km² (100 ha Coverage)", fontsize=11.5, fontweight="bold", pad=10)

# Draw 1 km2 boundary box
rect_outer = patches.Rectangle((0, 0), 10, 10, linewidth=2.5, edgecolor="#1e293b", facecolor="#f8fafc")
ax1.add_patch(rect_outer)

# Draw grid lines representing 1024x1024 tiles with overlap
n_div = 5
for i in range(1, n_div):
    ax1.plot([0, 10], [i * 2, i * 2], color="#cbd5e1", linestyle="--", linewidth=1.0)
    ax1.plot([i * 2, i * 2], [0, 10], color="#cbd5e1", linestyle="--", linewidth=1.0)

# Active tile and overlap buffer located at bottom-left quadrant
tile_x, tile_y = 2.0, 2.0
tile_w, tile_h = 2.0, 2.0
overlap_buf = 0.45

# Overlap buffer
rect_buf = patches.Rectangle((tile_x - overlap_buf, tile_y - overlap_buf), tile_w + 2*overlap_buf, tile_h + 2*overlap_buf,
                             linewidth=1.5, edgecolor="#d97706", facecolor="#fef3c7", alpha=0.55, linestyle="--")
ax1.add_patch(rect_buf)

# Active tile
rect_tile = patches.Rectangle((tile_x, tile_y), tile_w, tile_h, linewidth=2, edgecolor="#2563eb", facecolor="#dbeafe", alpha=0.75)
ax1.add_patch(rect_tile)

ax1.text(tile_x + tile_w/2, tile_y + tile_h/2, "Active Tile\n1024 × 1024", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#1e40af")
ax1.text(tile_x + tile_w/2, tile_y - 0.22, "128px Margin", ha="center", va="top", fontsize=7.5, fontweight="bold", color="#b45309")

# Summary specs box placed cleanly at the top-right inside panel 1
props = dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor='#94a3b8', alpha=0.95, linewidth=1.0)
specs_text = (
    "Tiling Specifications:\n"
    "• GSD 10 cm: 10k × 10k px\n"
    "  → 144 tiles (stride 896)\n"
    "• GSD 5 cm: 20k × 20k px\n"
    "  → 529 tiles (stride 896)\n"
    "• Overlap: 12.5% (128 px)"
)
ax1.text(9.5, 9.5, specs_text, fontsize=8.5, verticalalignment='top', horizontalalignment='right', bbox=props, family='sans-serif', linespacing=1.3)

ax1.set_xlim(-0.2, 10.2)
ax1.set_ylim(-0.2, 10.2)
ax1.set_aspect("equal")
ax1.axis("off")


# =========================================================================
# Panel 2: Compute & Latency Comparison on 1 x 1 km^2
# =========================================================================
ax2 = fig.add_subplot(gs[1])
ax2.set_title("(B) Wall-Clock Latency on 1 km²\n(529 Tiles @ 5 cm UAV GSD)", fontsize=11.5, fontweight="bold", pad=10)

models = ["SAM 2\n(AMG)", "Mask2Former\n(Swin-L)", "Mask R-CNN\n(ResNet-50)", "TreeFlowNet\n(Ours SOTA)"]
# Latency in minutes for 529 tiles
time_mins = [
    (529 * 13.45) / 60,  # ~118.5 mins (~2 hours)
    (529 * 1.45) / 60,   # ~12.8 mins
    (529 * 0.49) / 60,   # ~4.3 mins
    (529 * 0.14) / 60    # ~1.23 mins (74 seconds)
]
bar_colors = ["#ef4444", "#f97316", "#3b82f6", "#059669"]

bars = ax2.bar(range(len(models)), time_mins, color=bar_colors, edgecolor="black", linewidth=1.0, width=0.55, alpha=0.9)
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, fontsize=9.0, fontweight="bold")
ax2.set_ylabel("Processing Time (Minutes) [↓ Lower is Better]", fontsize=10.0, fontweight="bold")
ax2.set_yscale("log")
ax2.set_ylim(0.5, 300)
ax2.grid(axis="y", linestyle="--", alpha=0.5)

# Annotate bars
time_labels = ["118.5 min\n(~2.0 hours)", "12.8 min\n(767 sec)", "4.3 min\n(259 sec)", "1.2 min\n(74 sec) [Fastest]"]
for bar, lbl in zip(bars, time_labels):
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval * 1.25, lbl, ha="center", va="bottom", fontsize=8.0, fontweight="bold")

# Callout box
comp_box = dict(boxstyle='round,pad=0.4', facecolor='#ecfdf5', edgecolor='#059669', alpha=0.95, linewidth=1.2)
ax2.text(0.5, 0.05, "TreeFlowNet is ~100× Faster than SAM 2\nRequires <2.5 GB VRAM during inference\n(Full 1 km² in ~1.2 min on single GPU)", 
         transform=ax2.transAxes, fontsize=8.5, fontweight="bold", color="#065f46", ha="center", bbox=comp_box)


# =========================================================================
# Panel 3: Seamless Boundary Stitching & GIS Export
# =========================================================================
ax3 = fig.add_subplot(gs[2])
ax3.set_title("(C) Boundary Stitching & Vector Export\nContinuous Forest Crown Polygons", fontsize=11.5, fontweight="bold", pad=10)

# Step 1: Tile boundary with split crown
box_step1 = patches.FancyBboxPatch((0.04, 0.69), 0.92, 0.25, boxstyle="round,pad=0.03", edgecolor="#3b82f6", facecolor="#eff6ff", linewidth=1.2)
ax3.add_patch(box_step1)
ax3.text(0.50, 0.88, "1. Cross-Boundary Centripetal Flows", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1e40af")
ax3.text(0.50, 0.77, "• Vector flows in overlap margin point to same physical sink.\n• No truncated crown edges or split artifacts across seams.", 
         ha="center", va="center", fontsize=8.2, color="#1e293b", linespacing=1.2)

# Step 2: Sink matching and IoU Merge
box_step2 = patches.FancyBboxPatch((0.04, 0.37), 0.92, 0.25, boxstyle="round,pad=0.03", edgecolor="#8b5cf6", facecolor="#f5f3ff", linewidth=1.2)
ax3.add_patch(box_step2)
ax3.text(0.50, 0.56, "2. Sink Matching & Polygon Union", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#5b21b6")
ax3.text(0.50, 0.45, "• Global UTM/EPSG coordinate reprojection.\n• Overlap buffer crowns merged via Shapely unary_union.", 
         ha="center", va="center", fontsize=8.2, color="#1e293b", linespacing=1.2)

# Step 3: GIS Vector Layer
box_step3 = patches.FancyBboxPatch((0.04, 0.05), 0.92, 0.25, boxstyle="round,pad=0.03", edgecolor="#059669", facecolor="#ecfdf5", linewidth=1.2)
ax3.add_patch(box_step3)
ax3.text(0.50, 0.24, "3. Standard GIS Delivery Format", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#065f46")
ax3.text(0.50, 0.13, "• Direct export to GeoPackage (.gpkg) / Shapefile (.shp).\n• Ready for QGIS/ArcGIS forest inventory & carbon metrics.", 
         ha="center", va="center", fontsize=8.2, color="#1e293b", linespacing=1.2)

# Connecting downward arrows
ax3.annotate("", xy=(0.50, 0.64), xytext=(0.50, 0.68), arrowprops=dict(arrowstyle="->", lw=1.6, color="#64748b"))
ax3.annotate("", xy=(0.50, 0.32), xytext=(0.50, 0.36), arrowprops=dict(arrowstyle="->", lw=1.6, color="#64748b"))

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis("off")

# Main Title
fig.suptitle(
    "TreeFlowNet Large-Coverage (1 × 1 km² / 100 ha) Inference & Seamless Delineation Pipeline",
    fontsize=14.0, fontweight="bold", y=0.97
)

out_file = Path("meeting/week1/images/16_large_scale_coverage_workflow.png")
out_file.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close()
print("Saved refined large-scale workflow diagram to:", out_file)

