import sys
import torch
import numpy as np
from pathlib import Path
from crown_segmentation_research.evaluation.benchmark_bam_coco_ap import load_star_convex_model, infer_star_convex_tile

device = "cuda" if torch.cuda.is_available() else "cpu"

s_path = "crown_segmentation_research/experiments/star_convex_screen_v8/star_convex_screen_latest.pth"
print("Loading StarConvex v8 from", s_path)
star_convex = load_star_convex_model(s_path, "v8")

data = np.load("data/itc_benchmarks/extracted/quebec_cached_npz/quebec_z3_patch_0001.npz")
img = data["image"]  # [512, 512, 3] uint8

masks, scores = infer_star_convex_tile(star_convex, img, {"prob_threshold": 0.4, "min_peak_distance": 5, "nms_iou_threshold": 0.2})
print(f"StarConvex predictions: {len(masks)} crowns")
print("SUCCESS!")
