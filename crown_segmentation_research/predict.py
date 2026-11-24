#!/usr/bin/env python3
"""
TreeFlowNet: Standalone One-Command Reproducible Inference Script.

Runs Individual Tree Crown (ITC) Instance Segmentation on any aerial RGB orthomosaic.
Supports both single images (.png, .jpg, .tif) and batch directories.

Usage:
    python crown_segmentation_research/predict.py \
        --input path/to/image.tif \
        --output-dir results/ \
        --ckpt crown_segmentation_research/experiments/tree_flow_v3/best_model_v3.pth \
        --visualize
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from crown_segmentation_research.methods.tree_flow.model_v3 import TreeFlowNetV3
from crown_segmentation_research.methods.tree_flow.model_v2 import TreeFlowNetV2
from crown_segmentation_research.methods.tree_flow.decode import decode_organic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="TreeFlowNet Standalone Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input image file or directory")
    parser.add_argument("--output-dir", type=str, default="inference_results", help="Directory to save results")
    parser.add_argument("--ckpt", type=str, default="crown_segmentation_research/experiments/tree_flow_v3/best_model_v3.pth", help="Model checkpoint path")
    parser.add_argument("--version", type=str, default="v3", choices=["v2", "v3"], help="Architecture version")
    parser.add_argument("--conf-thresh", type=float, default=0.20, help="Confidence threshold (0.20 = balanced inventory)")
    parser.add_argument("--min-area", type=int, default=150, help="Minimum crown area in pixels")
    parser.add_argument("--stream-burning", action="store_true", default=True, help="Enable hydrological stream burning")
    parser.add_argument("--visualize", action="store_true", default=True, help="Save overlaid visualization PNGs")
    return parser.parse_args()

def load_model(ckpt_path: str, version: str = "v3"):
    print(f"Loading TreeFlowNet ({version.upper()}) from {ckpt_path}...")
    if version == "v3":
        model = TreeFlowNetV3().to(DEVICE)
    else:
        model = TreeFlowNetV2().to(DEVICE)
    
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model

def predict_single(model, img_np: np.ndarray, args):
    H, W = img_np.shape[:2]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = ((img_np.astype(np.float32) / 255.0) - mean) / std
    t_img = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        preds = model(t_img)
        preds_sq = {k: v[0] for k, v in preds.items()}

    instances = decode_organic(
        preds_sq,
        img_rgb=img_np,
        min_sink_dist=44,
        canopy_thresh=0.30,
        sdt_thresh=-0.15,
        min_area=args.min_area,
        gradient_mode="sobel",
        refine_mode="guided_filter",
        score_mode="calibrated",
        soft_nms=True,
        stream_burning=args.stream_burning,
        device=str(DEVICE),
    )

    # Filter by confidence threshold
    instances = [inst for inst in instances if inst["score"] >= args.conf_thresh]
    return instances

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model(args.ckpt, args.version)

    input_path = Path(args.input)
    if input_path.is_dir():
        image_files = sorted(list(input_path.glob("*.tif")) + list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")))
    else:
        image_files = [input_path]

    print(f"Found {len(image_files)} image(s) to process.")
    for idx, fpath in enumerate(image_files):
        t0 = time.time()
        pil_img = Image.open(fpath).convert("RGB")
        img_np = np.array(pil_img)

        instances = predict_single(model, img_np, args)
        elapsed = (time.time() - t0) * 1000

        print(f"[{idx+1}/{len(image_files)}] {fpath.name}: {len(instances)} crowns detected ({elapsed:.1f} ms)")

        # Save results JSON
        stem = fpath.stem
        out_json = os.path.join(args.output_dir, f"{stem}_crowns.json")
        crown_records = []
        for c_id, inst in enumerate(instances, 1):
            mask = inst["mask"].astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pts = contours[0].squeeze().tolist() if len(contours) > 0 and len(contours[0]) >= 3 else []
            crown_records.append({
                "crown_id": c_id,
                "score": float(inst["score"]),
                "area_px": int(inst["area"]),
                "polygon": pts
            })
        with open(out_json, "w") as f:
            json.dump({"file_name": fpath.name, "num_crowns": len(instances), "crowns": crown_records}, f, indent=2)

        # Save Visualization
        if args.visualize:
            vis = img_np.copy()
            for c_id, inst in enumerate(instances, 1):
                mask = inst["mask"].astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                color = (int(np.random.randint(50, 255)), int(np.random.randint(180, 255)), int(np.random.randint(50, 255)))
                cv2.drawContours(vis, contours, -1, color, 2)
            out_vis = os.path.join(args.output_dir, f"{stem}_delineation.png")
            Image.fromarray(vis).save(out_vis)

    print(f"\nAll results successfully saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
