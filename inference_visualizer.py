"""
inference_visualizer.py
======================
Generates high-quality heatmaps of change predictions.
Projects token-level change logits back onto SAM2 masks.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from token_change_reasoner import build_model, ReasonerConfig, build_batch

def generate_heatmap(img_p, res_t1, res_t2, logits, output_path):
    """Projects logits back to pixel space and overlays on image."""
    img = np.array(Image.open(img_p).convert("RGB"))
    h, w = img.shape[:2]
    
    # Heatmap setup
    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32) + 1e-6
    
    # res_t1["masks"] contains the list of boolean masks
    masks = res_t1.get("masks", [])
    if not masks:
        print("No masks found in token file! Run with --save_masks.")
        return
        
    for i, mask in enumerate(masks):
        score = torch.sigmoid(logits[i]).item()
        heatmap[mask] += score
        counts[mask] += 1
        
    heatmap = heatmap / counts
    
    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.imshow(heatmap, alpha=0.5, cmap="jet")
    plt.colorbar(label="Change Probability")
    plt.title(f"Change Detection Heatmap: {Path(img_p).name}")
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_p", type=str, required=True)
    parser.add_argument("--tok1", type=str, required=True)
    parser.add_argument("--tok2", type=str, required=True)
    parser.add_argument("--match", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model_type", default="hierarchical", choices=["base", "graph", "moe", "hierarchical"])
    parser.add_argument("--output", type=str, default="inference_viz.png")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data
    t1 = torch.load(args.tok1, weights_only=True)
    t2 = torch.load(args.tok2, weights_only=True)
    mtch = torch.load(args.match, weights_only=False)
    
    # Build batch
    from token_change_reasoner import SampleData
    sample = SampleData(
        tokens_t1=t1["tokens"].float(),
        tokens_t2=t2["tokens"].float(),
        centroids_t1=t1["centroids"].float(),
        centroids_t2=t2["centroids"].float(),
        areas_t1=t1["areas"].float(),
        areas_t2=t2["areas"].float(),
        cvs_t1=t1.get("cvs"),
        cvs_t2=t2.get("cvs"),
        match_pairs=torch.tensor(mtch["pairs"]).float(),
    )
    batch = build_batch([sample], device=device)
    
    # Load model based on type
    if args.model_type == "hierarchical":
        from token_hierarchical_reasoner import HierarchicalChangeReasoner, HierarchicalConfig
        cfg = HierarchicalConfig()
        model = HierarchicalChangeReasoner(cfg).to(device)
    elif args.model_type == "moe":
        from token_change_reasoner_moe import TokenChangeReasonerMoE, MoEConfig
        cfg = MoEConfig()
        model = TokenChangeReasonerMoE(cfg).to(device)
    elif args.model_type == "graph":
        from token_change_reasoner_graph import TokenChangeReasonerGraph, GraphReasonerConfig
        cfg = GraphReasonerConfig()
        model = TokenChangeReasonerGraph(cfg).to(device)
    else:
        cfg = ReasonerConfig()
        model = build_model(cfg).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    with torch.no_grad():
        outputs = model(batch)
        logits = outputs["change_logits"][0] # [N1+N2]
        
    generate_heatmap(args.img_p, t1, t2, logits[:len(t1["tokens"])], args.output)
    print(f"Heatmap saved to {args.output}")

if __name__ == "__main__":
    main()
