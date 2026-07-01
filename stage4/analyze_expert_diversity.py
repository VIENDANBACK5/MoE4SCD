"""
stage4/analyze_expert_diversity.py
==================================
MoE Diagnostic — Expert Diversity Collapse Test (Robust Version)

Independent of sklearn/seaborn/matplotlib (using torch + PIL as fallback).
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Search parent
sys.path.insert(0, str(Path(__file__).parent.parent))

from token_change_reasoner import SampleData, build_batch
from token_change_reasoner_moe import MoEConfig, TokenChangeReasonerMoE

# ─────────────────────────────────────────────────────────────────────────────
# Robust Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: Path) -> MoEConfig:
    d = json.loads(path.read_text())
    cfg = MoEConfig()
    for k, v in d.items():
        if hasattr(cfg, k): setattr(cfg, k, v)
    return cfg

def load_model(ckpt_path: Path, cfg: MoEConfig, device: torch.device) -> TokenChangeReasonerMoE:
    model = TokenChangeReasonerMoE(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "model_state" in state: state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    return model

def manual_pca_2d(X: torch.Tensor) -> torch.Tensor:
    """Implement PCA using SVD in torch."""
    X_mean = X.mean(dim=0)
    X_norm = X - X_mean
    # SVD
    U, S, V = torch.pca_lowrank(X_norm, q=2)
    return torch.mm(X_norm, V[:, :2])

def save_heatmap_pil(matrix: np.ndarray, path: Path, title: str):
    """Save a heatmap using PIL."""
    E = matrix.shape[0]
    cell_size = 100
    margin = 40
    img = Image.new("RGB", (E * cell_size + margin*2, E * cell_size + margin*2), "white")
    draw = ImageDraw.Draw(img)
    
    # Normalize for color
    m_min, m_max = matrix.min(), matrix.max()
    if m_max == m_min: m_max += 1e-6
    
    for i in range(E):
        for j in range(E):
            val = matrix[i, j]
            # Color: Blue-ish for similarity, Red-ish for distance
            norm = (val - m_min) / (m_max - m_min)
            color = (int(255 * (1-norm)), int(255 * norm), 180) if "Similarity" in title else (255, int(255*(1-norm)), int(255*(1-norm)))
            
            x0, y0 = margin + j*cell_size, margin + i*cell_size
            draw.rectangle([x0, y0, x0+cell_size, y0+cell_size], fill=color, outline="black")
            draw.text((x0+35, y0+40), f"{val:.2f}", fill="black")
    
    img.save(path)

def save_scatter_pil(coords: torch.Tensor, labels: List[int], path: Path, title: str):
    """Save a scatter plot using PIL."""
    W, H = 800, 600
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    
    # Normalize coords to fit
    c_min = coords.min(dim=0)[0]
    c_max = coords.max(dim=0)[0]
    c_range = c_max - c_min + 1e-6
    
    norm_coords = (coords - c_min) / c_range
    COLORS = [(255,0,0), (0,0,255), (0,200,0), (255,165,0), (128,0,128), (0,128,128)]
    
    for i in range(len(norm_coords)):
        cx, cy = norm_coords[i]
        px = int(cx * (W - 100)) + 50
        py = int(cy * (H - 100)) + 50
        e_id = labels[i]
        color = COLORS[e_id % len(COLORS)]
        draw.ellipse([px-3, py-3, px+3, py+3], fill=color)
        
    img.save(path)

# ─────────────────────────────────────────────────────────────────────────────
# Logic
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_analysis(model: TokenChangeReasonerMoE, stems: List[str], args: argparse.Namespace, device: torch.device):
    E = model.moe.num_experts
    t1_dir = Path(args.tokens_T1)
    t2_dir = Path(args.tokens_T2)
    match_dir = Path(args.matches)
    label_dir = Path(args.labels) if args.labels else None

    # 1. Parameter Distance (Static)
    weights = [model.moe.experts[e].net[0].weight.data.clone() for e in range(E)]
    dist_matrix = np.zeros((E, E))
    for i in range(E):
        for j in range(E):
            dist_matrix[i,j] = (weights[i] - weights[j]).norm().item()

    # 2. Inference Loop
    all_entropy = []
    per_expert_change = [[] for _ in range(E)]
    feat_samples = []
    common_inputs = []
    
    print(f"[diversity] Analyzing {len(stems)} samples...")
    for stem in tqdm(stems):
        t1_path, t2_path, m_path = t1_dir/f"{stem}.pt", t2_dir/f"{stem}.pt", match_dir/f"{stem}_matches.pt"
        if not (t1_path.exists() and t2_path.exists() and m_path.exists()): continue
        
        t1, t2 = torch.load(t1_path, weights_only=True), torch.load(t2_path, weights_only=True)
        mtch = torch.load(m_path, weights_only=True)
        pairs = mtch.get("pairs", [])
        if isinstance(pairs, list):
            p_tensor = torch.tensor([[float(p[0]),float(p[1]),float(p[2])] for p in pairs]) if len(pairs)>0 else torch.zeros(0,3)
        else: p_tensor = pairs.float()

        # Build batch
        sample = SampleData(
            tokens_t1=t1["tokens"].float(), tokens_t2=t2["tokens"].float(),
            centroids_t1=t1["centroids"].float(), centroids_t2=t2["centroids"].float(),
            areas_t1=t1["areas"].float(), areas_t2=t2["areas"].float(),
            match_pairs=p_tensor
        )
        batch = build_batch([sample], model.cfg, device)
        
        # Forward up to MoE
        B, N, _ = batch["tokens_pad"].shape
        flat_emb = batch["tokens_pad"].reshape(B*N, -1)
        flat_tid = batch["time_ids_pad"].reshape(B*N)
        flat_cen = batch["centroids_pad"].reshape(B*N, 2)
        flat_area = batch["log_areas_pad"].reshape(B*N)
        
        enc = model.token_encoder(flat_emb, flat_tid, flat_cen, flat_area).reshape(B, N, -1)
        ctx = model.reasoner(enc, batch["padding_mask"])
        ctx = model.graph(ctx, batch["padding_mask"], batch["centroids_pad"], batch["time_ids_pad"])
        
        x_flat = ctx.reshape(B*N, -1)
        mask = (~batch["padding_mask"]).reshape(B*N)
        x_valid = x_flat[mask]
        if len(x_valid) == 0: continue
        
        # Router
        logits = model.moe.router(x_valid)
        probs = torch.softmax(logits, dim=-1)
        all_entropy.append(-(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item())
        
        e_ids = probs.argmax(dim=-1)
        cp = torch.sigmoid(model.change_head(ctx.reshape(B*N,-1)[mask] + x_valid)).cpu().numpy()
        
        for e in range(E):
            e_mask = (e_ids == e)
            if e_mask.any():
                per_expert_change[e].extend(cp[e_mask.cpu().numpy()].tolist())
                if len(feat_samples) < 1000:
                    feat_samples.append((x_valid[e_mask][0].cpu(), e))
        
        if len(common_inputs) < 256:
            common_inputs.append(x_valid[:1].cpu())

    # 3. Output Similarity
    x_test = torch.cat(common_inputs, dim=0).to(device)
    sim_matrix = np.zeros((E, E))
    for i in range(E):
        out_i = F.normalize(model.moe.experts[i](x_test).reshape(len(x_test),-1), dim=1)
        for j in range(E):
            out_j = F.normalize(model.moe.experts[j](x_test).reshape(len(x_test),-1), dim=1)
            sim_matrix[i,j] = (out_i * out_j).sum(dim=1).mean().item()

    # 4. Save results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    save_heatmap_pil(sim_matrix, out_dir/"expert_sim_matrix.png", "Similarity")
    save_heatmap_pil(dist_matrix, out_dir/"expert_param_dist.png", "Distance")
    
    if feat_samples:
        fs = torch.stack([f[0] for f in feat_samples])
        ls = [f[1] for f in feat_samples]
        pca_coords = manual_pca_2d(fs)
        save_scatter_pil(pca_coords, ls, out_dir/"expert_pca.png", "PCA")

    # Metrics CSV
    with open(out_dir/"metrics.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["mean_entropy", np.mean(all_entropy)])
        w.writerow(["mean_off_diag_sim", (sim_matrix.sum()-E)/(E*(E-1))])
        for e in range(E):
            w.writerow([f"expert_{e}_change_prob", np.mean(per_expert_change[e])])

    # Markdown Report
    report_path = out_dir / "expert_diversity_report.md"
    mean_sim = (sim_matrix.sum()-E)/(E*(E-1))
    status = "STABLE" if mean_sim < 0.7 else "COLLAPSED"
    
    with open(report_path, "w") as f:
        f.write("# Expert Diversity Report\n\n")
        f.write(f"Status: **{status}** (Mean Off-Diag Similarity: {mean_sim:.3f})\n\n")
        f.write("![Similarity](expert_sim_matrix.png) ![Distance](expert_param_dist.png)\n\n")
        f.write("![PCA](expert_pca.png)\n\n")
        f.write(f"Mean Entropy: {np.mean(all_entropy):.3f}\n\n")
        f.write("### Expert Change Sensitivity\n")
        for e in range(E):
            f.write(f"- Expert {e}: {np.mean(per_expert_change[e]):.4f}\n")

    print(f"[✓] Diversity analysis complete: {out_dir}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--tokens_T1", default="SECOND/tokens_T1")
    p.add_argument("--tokens_T2", default="SECOND/tokens_T2")
    p.add_argument("--matches", default="SECOND/matches")
    p.add_argument("--labels", default="SECOND/label1")
    p.add_argument("--output", default="stage5/diversity")
    p.add_argument("--device", default="cuda")
    p.add_argument("--samples", type=int, default=100)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config(Path(args.config))
    model = load_model(Path(args.checkpoint), cfg, device)
    
    stems = sorted([p.stem.replace("_matches", "") for p in Path(args.matches).glob("*_matches.pt")])[:args.samples]
    run_analysis(model, stems, args, device)

if __name__ == "__main__":
    main()
