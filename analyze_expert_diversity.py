#!/usr/bin/env python3
"""
analyze_expert_diversity.py  — v2
==================================
MoE Expert Diversity & Collapse Test

Runs 6 diagnostic tests on a trained TokenChangeReasonerMoE checkpoint:

  Test 1  Expert Output Similarity     4×4 cosine-similarity matrix
                                       (same tokens forced through every expert)
  Test 2  Expert Feature Distribution  PCA scatter plot of expert outputs
  Test 3  Routing Entropy              per-token routing confidence
  Test 4  Expert Parameter Distance    4×4 L2 distance between expert weights
  Test 5  Expert Change Sensitivity    average change_prob per routing expert
  Test 6  Gradient Similarity          4×4 cosine-sim of per-expert gradient
                                       vectors (same tokens, forced dispatch)

Interpretation thresholds:
  similarity > 0.9  → collapse warning
  similarity < 0.6  → experts are functionally diverse

Output ({output_dir}/):
  expert_diversity_report.md
  metrics.csv
  output_similarity_matrix.png
  parameter_distance_matrix.png
  gradient_similarity_matrix.png
  expert_pca.png
  routing_distribution.png

Usage:
  # dynamic routing model
  python analyze_expert_diversity.py \\
      --checkpoint SECOND/stage5_6_dynamic/best_model.pt \\
      --config     SECOND/stage5_6_dynamic/config.json \\
      --output     stage5/diversity/dynamic \\
      --samples    200

  # semantic routing model (with labels for class breakdown)
  python analyze_expert_diversity.py \\
      --checkpoint SECOND/stage5_6_semantic/best_model.pt \\
      --config     SECOND/stage5_6_semantic/config.json \\
      --labels     SECOND/label1 \\
      --output     stage5/diversity/semantic \\
      --samples    200
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

# Allow importing from the phase1 root
sys.path.insert(0, str(Path(__file__).parent))

from token_change_reasoner import SampleData, build_batch
from token_change_reasoner_moe import MoEConfig, TokenChangeReasonerMoE

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Colorblind-friendly palette for up to 8 experts
EXPERT_COLORS = [
    (214,  39,  40),   # Expert 0 — red
    ( 31, 119, 180),   # Expert 1 — blue
    ( 44, 160,  44),   # Expert 2 — green
    (255, 127,  14),   # Expert 3 — orange
    (148, 103, 189),   # Expert 4 — purple
    ( 23, 190, 207),   # Expert 5 — teal
    (188, 189,  34),   # Expert 6 — yellow-green
    (140,  86,  75),   # Expert 7 — brown
]

CLASS_NAMES = [
    "background", "water", "soil/imp", "vegetation",
    "building",   "farmland", "low_veg",
]

COLLAPSE_THRESHOLD       = 0.90   # sim > this → likely collapsed
SPECIALIZATION_THRESHOLD = 0.60   # sim < this → functionally diverse

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: Path) -> MoEConfig:
    d = json.loads(path.read_text())
    cfg = MoEConfig()
    for k, v in d.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def load_model(
    ckpt_path: Path,
    cfg: MoEConfig,
    device: torch.device,
) -> TokenChangeReasonerMoE:
    model = TokenChangeReasonerMoE(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Expert Parameter Distance  (static: no data needed)
# ─────────────────────────────────────────────────────────────────────────────

def test_parameter_distance(model: TokenChangeReasonerMoE) -> np.ndarray:
    """L2 distance between the first FC-layer weights of each expert pair."""
    E = model.moe.num_experts
    weights = [
        model.moe.experts[e].net[0].weight.data.float().flatten()
        for e in range(E)
    ]
    dist = np.zeros((E, E), dtype=np.float64)
    for i in range(E):
        for j in range(E):
            dist[i, j] = (weights[i] - weights[j]).norm().item()
    return dist


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Expert Output Similarity
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_output_similarity(
    model: TokenChangeReasonerMoE,
    x_repr: torch.Tensor,   # [T, H]  post-GNN representations
    device: torch.device,
) -> np.ndarray:
    """
    Pass the SAME token batch through EVERY expert (forced dispatch).
    Compute pairwise mean cosine-similarity of output vectors.
    A value > 0.9 signals that expert pair is functionally identical.
    """
    E = model.moe.num_experts
    x = x_repr.to(device)

    outputs = []
    for e in range(E):
        out = model.moe.experts[e](x).float()          # [T, H]
        outputs.append(F.normalize(out, dim=-1))        # unit-norm

    sim = np.zeros((E, E), dtype=np.float64)
    for i in range(E):
        for j in range(E):
            sim[i, j] = (outputs[i] * outputs[j]).sum(dim=-1).mean().item()
    return sim


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — Gradient Similarity
# ─────────────────────────────────────────────────────────────────────────────

def test_gradient_similarity(
    model: TokenChangeReasonerMoE,
    x_repr: torch.Tensor,   # [T, H]  post-GNN representations
    device: torch.device,
) -> np.ndarray:
    """
    For each expert Ei, force ALL tokens → Ei → loss = mean(out²) → backward.
    Collect grad of Ei.net[0].weight, flatten to a vector.
    Compute pairwise cosine-similarity of those gradient vectors.

    Interpretation:
      grad_sim(Ei, Ej) > 0.9 → experts update in the same direction
                               (functionally equivalent, likely collapsed)
      grad_sim(Ei, Ej) < 0.5 → experts learn different transformations ✓

    Note: this is run OUTSIDE of any no_grad context so autograd works.
    """
    E = model.moe.num_experts
    x = x_repr.float().to(device)

    # Cap size for speed while keeping a representative sample
    x_test = x[:min(512, x.shape[0])].detach()

    grad_vecs: List[torch.Tensor] = []

    for e in range(E):
        # Zero all gradients
        model.zero_grad(set_to_none=True)

        # Temporarily enable grad on this expert's first weight only
        w = model.moe.experts[e].net[0].weight
        w.requires_grad_(True)

        # Forced forward through expert e
        out = model.moe.experts[e](x_test)          # [T, H]

        # Loss: mean of squared outputs — non-trivially depends on all weights
        loss = (out.float() ** 2).mean()
        loss.backward()

        if w.grad is not None:
            grad_vecs.append(w.grad.detach().float().flatten().clone())
        else:
            grad_vecs.append(torch.zeros(w.numel(), device=device))

        # Disable grad again so we don't pollute the next iteration
        w.requires_grad_(False)

    model.zero_grad(set_to_none=True)

    # Pairwise cosine similarity
    grad_matrix = np.zeros((E, E), dtype=np.float64)
    for i in range(E):
        for j in range(E):
            gi = F.normalize(grad_vecs[i].unsqueeze(0), dim=1)
            gj = F.normalize(grad_vecs[j].unsqueeze(0), dim=1)
            grad_matrix[i, j] = (gi * gj).sum().item()

    return grad_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Main inference pass — collects data for Tests 2, 3, 5 + gathers x_repr
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference_pass(
    model: TokenChangeReasonerMoE,
    stems: List[str],
    t1_dir: Path,
    t2_dir: Path,
    match_dir: Path,
    label_dir: Optional[Path],
    device: torch.device,
    max_feat_samples: int = 2000,
    max_common_inputs: int = 512,
) -> Dict:
    """
    Single forward-pass loop that collects:
      - per-sample routing entropy          → Test 3
      - per-expert change probabilities     → Test 5
      - expert output embeddings (PCA)      → Test 2
      - common post-GNN reprs               → Tests 1 & 6
      - routing load counts                 → bar chart
      - per-expert class counts (optional)  → bonus table
    """
    E   = model.moe.num_experts
    cfg = model.cfg

    all_entropy:            List[float]            = []
    per_expert_change:      List[List[float]]       = [[] for _ in range(E)]
    # (output_vector [H], expert_id, class_id)
    feat_samples:           List[Tuple[torch.Tensor, int, int]] = []
    common_reprs:           List[torch.Tensor]     = []   # post-GNN [T, H] chunks
    per_expert_class_counts: np.ndarray            = np.zeros((E, 7), dtype=np.int64)
    total_tokens_per_expert: np.ndarray            = np.zeros(E, dtype=np.int64)

    for stem in tqdm(stems, desc="[diversity] inference", ncols=80):
        t1_path = t1_dir   / f"{stem}.pt"
        t2_path = t2_dir   / f"{stem}.pt"
        m_path  = match_dir / f"{stem}_matches.pt"
        if not (t1_path.exists() and t2_path.exists() and m_path.exists()):
            continue

        t1   = torch.load(t1_path,  weights_only=True)
        t2   = torch.load(t2_path,  weights_only=True)
        mtch = torch.load(m_path,   weights_only=True)

        pairs = mtch.get("pairs", [])
        if isinstance(pairs, list):
            p_tensor = (
                torch.tensor(
                    [[float(p[0]), float(p[1]), float(p[2])] for p in pairs],
                    dtype=torch.float32,
                )
                if pairs
                else torch.zeros(0, 3)
            )
        else:
            p_tensor = pairs.float()

        sample = SampleData(
            tokens_t1    = t1["tokens"].float(),
            tokens_t2    = t2["tokens"].float(),
            centroids_t1 = t1["centroids"].float(),
            centroids_t2 = t2["centroids"].float(),
            areas_t1     = t1["areas"].float(),
            areas_t2     = t2["areas"].float(),
            match_pairs  = p_tensor,
        )

        batch = build_batch([sample], cfg, device)
        B, N, _ = batch["tokens_pad"].shape

        # ── Encode → Transformer → Graph  (post-GNN repr) ──────────────────
        flat_emb  = batch["tokens_pad"].reshape(B * N, -1)
        flat_tid  = batch["time_ids_pad"].reshape(B * N)
        flat_cen  = batch["centroids_pad"].reshape(B * N, 2)
        flat_area = batch["log_areas_pad"].reshape(B * N)

        enc = model.token_encoder(flat_emb, flat_tid, flat_cen, flat_area).reshape(B, N, -1)
        ctx = model.reasoner(enc, batch["padding_mask"])
        ctx = model.graph(ctx, batch["padding_mask"],
                          batch["centroids_pad"], batch["time_ids_pad"])

        # Valid (non-padding) tokens  [V, H]
        valid_mask_flat = (~batch["padding_mask"]).reshape(B * N)
        x_flat  = ctx.reshape(B * N, -1)
        x_valid = x_flat[valid_mask_flat]
        V = x_valid.shape[0]
        if V == 0:
            continue

        # Accumulate post-GNN reprs for Tests 1 & 6
        take = min(16, V, max_common_inputs - len(common_reprs))
        if take > 0:
            common_reprs.append(x_valid[:take].cpu())

        # ── Build router input ──────────────────────────────────────────────
        if cfg.router_version == "v2":
            la = batch["log_areas_pad"].reshape(B * N)[valid_mask_flat].unsqueeze(1)
            dh = torch.zeros(V, 1, device=device)
            router_in = torch.cat([x_valid, la, dh], dim=-1)
        elif cfg.router_version == "v3":
            router_in = torch.cat(
                [x_valid, torch.zeros(V, 7, device=device)], dim=-1
            )
        else:
            router_in = x_valid

        # ── Test 3: Routing Entropy ─────────────────────────────────────────
        logits = model.moe.router(router_in)             # [V, E]
        probs  = torch.softmax(logits, dim=-1)           # [V, E]
        ent    = -(probs * (probs + 1e-9).log()).sum(dim=-1)   # [V]
        all_entropy.append(ent.mean().item())
        e_ids  = probs.argmax(dim=-1)                    # [V]

        # Routing load
        for e in range(E):
            total_tokens_per_expert[e] += int((e_ids == e).sum().item())

        # ── Run through MoE for full change logits (post-residual) ──────────
        moe_out, _, _, _ = model.moe(
            ctx,
            log_areas       = batch["log_areas_pad"] if cfg.router_version == "v2" else None,
            delta_hints     = None,
            semantic_labels = None,
        )
        change_logits_all = model.change_head(moe_out.reshape(B * N, -1))  # [B*N]
        change_probs_all  = torch.sigmoid(change_logits_all)               # [B*N]
        change_probs_valid = change_probs_all[valid_mask_flat].cpu()       # [V]

        # ── Semantic label lookup (optional) ───────────────────────────────
        token_classes: Optional[torch.Tensor] = None
        if label_dir is not None:
            lp = label_dir / f"{stem}.png"
            if lp.exists():
                import numpy as _np
                _raw = Image.open(lp)
                # SECOND labels may be RGB palettized — convert to integer class map
                if _raw.mode != 'L':
                    _raw = _raw.convert('L')
                lbl_img = _np.array(_raw)               # [lH, lW] int
                lH, lW  = lbl_img.shape
                cen_flat = batch["centroids_pad"].reshape(B * N, 2)[valid_mask_flat]  # [V, 2]
                px = (cen_flat[:, 0] * lW).long().clamp(0, lW - 1)
                py = (cen_flat[:, 1] * lH).long().clamp(0, lH - 1)
                token_classes = torch.tensor(
                    [int(lbl_img[py[k].item(), px[k].item()]) for k in range(V)],
                    dtype=torch.long,
                )

        # ── Per-expert: change sensitivity + PCA features + class counts ───
        e_ids_cpu = e_ids.cpu()
        for e in range(E):
            e_mask = (e_ids_cpu == e)
            if not e_mask.any():
                continue

            # Test 5 — change sensitivity
            per_expert_change[e].extend(change_probs_valid[e_mask].tolist())

            # Bonus — class distribution
            if token_classes is not None:
                for cls_id in token_classes[e_mask].tolist():
                    cls_id = int(np.clip(cls_id, 0, 6))
                    per_expert_class_counts[e, cls_id] += 1

            # Test 2 — collect expert output embeddings for PCA
            if len(feat_samples) < max_feat_samples:
                e_out = model.moe.experts[e](x_valid[e_mask]).float().cpu()  # [n_e, H]
                cls_batch = (
                    token_classes[e_mask].tolist()
                    if token_classes is not None
                    else [-1] * int(e_mask.sum().item())
                )
                for out_vec, cls_id in zip(e_out, cls_batch):
                    feat_samples.append((out_vec.detach(), e, int(cls_id)))
                    if len(feat_samples) >= max_feat_samples:
                        break

    return {
        "all_entropy":            all_entropy,
        "per_expert_change":      per_expert_change,
        "feat_samples":           feat_samples,
        "common_reprs":           common_reprs,
        "per_expert_class_counts": per_expert_class_counts,
        "total_tokens_per_expert": total_tokens_per_expert,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — PCA helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_pca_2d(X: torch.Tensor) -> torch.Tensor:
    """Project [N, H] tensor to [N, 2] using truncated SVD (PCA)."""
    Xf   = X.float()
    mean = Xf.mean(dim=0)
    Xc   = Xf - mean
    _, _, V = torch.pca_lowrank(Xc, q=2, niter=4)
    return torch.mm(Xc, V[:, :2])   # [N, 2]


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers — PIL only (no matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def _lerp_color(
    c_lo: Tuple[int, int, int],
    c_hi: Tuple[int, int, int],
    t: float,
) -> Tuple[int, int, int]:
    return tuple(int(c_lo[k] + (c_hi[k] - c_lo[k]) * t) for k in range(3))


def save_heatmap(
    matrix: np.ndarray,
    path: Path,
    title: str,
    labels: Optional[List[str]] = None,
    cmap: str = "blue",   # "blue" | "red" | "green"
) -> None:
    """Save an N×N numeric heatmap as a PNG using PIL."""
    E        = matrix.shape[0]
    CELL     = 88
    MARGIN_T = 52   # top margin (title)
    MARGIN_L = 52   # left margin (row labels)
    MARGIN_R = 16
    MARGIN_B = 16

    W   = MARGIN_L + E * CELL + MARGIN_R
    H   = MARGIN_T + E * CELL + MARGIN_B
    img = Image.new("RGB", (W, H), (248, 248, 248))
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((W // 2 - len(title) * 4, 8), title, fill=(30, 30, 30))

    C_LO = (240, 240, 240)
    if cmap == "blue":
        C_HI = (18, 58, 176)
    elif cmap == "red":
        C_HI = (196, 28, 28)
    else:
        C_HI = (28, 148, 58)

    vmin = float(matrix.min())
    vmax = float(matrix.max())
    span = vmax - vmin if vmax != vmin else 1.0

    if labels is None:
        labels = [f"E{i}" for i in range(E)]

    for i in range(E):
        for j in range(E):
            val = float(matrix[i, j])
            t   = (val - vmin) / span
            color = _lerp_color(C_LO, C_HI, t)
            x0 = MARGIN_L + j * CELL
            y0 = MARGIN_T + i * CELL
            draw.rectangle(
                [x0, y0, x0 + CELL - 1, y0 + CELL - 1],
                fill=color, outline=(180, 180, 180),
            )
            txt = f"{val:.3f}"
            brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            txt_color  = (240, 240, 240) if brightness < 128 else (30, 30, 30)
            draw.text(
                (x0 + CELL // 2 - len(txt) * 3, y0 + CELL // 2 - 6),
                txt, fill=txt_color,
            )

    # Row & column labels
    for i, lbl in enumerate(labels):
        y0 = MARGIN_T + i * CELL + CELL // 2 - 6
        draw.text((2, y0), lbl[:6], fill=(60, 60, 60))
    for j, lbl in enumerate(labels):
        x0 = MARGIN_L + j * CELL + CELL // 2 - len(lbl) * 3
        draw.text((x0, MARGIN_T - 18), lbl[:6], fill=(60, 60, 60))

    img.save(path)


def save_scatter_pca(
    coords:     torch.Tensor,   # [N, 2]
    expert_ids: List[int],
    path:       Path,
    E:          int,
) -> None:
    """PCA scatter plot colored by expert, with a simple legend."""
    W, H   = 960, 720
    MARGIN = 60
    LEG_W  = 140
    PW     = W - 2 * MARGIN - LEG_W
    PH     = H - 2 * MARGIN - 30

    img  = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text(
        (W // 2 - 120, 10),
        "Expert Feature Distribution (PCA)",
        fill=(30, 30, 30),
    )

    coords_np = coords.numpy()
    c_min  = coords_np.min(axis=0)
    c_range = coords_np.max(axis=0) - c_min + 1e-6

    for idx, (cx, cy) in enumerate(coords_np):
        nx    = (cx - c_min[0]) / c_range[0]
        ny    = (cy - c_min[1]) / c_range[1]
        px    = int(nx * PW)  + MARGIN
        py    = int((1.0 - ny) * PH) + MARGIN   # flip y
        e_id  = expert_ids[idx]
        color = EXPERT_COLORS[e_id % len(EXPERT_COLORS)]
        draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=color)

    # Legend
    lx = W - LEG_W + 8
    draw.text((lx, MARGIN), "Experts:", fill=(40, 40, 40))
    for e in range(E):
        ly = MARGIN + 22 + e * 24
        draw.rectangle(
            [lx, ly, lx + 14, ly + 14],
            fill=EXPERT_COLORS[e % len(EXPERT_COLORS)],
        )
        draw.text((lx + 18, ly), f"Expert {e}", fill=(40, 40, 40))

    # Plot border + axis labels
    draw.rectangle(
        [MARGIN, MARGIN, MARGIN + PW, MARGIN + PH],
        outline=(180, 180, 180),
    )
    draw.text((MARGIN + PW // 2 - 12, H - 22), "PC1 →", fill=(100, 100, 100))
    draw.text((4, MARGIN + PH // 2 - 6), "PC2", fill=(100, 100, 100))

    img.save(path)


def save_bar_chart(
    values: np.ndarray,
    labels: List[str],
    path: Path,
    title: str,
) -> None:
    """Horizontal bar chart for routing load distribution."""
    BAR_H   = 36
    LABEL_W = 80
    PAD     = 12
    W       = 560
    H       = PAD * 2 + 30 + len(labels) * (BAR_H + 8)

    img  = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((W // 2 - len(title) * 4, 8), title, fill=(30, 30, 30))

    max_v    = float(max(values)) if max(values) > 0 else 1.0
    BAR_MAX  = W - LABEL_W - PAD * 2 - 60

    for i, (v, lbl) in enumerate(zip(values, labels)):
        y0    = PAD + 30 + i * (BAR_H + 8)
        bar_w = int((v / max_v) * BAR_MAX)
        draw.text((PAD, y0 + 10), lbl, fill=(60, 60, 60))
        draw.rectangle(
            [LABEL_W, y0, LABEL_W + bar_w, y0 + BAR_H],
            fill=EXPERT_COLORS[i % len(EXPERT_COLORS)],
            outline=(100, 100, 100),
        )
        draw.text(
            (LABEL_W + bar_w + 6, y0 + 10),
            f"{v:.1%}",
            fill=(30, 30, 30),
        )

    img.save(path)


# ─────────────────────────────────────────────────────────────────────────────
# Interpretation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _status(mean_sim: float) -> Tuple[str, str]:
    """Return (status_tag, icon)."""
    if mean_sim > COLLAPSE_THRESHOLD:
        return "COLLAPSED", "⚠️"
    if mean_sim > SPECIALIZATION_THRESHOLD:
        return "PARTIAL",   "⚡"
    return "STABLE",    "✓"


def _mean_off_diag(mat: np.ndarray) -> float:
    E = mat.shape[0]
    return float((mat.sum() - np.trace(mat)) / max(E * (E - 1), 1))


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    out_dir:                 Path,
    E:                       int,
    sim_matrix:              np.ndarray,
    param_dist:              np.ndarray,
    grad_sim:                np.ndarray,
    per_expert_change:       List[List[float]],
    all_entropy:             List[float],
    total_tokens_per_expert: np.ndarray,
    per_expert_class_counts: np.ndarray,
    has_labels:              bool,
) -> None:

    mean_sim      = _mean_off_diag(sim_matrix)
    mean_grad_sim = _mean_off_diag(grad_sim)
    mean_entropy  = float(np.mean(all_entropy)) if all_entropy else 0.0
    max_entropy   = float(np.log(E))
    total_tokens  = int(total_tokens_per_expert.sum())
    status, icon  = _status(mean_sim)

    expert_labels = [f"E{e}" for e in range(E)]
    row_sep = "|---|" + "---|" * E + "\n"
    hdr_row = "| | " + " | ".join(expert_labels) + " |\n"

    with open(out_dir / "expert_diversity_report.md", "w") as f:

        # ── Header ──────────────────────────────────────────────────────────
        f.write("# Expert Diversity Report\n\n")
        f.write(f"**Status: {icon} {status}**\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| Mean Off-Diagonal Output Similarity | `{mean_sim:.4f}` |\n")
        f.write(f"| Mean Gradient Similarity             | `{mean_grad_sim:.4f}` |\n")
        f.write(f"| Mean Router Entropy                  | `{mean_entropy:.4f}` |\n")
        f.write(f"| Normalized Entropy (0→1)             | `{mean_entropy/max(max_entropy,1e-9):.3f}` |\n")
        f.write(f"| Total Tokens Analyzed                | `{total_tokens:,}` |\n")
        f.write("\n---\n\n")

        # ── Test 1: Output Similarity ────────────────────────────────────────
        f.write("## Test 1 — Expert Output Similarity\n\n")
        f.write(
            "_Same tokens forced through every expert; "
            "cosine similarity of normalized output vectors._\n\n"
        )
        f.write(hdr_row + row_sep)
        for i in range(E):
            f.write(
                f"| {expert_labels[i]} | "
                + " | ".join(f"{sim_matrix[i,j]:.4f}" for j in range(E))
                + " |\n"
            )
        f.write("\n")

        if mean_sim > COLLAPSE_THRESHOLD:
            f.write(
                f"**⚠️ ALERT:** Mean similarity `{mean_sim:.3f}` > `{COLLAPSE_THRESHOLD}` "
                f"→ experts are likely collapsed into identical functions.\n\n"
            )
        elif mean_sim < SPECIALIZATION_THRESHOLD:
            f.write(
                f"**✓ GOOD:** Mean similarity `{mean_sim:.3f}` < `{SPECIALIZATION_THRESHOLD}` "
                f"→ experts are functionally diverse.\n\n"
            )
        else:
            f.write(
                f"**⚡ PARTIAL:** Mean similarity `{mean_sim:.3f}` — "
                f"moderate specialization between `{SPECIALIZATION_THRESHOLD}` and `{COLLAPSE_THRESHOLD}`.\n\n"
            )

        f.write("![Output Similarity](output_similarity_matrix.png)\n\n")

        # ── Test 2: PCA ──────────────────────────────────────────────────────
        f.write("## Test 2 — Expert Feature Distribution (PCA)\n\n")
        f.write(
            "Expert output embeddings projected to 2D via PCA. "
            "Overlapping clusters indicate experts produce similar representations.\n\n"
        )
        f.write("![PCA Scatter](expert_pca.png)\n\n")

        # ── Test 3: Routing Entropy ──────────────────────────────────────────
        f.write("## Test 3 — Routing Entropy\n\n")
        f.write("| Metric | Value | Note |\n|---|---|---|\n")
        f.write(
            f"| Mean per-token entropy | `{mean_entropy:.4f}` "
            f"| Maximum possible: `{max_entropy:.4f}` |\n"
        )
        f.write(
            f"| Normalized entropy     | `{mean_entropy/max(max_entropy,1e-9):.3f}` "
            f"| 0 = fully collapsed, 1 = uniform |\n\n"
        )
        if mean_entropy < 0.10:
            f.write(
                "**Sharp routing** — the router assigns each token to one expert "
                "with >95% confidence. Decision boundaries are well-formed.\n\n"
            )
        elif mean_entropy > 0.80 * max_entropy:
            f.write(
                "**Diffuse routing** — router entropy is near-maximum, indicating "
                "near-random expert assignment. The router has not learned discriminative boundaries.\n\n"
            )
        else:
            f.write(
                "**Moderate entropy** — router shows some uncertainty; "
                "consider increasing training steps or adding routing noise.\n\n"
            )

        # ── Test 4: Parameter Distance ───────────────────────────────────────
        f.write("## Test 4 — Expert Parameter Distance\n\n")
        f.write(
            "_L2 distance between `expert[i].net[0].weight` and `expert[j].net[0].weight`._\n\n"
        )
        f.write(hdr_row + row_sep)
        for i in range(E):
            f.write(
                f"| {expert_labels[i]} | "
                + " | ".join(f"{param_dist[i,j]:.2f}" for j in range(E))
                + " |\n"
            )
        f.write("\n")

        off_diag_mask = ~np.eye(E, dtype=bool)
        min_dist = float(param_dist[off_diag_mask].min()) if E > 1 else 0.0
        mean_dist = float(param_dist[off_diag_mask].mean()) if E > 1 else 0.0
        f.write(
            f"Min non-diagonal distance: `{min_dist:.2f}`, "
            f"Mean: `{mean_dist:.2f}`.\n"
        )
        if min_dist < 1.0:
            f.write(
                " **Very small** — at least one expert pair has barely diverged "
                "from initialization; consider expert dropout or diversity loss.\n\n"
            )
        else:
            f.write(" **Healthy weight-space separation.**\n\n")

        f.write("![Parameter Distance](parameter_distance_matrix.png)\n\n")

        # ── Test 5: Change Sensitivity ───────────────────────────────────────
        f.write("## Test 5 — Expert Change Sensitivity\n\n")
        f.write(
            "| Expert | Tokens Routed | Load % | Avg Change Prob | Role |\n"
            "|---|---|---|---|---|\n"
        )
        for e in range(E):
            n_e   = int(total_tokens_per_expert[e])
            pct   = float(n_e / max(total_tokens, 1) * 100)
            avg_cp = float(np.mean(per_expert_change[e])) if per_expert_change[e] else 0.0
            role  = "Change processor (high)" if avg_cp > 0.35 else "Stability validator (low)"
            f.write(f"| Expert {e} | {n_e:,} | {pct:.1f}% | {avg_cp:.4f} | {role} |\n")
        f.write("\n")
        f.write("![Routing Distribution](routing_distribution.png)\n\n")

        # ── Test 6: Gradient Similarity ──────────────────────────────────────
        f.write("## Test 6 — Gradient Similarity\n\n")
        f.write(
            "_Each expert's gradient `∂(mean(E_i(x)²)) / ∂W_i¹` computed on the same "
            "token batch (forced dispatch). Cosine similarity of flattened gradient vectors._\n\n"
        )
        f.write(hdr_row + row_sep)
        for i in range(E):
            f.write(
                f"| {expert_labels[i]} | "
                + " | ".join(f"{grad_sim[i,j]:.4f}" for j in range(E))
                + " |\n"
            )
        f.write("\n")

        if mean_grad_sim > 0.90:
            f.write(
                f"**⚠️ ALERT:** Gradient similarity `{mean_grad_sim:.3f}` > `0.9` "
                f"→ experts update in nearly identical directions. Strong collapse signal.\n\n"
            )
        elif mean_grad_sim > 0.70:
            f.write(
                f"**⚡ Warning:** Gradient similarity `{mean_grad_sim:.3f}` is moderately high.\n\n"
            )
        else:
            f.write(
                f"**✓ GOOD:** Gradient similarity `{mean_grad_sim:.3f}` "
                f"→ experts learn distinct parameter updates.\n\n"
            )

        f.write("![Gradient Similarity](gradient_similarity_matrix.png)\n\n")

        # ── Bonus: Class breakdown (if labels available) ─────────────────────
        if has_labels:
            f.write("## Bonus — Expert × Semantic Class Distribution\n\n")
            cls_hdr = "| Expert | " + " | ".join(CLASS_NAMES) + " |\n"
            cls_sep = "|---|" + "---|" * len(CLASS_NAMES) + "\n"
            f.write(cls_hdr + cls_sep)
            for e in range(E):
                row_total = int(per_expert_class_counts[e].sum())
                cells = " | ".join(
                    f"{per_expert_class_counts[e,c]} "
                    f"({per_expert_class_counts[e,c]/max(row_total,1):.0%})"
                    for c in range(7)
                )
                f.write(f"| Expert {e} | {cells} |\n")
            f.write("\n")

        # ── Overall Judgment ─────────────────────────────────────────────────
        f.write("---\n\n## Overall Judgment\n\n")

        collapse_signals = []
        if mean_sim > COLLAPSE_THRESHOLD:
            collapse_signals.append(
                f"Output similarity too high: `{mean_sim:.3f}` > `{COLLAPSE_THRESHOLD}`"
            )
        if mean_grad_sim > 0.90:
            collapse_signals.append(
                f"Gradient similarity too high: `{mean_grad_sim:.3f}`"
            )
        if mean_entropy < 0.05:
            collapse_signals.append(
                f"Routing entropy extremely low: `{mean_entropy:.4f}`"
            )

        if collapse_signals:
            f.write("**MoE Collapse Detected — signals:**\n\n")
            for sig in collapse_signals:
                f.write(f"- {sig}\n")
            f.write("\n")
        else:
            f.write(
                f"**No collapse detected.** "
                f"Experts are functionally diverse "
                f"(output_sim=`{mean_sim:.3f}`, grad_sim=`{mean_grad_sim:.3f}`).\n\n"
            )

        # Hidden task-level specialization (change vs stable)
        cps   = [
            float(np.mean(per_expert_change[e])) if per_expert_change[e] else 0.0
            for e in range(E)
        ]
        high  = [e for e in range(E) if cps[e] > 0.35]
        low   = [e for e in range(E) if cps[e] <= 0.35]
        if high and low and not collapse_signals:
            f.write("**Hidden task-level specialization detected:**\n\n")
            f.write(
                f"- Change processors "
                f"(avg change_prob > 0.35): "
                + ", ".join(f"Expert {e}" for e in high) + "\n"
            )
            f.write(
                f"- Stability validators "
                f"(avg change_prob ≤ 0.35): "
                + ", ".join(f"Expert {e}" for e in low) + "\n\n"
            )
            f.write(
                "Even without class-level purity, "
                "experts have specialized around the **task structure** "
                "(detecting change vs. confirming stability).\n\n"
            )

        # ── Improvement Suggestions ──────────────────────────────────────────
        f.write("## Improvement Suggestions\n\n")
        if collapse_signals:
            f.write(
                "Collapse detected. Recommended interventions "
                "(ordered by estimated impact):\n\n"
            )
            f.write(
                "1. **Expert dropout** "
                "(`expert_dropout_prob=0.1`): randomly disable one expert per step, "
                "forcing the remaining experts to compensate → stronger specialization.\n"
            )
            f.write(
                "2. **Diversity loss**: add "
                "`λ_div × Σ_{i≠j} cosine_sim(E_i(x), E_j(x))` to the training loss "
                "as a direct output-level penalty for similarity.\n"
            )
            f.write(
                "3. **Orthogonality regularization**: "
                "penalize `‖W_i^T W_j‖_F` across expert weight matrices.\n"
            )
            f.write(
                "4. **More experts**: increase `moe_num_experts` (e.g., 8). "
                "More experts → stronger competitive pressure to diverge.\n"
            )
            f.write(
                "5. **Routing noise** (Gumbel softmax): "
                "inject Gumbel noise into logits during training "
                "(`logits += Gumbel(0, 1)`) to encourage exploration.\n\n"
            )
        else:
            f.write("No collapse. Possible further improvements:\n\n")
            f.write(
                "1. **Imbalance-aware routing**: "
                "if all experts are dominated by one class (e.g., `low_veg`), "
                "remove the load-balance loss and switch to class-weighted expert assignment.\n"
            )
            f.write(
                "2. **Early semantic injection**: "
                "fuse class one-hot embeddings into `TokenEncoder` at Stage 1, not just the MoE router, "
                "so the entire network benefits from semantic context.\n"
            )
            f.write(
                "3. **Top-2 routing** (`use_top2=True`): "
                "let each token blend two expert outputs — "
                "smoother gradient flow, better utilization of all experts.\n"
            )
            f.write(
                "4. **Longer training**: 30 epochs may not be enough for specialization "
                "to emerge fully. Consider 50–100 epochs with a lower learning rate.\n\n"
            )

    print(f"[✓] Report saved: {out_dir / 'expert_diversity_report.md'}")


# ─────────────────────────────────────────────────────────────────────────────
# metrics.csv
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_csv(
    out_dir:                 Path,
    E:                       int,
    sim_matrix:              np.ndarray,
    param_dist:              np.ndarray,
    grad_sim:                np.ndarray,
    all_entropy:             List[float],
    per_expert_change:       List[List[float]],
    total_tokens_per_expert: np.ndarray,
) -> None:
    csv_path = out_dir / "metrics.csv"
    rows = []

    mean_entropy  = float(np.mean(all_entropy)) if all_entropy else 0.0
    mean_sim      = _mean_off_diag(sim_matrix)
    mean_grad_sim = _mean_off_diag(grad_sim)

    rows += [
        ("mean_entropy",       mean_entropy),
        ("mean_off_diag_sim",  mean_sim),
        ("mean_grad_sim",      mean_grad_sim),
    ]
    for e in range(E):
        cp  = float(np.mean(per_expert_change[e])) if per_expert_change[e] else 0.0
        n_e = int(total_tokens_per_expert[e])
        rows += [
            (f"expert_{e}_change_prob",   cp),
            (f"expert_{e}_token_count",   n_e),
        ]
    for i in range(E):
        for j in range(E):
            rows += [
                (f"sim_{i}_{j}",       round(float(sim_matrix[i, j]),  6)),
                (f"param_dist_{i}_{j}", round(float(param_dist[i, j]), 4)),
                (f"grad_sim_{i}_{j}",  round(float(grad_sim[i, j]),   6)),
            ]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(rows)

    print(f"[✓] Metrics saved: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(
    E:                       int,
    sim_matrix:              np.ndarray,
    grad_sim:                np.ndarray,
    param_dist:              np.ndarray,
    all_entropy:             List[float],
    per_expert_change:       List[List[float]],
    total_tokens_per_expert: np.ndarray,
) -> None:
    mean_sim      = _mean_off_diag(sim_matrix)
    mean_grad_sim = _mean_off_diag(grad_sim)
    mean_entropy  = float(np.mean(all_entropy)) if all_entropy else 0.0
    max_entropy   = float(np.log(E))
    status, icon  = _status(mean_sim)

    print(f"\n{'='*58}")
    print(f"  Expert Diversity Summary")
    print(f"{'='*58}")
    print(f"  Status                  : {icon}  {status}")
    print(f"  Output Sim (off-diag)   : {mean_sim:.4f}  "
          f"(collapse if > {COLLAPSE_THRESHOLD})")
    print(f"  Gradient Sim (off-diag) : {mean_grad_sim:.4f}  "
          f"(collapse if > 0.9)")
    print(f"  Router Entropy          : {mean_entropy:.4f}  "
          f"(max={max_entropy:.4f})")
    print(f"  {'─'*52}")
    total = int(total_tokens_per_expert.sum())
    for e in range(E):
        cp  = float(np.mean(per_expert_change[e])) if per_expert_change[e] else 0.0
        n_e = int(total_tokens_per_expert[e])
        pct = float(n_e / max(total, 1) * 100)
        role = "change" if cp > 0.35 else "stable"
        print(f"  Expert {e} : {n_e:6,} tokens ({pct:4.1f}%)  "
              f"change_prob={cp:.4f}  [{role}]")
    print(f"{'='*58}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="MoE Expert Diversity Collapse Test v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",  required=True,
                   help="Path to model checkpoint (best_model.pt or checkpoint.pt)")
    p.add_argument("--config",      required=True,
                   help="Path to config.json saved alongside the checkpoint")
    p.add_argument("--tokens_T1",   default="SECOND/tokens_T1")
    p.add_argument("--tokens_T2",   default="SECOND/tokens_T2")
    p.add_argument("--matches",     default="SECOND/matches")
    p.add_argument("--labels",      default=None,
                   help="Optional: label PNG dir (e.g. SECOND/label1) for class breakdown")
    p.add_argument("--output",      default="stage5/diversity",
                   help="Directory where all outputs are written")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--samples",     type=int, default=200,
                   help="Number of image pairs to use for the analysis")
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    device = torch.device(
        args.device if (args.device == "cpu" or not torch.cuda.is_available())
        else args.device
    )
    print(f"[diversity] Device      : {device}")

    cfg   = load_config(Path(args.config))
    model = load_model(Path(args.checkpoint), cfg, device)
    E     = model.moe.num_experts
    print(f"[diversity] Model loaded: E={E} experts, router_version={cfg.router_version}")

    # ── Sample stems ─────────────────────────────────────────────────────────
    all_stems = sorted([
        mp.stem.replace("_matches", "")
        for mp in Path(args.matches).glob("*_matches.pt")
    ])
    random.seed(args.seed)
    stems = (
        random.sample(all_stems, args.samples)
        if args.samples < len(all_stems)
        else all_stems
    )
    print(f"[diversity] Using {len(stems)} / {len(all_stems)} image pairs")

    # ── Test 4: Parameter Distance (no data needed) ──────────────────────────
    print("[diversity] Test 4 : computing parameter distances...")
    param_dist = test_parameter_distance(model)

    # ── Inference pass (Tests 2 / 3 / 5 + gather x_repr) ───────────────────
    print("[diversity] Tests 2,3,5: running inference pass...")
    results = run_inference_pass(
        model,
        stems      = stems,
        t1_dir     = Path(args.tokens_T1),
        t2_dir     = Path(args.tokens_T2),
        match_dir  = Path(args.matches),
        label_dir  = Path(args.labels) if args.labels else None,
        device     = device,
    )

    # ── Stack common reprs for Tests 1 & 6 ───────────────────────────────────
    if not results["common_reprs"]:
        print("[diversity] ERROR: No valid tokens collected — check paths.")
        return
    x_repr = torch.cat(results["common_reprs"], dim=0)   # [T, H]
    print(f"[diversity] Collected {x_repr.shape[0]} common repr tokens")

    # ── Test 1: Output Similarity ─────────────────────────────────────────────
    print("[diversity] Test 1 : computing output similarity...")
    sim_matrix = test_output_similarity(model, x_repr, device)

    # ── Test 6: Gradient Similarity ───────────────────────────────────────────
    print("[diversity] Test 6 : computing gradient similarity...")
    grad_sim = test_gradient_similarity(model, x_repr, device)

    # ── Test 2: PCA ───────────────────────────────────────────────────────────
    feat_samples = results["feat_samples"]
    pca_coords   = None
    if feat_samples:
        print(f"[diversity] Test 2 : PCA on {len(feat_samples)} feature samples...")
        fs_tensor  = torch.stack([s[0] for s in feat_samples])
        pca_coords = compute_pca_2d(fs_tensor)

    # ── Create output dir ─────────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    expert_labels = [f"E{e}" for e in range(E)]

    # ── Save visualizations ───────────────────────────────────────────────────
    print("[diversity] Saving visualizations...")

    save_heatmap(
        sim_matrix,
        out_dir / "output_similarity_matrix.png",
        "Expert Output Similarity",
        expert_labels, cmap="blue",
    )
    save_heatmap(
        param_dist,
        out_dir / "parameter_distance_matrix.png",
        "Expert Parameter Distance",
        expert_labels, cmap="red",
    )
    save_heatmap(
        grad_sim,
        out_dir / "gradient_similarity_matrix.png",
        "Expert Gradient Similarity",
        expert_labels, cmap="blue",
    )

    tot = results["total_tokens_per_expert"]
    fracs = tot.astype(float) / max(float(tot.sum()), 1.0)
    save_bar_chart(
        fracs,
        [f"Expert {e}" for e in range(E)],
        out_dir / "routing_distribution.png",
        "Routing Load Distribution",
    )

    if pca_coords is not None:
        save_scatter_pca(
            pca_coords,
            expert_ids = [s[1] for s in feat_samples],
            path       = out_dir / "expert_pca.png",
            E          = E,
        )

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(
        E,
        sim_matrix, grad_sim, param_dist,
        results["all_entropy"],
        results["per_expert_change"],
        results["total_tokens_per_expert"],
    )

    # ── Save report + CSV ─────────────────────────────────────────────────────
    generate_report(
        out_dir, E,
        sim_matrix              = sim_matrix,
        param_dist              = param_dist,
        grad_sim                = grad_sim,
        per_expert_change       = results["per_expert_change"],
        all_entropy             = results["all_entropy"],
        total_tokens_per_expert = results["total_tokens_per_expert"],
        per_expert_class_counts = results["per_expert_class_counts"],
        has_labels              = (args.labels is not None),
    )
    save_metrics_csv(
        out_dir, E,
        sim_matrix              = sim_matrix,
        param_dist              = param_dist,
        grad_sim                = grad_sim,
        all_entropy             = results["all_entropy"],
        per_expert_change       = results["per_expert_change"],
        total_tokens_per_expert = results["total_tokens_per_expert"],
    )

    print(f"[✓] All outputs saved to: {out_dir}/")


if __name__ == "__main__":
    main()
