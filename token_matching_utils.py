"""
token_matching_utils.py
=======================
Pure utility functions for Stage 3 - Token Matching.

All operations are vectorized (PyTorch), GPU-ready, with CPU fallback.
No Python loops over individual tokens except for visualization helpers.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Embedding utilities
# ─────────────────────────────────────────────────────────────────────────────

def normalize_embeddings(tokens: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    L2-normalize token embeddings along the feature dimension.

    Args:
        tokens: Tensor [N, D]
        eps:    small constant for numerical stability
    Returns:
        Tensor [N, D]  — unit-norm rows
    """
    norms = tokens.norm(dim=1, keepdim=True).clamp(min=eps)
    return tokens / norms


# ─────────────────────────────────────────────────────────────────────────────
# 2. Similarity / distance matrices  (fully vectorized)
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity_matrix(
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix.

    Args:
        A: Tensor [N1, D]  (already L2-normalised)
        B: Tensor [N2, D]  (already L2-normalised)
    Returns:
        Tensor [N1, N2]  in range [-1, 1]
    """
    return torch.mm(A, B.t())


def centroid_distance_matrix(
    C1: torch.Tensor,
    C2: torch.Tensor,
) -> torch.Tensor:
    """
    Pairwise Euclidean distance between normalised centroids (∈ [0,1]²).

    Args:
        C1: Tensor [N1, 2]
        C2: Tensor [N2, 2]
    Returns:
        Tensor [N1, N2]  — distances in [0, √2]
    """
    # Expand for broadcasting: (N1,1,2) - (1,N2,2) → (N1,N2,2)
    diff = C1.unsqueeze(1) - C2.unsqueeze(0)       # (N1, N2, 2)
    return diff.pow(2).sum(dim=2).sqrt()            # (N1, N2)


def fused_similarity_matrix(
    tokens_t1_norm: torch.Tensor,
    tokens_t2_norm: torch.Tensor,
    centroids_t1: torch.Tensor,
    centroids_t2: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    spatial_gate_dist: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Fused similarity:  S_ij = α·cos(i,j) − β·dist(i,j)

    With optional spatial gating: pairs with centroid distance > spatial_gate_dist
    are hard-zeroed (set to very negative value) before matching.

    Args:
        tokens_t1_norm:   [N1, D]  L2-normalized
        tokens_t2_norm:   [N2, D]  L2-normalized
        centroids_t1:     [N1, 2]
        centroids_t2:     [N2, 2]
        alpha:            cosine weight
        beta:             spatial penalty weight
        spatial_gate_dist: hard-gate threshold (0 = disabled)

    Returns:
        sim_matrix:  Tensor [N1, N2]  — fused similarity
        cos_matrix:  Tensor [N1, N2]  — raw cosine (for statistics)
        stats:       dict with mean_cosine, std_cosine, gated_fraction
    """
    cos_mat = cosine_similarity_matrix(tokens_t1_norm, tokens_t2_norm)   # [N1,N2]
    dist_mat = centroid_distance_matrix(centroids_t1, centroids_t2)       # [N1,N2]

    sim = alpha * cos_mat - beta * dist_mat                               # [N1,N2]

    # Spatial gating: impossible pairs → –∞
    gated_mask = torch.zeros_like(sim, dtype=torch.bool)
    if spatial_gate_dist > 0:
        gated_mask = dist_mat > spatial_gate_dist
        sim = sim.masked_fill(gated_mask, float("-inf"))

    # Similarity statistics on VALID (non-gated) cosine values
    valid_cos = cos_mat[~gated_mask]
    if valid_cos.numel() > 0:
        mean_cos = float(valid_cos.mean())
        std_cos  = float(valid_cos.std())
    else:
        mean_cos = 0.0
        std_cos  = 0.0

    stats = {
        "mean_cosine":    mean_cos,
        "std_cosine":     std_cos,
        "gated_fraction": float(gated_mask.float().mean()),
    }
    return sim, cos_mat, stats


# ─────────────────────────────────────────────────────────────────────────────
# 3. Top-K pruning helper  (for fast Hungarian)
# ─────────────────────────────────────────────────────────────────────────────

def topk_pruned_cost_matrix(
    sim_matrix: torch.Tensor,
    top_k: int,
    fill_val: float = -1e6,
) -> torch.Tensor:
    """
    For each T1 token (row), keep only its top-K T2 neighbours.
    All other entries are set to fill_val (very negative).

    This turns an N1×N2 dense assignment into a sparse one,
    making Hungarian 10× faster for large N.

    Args:
        sim_matrix: [N1, N2]
        top_k:      number of candidates to keep per T1 token
        fill_val:   penalty for pruned entries
    Returns:
        Tensor [N1, N2]  — same shape, non-topk cells set to fill_val
    """
    N1, N2 = sim_matrix.shape
    k = min(top_k, N2)

    # Indices of top-k in each row
    topk_idx = sim_matrix.topk(k, dim=1).indices     # (N1, k)

    pruned = torch.full_like(sim_matrix, fill_val)
    # Scatter top-k values back
    pruned.scatter_(1, topk_idx, sim_matrix.gather(1, topk_idx))
    return pruned


# ─────────────────────────────────────────────────────────────────────────────
# 4. Soft matching  (softmax / Sinkhorn)
# ─────────────────────────────────────────────────────────────────────────────

def soft_matrix_softmax(
    sim_matrix: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Row-wise softmax with temperature scaling.

    P_ij = softmax(S / T)_row-i

    Handles -inf entries from spatial gating gracefully
    (they become 0 after softmax).

    Args:
        sim_matrix:  [N1, N2]
        temperature: smaller → sharper / more peaked
    Returns:
        Tensor [N1, N2]  — rows sum to 1 (for valid rows)
    """
    return F.softmax(sim_matrix / temperature, dim=1)


def sinkhorn_normalize(
    M: torch.Tensor,
    n_iters: int = 20,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Sinkhorn–Knopp normalization: doubly-stochastic matrix from a non-negative matrix.

    Args:
        M:       Tensor [N1, N2]  — non-negative (e.g. after exp)
        n_iters: number of row/col normalization rounds
        eps:     small value to avoid division by zero
    Returns:
        Tensor [N1, N2]  — approximately doubly stochastic
    """
    # Replace -inf / nan from gating before Sinkhorn
    M = M.clone()
    M[~torch.isfinite(M)] = 0.0
    M = M.clamp(min=0.0)

    for _ in range(n_iters):
        M = M / (M.sum(dim=1, keepdim=True) + eps)   # row normalize
        M = M / (M.sum(dim=0, keepdim=True) + eps)   # col normalize
    return M


def soft_matrix_sinkhorn(
    sim_matrix: torch.Tensor,
    temperature: float = 0.1,
    n_iters: int = 20,
) -> torch.Tensor:
    """
    Sinkhorn-normalized soft matching matrix.

    Steps:
      1. Scale: S / T
      2. Exp (log-domain softmax base)
      3. Sinkhorn iterations

    Args:
        sim_matrix:  [N1, N2]
        temperature: scaling before exp
        n_iters:     Sinkhorn iterations
    Returns:
        Tensor [N1, N2]  — doubly-stochastic
    """
    # Clamp to avoid overflow in exp (finite entries only)
    S = sim_matrix / temperature
    S = S.clamp(max=80.0)           # exp(80) ≈ 5e34, safe float32
    S[~torch.isfinite(S)] = -1e6    # gated entries → near zero after exp

    K = S.exp()
    return sinkhorn_normalize(K, n_iters=n_iters)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cross-attention matching  (optional)
# ─────────────────────────────────────────────────────────────────────────────

def cross_attention_matching(
    tokens_t1: torch.Tensor,
    tokens_t2: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Single-head scaled dot-product cross-attention.

    Q = tokens_t1,  K = V = tokens_t2
    Output: attention weights [N1, N2] = matching probabilities.

    Args:
        tokens_t1: [N1, D]   (raw, not necessarily normalised)
        tokens_t2: [N2, D]
        scale:     optional override for 1/√D
    Returns:
        Tensor [N1, N2]  — attention probabilities (rows sum to 1)
    """
    D = tokens_t1.shape[1]
    if scale is None:
        scale = D ** -0.5

    # (N1, D) × (D, N2) → (N1, N2)
    attn_logits = torch.mm(tokens_t1, tokens_t2.t()) * scale
    return F.softmax(attn_logits, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Split / merge detection heuristic
# ─────────────────────────────────────────────────────────────────────────────

def detect_splits_merges(
    pairs: List[Tuple[int, int, float]],
    areas_t1: torch.Tensor,
    areas_t2: torch.Tensor,
    area_ratio_threshold: float = 0.6,
) -> dict:
    """
    Heuristic to flag split/merge events after Hungarian matching.

    Logic:
      - Split:  one T1 token maps to multiple T2 tokens with large combined area
      - Merge:  multiple T1 tokens map to one T2 token with large combined area

    Args:
        pairs:                list of (i, j, score) tuples / Tensor rows
        areas_t1:             [N1]
        areas_t2:             [N2]
        area_ratio_threshold: sum_matched_area / own_area to flag split/merge

    Returns:
        dict with keys:
            "splits":  list of T1 indices flagged as split
            "merges":  list of T2 indices flagged as merge
    """
    from collections import defaultdict

    t1_to_t2 = defaultdict(list)   # T1_i → [T2_j, ...]
    t2_to_t1 = defaultdict(list)   # T2_j → [T1_i, ...]

    for i, j, _ in pairs:
        t1_to_t2[int(i)].append(int(j))
        t2_to_t1[int(j)].append(int(i))

    splits = []
    for i, js in t1_to_t2.items():
        if len(js) <= 1:
            continue
        a_t1 = float(areas_t1[i])
        if a_t1 < 1e-8:
            continue
        a_sum_t2 = float(areas_t2[js].sum())
        if a_sum_t2 / a_t1 > area_ratio_threshold:
            splits.append(i)

    merges = []
    for j, is_ in t2_to_t1.items():
        if len(is_) <= 1:
            continue
        a_t2 = float(areas_t2[j])
        if a_t2 < 1e-8:
            continue
        a_sum_t1 = float(areas_t1[is_].sum())
        if a_sum_t1 / a_t2 > area_ratio_threshold:
            merges.append(j)

    return {"splits": splits, "merges": merges}


# ─────────────────────────────────────────────────────────────────────────────
# 7. Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_match_metrics(
    pairs: list,
    N1: int,
    N2: int,
    scores: Optional[List[float]] = None,
    gt_pairs: Optional[List[Tuple[int, int]]] = None,
) -> dict:
    """
    Compute matching quality metrics.

    If gt_pairs (ground-truth) is provided: compute precision/recall/F1.
    Otherwise compute proxy metrics.

    Args:
        pairs:    list of (i, j, score) — matched token index pairs
        N1:       number of T1 tokens
        N2:       number of T2 tokens
        scores:   optional list of match scores (same length as pairs)
        gt_pairs: optional list of ground-truth (i, j) pairs
    Returns:
        dict of metrics
    """
    matched_t1  = {int(p[0]) for p in pairs}
    matched_t2  = {int(p[1]) for p in pairs}
    unmatched_t1 = list(set(range(N1)) - matched_t1)
    unmatched_t2 = list(set(range(N2)) - matched_t2)

    match_count = len(pairs)
    unmatched_ratio = (len(unmatched_t1) + len(unmatched_t2)) / max(N1 + N2, 1)

    mean_score = float(np.mean([p[2] for p in pairs])) if pairs else 0.0

    metrics: dict = {
        "n_matches":       match_count,
        "n_t1":            N1,
        "n_t2":            N2,
        "unmatched_t1":    len(unmatched_t1),
        "unmatched_t2":    len(unmatched_t2),
        "unmatched_ratio": unmatched_ratio,
        "mean_score":      mean_score,
    }

    if gt_pairs is not None:
        gt_set = {(int(i), int(j)) for i, j in gt_pairs}
        pred_set = {(int(p[0]), int(p[1])) for p in pairs}
        tp = len(pred_set & gt_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall    = tp / len(gt_set)   if gt_set   else 0.0
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        metrics.update({
            "precision": precision,
            "recall":    recall,
            "F1":        f1,
        })
    else:
        # Proxy: treat matched_ratio as proxy recall (no GT available)
        proxy_recall = match_count / max(N1, 1)
        metrics.update({
            "precision": None,
            "recall":    None,
            "F1":        None,
            "proxy_recall": proxy_recall,
        })

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 8. IoU helper  (optional, if mask arrays available)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Compute IoU between two boolean/binary masks of the same shape.

    Args:
        mask_a: ndarray (H, W) bool
        mask_b: ndarray (H, W) bool
    Returns:
        float in [0, 1]
    """
    intersection = np.logical_and(mask_a, mask_b).sum()
    union        = np.logical_or(mask_a,  mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_image_optional(path: Optional[str]) -> Optional[np.ndarray]:
    """Try loading an image; return None if path is None or file missing."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        from PIL import Image as PILImage
        return np.array(PILImage.open(p).convert("RGB"))
    except Exception:
        return None


def plot_matches(
    stem: str,
    centroids_t1: torch.Tensor,
    centroids_t2: torch.Tensor,
    pairs: list,
    out_dir: Path,
    areas_t1: Optional[torch.Tensor] = None,
    areas_t2: Optional[torch.Tensor] = None,
    img_t1: Optional[np.ndarray] = None,
    img_t2: Optional[np.ndarray] = None,
    max_lines: int = 80,
) -> None:
    """
    Visualize matched token centroids.

    Plots T1 centroids on the left, T2 centroids on the right,
    connected by coloured lines (score-based colour).

    Args:
        stem:         image pair stem (for filename)
        centroids_t1: [N1, 2]  normalized (x,y)
        centroids_t2: [N2, 2]  normalized (x,y)
        pairs:        list of (i, j, score)
        out_dir:      output directory
        areas_t1/t2:  optional [N] tensors for point sizing
        img_t1/t2:    optional RGB images for background
        max_lines:    maximum match lines to draw (for clarity)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")

    # Axis titles
    axes[0].set_title("T1 Tokens", color="white", fontsize=12, fontweight="bold")
    axes[1].set_title("T2 Tokens", color="white", fontsize=12, fontweight="bold")

    c1 = centroids_t1.cpu().numpy()   # (N1, 2)
    c2 = centroids_t2.cpu().numpy()   # (N2, 2)

    # Background image
    for ax, img in zip(axes, [img_t1, img_t2]):
        if img is not None:
            ax.imshow(img, extent=[0, 1, 1, 0], aspect="auto", alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.axis("off")

    # Point size from area (if provided), else uniform
    def _sizes(areas, centroids):
        N = len(centroids)
        if areas is not None:
            a = areas.cpu().numpy()
            return np.clip(a * 4000, 20, 300)
        return np.full(N, 60)

    s1 = _sizes(areas_t1, c1)
    s2 = _sizes(areas_t2, c2)

    # Colour-map for matched tokens (by score)
    scores = [float(p[2]) for p in pairs] if pairs else [0.0]
    score_min, score_max = min(scores), max(scores)
    cmap = plt.cm.plasma
    norm = Normalize(vmin=score_min, vmax=score_max)
    sm   = ScalarMappable(cmap=cmap, norm=norm)

    matched_t1 = {int(p[0]) for p in pairs}
    matched_t2 = {int(p[1]) for p in pairs}

    # Unmatched tokens — grey
    um1 = [i for i in range(len(c1)) if i not in matched_t1]
    um2 = [j for j in range(len(c2)) if j not in matched_t2]

    if um1:
        axes[0].scatter(c1[um1, 0], c1[um1, 1], s=s1[um1], c="grey",
                        alpha=0.5, edgecolors="white", linewidths=0.3, zorder=3)
    if um2:
        axes[1].scatter(c2[um2, 0], c2[um2, 1], s=s2[um2], c="grey",
                        alpha=0.5, edgecolors="white", linewidths=0.3, zorder=3)

    # Matched tokens — plasma colour
    for rank, (i, j, sc) in enumerate(pairs[:max_lines]):
        col = cmap(norm(float(sc)))
        axes[0].scatter(c1[i, 0], c1[i, 1], s=s1[i], c=[col],
                        edgecolors="white", linewidths=0.5, zorder=4)
        axes[1].scatter(c2[j, 0], c2[j, 1], s=s2[j], c=[col],
                        edgecolors="white", linewidths=0.5, zorder=4)

    # Add connection lines between panels using axes transform
    # (draw in figure coordinates)
    for i, j, sc in pairs[:max_lines]:
        col = list(cmap(norm(float(sc))))
        col[3] = 0.5   # alpha
        # Convert data → figure coords
        xy1 = axes[0].transData.transform([c1[i, 0], c1[i, 1]])
        xy2 = axes[1].transData.transform([c2[j, 0], c2[j, 1]])
        xy1_fig = fig.transFigure.inverted().transform(xy1)
        xy2_fig = fig.transFigure.inverted().transform(xy2)
        fig.add_artist(
            mpatches.FancyArrowPatch(
                xy1_fig, xy2_fig,
                transform=fig.transFigure,
                arrowstyle="-",
                color=col,
                linewidth=0.8,
            )
        )

    # Colourbar
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label("Match Score", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    n_matched = len(pairs)
    n_shown   = min(n_matched, max_lines)
    fig.suptitle(
        f"{stem}  |  Matched: {n_matched}  |  T1: {len(c1)}  T2: {len(c2)}"
        + (f"  (showing {n_shown})" if n_shown < n_matched else ""),
        color="white", fontsize=11, y=1.01,
    )

    out_path = out_dir / f"{stem}_matches.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.debug(f"Saved match viz → {out_path}")


def plot_soft_heatmap(
    stem: str,
    soft_matrix: torch.Tensor,
    out_dir: Path,
    max_size: int = 64,
) -> None:
    """
    Plot soft matching matrix as a heatmap.

    Args:
        stem:        image pair stem
        soft_matrix: [N1, N2]
        out_dir:     output directory
        max_size:    downsample if matrix dimension exceeds this
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    M = soft_matrix.cpu().float().numpy()

    # Subsample for very large matrices
    if M.shape[0] > max_size or M.shape[1] > max_size:
        step_r = max(1, M.shape[0] // max_size)
        step_c = max(1, M.shape[1] // max_size)
        M = M[::step_r, ::step_c]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    im = ax.imshow(M, cmap="viridis", aspect="auto", origin="upper",
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Matching Probability", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ax.set_xlabel("T2 Token Index", color="white")
    ax.set_ylabel("T1 Token Index", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    ax.set_title(f"{stem} — Soft Matching Matrix", color="white", fontsize=12)

    out_path = out_dir / f"{stem}_soft_heatmap.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.debug(f"Saved soft heatmap → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Misc / timing
# ─────────────────────────────────────────────────────────────────────────────

class Timer:
    """Simple context-manager timer."""
    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed: float = 0.0

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._t
        if self.name:
            log.debug(f"[Timer] {self.name}: {self.elapsed:.4f}s")


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log.debug(f"Seed set to {seed}")
