"""
token_change_reasoner_graph.py
==============================
Stage 4B — Token Change Reasoner with Spatial Graph Reasoning.

Extends Stage 4A by adding GraphSAGE layers after the Transformer.
The graph is built per-sample using token centroid coordinates (k-NN).

Architecture:
    tokens [N, 256]
        ↓ TokenEncoder
    [N, H]  (H = hidden_dim = 384)
        ↓ TransformerReasoner  (global context)
    [N, H]
        ↓ GraphReasoner        (local spatial context, 2 × GraphSAGE)
    [N, H]
        ↓ ChangePredictionHead / DeltaHead
    change_logits [N], delta_pred [M]

Tensor notation:
    B  = batch size
    N  = max total tokens per pair (N1 + N2, padded)
    H  = hidden_dim  (384)
    k  = k-NN neighbours  (6)
    M  = number of matched pairs across the batch

Usage (same CLI as Stage 4A):
    python train_reasoner.py \\
        --tokens_T1 SECOND/tokens_T1 \\
        --tokens_T2 SECOND/tokens_T2 \\
        --matches   SECOND/matches   \\
        --output    SECOND/stage4B   \\
        --epochs 30 --batch_size 8   \\
        --device cuda --model_type graph
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-use everything from Stage 4A that doesn't change
from token_change_reasoner import (
    ChangePredictionHead,
    DeltaHead,
    ReasonerConfig,
    SampleData,
    TokenEncoder,
    TransformerReasoner,
    build_batch,
    compute_loss,
    count_parameters,
    make_dummy_batch,
    training_step,
    _proxy_labels,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config extension
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphReasonerConfig(ReasonerConfig):
    """Stage 4B config: inherits all Stage 4A fields and adds graph params."""
    graph_k: int          = 6     # k-nearest neighbours
    graph_layers: int     = 2     # number of GraphSAGE layers stacked
    graph_dropout: float  = 0.2   # edge dropout prob during training
    graph_residual: bool  = True  # h = h + GraphSAGE(h)
    # Hybrid edge weight coefficients
    alpha_spatial: float  = 0.6   # weight of spatial (inverse-distance) term
    beta_semantic: float  = 0.4   # weight of cosine-similarity term
    # Cross-time edge scaling
    gamma_cross: float    = 1.2   # edge weight multiplier for T1↔T2 edges



# ─────────────────────────────────────────────────────────────────────────────
# GraphBuilder — builds BATCHED k-NN tensors (fully vectorized)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_batch_graph(
    centroids_pad: torch.Tensor,             # [B, N, 2]
    padding_mask:  torch.Tensor,             # [B, N]  bool True=pad
    k:             int,
    h:             Optional[torch.Tensor] = None,  # [B, N, H] Transformer repr
    time_ids_pad:  Optional[torch.Tensor] = None,  # [B, N]   long  0=T1, 1=T2
    alpha:         float = 0.6,   # spatial weight coefficient
    beta:          float = 0.4,   # semantic weight coefficient
    gamma_cross:   float = 1.2,   # cross-time edge multiplier
    graph_dropout: float = 0.0,   # edge drop probability (0 = off)
    training:      bool  = False, # apply dropout only during training
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch-vectorized hybrid k-NN graph construction.

    Pipeline:
        1. Spatial L2 distances  →  top-k neighbours
        2. Hybrid weights: α · inv-dist + β · cosine-sim  (if h provided)
        3. Cross-time scaling: γ_cross for T1↔T2 edges
        4. Graph dropout (training only)

    Tensor shapes:
        centroids_pad : [B, N, 2]
        padding_mask  : [B, N]       True = pad
        h             : [B, N, H]    optional; used for semantic similarity
        time_ids_pad  : [B, N]       optional; 0=T1, 1=T2
        — returns —
        nbr_idx : [B, N, k]  LongTensor  — neighbour global indices
        nbr_w   : [B, N, k]  FloatTensor — normalised hybrid weights
    """
    B, N, _ = centroids_pad.shape
    device  = centroids_pad.device
    k       = min(k, N - 1)

    # ── 1. Pairwise L2 distances [B, N, N] ─────────────────────────────────
    dist = torch.cdist(centroids_pad.float(), centroids_pad.float(), p=2)

    INF = 1e9
    eye     = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
    pad_col = padding_mask.unsqueeze(1).expand(B, N, N)
    dist = dist.masked_fill(eye,     INF)
    dist = dist.masked_fill(pad_col, INF)

    # ── 2. Top-k by spatial distance ──────────────────────────────────────
    topk_dist, nbr_idx = dist.topk(k, dim=2, largest=False)  # [B,N,k]
    valid_nbr = topk_dist < (INF / 2)                         # [B,N,k] bool

    # ── 3. Spatial weights (normalised inverse distance) ──────────────────
    inv_d = (1.0 / topk_dist.clamp(min=1e-4)) * valid_nbr.float()  # [B,N,k]
    inv_d_sum = inv_d.sum(dim=2, keepdim=True).clamp(min=1e-8)
    w_spatial = inv_d / inv_d_sum                                    # [B,N,k]

    # ── 4. Semantic weights (cosine similarity of Transformer embeddings) ───
    if h is not None and beta > 0.0:
        H = h.shape[-1]
        h_norm = F.normalize(h.float(), dim=-1)                # [B, N, H]

        # Gather normalised neighbour embeddings via flat index
        flat_idx     = nbr_idx.reshape(B, N * k)               # [B, N*k]
        flat_idx_exp = flat_idx.unsqueeze(-1).expand(B, N*k, H) # [B, N*k, H]
        h_nbr        = torch.gather(h_norm, 1, flat_idx_exp)   # [B, N*k, H]
        h_nbr        = h_nbr.reshape(B, N, k, H)               # [B, N, k, H]

        # Cosine similarity: dot product (h already l2-normalised)
        h_src = h_norm.unsqueeze(2)                            # [B, N, 1, H]
        sim   = (h_src * h_nbr).sum(-1)                       # [B, N, k]
        sim   = sim.clamp(min=0.0) * valid_nbr.float()        # relu + validity

        sim_sum  = sim.sum(dim=2, keepdim=True).clamp(min=1e-8)
        w_sem    = sim / sim_sum                               # [B, N, k]

        nbr_w = alpha * w_spatial + beta * w_sem              # [B, N, k]
    else:
        nbr_w = w_spatial                                     # [B, N, k]

    # ── 5. Cross-time edge scaling ─────────────────────────────────────
    # T1↔T2 edges multiplied by gamma_cross
    if time_ids_pad is not None and gamma_cross != 1.0:
        flat_idx = nbr_idx.reshape(B, N * k)                  # [B, N*k]
        t_nbr  = time_ids_pad.gather(1, flat_idx)             # [B, N*k]
        t_nbr  = t_nbr.reshape(B, N, k)                       # [B, N, k]
        t_src  = time_ids_pad.unsqueeze(2).expand(B, N, k)    # [B, N, k]
        is_cross = (t_src != t_nbr) & valid_nbr               # [B, N, k]
        gamma    = torch.where(is_cross,
                               torch.full_like(nbr_w, gamma_cross),
                               torch.ones_like(nbr_w))         # [B, N, k]
        nbr_w = nbr_w * gamma

    # ── 6. Re-normalise after cross-time scaling ───────────────────────
    w_sum = nbr_w.sum(dim=2, keepdim=True).clamp(min=1e-8)
    nbr_w = nbr_w / w_sum                                     # [B, N, k]

    # ── 7. Graph dropout (training only) ─────────────────────────────
    if training and graph_dropout > 0.0:
        keep  = torch.rand(B, N, k, device=device) >= graph_dropout
        keep  = keep | ~valid_nbr                              # never drop invalid edges
        nbr_w = nbr_w * keep.float()
        # Re-normalise; nodes with all edges dropped get zero output (safe)
        w_sum = nbr_w.sum(dim=2, keepdim=True).clamp(min=1e-8)
        has_keep = keep.any(dim=2, keepdim=True).float()
        nbr_w = (nbr_w / w_sum) * has_keep                   # [B, N, k]

    # ── 8. Zero out pad query positions ───────────────────────────────
    pad_q   = padding_mask.unsqueeze(2)                       # [B, N, 1]
    nbr_idx = nbr_idx.masked_fill(pad_q, 0)
    nbr_w   = nbr_w.masked_fill(pad_q, 0.0)

    return nbr_idx, nbr_w                                     # [B,N,k], [B,N,k]



# ─────────────────────────────────────────────────────────────────────────────
# GraphBuilder (kept for tests that use the per-sample API)
# ─────────────────────────────────────────────────────────────────────────────

class GraphBuilder:
    """
    Per-sample k-NN graph builder used by unit tests.
    The training path uses build_batch_graph() instead.
    """

    def __init__(self, k: int = 6):
        self.k = k

    @torch.no_grad()
    def build(
        self,
        centroids: torch.Tensor,   # [N, 2]  CPU
        valid_mask: torch.Tensor,  # [N] bool (True = valid)
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        N        = centroids.shape[0]
        valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(1)
        n_valid   = valid_idx.shape[0]

        adj          = [torch.zeros(0, dtype=torch.long)  for _ in range(N)]
        edge_weights = [torch.zeros(0, dtype=torch.float) for _ in range(N)]

        if n_valid < 2:
            return adj, edge_weights

        cen_valid = centroids[valid_idx]                          # [n, 2]
        diff      = cen_valid.unsqueeze(0) - cen_valid.unsqueeze(1)  # [n,n,2]
        dist      = diff.pow(2).sum(-1).sqrt()                    # [n,n]
        dist.fill_diagonal_(float("inf"))

        k = min(self.k, n_valid - 1)
        topk_dist, topk_local = dist.topk(k, dim=1, largest=False)  # [n,k]

        for li, gi in enumerate(valid_idx.tolist()):
            nbr_global = valid_idx[topk_local[li]]
            d   = topk_dist[li].clamp(min=1e-4)
            inv = 1.0 / d
            w   = inv / inv.sum()
            adj[gi]          = nbr_global
            edge_weights[gi] = w

        return adj, edge_weights


# ─────────────────────────────────────────────────────────────────────────────
# GraphSAGELayer — vectorized gather-based aggregation
# ─────────────────────────────────────────────────────────────────────────────

class GraphSAGELayer(nn.Module):
    """
    Batched GraphSAGE layer using torch.gather.

    Inputs:
        h        : [B, N, H]
        nbr_idx  : [B, N, k]   neighbour indices from build_batch_graph
        nbr_w    : [B, N, k]   inverse-distance weights

    Output: [B, N, H]

    Also accepts the old per-sample (adj, edge_weights) API for tests.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W_self  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_neigh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm    = nn.LayerNorm(hidden_dim)
        self.drop    = nn.Dropout(dropout)
        self.act     = nn.GELU()

    # ── Batched path (training) ────────────────────────────────────────────
    def forward_batched(
        self,
        h: torch.Tensor,        # [B, N, H]
        nbr_idx: torch.Tensor,  # [B, N, k]  long
        nbr_w: torch.Tensor,    # [B, N, k]  float
    ) -> torch.Tensor:          # [B, N, H]
        B, N, H = h.shape
        k        = nbr_idx.shape[2]

        # Gather neighbour embeddings: [B, N, k, H]
        idx_exp  = nbr_idx.unsqueeze(-1).expand(B, N, k, H)  # [B,N,k,H]
        h_exp    = h.unsqueeze(2).expand(B, N, N, H)         # [B,N,N,H] — expensive
        # Use 2-D gather instead to avoid [B,N,N,H] allocation
        # h_exp along dim=1: gather dim 1 using nbr_idx
        # nbr_idx: [B, N, k] — for each (b,i,j) pick h[b, nbr_idx[b,i,j]]
        #   → reshape to [B, N*k], index, then reshape back
        flat_idx  = nbr_idx.reshape(B, N * k)                      # [B, N*k]
        flat_idx_exp = flat_idx.unsqueeze(-1).expand(B, N * k, H)  # [B, N*k, H]
        h_flat    = torch.gather(h, 1, flat_idx_exp)               # [B, N*k, H]
        h_nbr     = h_flat.reshape(B, N, k, H)                     # [B, N, k, H]

        # Weighted mean: [B, N, H]
        h_agg = (nbr_w.unsqueeze(-1) * h_nbr).sum(dim=2)          # [B, N, H]

        out = self.act(self.W_self(h) + self.W_neigh(h_agg))
        return self.norm(self.drop(out))

    # ── Per-sample path (unit tests) ─────────────────────────────────────
    def forward(
        self,
        h: torch.Tensor,
        adj=None,
        edge_weights=None,
        nbr_idx: Optional[torch.Tensor] = None,
        nbr_w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if nbr_idx is not None:
            return self.forward_batched(h, nbr_idx, nbr_w)

        # Legacy per-sample API (list-of-tensors) — used only by tests
        if h.dim() == 2:
            N, H = h.shape
            h_agg = torch.zeros_like(h)
            for i in range(N):
                nbrs = adj[i]
                if len(nbrs) == 0:
                    continue
                w = edge_weights[i].to(h.device).unsqueeze(1)
                h_agg[i] = (w * h[nbrs]).sum(0)
            out = self.act(self.W_self(h) + self.W_neigh(h_agg))
            return self.norm(self.drop(out))

        raise ValueError("Provide either (adj/edge_weights) or (nbr_idx/nbr_w).")


# ─────────────────────────────────────────────────────────────────────────────
# GraphReasoner — vectorized batched forward
# ─────────────────────────────────────────────────────────────────────────────

class GraphReasoner(nn.Module):
    """
    Two stacked GraphSAGE layers — fully batched, runs on GPU.

    Graph is built once per batch using build_batch_graph (pure tensor ops).
    No Python-level loop over nodes or samples during forward.

    Residual:  h = h + GraphSAGE(h)

    Shapes:
        h_in  : [B, N, H]
        h_out : [B, N, H]
    """

    def __init__(self, cfg: GraphReasonerConfig):
        super().__init__()
        self.cfg = cfg
        self.k   = cfg.graph_k
        self.layers = nn.ModuleList([
            GraphSAGELayer(cfg.hidden_dim, cfg.graph_dropout)
            for _ in range(cfg.graph_layers)
        ])

    def forward(
        self,
        repr_pad: torch.Tensor,       # [B, N, H]
        padding_mask: torch.Tensor,   # [B, N]  True=pad
        centroids_pad: torch.Tensor,  # [B, N, 2]
        time_ids_pad:  Optional[torch.Tensor] = None,  # [B, N] 0=T1, 1=T2
    ) -> torch.Tensor:                # [B, N, H]

        # Build hybrid graph for the whole batch (no Python loop)
        # Pass repr_pad as h so semantic similarity uses Transformer embeddings.
        nbr_idx, nbr_w = build_batch_graph(
            centroids_pad, padding_mask, self.k,
            h            = repr_pad,
            time_ids_pad = time_ids_pad,
            alpha        = self.cfg.alpha_spatial,
            beta         = self.cfg.beta_semantic,
            gamma_cross  = self.cfg.gamma_cross,
            graph_dropout= self.cfg.graph_dropout,
            training     = self.training,
        )   # [B,N,k], [B,N,k]

        out = repr_pad
        for layer in self.layers:
            layer_out = layer.forward_batched(out, nbr_idx, nbr_w)
            if self.cfg.graph_residual:
                out = out + layer_out
            else:
                out = layer_out

        # Zero out padded positions (keep representational integrity)
        out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return out



# ─────────────────────────────────────────────────────────────────────────────
# Full Model — Stage 4B
# ─────────────────────────────────────────────────────────────────────────────

class TokenChangeReasonerGraph(nn.Module):
    """
    Stage 4B model.

    Forward flow:
        1. TokenEncoder:       [B*N, 256] → [B*N, H]
        2. TransformerReasoner: [B, N, H] → [B, N, H]  (global context)
        3. GraphReasoner:       [B, N, H] → [B, N, H]  (local spatial)
        4. ChangePredictionHead: [B, N, H] → [B, N]
        5. DeltaHead:  [M, H] × [M, H] → [M]
    """

    def __init__(self, cfg: GraphReasonerConfig):
        super().__init__()
        self.cfg           = cfg
        self.token_encoder = TokenEncoder(cfg)
        self.reasoner      = TransformerReasoner(cfg)
        self.graph         = GraphReasoner(cfg)
        self.change_head   = ChangePredictionHead(cfg)
        self.delta_head    = DeltaHead(cfg)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        batch keys (from build_batch):
            tokens_pad    : [B, N, 256]
            time_ids_pad  : [B, N]
            centroids_pad : [B, N, 2]
            log_areas_pad : [B, N]
            padding_mask  : [B, N]   True=pad
            pair_b, pair_i, pair_j   : [M]
            delta_target  : [M]
        """
        B, N, _ = batch["tokens_pad"].shape

        # ── 1. Encode ──────────────────────────────────────────────────────
        flat_emb  = batch["tokens_pad"].reshape(B * N, -1)      # [B*N, 256]
        flat_tid  = batch["time_ids_pad"].reshape(B * N)         # [B*N]
        flat_cen  = batch["centroids_pad"].reshape(B * N, 2)     # [B*N, 2]
        flat_area = batch["log_areas_pad"].reshape(B * N)        # [B*N]

        flat_repr = self.token_encoder(flat_emb, flat_tid, flat_cen, flat_area)
        repr_pad  = flat_repr.reshape(B, N, -1)                  # [B, N, H]

        # ── 2. Transformer (global context) ───────────────────────────────
        repr_ctx = self.reasoner(repr_pad, batch["padding_mask"])   # [B, N, H]

        # ── 3. Graph (hybrid spatial + semantic context) ───────────────────────
        repr_ctx = self.graph(
            repr_ctx,
            batch["padding_mask"],
            batch["centroids_pad"],
            batch["time_ids_pad"],   # enables cross-time edge scaling
        )   # [B, N, H]


        # ── 4. Change logits ───────────────────────────────────────────────
        change_logits = self.change_head(repr_ctx)   # [B, N]

        # ── 5. Delta predictions ───────────────────────────────────────────
        pair_b = batch["pair_b"]
        pair_i = batch["pair_i"]
        pair_j = batch["pair_j"]

        if len(pair_b) > 0:
            repr_i = repr_ctx[pair_b, pair_i]   # [M, H]
            repr_j = repr_ctx[pair_b, pair_j]   # [M, H]
            delta_pred = self.delta_head(repr_i, repr_j)   # [M]
        else:
            delta_pred = torch.zeros(0, device=repr_ctx.device)

        return {
            "change_logits": change_logits,
            "delta_pred":    delta_pred,
            "delta_target":  batch["delta_target"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_graph_model(cfg: Optional[GraphReasonerConfig] = None) -> TokenChangeReasonerGraph:
    if cfg is None:
        cfg = GraphReasonerConfig()
    return TokenChangeReasonerGraph(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("Stage 4B — Graph Reasoner Smoke Test")
    print("=" * 60)

    cfg   = GraphReasonerConfig(hidden_dim=64, num_layers=2, num_heads=4,
                                 ff_dim=128, graph_k=4, graph_layers=2)
    model = build_graph_model(cfg)
    print(f"Parameters: {count_parameters(model):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    _, batch, _ = make_dummy_batch(batch_size=2, n1=20, n2=25,
                                    n_pairs=10, device=device, seed=42)

    # Forward
    t0  = time.perf_counter()
    out = model(batch)
    t1  = time.perf_counter()

    B  = batch["tokens_pad"].shape[0]
    N  = batch["tokens_pad"].shape[1]
    M  = len(batch["pair_b"])
    print(f"change_logits : {tuple(out['change_logits'].shape)}  expected ({B},{N})")
    print(f"delta_pred    : {tuple(out['delta_pred'].shape)}  expected ({M},)")
    print(f"Forward time  : {(t1-t0)*1000:.1f} ms")

    # Loss + backward
    losses = compute_loss(out, batch, cfg)
    losses["total_loss"].backward()
    print(f"Total loss    : {float(losses['total_loss'].detach()):.4f}")
    print(f"Change loss   : {float(losses['change_loss'].detach()):.4f}")
    print(f"Delta  loss   : {float(losses['delta_loss'].detach()):.4f}")
    print("Backward      : OK ✅")

    # Training step
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    step_losses = training_step(model, batch, opt, scaler=None)
    print(f"Training step : {step_losses}")
    print("=" * 60)
    print("Smoke test PASSED ✅")
