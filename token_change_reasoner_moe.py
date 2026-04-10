"""
token_change_reasoner_moe.py
============================
Stage 4C / 4D — Token Change Reasoner with Mixture-of-Experts (MoE)

Architecture:
    TokenEncoder  → [B, N, H]
    Transformer   → [B, N, H]   (global context, 4 layers)
    GraphReasoner → [B, N, H]   (local spatial, 2 × GraphSAGE)
    MoELayer      → [B, N, H]   (expert specialisation, E experts)
    ChangePredictionHead + DeltaHead

MoE design (Stage 4C — v1 router)
────────────────────────────────────────────────────────────────
• 4 experts, each:  Linear(H → expert_dim) → GELU → Linear(expert_dim → H)
• Router (v1):      Linear(H → E) → softmax
• Top-1 routing:    each token dispatched to its highest-prob expert
• Residual:         out = x + MoE(x)
• Aux losses:
    L_balance  = num_experts * Σ p_i²       (load balancing)
    L_entropy  = −mean Σ p * log(p)         (exploration)

Stage 4D additions (router_version='v2', expert_dropout_prob, use_top2)
────────────────────────────────────────────────────────────────
• Router (v2): Linear(H+2 → E)
    router_input = concat(h, log_area, delta_hint)
    log_area  : explicit object size cue for each token
    delta_hint: precomputed T1→T2 embedding delta norm (or 0 for unmatched)
    This helps the router differentiate texture / geometric / large-object tokens.

• Expert dropout (expert_dropout_prob > 0):
    During training, randomly disable one expert with probability p,
    forcing remaining experts to compensate → stronger specialization.

• Top-2 routing (use_top2=True):
    Each token is routed to its top-2 experts.
    out = w1*Expert_a(x) + w2*Expert_b(x)   (weights renormalised to sum=1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from token_change_reasoner import (
    ChangeReasonerModel,
    ReasonerConfig,
    SampleData,
    build_batch,
    build_model,
    compute_loss,
    count_parameters,
    make_dummy_batch,
    training_step,
)
from token_change_reasoner_graph import (
    GraphReasoner,
    GraphReasonerConfig,
    TokenChangeReasonerGraph,
    build_graph_model,
    ChangePredictionHead,
    DeltaHead,
    TokenEncoder,
    TransformerReasoner,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MoEConfig(GraphReasonerConfig):
    """Stage 4C/4D config: inherits Stage 4B fields and adds MoE params."""
    # Expert FFN dimensions
    moe_num_experts: int   = 4     # number of expert FFNs
    moe_expert_dim:  int   = 512   # hidden dim inside each expert (H → D_exp → H)

    # Aux loss weights
    lambda_balance:  float = 0.01  # load-balancing loss coefficient
    lambda_entropy:  float = 0.001 # entropy regularisation coefficient

    # ── Stage 4D additions ────────────────────────────────────────────────
    # Router version: 'v1' = Linear(H→E)   'v2' = Linear(H+2→E)   'v3' = Linear(H+7→E)
    router_version:       str   = "v1"
    # Expert dropout: randomly disable one expert during training
    expert_dropout_prob:  float = 0.0
    # Top-2 routing: weighted sum of two expert outputs per token
    use_top2:             bool  = False


# ─────────────────────────────────────────────────────────────────────────────
# Expert FFN
# ─────────────────────────────────────────────────────────────────────────────

class Expert(nn.Module):
    """Single expert: Linear(H→D_exp) → GELU → Linear(D_exp→H)."""

    def __init__(self, hidden_dim: int, expert_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, expert_dim),
            nn.GELU(),
            nn.Linear(expert_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# MoE Layer
# ─────────────────────────────────────────────────────────────────────────────

class MoELayer(nn.Module):
    """
    Mixture-of-Experts layer.

    Supports:
        router_version    'v1' — routes on h only (Stage 4C default)
                          'v2' — routes on concat(h, log_area, delta_hint)
                                 richer context for specialization
        use_top2          False — Top-1 (Stage 4C default)
                          True  — Top-2 weighted sum
        expert_dropout_prob 0   — no dropout (Stage 4C default)
                          0.1   — 10% chance to disable one expert/step

    Forward flow (Top-1):
        x_flat   [T, H]  = reshape(x)
        r_input  [T, ?]  = x_flat (v1) or concat(x_flat, log_area, delta) (v2)
        logits   [T, E]  = router(r_input)
        probs    [T, E]  = softmax(logits)
        idx      [T]     = argmax(probs)
        out_flat [T, H]  = dispatch → experts → collect
        out      [B,N,H] = reshape + x (residual)
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        H = cfg.hidden_dim
        E = cfg.moe_num_experts

        self.num_experts        = E
        self.expert_dropout_prob= cfg.expert_dropout_prob
        self.use_top2           = cfg.use_top2
        self.router_version     = cfg.router_version

        self.experts = nn.ModuleList([
            Expert(H, cfg.moe_expert_dim) for _ in range(E)
        ])

        # Router input dim: H for v1, H+2 for v2, H+7 for v3 (semantic one-hot)
        if cfg.router_version == "v2":
            router_in = H + 2
        elif cfg.router_version == "v3":
            router_in = H + 7
        else:
            router_in = H
            
        self.router = nn.Linear(router_in, E)

    def forward(
        self,
        x: torch.Tensor,             # [B, N, H]
        log_areas: Optional[torch.Tensor] = None,   # [B, N]
        delta_hints: Optional[torch.Tensor] = None, # [B, N]
        semantic_labels: Optional[torch.Tensor] = None, # [B, N] long
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            out              : [B, N, H]  — x + routed expert output (residual)
            balance_loss     : scalar
            entropy_loss     : scalar
            tokens_per_expert: [E] long
        """
        B, N, H = x.shape
        E = self.num_experts

        # ── 1. Flatten ──────────────────────────────────────────────────────
        x_flat = x.reshape(B * N, H)        # [T, H]

        # ── 2. Build router input ───────────────────────────────────────────
        if self.router_version == "v2":
            if log_areas is not None:
                la = log_areas.reshape(B * N, 1)          # [T, 1]
            else:
                la = torch.zeros(B * N, 1, device=x.device, dtype=x.dtype)

            if delta_hints is not None:
                dh = delta_hints.reshape(B * N, 1)        # [T, 1]
            else:
                dh = torch.zeros(B * N, 1, device=x.device, dtype=x.dtype)

            router_input = torch.cat([x_flat, la, dh], dim=-1)  # [T, H+2]
            
        elif self.router_version == "v3":
            if semantic_labels is not None:
                # Expect semantic_labels in [B, N] mapped 0-6. Flat to [T]
                sl = semantic_labels.reshape(B * N).long()
                sl_onehot = F.one_hot(sl, num_classes=7).to(x.dtype)  # [T, 7]
            else:
                sl_onehot = torch.zeros(B * N, 7, device=x.device, dtype=x.dtype)
                
            router_input = torch.cat([x_flat, sl_onehot], dim=-1) # [T, H+7]
            
        else:
            router_input = x_flat                          # [T, H]

        # ── 3. Router: compute per-token expert probabilities ───────────────
        router_logits = self.router(router_input)          # [T, E]
        router_probs  = torch.softmax(router_logits, dim=-1)  # [T, E]

        # ── 4. Expert dropout (training only) ───────────────────────────────
        if self.training and self.expert_dropout_prob > 0.0 and E > 1:
            if torch.rand(1).item() < self.expert_dropout_prob:
                # Randomly disable one expert: zero its probability column
                drop_e = torch.randint(0, E, (1,)).item()
                router_probs = router_probs.clone()
                router_probs[:, drop_e] = 0.0
                # Re-normalise
                router_probs = router_probs / router_probs.sum(-1, keepdim=True).clamp(min=1e-8)

        # ── 5. Expert selection ──────────────────────────────────────────────
        if self.use_top2 and E >= 2:
            top2_vals, top2_idx = router_probs.topk(2, dim=-1)   # [T, 2]
            # Renormalise top-2 weights to sum=1
            top2_w = top2_vals / top2_vals.sum(-1, keepdim=True).clamp(min=1e-8)
        else:
            expert_idx = router_probs.argmax(dim=-1)              # [T]

        # ── 6. Dispatch → experts → collect ─────────────────────────────────
        # out_flat: float32 (same dtype as x).  Cast expert outs back for AMP.
        out_flat = torch.zeros(B * N, H, dtype=x.dtype, device=x.device)
        tokens_per_expert = torch.zeros(E, dtype=torch.long, device=x.device)

        if self.use_top2 and E >= 2:
            # Each token contributes to 2 experts, weighted
            for k in range(2):
                idx_k = top2_idx[:, k]     # [T]
                w_k   = top2_w[:, k]       # [T]
                for e in range(E):
                    mask = (idx_k == e)
                    n_e  = mask.sum()
                    if k == 0:
                        tokens_per_expert[e] += n_e
                    if n_e > 0:
                        eout = self.experts[e](x_flat[mask]).to(x.dtype)
                        out_flat[mask] = out_flat[mask] + w_k[mask].unsqueeze(1) * eout
        else:
            for e in range(E):
                mask = (expert_idx == e)
                n_e  = mask.sum()
                tokens_per_expert[e] = n_e
                if n_e > 0:
                    expert_out       = self.experts[e](x_flat[mask])
                    out_flat[mask]   = expert_out.to(x.dtype)

        # ── 7. Residual ──────────────────────────────────────────────────────
        out = x + out_flat.reshape(B, N, H)

        # ── 8. Aux loss 1: load balancing ────────────────────────────────────
        # p_i = mean fraction of tokens routed to expert i
        # L_balance = E * Σ p_i²   (=1.0 when perfectly uniform)
        p_i          = router_probs.mean(dim=0)             # [E]
        balance_loss = (E * (p_i ** 2).sum()).unsqueeze(0).squeeze()

        # ── 9. Aux loss 2: entropy regularisation ────────────────────────────
        eps          = 1e-8
        entropy      = -(router_probs * (router_probs + eps).log()).sum(dim=-1)  # [T]
        entropy_loss = entropy.mean()

        return out, balance_loss, entropy_loss, tokens_per_expert


# ─────────────────────────────────────────────────────────────────────────────
# Full Model — Stage 4C / 4D
# ─────────────────────────────────────────────────────────────────────────────

class TokenChangeReasonerMoE(nn.Module):
    """
    Stage 4C / 4D model.

    Forward flow:
        1. TokenEncoder        [B*N, 256] → [B*N, H]
        2. TransformerReasoner [B, N, H]  → [B, N, H]   (global context)
        3. GraphReasoner       [B, N, H]  → [B, N, H]   (local spatial)
        4. MoELayer            [B, N, H]  → [B, N, H]   (expert specialisation)
        5. ChangePredictionHead                → [B, N]
        6. DeltaHead                           → [M]

    Stage 4D:
        With router_version='v2', log_areas_pad and a delta_hint are passed
        to MoELayer so the router can condition on object size and change signal.
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg           = cfg
        self.token_encoder = TokenEncoder(cfg)
        self.reasoner      = TransformerReasoner(cfg)
        self.graph         = GraphReasoner(cfg)
        self.moe           = MoELayer(cfg)
        self.change_head   = ChangePredictionHead(cfg)
        self.delta_head    = DeltaHead(cfg)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        batch keys (from build_batch):
            tokens_pad    : [B, N, 256]
            time_ids_pad  : [B, N]         long  0=T1, 1=T2
            centroids_pad : [B, N, 2]
            log_areas_pad : [B, N]
            padding_mask  : [B, N]         bool  True=pad
            pair_b, pair_i, pair_j : [M]   long
            delta_target  : [M]            float

        Extra outputs:
            balance_loss      : scalar
            entropy_loss      : scalar
            tokens_per_expert : [E] long
        """
        B, N, _ = batch["tokens_pad"].shape

        # ── 1. Encode tokens ────────────────────────────────────────────────
        flat_emb  = batch["tokens_pad"].reshape(B * N, -1)
        flat_tid  = batch["time_ids_pad"].reshape(B * N)
        flat_cen  = batch["centroids_pad"].reshape(B * N, 2)
        flat_area = batch["log_areas_pad"].reshape(B * N)

        flat_repr = self.token_encoder(flat_emb, flat_tid, flat_cen, flat_area)
        repr_pad  = flat_repr.reshape(B, N, -1)

        # ── 2. Transformer ──────────────────────────────────────────────────
        repr_ctx = self.reasoner(repr_pad, batch["padding_mask"])

        # ── 3. Graph ────────────────────────────────────────────────────────
        repr_ctx = self.graph(
            repr_ctx,
            batch["padding_mask"],
            batch["centroids_pad"],
            batch["time_ids_pad"],
        )

        # ── 4. Precompute delta_hint for v2 router ──────────────────────────
        # delta_hint[b, n] = L2 distance between T1 and T2 representations
        # for matching token pairs; 0 for unmatched tokens.
        delta_hint = None
        if self.cfg.router_version == "v2":
            delta_hint = torch.zeros(B, N, device=repr_ctx.device, dtype=repr_ctx.dtype)
            pair_b = batch["pair_b"]
            pair_i = batch["pair_i"]
            pair_j = batch["pair_j"]
            if len(pair_b) > 0:
                diff = repr_ctx[pair_b, pair_i] - repr_ctx[pair_b, pair_j]
                dist = diff.norm(dim=-1)         # [M]
                delta_hint.index_put_(
                    (pair_b, pair_i), dist, accumulate=False
                )

        # ── 5. MoE ─────────────────────────────────────────────────────────
        repr_ctx, balance_loss, entropy_loss, tokens_per_expert = self.moe(
            repr_ctx,
            log_areas   = batch["log_areas_pad"] if self.cfg.router_version == "v2" else None,
            delta_hints = delta_hint,
            semantic_labels = batch.get("semantic_labels_pad") if self.cfg.router_version == "v3" else None,
        )

        # ── 6. Change logits ────────────────────────────────────────────────
        change_logits = self.change_head(repr_ctx)

        # ── 7. Delta predictions ────────────────────────────────────────────
        pair_b = batch["pair_b"]
        pair_i = batch["pair_i"]
        pair_j = batch["pair_j"]

        if len(pair_b) > 0:
            repr_i = repr_ctx[pair_b, pair_i]
            repr_j = repr_ctx[pair_b, pair_j]
            delta_pred = self.delta_head(repr_i, repr_j)
        else:
            delta_pred = torch.zeros(0, device=repr_ctx.device)

        return {
            "change_logits":     change_logits,
            "delta_pred":        delta_pred,
            "delta_target":      batch["delta_target"],
            "balance_loss":      balance_loss,
            "entropy_loss":      entropy_loss,
            "tokens_per_expert": tokens_per_expert,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Loss function (extends compute_loss with MoE aux terms)
# ─────────────────────────────────────────────────────────────────────────────

def compute_moe_loss(
    outputs: Dict[str, torch.Tensor],
    batch:   Dict[str, torch.Tensor],
    cfg:     MoEConfig,
) -> Dict[str, torch.Tensor]:
    """
    Total loss:
        L_total = L_change + λ_delta * L_delta
                + λ_balance * L_balance
                + λ_entropy * L_entropy

    Returns dict with keys:
        total_loss, change_loss, delta_loss, balance_loss, entropy_loss
    """
    base = compute_loss(outputs, batch, cfg)

    balance_loss = outputs["balance_loss"]
    entropy_loss = outputs["entropy_loss"]

    total = (base["total_loss"]
             + cfg.lambda_balance * balance_loss
             + cfg.lambda_entropy * entropy_loss)

    return {
        "total_loss":   total,
        "change_loss":  base["change_loss"],
        "delta_loss":   base["delta_loss"],
        "balance_loss": balance_loss,
        "entropy_loss": entropy_loss,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_moe_model(cfg: Optional[MoEConfig] = None) -> TokenChangeReasonerMoE:
    if cfg is None:
        cfg = MoEConfig()
    return TokenChangeReasonerMoE(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    def _run_smoke(label, cfg):
        model  = build_moe_model(cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = model.to(device)
        _, batch, _ = make_dummy_batch(batch_size=2, n1=20, n2=25,
                                        n_pairs=8, device=device, seed=42)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        model.train()
        losses = compute_moe_loss(model(batch), batch, cfg)
        losses["total_loss"].backward()
        tpe = out["tokens_per_expert"].tolist()
        print(f"  {label}: loss={float(losses['total_loss'].detach()):.3f}  "
              f"bal={float(out['balance_loss']):.3f}  "
              f"experts={tpe}  OK ✓")

    print("=" * 60)
    print("Stage 4C/4D — MoE Smoke Tests")
    print("=" * 60)

    base_cfg = dict(hidden_dim=32, num_layers=2, num_heads=4, ff_dim=64,
                    graph_k=4, graph_layers=2,
                    moe_num_experts=4, moe_expert_dim=64)

    _run_smoke("v1 / Top-1",           MoEConfig(**base_cfg, router_version="v1"))
    _run_smoke("v2 / Top-1 / dropout", MoEConfig(**base_cfg, router_version="v2",
                                                  expert_dropout_prob=0.1))
    _run_smoke("v1 / Top-2",           MoEConfig(**base_cfg, use_top2=True))
    _run_smoke("v2 / Top-2 / dropout", MoEConfig(**base_cfg, router_version="v2",
                                                  use_top2=True, expert_dropout_prob=0.1))

    print("=" * 60)
    print("All smoke tests PASSED ✅")
