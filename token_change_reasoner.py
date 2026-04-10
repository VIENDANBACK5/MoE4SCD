"""
token_change_reasoner.py
========================
Stage 4A — Baseline Token Change Reasoner

Architecture:
  TokenEncoder        → project 256→384, fuse time/pos/area embeddings
  TransformerReasoner → 4-layer self-attention encoder
  ChangePredictionHead → binary changed/unchanged logit per token
  DeltaHead           → predicted embedding delta for matched pairs

Usage (programmatic):
    from token_change_reasoner import ChangeReasonerModel, build_batch, training_step
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReasonerConfig:
    # Token encoder
    token_dim: int = 256          # SAM2 embedding dimension
    hidden_dim: int = 384         # model width
    pos_mlp_hidden: int = 64      # hidden dim of positional MLP
    area_mlp_hidden: int = 64     # hidden dim of area MLP
    dropout: float = 0.1

    # Transformer
    num_layers: int = 4
    num_heads: int = 8
    ff_dim: int = 1536            # 4 × hidden_dim

    # Loss weights
    delta_loss_weight: float = 0.2

    # Proxy label threshold  (mean+std from diagnostics: 9.56)
    # used when GT labels are not provided
    proxy_delta_threshold: float = 9.56


# ─────────────────────────────────────────────────────────────────────────────
# Helper: small MLP
# ─────────────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Token Encoder
# ─────────────────────────────────────────────────────────────────────────────

class TokenEncoder(nn.Module):
    """
    Projects raw token data into a unified hidden representation.

    Input (per token):
        embedding  : [*, 256]  raw SAM2 feature
        time_id    : [*]       long  0=T1, 1=T2
        centroid   : [*, 2]    float (x, y) in [0, 1]
        log_area   : [*]       float log(area + 1)

    Output:
        [*, hidden_dim]  — fused representation
    """

    def __init__(self, cfg: ReasonerConfig):
        super().__init__()
        H = cfg.hidden_dim
        # Project raw embedding
        self.proj = nn.Linear(cfg.token_dim, H)

        # Time embedding: 0=T1, 1=T2
        self.time_embed = nn.Embedding(2, H)

        # Positional embedding from (x, y) centroid
        self.pos_mlp = _mlp(2, cfg.pos_mlp_hidden, H, cfg.dropout)

        # Area embedding from log(area + 1) — scalar → hidden
        self.area_mlp = _mlp(1, cfg.area_mlp_hidden, H, cfg.dropout)

        self.norm = nn.LayerNorm(H)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        embedding: torch.Tensor,   # [*, D]
        time_id: torch.Tensor,     # [*]  long
        centroid: torch.Tensor,    # [*, 2]
        log_area: torch.Tensor,    # [*]
    ) -> torch.Tensor:             # [*, H]

        x = self.proj(embedding)                   # [*, H]
        x = x + self.time_embed(time_id)           # [*, H]
        x = x + self.pos_mlp(centroid)             # [*, H]
        x = x + self.area_mlp(log_area.unsqueeze(-1))  # [*, H]
        return self.drop(self.norm(x))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Transformer Reasoner
# ─────────────────────────────────────────────────────────────────────────────

class TransformerReasoner(nn.Module):
    """
    Standard Transformer encoder.

    Input:
        x    : [B, N_pad, H]
        mask : [B, N_pad]  bool — True = padding (ignored)

    Output:
        [B, N_pad, H]
    """

    def __init__(self, cfg: ReasonerConfig):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,      # Pre-LN for more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers,
            enable_nested_tensor=False,   # avoid warning with variable lengths
        )

    def forward(
        self,
        x: torch.Tensor,                        # [B, N_pad, H]
        padding_mask: Optional[torch.Tensor],   # [B, N_pad] bool
    ) -> torch.Tensor:                          # [B, N_pad, H]
        return self.encoder(x, src_key_padding_mask=padding_mask)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Prediction Heads
# ─────────────────────────────────────────────────────────────────────────────

class ChangePredictionHead(nn.Module):
    """
    Predicts a change logit for each token.

    Input : [*, H]
    Output: [*]   raw logit (apply sigmoid for probability)
    """

    def __init__(self, cfg: ReasonerConfig):
        super().__init__()
        H = cfg.hidden_dim
        self.head = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(H // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)   # [*]


class DeltaHead(nn.Module):
    """
    Predicts the embedding-delta magnitude for a matched pair.

    Input : repr_i [*, H], repr_j [*, H]
    Output: [*]  predicted ||emb_T2 - emb_T1||
    """

    def __init__(self, cfg: ReasonerConfig):
        super().__init__()
        H = cfg.hidden_dim
        self.head = nn.Sequential(
            nn.LayerNorm(2 * H),
            nn.Linear(2 * H, H),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(H, 1),
            nn.Softplus(),          # output >= 0
        )

    def forward(self, repr_i: torch.Tensor, repr_j: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([repr_i, repr_j], dim=-1)).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. End-to-End Model
# ─────────────────────────────────────────────────────────────────────────────

class ChangeReasonerModel(nn.Module):
    """
    Full Stage 4A model.

    Processes a BATCH of image pairs — each sample may have a different
    number of tokens; they are padded to the max length in the batch.

    Forward inputs (per sample, stored in a batch dict — see build_batch()):
        tokens_pad   [B, N_max, 256]  padded embeddings (T1 concat T2)
        time_ids_pad [B, N_max]       long  0=T1 1=T2
        centroids_pad[B, N_max, 2]
        log_areas_pad[B, N_max]
        padding_mask [B, N_max]       bool  True=pad position

    Forward outputs:
        change_logits [B, N_max]  (pad positions have undefined values)
        delta_pred    [M_total]   flat tensor of pair delta predictions
        delta_target  [M_total]   flat tensor of pair delta targets
    """

    def __init__(self, cfg: ReasonerConfig):
        super().__init__()
        self.cfg = cfg
        self.token_encoder = TokenEncoder(cfg)
        self.reasoner = TransformerReasoner(cfg)
        self.change_head = ChangePredictionHead(cfg)
        self.delta_head = DeltaHead(cfg)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        batch keys (all tensors already on model device):
            tokens_pad    [B, N, 256]
            time_ids_pad  [B, N]     long
            centroids_pad [B, N, 2]
            log_areas_pad [B, N]
            padding_mask  [B, N]     bool  True=ignore
            pair_b        [M]        long  — batch index for each match pair
            pair_i        [M]        long  — T1 token index within sample
            pair_j        [M]        long  — T2 token index within sample (offset already applied)
            delta_target  [M]        float — ||emb_T1_i - emb_T2_j||

        Returns dict:
            change_logits [B, N]
            delta_pred    [M]
            delta_target  [M]   (pass-through for loss computation)
        """
        B, N, _ = batch["tokens_pad"].shape

        # ── 1. Encode all tokens ───────────────────────────────────────────
        flat_emb  = batch["tokens_pad"].reshape(B * N, -1)          # [B*N, 256]
        flat_tid  = batch["time_ids_pad"].reshape(B * N)             # [B*N]
        flat_cen  = batch["centroids_pad"].reshape(B * N, 2)         # [B*N, 2]
        flat_area = batch["log_areas_pad"].reshape(B * N)            # [B*N]

        flat_repr = self.token_encoder(flat_emb, flat_tid, flat_cen, flat_area)
        repr_pad  = flat_repr.reshape(B, N, -1)                      # [B, N, H]

        # ── 2. Transformer context ─────────────────────────────────────────
        repr_ctx = self.reasoner(repr_pad, batch["padding_mask"])    # [B, N, H]

        # ── 3. Change logits ───────────────────────────────────────────────
        change_logits = self.change_head(repr_ctx)                   # [B, N]

        # ── 4. Delta predictions for matched pairs ─────────────────────────
        pair_b = batch["pair_b"]   # [M]
        pair_i = batch["pair_i"]   # [M]   index of T1 token in padded seq
        pair_j = batch["pair_j"]   # [M]   index of T2 token in padded seq

        if len(pair_b) > 0:
            repr_i = repr_ctx[pair_b, pair_i]   # [M, H]
            repr_j = repr_ctx[pair_b, pair_j]   # [M, H]
            delta_pred = self.delta_head(repr_i, repr_j)   # [M]
        else:
            delta_pred = torch.zeros(0, device=repr_ctx.device)

        return {
            "change_logits": change_logits,          # [B, N]
            "delta_pred": delta_pred,                # [M]
            "delta_target": batch["delta_target"],   # [M]
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Batch Builder
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SampleData:
    """One image-pair sample — raw (unpadded) tensors."""
    tokens_t1: torch.Tensor    # [N1, 256]
    tokens_t2: torch.Tensor    # [N2, 256]
    centroids_t1: torch.Tensor # [N1, 2]
    centroids_t2: torch.Tensor # [N2, 2]
    areas_t1: torch.Tensor     # [N1]
    areas_t2: torch.Tensor     # [N2]
    # Stage 3 match output: [[i, j, score], ...]  already as tensor [M_pairs, 3]
    match_pairs: torch.Tensor
    # Optional: per-token change labels  [N1+N2]  float 0/1
    # Pass None to use proxy labels derived from delta_norm
    change_labels: Optional[torch.Tensor] = None
    # Optional: per-token semantic labels [N1+N2] long (0-6)
    semantic_labels: Optional[torch.Tensor] = None


def _proxy_labels(tokens_t1: torch.Tensor, tokens_t2: torch.Tensor,
                  pairs: torch.Tensor, threshold: float,
                  n_total: int) -> torch.Tensor:
    """
    Generate proxy change labels from embedding delta norm.
    label[k] = 1  if token k is involved in a pair with delta_norm > threshold.
    Unmatched tokens get label = 0 (conservative).
    """
    labels = torch.zeros(n_total, dtype=torch.float32)
    n1 = len(tokens_t1)
    if len(pairs) == 0:
        return labels
    for p in pairs:
        i, j = int(p[0]), int(p[1])
        dn = (tokens_t2[j] - tokens_t1[i]).norm().item()
        lbl = 1.0 if dn > threshold else 0.0
        labels[i] = lbl               # T1 token index
        labels[n1 + j] = lbl          # T2 token index (offset by n1)
    return labels


def build_batch(
    samples: List[SampleData],
    cfg: ReasonerConfig,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Collate a list of SampleData into a padded batch dict ready for the model.

    Concatenation order within each sample:
        seq = [tok_T1_0, ..., tok_T1_{N1-1}, tok_T2_0, ..., tok_T2_{N2-1}]

    Match pair indices (pair_i, pair_j) are expressed as positions in this
    concatenated+padded sequence:
        pair_i = i          (T1 index, 0-based)
        pair_j = N1 + j     (T2 index, offset by N1)
    """
    B = len(samples)

    # ── 1. Compute sizes ───────────────────────────────────────────────────
    seq_lens = []
    for s in samples:
        n1 = len(s.tokens_t1)
        n2 = len(s.tokens_t2)
        seq_lens.append(n1 + n2)
    N_max = max(seq_lens)

    # ── 2. Allocate padded tensors ─────────────────────────────────────────
    tokens_pad    = torch.zeros(B, N_max, samples[0].tokens_t1.shape[-1])
    time_ids_pad  = torch.zeros(B, N_max, dtype=torch.long)
    centroids_pad = torch.zeros(B, N_max, 2)
    log_areas_pad = torch.zeros(B, N_max)
    padding_mask  = torch.ones(B, N_max, dtype=torch.bool)   # True = pad

    has_semantic = any(s.semantic_labels is not None for s in samples)
    if has_semantic:
        semantic_labels_pad = torch.zeros(B, N_max, dtype=torch.long)
    else:
        semantic_labels_pad = None

    # ── 3. Pair lists ──────────────────────────────────────────────────────
    pair_b_list: List[int] = []
    pair_i_list: List[int] = []
    pair_j_list: List[int] = []
    delta_target_list: List[float] = []

    # ── 4. Change label list ───────────────────────────────────────────────
    change_labels_list: List[torch.Tensor] = []

    for b, s in enumerate(samples):
        n1 = len(s.tokens_t1)
        n2 = len(s.tokens_t2)
        n  = n1 + n2

        # -- Tokens
        tokens_pad[b, :n1] = s.tokens_t1.float()
        tokens_pad[b, n1:n] = s.tokens_t2.float()

        # -- Time IDs  (0 = T1, 1 = T2)
        time_ids_pad[b, :n1]  = 0
        time_ids_pad[b, n1:n] = 1

        # -- Centroids
        centroids_pad[b, :n1]  = s.centroids_t1.float()
        centroids_pad[b, n1:n] = s.centroids_t2.float()

        # -- Log areas
        log_areas_pad[b, :n1]  = torch.log1p(s.areas_t1.float())
        log_areas_pad[b, n1:n] = torch.log1p(s.areas_t2.float())

        # -- Padding mask  (False = valid)
        padding_mask[b, :n] = False

        # -- Semantic labels
        if has_semantic and s.semantic_labels is not None:
            semantic_labels_pad[b, :n] = s.semantic_labels.long()

        # -- Match pairs
        for p in s.match_pairs:
            i, j, score = int(p[0]), int(p[1]), float(p[2])
            pair_b_list.append(b)
            pair_i_list.append(i)         # T1 position in seq
            pair_j_list.append(n1 + j)   # T2 position in seq (offset)
            delta_norm = (s.tokens_t2[j] - s.tokens_t1[i]).norm().item()
            delta_target_list.append(delta_norm)

        # -- Change labels
        if s.change_labels is not None:
            lbl = s.change_labels.float()
        else:
            lbl = _proxy_labels(
                s.tokens_t1, s.tokens_t2, s.match_pairs,
                cfg.proxy_delta_threshold, n
            )
        change_labels_list.append(lbl)

    # Pad change_labels to N_max
    change_labels_pad = torch.zeros(B, N_max)
    for b, lbl in enumerate(change_labels_list):
        n = len(lbl)
        change_labels_pad[b, :n] = lbl

    # ── 5. Assemble pair tensors ───────────────────────────────────────────
    pair_b = torch.tensor(pair_b_list, dtype=torch.long)
    pair_i = torch.tensor(pair_i_list, dtype=torch.long)
    pair_j = torch.tensor(pair_j_list, dtype=torch.long)
    delta_target = torch.tensor(delta_target_list, dtype=torch.float32)

    batch = {
        "tokens_pad":     tokens_pad.to(device),
        "time_ids_pad":   time_ids_pad.to(device),
        "centroids_pad":  centroids_pad.to(device),
        "log_areas_pad":  log_areas_pad.to(device),
        "padding_mask":   padding_mask.to(device),
        "change_labels":  change_labels_pad.to(device),
        "pair_b":         pair_b.to(device),
        "pair_i":         pair_i.to(device),
        "pair_j":         pair_j.to(device),
        "delta_target":   delta_target.to(device),
        "seq_lens":       torch.tensor(seq_lens, dtype=torch.long),
    }
    if has_semantic:
        batch["semantic_labels_pad"] = semantic_labels_pad.to(device)
    
    return batch


# ─────────────────────────────────────────────────────────────────────────────
# 6. Loss + Training Step
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: ReasonerConfig,
) -> Dict[str, torch.Tensor]:
    """
    Computes:
        change_loss  = BCE(change_logits, change_labels)  [valid tokens only]
        delta_loss   = MSE(delta_pred, delta_target)       [matched pairs]
        total_loss   = change_loss + cfg.delta_loss_weight * delta_loss

    Returns dict with keys: total_loss, change_loss, delta_loss.
    """
    change_logits  = outputs["change_logits"]           # [B, N]
    change_labels  = batch["change_labels"]             # [B, N]
    padding_mask   = batch["padding_mask"]              # [B, N] bool (True=pad)

    # ── Change loss (mask out pad positions) ───────────────────────────────
    valid = ~padding_mask                               # [B, N] bool
    logits_valid = change_logits[valid]                 # [K]
    labels_valid = change_labels[valid]                 # [K]

    # Compute class weight to handle imbalance (more unchanged than changed)
    pos_frac = labels_valid.mean().clamp(min=1e-3, max=1 - 1e-3)
    pos_weight = torch.tensor(
        [(1 - pos_frac) / pos_frac], device=change_logits.device
    )
    change_loss = F.binary_cross_entropy_with_logits(
        logits_valid, labels_valid, pos_weight=pos_weight
    )

    # ── Delta loss ─────────────────────────────────────────────────────────
    delta_pred   = outputs["delta_pred"]    # [M]
    delta_target = outputs["delta_target"]  # [M]

    if len(delta_pred) > 0:
        delta_loss = F.mse_loss(delta_pred, delta_target)
    else:
        delta_loss = torch.tensor(0.0, device=change_loss.device)

    total_loss = change_loss + cfg.delta_loss_weight * delta_loss

    return {
        "total_loss":  total_loss,
        "change_loss": change_loss,
        "delta_loss":  delta_loss,
    }


def training_step(
    model: ChangeReasonerModel,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    """
    Single training step with optional mixed precision.

    Returns dict of scalar losses for logging.

    Example:
        scaler = torch.cuda.amp.GradScaler()
        for batch in dataloader:
            log = training_step(model, batch, optimizer, scaler)
            print(log)
    """
    model.train()
    optimizer.zero_grad()
    cfg = model.cfg

    if scaler is not None:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(batch)
            losses  = compute_loss(outputs, batch, cfg)
        scaler.scale(losses["total_loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs = model(batch)
        losses  = compute_loss(outputs, batch, cfg)
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return {k: float(v.detach()) for k, v in losses.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 7. Dummy batch for testing
# ─────────────────────────────────────────────────────────────────────────────

def make_dummy_batch(
    batch_size: int = 2,
    n1: int = 10,
    n2: int = 12,
    n_pairs: int = 8,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> Tuple[List[SampleData], Dict[str, torch.Tensor], ReasonerConfig]:
    """
    Creates a small synthetic batch for unit tests and sanity checks.

    Returns:
        samples  : list of SampleData
        batch    : packed batch dict
        cfg      : ReasonerConfig used
    """
    torch.manual_seed(seed)
    cfg = ReasonerConfig()
    samples = []

    for _ in range(batch_size):
        n1_ = n1 + torch.randint(0, 3, ()).item()  # slight variation
        n2_ = n2 + torch.randint(0, 3, ()).item()

        pairs_i = torch.randperm(min(n1_, n_pairs))[:n_pairs]
        pairs_j = torch.randperm(min(n2_, n_pairs))[:n_pairs]
        scores  = torch.rand(n_pairs)
        pairs   = torch.stack([pairs_i.float(), pairs_j.float(), scores], dim=1)

        samples.append(SampleData(
            tokens_t1    = torch.randn(n1_, 256),
            tokens_t2    = torch.randn(n2_, 256),
            centroids_t1 = torch.rand(n1_, 2),
            centroids_t2 = torch.rand(n2_, 2),
            areas_t1     = torch.randint(100, 10000, (n1_,)).float(),
            areas_t2     = torch.randint(100, 10000, (n2_,)).float(),
            match_pairs  = pairs,
            change_labels= None,   # use proxy labels
        ))

    batch = build_batch(samples, cfg, device)
    return samples, batch, cfg


# ─────────────────────────────────────────────────────────────────────────────
# 8. Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: Optional[ReasonerConfig] = None) -> ChangeReasonerModel:
    """Convenience constructor."""
    if cfg is None:
        cfg = ReasonerConfig()
    return ChangeReasonerModel(cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = ReasonerConfig()
    model = build_model(cfg).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # ── Dummy forward pass ─────────────────────────────────────────────────
    samples, batch, _ = make_dummy_batch(batch_size=4, device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(batch)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"change_logits : {outputs['change_logits'].shape}")
    print(f"delta_pred    : {outputs['delta_pred'].shape}")
    print(f"delta_target  : {outputs['delta_target'].shape}")
    print(f"Forward pass  : {elapsed:.1f} ms")

    # ── Dummy training step ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    losses = training_step(model, batch, optimizer, scaler)
    print("\nTraining step losses:")
    for k, v in losses.items():
        print(f"  {k}: {v:.4f}")

    # ── Shape summary ──────────────────────────────────────────────────────
    print("\nTensor shape summary:")
    B = batch["tokens_pad"].shape[0]
    N = batch["tokens_pad"].shape[1]
    print(f"  Batch size       B = {B}")
    print(f"  Max seq length   N = {N}  (T1 + T2 tokens, padded)")
    print(f"  D_model            = {cfg.hidden_dim}")
    print(f"  M (match pairs)    = {len(batch['pair_b'])}")
