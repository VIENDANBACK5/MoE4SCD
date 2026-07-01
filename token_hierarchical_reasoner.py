"""
token_hierarchical_reasoner.py
==============================
Stage 4C — Hierarchical Token Change Reasoner (MOB-GCN 2025).

Integrates Multiresolution Graph Networks (MGN) and Gumbel-Softmax pooling.
Adds Smoothness Regularization to the loss function.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from token_change_reasoner import (
    TokenEncoder, TransformerReasoner, ChangePredictionHead, DeltaHead,
    SampleData, build_batch, ReasonerConfig
)
from token_change_reasoner_graph import (
    GraphReasonerConfig, build_batch_graph, GraphSAGELayer
)
from mob_gcn_modules import GumbelSoftmaxPool, compute_smoothness_loss

@dataclass
class HierarchicalConfig(GraphReasonerConfig):
    num_clusters: int = 20
    smoothness_weight: float = 0.1
    multiscale_fuse: bool = True

class HierarchicalGraphReasoner(nn.Module):
    """
    Multiresolution Graph Reasoner implementing MOB-GCN concepts.
    1. Local GraphSAGE layers.
    2. Gumbel-Softmax pooling to clusters.
    3. Global cluster interaction.
    4. Feedback / Broadcast to local nodes.
    """
    def __init__(self, cfg: HierarchicalConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_dim
        
        # Local GNN layers
        self.local_layers = nn.ModuleList([
            GraphSAGELayer(H, cfg.graph_dropout)
            for _ in range(cfg.graph_layers)
        ])
        
        # Hierarchical Pooling
        self.pool = GumbelSoftmaxPool(H, cfg.num_clusters)
        
        # Global Cluster Reasoner (Simplified Transformer or MLP)
        self.cluster_reasoner = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.LayerNorm(H),
            nn.Linear(H, H)
        )
        
        # Multiscale Fusion
        if cfg.multiscale_fuse:
            self.fuse = nn.Sequential(
                nn.Linear(2 * H, H),
                nn.LayerNorm(H),
                nn.GELU()
            )

    def forward(
        self,
        h: torch.Tensor,
        padding_mask: torch.Tensor,
        centroids: torch.Tensor,
        time_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        h: [B, N, H] (from Transformer)
        Returns:
            h_fused: [B, N, H]
            nbr_idx, nbr_w: (for smoothness loss)
        """
        B, N, H = h.shape
        
        # 1. Build Local Graph
        nbr_idx, nbr_w = build_batch_graph(
            centroids, padding_mask, self.cfg.graph_k,
            h=h, time_ids_pad=time_ids,
            alpha=self.cfg.alpha_spatial,
            beta=self.cfg.beta_semantic,
            gamma_cross=self.cfg.gamma_cross,
            graph_dropout=self.cfg.graph_dropout,
            training=self.training
        )

        # 2. Local Reasoning
        h_local = h
        for layer in self.local_layers:
            h_local = h_local + layer.forward_batched(h_local, nbr_idx, nbr_w)
            
        # 3. Hierarchical Pooling (MOB-GCN MiddlePool)
        h_cluster, s = self.pool(h_local, padding_mask) # h_cluster: [B, C, H], s: [B, N, C]
        
        # 4. Global reasoning among clusters
        h_global = h_cluster + self.cluster_reasoner(h_cluster) # [B, C, H]
        
        # 5. Broadcast back to local nodes
        h_context = torch.matmul(s, h_global) # [B, N, H]
        
        # 6. Multiscale Fusion
        if self.cfg.multiscale_fuse:
            h_fused = self.fuse(torch.cat([h_local, h_context], dim=-1))
        else:
            h_fused = h_local + h_context
            
        return h_fused, nbr_idx, nbr_w, s

class HierarchicalChangeReasoner(nn.Module):
    """
    Full Stage 4C Architecture.
    """
    def __init__(self, cfg: HierarchicalConfig):
        super().__init__()
        self.cfg = cfg
        self.token_encoder = TokenEncoder(cfg)
        self.transformer = TransformerReasoner(cfg)
        self.hierarchical_graph = HierarchicalGraphReasoner(cfg)
        self.change_head = ChangePredictionHead(cfg)
        self.delta_head = DeltaHead(cfg)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        B, N, _ = batch["tokens_pad"].shape
        
        h = self.token_encoder(
            batch["tokens_pad"].reshape(B*N, -1),
            batch["time_ids_pad"].reshape(B*N),
            batch["centroids_pad"].reshape(B*N, 2),
            batch["log_areas_pad"].reshape(B*N)
        ).reshape(B, N, -1)
        
        h = self.transformer(h, batch["padding_mask"])
        
        h_fused, nbr_idx, nbr_w, s = self.hierarchical_graph(
            h, batch["padding_mask"], batch["centroids_pad"], batch["time_ids_pad"]
        )
        
        change_logits = self.change_head(h_fused)
        
        # Delta predictions
        pair_b = batch["pair_b"]
        pair_i = batch["pair_i"]
        pair_j = batch["pair_j"]
        if len(pair_b) > 0:
            delta_pred = self.delta_head(h_fused[pair_b, pair_i], h_fused[pair_b, pair_j])
        else:
            delta_pred = torch.zeros(0, device=h.device)
            
        return {
            "change_logits": change_logits,
            "delta_pred": delta_pred,
            "delta_target": batch["delta_target"],
            "nbr_idx": nbr_idx,
            "nbr_w": nbr_w,
            "assignment": s
        }

def compute_hierarchical_loss(outputs, batch, cfg: HierarchicalConfig):
    """
    Custom loss with Smoothness Regularization.
    """
    from token_change_reasoner import compute_loss as base_compute_loss
    
    # Base losses (Change BCE + Delta MSE)
    losses = base_compute_loss(outputs, batch, cfg)
    
    # Smoothness Regularization
    L_smooth = compute_smoothness_loss(
        outputs["change_logits"], 
        outputs["nbr_idx"], 
        outputs["nbr_w"],
        mask=batch["padding_mask"]
    )
    
    losses["smoothness_loss"] = L_smooth
    losses["total_loss"] = losses["total_loss"] + cfg.smoothness_weight * L_smooth
    
    return losses
