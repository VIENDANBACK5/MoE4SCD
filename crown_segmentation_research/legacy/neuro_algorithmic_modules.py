"""Core Neuro-Algorithmic PyTorch Modules for DeadTrees Segmentation.

Implements the theoretical frameworks from Prof. Yusu Wang's seminar (UCSD)
and recent papers (NeurIPS 2025, ICLR 2025):
  1. DeadwoodDirectionalExtentHead: Anisotropic eps-Kernel SumFormer extent processor
     with Paraboloid Lifting space.
  2. SpectralModularityPartition: GNN Power Iteration + Virtual Node global normalization
     optimizing graph Modularity Q to disentangle crisscrossing fallen deadwood.
  3. MedialAxisBellmanFord: Algorithmic-aligned shortest-path dynamic programming
     relaxation layer with sparsity regularization to bridge shadow gaps on fallen logs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeadwoodDirectionalExtentHead(nn.Module):
    """Anisotropic eps-Kernel Directional Extent Head (NeurIPS 2025).

    Evaluates directional support function h_P(u) = max_{p in P} <p, u> along an
    adaptive directional basis oriented by the predicted longitudinal field phi.
    Uses Paraboloid Lifting (x, y, x^2 + y^2) for non-convex hull fitting.
    """
    def __init__(self, in_channels: int = 256, num_support_directions: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.num_directions = num_support_directions
        
        # 1. Longitudinal Orientation Head: predicts [sin(2*phi), cos(2*phi)]
        self.orient_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=1)
        )
        
        # 2. Canonical Directional Basis on S^1 (equi-spaced in [0, 2*pi))
        canonical_angles = torch.linspace(0, 2 * torch.pi, num_support_directions + 1)[:-1]
        canonical_dirs = torch.stack([torch.cos(canonical_angles), torch.sin(canonical_angles)], dim=1) # (K, 2)
        self.register_buffer("canonical_dirs", canonical_dirs)
        
        # 3. SumFormer Linear Attention Projector in Paraboloid Lifting Space
        # Input features: feature (C) + coords (2) + lifted (1: x^2+y^2) + orient (2) = C + 5
        self.support_proj = nn.Sequential(
            nn.Linear(in_channels + 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_support_directions)
        )

    def forward(
        self,
        feat_map: torch.Tensor,
        coords_grid: torch.Tensor,
        foreground_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        feat_map: (B, C, H, W) - FPN dense feature map
        coords_grid: (B, H, W, 2) - normalized relative coordinates in [-1, 1]
        foreground_mask: (B, 1, H, W) - optional soft probability / canopy gating mask
        """
        B, C, H, W = feat_map.shape
        N = H * W
        
        # 1. Orientation field
        orient_logits = self.orient_conv(feat_map) # (B, 2, H, W)
        sin_2phi = orient_logits[:, 0:1] # (B, 1, H, W)
        cos_2phi = orient_logits[:, 1:2] # (B, 1, H, W)
        phi = 0.5 * torch.atan2(sin_2phi, cos_2phi) # (B, 1, H, W) in [-pi/2, pi/2]
        
        # 2. Lifting Space: (x, y) -> (x, y, x^2 + y^2)
        flat_coords = coords_grid.view(B, N, 2) # (B, N, 2)
        r_sq = torch.sum(flat_coords ** 2, dim=-1, keepdim=True) # (B, N, 1)
        flat_orient = orient_logits.view(B, 2, N).transpose(1, 2) # (B, N, 2)
        
        flat_feats = feat_map.view(B, C, N).transpose(1, 2) # (B, N, C)
        x_lifted = torch.cat([flat_feats, flat_coords, r_sq, flat_orient], dim=-1) # (B, N, C + 5)
        
        # 3. Directional Support Logits
        support_logits = self.support_proj(x_lifted) # (B, N, K)
        
        if foreground_mask is not None:
            flat_mask = foreground_mask.view(B, N, 1).float()
            # Masked softmax so only foreground tokens contribute to extent extreme points
            masked_logits = support_logits.float() + (1.0 - flat_mask.clamp(0.0, 1.0)) * (-1e4)
        else:
            masked_logits = support_logits.float()
            
        attn_weights = F.softmax(masked_logits, dim=1) # (B, N, K) sum over N equals 1
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # 4. Extreme Coreset Points Q_k for each direction k: Q_k = sum_i w_{ik} * p_i
        coreset_points = torch.einsum("bnk,bnd->bkd", attn_weights, flat_coords.float()) # (B, K, 2)
        
        # 5. Rotate canonical directions by average local orientation
        mean_phi = phi.mean(dim=[2, 3]) # (B, 1)
        cos_p = torch.cos(mean_phi).unsqueeze(-1) # (B, 1, 1)
        sin_p = torch.sin(mean_phi).unsqueeze(-1) # (B, 1, 1)
        rot_mat = torch.cat([
            torch.cat([cos_p, -sin_p], dim=-1),
            torch.cat([sin_p, cos_p], dim=-1)
        ], dim=1) # (B, 2, 2)
        
        rotated_dirs = torch.matmul(self.canonical_dirs.unsqueeze(0).to(rot_mat.dtype), rot_mat.transpose(1, 2)) # (B, K, 2)
        
        # Directional support value (radius along u_k): h_k = <Q_k, u_k>
        extent_radii = torch.sum(coreset_points * rotated_dirs, dim=-1) # (B, K)
        extent_radii = torch.nan_to_num(extent_radii, nan=0.0)
        
        return {
            "orientation_field": phi, # (B, 1, H, W)
            "extent_radii": extent_radii, # (B, K)
            "coreset_points": coreset_points, # (B, K, 2)
            "rotated_dirs": rotated_dirs, # (B, K, 2)
            "attn_weights": attn_weights, # (B, N, K)
        }


class SpectralModularityPartition(nn.Module):
    """Spectral Graph Modularity Partitioning via Neural Power Iteration.

    Maximizes Newman's Modularity Q = (1 / 2m) Tr(S^T B S) to disentangle
    crisscrossing or tangled fallen deadwood without requiring centroid peaks.
    Uses adaptive graph pooling for O(1) memory scalability.
    """
    def __init__(
        self,
        in_channels: int = 256,
        num_clusters: int = 8,
        num_power_iters: int = 3,
        hidden_dim: int = 64,
        graph_grid_size: int = 16,
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.num_power_iters = num_power_iters
        self.graph_grid_size = graph_grid_size
        
        self.node_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        # Global Rayleigh Quotient Normalizer (Virtual Node)
        self.virtual_node = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        
        # Cluster Assignment Head
        self.assign_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_clusters)
        )

    def forward(self, feat_map: torch.Tensor, coords_grid: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        feat_map: (B, C, H, W)
        coords_grid: (B, H, W, 2) in [-1, 1]
        """
        B, C, H, W = feat_map.shape
        G = self.graph_grid_size
        
        # Adaptive pooling for tractable graph size (e.g. 16x16 -> N=256)
        if H > G or W > G:
            pooled_feats = F.adaptive_avg_pool2d(feat_map, (G, G))
            pooled_coords = F.adaptive_avg_pool2d(coords_grid.permute(0, 3, 1, 2), (G, G)).permute(0, 2, 3, 1)
        else:
            pooled_feats = feat_map
            pooled_coords = coords_grid
            
        N = pooled_feats.shape[2] * pooled_feats.shape[3]
        node_feats = self.node_proj(pooled_feats).view(B, -1, N).transpose(1, 2).float() # (B, N, hidden_dim)
        flat_coords = pooled_coords.view(B, N, 2).float() # (B, N, 2)
        
        # 1. Spatial-Spectral Affinity Matrix A
        dist_sq = torch.cdist(flat_coords, flat_coords) ** 2 # (B, N, N)
        feat_dist_sq = torch.cdist(node_feats, node_feats) ** 2 # (B, N, N)
        
        A = torch.exp(-dist_sq / 0.1 - feat_dist_sq / 2.0).clamp(min=1e-6, max=1.0) # (B, N, N)
        
        # 2. Power Iteration with Virtual Node Normalization
        d = torch.sum(A, dim=-1, keepdim=True).clamp(min=1e-4) # Degree (B, N, 1)
        d_inv_sqrt = torch.pow(d, -0.5)
        L_norm = d_inv_sqrt * A * d_inv_sqrt.transpose(1, 2) # (B, N, N)
        
        v = node_feats
        for _ in range(self.num_power_iters):
            v_next = torch.bmm(L_norm, v) + self.virtual_node.unsqueeze(0).float()
            v = F.normalize(v_next, p=2, dim=-1) # (B, N, hidden_dim)
            
        # 3. Soft Cluster Assignment S in R^{N x K}
        node_context = torch.cat([v, flat_coords], dim=-1) # (B, N, hidden_dim + 2)
        logits = self.assign_head(node_context) # (B, N, K)
        S = F.softmax(logits, dim=-1) # (B, N, K)
        
        # 4. Modularity Matrix B = A - (d * d^T) / (2m)
        m = torch.sum(d, dim=1, keepdim=True) * 0.5 # (B, 1, 1) total edge weight
        B_mod = A - torch.bmm(d, d.transpose(1, 2)) / (2.0 * m.clamp(min=1e-4)) # (B, N, N)
        
        # Modularity Q = Tr(S^T B S) / (2m)
        SB = torch.bmm(S.transpose(1, 2), B_mod) # (B, K, N)
        SBS = torch.bmm(SB, S) # (B, K, K)
        modularity_trace = torch.diagonal(SBS, dim1=-2, dim2=-1).sum(dim=-1) # (B,)
        modularity_q = modularity_trace / (2.0 * m.squeeze(-1).squeeze(-1).clamp(min=1e-4)) # (B,)
        modularity_q = torch.nan_to_num(modularity_q, nan=0.0).clamp(-1.0, 1.0)
        
        # Modularity Loss: maximize Q -> minimize -Q
        loss_modularity = -torch.mean(modularity_q)
        
        return {
            "cluster_assignments": S, # (B, N, K)
            "modularity_q": modularity_q, # (B,)
            "loss_modularity": loss_modularity,
            "eigen_embeddings": v # (B, N, hidden_dim)
        }


class MedialAxisBellmanFord(nn.Module):
    """Algorithmic-Aligned Shortest Path Dynamic Programming Layer.

    Applies (min, +) dynamic programming updates to trace longitudinal medial axes
    of fallen logs across shadow interruptions, with sparsity regularization.
    Uses adaptive graph pooling and feature projection for memory efficiency.
    """
    def __init__(
        self,
        in_channels: int = 256,
        num_iterations: int = 4,
        proj_dim: int = 32,
        hidden_dim: int = 32,
        graph_grid_size: int = 16,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.graph_grid_size = graph_grid_size
        
        self.feat_proj = nn.Sequential(
            nn.Conv2d(in_channels, proj_dim, kernel_size=1),
            nn.GELU()
        )
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(proj_dim * 2 + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus() # strictly non-negative edge costs
        )

    def forward(
        self,
        feat_map: torch.Tensor,
        coords_grid: torch.Tensor,
        orientation_field: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        feat_map: (B, C, H, W)
        coords_grid: (B, H, W, 2)
        orientation_field: (B, 1, H, W)
        """
        B, C, H, W = feat_map.shape
        G = self.graph_grid_size
        
        # Project channels and downsample for graph operations
        proj_feats = self.feat_proj(feat_map) # (B, proj_dim, H, W)
        
        if H > G or W > G:
            pooled_feats = F.adaptive_avg_pool2d(proj_feats, (G, G))
            pooled_coords = F.adaptive_avg_pool2d(coords_grid.permute(0, 3, 1, 2), (G, G)).permute(0, 2, 3, 1)
            pooled_orient = F.adaptive_avg_pool2d(orientation_field, (G, G))
        else:
            pooled_feats = proj_feats
            pooled_coords = coords_grid
            pooled_orient = orientation_field
            
        N = pooled_feats.shape[2] * pooled_feats.shape[3]
        flat_feats = pooled_feats.view(B, -1, N).transpose(1, 2) # (B, N, proj_dim)
        flat_coords = pooled_coords.view(B, N, 2) # (B, N, 2)
        flat_orient = pooled_orient.view(B, 1, N).transpose(1, 2) # (B, N, 1)
        
        # Compute spatial distance and angular alignment
        dist = torch.cdist(flat_coords, flat_coords) # (B, N, N)
        delta_phi = torch.abs(flat_orient - flat_orient.transpose(1, 2)) # (B, N, N)
        
        # Spatial gating: only consider local neighbors (d < 0.25)
        local_mask = dist < 0.25
        
        # Pairwise feature representation
        f_i = flat_feats.unsqueeze(2).expand(B, N, N, flat_feats.shape[-1])
        f_j = flat_feats.unsqueeze(1).expand(B, N, N, flat_feats.shape[-1])
        geom_feat = torch.cat([f_i, f_j, dist.unsqueeze(-1), delta_phi.unsqueeze(-1)], dim=-1) # (B, N, N, 2*proj_dim + 2)
        
        # Edge transition cost w_ij (non-negative)
        raw_edge_costs = self.edge_mlp(geom_feat).squeeze(-1).float() # (B, N, N)
        # Apply local mask
        edge_costs = raw_edge_costs + (~local_mask).float() * 1e4
        
        # Algorithmic Loop: Bellman-Ford (min, +) relaxation
        path_costs = edge_costs.clone()
        for _ in range(self.num_iterations):
            min_incoming, _ = torch.min(path_costs, dim=1, keepdim=True) # (B, 1, N)
            path_costs = torch.minimum(path_costs, edge_costs + min_incoming)
            
        # Sparsity Regularization: penalize redundant edge connections on valid local edges
        num_valid = local_mask.float().sum().clamp(min=1.0)
        loss_sparse = (raw_edge_costs * local_mask.float()).sum() / num_valid
        loss_sparse = torch.nan_to_num(loss_sparse, nan=0.0)
        
        # Continuity affinity map (high affinity = low shortest-path cost)
        continuity_affinity = torch.exp(-path_costs.clamp(max=20.0) / 1.0) # (B, N, N)
        
        return {
            "continuity_affinity": continuity_affinity,
            "loss_sparse": loss_sparse,
            "path_costs": path_costs
        }
