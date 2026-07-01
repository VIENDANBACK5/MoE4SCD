"""
mob_gcn_modules.py
==================
Core modules for Multiresolution Graph Networks (MGN) inspired by MOB-GCN (2025).
Includes Gumbel-Softmax pooling and hierarchical GNN layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelSoftmaxPool(nn.Module):
    """
    Differentiable pooling using Gumbel-Softmax to assign nodes to clusters.
    """
    def __init__(self, in_channels, num_clusters, temperature=1.0):
        super().__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature
        self.assign_net = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.GELU(),
            nn.Linear(in_channels // 2, num_clusters)
        )

    def forward(self, x, mask=None):
        """
        x: [B, N, H]
        mask: [B, N] bool (True=pad)
        """
        B, N, H = x.shape
        logits = self.assign_net(x) # [B, N, C]
        
        if mask is not None:
            # Mask out padding nodes in assignment
            logits = logits.masked_fill(mask.unsqueeze(-1), -1e9)
            
        # Gumbel-Softmax assignment matrix S
        if self.training:
            s = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
        else:
            s = F.softmax(logits, dim=-1)
            # Optional: hard assignment during inference
            # s_hard = torch.zeros_like(s).scatter_(-1, s.argmax(dim=-1, keepdim=True), 1.0)
            # s = s_hard
            
        # Coarsen features: [B, C, H]
        x_pool = torch.matmul(s.transpose(1, 2), x)
        
        # Count nodes per cluster (for normalization)
        cluster_counts = s.sum(dim=1, keepdim=True).clamp(min=1.0) # [B, 1, C]
        x_pool = x_pool / cluster_counts.transpose(1, 2)
        
        return x_pool, s

class MGNLayer(nn.Module):
    """
    Multiresolution Graph Network Layer.
    Processes features at current level and interacts with global clusters.
    """
    def __init__(self, channels, num_clusters, k=6):
        super().__init__()
        self.k = k
        self.pool = GumbelSoftmaxPool(channels, num_clusters)
        self.gnn_local = nn.Linear(channels, channels) # Placeholder for real GNN
        self.gnn_global = nn.Linear(channels, channels)
        
        self.fuse = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.LayerNorm(channels),
            nn.GELU()
        )

    def forward(self, x, nbr_idx, nbr_w, mask=None):
        """
        x: [B, N, H]
        nbr_idx, nbr_w: Local graph indices/weights
        """
        # 1. Local GNN update (Simplified aggregation)
        # In actual implementation, we use GraphSAGE aggregation here
        # ... (implementation inside HierarchicalGraphReasoner) ...
        
        # 2. Global Pooling
        x_global, s = self.pool(x, mask) # [B, C, H]
        
        # 3. Global update (Interaction between clusters)
        # Note: Clusters are fully connected or have their own graph
        x_global = self.gnn_global(x_global)
        
        # 4. Broadcast back to local nodes: [B, N, H]
        x_context = torch.matmul(s, x_global)
        
        return x_context, x_global, s

def compute_smoothness_loss(logits, nbr_idx, nbr_w, mask=None):
    """
    Encourages neighboring nodes to have similar predictions.
    L_smooth = sum_{i,j} w_ij * (y_i - y_j)^2
    """
    B, N = logits.shape
    k = nbr_idx.shape[2]
    H = 1 # scalar logits
    
    # Sigmoid to get probabilities [B, N]
    probs = torch.sigmoid(logits)
    
    # Gather neighbor probabilities: [B, N, k]
    flat_idx = nbr_idx.reshape(B, N * k)
    probs_flat = torch.gather(probs, 1, flat_idx)
    probs_nbr = probs_flat.reshape(B, N, k)
    
    # Square difference: [B, N, k]
    diff_sq = (probs.unsqueeze(-1) - probs_nbr).pow(2)
    
    # Weighted sum
    loss_val = (nbr_w * diff_sq).sum(dim=-1) # [B, N]
    
    if mask is not None:
        loss_val = loss_val.masked_fill(mask, 0.0)
        return loss_val.sum() / (~mask).sum().clamp(min=1.0)
    else:
        return loss_val.mean()
