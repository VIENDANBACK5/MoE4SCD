"""Unit tests for Neuro-Algorithmic Deadwood Segmentation modules."""
from __future__ import annotations


# Ensure workspace root is in sys.path
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "crown_segmentation_research").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

import pytest
import torch
import numpy as np

from crown_segmentation_research.legacy.neuro_algorithmic_modules import (
    DeadwoodDirectionalExtentHead,
    SpectralModularityPartition,
    MedialAxisBellmanFord,
)
from crown_segmentation_research.methods.dynamic_mask.neuro_deadwood_net import NeuroDeadwoodNet
from crown_segmentation_research.methods.dynamic_mask.neuro_deadwood_loss import NeuroDeadwoodLoss
from crown_segmentation_research.methods.dynamic_mask.neuro_deadwood_decode import decode_neuro_deadwood


def test_extent_head_forward():
    B, C, H, W = 2, 256, 32, 32
    head = DeadwoodDirectionalExtentHead(in_channels=C, num_support_directions=16, hidden_dim=64)
    feat_map = torch.randn(B, C, H, W)
    coords = torch.randn(B, H, W, 2)
    mask = torch.sigmoid(torch.randn(B, 1, H, W))
    
    out = head(feat_map, coords, foreground_mask=mask)
    assert "orientation_field" in out
    assert "extent_radii" in out
    assert "coreset_points" in out
    assert out["orientation_field"].shape == (B, 1, H, W)
    assert out["extent_radii"].shape == (B, 16)
    assert out["coreset_points"].shape == (B, 16, 2)


def test_spectral_modularity_forward():
    B, C, H, W = 2, 256, 16, 16 # smaller spatial size for fast unit test
    mod = SpectralModularityPartition(in_channels=C, num_clusters=4, num_power_iters=2, hidden_dim=32)
    feat_map = torch.randn(B, C, H, W)
    coords = torch.randn(B, H, W, 2)
    
    out = mod(feat_map, coords)
    assert "cluster_assignments" in out
    assert "loss_modularity" in out
    assert out["cluster_assignments"].shape == (B, H * W, 4)
    assert out["loss_modularity"].ndim == 0 # scalar loss


def test_bellman_ford_forward():
    B, C, H, W = 2, 256, 16, 16
    bf = MedialAxisBellmanFord(in_channels=C, num_iterations=2, hidden_dim=32)
    feat_map = torch.randn(B, C, H, W)
    coords = torch.randn(B, H, W, 2)
    orient = torch.randn(B, 1, H, W)
    
    out = bf(feat_map, coords, orient)
    assert "continuity_affinity" in out
    assert "loss_sparse" in out
    assert out["continuity_affinity"].shape == (B, H * W, H * W)
    assert out["loss_sparse"].ndim == 0


def test_neuro_deadwood_net_end_to_end():
    B, H, W = 2, 128, 128
    model = NeuroDeadwoodNet(
        n_directions=16,
        pretrained_backbone=False,
        use_eps_kernel=True,
        use_modularity=True,
        use_bellman_ford=True,
        num_modularity_clusters=4,
    )
    images = torch.randn(B, 3, H, W)
    outputs = model(images)
    
    assert outputs["probability"].shape == (B, 1, H, W)
    assert outputs["canopy"].shape == (B, 1, H, W)
    assert outputs["orientation"].shape == (B, 1, H, W)
    assert outputs["extent_radii"].shape == (B, 16)
    assert "loss_modularity" in outputs
    assert "loss_sparse" in outputs
    
    # Test loss & backward pass
    criterion = NeuroDeadwoodLoss()
    targets = {
        "probability": (torch.rand(B, 1, H, W) > 0.8).float(),
        "canopy": (torch.rand(B, 1, H, W) > 0.5).float(),
        "rays": torch.rand(B, 16),
    }
    loss_dict = criterion(outputs, targets)
    total_loss = loss_dict["total_loss"]
    assert torch.isfinite(total_loss)
    
    total_loss.backward()
    # Check gradients exist
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0.0


def test_decode_function():
    h, w = 128, 128
    prob = np.zeros((h, w), dtype=np.float32)
    # Add two artificial peaks
    prob[30, 40] = 0.95
    prob[70, 80] = 0.88
    
    orient = np.full((h, w), 0.5, dtype=np.float32)
    radii = np.full((16,), 12.0, dtype=np.float32)
    
    polys = decode_neuro_deadwood(
        prob,
        orientation=orient,
        extent_radii=radii,
        prob_threshold=0.5,
        min_peak_distance=5,
    )
    assert len(polys) == 2
    assert all(p.is_valid and p.area > 0 for p in polys)
