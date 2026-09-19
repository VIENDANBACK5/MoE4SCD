"""NeuroDeadwoodNet: Integrated End-to-End Neuro-Algorithmic Deadwood Segmentation Model.

Combines:
  - ResNet50-FPN Backbone (stride 4 feature pyramid)
  - Dense Foreground / Canopy Probability Heads
  - Anisotropic eps-Kernel SumFormer Directional Extent Head
  - Spectral Modularity Graph Partitioning Head (GNN Power Iteration)
  - Medial Axis Bellman-Ford Shortest Path Dynamic Programming Head
"""
from __future__ import annotations


# Ensure workspace root is in sys.path
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "crown_segmentation_research").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from crown_segmentation_research.legacy.neuro_algorithmic_modules import (
    DeadwoodDirectionalExtentHead,
    SpectralModularityPartition,
    MedialAxisBellmanFord,
)


class DenseHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class NeuroDeadwoodNet(nn.Module):
    """End-to-End Neuro-Algorithmic Model for DeadTrees Segmentation."""
    def __init__(
        self,
        n_directions: int = 16,
        pretrained_backbone: bool = True,
        backbone_ckpt: str | None = None,
        use_eps_kernel: bool = True,
        use_modularity: bool = True,
        use_bellman_ford: bool = True,
        num_modularity_clusters: int = 8,
    ):
        super().__init__()
        self.n_directions = n_directions
        self.use_eps_kernel = use_eps_kernel
        self.use_modularity = use_modularity
        self.use_bellman_ford = use_bellman_ford
        
        # 1. ResNet50-FPN Backbone
        weights_name = "IMAGENET1K_V1" if pretrained_backbone else None
        self.backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=weights_name,
            trainable_layers=5,
        )
        
        # Optionally warm-start backbone from TreeFlowNet or other checkpoint
        if backbone_ckpt and Path(backbone_ckpt).is_file():
            print(f"Warm-starting backbone from {backbone_ckpt}...")
            ckpt = torch.load(backbone_ckpt, map_location="cpu", weights_only=True)
            backbone_state = {k.replace("backbone.", ""): v for k, v in ckpt.items() if k.startswith("backbone.")}
            if backbone_state:
                self.backbone.load_state_dict(backbone_state, strict=False)
                print("  --> Backbone weights successfully loaded!")
        fpn_channels = 256
        
        # 2. Dense Foreground & Canopy Heads (Stride 4 -> Stride 1)
        self.probability_head = DenseHead(fpn_channels, 1)
        self.canopy_head = DenseHead(fpn_channels, 1)
        
        # 3. Anisotropic eps-Kernel Extent Head (SumFormer)
        if self.use_eps_kernel:
            self.extent_head = DeadwoodDirectionalExtentHead(
                in_channels=fpn_channels,
                num_support_directions=n_directions,
                hidden_dim=128,
            )
            
        # 4. Spectral Modularity Partitioning Head
        if self.use_modularity:
            self.modularity_head = SpectralModularityPartition(
                in_channels=fpn_channels,
                num_clusters=num_modularity_clusters,
                num_power_iters=3,
                hidden_dim=64,
            )
            
        # 5. Medial Axis Bellman-Ford Shortest Path Head
        if self.use_bellman_ford:
            self.bellman_ford_head = MedialAxisBellmanFord(
                in_channels=fpn_channels,
                num_iterations=4,
                hidden_dim=64,
            )

    def _get_coords_grid(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Generates normalized coordinate grid in [-1, 1]."""
        ys = torch.linspace(-1, 1, h, device=device)
        xs = torch.linspace(-1, 1, w, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx, yy], dim=-1) # (H, W, 2)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        images: (B, 3, H, W) in [0, 1]
        """
        B, _, H, W = images.shape
        device = images.device
        
        # 1. Feature Extraction via FPN
        fpn_features = self.backbone(images)
        # Use stride 4 finest feature map (torchvision key '0')
        f0 = fpn_features["0"] # (B, 256, H/4, W/4)
        _, _, h_feat, w_feat = f0.shape
        
        # Coordinate grid at feature resolution
        coords_grid = self._get_coords_grid(h_feat, w_feat, device).unsqueeze(0).expand(B, h_feat, w_feat, 2)
        
        # 2. Foreground Deadwood & Canopy Logits & Probabilities
        prob_low = self.probability_head(f0) # (B, 1, H/4, W/4)
        canopy_low = self.canopy_head(f0) # (B, 1, H/4, W/4)
        
        # Upsample raw logits to full image resolution
        prob_logits = F.interpolate(prob_low, size=(H, W), mode="bilinear", align_corners=False)
        canopy_logits = F.interpolate(canopy_low, size=(H, W), mode="bilinear", align_corners=False)
        probability = torch.sigmoid(prob_logits)
        canopy = torch.sigmoid(canopy_logits)
        
        output: dict[str, torch.Tensor] = {
            "prob_logits": prob_logits,
            "canopy_logits": canopy_logits,
            "probability": probability,
            "canopy": canopy,
            "feat_map": f0,
        }
        
        # 3. Anisotropic eps-Kernel Extent
        if self.use_eps_kernel:
            extent_out = self.extent_head(f0, coords_grid, foreground_mask=torch.sigmoid(prob_low))
            # Upsample orientation field to full image resolution
            orient_full = F.interpolate(extent_out["orientation_field"], size=(H, W), mode="bilinear", align_corners=False)
            output["orientation"] = orient_full
            output["extent_radii"] = extent_out["extent_radii"]
            output["coreset_points"] = extent_out["coreset_points"]
            output["rotated_dirs"] = extent_out["rotated_dirs"]
            
        # 4. Spectral Modularity Partitioning
        if self.use_modularity:
            mod_out = self.modularity_head(f0, coords_grid)
            output["cluster_assignments"] = mod_out["cluster_assignments"]
            output["modularity_q"] = mod_out["modularity_q"]
            output["loss_modularity"] = mod_out["loss_modularity"]
            
        # 5. Medial Axis Bellman-Ford Shortest Path
        if self.use_bellman_ford and self.use_eps_kernel:
            # Downsample feature map for Bellman-Ford if needed for memory
            bf_out = self.bellman_ford_head(f0, coords_grid, extent_out["orientation_field"])
            output["continuity_affinity"] = bf_out["continuity_affinity"]
            output["loss_sparse"] = bf_out["loss_sparse"]
            
        return output
