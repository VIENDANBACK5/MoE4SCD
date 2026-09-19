"""Dense star-convex prediction network.

Backbone: torchvision's ResNet50-FPN, the same backbone family already used
by the frozen G1B Mask R-CNN checkpoint (`experiments/g1b_baselines/`), so
the encoder can be warm-started from that checkpoint's backbone weights
later if useful -- not done automatically here, since this is architecture
code, not a trained model yet.

Two dense heads run on the finest FPN level (stride 4, torchvision's key
"0"), matching this project's "predict at every pixel, no region-proposal
step" design principle (Section 2 of
design_docs/method_design_dense_crown_separation_v1.md) -- unlike Mask
R-CNN, there is no RPN here, so this architecture cannot reproduce the
proposal-drop failure mode G4D was checking for, though it can still fail
for other reasons the G4D/oracle-test results already flagged (classifier-
style under-confidence has no direct analogue tested yet for this head).

- object_probability: 1 channel, sigmoid. Target: star_convex_targets.object_probability_map.
- ray_distances: n_rays channels, softplus (>=0). Target: star_convex_targets.ray_distance_maps.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class DenseHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class StarConvexNet(nn.Module):
    """Predicts dense (object_probability, ray_distances) maps at stride 4.

    Callers must upsample outputs back to input resolution (stride 4 -> 1)
    before comparing to full-resolution targets from
    star_convex_targets.build_targets; this is done in forward() via
    bilinear interpolation, matching the resolution the training targets
    are generated at.
    """

    def __init__(
        self, n_rays: int = 32, pretrained_backbone: bool = True, use_canopy_head: bool = False,
        use_embedding_head: bool = False, embedding_dim: int = 8,
        use_centroid_head: bool = False, use_sdt_head: bool = False,
    ):
        super().__init__()
        self.n_rays = n_rays
        self.use_canopy_head = use_canopy_head
        self.use_embedding_head = use_embedding_head
        self.use_centroid_head = use_centroid_head
        self.use_sdt_head = use_sdt_head
        weights_name = "IMAGENET1K_V1" if pretrained_backbone else None
        self.backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=weights_name, trainable_layers=5)
        fpn_out_channels = self.backbone.out_channels
        self.probability_head = DenseHead(fpn_out_channels, 1)
        self.ray_head = DenseHead(fpn_out_channels, n_rays)
        if use_centroid_head:
            # Gaussian instance-centroid heatmap, sigmoid + MSE (TreeMort-3T-UNet,
            # arXiv:2503.21438, research.md Addendum 6/8): a second, independent
            # "where is an instance center" signal distinct from object_probability
            # (which peaks at every foreground pixel's own distance-to-boundary,
            # not specifically at instance centers). Motivation: this project's
            # measured bottleneck on DeadTrees/DTE-aerial-bench is catastrophically
            # low recall (2-34%), and TreeMort's ablation is the strongest
            # task-matched (standing dead tree) evidence found that this specific
            # head lifts recall (0.467->0.669 in their own numbers).
            self.centroid_head = DenseHead(fpn_out_channels, 1)
        if use_sdt_head:
            # Signed-distance-transform / boundary head (TreeMort's third head):
            # regresses a smooth signed-distance surface (positive inside an
            # instance growing toward its medial axis, negative-approaching
            # outside near the boundary) rather than the hard 0/1 probability
            # target, giving a continuous signal for both instance interiors
            # and the thin inter-instance boundary ridge. tanh-bounded output
            # since targets are normalized to [-1, 1] (see sdt_target in
            # train_star_convex.py's on-the-fly target computation).
            self.sdt_head = DenseHead(fpn_out_channels, 1)
        if use_canopy_head:
            # Binary canopy/background head, separate from object_probability:
            # object_probability answers "how deep inside *an* instance is
            # this pixel" (0 at every instance's own boundary too), while
            # canopy answers the coarser "is this crown material at all"
            # question. Diagnosed in star_convex_v3_failure_diagnosis.md:
            # 70% of false positives have zero overlap with any real crown
            # (background texture mistaken for a crown), a problem plain
            # instance-probability thresholding can't fix on its own.
            self.canopy_head = DenseHead(fpn_out_channels, 1)
        if use_embedding_head:
            # Per-pixel embedding trained with a discriminative (pull/push)
            # loss (De Brabandere et al. 2017), an explicit inter-instance
            # repulsion signal that star_convex_v6_boundary_weight_result.md
            # found plain probability-loss reweighting cannot provide (it
            # can raise signal near a boundary but not teach the network
            # *whose* boundary it is). No activation -- raw embedding
            # coordinates, the loss itself shapes the space.
            self.embedding_head = DenseHead(fpn_out_channels, embedding_dim)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """images: (N, 3, H, W) float in [0, 1]. Returns full-resolution maps."""
        features = self.backbone(images)
        finest = features["0"]  # stride 4
        target_size = images.shape[-2:]

        probability_logits = self.probability_head(finest)
        ray_raw = self.ray_head(finest)

        probability = torch.sigmoid(probability_logits)
        rays = torch.nn.functional.softplus(ray_raw)

        probability = torch.nn.functional.interpolate(
            probability, size=target_size, mode="bilinear", align_corners=False
        )
        rays = torch.nn.functional.interpolate(
            rays, size=target_size, mode="bilinear", align_corners=False
        )
        output = {"probability": probability, "rays": rays}

        if self.use_centroid_head:
            centroid_logits = self.centroid_head(finest)
            centroid = torch.sigmoid(centroid_logits)
            output["centroid"] = torch.nn.functional.interpolate(
                centroid, size=target_size, mode="bilinear", align_corners=False
            )
        if self.use_sdt_head:
            sdt_raw = self.sdt_head(finest)
            sdt = torch.tanh(sdt_raw)
            output["sdt"] = torch.nn.functional.interpolate(
                sdt, size=target_size, mode="bilinear", align_corners=False
            )
        if self.use_canopy_head:
            canopy_logits = self.canopy_head(finest)
            canopy = torch.sigmoid(canopy_logits)
            output["canopy"] = torch.nn.functional.interpolate(
                canopy, size=target_size, mode="bilinear", align_corners=False
            )
        if self.use_embedding_head:
            embedding = self.embedding_head(finest)
            output["embedding"] = torch.nn.functional.interpolate(
                embedding, size=target_size, mode="bilinear", align_corners=False
            )
        return output
