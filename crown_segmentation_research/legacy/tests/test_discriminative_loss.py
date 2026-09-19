
# Ensure workspace root is in sys.path
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "crown_segmentation_research").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

import torch

from crown_segmentation_research.legacy.discriminative_loss import discriminative_loss


def test_zero_instances_returns_zero_loss():
    embedding = torch.randn(4, 10, 10)
    instance_label = torch.zeros(10, 10, dtype=torch.long)
    losses = discriminative_loss(embedding, instance_label)
    assert float(losses["total"]) == 0.0


def test_already_separated_clusters_have_near_zero_loss():
    # two instances whose embeddings are already tight (var~0) and far apart
    # (dist >> delta_d) should incur ~0 loss under the hinge formulation
    embedding = torch.zeros(2, 10, 10)
    embedding[:, :5, :] = torch.tensor([0.0, 0.0]).view(2, 1, 1)
    embedding[:, 5:, :] = torch.tensor([10.0, 10.0]).view(2, 1, 1)
    instance_label = torch.zeros(10, 10, dtype=torch.long)
    instance_label[:5, :] = 1
    instance_label[5:, :] = 2
    losses = discriminative_loss(embedding, instance_label, delta_v=0.5, delta_d=1.5)
    assert float(losses["var_loss"]) < 1e-6
    assert float(losses["dist_loss"]) < 1e-6


def test_overlapping_clusters_incur_distance_loss():
    # two instances with identical embeddings (distance 0 << delta_d) should
    # incur a large distance-hinge loss
    embedding = torch.zeros(2, 10, 10)
    instance_label = torch.zeros(10, 10, dtype=torch.long)
    instance_label[:5, :] = 1
    instance_label[5:, :] = 2
    losses = discriminative_loss(embedding, instance_label, delta_v=0.5, delta_d=1.5)
    assert float(losses["dist_loss"]) > 1.0


def test_scattered_cluster_incurs_variance_loss():
    # one instance whose pixels are scattered far from their own mean should
    # incur a large variance-hinge loss
    torch.manual_seed(0)
    embedding = torch.randn(2, 10, 10) * 100.0
    instance_label = torch.ones(10, 10, dtype=torch.long)
    losses = discriminative_loss(embedding, instance_label, delta_v=0.5, delta_d=1.5)
    assert float(losses["var_loss"]) > 1.0
    assert float(losses["dist_loss"]) == 0.0  # only one instance, no pairs to push apart
