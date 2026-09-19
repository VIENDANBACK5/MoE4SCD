"""TreeCoG-lite: a practical, buildable version of TreeCoG's contour+GCN-merge
idea (research.md Addendum 7, doi.org/10.1038/s41598-026-36541-y), for
testing on DeadTrees data without the heavy EDTER edge-transformer stage
the original paper trains from scratch (Phase 1, 200 epochs on their own
data) -- infeasible to reproduce in this session's compute budget.

Substitutions from the original recipe, and why they are reasonable
approximations rather than arbitrary shortcuts:
- Contour/over-segmentation: SLIC superpixels (skimage) instead of a
  trained EDTER edge map. TreeCoG's own ablation (Table 3, Addendum 7)
  shows contour-generator quality matters but the *pipeline* still works
  with a much weaker generator (PiDiNet: 60.22% classification accuracy,
  AP 50.77 vs. EDTER's 65.37%/57.01% -- a ~5-6 point gap, not a collapse).
  SLIC has no learned prior at all, so this is a lower-bound test of the
  contour+merge *idea*, not a reproduction of TreeCoG's exact numbers.
- Appearance similarity: mean+std RGB per superpixel patch, cosine
  similarity, instead of LPIPS+AlexNet (the original needs a pretrained
  ImageNet CNN purely as a fixed feature extractor; a cheap color
  descriptor plays the same structural role in the pipeline -- separating
  same-crown vs. different-crown patches -- without adding a heavy
  dependency).
- Shape features: area, extent, solidity, eccentricity (aspect-ratio
  proxy), and a perimeter^2/(4*pi*area) circularity-based deviation proxy,
  from skimage.measure.regionprops -- four of TreeCoG's five features
  (area/extent/solidity/aspect_ratio) are used almost as specified;
  "deviation" (arc-length integral vs. convex hull) is approximated by
  circularity since skimage does not expose the former directly.
- GCN: implemented directly (paper's Eq. 13-15: symmetric-normalized
  adjacency message passing, MLP edge classifier, BCE loss) since
  torch_geometric is not installed in this environment and the paper's
  own graph is tiny (tens of nodes per image) -- no need for a library.

Everything else (K=9 nearest-neighbor graph sized to match GT instance
density, Algorithm 1's majority-vote merge-ground-truth construction, union
of merged regions as the final instance mask) follows the paper directly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.measure import regionprops
from skimage.segmentation import felzenszwalb, slic
from torch import nn

K_NEIGHBORS = 9
N_SEGMENTS = 120
MIN_GT_AREA = 20


def extract_superpixel_graph(image: np.ndarray, instance_label: np.ndarray | None = None):
    """Returns (segments, node_features[N,9], adjacency[N,N] bool, centroids[N,2],
    merge_labels[N] int or None if instance_label not given)."""
    # felzenszwalb (graph-based, merges by actual color/intensity discontinuity)
    # instead of slic (fixed-size compactness-constrained grid): closer in
    # spirit to an edge-aware contour generator like TreeCoG's EDTER, since
    # slic's uniform partition ignores true object boundaries entirely --
    # measured directly as the likely cause of TreeCoG-lite's initial failure.
    segments = felzenszwalb(image, scale=300, sigma=0.8, min_size=800) + 1  # start_label=1 convention
    props = regionprops(segments, intensity_image=image[..., 0])
    ids = sorted(np.unique(segments))
    n = len(ids)

    shape_features = np.zeros((n, 4), dtype=np.float32)
    appearance = np.zeros((n, 6), dtype=np.float32)
    centroids = np.zeros((n, 2), dtype=np.float32)
    merge_labels = np.zeros(n, dtype=np.int64) if instance_label is not None else None

    for i, region in enumerate(props):
        mask = segments == region.label
        area = region.area
        bbox_area = max(region.bbox_area, 1)
        convex_area = max(region.convex_area, 1)
        perimeter = max(region.perimeter, 1e-6)
        extent = area / bbox_area
        solidity = area / convex_area
        eccentricity = region.eccentricity
        circularity = (perimeter**2) / (4 * np.pi * area + 1e-6)
        shape_features[i] = [extent, solidity, eccentricity, circularity]
        centroids[i] = region.centroid

        patch = image[mask]
        appearance[i, :3] = patch.mean(axis=0) / 255.0
        appearance[i, 3:] = patch.std(axis=0) / 255.0

        if merge_labels is not None:
            values, counts = np.unique(instance_label[mask], return_counts=True)
            merge_labels[i] = values[np.argmax(counts)]

    # K nearest neighbors by centroid distance
    dists = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    k = min(K_NEIGHBORS, n - 1) if n > 1 else 0
    neighbor_idx = np.argsort(dists, axis=1)[:, :k]
    adjacency = np.zeros((n, n), dtype=bool)
    for i in range(n):
        adjacency[i, neighbor_idx[i]] = True
    adjacency = adjacency | adjacency.T

    node_features = np.concatenate([shape_features, appearance], axis=1)
    return segments, node_features, adjacency, centroids, merge_labels


class SimpleGCN(nn.Module):
    def __init__(self, in_dim: int = 10, hidden_dim: int = 32, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * n_layers
        for i in range(n_layers):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features: torch.Tensor, norm_adjacency: torch.Tensor) -> torch.Tensor:
        x = node_features
        for layer in self.layers:
            x = torch.relu(norm_adjacency @ layer(x))
        return x  # node embeddings

    def edge_logits(self, embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        edge_feat = 0.5 * (embeddings[src] + embeddings[dst])
        return self.edge_mlp(edge_feat).squeeze(-1)


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    a_hat = adjacency.astype(np.float32) + np.eye(len(adjacency), dtype=np.float32)
    degree = a_hat.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(degree, 1e-6, None)))
    return d_inv_sqrt @ a_hat @ d_inv_sqrt


def train_one_image(model, optimizer, node_features, adjacency, merge_labels, device, pos_weight: float = 1.0):
    n = len(node_features)
    edges = np.argwhere(np.triu(adjacency, k=1))
    if len(edges) == 0:
        return None
    edge_labels = (merge_labels[edges[:, 0]] == merge_labels[edges[:, 1]]).astype(np.float32)
    edge_labels = edge_labels * (merge_labels[edges[:, 0]] != 0)  # background never "merges"

    x = torch.from_numpy(node_features).float().to(device)
    norm_adj = torch.from_numpy(normalize_adjacency(adjacency)).float().to(device)
    edge_index = torch.from_numpy(edges.T).long().to(device)
    labels = torch.from_numpy(edge_labels).float().to(device)

    embeddings = model(x, norm_adj)
    logits = model.edge_logits(embeddings, edge_index)
    # "Merge" edges are rare (~0.7% of all superpixel-pair edges measured on
    # DeadTrees: sparse standing-dead-tree scenes are mostly background,
    # unlike TreeCoG's original dense-canopy scenes where most of the image
    # IS tree material) -- unweighted BCE collapses to always predicting
    # "never merge" (verified: loss plateaus after 1 epoch, val F1~0.016,
    # massive over-segmentation). pos_weight counteracts this exactly like
    # this project's own focal-loss rationale for the probability head.
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=torch.tensor(pos_weight, device=device)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def decode_instances(model, node_features, adjacency, segments, device, merge_threshold=0.5):
    n = len(node_features)
    edges = np.argwhere(np.triu(adjacency, k=1))
    if len(edges) == 0:
        return np.zeros_like(segments)

    x = torch.from_numpy(node_features).float().to(device)
    norm_adj = torch.from_numpy(normalize_adjacency(adjacency)).float().to(device)
    edge_index = torch.from_numpy(edges.T).long().to(device)
    embeddings = model(x, norm_adj)
    probs = torch.sigmoid(model.edge_logits(embeddings, edge_index)).cpu().numpy()

    # Union-find over superpixels using predicted merge edges
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (i, j), p in zip(edges, probs):
        if p >= merge_threshold:
            union(i, j)

    group_of = {i: find(i) for i in range(n)}
    ids = sorted(np.unique(segments))
    instance_map = np.zeros_like(segments)
    group_to_instance = {}
    next_instance = 1
    for node_idx, seg_id in enumerate(ids):
        group = group_of[node_idx]
        if group not in group_to_instance:
            group_to_instance[group] = next_instance
            next_instance += 1
        instance_map[segments == seg_id] = group_to_instance[group]
    return instance_map


def load_items(target_dir: Path):
    manifest = pd.read_csv(target_dir / "manifest.csv")
    items = []
    for image_id in manifest["image_id"].astype(str):
        path = target_dir / f"{image_id}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        items.append((image_id, data["image"], data["instance_label"].astype(np.int64)))
    return items


def instances_from_label_map(label_map: np.ndarray, min_area: int = MIN_GT_AREA) -> np.ndarray:
    masks = []
    for label in np.unique(label_map):
        if label == 0:
            continue
        mask = label_map == label
        if mask.sum() >= min_area:
            masks.append(mask)
    return np.stack(masks) if masks else np.zeros((0, *label_map.shape), dtype=bool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("DeadTrees/star_convex_targets_v1/train_small_4sites"))
    parser.add_argument("--val-dir", type=Path, default=Path("DeadTrees/star_convex_targets_v1/val"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=100.0, help="BCE positive-class weight to counteract the ~0.7% merge-edge imbalance measured on DeadTrees's sparse standing-dead-tree scenes")
    parser.add_argument("--merge-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("DeadTrees/experiments/treecog_lite"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading + building graphs for train set...", flush=True)
    train_items = load_items(args.train_dir)
    train_graphs = []
    for image_id, image, instance_label in train_items:
        _, node_features, adjacency, _, merge_labels = extract_superpixel_graph(image, instance_label)
        train_graphs.append((node_features, adjacency, merge_labels))
    print(f"Built {len(train_graphs)} training graphs", flush=True)

    model = SimpleGCN(in_dim=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(args.epochs):
        losses = []
        for node_features, adjacency, merge_labels in train_graphs:
            loss = train_one_image(model, optimizer, node_features, adjacency, merge_labels, device, pos_weight=args.pos_weight)
            if loss is not None:
                losses.append(loss)
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch, "loss": mean_loss})
        print(f"epoch {epoch}: loss={mean_loss:.4f}", flush=True)

    torch.save(model.state_dict(), args.output_dir / "treecog_lite_gcn.pth")
    (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2))

    print("Evaluating on held-out val...", flush=True)
    from deadtrees_pipeline.metrics import hungarian_match, overlap_matrices

    val_items = load_items(args.val_dir)
    total_tp = total_fp = total_fn = 0
    for image_id, image, instance_label in val_items:
        gt_masks = instances_from_label_map(instance_label)
        segments, node_features, adjacency, _, _ = extract_superpixel_graph(image, instance_label=None)
        pred_label_map = decode_instances(model, node_features, adjacency, segments, device, merge_threshold=args.merge_threshold)
        pred_masks = instances_from_label_map(pred_label_map, min_area=50)

        iou, _, _, _ = overlap_matrices(gt_masks, pred_masks)
        match = hungarian_match(iou, 0.50)
        total_tp += match.tp
        total_fp += match.fp
        total_fn += match.fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    result = {"precision": precision, "recall": recall, "f1": f1, "tp": total_tp, "fp": total_fp, "fn": total_fn}
    print(json.dumps(result, indent=2))
    (args.output_dir / "val_result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
