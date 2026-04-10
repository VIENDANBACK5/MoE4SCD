# Stage 3: Token Matching

**Pipeline**: SAM2 features (Stage 1) → Region Tokenization (Stage 2) → **Token Matching (Stage 3)** → Token Change Reasoner (Stage 4)

Token matching computes correspondences between SAM2 region tokens extracted at T1 and T2. It is the *alignment layer* before Stage 4 (the novelty stage).

---

## Files

| File | Description |
|------|-------------|
| `token_matching.py` | Main module: `TokenMatcher` class + CLI + dataset pipeline |
| `token_matching_utils.py` | Vectorized math helpers: similarity, Sinkhorn, metrics, visualization |
| `tests/test_matching.py` | Unit tests (pytest) |
| `notebooks/demo_matching.ipynb` | Interactive demo |

---

## Quick Start

```bash
# Match all training pairs with Hungarian (default)
python token_matching.py \
  --tokens_T1 SECOND/tokens_T1 \
  --tokens_T2 SECOND/tokens_T2 \
  --output SECOND/matches \
  --method hungarian \
  --device cuda \
  --visualize \
  --n_vis 20
```

Run tests:
```bash
python -m pytest tests/test_matching.py -v
```

---

## Matching Methods

| Method | Description | Speed | Quality |
|--------|-------------|-------|---------|
| `hungarian` | Optimal 1-to-1 with top-k pruning (default) | ★★★★ | ★★★★★ |
| `nearest_neighbor` | Top-K cosine NN (one-to-many) | ★★★★★ | ★★★ |
| `cosine_spatial` | Greedy argmax of fused score | ★★★★★ | ★★★ |
| `soft` | Soft probability matrix (softmax or Sinkhorn) | ★★★★ | ★★★★ |
| `cross_attention` | Scaled dot-product attention | ★★★★ | ★★★★ |
| `graph` | GNN cross-graph matching *(skeleton)* | — | — |

---

## Key Design Decisions

### 1. Top-K Pruning before Hungarian
Rather than running Hungarian on the full `N1×N2` cost matrix, each T1 token only considers its top-10 nearest T2 neighbours. This yields a **~10× speedup** with negligible quality loss.

### 2. Spatial Gating
Pairs with centroid distance > `spatial_gate_dist` (default: **0.3**) are hard-rejected before any matching. This removes physically impossible correspondences and reduces noise.

### 3. Similarity Statistics
Every pair output in `metadata` includes:
- `mean_cosine` — average cosine over valid (non-gated) pairs
- `std_cosine` — spread of cosine scores
- `gated_fraction` — fraction of pairs rejected by spatial gate

These are aggregated into `matching_report.json` for dataset-wide debugging.

### 4. Split / Merge Detection
After Hungarian matching, area ratios are compared:
- **Split**: one T1 token → multiple T2 tokens with `sum(area_T2) / area_T1 > threshold`
- **Merge**: multiple T1 tokens → one T2 token (symmetric check)

---

## Hyperparameters

| Parameter | Default | CLI flag |
|-----------|---------|----------|
| `alpha_cos` | 1.0 | `--alpha_cos` |
| `beta_geo` | 0.5 | `--beta_geo` |
| `spatial_gate_dist` | 0.3 | `--spatial_gate_dist` |
| `top_k` | 10 | `--top_k` |
| `hungarian_threshold` | 0.2 | `--hungarian_threshold` |
| `softmax_temp` | 0.1 | `--softmax_temp` |
| `split_area_ratio` | 0.6 | `--split_area_ratio` |
| `sinkhorn_iters` | 20 | `--sinkhorn_iters` |
| `device` | `cuda` | `--device` |
| `seed` | 42 | `--seed` |

---

## Output Structure

```
SECOND/matches/
    config.json               # Saved hyperparameters
    matching_report.json       # Dataset-wide metrics
    000003_matches.pt
    000011_matches.pt
    ...

SECOND/diagnostics/matches/
    000003_matches.png         # Centroid correspondence visualization
    000003_soft_heatmap.png    # Soft matching matrix heatmap
    ...
```

### Per-pair `.pt` format

```python
{
  "pairs":        [[i, j, score], ...],  # matched token pairs
  "soft_matrix":  Tensor[N1, N2],         # soft matching probabilities
  "unmatched_T1": [idx, ...],             # unmatched T1 token indices
  "unmatched_T2": [idx, ...],             # unmatched T2 token indices
  "metadata": {
    "n_t1":           int,
    "n_t2":           int,
    "n_matches":      int,
    "method":         str,
    "mean_cosine":    float,
    "std_cosine":     float,
    "gated_fraction": float,
    "splits":         [t1_idx, ...],
    "merges":         [t2_idx, ...],
    "runtime":        float,
  }
}
```

### `matching_report.json` format

```json
{
  "method": "hungarian",
  "total_pairs": 2968,
  "avg_matches_per_pair": 18.4,
  "unmatched_ratio": 0.21,
  "avg_mean_cosine": 0.412,
  "avg_std_cosine": 0.089,
  "avg_gated_fraction": 0.38,
  "avg_runtime_s": 0.034,
  "total_runtime_s": 100.9
}
```

---

## Note on Stage 4

Stage 3 is the **alignment layer** only. The novelty of the SCD pipeline lies in **Stage 4 — Token Change Reasoner** (MoE/transformer). The `soft_matrix` and `pairs` outputs from this stage are designed to serve as direct inputs to Stage 4.
