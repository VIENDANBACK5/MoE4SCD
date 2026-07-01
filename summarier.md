# Project Summary: Token-based Semantic Change Detection with Mixture of Experts (MoE)

**Last updated:** 2026-03-12

---

## 1. Context & Objective

Build an advanced **Semantic Change Detection (SCD)** pipeline for remote sensing images using the **SECOND dataset**. The system takes SAM2-tokenized region representations of image pairs (T1, T2) and predicts token-level changes, mapping them to semantic classes: Water, Soil/Impervious, Vegetation, Building, Farmland, Low Vegetation.

**Core research question:** Can MoE routing improve object-specific reasoning? Does **Semantic-guided routing** (injecting class labels) produce better expert specialization than **Dynamic routing** based solely on embedding features?

---

## 2. Dataset & Environment

| | |
|---|---|
| **Dataset** | SECOND (Remote Sensing SCD) |
| **Train pairs** | 2,968 image pairs (512×512) |
| **Test pairs** | 1,694 image pairs (512×512) |
| **Semantic classes** | 7: background, water, soil/impervious, vegetation, building, farmland, low_veg |
| **Python env** | `/home/chung/RS/.venv` |
| **SAM2 checkpoint** | `phase1/sam2/checkpoints/sam2.1_hiera_large.pt` |
| **Data root** | `phase1/SECOND/` |

---

## 3. Full Pipeline Architecture

```
Stage 1: SAM2 Encoder
  Images (T1, T2) → SAM2 Hiera-Large → embeddings (1,256,64,64) + high_res_feats
  Output: SECOND/embeddings_T1/, embeddings_T2/, highres_T1/, highres_T2/  (train + test)

Stage 2: Region Tokenization
  SAM2AutoMaskGenerator → masks @ 512×512 → downsample 8× →
  masked avg pool on image_embed → tokens (N,256), centroids (N,2), areas (N)
  Output: SECOND/tokens_T1/, tokens_T2/  (train + test)

Stage 3: Token Matching
  TokenMatcher (Hungarian + Top-K pruning + spatial gating) →
  per-pair match indices + confidence scores
  Output: SECOND/matches/  (2,968 *_matches.pt files)

Stage 4+: Token Change Reasoner (Transformer + GNN + MoE)
  tokens_T1/T2 + matches + semantic_labels →
  TokenEncoder → Transformer (4L×8H) → GraphReasoner → MoE Layer →
  ChangePredictionHead + DeltaHead
  Output: SECOND/stage4/, stage4B/, stage4B_v2/, stage4C/,
          stage5_6_dynamic/, stage5_6_semantic/
```

**Full current architecture:**
```
Tokens (T1+T2) → TokenEncoder (Linear + time_embed + pos_mlp + area_mlp)
              → TransformerReasoner (4 layers, 8 heads, d=384, ff=1536)
              → GraphReasoner (GNN k=6, 2 layers, hybrid edge weights, cross-time edges)
              → MoELayer (4 experts, expert_dim=512, Top-1 routing, residual)
              → ChangePredictionHead + DeltaHead
```

---

## 4. Stage Execution Results

### Stage 1 — SAM2 Feature Extraction (`extract_sam2_features.py`)
| Split | Pairs | Errors | Time |
|-------|-------|--------|------|
| train | 2,968 | 0 | ~16 min |
| test | 1,694 | 0 | ~9 min |
- Sample shape: `torch.Size([1, 256, 64, 64])`, dtype `float32`

### Stage 2 — Region Tokenization (`tokenize_regions.py`)
| Split | Pairs | Errors | Avg tokens/img | Time |
|-------|-------|--------|----------------|------|
| train | 2,968 | 0 | 99.4 | ~2h 13m |
| test | 1,694 | 0 | 91.8 | ~1h 13m |
- Note: harmless `UserWarning: cannot import name '_C' from 'sam2'` (skips C++ postprocessing only)

### Stage 3 — Token Matching (`token_matching.py`)
- 2,968 `*_matches.pt` files created + `config.json` + `matching_report.json`
- Method: Hungarian with Top-10 pruning + spatial gate (dist < 0.3)

---

## 5. Model Training Results

**Shared hyperparameters:** `token_dim=256, hidden_dim=384, num_layers=4, num_heads=8, ff_dim=1536, dropout=0.1, pos_mlp_hidden=64, area_mlp_hidden=64, delta_loss_weight=0.2, proxy_delta_threshold=9.56`, **30 epochs each**

| Model | Key Innovation | val_change (ep30) | val_delta | val_f1 | val_iou | Checkpoint |
|-------|----------------|:-----------------:|:---------:|:------:|:-------:|-----------|
| **stage4** | Baseline Transformer | 0.6912 | 0.4135 | — | — | `SECOND/stage4/` |
| **stage4B** | + GNN (k=6, 2L, residual) | 0.8051 | 0.5890 | — | — | `SECOND/stage4B/` |
| **stage4B_v2** | + weighted matching (α=0.6, β=0.4, γ=1.2) | 0.7912 | 0.5879 | — | — | `SECOND/stage4B_v2/` |
| **stage4C** | + MoE (4 experts, λ_balance=0.01, λ_entropy=0.001) | 0.7908 | 0.5567 | tracked | tracked | `SECOND/stage4C/` |
| **stage5_6_dynamic** | + router_v1 (dynamic routing) | 0.7879 | 0.4583 | 0.5469 | 0.3764 | `SECOND/stage5_6_dynamic/` |
| **stage5_6_semantic** | + router_v3 (semantic routing) | 0.8234 | 0.4585 | **0.5470** | **0.3764** | `SECOND/stage5_6_semantic/` |

**Best model: `stage5_6_semantic` — val_f1=0.547, val_iou=0.376**

Each model dir contains: `best_model.pt`, `checkpoint.pt`, `final_model.pt` (+ `config.json`, `training_log.csv`, `train_stdout.log`).

---

## 6. Expert Specialization Analysis (Stage 5)

### Stage 4C — Initial Specialization (30 epochs, validation subset ~51k tokens)

| Expert | Tokens | Load | Dominant Class | Purity | Change Ratio |
|--------|--------|------|----------------|--------|-------------|
| Expert 0 | 12,245 | 23.9% | low_veg | 0.820 | 0.46 |
| Expert 1 | 12,657 | 24.7% | low_veg | 0.820 | 0.24 |
| Expert 2 | 13,424 | 26.2% | low_veg | 0.777 | 0.25 |
| Expert 3 | 12,840 | 25.1% | low_veg | 0.805 | 0.45 |

**Key finding:** All 4 experts route ~80% `low_veg` tokens — purity is "strong" but only because the dataset itself is ~80% `low_veg`. No meaningful semantic specialization achieved.

### stage5_6_dynamic — Dynamic MoE (~590k tokens, full val set)

| Expert | Tokens | Load | Dominant Class | Purity | Change Ratio |
|--------|--------|------|----------------|--------|-------------|
| Expert 0 | 143,281 | 24.3% | low_veg | 0.810 | 0.44 |
| Expert 1 | 144,897 | 24.6% | low_veg | 0.838 | 0.20 |
| Expert 2 | 146,392 | 24.8% | low_veg | 0.808 | 0.20 |
| Expert 3 | 155,351 | 26.3% | low_veg | 0.797 | 0.44 |

### stage5_6_semantic — Semantic MoE (~590k tokens)

| Expert | Tokens | Load | Dominant Class | Purity | Change Ratio |
|--------|--------|------|----------------|--------|-------------|
| Expert 0 | 147,886 | 25.1% | low_veg | 0.831 | 0.21 |
| Expert 1 | 147,737 | 25.0% | low_veg | 0.796 | 0.24 |
| Expert 2 | 140,270 | 23.8% | low_veg | 0.817 | 0.43 |
| Expert 3 | 154,028 | 26.1% | low_veg | 0.808 | 0.43 |

**A/B comparison conclusion:** Semantic routing vs Dynamic routing shows nearly identical purity scores and F1/IoU. The `low_veg` class dominance (~80% of tokens) overwhelms router signals regardless of routing strategy. A notable structural finding: in both models, experts naturally polarize into two "change-sensitive" experts (change_ratio ~0.44) and two "stable-content" experts (change_ratio ~0.20).

---

## 7. Expert Diversity Collapse Test (stage5_6_dynamic)

Results from `stage4/analyze_expert_diversity.py` → `stage5/diversity/dynamic/`:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean Off-Diagonal Cosine Similarity | 0.224 | **STABLE** — no collapse |
| Mean Router Entropy | 0.062 | Very sharp (near one-hot) routing |
| Expert 0 change_prob | 0.408 | High sensitivity |
| Expert 1 change_prob | 0.197 | Low sensitivity |
| Expert 2 change_prob | 0.207 | Low sensitivity |
| Expert 3 change_prob | 0.408 | High sensitivity |

**Interpretation:** Experts are functionally distinct (sim=0.224, not collapsing) but the router is very sharp (entropy=0.062), meaning nearly all tokens get hard-routed to a single expert. Parameters diverged enough to produce different functions, but semantic specialization is blocked by class imbalance.

*(Diversity test for `stage5_6_semantic` not yet run — only `dynamic` analyzed.)*

---

## 8. Key Source Files

| File | Description |
|------|-------------|
| `extract_sam2_features.py` | Stage 1: SAM2 Hiera-Large feature extraction |
| `tokenize_regions.py` | Stage 2: AutoMaskGenerator + masked avg pool → tokens |
| `token_matching.py` | Stage 3: Hungarian matcher with Top-K pruning |
| `token_matching_utils.py` | Vectorized math: similarity, Sinkhorn, metrics |
| `token_change_reasoner.py` | `MatchDataset`, `build_batch`, `SampleData`, baseline model |
| `token_change_reasoner_graph.py` | `GraphReasoner` (GNN) component |
| `token_change_reasoner_moe.py` | `TokenChangeReasonerMoE`, `MoELayer`, `MoEConfig`, router v1/v2/v3 |
| `train_reasoner.py` | Main training loop: AdamW, AMP, cosine LR, CSV logs, checkpoints |
| `diagnostics.py` | Simple token dataset diagnostics (norms, coverage, viz) |
| `dataset_diagnostics.py` | Full 10-section diagnostics + K-Means + JSON report |
| `matching_diagnostics.py` | Stage 3 match quality diagnostics |
| `stage4/analyze_specialization.py` | Expert-class matrix + purity scoring |
| `stage4/analyze_expert_diversity.py` | Diversity collapse test (cosine sim, PCA, entropy, change sensitivity) |
| `stage4/visualize_expert_map.py` | Spatial visualization of expert assignments |

---

## 9. Technical Hurdles & Fixes

- **Label NameError**: `labels` variable in `MatchDataset` out of scope when parsing PNGs → fixed by ensuring proper variable persistence in the loop.
- **NumPy 2.x incompatibility**: Upgrading NumPy broke `seaborn`, `sklearn`, `pandas`. All analysis scripts rewritten to use pure `torch` + `PIL` — no sklearn/seaborn dependency anywhere.
- **Imbalance bottleneck (root cause)**: SECOND dataset is ~80% `low_veg` tokens. The load-balancing loss forces all experts to process `low_veg` proportionally, completely neutralizing semantic purity gains from routing. This is the primary ceiling on MoE effectiveness.

---

## 10. Results Summary & Next Steps

| Aspect | Finding |
|--------|---------|
| **Best F1** | 0.547 (stage5_6_semantic, ep 30) |
| **Best IoU** | 0.376 (stage5_6_semantic, ep 30) |
| **Expert specialization** | None (all experts dominated by low_veg ~80%) |
| **Router collapse** | No — experts are functionally distinct (sim=0.224) |
| **Semantic vs Dynamic routing** | No measurable difference (~0.0001 F1 gap) |
| **Root bottleneck** | Class imbalance (80% low_veg overwhelms balancing loss) |

**Potential next directions:**
1. **Imbalance-aware routing**: Remove load balancing penalty; instead use class-weighted expert loss or unbalanced routing (e.g., Top-2 without rebalancing).
2. **Longer training**: Both stage5_6 models trained only 30 epochs; F1 may not have converged.
3. **Test set evaluation**: Run `best_model.pt` of `stage5_6_semantic` on the 1,694 test pairs.
4. **Expert capacity tuning**: Increase `expert_dim` (currently 512) or use more experts for rare classes.
5. **Phase 2**: Use extracted embeddings/tokens for a downstream pixel-level decoder (e.g., UperNet or Mask2Former head).
