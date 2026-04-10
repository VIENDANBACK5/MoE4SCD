# Stage 4A — Token Change Reasoner

Stage 4A of the SCD pipeline: a Transformer-based model that predicts
which tokens (region proposals) represent semantic changes between T1 and T2.

---

## Architecture

```
Input (per image pair):
  tokens_T1 [N1, 256]  →  concat  →  [N, 256]
  tokens_T2 [N2, 256]  →            (N = N1 + N2)

         ┌─────────────────────────────────┐
         │         TokenEncoder            │
         │  Linear(256→384)                │
         │  + time_embed  (T1=0 / T2=1)   │
         │  + pos_mlp  (centroid x,y)      │
         │  + area_mlp (log area)          │
         └──────────┬──────────────────────┘
                    │ [N, 384]
         ┌──────────▼──────────────────────┐
         │   TransformerReasoner (4L×8H)   │
         │   d_model=384  ff_dim=1536      │
         │   Pre-LN, dropout=0.1           │
         └──────────┬──────────────────────┘
            [N, 384]│
          ┌─────────┴──────────┐
          ▼                    ▼
  ChangePredictionHead     DeltaHead
  Linear→GELU→Linear       Linear→GELU→Softplus
  [N] logits               [M_pairs] delta prediction
```

**Parameters**: ~6.0M (default config: hidden=384, 4 layers, 8 heads)

---

## Files

| File | Description |
|------|-------------|
| `token_change_reasoner.py` | Model: `TokenEncoder`, `TransformerReasoner`, `ChangePredictionHead`, `DeltaHead`, `ChangeReasonerModel`, `build_batch`, `training_step` |
| `train_reasoner.py` | Training loop: `MatchDataset`, AdamW, AMP, cosine LR, CSV log, checkpoints |
| `tests/test_reasoner.py` | 28 unit tests covering all modules |

---

## Quick Start

### Smoke test (model instantiation + forward)
```bash
python token_change_reasoner.py
```

### Run unit tests
```bash
pytest tests/test_reasoner.py -v
```

### Train on subset (50 pairs, 5 epochs)
```bash
python train_reasoner.py \
  --tokens_T1 SECOND/tokens_T1 \
  --tokens_T2 SECOND/tokens_T2 \
  --matches   SECOND/matches   \
  --output    SECOND/stage4    \
  --epochs 5 --n_samples 50 --batch_size 4 --device cuda
```

### Full training (2968 pairs, 30 epochs)
```bash
python train_reasoner.py \
  --tokens_T1 SECOND/tokens_T1 \
  --tokens_T2 SECOND/tokens_T2 \
  --matches   SECOND/matches   \
  --output    SECOND/stage4    \
  --epochs 30 --batch_size 8 --device cuda
```

---

## Loss Functions

```
change_loss = BCE(change_logits[valid], change_labels[valid])   pos_weight auto-computed
delta_loss  = MSE(delta_pred, ||emb_T2_j − emb_T1_i||)
total_loss  = change_loss + 0.2 × delta_loss
```

**Proxy labels** (when GT not available):
```
label[token] = 1 if ||emb_T2_j − emb_T1_i|| > 9.56  else 0
```
Threshold = mean + σ from Stage 3 diagnostics (17.6% labeled as changed).

---

## Tensor Shape Guide

| Tensor | Shape | Description |
|--------|-------|-------------|
| `tokens_pad` | `[B, N_max, 256]` | Raw SAM2 embeddings, padded |
| `time_ids_pad` | `[B, N_max]` | 0=T1, 1=T2 |
| `centroids_pad` | `[B, N_max, 2]` | Normalized (x,y) |
| `log_areas_pad` | `[B, N_max]` | log(area+1) |
| `padding_mask` | `[B, N_max]` bool | True = pad position |
| After TokenEncoder | `[B, N_max, 384]` | Fused representations |
| After Transformer | `[B, N_max, 384]` | Context-aware representations |
| `change_logits` | `[B, N_max]` | Raw change logits |
| `delta_pred` | `[M]` | Predicted ‖T2−T1‖ per matched pair |
| `delta_target` | `[M]` | True ‖emb_T2−emb_T1‖ |

N_max = max(N1_b + N2_b) over the batch.

---

## Output Files

```
SECOND/stage4/
  config.json          # ReasonerConfig
  best_model.pt        # checkpoint with lowest val_loss
  final_model.pt       # last epoch checkpoint
  checkpoint.pt        # periodic checkpoint
  training_log.csv     # epoch losses
```

---

## Training Setup

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-2 |
| LR schedule | CosineAnnealingLR |
| Mixed precision | ✅ (torch.amp) |
| Grad clipping | max_norm=1.0 |
| Val split | 10% |

---

## Future Extensions (Stage 4B+)

1. **MoE FFN** — replace TransformerReasoner FFN with Mixture of Experts (top-k=2)
2. **Cross-attention T1↔T2** — bipartite attention using Stage 3 match topology
3. **GNN** — graph neural network over token adjacency graph
4. **GT labels** — plug in SECOND mask labels via `--labels SECOND/labels/`
5. **Pixel refinement** — upsample token predictions back to pixel masks via SAM2 decoder
