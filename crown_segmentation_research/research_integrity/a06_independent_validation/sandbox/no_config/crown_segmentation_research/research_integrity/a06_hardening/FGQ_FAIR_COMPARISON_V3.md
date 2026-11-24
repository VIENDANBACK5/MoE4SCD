# TreeFlowNet F–G–Q Fair Comparison Contract V3

**Version:** 3.0.0  
**Effective Date:** 2026-10-15  
**Governing Standard:** Equal scientific questioning, explicit resource accounting, and prevention of architectural bias.

---

## 1. Dual Study Architecture

To avoid confounding head mechanism with disparate tuning budgets, all experimental evaluations comparing F, G, and Q must explicitly declare whether they belong to Study Type 1 or Study Type 2:

### Study Type 1: Controlled Mechanism Comparison
- **Scientific Question:** Under identical frozen representation features and identical spatial receptive fields, how do flow streamlines (F), graph relations (G), and cross-attention queries (Q) compare in assembling crown instances?
- **Requirements:**
  1. Shared identical backbone weights (e.g., Swin-T/FPN P2 checkpoint frozen at epoch 25, seed 42).
  2. Identical input crops, augmentations, and random seeds.
  3. Fixed training update budget (e.g., 400 updates on identical batches).
  4. Explicit parameter accounting (logging trainable head parameters: F ~150k, G ~180k, Q ~257k).

### Study Type 2: Practical System Comparison
- **Scientific Question:** When each complete paradigm (encoder + decoder + post-processing) is independently trained and tuned to its optimal capacity, what is the best achievable instance segmentation quality on uncropped scenes?
- **Requirements:**
  1. Disclosed total compute budget (GPU hours, batch size, epochs).
  2. Predeclared hyperparameter tuning protocol on the `DEVELOPMENT` split.
  3. Explicit inference latency, VRAM consumption, and FLOPs reported per 1024×1024 tile.

---

## 2. Paradigm-Specific Formulations

### Paradigm F (Field Decoder)
- **Mechanisms:**
  - Dense direction vectors $\vec{v}(x, y)$ pointing to crown topological medoid/centroid.
  - Truncated Signed Distance Transform (SDT) providing boundary repelling force.
  - Centroid peak detection followed by streamline tracking or Hamiltonian watershed.
- **Inference Parameterization:** Decouple candidate seed threshold $\tau_{seed}$ from mask boundary threshold $\tau_{boundary}$.

### Paradigm G (Relational Graph Partitioning)
- **Sub-Typing Requirements:**
  - **G0 (Merge-Only Grouping):** Takes over-segmented base basins (e.g., watershed from high-frequency gradient) and predicts binary merge probabilities $P(B_i \cup B_j)$.
    - *Theoretical Bound:* G0 CANNOT split an under-segmented basin containing multiple tree crowns ($N_{GT} > 1$). Empirical failure of G0 (EXP020/021) is an intrinsic property of merge-only operators, NOT evidence against graph-based partitioning.
  - **G1 (Cut-and-Merge / Repartitioning):** Operates with joint split-and-merge or multicut formulation capable of both partitioning merged basins and joining fragmented crowns.
  - **Scientific Rule:** Any paper claim citing G performance must explicitly state whether G0 or G1 was evaluated.

### Paradigm Q (Transformer Set Prediction)
- **Mechanisms:**
  - Fixed query bank $Q \in \{16, 32, 64, 100\}$.
  - Hungarian bipartite matching between active queries and ground-truth crowns.
  - Combined loss: $\mathcal{L} = \lambda_{bce} \mathcal{L}_{bce} + \lambda_{dice} \mathcal{L}_{dice} + \lambda_{obj} \mathcal{L}_{obj}$.
- **Fairness Rules:**
  1. Query capacity $Q$ must strictly exceed the maximum expected crowns in the training crop field of view ($Q \ge \max N_{GT}^{crop}$).
  2. Training crops exceeding $Q$ must trigger an explicit capacity overflow warning.
  3. Accepted instance selection must apply a fixed objectness threshold ($\tau_{obj} \ge 0.5$) before evaluating instance metrics.

---

## 3. Comparison Parameter Freeze Table

The following configuration parameters must be frozen and logged in all comparative benchmark manifests:

| Parameter | Specification | Verification Method |
|---|---|---|
| Encoder Backbone | Swin-T with FPN (stride 4, P2 feature width 256) | SHA256 checksum of checkpoint |
| Input Resolution | $1024 \times 1024$ (train), full tile $H \times W$ (eval) | Tensor shape assertion |
| Physical GSD | $0.03\text{ m}$ per pixel ($30.72\text{ m} \times 30.72\text{ m}$ FOV) | Affine transform check |
| Coordinate System | `EPSG:32618` | GDAL / Rasterio metadata check |
| Ground-Truth Matching | 1-to-1 Cardinality-First IoU $\ge 0.5$ | Synthetic fixture test |
| Output Stage | Stage 2 (Accepted Instances) | Output Contract V3 test |
| Random Seeds | Master seed `20261015`, Seeds `{42, 43, 44}` | Deterministic RNG test |
| Evaluation Denominator | Deduplicated valid ground area (m² or hectares) | Valid mask union check |
