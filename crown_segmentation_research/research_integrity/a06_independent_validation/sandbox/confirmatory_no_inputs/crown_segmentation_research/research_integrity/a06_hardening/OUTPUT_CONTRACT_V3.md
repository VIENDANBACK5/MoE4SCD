# TreeFlowNet Output Contract V3

**Version:** 3.0.0  
**Effective Date:** 2026-10-15  
**Governing Principle:** Architectural neutrality with strict semantic alignment across stages.

---

## 1. Four-Stage Output Hierarchy

To allow scientifically fair comparison between radically different instance segmentation paradigms—Field Decoders (F), Relational Basin Graph Partitioners (G), and Query-based Set Transformers (Q)—all model outputs are structured into four strictly delineated stages:

```
[Stage 0: Model-Native] ──> [Stage 1: Materialized Proposals] ──> [Stage 2: Accepted Instances] ──> [Stage 3: Scene Instances]
```

---

### Stage 0 — Model-Native Outputs
Architecture-specific raw tensor outputs directly produced by the neural network head:
- **Paradigm F (Flow / Field)**:
  - Vector flow field $\vec{v}(x, y) \in \mathbb{R}^2$
  - Truncated Signed Distance Transform (SDT) $s(x, y) \in \mathbb{R}$
  - Centroid heatmaps $c(x, y) \in [0, 1]$
  - Canopy boundary / seam logit maps
- **Paradigm G (Basin Graph / Relational)**:
  - Base basin segmentation map $\mathcal{B}(x, y)$
  - Basin node feature embeddings
  - Pairwise adjacency relation logits / affinity matrix $\mathcal{A}_{ij} \in \mathbb{R}$
- **Paradigm Q (Transformer Queries)**:
  - Raw query mask logits $M_q(x, y) \in \mathbb{R}^{Q \times H \times W}$
  - Raw query objectness / class logits $o_q \in \mathbb{R}^Q$
  - No-object prediction probabilities

*Invariant:* Stage 0 outputs are purely internal representations. No cross-paradigm instance metrics (such as COCO AP or 1-to-1 matching) may be evaluated at Stage 0.

---

### Stage 1 — Materialized Proposals
Explicit geometric candidate instances generated under a documented, deterministic materialization function:
- **Paradigm F**: Connected component / watershed basins extracted from flow streamlines or centroid seeds before score pruning.
- **Paradigm G**: Connected subgraph clusters or merged basin candidates before confidence gating.
- **Paradigm Q**: Active query mask slices where query $q$ is materialized if and only if its objectness score exceeds an exploratory threshold $\tau_{mat}$ (e.g. $\tau_{mat} = 0.05$). Unmatched or background queries MUST NOT be materialized as instances.

*Invariant:* Every materialized proposal must have:
1. Binary spatial mask $m_k \in \{0, 1\}^{H \times W}$.
2. Proposal scalar confidence score $s_k \in [0, 1]$.
3. Traced provenance back to Stage 0 tensors (e.g., source query index, seed basin ID).

---

### Stage 2 — Accepted Instances (Primary Comparative Benchmark)
The primary evaluation level for comparable instance segmentation metrics. Predictions after applying a frozen, predeclared selection policy:
1. **Confidence Thresholding**: $s_k \ge \tau_{accept}$ (fixed per model or selected on validation fold).
2. **Duplicate & Overlap Resolution**: Documented Non-Maximum Suppression (NMS) or argmax pixel assignment if overlapping masks are resolved.
3. **Validity & Area Filtering**: Rejection of masks with $A_k^{m^2} < A_{min}$ (default $0.25\text{ m}^2$) or masks falling entirely outside valid support $V$.
4. **Permutation Invariance**: The set of accepted instances $\mathcal{I} = \{(m_1, s_1), \dots, (m_N, s_N)\}$ must produce identical evaluation metrics under arbitrary reordering of instance indices.

*Invariant:* All end-to-end instance metrics (COCO AP, Cardinality-First 1-to-1 Recall, Count MAE) MUST be evaluated strictly on Stage 2 outputs.

---

### Stage 3 — Scene-Level Instances
Global instances formed after tile stitching and cross-tile spatial association:
1. **Boundary Stitching**: Merging truncated crown fragments along tile edges using spatial proximity and feature embedding similarity.
2. **Global Coordinate Mapping**: Masks transformed into projected CRS coordinates (`EPSG:32618`).
3. **Global ID Deduplication**: Stable unique identifier assigned to each reconstructed tree in the orthomosaic.
4. **Tile-Origin Invariance**: Reconstructed scene crown boundaries must remain stable against arbitrary sub-tile grid origin shifts.

---

## 2. Inactive Query Policy (Q-Specific Guardrail)

In transformer query architectures (e.g., Mask2Former, DETR-style heads):
- Unmatched queries during training receive no positive mask loss.
- In zero-crown or low-density crops, unused queries produce high-entropy or background mask logits.
- **Rule:** Unmatched / inactive query slots MUST NOT be passed into structural grouping or instance matching as accepted detections. Treating raw inactive queries as false positives without objectness thresholding corrupts scientific comparison against F and G.
