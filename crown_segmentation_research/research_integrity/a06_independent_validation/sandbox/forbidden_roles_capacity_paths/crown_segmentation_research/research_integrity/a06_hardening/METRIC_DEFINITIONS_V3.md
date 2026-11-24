# TreeFlowNet Metric Definitions & Evaluation Standards V3

**Version:** 3.0.0  
**Effective Date:** 2026-10-15  
**Governing Standard:** Mathematical rigor, metric gamification defense, and separation of geometry from confidence.

---

## 1. Instance Matching Protocols

### 1.1. Cardinality-First 1-to-1 Matching
Let $\mathcal{G} = \{g_1, \dots, g_M\}$ be the ground-truth instance masks and $\mathcal{P} = \{p_1, \dots, p_N\}$ be the Stage 2 accepted predicted instance masks.  
Construct the bipartite matching cost/affinity matrix $\mathbf{C} \in [0, 1]^{M \times N}$, where:
$$C_{ij} = \text{IoU}(g_i, p_j) = \frac{|g_i \cap p_j|}{|g_i \cup p_j|}$$

A candidate edge $(i, j)$ is eligible if and only if $C_{ij} \ge \tau_{match}$ (canonical threshold $\tau_{match} = 0.5$).
1. **Primary Objective (Max Cardinality)**: Find a matching $\pi \subset \{1, \dots, M\} \times \{1, \dots, N\}$ that maximizes the total number of matched pairs:
   $$\max |\pi| \quad \text{subject to } \forall (i, j) \in \pi, C_{ij} \ge 0.5$$
2. **Secondary Tiebreaker (Max Total IoU)**: Among all matchings achieving maximum cardinality, select the one maximizing total overlap:
   $$\max \sum_{(i, j) \in \pi} C_{ij}$$

*Implementation Note:* Solved via Maximum Weight Bipartite Matching with edge weights $W_{ij} = \mathbf{1}_{C_{ij} \ge 0.5} \cdot (1000 + C_{ij})$.

---

## 2. Standard Detection & Instance Metrics

1. **Precision & Recall**:
   $$\text{Precision} = \frac{|\pi|}{N}, \quad \text{Recall} = \frac{|\pi|}{M}$$
2. **COCO AP Metrics**:
   - $\text{AP}_{50:95}$: Average Precision averaged across 10 IoU thresholds from 0.50 to 0.95 with step 0.05.
   - Size-stratified AP evaluated using canonical physical area $A_i^{m^2}$:
     - $\text{AP}_{<1m^2}$, $\text{AP}_{1-2m^2}$, $\text{AP}_{2-4m^2}$, $\text{AP}_{4-9m^2}$, $\text{AP}_{\ge 9m^2}$.
3. **Count Bias & Absolute Error**:
   $$\text{Signed Bias} = \frac{N - M}{\text{Area}_{ha}}, \quad \text{Count MAE} = \frac{|N - M|}{\text{Area}_{ha}}$$
   where $\text{Area}_{ha}$ is the deduplicated valid ground area in hectares.

---

## 3. Structural vs. End-to-End Grouping Metrics

Grouping evaluates whether a model correctly individuates different points within the forest canopy.
Anchor points $a_k \in U_{g(k)}^{valid}$ are sampled from the unique valid ground truth of tree $g(k)$ with erosion radius $r \in \{1\text{ px}, 4\text{ px}, \text{adaptive}\}$.
Pairs of anchors $(a_u, a_v)$ are formed within distance bins $d(a_u, a_v) \in [0, 0.25, 0.5, 1, 2, 4, \infty)\text{ m}$.
Let $y_{uv} = \mathbf{1}_{g(u) = g(v)}$ indicate whether the anchors belong to the same biological tree.

### 3.1. Unique Coverage Rule
A prediction mask set assigns instance ID $\hat{I}(a)$ to anchor $a$ if and only if:
$$\sum_{j=1}^N p_j(a) = 1 \implies \hat{I}(a) = \arg\max_j p_j(a)$$
- If $\sum_j p_j(a) = 0$: **Under-coverage (Background Abstention)**.
- If $\sum_j p_j(a) > 1$: **Ambiguous Overlap (Conflict Abstention)**.
- Both cases assign $\hat{I}(a) = 0$.

### 3.2. Structural vs. E2E Definitions
- **Structural Grouping**: Evaluated strictly on the frozen mask geometry without confidence re-ranking or score thresholds.
- **End-to-End (E2E) Grouping**: Evaluated on Stage 2 accepted masks after applying confidence threshold $\tau_{accept}$.
- **Metrics Reported**:
  - $\text{Same-Tree Accuracy} = P(\hat{I}(a_u) = \hat{I}(a_v) \ne 0 \mid y_{uv} = 1)$
  - $\text{Different-Tree Accuracy} = P(\hat{I}(a_u) \ne \hat{I}(a_v) \text{ and } \hat{I}(a_u) \ne 0 \text{ and } \hat{I}(a_v) \ne 0 \mid y_{uv} = 0)$
  - $\text{Balanced Grouping} = \frac{1}{2}(\text{Same Accuracy} + \text{Different Accuracy})$
  - $\text{Pair Coverage} = P(\hat{I}(a_u) \ne 0 \text{ and } \hat{I}(a_v) \ne 0)$

---

## 4. Defense Against Metric Gamification

| Gamification Strategy | Trivial Attack Behavior | Counter-Metric Guardrail |
|---|---|---|
| **Empty Prediction** | Predicts $N=0$ instances everywhere. | Same-Tree Acc = 0%, Recall = 0, Coverage = 0%. |
| **All-Merge** | Single giant mask covering the entire forest tile. | Different-Tree Acc = 0%, AP = 0, FP rate penalized. |
| **Duplicate Flooding** | Duplicating masks 100 times to increase chance of IoU overlap. | Ambiguous Overlap Rule causes structural abstention; NMS penalizes precision. |
| **Score Gaming** | Artificially shifting all confidence scores to 0.99. | Structural grouping is mathematically invariant to score shifts. |
| **Excluding Hard Crowns** | Omitting small crowns ($<1\text{ m}^2$) from the denominator. | Mandatory joint matching of ALL ground truth before stratum partitioning. |
