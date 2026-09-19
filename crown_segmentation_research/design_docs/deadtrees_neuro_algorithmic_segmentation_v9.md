# Neuro-Algorithmic DeadTrees Segmentation (Method Design v9)
## Systematic Research Blueprint: Solving Irregular, Fallen, and Skeletal Deadwood Delineation via $\varepsilon$-Kernels, Medial Axis Bellman-Ford, and Spectral Modularity

**Target Task:** Track 2 — High-Fidelity Segmentation on the **DeadTrees** Dataset  
**Date:** September 2026  
**Theoretical Foundation:** Prof. Yusu Wang's Neuro-Algorithmic Paradigm, Algorithmic Alignment, Computational Geometry $\varepsilon$-Kernels (NeurIPS 2025), and Spectral Modularity Graph Theory  
**Cognitive Framework:** Applied `/creative-thinking-for-research` (8 Empirical Cognitive Frameworks)

---

## 1. Context & The Core DeadTrees Segmentation Bottleneck

### 1.1 Why Standard Star-Convex (StarDist) Fails on DeadTrees
While Star-Convex architectures (v1–v8) were designed for rounded, compact living tree crowns (e.g., BAM), the **DeadTrees** dataset presents fundamentally different geometric topologies:
1. **Fallen Deadwood (High Aspect Ratio)**: Fallen tree trunks form elongated, thin linear structures (aspect ratio $L/W > 10:1$, width often only 2–5 pixels).
   * *StarDist Failure*: Casting $K=16$ or $32$ equi-angular radial rays around a centroid wastes $>85\%$ of the rays shooting into empty surrounding forest, while failing to reach the far longitudinal ends.
2. **Standing Dead Snags (Skeletal & Branching)**: Defoliated dead trees consist of bare, protruding branches without a solid dome canopy.
   * *StarDist Failure*: Violates star-convexity; rays intersect empty air or live canopy understory between branches.
3. **Crisscrossing & Tangled Deadwood ("Pick-up Sticks")**: Storm damage and bark beetle infestations produce overlapping, crisscrossing fallen logs.
   * *Centroid & NMS Failure*: The geometric centroid of two crossed logs often lies at their intersection or completely outside the individual trunks, causing NMS and embedding heads to either collapse them into a single blob or miss them entirely.
4. **Extreme Class Imbalance**: Deadwood occupies $<2\%$ of total pixels in 1024x1024 global aerial tiles, surrounded by live canopy, rocks, bare soil, and harsh shadow artifacts.

---

## 2. Creative Thinking Frameworks Applied to DeadTrees (`/creative-thinking-for-research`)

```
               ┌─────────────────────────────────────────────────────────────┐
               │              DEADTREES SEGMENTATION CHALLENGES              │
               │   • High aspect ratio (fallen logs: L/W > 10:1)             │
               │   • Non-convex skeletal snags (bare branches)               │
               │   • Tangled crisscrossing logs & 2% class imbalance         │
               └──────────────────────────────┬──────────────────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
     ┌──────────────────────────────────┐            ┌──────────────────────────────────┐
     │ 1. DIRECTIONAL EXTENT ε-KERNEL   │            │ 2. SKELETON BELLMAN-FORD GNN     │
     │    (NeurIPS 2025 SumFormer)      │            │    (Medial Axis Dynamic Prog)    │
     ├──────────────────────────────────┤            ├──────────────────────────────────┤
     │ • Replaces radial equiangular ray│            │ • Traces fallen trunks & branches│
     │ • Learns directional width along │            │ • Provable length extrapolation  │
     │   principal axes (u_∥, u_⊥)      │            │ • End-to-end contour continuity  │
     │ • Bounded complexity O(N)        │            │ • Resilient to shadow gaps       │
     └──────────────────────────────────┘            └──────────────────────────────────┘
                                              │
                                              ▼
                             ┌──────────────────────────────────┐
                             │ 3. SPECTRAL MODULARITY PARTITION │
                             │    (Unsupervised Cluster Cut)    │
                             ├──────────────────────────────────┤
                             │ • Disentangles crisscrossing logs│
                             │ • Rayleigh Quotient Optimization │
                             │ • Zero centroid-peak dependence  │
                             └──────────────────────────────────┘
```

### Applying the 8 Cognitive Frameworks:

| Framework | Cognitive Move | DeadTrees Research Innovation |
| :--- | :--- | :--- |
| **F1: Bisociation (Combinatorial)** | $\varepsilon$-Kernel Directional Core-sets $\times$ Remote Sensing Deadwood | Representing fallen deadwood as **Minimum Enclosing Directional Polytope / Cylinder Core-sets** rather than pixel masks or radial rays. |
| **F2: Problem Reformulation** | "Detect Center $\to$ Cast Rays" $\to$ "Trace Medial Axis $\to$ Expand Extents" | Shifting representation from centroid-radial coordinates $(r, \theta)$ to **Medial Axis Skeleton + Transverse Extent ($L_\parallel, W_\perp, \phi$)**. |
| **F3: Analogical Reasoning** | Shortest Path / Bellman-Ford on Graphs $\leftrightarrow$ Fallen Log Tracing | Modeling the longitudinal spine of a fallen tree as a shortest path on the image gradient grid, solved via **Algorithmic-Aligned GNN**. |
| **F4: Constraint Manipulation** | Drop Star-Convexity & Drop Fixed Ray Counts | Replacing equiangular rays with **Anisotropic Principal Support Functions** ($u_1, \dots, u_K$ adapted to object orientation). |
| **F5: Negation & Inversion** | "Deadwood requires single-point peak detection" $\to$ **FALSE** | Detecting deadwood by its extreme endpoints and orientation field, spanning the $\varepsilon$-kernel between endpoints. |
| **F6: Abstraction Laddering** | Unifying Standing Snags, Fallen Trunks & Branching Debris | A unified 3-parameter geometric formulation: **$\text{Crown}_{\text{dead}} = \text{Skeleton}(E) \oplus \text{Envelope}_{\varepsilon}(u)$**. |
| **F7: Adjacent Possible** | NeurIPS 2025 SumFormer + SDT (Signed Distance Transform) + SAM2 Feats | Combining SAM2 zero-shot high-res feature grids with Linear SumFormer extent pooling. |
| **F8: Janusian Thinking** | Reconciling Thin Skeletal Precision vs. Global Environmental Context | Dual-path network: Local High-Frequency Skeleton Head + Global Spectral Modularity Graph. |

---

## 3. Mathematical Formulations for DeadTrees Segmentation

### 3.1 Directional Extent $\varepsilon$-Kernel Support Formulation (NeurIPS 2025)
For any deadwood instance $P \subset \mathbb{R}^2$ (whether a straight fallen log or a branched snag):
1. **Directional Support Function**:
   $$h_P(u) = \max_{p \in P} \langle p, u \rangle, \quad \forall u \in \mathbb{S}^1$$
2. **Principal Anisotropic Decomposition**:
   Instead of uniform angles, the network predicts the principal longitudinal orientation $\phi \in [-\frac{\pi}{2}, \frac{\pi}{2}]$ and evaluates extents along an adaptive directional basis $\{u_\parallel, u_\perp\}$:
   $$u_\parallel = (\cos \phi, \sin \phi), \quad u_\perp = (-\sin \phi, \cos \phi)$$
3. **$\varepsilon$-Kernel Support Loss**:
   $$\mathcal{L}_{\varepsilon\text{-deadwood}} = \frac{1}{K} \sum_{k=1}^K \left| \hat{h}_P(u_k) - h_P^{\text{GT}}(u_k) \right|_1 + \lambda_{\text{aspect}} \left| \frac{\hat{h}(u_\parallel)}{\hat{h}(u_\perp)} - \frac{h^{\text{GT}}(u_\parallel)}{h^{\text{GT}}(u_\perp)} \right|$$

### 3.2 Medial Axis Bellman-Ford Relaxation Layer
To ensure long fallen logs are not fragmented by intervening canopy shadows:
1. Define the graph over deadwood candidate pixels where edge weights $w_{ij}$ represent inverse continuity:
   $$w_{ij} = \text{Softplus}\left( \mathbf{W}_e \cdot [f_i, f_j, \|\Delta c_{ij}\|, |\Delta \phi_{ij}|] \right)$$
2. Perform $T$ steps of Bellman-Ford dynamic programming relaxation to connect log endpoints:
   $$d_j^{(t)} = \min\left( d_j^{(t-1)}, \min_{i \in \mathcal{N}(j)} (d_i^{(t-1)} + w_{ij}) \right)$$
3. **Sparsity Regularizer**:
   $$\mathcal{L}_{\text{sparse}} = \frac{1}{|\mathcal{E}|} \sum_{(i,j) \in \mathcal{E}} |w_{ij}|$$

### 3.3 Spectral Modularity Cut for Crisscrossing Deadwood
For entangled wood piles, the modularity matrix $B = A - \frac{d d^T}{2m}$ on the local spatial-spectral graph partitions intersecting trunks:
$$\mathcal{L}_{\text{modularity}} = - \frac{1}{2m} \text{Tr}\left( S^T B S \right)$$

---

## 4. PyTorch Architectural Implementation Blueprints

### 4.1 `DeadwoodDirectionalExtentHead` (Adaptive Anisotropic $\varepsilon$-Kernel)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeadwoodDirectionalExtentHead(nn.Module):
    """
    Anisotropic eps-Kernel extent head tailored for fallen deadwood (high aspect ratio)
    and skeletal standing snags.
    """
    def __init__(self, in_channels=256, num_support_directions=16):
        super().__init__()
        self.num_directions = num_support_directions
        
        # Orientation & Anisotropy Head (predicts angle phi and aspect ratio)
        self.orient_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1) # [sin(2*phi), cos(2*phi)]
        )
        
        # Directional Support Processor (SumFormer Linear Attention)
        self.support_proj = nn.Sequential(
            nn.Linear(in_channels + 4, 128), # feat + x + y + x^2 + y^2
            nn.GELU(),
            nn.Linear(128, num_support_directions)
        )

    def forward(self, feat_map, coords_grid, deadwood_mask):
        """
        feat_map: (B, C, H, W)
        coords_grid: (B, H, W, 2)
        deadwood_mask: (B, 1, H, W) soft probability of deadwood
        """
        B, C, H, W = feat_map.shape
        N = H * W
        
        # 1. Predict local orientation vector field
        orient_logits = self.orient_head(feat_map) # (B, 2, H, W)
        sin_2phi = orient_logits[:, 0]
        cos_2phi = orient_logits[:, 1]
        phi = 0.5 * torch.atan2(sin_2phi, cos_2phi) # (B, H, W)
        
        # 2. Construct Paraboloid Lifting Coordinates
        flat_coords = coords_grid.view(B, N, 2)
        r_sq = torch.sum(flat_coords ** 2, dim=-1, keepdim=True) # (B, N, 1)
        lifted_coords = torch.cat([flat_coords, r_sq], dim=-1) # (B, N, 3)
        
        flat_feats = feat_map.view(B, C, N).transpose(1, 2) # (B, N, C)
        flat_mask = deadwood_mask.view(B, N, 1) # (B, N, 1)
        
        x_in = torch.cat([flat_feats, flat_coords, lifted_coords], dim=-1) # (B, N, C+5)
        support_logits = self.support_proj(x_in) # (B, N, K)
        
        # Masked Linear SumFormer Attention
        masked_logits = support_logits + torch.log(flat_mask.clamp(min=1e-6))
        attn_weights = F.softmax(masked_logits, dim=1) # (B, N, K)
        
        # Extreme coreset points Q_k
        coreset_points = torch.einsum('bnk,bnd->bkd', attn_weights, flat_coords) # (B, K, 2)
        
        return {
            "orientation": phi,
            "coreset_points": coreset_points,
            "attn_weights": attn_weights
        }
```

---

## 5. Concrete Action Plan & Experimental Roadmap for DeadTrees

### Phase 1: DeadTrees Geometric Representation Benchmark
1. **Benchmark StarDist (v8) vs. Anisotropic $\varepsilon$-Kernel on DeadTrees**:
   * Measure IoU and boundary Hausdorff distance on fallen logs ($L/W > 5$) vs. standing snags.
   * Verify elimination of "ray leakage" into surrounding live canopy.

### Phase 2: Full DeadTrees Training (181 Sites / Global Aerial Tiles)
1. Train with composite objective:
   $$\mathcal{L} = \mathcal{L}_{\text{deadwood\_focal}} + \alpha \mathcal{L}_{\varepsilon\text{-deadwood}} + \beta \mathcal{L}_{\text{modularity}} + \gamma \mathcal{L}_{\text{sparse}}$$
2. Target Metrics on held-out test site 5737:
   * **F1@IoU0.5 $\ge 0.65$** (surpassing existing scratch best of ~0.42).
   * Precision $\ge 0.70$, Recall $\ge 0.62$.
   * Elimination of fallen trunk fragmentation.

### Phase 3: Cross-Sensor & Scale Invariance Verification
1. Test on varying Ground Sampling Distances (GSD from 0.05m to 0.3m).
2. Validate bounded complexity and linear inference speed on full 1024x1024 tiles without memory overflow.
