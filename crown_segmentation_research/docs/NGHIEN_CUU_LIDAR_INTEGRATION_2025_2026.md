# 🛰️ DEEP RESEARCH × CREATIVE THINKING: LiDAR Integration for Tree Crown Segmentation

> **Ngày:** 2026-09-26 | **Mode:** Full Research + Creative Thinking
> **Gợi ý từ:** Thầy Phương & Anh Khương — "incorporate LiDAR in your segmentation"
> **Pipeline:** 13-agent deep-research + 8-framework creative-thinking

---

## 📋 EXECUTIVE SUMMARY

LiDAR (Light Detection and Ranging) — cụ thể là USGS 3DEP ALS (Airborne Laser Scanning) — có thể đóng vai trò **chiến lược kép** trong dự án phân đoạn tán lá cây:

1. **Path A — Pseudo-label Generator (Tier 1):** Dùng LiDAR-derived CHM → ITS → raster masks làm training labels tự động, giảm gánh nặng annotation thủ công.
2. **Path B — Independent Validator (Tier 2):** Dùng CHM-based crown polygons kiểm tra chất lượng dự đoán của TreeFlowNet trên vùng không có manual labels.
3. **Path C — Boundary Refinement Signal (Tier 3 — NOVEL):** Dùng CHM gradient làm prior cho boundary head của TreeFlowNet-HR, giống cách BalSAM dùng DSM làm additional channel.

**Bottom line:** Anh Khương đã đúng — LiDAR không nên là primary RGB-training input — mà là *evidence layer* bổ sung. Với pipeline 2025-2026, Path A + C kết hợp tạo ra **cơ hội đặc biệt mạnh**.

---

## PHASE 1: SCOPING

### Research Questions (FINER-scored)

**Primary RQ:**
> Làm thế nào để tích hợp LiDAR point cloud data (USGS 3DEP) vào pipeline huấn luyện TreeFlowNet (RGB raster) — từ `.laz` → CHM → ITS masks → pseudo-label hoặc boundary prior — theo cách nào hiệu quả nhất về chi phí/chất lượng?

**Sub-questions:**
- **SQ1:** Thuật toán ITS nào (Li2012, Dalponte2016, FG-TreeSeg) cho boundary quality tốt nhất với ALS data mật độ thấp (3DEP: 1-8 pts/m²)?
- **SQ2:** Pipeline `.laz → CHM → polygon masks → raster labels` có thể tự động hóa hoàn toàn bằng Python không?
- **SQ3:** CHM gradient (∇CHM) có thể làm boundary prior cho TreeFlowNet-HR boundary head không?
- **SQ4:** Chiến lược semi-supervised nào (LiDAR pseudo-label + SAM2 refinement) tốt nhất?

**FINER Score:**

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Feasible | 9/10 | USGS 3DEP free, lidR/PDAL mature, Python ecosystem đủ |
| Interesting | 10/10 | Mentor confirmation + multi-source label generation |
| Novel | 8/10 | LiDAR→RGB pseudo-label + flow field bridge còn ít paper |
| Ethical | 10/10 | Public data, no human subjects |
| Relevant | 10/10 | Trực tiếp giải quyết boundary bottleneck của TreeFlowNet |

---

## PHASE 2: LITERATURE INVESTIGATION

### Tier 1 — Core Papers (APA 7.0)

**1. FG-TreeSeg (Chen et al., 2025) — ⭐ ISOMORPHIC VỚI TREEFLOWNET**
- **WHY:** Cần segmentation tán lá mà không cần instance annotations
- **HOW:** SegFormer binary mask → Cellpose-SAM flow field → sinks delineation
- **WHAT:** Competitive với supervised baselines trên NEON + BAMFOREST datasets
- **Relevance:** Flow field approach ISOMORPHIC với TreeFlowNet — **paper closest match!**
- Source: arXiv 2025; GitHub: University of South Carolina & Virginia Tech

**2. "Learning Image-based Tree Crown Segmentation from Enhanced LiDAR-based Pseudo-labels" (2026)**
- **WHY:** Thiếu annotation → dùng LiDAR làm pseudo-label source
- **HOW:** ALS-based ITS → SAM2 refinement → train RGB model
- **WHAT:** SAM2-refined pseudo-labels significantly outperform raw LiDAR masks
- **Relevance:** Direct blueprint cho Path A | arXiv:2602.13022

**3. BalSAM (2025) — CHM as input channel**
- **WHY:** RGB alone insufficient trong dense canopy; LiDAR gives 3D structure
- **HOW:** DSM/CHM as additional input channel to SAM-based model
- **WHAT:** Generalizes across boreal, temperate, tropical forests
- **Relevance:** Path C — CHM channel cho TreeFlowNet-HR

**4. Dalponte & Coomes (2016) — Classic ITS baseline**
- CHM-based seeded region growing; high accuracy in mature forests
- Methods in Ecology and Evolution, 7(10), 1236-1245

**5. Li et al. (2012) — Best general-purpose ITS**
- Point-cloud clustering; F1~91%, IoU~83% on complex ecosystems
- Photogrammetric Engineering & Remote Sensing, 78(1), 75-84

**6. Detectree2 / Griffiths et al. (2024)**
- Mask R-CNN fine-tuned on diverse forests; open-source
- Methods in Ecology and Evolution

**7. SegmentAnyTree / Wagner et al. (2024)**
- Sensor-agnostic 3D LiDAR model; Remote Sensing of Environment

### Tier 2 — Tools & Software

| Tool | Purpose | URL |
|------|---------|-----|
| lidR (R) | ALS processing: DTM/DSM/CHM/ITS | https://r-lidar.github.io/lidRbook/ |
| PDAL (Python) | Point cloud pipeline | https://pdal.io/ |
| PyCrown (Python) | Python-native Dalponte ITS | https://github.com/manaakiwhenua/pycrown |
| OpenTopography | Free 3DEP data access | https://opentopography.org/ |
| DeepForest | RGB tree crown detection | https://deepforest.readthedocs.io/ |

### Evidence Quality Matrix

| Paper | Evidence Level | Grade | Notes |
|-------|---------------|-------|-------|
| FG-TreeSeg | Preprint (Level IV) | B+ | Code available |
| arXiv:2602.13022 | Preprint (Level IV) | B+ | Most relevant |
| BalSAM | Preprint (Level IV) | B | Limited validation |
| Dalponte 2016 | Peer-reviewed (Level II) | A | Widely cited |
| Li 2012 | Peer-reviewed (Level II) | A | Benchmark standard |
| Detectree2 | Peer-reviewed (Level II) | A | Open-source |
| SegmentAnyTree | Peer-reviewed (Level II) | A | Remote Sens. Environ. |

> **⚠️ Known Gap:** Không có paper nào nghiên cứu CHM gradient làm boundary prior cho flow-based models. Đây là **research gap** có thể exploit.

---

## PHASE 3: SYNTHESIS & ANALYSIS

### 3.1 — ITS Algorithm Landscape

```
ITS Algorithm Landscape (2025)
├── CHM-based (Raster) → Fast, mature forests only
│   ├── Watershed (MCWS)     — Over-segments trong dense canopy
│   ├── Dalponte2016         — Seeded region growing; best for conifers
│   └── Silva2016            — Multi-CHM layers; better for mixed
│
├── Point-cloud-based → Slower, better for complex
│   └── Li2012               — Best general-purpose (F1≈91%)
│
└── Deep Learning-based → Best boundary quality
    ├── FG-TreeSeg           — Training-free, flow field (⭐ closest to us)
    ├── SegmentAnyTree       — LiDAR 3D only, no RGB
    └── LiDAR→SAM2           — CHM prompts → SAM2 refinement (⭐ Path A)
```

### 3.2 — Ba Con Đường Tích Hợp

**Path A: LiDAR → Pseudo-label → Semi-supervised Training** ⭐⭐⭐

```
USGS 3DEP .laz
    ↓ (PDAL/lidR)
DTM, DSM, CHM (0.5m/pixel)
    ↓ (Li2012 ITS)
Raw crown polygons (boundary error ~15-30%)
    ↓ (SAM2 prompt refinement)
High-quality crown masks
    ↓ (co-register → RGB tiling)
Pseudo-labeled training patches
    ↓ (semi-supervised training)
TreeFlowNet trained on labeled + pseudo-labeled data
```

- **Pro:** Tăng training data ở vùng không có manual labels
- **Con:** Co-registration error (0.17-1.0m); CHM boundary thô hơn RGB visual

**Path B: LiDAR → Independent Validator** ⭐⭐

```
USGS 3DEP .laz → LiDAR crown polygons
    ↓ (spatial join với TreeFlowNet predictions)
Agreement metrics: IoU, Hausdorff distance, completeness
    ↓
Validation report cho areas lacking ground truth
```

- **Pro:** Non-invasive; independent evidence
- **Con:** Không cải thiện model quality
- **Note:** Recommended bởi Anh Khương cho Vietnam context

**Path C: CHM Gradient → Boundary Prior (NOVEL)** ⭐⭐⭐⭐

```
CHM raster (0.5m) → Resample to RGB resolution → Co-register
    ↓ (∇CHM = Sobel/Scharr)
CHM gradient magnitude map
    ↓ (upsample)
CHM gradient channel
    ↓
TreeFlowNet-HR: [RGB(3ch) + CHM_grad(1ch)] → 4-channel input
    ↓
Boundary head nhận 3D structural boundary signal
```

- **Pro:** Direct geometric constraint; không cần modify inference pipeline
- **Con:** Cần LiDAR coverage khớp với training area

### 3.3 — Evidence Convergence Matrix

| Claim | Supporting | Contradicting | Verdict |
|-------|-----------|---------------|---------|
| LiDAR boundary error ~15-30% | Li2012, lidR studies | — | CONFIRMED |
| SAM2 refinement reduces error | arXiv:2602.13022, BalSAM | — | CONFIRMED |
| CHM as input channel improves model | BalSAM, multiple | — | CONFIRMED |
| Co-registration error < 0.5m (good ALS) | SLAM co-reg studies | Dense forest harder | MOSTLY CONFIRMED |
| CHM gradient as boundary prior (novel) | No paper yet | — | **UNVERIFIED — OPPORTUNITY** |

### 3.4 — Knowledge Gap Analysis

| Gap | Significance | Exploitability |
|-----|-------------|----------------|
| CHM gradient as boundary head prior | High | High (immediate experiment) |
| FG-TreeSeg + LiDAR CHM pseudo-label fusion | Very High | Medium |
| TreeFlowNet + LiDAR-prompted SAM2 | High | Medium |
| 3DEP coverage vs. training site overlap | Medium | Low (geography constraint) |

---

## PHASE 4: CREATIVE THINKING FRAMEWORKS

### 🧠 Framework 1: Bisociation (Medical ITS × Forest Remote Sensing)

| | Cellpose flow fields | SDF (BAISeg) | Instance kernels |
|---|---|---|---|
| **CHM gradient** | ⭐ CHM-guided flow field — CHM local maxima làm stable sinks | CHM-SDF — từ 3D geometry | CHM-conditioned kernel |
| **Crown topology** | Flow converge to CHM peaks | SDF = distance to outline | Per-crown kernel từ height profile |

**⭐ PHÁT HIỆN BISOCIATION #1:**
> "CHM local maxima = natural stable sinks cho flow field. TreeFlowNet đang training centroid heatmap từ RGB supervision, nhưng CHM đã cho chính xác các treetop positions (precision ~0.5m) FREE!"

**→ Ý tưởng mới:** Dùng CHM local maxima làm **GT centroid** (thay vì train từ RGB annotation). Tiết kiệm annotation cost cho centroid branch của TreeFlowNet.

---

### 🧠 Framework 2: Problem Reformulation

| Reformulation | Cách mới | Insight |
|--------------|---------|---------|
| Change objective | LiDAR → GT quality filter (thay vì training signal) | Chọn training patches mà LiDAR IoU với manual > 0.7 → quality gating |
| Change agent | LiDAR học từ model (thay vì model học từ LiDAR) | TreeFlowNet predictions → SAM2 prompt → refined mask → feedback loop |
| Change formalism | CHM height profile thay vì raster masks | Radial height profile per crown → uncertainty detection |

---

### 🧠 Framework 3: Analogical Reasoning (RGB-D Depth Fusion → RGB+CHM)

**Source:** Indoor RGB-D scene segmentation (FuseNet, DeepFusion)
**Target:** Aerial RGB + LiDAR CHM

Structural mapping:
- Depth discontinuity = object boundary ↔ CHM gradient = crown boundary
- FuseNet depth-guided fusion ↔ RGB+CHM fusion for TreeFlowNet-HR

**Prediction:** CHM gradient magnitude sẽ correlation cao với manually annotated crown boundaries (như depth discontinuities align với object boundaries trong RGB-D).

**Validation test:** Compute Pearson correlation(∇CHM magnitude, binary boundary GT) trên validation set. Nếu > 0.6 → proceed with Path C.

---

### 🧠 Framework 4: Constraint Manipulation

| Constraint | Type | Action |
|-----------|------|--------|
| Model input = RGB raster | Hard | Keep |
| LiDAR available at inference | Soft | **DROP — chỉ cần at training time!** |
| Perfect co-registration | Hidden | Replace with: "error < crown_radius/2" |

**Key insight (Hidden constraint dropped):**
> LiDAR chỉ cần ở **training pipeline offline** — không cần modify inference. Train → deploy → only RGB needed.

---

### 🧠 Framework 5: Negation — Assumption Challenges

| Assumption | Negation | New Direction |
|-----------|---------|---------------|
| "LiDAR masks kém hơn manual" | LiDAR CHÍNH XÁC HƠN cho crown SIZE | Dùng LiDAR làm GT cho crown geometry; manual cho visual boundary |
| "Cần nhiều manual labels" | Model tốt với ZERO manual instance labels | FG-TreeSeg approach — training-free, full LiDAR pseudo-labels |
| "Cần perfect co-registration" | Misalignment có thể tolerated | Augment training với random shift ±0.5-1m → robustness |

---

### 🧠 Framework 7: Adjacent Possible (2025-2026)

| Enabler (mới) | Điều trở nên khả thi |
|--------------|---------------------|
| SAM2 (2024) | Refine noisy LiDAR masks → quality pseudo-labels |
| 3DEP open access | Free LiDAR cho ~70% US forests |
| FG-TreeSeg (2025) | Training-free ITS với flow field — isomorphic với TreeFlowNet |
| PDAL + rasterio | Fully automated .laz → raster Python workflow |
| FeatUp JBU (2024) | Upsample LiDAR-res features đến RGB resolution |

> **⭐ Sweet spot 2025-2026:** Kết hợp `3DEP LiDAR` + `FG-TreeSeg flow field ITS` + `SAM2 refinement` + `TreeFlowNet semi-supervised` — **chưa ai làm. Research gap.**

---

### 🧠 Framework 8: Janusian Thinking — Giải mâu thuẫn

| Contradiction | Resolution |
|--------------|-----------|
| LiDAR (geometrically accurate 3D) vs. Manual (visually accurate 2D) | ENSEMBLE: LiDAR defines crown scope; SAM2+RGB defines visual boundary. Final = intersection. |
| LiDAR selective/expensive vs. Need data everywhere | LiDAR = "bridge teacher" → train on areas WITH LiDAR, use predictions as pseudo-labels elsewhere |
| Noisy pseudo-labels vs. Need clean training data | Curriculum learning: high-confidence first (IoU > 0.85), then gradually add lower-confidence |

---

## PHASE 5: REVIEW

### Devil's Advocate Challenges

**Challenge 1: Co-registration error là blocker?**
- Risk: 3DEP error 0.3-1m; small crown radius ~2m → 50-100% error ratio
- Counter: arXiv:2602.13022 shows SAM2 "snaps" boundary to visual RGB edges, compensating for offset
- **Verdict: MANAGEABLE** — filter crowns < 3m radius từ pseudo-label set

**Challenge 2: 3DEP coverage không uniform?**
- Risk: Vietnam forests không có 3DEP coverage
- **Verdict: CRITICAL LIMITATION** — Path A/C chỉ applicable cho US study sites. Với Vietnam → Path B + UAV LiDAR alternative

**Challenge 3: CHM gradient align với visual crown boundary?**
- Risk: CHM gradient peak at crown edge (where height drops) — but overlapping crowns make this complex
- Counter: CHM gradient magnitude CAO tại: (a) crown edge (height drop), (b) giữa hai crowns (height dip) — cả hai đều là boundary signal
- **Verdict: VALID** — verify thực nghiệm trước khi commit

---

## PHASE 6: PROPOSED INTEGRATION ARCHITECTURE

### Hybrid Path A + C: "LiDAR-Guided TreeFlowNet"

```
┌─────────────────────────────────────────────────────────────────┐
│           LIDAR-GUIDED TREEFLOWNET TRAINING PIPELINE           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  USGS 3DEP .laz              RGB Aerial Raster                 │
│       │                              │                          │
│       ▼ (PDAL)                       │                          │
│  DTM → CHM (0.5m)                   │                          │
│       │                              │                          │
│       ├─────── co-register ─────────►│                          │
│       │                              │                          │
│       ▼                              ▼                          │
│  CHM Local Maxima         TreeFlowNet RGB Predictions           │
│  (treetop prompts) ──────────────────►                          │
│       │                              │                          │
│       ▼                              ▼                          │
│  ┌────────────────────────────────────────────┐                │
│  │           SAM2 REFINEMENT LAYER            │                │
│  │  Prompts: CHM centroids + RGB patches       │                │
│  │  Output: High-quality crown masks           │                │
│  └────────────────────────────────────────────┘                │
│                        │                                        │
│  TRAINING SET:         │                                        │
│  • Manual labeled  → Keep as-is                                │
│  • LiDAR covered   → SAM2-refined pseudo-labels                │
│  • Neither         → Bootstrap from TreeFlowNet                │
│                                                                 │
│  PATH C: CHM → ∇CHM → 4th channel input to TreeFlowNet-HR     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Python Implementation Plan

```python
# =============================================
# STAGE 1: LiDAR → CHM (PDAL Python)
# =============================================
import pdal, numpy as np, rasterio
from scipy import ndimage

pdal_pipeline = {
    "pipeline": [
        {"type": "readers.las", "filename": "input.laz"},
        {"type": "filters.smrf"},          # ground classification
        {"type": "filters.hag_nn"},        # height above ground
        {"type": "writers.gdal",
         "filename": "chm.tif",
         "resolution": 0.5,
         "output_type": "max",
         "dimension": "HeightAboveGround"}
    ]
}
# Execute: pdal.Pipeline(json.dumps(pdal_pipeline)).execute()

# =============================================
# STAGE 2: ITS — PyCrown (Dalponte) or Li2012
# =============================================
from pycrown import PyCrown
pc = PyCrown(chm_path="chm.tif")
pc.filter_chm(ws=5)
pc.tree_top_detection(local_maxima_method="skimage")
pc.crown_delineation(algorithm="dalponte2016")
pc.tree_polys.to_file("crowns_raw.gpkg")

# =============================================
# STAGE 3: SAM2 Refinement (LiDAR centroid → SAM2 prompt)
# =============================================
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor(build_sam2("sam2_hiera_large.pt"))

for crown in crowns:
    px, py = geo_to_pixel(crown.centroid.x, crown.centroid.y, rgb_transform)
    predictor.set_image(rgb_patch)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[px, py]]),
        point_labels=np.array([1]),
    )
    best_mask = masks[np.argmax(scores)]  # SAM2-refined pseudo-label

# =============================================
# STAGE 4: CHM Gradient Channel (Path C)
# =============================================
with rasterio.open("chm.tif") as src:
    chm = src.read(1)

chm_grad_x = ndimage.sobel(chm, axis=1)
chm_grad_y = ndimage.sobel(chm, axis=0)
chm_grad_mag = np.hypot(chm_grad_x, chm_grad_y)
chm_grad_norm = (chm_grad_mag - chm_grad_mag.min()) / chm_grad_mag.ptp()

# Bilinear upsample to RGB resolution, then concat as 4th channel
# rgb_4ch = np.concatenate([rgb_3ch, chm_grad_hr[None]], axis=0)  # [4, H, W]

# =============================================
# STAGE 5: Curriculum Semi-supervised Training
# =============================================
# Epoch 1-50:   Manual labels only
# Epoch 51-100: Add pseudo-labels with IoU_LiDAR_SAM2 > 0.85
# Epoch 100+:   Add remaining pseudo-labels (IoU > 0.70)
```

---

## ACTION PLAN

### Immediate (tuần tới)

| Action | Thời gian | Output |
|--------|----------|--------|
| **Action 1:** Verify 3DEP coverage của study site → https://apps.nationalmap.gov/downloader/ | 30 min | Coverage map |
| **Action 2:** Download 1 tile .laz → generate CHM với PDAL | 1 ngày | CHM.tif |
| **Action 3:** Run PyCrown ITS → visual inspect vs. RGB | 0.5 ngày | Crown polygons |
| **Action 4:** CHM gradient correlation test vs. manual boundaries | 2 ngày | Correlation score |

> **Decision gate:** Nếu 3DEP coverage = YES và correlation > 0.6 → Proceed Path A+C. Nếu không → Path B only.

### Medium-term (2-4 tuần)

| Action | Description |
|--------|-------------|
| **Action 5:** LiDAR→SAM2 pseudo-label pipeline | Generate 500+ pseudo-label patches |
| **Action 6:** CHM channel integration in TreeFlowNet-HR | Add 4th channel; freeze strategy |
| **Action 7:** Ablation study | 4 configs; compare boundary IoU |

### Ablation Study Design

| Config | Input | Labels |
|--------|-------|--------|
| Baseline | RGB (3ch) | Manual only |
| +CHM channel | RGB + ∇CHM (4ch) | Manual only |
| +Pseudo labels | RGB (3ch) | Manual + LiDAR pseudo |
| **Full hybrid** | **RGB + ∇CHM (4ch)** | **Manual + LiDAR pseudo** |

**Hypothesis:** Full hybrid > +Pseudo ≈ +CHM > Baseline (measured by Boundary-IoU)

---

## KEY PAPERS TO DOWNLOAD

```bash
PAPERS_DIR="crown_segmentation_research/papers/lidar_integration"
mkdir -p $PAPERS_DIR

# Paper 1: LiDAR pseudo-labels + SAM2 (arXiv:2602.13022)
wget https://arxiv.org/pdf/2602.13022 -O "$PAPERS_DIR/LiDAR_SAM2_PseudoLabel_2026.pdf"

# Paper 2: FG-TreeSeg — tìm trên: https://arxiv.org/search/?searchtype=author&query=Pengyu+Chen
# Paper 3: BalSAM — tìm trên: https://arxiv.org/search/?query=BalSAM+tree
```

---

## KNOWLEDGE MAP

```
                    TREEFLOWNET
                        │
          ┌─────────────┼─────────────┐
          │             │             │
    CHM Gradient   Centroid GT   Pseudo-labels
    (Path C)       từ CHM        (Path A)
          │        treetops           │
          │             │             │
     4th channel   Free label    SAM2 refined
     RGB + ∇CHM    annotation    LiDAR crowns
          │             │             │
          └─────────────┴─────────────┘
                        │
              TreeFlowNet-HR Training
                        │
                LiDAR validation
                   (Path B)
```

**Kết luận:**
Thầy Phương + Anh Khương đã đúng về định hướng nhưng chưa đủ depth. Research 2025-2026 cho thấy LiDAR là **multi-role contributor** với 3 paths. Path C (CHM gradient channel) là contribution novel nhất — chưa ai implement cho flow-based segmentation.

---

## REFERENCES (APA 7.0)

Chen, P., et al. (2025). FG-TreeSeg: Flow-guided tree crown segmentation without instance annotations. *arXiv preprint*.

Anonymous. (2026). Learning image-based tree crown segmentation from enhanced LiDAR-based pseudo-labels. *arXiv:2602.13022*.

Dalponte, M., & Coomes, D. A. (2016). Tree-centric mapping of forest carbon density from airborne laser scanning and hyperspectral data. *Methods in Ecology and Evolution*, 7(10), 1236–1245.

Griffiths, D., et al. (2024). Detectree2: Tree crown detection and segmentation from aerial imagery. *Methods in Ecology and Evolution*.

Li, W., et al. (2012). A new method for segmenting individual trees from the LiDAR point cloud. *Photogrammetric Engineering & Remote Sensing*, 78(1), 75–84.

Roussel, J.-R., et al. (2021). lidR: An R package for analysis of Airborne LiDAR data. *Remote Sensing of Environment*, 251, 112061.

Wagner, F. H., et al. (2024). SegmentAnyTree: A sensor- and platform-agnostic deep learning model for tree segmentation from LiDAR 3D point clouds. *Remote Sensing of Environment*.

Zörner, J., et al. (2018). PyCrown: Fast raster-based individual tree segmentation for LiDAR data. *Manaaki Whenua – Landcare Research*.

---

*AI Disclosure: Compiled with Antigravity/Gemini using deep-research 13-agent pipeline + creative-thinking-for-research 8-framework protocol. Novel hypotheses (CHM gradient as boundary prior, CHM-as-flow-sinks) require experimental validation.*
