# TreeFlowNet Unresolved Assumptions & Epistemic Limitations

**Version:** 3.0.0  
**Effective Date:** 2026-10-15  
**Policy Requirement:** All items in this document must remain explicitly classified as unresolved. They CANNOT be silently treated as resolved or converted into confirmatory claims.

---

## 1. Domain & Sensor Assumptions (Blocker B03)

### Assumption U01: Site and Survey Independence
- **Recorded State:** Mosaics are labeled `sblz1`, `sblz2`, and `sblz3`. The metadata contains date-like string `20210902`.
- **Epistemic Status:** `UNRESOLVED_ASSUMPTION`.
- **Limitation:** The presence of three raster identifiers does NOT prove independent geographical sites, distinct forest biomes, different flight altitudes, or separate acquisition campaigns. 
- **Scientific Consequence:** High performance across these three mosaics cannot be claimed as "cross-site ecological generalization" or "sensor invariance."
- **Required Action:** Obtain raw flight logs, sensor calibration manifests, and geographic coordinate bounding polygons from the data provider before claiming cross-site evaluation.

### Assumption U02: Biological Individuation vs. Orthomosaic Canopy Segmentation
- **Recorded State:** Ground truth consists of 2D COCO polygons traced on 0.03 m RGB orthomosaics.
- **Epistemic Status:** `REQUIRES_SCIENTIFIC_DECISION` / `UNRESOLVED_ASSUMPTION`.
- **Limitation:** A 2D orthomosaic displays top-of-canopy foliage. It cannot observe suppressed understory trees completely occluded by dominant overstory crowns, nor can it definitively differentiate multiple interlocking branches of a single multi-stem tree from touching distinct stems without field-surveyed stem locations.
- **Scientific Consequence:** TreeFlowNet benchmarks evaluate *orthomosaic crown delineation*, which is a surrogate for, but not identical to, biological stem inventory.

---

## 2. Benchmark & Evaluation Assumptions (Blockers B01, B02, B05)

### Assumption U03: Confirmatory Holdout Purity
- **Recorded State:** Split consists of 148 train, 139 validation, and 168 test tiles.
- **Epistemic Status:** `FALSIFIED_FOR_CONFIRMATORY_USE`.
- **Limitation:** Historical test set tiles have been inspected, diagnosed, and used to measure AP in prior stages (EXP003, EXP015, EXP020, etc.). Furthermore, 235 tiles across train, validation, and test form a single connected component with touching boundaries.
- **Scientific Consequence:** No confirmatory benchmark can be published on the historical test set without compromising scientific integrity.
- **Required Action:** Level 3 readiness must report `CONFIRMATORY_NOT_READY` and output `EXTERNAL_HOLDOUT_REQUIRED`.

### Assumption U04: Equivalence of Model Capacities and Supervision
- **Recorded State:** Comparisons between F (Field Decoder), G (Basin Graph), and Q (Transformer Query Head) share an encoder backbone.
- **Epistemic Status:** `UNRESOLVED_ASSUMPTION`.
- **Limitation:** Sharing a frozen Swin-T backbone does NOT establish computational or supervisory parity. F learns dense flow fields and distance transforms; G operates on segmented basin adjacency graphs; Q learns Hungarian-matched set queries with class/no-object heads and cross-attention.
- **Scientific Consequence:** Differences in AP or recall between F, G, and Q reflect the combination of loss formulation, head capacity, optimization dynamics, and proposal materialization—not an isolated test of "flow vs. query" in the abstract.

---

## 3. Methodological Assumptions (Blockers B04, B07, B08)

### Assumption U05: Inactive Query Semantics in Transformer Decoders
- **Recorded State:** Query heads output $Q$ slots (e.g., $Q=16$ or $Q=100$). Unmatched queries receive no positive mask loss during Hungarian matching.
- **Epistemic Status:** `RESOLVED_BY_CONTRACT_V3`.
- **Specification:** Raw query slots before thresholding (Stage 0) are model-native hypotheses, NOT accepted instance proposals. They must be materialized through an explicit confidence/objectness threshold (Stage 1/2) before participating in structural grouping or instance matching.

### Assumption U06: Spatial Independence of Out-of-Fold (OOF) Scores
- **Recorded State:** Historical OOF validation scores partitioned tiles without spatial buffering.
- **Epistemic Status:** `FALSIFIED`.
- **Specification:** 378 cross-fold tile pairs share valid source pixels, with 370 pairs sharing ground-truth crowns. Historical OOF scores are spatially dependent. Spatial calibration must use buffered block folds.
