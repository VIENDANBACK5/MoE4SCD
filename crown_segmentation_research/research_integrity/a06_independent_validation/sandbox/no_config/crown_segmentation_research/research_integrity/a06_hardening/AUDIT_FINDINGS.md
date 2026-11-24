# TreeFlowNet A06 Scientific Audit Findings

**Version:** 3.0.0  
**Audit Date:** 2026-10-15  
**Auditor:** Senior Research Software Engineer, Computer Vision Researcher, Scientific Auditor  
**Scope:** Data pipeline, coordinate reference systems, annotation integrity, spatial split independence, training crop sampling, F–G–Q comparison parity, and evaluation metrics across 455 tiles and 90,462 annotations.

---

## 1. Executive Summary of Audit Findings

The audit systematically examined all 455 prepared tiles (148 train, 139 validation, 168 test) and 90,462 annotation records, historical evaluation scripts (`study.py`, `grouping.py`, `evaluate.py`), dataset loaders (`CanopyCropDataset`), and historical sanity/benchmark reports (`q_sanity.py`, `exp020`/`exp021`).

Every finding is assigned one of six standardized epistemic classifications:
- `VERIFIED_CORRECT`: Inspected against source code/artifacts and proven sound.
- `DEFECT_CONFIRMED`: Reproducible flaw identified, root cause traced, and corrective fix required.
- `SEMANTIC_AMBIGUITY`: Mathematical or conceptual definition under-specified, leading to potential metric gaming or divergence.
- `MISSING_CONTROL`: Necessary guardrail, validation gate, or invariant check absent from infrastructure.
- `NOT_VERIFIABLE`: External information or raw uncurated source inaccessible from current repository.
- `REQUIRES_SCIENTIFIC_DECISION`: Domain assumption that cannot be resolved solely by software engineering.

---

## 2. Detailed Findings Catalog

### FINDING-01: Boundary-Localized Valid-Pixel Overlap in Same-Grid Split
- **Classification:** `VERIFIED_CORRECT` (as an audit finding) & `MISSING_CONTROL` (in dataset splitting infrastructure).
- **Evidence:** 
  - Train–Val: 14,094 shared source pixels (12.6846 m²), 114 tile pairs, 5 GT-positive pairs.
  - Train–Test: 12,966 shared source pixels (11.6694 m²), 89 tile pairs, 6 GT-positive pairs.
  - Val–Test: 11,895 shared source pixels (10.7055 m²), 69 tile pairs, 0 GT-positive pairs.
  - Sensitivity analysis: 1-pixel binary erosion of both valid masks eliminates 100% of same-grid shared valid pixels (0 pairs remain).
- **Scientific Impact:** The pixel overlap is strictly boundary-localized (1-pixel seam between adjacent cropped tiles). It does NOT constitute extensive test contamination or duplicate image content. However, eliminating shared boundary pixels through 1-pixel erosion does NOT confer spatial independence; spatial autocorrelation persists across the seam.
- **Status:** Documented, bounded, and controlled. Confirmatory use of this split remains strictly prohibited without an explicit spatial buffer.

---

### FINDING-02: Cross-Grid Ground Footprint Overlap Across Mosaics
- **Classification:** `VERIFIED_CORRECT` (audit metric) & `SEMANTIC_AMBIGUITY` (spatial definition).
- **Evidence:** 64 pairs between different mosaic grids (sblz1, sblz2, sblz3) have non-integer grid offsets. Polygon intersection of cell bounds reveals 2 pairs with valid ground footprint overlap remaining after 1-pixel erosion (0 GT-positive pairs).
- **Scientific Impact:** Ground footprint overlap in geospatial coordinates indicates geographic proximity or partial re-flight coverage, but does not prove byte-identical imagery or identical acquisition date/illumination.
- **Status:** Explicit distinction between source-grid pixel overlap and geospatial ground footprint overlap enforced.

---

### FINDING-03: Spatial Component Connectivity Across Historical Splits
- **Classification:** `VERIFIED_CORRECT` & `REQUIRES_SCIENTIFIC_DECISION`.
- **Evidence:** 272 valid-overlap tile pairs connect 235 Z1 tiles into a single continuous connected component spanning Train, Validation, and Test.
- **Scientific Impact:** The historical split was created by contiguous block slicing with touching boundaries. Silently dropping graph edges to artificially claim a "clean split" violates research integrity.
- **Status:** Confirmatory testing on this split is BLOCKED (`CONFIRMATORY_NOT_READY`). Only exploratory model prototyping is permitted on Z1.

---

### FINDING-04: Non-Equivalence of Unique Support and Priority Owner Support
- **Classification:** `DEFECT_CONFIRMED` (in historical documentation conflating the two) & `SEMANTIC_AMBIGUITY`.
- **Evidence:**
  - $U_i = M_i \setminus \bigcup_{j \ne i} M_j$ (Unique Support).
  - $S_i^{priority} = \{p \in M_i \mid \text{argmax/priority assignment} = i\}$ (Priority Owner).
  - Train: 63 annotations have empty unique support ($U_i = \emptyset$), but only 10 have empty priority support.
  - Validation: 4 annotations have empty unique support and 4 have empty priority support.
- **Root Cause:** Converting overlapping COCO instance polygons into a single-channel label raster assigns overlapping pixels to a single instance (usually the last rendered or index-ordered). This generates synthetic boundaries and alters the geometric support.
- **Scientific Consequence:** If primary instance matching uses rasterized priority ownership, instances completely enclosed inside larger crowns are silently erased from the evaluation ground truth.
- **Correction:** Enforce COCO original RLE polygons as the immutable reference for instance matching. Retain unique support strictly for anchor sampling and core erosion.

---

### FINDING-05: Metadata Area vs. RLE Pixel Count Discrepancy
- **Classification:** `VERIFIED_CORRECT` & `MISSING_CONTROL`.
- **Evidence:**
  - Median absolute difference: Train 5.33 px (0.051%), Val 4.88 px (0.059%), Test 4.43 px (0.071%).
  - Annotations shifting size bins: Train 61, Val 4, Test 22.
- **Root Cause:** COCO `area` metadata field stores floating-point polygon area (from original vector geometry), whereas raster IoU and mask operations evaluate discrete RLE counts ($N_i^{RLE}$).
- **Scientific Consequence:** Mismatch causes annotations near size-bin boundaries (e.g., exactly 1.0 m² or 4.0 m²) to shift strata, corrupting size-stratified recall metrics if different tools use different area fields.
- **Correction:** Adopt deterministic formula $A_i^{m^2} = N_i^{RLE} \cdot |\det(A)|$ where $|\det(A)|$ is the pixel ground area ($a_x \cdot a_y \approx 0.0009\text{ m}^2$). Retain native metadata area separately.

---

### FINDING-06: PyTorch Dataset Worker RNG Reseeding Defect
- **Classification:** `DEFECT_CONFIRMED`.
- **Evidence:** In `CanopyCropDataset` (`clean_benchmark/dataset.py:L46`):
  ```python
  rng = np.random.default_rng(i if self.fixed else (torch.initial_seed() + i) % (2 ** 32))
  ```
- **Root Cause:** In PyTorch `DataLoader` with multi-epoch training, `torch.initial_seed()` is fixed per worker process across epochs unless the DataLoader sampler or `worker_init_fn` explicitly seeds with `(epoch, worker_id)`.
- **Scientific Consequence:** In multi-epoch training, every worker extracts the exact same sequence of crops in epoch 1, epoch 2, ..., epoch $N$. The model encounters zero crop-position diversity across epochs.
- **Correction:** Implement explicit epoch-aware deterministic seeding: `seed = hash(master_seed, epoch, split, tile_idx)`. Add regression test asserting different crop coordinates across epochs.

---

### FINDING-07: Fragment-Only Training Targets in Crop Sampler
- **Classification:** `DEFECT_CONFIRMED` & `MISSING_CONTROL`.
- **Evidence:**
  - With `target_margin=0` in `CanopyCropDataset`, 41.7% of crowns in 1024-px crop touch the boundary and are truncated into fragments.
  - In historical Q sanity (`q_sanity.py`), 100% of positive targets (all 11 targets across 4 crops) were fragments cut by the 256×256 window.
- **Scientific Consequence:** The network is penalized or trained to treat partial boundary slices as complete botanical individuals with synthetic centroids and distorted distance transforms. Complete-crown instance formation cannot be verified.
- **Correction:** Implement a structured full-crown sampling policy that categorizes and records: complete crowns, partial crowns, and background-only crops. Add validation ensuring target queries do not exceed capacity.

---

### FINDING-08: Conflation of Raw Q Slots and Accepted Proposals
- **Classification:** `DEFECT_CONFIRMED` & `SEMANTIC_AMBIGUITY`.
- **Evidence:** In historical Q sanity, 16 raw queries produced 16 non-empty mask logits per crop, even in zero-crown crops. Unmatched queries receive no mask supervision during Hungarian loss matching.
- **Scientific Consequence:** Evaluating raw Q slots without thresholding under unique-cover grouping forces false abstention and massive duplicate/FP penalties. Conversely, post-filtering Q with ground-truth assignment violates fair comparison against F and G.
- **Correction:** Enforce a strict 4-stage output contract (Stage 0: Model-native, Stage 1: Materialized proposals, Stage 2: Accepted instances, Stage 3: Scene instances). All cross-model instance metrics MUST be evaluated strictly at Stage 2.

---

### FINDING-09: Spatial Leakage in Historical Within-Validation OOF Scorer
- **Classification:** `DEFECT_CONFIRMED` & `MISSING_CONTROL`.
- **Evidence:** Historical Out-Of-Fold (OOF) scoring on validation set had 378 cross-fold tile pairs sharing valid pixels, including 370 pairs sharing ground-truth positive crown support.
- **Scientific Consequence:** Folds created by random or hash-based tile partitioning without spatial buffering leak features and crowns across folds, inflating OOF calibration metrics.
- **Correction:** Forbid claim of spatial independence for historical OOF scores. Require spatially buffered block cross-validation for all future confidence calibration.

---

### FINDING-10: Test Set Re-Use Precluding Confirmatory Holdout
- **Classification:** `VERIFIED_CORRECT` & `MISSING_CONTROL`.
- **Evidence:** 168 test tiles were repeatedly used for diagnostic evaluations, border overlap analyses, and benchmark scripts (`eval_test_set.py`, `eval_test_run.log`).
- **Scientific Consequence:** The historical test set has participated in iterative research feedback loops. It is methodologically contaminated and cannot serve as an independent confirmatory holdout.
- **Correction:** Formally declare `CONFIRMATORY_BENCHMARK_READY = false`. Require `EXTERNAL_HOLDOUT_REQUIRED` for any future confirmatory publication claims.

---

### FINDING-11: Fundamental Limitations of G0 Merge-Only Formulation
- **Classification:** `VERIFIED_CORRECT` & `SEMANTIC_AMBIGUITY`.
- **Evidence:** Experiments EXP020/021 showed merge-only grouping achieved 36.48–36.58 AP vs baseline 36.50 AP.
- **Root Cause:** A merge-only operator can only join adjacent over-segmented basins. It is mathematically incapable of splitting an under-segmented basin containing multiple tree crowns.
- **Scientific Consequence:** The empirical failure of G0 is a property of the merge-only formulation, not evidence against relational graph or partitioning methods in general.
- **Correction:** Formally distinguish G0 (merge-only), G1 (repartitioning/cut-and-merge), and G-relation in all comparison documentation.

---

### FINDING-12: Absence of Automated Preflight Gates
- **Classification:** `MISSING_CONTROL`.
- **Evidence:** Scripts previously ran ad-hoc without verifying configuration contracts, data hashes, or split compliance, allowing silent execution of invalid configurations.
- **Correction:** Implement `crown_segmentation_research.research_integrity.a06_hardening.preflight` with dual modes (`exploratory` and `confirmatory`), enforcing fail-closed exits on critical violations.
