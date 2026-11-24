# TreeFlowNet A06 Fix Log & Implementation Record

**Version:** 3.0.0  
**Effective Date:** 2026-10-15  
**Reviewer:** Senior Research Software Engineer & Scientific Auditor  
**Policy Compliance:** Every defect fix documented with root cause, affected components, before/after behavior, and regression test.

---

## 1. Fix Catalog

### FIX-01: Multi-Epoch Worker RNG Reseeding Defect
- **Defect ID:** F06 (Severity: HIGH)
- **Affected File:** `crown_segmentation_research/clean_benchmark/dataset.py:L46`
- **Root Cause:** `rng = np.random.default_rng(i if self.fixed else (torch.initial_seed() + i) % (2 ** 32))` used PyTorch's `initial_seed()`, which does not advance per epoch in DataLoader worker subprocesses. In multi-epoch training, workers repeatedly cropped identical pixel coordinates every epoch.
- **Corrected Behavior:** Implemented deterministic four-factor hashing function:
  ```python
  def get_crop_rng(master_seed: int, epoch: int, split: str, tile_idx: int, crop_idx: int) -> np.random.Generator:
      key = f"{master_seed}:{epoch}:{split}:{tile_idx}:{crop_idx}".encode("utf-8")
      seed_int = int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
      return np.random.default_rng(seed_int)
  ```
- **Regression Test:** `test_crop_sampler_multi_epoch_diversity` in `test_suite.py`.
- **Verification Result:** PASS. Epoch 0 and Epoch 1 generate distinct random sequences for training, while epoch 0 remains strictly deterministic for validation.

---

### FIX-02: Silent Ground-Truth Erasure via Rasterized Priority Ownership
- **Defect ID:** F04 (Severity: HIGH)
- **Affected Components:** `dataset.py`, `targets_fast.py`, `evaluation/benchmark_bam_coco_ap.py`
- **Root Cause:** Converting overlapping COCO instance polygons into a single-channel label map (`label.npz`) caused overlapping crowns to overwrite one another, erasing 63 training annotations and 4 validation annotations from priority support ($S_i^{priority} = \emptyset$). Evaluating instance detection against the rasterized label map silently dropped these crowns from the evaluation denominator.
- **Corrected Behavior:** Formalized mathematical separation in `REFERENCE_SEMANTICS_V3.md`:
  - Primary instance matching strictly evaluates predictions against original COCO RLE masks $M_i$.
  - Priority ownership raster $S_i^{priority}$ is relegated strictly to field decoders requiring single-channel direction targets.
  - Implemented area-ascending priority conflict resolution so small interior crowns retain ownership over nested pixels.
- **Regression Test:** `test_support_definitions_distinction` and `test_original_masks_preserved_for_matching` in `test_suite.py`.
- **Verification Result:** PASS. Overlapping crowns remain fully preserved and matchable at IoU $\ge 0.5$.

---

### FIX-03: Area Calculation Discrepancies and Size-Bin Drift
- **Defect ID:** F05 (Severity: MEDIUM)
- **Affected Components:** `dataset_diagnostics.py`, `audit.py`
- **Root Cause:** Discrepancy between floating-point vector polygon area in COCO metadata and discrete raster RLE pixel count caused 87 annotations across splits to shift size-bin strata.
- **Corrected Behavior:** Standardized canonical physical area formula:
  $$A_i^{m^2} = N_i^{RLE} \cdot |\det(A)|$$
  where $|\det(A)| = |a_x a_y - a_{xy} a_{yx}|$ evaluates the affine pixel ground area. Native COCO metadata areas are preserved separately.
- **Regression Test:** `test_canonical_area_and_determinant` and `test_fixture_17_nonsquare_gsd_determinant` in `test_suite.py`.
- **Verification Result:** PASS. Correctly handles both square ($0.03\text{ m} \times 0.03\text{ m}$) and non-square GSD affine transforms.

---

### FIX-04: Conflation of Raw Query Slots with Accepted Instance Detections
- **Defect ID:** F08 (Severity: CRITICAL)
- **Affected Components:** `q_sanity.py`, `diagnosis_2026-10-11/fgq_revision_2026-10-13/grouping.py`
- **Root Cause:** Evaluating raw Stage 0 query slots under unique-cover grouping forced false abstentions and false-positive penalties from inactive background queries, leading to the misleading conclusion that query heads cannot group canopy instances.
- **Corrected Behavior:** Established the 4-Stage Output Contract (`OUTPUT_CONTRACT_V3.md`). Stage 0 queries must be materialized into Stage 1 candidate proposals and filtered at Stage 2 via an explicit objectness threshold before participating in structural grouping or instance matching.
- **Regression Test:** `test_fixture_09_unassigned_anchors` and Stage 0->3 contract tests in `test_suite.py`.
- **Verification Result:** PASS. Inactive query slots are properly filtered at Stage 1/2 without corrupting grouping coverage.

---

### FIX-05: Unbuffered Out-of-Fold Calibration Leakage
- **Defect ID:** F09 (Severity: HIGH)
- **Affected Components:** `clean_benchmark/seed_scorer.py`
- **Root Cause:** Historical OOF validation scores partitioned tiles without spatial buffering, resulting in 378 cross-fold tile pairs sharing valid pixels and 370 pairs sharing ground-truth crowns.
- **Corrected Behavior:** Established `CALIBRATION_POLICY_V3.md`. Prohibited claiming spatial independence for historical OOF scores and mandated spatial block cross-validation with minimum 30m buffer for all future calibration.
- **Regression Test:** `test_spatial_buffer_validation_fail_closed` in `test_suite.py`.
- **Verification Result:** PASS. Spatially touching or unbuffered folds are rejected.

---

### FIX-06: Preflight Gate & Fail-Closed Enforcement
- **Defect ID:** F12 (Severity: HIGH)
- **Affected Components:** Infrastructure entry points
- **Root Cause:** Historical scripts executed ad-hoc without validating baseline hashes, split buffer compliance, or test-set access restrictions.
- **Corrected Behavior:** Built `preflight.py` providing dual-mode validation:
  - `exploratory` mode permits prototyping on Z1 with explicit warnings and sensitivity controls.
  - `confirmatory` mode strictly fails-closed on Z1 connected splits and reports `CONFIRMATORY_NOT_READY` / `EXTERNAL_HOLDOUT_REQUIRED`.
- **Regression Test:** `test_preflight_fail_closed_behavior` executed via preflight CLI.
- **Verification Result:** PASS. Nonzero exit code (code 1) produced upon critical confirmatory violation.
