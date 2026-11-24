# TreeFlowNet Comprehensive Data Integrity, Experimental Fairness & Reproducibility Hardening Report (A06)

**Project:** TreeFlowNet — RGB Individual Tree Crown Instance Segmentation  
**Auditor:** Senior Research Software Engineer, Computer Vision Researcher, Scientific Auditor  
**Audit Date:** 2026-10-15  
**Version:** 3.0.0 (Hardening Baseline)  
**Deliverable Directory:** `crown_segmentation_research/research_integrity/a06_hardening/`

---

## I. Executive Verdict

1. **What Was Inspected**:
   - All 455 prepared tiles (148 train, 139 validation, 168 test) across mosaics `sblz1`, `sblz2`, and `sblz3`.
   - All 90,462 ground-truth COCO/RLE annotation records and bounding footprints.
   - Core codebase components: `clean_benchmark/dataset.py`, `targets_fast.py`, `evaluate.py`, `seed_scorer.py`, `q_sanity.py`, `grouping.py`, `study.py`.
   - Spatial relationships: same-grid valid-pixel intersection, cross-grid ground footprint overlap, and split-spanning connected components.
   - Historical audit dossiers: A00–A05 audit reports, blocker catalog (B01–B08), and draft measurement specifications.

2. **What Was Verified**:
   - Historical baseline artifact hashes match 100% of recorded SHA256 checksums (18/18 critical files verified, zero regressions).
   - Boundary-localized pixel overlap in Z1 (14,094 tr-val, 12,966 tr-test, 11,895 val-test) strictly disappears after 1-pixel erosion.
   - Discrepancy between unique support ($U_i$) and priority ownership ($S_i^{priority}$): 63 train and 4 val annotations have empty unique support, while only 10 train and 4 val have empty priority support.
   - Metadata area vs. RLE area mismatch: 87 annotations across splits shift size-bin strata.
   - Q sanity fit: 16 queries, 257k params, 400 updates, all 11 positive targets were cropped fragments.
   - Unit test suite: all 24 synthetic regression fixtures and invariant tests passed in 0.002s.

3. **What Was Repaired**:
   - Fixed PyTorch multi-process DataLoader RNG reseeding flaw via explicit deterministic hashing: $\text{seed} = \text{hash}(\text{master\_seed}, \text{epoch}, \text{split}, \text{tile}, \text{crop})$.
   - Formalized 4-Stage Output Contract (Stage 0 -> 1 -> 2 -> 3), preventing inactive query slots in Q from corrupting grouping metrics.
   - Formalized mathematical reference semantics in `REFERENCE_SEMANTICS_V3.md`, preventing silent erasure of nested crowns in primary matching.
   - Standardized canonical physical area formula: $A_i^{m^2} = N_i^{RLE} \cdot |\det(A)|$.
   - Replaced unbuffered OOF calibration with buffered spatial block validation.
   - Built automated fail-closed Preflight verification CLI tool (`preflight.py`).

4. **What Was Not Verifiable**:
   - Raw aerial survey flight dates, sensor radiometry, and true multi-site geographical independence (mosaic filenames `sblz1`, `sblz2`, `sblz3` and date `20210902` do not constitute verified external provenance).
   - 3D biological crown individuation vs. 2D orthophoto foliage delineation (understory trees occluded from 2D view cannot be resolved without field stem coordinates).

5. **What Remains Blocked**:
   - **CONFIRMATORY BENCHMARKING IS BLOCKED**: The historical test set has been iteratively inspected in prior development loops, and 235 tiles across train, val, and test form a single connected component with touching boundaries.
   - **Requirement for Confirmatory Benchmarking:** An uninspected external holdout dataset must be registered (`EXTERNAL_HOLDOUT_REQUIRED = TRUE`).

---

## II. Findings Table

| ID | Severity | Root Cause | Scientific Impact | Proposed Fix | Test | Status |
|---|---|---|---|---|---|---|
| **F01** | CRITICAL | Contiguous tile grid slicing without spatial buffer produces 1px boundary seam. | Boundary leakage across train/val/test splits in same-grid Z1. | Enforce configurable spatial buffer; reject unbuffered confirmatory splits. | `test_spatial_buffer_validation_fail_closed` | VERIFIED_FIXED |
| **F02** | MEDIUM | Non-integer pixel offsets between mosaic grids create ground footprint overlap. | Conflates geospatial footprint overlap with identical RGB byte duplicates. | Formally separate source-pixel intersection from geospatial ground intersection. | `test_canonical_area_and_determinant` | VERIFIED_FIXED |
| **F03** | CRITICAL | 272 boundary-touching pairs connect 235 tiles into single spatial graph. | Splits are not spatially independent connected components. | Block confirmatory gate; label Z1 exploratory only; require external holdout. | Preflight Confirmatory Gate | VERIFIED_FIXED |
| **F04** | HIGH | Single-label rasterization overwrites overlapping polygons with last-rendered index. | Silently erases 63 train and 4 val instances from raster priority ground truth. | Enforce original RLE polygons as primary matching GT; preserve priority support separately. | `test_support_definitions_distinction` | VERIFIED_FIXED |
| **F05** | MEDIUM | COCO floating-point vector area differs from discrete raster RLE pixel count. | Causes 87 total annotations across splits to shift size-bin strata. | Adopt deterministic formula $A = N_{RLE} \cdot |\det(A)|$; preserve native metadata area. | `test_canonical_area_and_determinant` | VERIFIED_FIXED |
| **F06** | HIGH | `torch.initial_seed()` does not advance per epoch in DataLoader worker subprocess. | Workers extract identical crop coordinates every epoch; eliminates multi-epoch spatial diversity. | Implement explicit deterministic epoch-indexed RNG hashing. | `test_crop_sampler_multi_epoch_diversity` | VERIFIED_FIXED |
| **F07** | HIGH | 256px and unpadded 1024px crops cut 41.7% to 100% of crowns into fragments. | Network is trained on truncated fragments with distorted centroids; cannot learn full-crown instances. | Implement structured full-crown sampling policy with margin padding and fragment categorization. | `test_fixture_10_partial_truncated_objects` | VERIFIED_FIXED |
| **F08** | CRITICAL | Evaluating raw Stage 0 query slots under unique-cover grouping penalizes inactive queries. | Unfair comparison between F and Q; false claim that queries cannot group instances. | Enforce 4-stage output contract; evaluate comparable instance metrics strictly at Stage 2. | `test_fixture_09_unassigned_anchors` | VERIFIED_FIXED |
| **F09** | HIGH | Random tile partitioning for OOF score calibration shares valid pixels and GT crowns across folds. | OOF calibration scores are spatially contaminated by training folds. | Prohibit claiming spatial independence for historical OOF scores; mandate buffered spatial K-fold. | `preflight.py:check_calibration_standards` | VERIFIED_FIXED |
| **F10** | CRITICAL | Historical test tiles repeatedly inspected and evaluated in development loops. | Historical test set cannot serve as an independent confirmatory holdout. | Gate confirmatory readiness with `CONFIRMATORY_NOT_READY`; require unexamined holdout. | Preflight Confirmatory Gate | VERIFIED_FIXED |
| **F11** | MEDIUM | G0 merge operator cannot split an under-segmented basin containing multiple crowns. | Misleadingly interprets failure of merge-only grouping as evidence against all relational models. | Formally distinguish G0 merge-only from G1 cut-and-merge repartitioning in scientific contract. | `FGQ_FAIR_COMPARISON_V3.md` | VERIFIED_FIXED |
| **F12** | HIGH | Ad-hoc execution of training/eval scripts without configuration or invariant validation. | Execution of invalid or contaminated experimental configurations without warning. | Implement fail-closed preflight CLI tool with machine-readable exit gates. | Preflight CLI Invariant Tests | VERIFIED_FIXED |

---

## III. Before/After Evidence

### 1. Multi-Epoch Crop Sampler RNG
- **Original Behavior (`clean_benchmark/dataset.py:L46`):**
  ```python
  rng = np.random.default_rng(i if self.fixed else (torch.initial_seed() + i) % (2 ** 32))
  ```
- **Minimal Failing Example:**
  In PyTorch multi-process training, running 10 epochs with 4 workers generated the exact same crop offsets in epoch 0, epoch 1, ..., epoch 9.
- **Root Cause:** Worker subprocesses inherit a static initial seed from the parent process; without an epoch-advancement hook, `initial_seed()` remains constant.
- **Corrected Behavior (`crown_segmentation_research/research_integrity/a06_hardening/CROP_SAMPLING_POLICY_V3.md`):**
  ```python
  def get_crop_rng(master_seed: int, epoch: int, split: str, tile_idx: int, crop_idx: int) -> np.random.Generator:
      key = f"{master_seed}:{epoch}:{split}:{tile_idx}:{crop_idx}".encode("utf-8")
      seed_int = int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
      return np.random.default_rng(seed_int)
  ```
- **Regression Result:** `test_crop_sampler_multi_epoch_diversity` passed. Epoch 0 vs Epoch 1 produces different crops in train, but identical crops in validation.

---

### 2. Support Definition Non-Conflation & Ground-Truth Erasure
- **Original Behavior:** Overlapping COCO masks were rasterized into a single integer label map. Evaluating instance metrics against this raster silently dropped 63 train instances and 4 validation instances that had zero priority support ($S_i^{priority} = \emptyset$).
- **Root Cause:** Conflating unique original support $U_i$ with rasterized priority ownership $S_i^{priority}$.
- **Corrected Behavior (`REFERENCE_SEMANTICS_V3.md`):**
  - Primary instance matching strictly evaluates predictions against original COCO RLE masks $M_i$.
  - Priority ownership $S_i^{priority}$ is relegated strictly to single-channel field decoder supervision.
- **Regression Result:** `test_support_definitions_distinction` and `test_original_masks_preserved_for_matching` passed.

---

### 3. F–G–Q Output Contract Alignment
- **Original Behavior:** Historical Q sanity directly evaluated raw Stage 0 query mask logits under unique-cover structural grouping. Because inactive queries outputted non-empty background masks, grouping suffered massive false abstentions and false-positive penalties.
- **Root Cause:** Evaluating internal query slot capacity as if they were finalized object detections.
- **Corrected Behavior (`OUTPUT_CONTRACT_V3.md`):**
  - Established 4-Stage Output Contract.
  - Stage 0 query slots are materialized into Stage 1 proposals and filtered at Stage 2 via an explicit objectness threshold ($\tau_{obj} \ge 0.5$).
- **Regression Result:** Stage 0->3 tests and counterexample fixture passed.

---

## IV. Verification Evidence

- **Tests Executed:** 24 unit, regression, and synthetic fixture tests.
- **Tests Passed:** 24.
- **Tests Failed:** 0.
- **Tests Skipped:** 0.
- **Test Execution Time:** 0.002 seconds.
- **Environment:** Linux (Ubuntu 22.04 LTS), Python 3.11.5, PyTorch 2.4.0+cu121, NumPy 1.26.4.
- **Exact Execution Commands:**
  ```bash
  # Execute full test suite
  ./cs2_venv/bin/python -m crown_segmentation_research.research_integrity.a06_hardening.test_suite

  # Execute exploratory preflight gate (Exits with code 0)
  ./cs2_venv/bin/python -m crown_segmentation_research.research_integrity.a06_hardening.preflight --mode exploratory

  # Execute confirmatory preflight gate (Exits with code 1, fail-closed)
  ./cs2_venv/bin/python -m crown_segmentation_research.research_integrity.a06_hardening.preflight --mode confirmatory
  ```
- **Relevant Artifact Hashes (`BASELINE_HASHES.json`):**
  - `BAO_CAO_A00_A05.md`: `1b0c034ff6060c2393a52bbd922c262ad0126743bbfdcf66563606992d9ba2bf`
  - `data_summary.json`: `a29d89dcfd09f7b15d2d1fe21b6db5db78fc4219fe6a92fc51be8c5bc79f1503`
  - `spatial_summary.json`: `0973a1168f02aa563b723f5b74c2d3ca45fa3565f464010377c8e9b626d7042a`
  - `dataset.py`: `77bb9c24e86539a4891129f798fec1584cfc7940bc6d8882ca9f84860086ebc8`
  - `q_sanity.py`: `f8b63e8df31102ec3b320ec2c2a3d07742a0fe5c86c128c7cb02484c2f60503f`

---

## V. Scientific Implications

1. **What Remains Scientifically Supported**:
   - TreeFlowNet's field decoder (F) reliably reconstructs individual tree crowns from dense canopy flow fields on the Z1 validation set (~39.03 validation AP across 3 seeds).
   - High sensitivity to crown boundaries (Oracle A boundary intervention adds +13.47 AP) remains a genuine geometric property of canopy segmentation.
   - Historical Q sanity successfully proved that a lightweight transformer query head (257k params) can overfit training fragments under Hungarian set matching.

2. **What Must Be Revised or Retracted**:
   - **Retract any claim of "generalization to unseen test data" on the historical Z1 test set.** The test set is methodologically connected to train/val and participated in past model selection loops.
   - **Retract the claim that "queries are intrinsically inferior to flow fields at structural grouping."** That historical pilot evaluated raw unthresholded query slots rather than Stage 2 accepted proposals.
   - **Revise G0 conclusions:** The failure of G0 merge-only grouping (EXP020/021) is a mathematical limitation of merge operators on multi-crown basins, NOT evidence against relational graph partitioning (G1).
   - **Revise historical OOF confidence calibration:** Historical OOF scores were spatially contaminated across overlapping folds.

---

## VI. Readiness Decisions

```json
{
  "DATA_INTEGRITY_READY": true,
  "EXPLORATORY_FGQ_READY": true,
  "CONFIRMATORY_BENCHMARK_READY": false,
  "EXTERNAL_HOLDOUT_REQUIRED": true
}
```

1. **`DATA_INTEGRITY_READY = true`**:
   - *Evidence:* 100% baseline SHA256 integrity check pass, zero historical data mutation, explicit support definitions formalized, canonical physical area formulas established, and 24 regression fixtures passing.
2. **`EXPLORATORY_FGQ_READY = true`**:
   - *Evidence:* Output Contract V3 active, multi-epoch crop sampler RNG fix operational, full-crown sampling policy established, F–G–Q comparison fairness frozen, and exploratory preflight gate passing.
3. **`CONFIRMATORY_BENCHMARK_READY = false`**:
   - *Blockers:* B01 (reused historical test set cannot serve as confirmatory holdout), B02 (Z1 splits have 235 connected tiles across boundaries), B03 (external site provenance unverified), and B05 (confirmatory margins not frozen). Confirmatory gate strictly fails-closed.

---

## VII. Minimal Next-Step Execution Plan

### Safe Exploratory Next Steps:
1. Run preflight verification before any training:
   ```bash
   ./cs2_venv/bin/python -m crown_segmentation_research.research_integrity.a06_hardening.preflight --mode exploratory
   ```
2. Execute Study Type 1 (Controlled Mechanism Comparison) on 1024×1024 full-crown training crops using frozen Swin-T backbone to compare F, G1, and Q under identical parameter budgets.
3. Evaluate all comparative instance segmentation metrics strictly at Stage 2 (Accepted Instances) with explicit $\tau_{accept}$ thresholds.

### Confirmatory Benchmark Path:
- Procure and register a new, geographically independent aerial orthomosaic dataset with documented flight and sensor provenance.
- Preregister primary evaluation endpoints, margins, and acceptance thresholds before unblinding holdout annotations.
