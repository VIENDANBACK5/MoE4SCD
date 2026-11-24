# TreeFlowNet Epistemic Hierarchy & Evidence Levels

**Version:** 3.0.0  
**Effective Date:** 2026-10-15  
**Policy Purpose:** Prevent epistemic slippage, unverified promotion of claims, and retroactive reinterpretation of experimental findings.

---

## 1. Formal Taxonomy of Evidence Levels

To maintain scientific integrity across all TreeFlowNet documentation, code, and reports, every statement, metric, table cell, and conclusion MUST be strictly categorized into exactly one of the following seven levels:

| Level | Identifier | Epistemic Definition | Allowable Usage in Research |
|---|---|---|---|
| **L1** | `HISTORICAL_REPORTED` | A claim or metric recorded in historical text, slide, or log without verifiable local artifact recalculation. | Background context, historical narrative. Cannot be used as confirmatory evidence. |
| **L2** | `INSPECTED_ARTIFACT` | A static file, JSON summary, CSV ledger, or image existing in the repository that has been read, parsed, and hash-checked. | Descriptive audit of recorded state. Proves artifact existence and content, not validity of the producing code. |
| **L3** | `RECOMPUTED_NUMERICAL` | A metric re-derived deterministically from an existing inspected artifact using verified calculation logic. | Verifies mathematical consistency of historical tables and aggregations. |
| **L4** | `TEST_EXECUTED_CURRENT` | An invariant, regression test, or synthetic validation executed directly against the current repository codebase in this session. | Proves code functionality and regression defense on specified synthetic/fixture inputs. |
| **L5** | `REPRODUCED_FROM_SOURCE` | An end-to-end pipeline run from raw primary imagery to final evaluation metrics with identical outputs. | Full scientific reproduction. (Not authorized for full training in A06 scope). |
| **L6** | `SCIENTIFIC_HYPOTHESIS` | A proposed explanation, anticipated mechanism, or theoretical conjecture. | Stated explicitly as an unproven hypothesis for future testing. |
| **L7** | `UNRESOLVED_ASSUMPTION` | A premise, metadata field, or external factor that cannot be verified from available local artifacts. | Explicit blocker or limitation; cannot be treated as true by default. |

---

## 2. Mapping of TreeFlowNet Project Facts to Evidence Levels

### Category A: Core Numerical Results
1. **Validation AP ~39.03 (Field Decoder, 3 seeds)**:
   - Level: `HISTORICAL_REPORTED` (L1) / `INSPECTED_ARTIFACT` (L2).
   - Artifacts: `clean_benchmark/results/exp003_swin_t_s*_log.json`.
   - Verification Note: Log files exist and match reported AP; full inference was NOT rerun in A06.

2. **Oracle A (+13.47 AP) & Oracle B (+38.83 AP)**:
   - Level: `HISTORICAL_REPORTED` (L1) / `INSPECTED_ARTIFACT` (L2).
   - Artifacts: `diagnosis_2026-10-11/oracles/final_s42/metrics.json`.
   - Verification Note: Legacy diagnostics proving boundary sensitivity; not additive components of physical error.

3. **Structural Grouping (Core 4px: Old 66.79%, Final 72.83%)**:
   - Level: `HISTORICAL_REPORTED` (L1) / `INSPECTED_ARTIFACT` (L2).
   - Artifacts: `diagnosis_2026-10-11/fgq_revision_2026-10-13/pair_counts_by_tile.csv`.
   - Verification Note: Verified against pilot cache; confirmed to evaluate saved masks, not raw head representations.

4. **Q Sanity Fit (16 queries, 257k params, 400 updates, 11 fragments)**:
   - Level: `INSPECTED_ARTIFACT` (L2) / `RECOMPUTED_NUMERICAL` (L3).
   - Artifacts: `q_sanity/metrics_400.json`, `completion.json`, `crop_manifest.json`.
   - Verification Note: Proves query head can overfit 4 specific crops of fragments; does NOT prove instance segmentation generalization.

---

### Category B: Data Integrity & Spatial Audits
5. **Tile and Annotation Counts (455 tiles, 90,462 annotations)**:
   - Level: `RECOMPUTED_NUMERICAL` (L3) & `TEST_EXECUTED_CURRENT` (L4).
   - Artifacts: `registry/tiles.csv`, `registry/annotations.csv`.
   - Recomputation: Exactly 148 train, 139 val, 168 test tiles; 53,289 train, 15,155 val, 22,018 test annotations.

6. **Boundary Pixel Overlap in Same-Grid Z1 (14,094 tr-val, 12,966 tr-test, 11,895 val-test)**:
   - Level: `RECOMPUTED_NUMERICAL` (L3).
   - Artifacts: `spatial_overlap_pairs.csv`, `border_overlap_summary.json`.
   - Recomputation: Overlap strictly eliminated by 1-px erosion on both masks.

7. **Connected Component Across Splits (235 tiles connected by 272 pairs)**:
   - Level: `RECOMPUTED_NUMERICAL` (L3).
   - Artifacts: `overlap_components.json`.
   - Implication: Spatial graph connection is proven; independence assumption is falsified for Z1 contiguous blocks.

---

### Category C: Unresolved Items & Hypotheses
8. **Flight Campaigns & Acquisition Sensor Provenance**:
   - Level: `UNRESOLVED_ASSUMPTION` (L7).
   - Detail: Mosaic names `sblz1`, `sblz2`, `sblz3` and date string `20210902` do NOT establish independent surveys, sensor calibrations, or multi-site variability.

9. **Generalization of Q (Query Architecture) vs F (Flow)**:
   - Level: `SCIENTIFIC_HYPOTHESIS` (L6).
   - Detail: Whether query-based set prediction outperforms dense vector fields on uncropped individual tree crowns remains an open hypothesis to be tested under fair experimental protocols.

---

## 3. Iron Rule of Scientific Reporting

**NO claim may be promoted up the hierarchy without executing the corresponding verification step.**
- An unverified log cannot be cited as "reproduced".
- An audit finding cannot be claimed as a "new model result".
- An exploratory fit cannot be described as a "generalization benchmark".
