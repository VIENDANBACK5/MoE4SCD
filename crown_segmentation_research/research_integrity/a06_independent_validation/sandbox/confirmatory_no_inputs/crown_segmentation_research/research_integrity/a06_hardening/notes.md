# Notes: TreeFlowNet A06 Hardening Research & Findings

## 1. Historical Context & Evidence Sources

### Sources Inspected
- `crown_segmentation_research/execution_plan_audit_2026-10-15/BAO_CAO_A00_A05.md`: Core audit summary (15/10/2026).
- `crown_segmentation_research/execution_plan_audit_2026-10-15/blockers.json`: Blockers B01–B08.
- `crown_segmentation_research/execution_plan_audit_2026-10-15/measurement_spec_v3_draft.md`: Draft measurement specification.
- `crown_segmentation_research/TREEFLOWNET_AGENT_EVIDENCE_DOSSIER_ALL_IN_ONE.md`: Comprehensive 104-artifact self-contained package.
- `crown_segmentation_research/clean_benchmark/dataset.py`: CanopyCropDataset implementation.
- `crown_segmentation_research/clean_benchmark/targets_fast.py`: Fast raster targets (v1 and v2).
- `crown_segmentation_research/diagnosis_2026-10-11/fgq_revision_2026-10-13/q_sanity.py`: Q sanity fit implementation.
- `crown_segmentation_research/diagnosis_2026-10-11/fgq_revision_2026-10-13/grouping.py`: Grouping metrics & unit tests.
- `crown_segmentation_research/diagnosis_2026-10-11/study.py`: Oracle studies & matching tests.

### Key Numerical Baselines
- Tiles: Train 148, Val 139, Test 168 (Total 455).
- Annotations: Train 53,289, Val 15,155, Test 22,018 (Total 90,462).
- Spatial overlap (same-grid):
  - Train–Val: 14,094 shared source pixels (12.68 m²), 114 pairs, 5 GT-positive pairs, 0 after 1-px erosion.
  - Train–Test: 12,966 shared source pixels (11.67 m²), 89 pairs, 6 GT-positive pairs, 0 after 1-px erosion.
  - Val–Test: 11,895 shared source pixels (10.71 m²), 69 pairs, 0 GT-positive pairs, 0 after 1-px erosion.
  - Connected component: 272 valid-overlap pairs connect 235 Z1 tiles across all 3 splits.
- Cross-grid: 64 pairs between different mosaics; 2 pairs have valid ground support overlap remaining after 1-px erosion (0 GT-positive).
- Area differences (Metadata vs RLE):
  - Train: median Δ = 5.33 px (0.0511%), 61 change size bin.
  - Val: median Δ = 4.88 px (0.0594%), 4 change size bin.
  - Test: median Δ = 4.43 px (0.0714%), 22 change size bin.
- Q Sanity fit: 16 queries, 257,537 trainable params, 400 updates, all 11 positive targets were cropped fragments.
- Within-val OOF scorer: 378 cross-fold tile pairs share valid pixels, 370 have shared GT-positive support.

## 2. Identified Defect Catalog & Root Cause Hypotheses

1. **RNG Reseeding in Crop Dataset (`dataset.py`)**:
   - `rng = np.random.default_rng(i if self.fixed else (torch.initial_seed() + i) % (2 ** 32))`
   - In PyTorch multi-epoch training without DataLoader worker init or explicit epoch seed, `torch.initial_seed()` does not change across epochs in the same worker.
   - Consequence: Identical crop coordinates sampled every epoch, severely degrading data diversity.

2. **Crop Edge Truncation & Lack of Full-Crown Guarantee**:
   - `margin=0` causes 41.7% of crowns in 1024-px crop to touch the edge, learning fragmented targets.
   - In Q sanity (256x256), 100% of targets were fragments. Complete crown formation was untested.

3. **Conflation of Support Definitions**:
   - `unique_support` ($M_i \setminus \bigcup_{j \ne i} M_j$) differs from `priority_owner` (argmax/label raster assignment).
   - In Train: 63 annotations have empty unique support, but only 10 have empty priority support.
   - Rasterizing overlapping polygons into a single-label map changes instance boundaries and can drop instances.

4. **Raw Q-Slot vs Accepted Proposal Conflation**:
   - Evaluating raw Q slots (16 queries) under unique-cover structural grouping forces false abstention / overcounting because inactive queries still project mask logits.
   - Evaluation without explicit Stage 1/2 materialization corrupts fair comparison against F (field decoder) and G (graph/basin).

5. **Historical Test Set Contamination / Leakage**:
   - Test set tiles were repeatedly evaluated during model tuning, threshold sweeps, and diagnostic audits.
   - Boundary overlap connects test to train/val in same-grid graph.
   - Historical test set cannot serve as confirmatory holdout. Level 3 must be blocked (`CONFIRMATORY_NOT_READY`).
