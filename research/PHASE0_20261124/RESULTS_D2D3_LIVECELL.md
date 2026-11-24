# Phase 0 D2 + D3: LIVECell validation (2026-11-24). Code: code/d2_livecell.py, code/d3_livecell_cellpose.py.

## Data and protocol (Edlund et al. 2021, Nature Methods; read in full)
- 5,239 phase-contrast images of 704×520 px; 1.69 M cells; 8 cell lines. Annotation by professional annotators with two QA
  rounds (annotation manager, then a cell biologist).
- **Partially annotated by design** (p. 1044): "we intentionally chose not to attempt separation of a large cell mass into
  individual cells". So LIVECell is non-exhaustive, like QuebecTrees.
- Licence: CC BY-NC 4.0.
- Released baselines: Cascade Mask R-CNN ResNeSt-200 and CenterMask2 VoVNet, both detectron2 (AP about 47.9 / 47.8).
- Only train and val annotations were downloaded; the test split was not read.

## D2 (val, annotations only, border cells excluded)
- 160,540 cells; strong-stratum (r ≥ 4, d = 2 px) 12,479 (7.8%).
- By cell type: A172 11.8%, BT474 6.1%, BV2 5.5%, Huh7 11.0%, MCF7 4.0%, SHSY5Y 8.3%, SKOV3 19.6%, SkBr3 12.5%.
- About 13× more strong-stratum instances than QuebecTrees (931).

## D3, mechanism baseline: Cellpose 3 `livecell_cp3` (released; flow family), val

| d (px) | Absorption: non-touching / balanced / moderate / strong | Standardised diff (size quintile × density tertile), 95% image-bootstrap CI | Ratio |
|---|---|---|---|
| 1 | 0 / 0.053 / 0.142 / 0.288 | **+0.173 [0.133, 0.211]** | 2.49 |
| 2 | 0 / 0.051 / 0.141 / 0.284 | **+0.176 [0.137, 0.214]** | 2.64 |
| 5 | 0 / 0.047 / 0.136 / 0.272 | **+0.179 [0.141, 0.215]** | 2.92 |

Per cell type at d = 2, with the standardisation done within each cell type:

| Cell type | n strong | Strong rate | Standardised diff [95% CI] |
|---|---|---|---|
| A172 | 1,498 | 0.287 | +0.097 [0.029, 0.150] |
| BT474 | 775 | 0.196 | −0.078 [−0.155, 0.010] |
| BV2 | 2,298 | 0.443 | +0.227 [0.141, 0.432] |
| Huh7 | 329 | 0.237 | +0.058 [−0.028, 0.144] |
| MCF7 | 1,422 | 0.306 | +0.220 [0.125, 0.288] |
| SHSY5Y | 2,035 | 0.248 | +0.063 [0.019, 0.102] |
| SKOV3 | 822 | 0.283 | +0.139 [0.025, 0.247] |
| SkBr3 | 3,300 | 0.211 | +0.135 [0.046, 0.245] |

Consequence:
- recall 0.616;
- absorption is 22% of all misses;
- recall +0.085 if all absorptions were hits, +0.022 for the strong stratum alone.

## Findings
1. **On LIVECell (Cellpose), neighbour asymmetry raises absorption beyond what absolute size and crowding explain.** The
   effect is +0.18 overall, robust over d, and significantly positive within 6 of 8 cell types. In one type (BT474) it is
   negative and not significant.
2. **On QuebecTrees it does not** (9 models from 3 families; all standardised CIs contain 0; RESULTS_D2D3_QUEBECTREES.md).
3. So the asymmetry effect is dataset- and/or model-dependent. Two readings are still possible:
   - (a) LIVECell has a real asymmetry mechanism that QuebecTrees lacks, or that is masked there by coarse size bins and a
     small n (931 strong);
   - (b) the effect is specific to the flow-family decoder, since the only LIVECell baseline so far is Cellpose.
   
   **A task baseline on LIVECell is required** before (a) can be told apart from (b).

## Status of D3 on LIVECell
- Mechanism baseline: done.
- Task baseline: blocked. Installing detectron2 / CenterMask2 from GitHub (needed for LIVECell's released Mask R-CNN /
  CenterMask models) was denied by the permission system on 2026-11-24. Not attempted again; the owner decides.

## D3, task baseline: LIVECell's released anchor-free CenterMask2 V-99-eSE (ALL), val (added 2026-11-24, after the owner allowed the detectron2 install)
- Setup:
  - detectron2 was built in place against the system torch 2.4;
  - the authors' config was used, with only WEIGHTS changed to the local file;
  - the validation F1@0.5 cut-off is 0.55.
- Overall: recall 0.790; absorption is 34% of all misses; recall +0.071 if all absorptions were hits, +0.024 for the strong
  stratum alone.

| d (px) | Absorption: balanced / moderate / strong | Standardised diff [95% CI] |
|---|---|---|
| 1 | 0.038 / 0.111 / 0.318 | **+0.258 [0.157, 0.356]** |
| 2 | 0.037 / 0.110 / 0.310 | **+0.254 [0.159, 0.351]** (ratio 5.6) |
| 5 | 0.033 / 0.106 / 0.291 | **+0.246 [0.157, 0.339]** |

Per cell type, d = 2 (standardised diff [95% CI]):

| Cell type | Standardised diff [95% CI] |
|---|---|
| A172 | +0.068 [−0.020, 0.127] |
| BT474 | **−0.098 [−0.174, −0.020]** |
| BV2 | **+0.581 [0.246, 0.672]** |
| Huh7 | +0.011 [−0.075, 0.106] |
| MCF7 | **+0.262 [0.100, 0.394]** |
| SHSY5Y | **+0.100 [0.045, 0.149]** |
| SKOV3 | +0.126 [−0.024, 0.251] |
| SkBr3 | **+0.120 [0.019, 0.243]** |

## Updated findings (two families on LIVECell)
1. The asymmetry effect on LIVECell is NOT specific to the flow decoder. The detection-based CenterMask2 shows it more
   strongly than Cellpose: +0.25 vs +0.18 overall, with positive and significant effects in the same direction for BV2,
   MCF7, SHSY5Y and SkBr3.
   - So reading (b), "a flow-decoder artefact", is rejected for LIVECell.
2. It is cell-type dependent: strong in BV2 and MCF7, absent in Huh7 and A172 (CenterMask), and REVERSED in BT474 for both
   models.
3. QuebecTrees shows no asymmetry effect for 3 families. The effect therefore depends on the dataset (or the object
   regime), not only on the model family.
4. **Open check before interpreting (annotation artefact risk).** LIVECell leaves dense cell masses unseparated by design.
   - A "strong" small cell next to a ≥ 4× larger GT object might sit next to an annotated multi-cell blob rather than a
     single large cell.
   - Previews of strong-stratum absorptions (BV2, MCF7) must be inspected before the effect is attributed to instance
     formation.
