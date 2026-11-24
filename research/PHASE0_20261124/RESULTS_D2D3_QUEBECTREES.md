# Phase 0 D2 + D3: QuebecTrees validation (2026-11-24). Code: code/absorption.py, code/d2d3_quebectrees.py.

Setup:
- Existing predictions only.
- Cut-offs are the validation F1@0.5 maxima:
  - Mask R-CNN s42 / 43 / 44: 0.45 / 0.65 / 0.70;
  - TFN-c5: 0.75 / 0.75 / 0.70;
  - TFN-M0': 0.75 for all three seeds.
- One-to-one matching.
- 3,671 unique non-edge GT crowns after de-duplicating the 50%-overlapping tiles (the occurrence nearest its tile centre is
  kept).

## D2 (annotations only)
- Crown area: median 7.5 m²; 5–95% range 1.5–32.6 m².
- Strata at d = 2 px:

| Stratum | n |
|---|---|
| non-touching | 103 |
| balanced (r < 2) | 1,638 |
| moderate (2 ≤ r < 4) | 999 |
| strong (r ≥ 4) | 931 (25%) |

- Strong crowns < 2 m²: 190.
- d = 1 and d = 5 px give nearly the same counts (strong 917 / 970).

## D3 (absorption = not hit AND one prediction covers ≥ 50% of the crown and ≥ 50% of a neighbour)

| Model | Raw absorption: strong / balanced | Standardised diff (size quintile × density tertile), 95% tile-bootstrap CI | Absorption share of all misses | Recall gain if all absorptions were hits |
|---|---|---|---|---|
| Mask R-CNN s42 | 0.190 / 0.089 | +0.004 [−0.061, +0.072] | 0.44 | +0.125 |
| Mask R-CNN s43 | 0.179 / 0.076 | −0.006 [−0.066, +0.052] | 0.40 | +0.114 |
| Mask R-CNN s44 | 0.142 / 0.051 | +0.013 [−0.037, +0.062] | 0.31 | +0.087 |
| TFN-c5 s42 | 0.125 / 0.063 | +0.018 [−0.023, +0.063] | 0.31 | +0.089 |
| TFN-c5 s43 | 0.131 / 0.058 | +0.017 [−0.029, +0.059] | 0.32 | +0.088 |
| TFN-c5 s44 | 0.125 / 0.061 | −0.004 [−0.056, +0.045] | 0.31 | +0.086 |
| TFN-M0' s42 | 0.117 / 0.051 | +0.004 [−0.037, +0.048] | 0.27 | +0.075 |
| TFN-M0' s43 | 0.121 / 0.051 | +0.031 [−0.008, +0.068] | 0.28 | +0.080 |
| TFN-M0' s44 | 0.126 / 0.056 | −0.003 [−0.051, +0.044] | 0.29 | +0.084 |

The same pattern holds at d = 1 and d = 5 px (standardised diffs between −0.004 and +0.035, all CIs contain 0).

## Findings (verified on this data, 9 models from 3 families)
1. **Absorption is a major failure for every family.** It is 27–44% of all missed crowns and worth +7.5 to +12.5 recall
   points. Mask R-CNN absorbs MORE than TFN: raw strong-stratum rate 14–19% vs 12–13%.
2. **Raw absorption is about 2× higher next to a ≥ 4× larger neighbour. But after standardising for absolute crown size
   and local density, the asymmetry adds nothing measurable** (all 27 model × d CIs contain 0; point estimates ≤ 0.035).
3. Therefore, on QuebecTrees, the driver is **small absolute size in dense touching scenes**, not neighbour asymmetry as
   such.
   - The working name "small-to-large absorption" describes WHERE it happens: small crowns often have big neighbours. It
     does not describe WHY.
4. Restricted to crowns < 2 m², the balanced stratum has only 35 crowns, which is too few to test asymmetry there; TFN
   point estimates are positive (+0.005 to +0.17). Unresolved.

## Limits
- Size quintiles are coarse: the lowest is < 3.3 m², so some residual confounding within a bin is possible.
- One dataset. The validation tiles' spatial blocks are few, so the CIs are approximate.
- Non-exhaustive labels (Cloutier 2023): absorption into an UNLABELLED neighbour is not counted, because a neighbour must
  be GT.

## Consequence for Phase 0 (not silently applied; needs the owner's decision)
- G1/G2 support "absorption of small instances in dense touching scenes, shared across architectures" as an observable
  failure.
- They do NOT support "neighbour size asymmetry" as the defining variable.
- Proposed revision of the provisional target: *instance absorption as a function of absolute instance size and crowding*.
  To be confirmed on LIVECell before any wording is locked.
