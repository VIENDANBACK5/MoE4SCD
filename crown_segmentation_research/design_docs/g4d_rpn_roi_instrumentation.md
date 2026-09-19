# G4D — RPN vs ROI/Mask Head Instrumentation

## Material Passport

- Origin: direct execution of the "Correct Next Gate" specified in
  `reports/g4c_missed_crown_characterization.md` ("Do not build GSD-conditioned
  anchors yet. Instrument the frozen model at three stages...").
- Verification Status: VERIFIED — full BAM_val (382 images, 8,747 GT crowns),
  frozen `maskrcnn_seed42_best.pth`, native + 10cm-degraded conditions.
- Script: `crown_segmentation_research/code/rpn_roi_instrumentation.py`.
- Output: `crown_segmentation_research/experiments/instrumentation_results/rpn_stage_recall.csv`
  (17,494 rows = 8,747 GT crowns x 2 conditions), joined to the existing
  `per_crown_transitions.csv` (G4C) in `crown_segmentation_research/experiments/instrumentation_results/rpn_stage_recall_joined.csv`.

## Method

For each GT crown, the frozen model's `RegionProposalNetwork` submodules are
invoked directly (not reimplemented) to expose two intermediate stages that
the normal `model(...)` forward call does not return:

- **Stage A (pre-NMS)**: proposals after per-FPN-level top-k selection
  (`RegionProposalNetwork._get_top_n_idx`), before NMS.
- **Stage B (post-NMS/top-k)**: the exact proposals `filter_proposals` hands
  to the ROI heads — what the ROI/mask head actually sees.

Both are box IoU (`torchvision.ops.box_iou`) against GT boxes, computed in the
model's internally-resized coordinate space (the default
`maskrcnn_resnet50_fpn` transform resizes the 2048x2048 input to ~800x800;
GT boxes are scaled by the same factor before comparison).

**Stage C (final detection survival)** reuses the already-validated mask-IoU
Hungarian match already stored in `per_crown_transitions.csv`
(`native_matched` / `target_10cm_matched` / `transition`) — not recomputed.

## Result

Stage A/B recall at IoU >= 0.50, computed **only for crowns whose final Stage-C
outcome is a failure at 10cm** (`lost_strict_miss`, `lost_partial_overlap`),
compared against the `retained_match` population:

| Transition (10cm) | n | Stage A recall@0.5 | Stage B recall@0.5 |
|---|---:|---:|---:|
| `lost_strict_miss` | 314 | 95.5% | 94.3% |
| `lost_partial_overlap` | 354 | 99.2% | 97.5% |
| `retained_match` | 6,341 | 99.98% | 99.97% |

The same pattern holds at native resolution (proposals survive at 100%/100%
for both lost groups there too — consistent, since these crowns are matched
at native and only fail after degradation).

## Interpretation

The RPN recall gap between failing crowns and retained crowns is small (94-99%
vs 99.97-99.98%) — **proposals overwhelmingly survive** even for crowns the
final model output ultimately fails to match, including the `lost_strict_miss`
population (small + low-contrast crowns), not only `lost_partial_overlap` as
originally hypothesized in G4C.

Per the gate's own pre-declared decision rule ("A proposal method is justified
only if proposal recall collapses disproportionately for the
strict-miss/small-crown group. If proposals survive but ROI/final masks fail,
the method target must move downstream."): **proposals survive; the method
target moves downstream to the ROI classification and mask head.**

## Consequence for method design

This directly revises the risk assessment in
`reports/method_design_dense_crown_separation_v1.md` Section 6, which left
open whether a proposal-free architecture was needed because the strict-miss
mechanism was unconfirmed. It is now confirmed to be a downstream (ROI/mask)
effect for both failure populations, not an upstream (RPN/encoder-activation)
effect. This does not disqualify a dense/proposal-free redesign on its own
merits, but removes the specific justification "SAM/Mask R-CNN loses crowns
because the proposal is never generated" as the mechanism for BAM's measured
failures — that mechanism is not what is happening here.

## Follow-up: classification score vs mask quality (RESOLVED)

Script: `crown_segmentation_research/code/roi_score_vs_mask_split.py`. For each of
the 668 `lost_strict_miss`/`lost_partial_overlap` crowns, scored the exact
best-IoU Stage-B proposal through `model.roi_heads.box_predictor` (tree-class
softmax score) and `model.roi_heads.mask_predictor` (mask IoU against GT,
independent of the score threshold), bypassing box-regression refinement
(simplification noted in the script docstring).

| Cause | n | % |
|---|---:|---:|
| `score_filtered` (tree-class score < 0.40) | 541 | 81.0% |
| `unexplained_by_score_or_mask` (score & mask both fine) | 108 | 16.2% |
| `mask_quality_fail` (score >= 0.40, mask IoU < 0.50) | 19 | 2.8% |

For the 541 `score_filtered` crowns, scores are not borderline: median 0.18,
90.4% below 0.35, only 11.8% below 0.05 (i.e. not "confidently rejected as
background" either — mostly *uncertain*, not confidently wrong). Lowering the
0.40 threshold to catch half of them would require dropping to ~0.18,
trading away precision broadly rather than fixing these crowns specifically.

**Conclusion: the dominant failure mode (81%) is classifier under-confidence
on otherwise well-localized proposals, not mask decoding quality.** This
reprioritizes the method-design intervention below.

## Follow-up: is under-confidence explained by size or contrast? (NO — null result)

The natural next hypothesis was that the same covariates driving G4C's
strict-miss population (small area, low local RGB contrast) also drive this
81% classifier under-confidence. Joined `roi_score_vs_mask_split.csv` to the
`area_m2`/`rgb_contrast_l2` columns already in `per_crown_transitions.csv`
(no new model run, join + correlation only):

- `tree_class_score` vs `area_m2`: r = **0.137** (weak).
- `tree_class_score` vs `rgb_contrast_l2`: r = **0.021** (negligible).
- By area quartile: mean score 0.20 (smallest) -> 0.32 (q3) -> 0.30 (largest)
  — a mild, non-monotonic trend, not the strong size effect seen in G4C.
- By contrast quartile: 0.25 / 0.29 / 0.26 / 0.28 — flat, no trend.

**This does not support the Section 4bis hypothesis in
`method_design_dense_crown_separation_v1.md` that size/contrast explain the
under-confidence** — that document has been corrected to say so rather than
proceeding on an unsupported premise. The classifier is broadly unconfident
across this population regardless of size or contrast; the actual driver is
unidentified. The next honest diagnostic is qualitative (look at a sample of
the 541 crowns directly, images + their proposal crops), not another
tabular covariate correlation, since the two covariates this project has
instrumented so far do not explain the effect.
