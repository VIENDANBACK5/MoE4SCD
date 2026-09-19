# Crown Segmentation Method Design v2 — FINAL for this design cycle

## Material Passport

- Origin: supersedes `reports/method_design_dense_crown_separation_v1.md`,
  reordered to match evidence from `reports/g4d_rpn_roi_instrumentation.md`
  (RPN instrumentation + score-vs-mask split + size/contrast correlation,
  all run this session on the real frozen `maskrcnn_seed42_best.pth`
  checkpoint and full BAM_val, 382 images / 8,747 crowns).
- Verification Status: DESIGN ONLY — not implemented, not trained.
- v1 is not deleted; its multi-seed merge design (large/multi-lobed crowns)
  is carried forward unchanged into Section 3 below, because that failure
  mode is independently confirmed by Freudenberg et al. (2022) and is
  untouched by this session's RPN/ROI findings.

## 1. What the evidence actually says, in one paragraph

SAM2/Mask R-CNN's headline miss rate on this data is **not** a proposal-
generation problem: RPN recall for crowns the model ultimately fails on is
94-99.98% at IoU>=0.50 (G4D). Of 668 measured failures, 81% (541) are the box
classifier assigning a well-localized proposal a tree-class score with
median 0.18 against a 0.40 operating threshold — not "almost passing," 90%
score below 0.35. That under-confidence is **not explained** by crown size
(r=0.137) or local RGB contrast (r=0.021), the two covariates this project
has instrumented. Mask quality itself is fine for 97% of failures. Given
this, the method with the largest expected effect is one that (a) improves
box-classifier confidence through richer evidence per proposal rather than
guessing at a cause, and (b) separately fixes the independently-confirmed
large-crown over-segmentation problem, which is architecturally unrelated.

## 2. Two independent problems, two independent fixes

Do not conflate these into one representation change — they have different
root causes and different owners in the architecture.

| Problem | Confirmed mechanism | Fix target |
|---|---|---|
| 81% of failures: proposal fine, classifier unconfident | Unknown visual cause; ruled out size/contrast | Box classification head — Section 3 |
| Large/multi-lobed crown over-segmentation (independent confirmation: Freudenberg et al. 2022, +Fan et al. 2024 baseline comparison) | Single-center-per-instance assumption shared by every reviewed representation family | Post-hoc region merge — Section 4 (unchanged from v1 Section 4) |

## 3. Fix for classifier under-confidence: multi-view evidence pooling

Since the cause is not yet visually diagnosed, the design commits to a
mechanism that is robust to *not knowing* the exact cause — give the
classifier strictly more evidence per proposal, rather than a targeted patch
for one hypothesized cause:

\[
\text{score}(R) = \text{box\_predictor}\Big(\text{pool}\big(f(R_{\text{tight}}),\; f(R_{\text{ctx}})\big)\Big)
\]

- \(R_{\text{tight}}\): the original RPN proposal box (current behavior).
- \(R_{\text{ctx}}\): the same box dilated by a fixed physical margin
  (metres via GSD, reusing the `A_physical = A_pixels * GSD^2` convention
  from `experiments/g4a_physbound/model.py`, not a pixel-count margin, so it
  transfers across 5/10/20cm the same way the multi-seed merge rule does).
- \(f(\cdot)\): the same `box_roi_pool -> box_head` path already in the
  frozen model, run twice (tight + context), fused before
  `box_predictor` by concatenation + a small learned linear layer
  (initialized so the tight-only pathway is preserved at init, matching this
  project's established pattern of identity/near-identity initialization for
  new branches so a warm-started fine-tune does not regress before it
  improves — same principle as G4A's zero-initialized FiLM layer).

This is deliberately *not* presented as a literature contribution — context-
padded RoI pooling for small/ambiguous object classification is standard
practice in the object-detection literature generally, not from the 10-paper
segmentation-representation survey. What is specific to this project is
attaching it precisely where G4D localized the failure, instead of applying
it everywhere by default.

**Pre-registered promotion rule** (mirrors G4A's decision-gate format): promote
this branch only if, on the same 541-crown validation set:
1. Mean tree-class score on those 541 crowns increases by >= 0.05 absolute, AND
2. Recall@IoU0.50 on `retained_match` crowns does not regress by more than
   0.005 absolute (i.e. it must not buy confidence on hard crowns by
   generally loosening the classifier).

Failure is informative: if score does not move, the missing evidence is not
spatial context, and the next hypothesis must come from the qualitative
image inspection this document still recommends before further architecture
changes (Section 6).

## 4. Fix for large-crown over-segmentation (carried forward from v1, unchanged)

\[
I \rightarrow \big(P_{fg}(x,y),\; H(x,y),\; V(x,y)\big) \rightarrow \text{marker-controlled watershed} \rightarrow \{R_1,\dots,R_K\}
\]

Merge adjacent regions:

\[
\text{merge}(R_i, R_j) \iff \| E(R_i) - E(R_j) \|_2 < \delta_E \;\wedge\;
d_{\text{phys}}(c_i, c_j) < \delta_d \cdot \text{GSD}
\]

\(E(R)\) a learned per-region appearance embedding (discriminative-loss style,
De Brabandere et al. 2017, applied post-hoc to watershed regions rather than
replacing watershed). \(\delta_E, \delta_d\) unvalidated; grid search on
BAM_val only, per `benchmark/experiment_protocol.yaml`.

This piece operates on the **existing Mask R-CNN mask output**, not only on a
future dense H/V rewrite — it can run as a standalone post-processing pass
over any set of per-image instance masks, so it does not require Sections
1-3's classifier fix to land first, and does not require replacing Mask R-CNN
at all. The full dense H/V representation from v1 remains a valid longer-term
direction but is no longer the lead recommendation, since it was motivated by
an RPN-drop mechanism G4D showed is not what is actually happening.

## 5. GSD conditioning (unchanged, reuse)

`experiments/g4a_physbound/model.py`'s identity-initialized FiLM module
attaches to either the classifier fix (Section 3) or the merge module
(Section 4) unchanged; blocked previously only by data availability
(`Bamberg_coco2048.zip`), which is now resolved.

## 6. What remains open — do not implement past this without it

The 541-crown under-confidence cause is unidentified. Section 3 is designed
to be robust to that (more evidence, not a targeted patch), but the
pre-registered promotion rule in Section 3 is exactly the check that will
tell us whether "more context" was ever the right kind of additional
evidence. Qualitative image inspection of a sample of the 541 crowns (same
method as this session's DTE-aerial visual QC) is still the cheapest
possible next diagnostic and should run **before or alongside** Section 3
implementation, not be skipped because a design now exists on paper.

## 7. Implementation order

1. Section 4 merge module — standalone, no dependency on anything blocked,
   testable today on existing G1B mask outputs.
2. Section 3 context-pooling classifier fix — requires a short fine-tune from
   the G1B checkpoint (warm-start, same pattern as G4A), gated by the
   promotion rule above.
3. Section 5 GSD-FiLM attaches to either once its host module is validated.
4. Qualitative inspection (Section 6) can and should run in parallel with (1).
