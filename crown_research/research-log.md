# Research Log

## 2026-09-19 — Bootstrap

Entered an existing project: `crown_segmentation_research/` already contains six ITC
segmentation paradigms benchmarked on BAMFORESTS, plus a just-completed boundary
refinement effort (MorphSnakes / adaptive spline smoothing / Dao ED metrics).

Reading the published table rather than accepting it: `PQ = SQ x RQ` with SQ = 67.7%
and RQ = 31.3% says the masks are fine and the *recognition* is broken. Boundary
refinement had moved mAP@50 by +1.66 — consistent with contour geometry being a
second-order term. Set the research question around locating the real error.

## 2026-09-19 — H0 locked and run

Pre-registered a 2x2 oracle ablation ({pred, oracle} seeds x {pred, oracle} energy)
with three numeric predictions, committed at `eb418ef` **before** running.

*Honesty note on the lock:* the repo's `.gitignore` carried a generic `experiments/`
rule, so `experiments/h0-error-budget/protocol.md` was silently excluded from
`eb418ef`; only `research-state.yaml` landed. That file did carry H0's statement and
the "> 55% mAP@50" prediction, so the prediction is still time-stamped pre-run, but
the full protocol was not. Fixed by adding `!crown_research/experiments/` to
`.gitignore`; future protocol locks will commit in full. Verified with
`git check-ignore` before proceeding.

**Result — all three predictions confirmed, decisively.**

- Oracle seeds + *predicted* energy: mAP@50 **31.3 -> 70.8** (predicted: > 55%)
- Predicted seeds + oracle energy: **29.9** (predicted: < 42%)
- Seed density `n_pred/n_gt` = **1.904** (predicted: > 1.15)

Marker selection owns essentially the entire recoverable error budget. The learned
boundary head is not the bottleneck for recognition, though it does cost ~7 points of
SQ and over half the attainable B-IoU (visible in cell C).

**Direction: DEEPEN.** The whole project pivots from contour refinement to seed
selection. Cell B (70.79 mAP@50) becomes the target to approach without an oracle.
Next: H1 scale-adaptive seed suppression, using the surface head U as its own local
crown-radius estimate.
