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

## 2026-09-19 — H1 run: two of three predictions refuted

Locked H1 at `59ec567`. Arm A (global radius) vs Arm B (scale-adaptive radius from
superlevel-set extent).

**Both headline predictions failed.** Arm B lost to Arm A (51.20 vs 52.87), and Arm A
blew past its predicted <= 48% ceiling. The cause of the second failure was my own
grid: it peaked at its upper corner (md=56, ma=0.45), which means it never found an
optimum. Extending to md <= 192 revealed an interior plateau at **66.15** (md=160,
ma=0.50).

Root cause, in hindsight obvious: BAMFORESTS crowns average ~146 px equivalent
diameter at 1024 px crops, and the published decode constant was `min_distance = 24`.
Off by ~6x. That single constant accounts for most of the gap H0 measured.

Dropped H1-b. The superlevel-set radius estimator is too noisy — components merge at
low `t` and shatter at high `t` — and per-instance target normalisation means scale
would have to be *supervised* to be usable, not recovered at decode time.

**Integrity check run before believing any of it.** The settings were selected on the
same 30 crops used to report, so I re-scored frozen settings on 60 disjoint crops
(eval idx 100-159). Tuned setting transfers at -0.61 (66.15 -> 65.54): not overfit.
But the baseline jumps 31.33 -> 44.29 on the hold-out, so **the original 30-crop
subset is unrepresentatively hard** and all previously published numbers on it —
including the MorphSnakes 32.99 — are pessimistic by ~13 points. Honest headline is
the hold-out delta: **44.29 -> 65.54 (+21.2)**.

**Direction: DEEPEN.** The seed audit found the real structure: loose suppression hits
83.1% of crowns with 0.82 duplicates each; tight suppression kills duplicates but
misses 37.5% of crowns. Global NMS is stuck on that frontier. Next (H2): over-seed to
keep the 83% hit rate, then merge fragments using the boundary head's seam evidence.
