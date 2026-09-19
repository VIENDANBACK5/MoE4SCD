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

## 2026-09-19 — H2 through H5: the bottleneck moved, twice

**H2 (over-seed then merge)** — all three predictions refuted. The oracle merge, which
bounds any possible merge rule, reached only 69.74. Combined with H0's oracle seeds
(69.03), three independent perfect-marker routes converge at 69-70. Marker selection,
which H0 had identified as owning the budget, is now closed as a direction.

**H3 (canopy gate)** — two of three refuted, and the sharpest result of the session.
H0's 2x2 had silently held the canopy gate fixed in all four cells; the full 2^3
factorial shows every single-factor oracle *hurts* (canopy alone -24.3, energy alone
-32.5) while the pair gains +12.4, reaching **97.30 mAP@50** with oracle seeds. The
decoder H2 had implicated is nearly perfect. ~30 points sit in the heads.

**H4 (decode-time coupling)** — null, and diagnostic: `canopy - lambda*B` moved mAP@50
by <0.2 across lambda in [0,2], which can only happen if B is ~0 everywhere. Triggered
a head-quality diagnostic.

**Head diagnostic** — B scores AUROC 0.866 against GT seams with median 0.0011 and mean
0.105 on true seam pixels. A calibration failure from the ~6% positive rate, not a
learning failure. It retro-explains H2 (no seam ever reached the 0.35 threshold) and H4
(subtracting 0.019 changes nothing).

**H5 (rank-based use)** — two of three refuted. Rank-transforming B cost 9.4 points:
the ordering inside the near-zero bulk is noise, and spreading it manufactures ridges.
Percentile merging gave +1.4 and is the best configuration found (66.91 hold-out).
Arm D confirmed a corrected mechanism for H3: the canopy gate cuts 32.1% of true seam
pixels as hard background barriers, at 6x the rate of crown interiors — while ranking
seams at only AUROC 0.567. I withdrew H3's "canopy is the seam detector" claim in H5's
protocol and tested the replacement rather than leaving it standing.

**Direction: PIVOT to training.** Five decode-side hypotheses after H1 have returned
+1.4 in total. Next is H6: retrain the boundary head with a class-balanced objective,
success criterion mean B on true seams > ~0.4. Progress report published for review.

## 2026-09-19 — H6 launched; two findings while it trains

**H6 locked** (`train_calibrated.py`, a copy of `train_sharp.py` with only the boundary
term changed: per-batch class-balanced BCE + asymmetric Tversky alpha=0.3/beta=0.7).
`train_sharp.py` left untouched so the baseline stays reproducible. 12 epochs,
warm-started from `best_sharp_watershed.pth`, seeded. ~76 min.

The training log corroborated the diagnosis before the run even started: under the old
loss the boundary term sits at 1.08 -> 1.05 through all of epoch 1 and never moves. It
is essentially the Dice term pinned near 1.0, because predicted-vs-target overlap never
materialises.

**Open question 3 answered.** Missed crowns are not suppressed, they are `never_lit`:
16.5% of GT crowns (loose) and 28.1% (tuned) never raise the surface head above
`min_apex` anywhere inside them, and their median area is 1,899 px against 24,382 px
for all crowns. Per-instance normalisation strikes twice — it defeated H1's adaptive
radius and it caps recall here.

**A discarded asset found.** `targets_sharp` computes a fourth target, a Gaussian apex
peak with sigma fixed independent of crown size, and `dataset.py` ships it as
`batch["apex"]` every step. Nothing reads it; the model has three channels. The one
scale-invariant signal in the project is being thrown away, and it is exactly what the
small-crown failure needs.

**H7 locked now, queued behind H6** so its predictions are on record before H6's results
are known. The headroom decomposes cleanly from the H3 cube: boundary+canopy quality is
worth ~+12 (65.5 -> 78.0) and seeding is worth ~+19 (78.0 -> 97.3).

## 2026-09-19 — metric integrity closed out; H6 epoch-1 probe

**True COCO AP implemented** (`crown_research/src/coco_ap.py`) and the key
configurations re-scored. The ordering is unchanged — every conclusion survives — but
the absolute numbers are far lower and the honest headline for the decode-side work is
**COCO AP 15.81 -> 24.32**, not F1 44 -> 67. AP@[.5:.95] sits at 24.3 against AP50
56.6, while the oracle holds 84.5 against 96.0, which independently confirms that mask
quality at strict IoU is the larger deficit.

**H6 epoch-1 probe** (early read, 1 of 12 epochs, not the result):
- mean B on true seams 0.105 -> 0.161, still far from the 0.40 criterion
- mean B off seams 0.013 -> 0.043 — rising *faster* in proportion, so the separation
  ratio has fallen from 8.0x to 3.8x. Worth watching: the balanced loss may be lifting
  the whole field rather than the seams specifically.
- AUROC holds (0.866 -> 0.862), as predicted
- geometry improves (mAP@75 28.5 -> 32.8, B-IoU 11.7 -> 13.9, SQ 72.1 -> 73.7) while
  recognition drops (F1@50 65.5 -> 62.7) and the instance ratio falls 0.756 -> 0.684

That trade is mechanically sensible — a stronger ridge tightens masks and over-separates
— but it means P4 may fail even if P1 succeeds. Letting the run finish before judging.

## 2026-09-19 — H8: the rule transfers, the model does not

All six pre-registered predictions passed — the first clean sweep since H0. One
`eval`-chosen suppression setting is within 1.35 F1@50 of each test split's own
optimum, and the published 24 px collapses off its tuning split (3.18 seeds/crown and
COCO AP 0.86 on TestSet1). The practical recommendation is now supported at three
splits.

**But the more important result is the one the predictions did not ask about.**
TestSet1 scores COCO AP 2.98 even at its own optimum, against 23.85 on the eval
hold-out, while TestSet2 matches eval (21.3). The distinguishing statistic is crown
scale: TestSet1's crowns are 2.2x larger in area (D = 314 px vs 211 px). The decoder
conclusions hold there; the heads do not. Every ceiling and error budget in this
project was measured at one crown scale, so their external validity is now explicitly
bounded.

Caught my own repeated mistake: test1's swept optimum landed on the upper edge of the
grid (240 of a 40-240 range) — the same boundary failure that made H1 report a false
optimum. Extending the grid before quoting the exploratory `min_dist ~ 0.76 x D` ratio
as a rule; it currently rests on two resolved points (eval 0.760, test2 0.756), not
three. Added to Lessons so it stops recurring.

**Priority change.** Scale generalisation is now the top open question, ahead of
further head work — if the model cannot survive a 1.5x scale shift, H6 and H7 are
optimising inside a regime it cannot leave.

## 2026-09-19 — H8 follow-up: the ratio did not survive the boundary check

Extended test1's grid to 460 and its optimum resolved at md=280, interior. The
exploratory `min_dist ~ 0.76 x D` regularity **does not hold**: ratios are 0.760
(eval), 0.756 (test2), 0.892 (test1). The first two agree to three decimals because
they share a crown scale (D ~ 211 px), not because a universal coefficient exists.

This is the second time the boundary check changed a conclusion, and the first time it
stopped me reporting something false rather than merely finding a better number. The
honest statement is weaker: the optimum sits at roughly **0.75-0.9 x mean crown
diameter**, enough to replace a blind 24 px constant, not a calibration law.

The pre-registered ±30% prediction actually passes more comfortably after the
extension (11% off instead of 24%), so H8's verdicts are unchanged.

Also noted: on test1 the two metrics disagree about the optimum (F1@50 peaks at 280,
COCO AP at 240) — further evidence that test1 is a degenerate regime, not a harder one.

## 2026-09-19 — H6 result: predictions wrong, intervention right

Three of four predictions failed, and the run is still the biggest single step since
the decode constants: **COCO AP 24.32 -> 26.84, AP75 16.48 -> 20.40**, F1@75
30.45 -> 37.06, SQ +1.8, B-IoU +2.4, ED 0.638 -> 0.595. F1@50 flat (+0.07).

The failures were mine, not the data's:
- **P1's 0.40 threshold was invented.** The flood relief is `boundary*255`, so
  0.105 -> 0.186 raises the inter-crown barrier from ~27/255 to ~47/255 — nearly double,
  and plainly enough to change flooding. The criterion failed; the mechanism held.
- **P4 predicted a recognition gain from a boundary intervention**, which was an error
  in reasoning: recognition is seeding-bound and H6 did not touch seeding.
- **P3 failed informatively.** Under the new head merging now *hurts* (26.84 no-merge
  vs 26.29 percentile, 25.66 absolute). Merging existed to repair over-separation from
  a weak ridge; the ridge is stronger, so the repair deletes real boundaries. **Merge
  step retired.**

Calibration is half-fixed: seams 0.105 -> 0.186 but background 0.013 -> 0.041, so the
separation ratio *fell* 8.0x -> 4.5x. Balanced weighting lifts the whole field instead
of sharpening contrast. The protocol's pre-registered fallback now applies — change the
**target** (3 px band at sigma 0.8 may be too thin to supervise), not the loss.

Next: H7 apex head, warm-started from the H6 checkpoint.

## 2026-09-19 — literature search (overdue), and two scoping corrections

Stepped back and noticed a real gap in method: I entered an existing project and ran
eight hypotheses without ever establishing the benchmark's published numbers. Fixed
that. Notes in `crown_research/literature/survey.md`.

**Where the project stands.** TreeCoG reports **AP 53.21 / AP@50 73.14** on BAMFORESTS;
FG-TreeSeg reports mAP@50 67.31. This project's best is **AP 26.84 / AP@50 58.06** —
roughly half. Protocols are not verified to match, but a 2x gap is too wide to be
protocol alone. The contribution here is **diagnostic, not competitive**, and any
write-up has to say so.

**Correction 1 — H2 was scoped too widely.** TreeCoG uses exactly H2's paradigm
(deliberate over-segmentation then merging) and is the SOTA. H2's oracle bounded
merging *of this pipeline's basins*, not the paradigm. TreeCoG differs in both things
H2 held fixed: a contour network for over-segmentation, and a learned GNN merger rather
than a threshold on one scalar — precisely the failure H2 and H5 diagnosed. H6 supports
the reading, since a stronger ridge made merging harmful.

**Correction 2 — H8 under-described the TestSet1 collapse.** TestSet1 is the
benchmark's *designated* out-of-domain split: the complete Hain-AOI, a city park rather
than managed forest, "of severely different structure", captured with another UAV and
sensor, at a different altitude, date and time of day. Crown scale is one symptom of a
multi-factor shift, not the cause. A scale-only remedy may not help, so the next
experiment must separate the factors — cheapest is test-time rescaling of TestSet1.

Both corrections applied in place to the affected analyses rather than left standing.

## 2026-09-20 — H9 and H10: the domain gap, decomposed

**H9** (test-time rescaling) came back a clean, well-powered null on its main
prediction. Both controls behaved: TestSet1's optimum landed *exactly* on the
scale-matching factor s = 0.672, and the TestSet2 control was perfectly flat (delta
0.00, rescaling strictly harmful). So the effect measured really is the scale axis —
and it recovers only **7.3%** of the gap. ~93% of the transfer failure is appearance and
structure. Crown-diameter framing dropped as an explanation of the domain gap;
multi-scale training deprioritised.

**H10** resolved the confound H9 logged rather than leaving it hanging. Both checkpoints
swept the same 15-cell decode grid on the same crops. **P3 failed — and P3 was my bet
against my own earlier observation.** The +83% was not a decode artefact: at matched
decode it is +83.0% out-of-domain against +18.6% in-domain, and the in/out ratio narrows
7.15x -> 4.63x. First intervention in the project to move that ratio.

**The mechanism was not what I assumed.** I expected calibrated heads to degrade more
gracefully. They do not: both lose the same ~40% of seam magnitude under the shift and
their AUROC degrades identically. What differs is the absolute level — the flood runs on
`boundary*255`, so out-of-domain the old head sits at ~14/255 and the calibrated one at
~27/255. **There is an absolute usability floor; the shift costs both heads the same and
only one can afford it.** Confirmed by the old head's seam/background *ratio* actually
being higher out-of-domain (8.69 vs 7.23) — ranking is fine, magnitude is not — and by
the canopy head barely degrading.

This makes H6's pre-registered fallback (widen the seam target) the highest-value
remaining training change: it is now a robustness lever, not just an accuracy one.

Also caught a real bug: `pred_cache`'s CLI ignored `--ckpt` and always loaded
`best_sharp`, so H10's first caches were duplicates. Verified by diffing cached boundary
means before trusting them. Added to Lessons.
