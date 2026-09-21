# H12 Analysis — Magnitude is necessary, not sufficient: the decoder has a window, not a floor

**Run:** 2026-09-20/21 · `train_wide.py` (seam target dilated 3 px → 7 px, nothing else
changed), 12 epochs from the H6 checkpoint. Control is H11 (same epochs, narrow target).

## Pre-registered predictions vs outcome

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | mean B on true seams > 0.30 | 0.186 → **0.357** | **CONFIRMED** |
| 2 | in-domain ΔAP in [−1.0, +2.0] | **−5.90** | **FAILED (badly)** |
| 3 | out-of-domain relative Δ > in-domain relative Δ | −19.5% vs −22.1% | CONFIRMED, *inverted* |

## Results

| split | H11 narrow | H12 wide | ΔAP | Δ rel |
|---|---:|---:|---:|---:|
| evalhold60 | 27.85 | 21.95 | **−5.90** | −21.2% |
| test2 (in-domain) | 26.03 | 20.29 | −5.74 | −22.1% |
| test1 (out-of-domain) | 5.27 | 4.25 | −1.03 | −19.5% |

Head diagnostic, seam magnitude / AUROC:

| split | H11 narrow | H12 wide |
|---|---|---|
| evalhold60 | 0.186 / 0.871 | **0.357** / 0.873 |
| test2 | 0.180 / 0.857 | **0.352** / 0.857 |
| test1 | 0.113 / 0.781 | **0.242** / 0.773 |

## What happened

**Widening the target did exactly what it was meant to do at the head.** Seam magnitude
nearly doubled again, ranking was preserved, and out-of-domain barrier height went from
29/255 to 62/255 — far clear of the floor the original head fell through at 14/255.
Prediction 1 is the cleanest confirmation yet of the project's recurring lesson: target
geometry moves magnitude where loss reweighting could not.

**And it cost 21% of AP.** The mechanism is not subtle in hindsight. A 7 px training
target produces a thick predicted ridge; the flat-basin watershed then erodes every
crown by the ridge half-width at every seam it touches. That is a systematic loss of
2–4 px on each side of each contact, which AP@[.5:.95] punishes hard. The off-seam
background also rose (0.041 → 0.107), so the flood now meets spurious barriers inside
crowns.

**Prediction 3 passed only in the inverted sense.** Out-of-domain lost *less* than
in-domain (−19.5% vs −22.1%). The sign is consistent with H10's floor account — the
split that was further below the floor is the one least harmed by overshooting it — but
it is not the confirmation I set out to get. H12 was framed as "extra magnitude pays
disproportionately out-of-domain", and what it shows is "excess ridge width hurts
disproportionately in-domain". Those are not the same claim, and I am not going to
pretend they are.

## What this adds to the sparsity law

Finding 18 said magnitude collapses with sparsity and every consumer reads magnitude.
H12 adds the other bound: **the decoder has a usability *window*, not just a floor.**
Below ~15/255 the flood leaks across seams (H10's out-of-domain failure); with a ridge
this thick it over-partitions and erodes (H12). The right target width is one that puts
the predicted ridge inside that window — and the 3 px band at sigma 0.8, which I called
"probably too thin to supervise" in H6's fallback, turns out to have been on the right
side of it.

## Does the H12 head survive a decode fix?

The obvious question is whether the H12 *head* is worse, or whether the *decode* is
mismatched to a thicker ridge — thinning or sharpening B at inference, with no
retraining, could recover the geometry while keeping the magnitude. That is an
exploratory probe (`code/thin_probe.py`), run after this analysis was written and
appended below rather than folded into the conclusions above.

## Status of H10's mechanism after H12

H12 does **not** falsify H10. Both H10's direct measurement (same 40% loss under shift,
different absolute landing points) and H12's inverted P3 point the same way. But H12
removes the naive corollary "more magnitude is always better out-of-domain" — the
benefit exists only up to the top of the window. H10 stays a mechanism; the *unbounded*
version of its implication is dropped.

## Carried forward

The H12 checkpoint is **not** the new best. Best remains H7's checkpoint at COCO AP
29.07 on validation (attribution settled by H11). The wide target is retained as a
negative result and as the upper bound of the usability window.
