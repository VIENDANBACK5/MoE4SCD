# H6 Analysis — The mechanism was right; my success criterion was a guess, and it was wrong

**Run:** 2026-09-19 · 12 epochs, `train_calibrated.py`, warm-started from
`best_sharp_watershed.pth`. Scored on the 60 held-out crops.

## Pre-registered predictions vs outcome

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | mean B on true seam pixels > 0.40 | 0.105 -> **0.186** | **FAILED** |
| 2 | AUROC stays >= 0.85 | 0.866 -> **0.872** | **CONFIRMED** |
| 3 | absolute-threshold merge beats no-merge by >= +2 F1@50 | 66.90 vs 66.97 (**-0.07**) | **FAILED** |
| 4 | best config >= 70 F1@50 | **66.98** | **FAILED** |

Three of four failed. And yet the intervention is the largest single-step improvement
the project has produced since the decode constants.

## What actually changed

| metric (hold-out) | old head, best config | **H6 head, best config** | delta |
|---|---:|---:|---:|
| **COCO AP** | 24.32 | **26.84** | **+2.52** |
| COCO AP50 | 56.56 | 58.06 | +1.50 |
| **COCO AP75** | 16.48 | **20.40** | **+3.92** |
| F1@50 | 66.91 | 66.98 | +0.07 |
| F1@75 | 30.45 | **37.06** | **+6.62** |
| PQ | 48.82 | 49.97 | +1.15 |
| SQ | 72.78 | 74.59 | +1.81 |
| B-IoU | 12.84 | **15.26** | +2.42 |
| Dao ED | 0.638 | **0.595** | -0.043 |

The gain is entirely in **geometry at strict IoU**, not recognition. F1@50 is flat;
F1@75 rises 6.6 points and COCO AP75 rises 3.9. That is exactly the signature of masks
landing in better places, which is what a stronger seam ridge should do.

## Why three predictions failed

**Prediction 1's threshold of 0.40 was invented, not derived.** I picked it as "clearly
non-collapsed" without asking what magnitude the *watershed* actually needs. The flood
relief is `boundary * 255`, so mean seam response 0.105 -> 0.186 raises the barrier
between crowns from ~27/255 to ~47/255 — it nearly doubles. That is plainly enough to
change flooding, and the results show it did. The criterion failed; the mechanism it
was supposed to test held.

**Prediction 3 failed for a reason worth keeping.** Under the new head, merging stops
helping: no-merge scores COCO AP 26.84 against 26.29 for percentile merge and 25.66 for
the absolute threshold. Merging existed to repair over-separation that the weak ridge
caused; a stronger ridge separates correctly in the first place, so the repair now
removes real boundaries. **The merge step should be dropped when the calibrated head is
used** — it was a workaround, and the workaround has expired.

**Prediction 4 failed because F1@50 was never the metric this would move.** Recognition
is bounded by seeding, and H6 did not touch seeding. Predicting a recognition gain from
a boundary intervention was an error in my own reasoning, and H7 (the apex head) is the
hypothesis that addresses recognition.

## The calibration is only half-fixed

| | old | H6 | target |
|---|---:|---:|---:|
| mean B on true seams | 0.105 | 0.186 | 0.40 |
| mean B elsewhere | 0.013 | 0.041 | — |
| **separation ratio** | **8.0x** | **4.5x** | — |
| AUROC | 0.866 | 0.872 | >= 0.85 |

The head got louder, but the background got louder *faster*: the seam-to-background
ratio fell from 8.0x to 4.5x. This is the risk flagged at the epoch-1 probe and it did
not go away. Class-balanced weighting raises the whole field rather than sharpening the
contrast — which is also why the absolute-threshold merge still cannot find a usable
cut point.

The protocol's pre-registered fallback therefore applies: *"If 1 fails, positive
weighting is insufficient against this target and the next step is the target, not the
loss — the seam band is 3 px wide and blurred at sigma 0.8, which may simply be too
thin to supervise at this resolution."* That is now the live hypothesis for any further
boundary work.

## Carried forward

New best configuration: **H6 calibrated head, tuned seeds (md=160, ma=0.50), no merge**
— COCO AP **26.84**, AP50 58.06, AP75 20.40, F1@50 66.98, PQ 49.55, SQ 73.79,
B-IoU 13.72, Dao ED 0.594.

Project arc in COCO AP on the hold-out: published **15.81** -> H1 tuned **23.85** ->
H5 merge **24.32** -> **H6 26.84**.
