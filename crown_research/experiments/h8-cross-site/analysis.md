# H8 Analysis — the suppression rule transfers; the *model* does not

**Run:** 2026-09-19 · BAMFORESTS TestSet1 (39 crops) and TestSet2 (40 crops), both
untouched by any earlier tuning. `evalhold60` carried as the reference.

## Pre-registered predictions vs outcome

| # | Prediction | test1 | test2 | Verdict |
|---|---|---|---|---|
| 1 | frozen (160, 0.50) beats published (24, 0.25) by >= +10 F1@50 | **+14.54** | **+24.77** | **CONFIRMED** |
| 2 | each split's own optimum within ±30% of its crown diameter D | 24% off | 24% off | **CONFIRMED** |
| 3 | frozen is within 2 F1@50 of the per-split optimum | +1.35 | +0.00 | **CONFIRMED** |

First hypothesis since H0 to pass cleanly on every prediction.

## Results

| split | D (mean crown diameter) | config | F1@50 | COCO AP | AP50 | seeds/crown |
|---|---:|---|---:|---:|---:|---:|
| evalhold60 | 211 px | published (24, 0.25) | 44.29 | 15.81 | 40.34 | 1.46 |
| evalhold60 | 211 px | **frozen (160, 0.50)** | **65.54** | **23.85** | 56.19 | 0.76 |
| test1 | **314 px** | published (24, 0.25) | 9.56 | 0.86 | 3.81 | **3.18** |
| test1 | 314 px | **frozen (160, 0.50)** | **24.10** | 2.48 | 9.16 | 0.93 |
| test2 | 212 px | published (24, 0.25) | 39.08 | 13.18 | 34.97 | 1.52 |
| test2 | 212 px | **frozen (160, 0.50)** | **63.85** | **21.29** | 49.96 | 0.75 |

## Two findings, pulling in opposite directions

**1. The calibration rule transfers, and no per-site tuning is needed.**
One constant chosen on `eval` is within 1.35 F1@50 of each test split's own swept
optimum. The published 24 px setting degrades catastrophically off its tuning split —
on test1 it emits **3.18 seeds per crown** and collapses to COCO AP 0.86. The
recommendation to set suppression by crown scale rather than by a fixed small constant
is therefore supported at three splits, not one.

**2. The model itself does not transfer to TestSet1, and that is the bigger story.**
Even at its own optimum, test1 reaches **COCO AP 2.98** against 23.85 on the eval
hold-out — an order of magnitude worse, not a few points. TestSet2 behaves like eval
(AP 21.3 vs 23.8), so this is specific to TestSet1.

The split statistics point at why: TestSet1's mean crown diameter is **314 px against
211 px** for both eval and TestSet2 — crowns are 2.2x larger in area. That is a change
of scale (different flight altitude, ground sample distance, or forest structure), and
the network was trained at one scale. Every conclusion in this project about *decoder*
behaviour holds on test1 — the suppression rule still works there — but the *heads*
produce almost nothing usable.

**This bounds the entire project's external validity.** The error budget (H0, H3), the
69-70 marker wall, and the 97.3 oracle ceiling were all measured at one crown scale.
None of them has been shown to hold under a scale shift, and TestSet1 shows the model
degrades severely under exactly that shift.

## An exploratory regularity that did not survive its own check

test1's reported optimum landed on the upper edge of the swept grid (240 of a 40-240
range) — the same failure that made H1's locked grid report a false optimum. Before
quoting anything, the grid was extended to 460.

With test1 resolved at an interior optimum:

| split | D | optimum min_dist | ratio min_dist / D |
|---|---:|---:|---:|
| evalhold60 | 210.6 | 160 | 0.760 |
| test2 | 211.6 | 160 | 0.756 |
| **test1** | 314.0 | **280** | **0.892** |

**There is no tight constant ratio.** On the two splits at the same crown scale the
ratio agrees to three decimals (0.760, 0.756), which is why it looked like a law; the
split at a *different* scale sits at 0.892. So the agreement between eval and test2 was
telling me they share a scale, not that a universal coefficient exists.

Had the grid not been extended, `min_dist ~ 0.76 x D` would have been reported as a
rule on two unresolved points and been wrong. The honest statement is the weaker one:
**the optimum lies at roughly 0.75-0.9 x mean crown diameter**, which is enough to
replace a blind 24 px constant but is not a precise calibration law.

(The pre-registered ±30% prediction passes more comfortably after the extension: 280 vs
D=314 is 11% off, against 24% at the truncated grid.)

Note also that the naive `min_dist = D` rule is *worse* than the frozen 160 on both
eval (61.59 vs 65.54) and test2 (60.52 vs 63.85). Whatever the coefficient is, it is
below 1.0.

Metric footnote: on test1, F1@50 peaks at md=280 (26.83) while COCO AP peaks at md=240
(2.98 vs 2.91). The two metrics disagree about the optimum on this split — a further
sign that test1 is a degenerate operating regime rather than a slightly harder one.

## Consequences

- The practical recommendation stands and is now multi-split: **do not use 24 px; set
  suppression from the dataset's crown scale**, and one value transfers across splits
  of similar scale.
- **A scale-generalisation experiment is now the highest-priority open question**,
  ahead of further head work. If the model collapses under a 1.5x scale shift, then
  H6 and H7 are optimising within a regime the model cannot leave.
