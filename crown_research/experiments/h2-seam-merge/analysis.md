# H2 Analysis — Merging works, but the merge *framework* has a ceiling of ~70. The bottleneck has moved.

**Run:** 2026-09-19 · tuned on `eval30`, all claims quoted on `evalhold60` (60 disjoint crops)

## Pre-registered predictions vs outcome

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | C beats B by >= 3 mAP@50 on hold-out | C = 65.96 vs B = 65.54 -> **+0.42** | **FAILED** |
| 2 | D (oracle merge) >= 80 mAP@50 | D = **69.74** | **FAILED** |
| 3 | `(C-B)/(D-B) >= 0.40` | 0.100 | **FAILED** |

## Results (hold-out, 60 crops)

| Arm | mAP@50 | mAP@75 | PQ | SQ | RQ | B-IoU | Dao ED | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A loose seeds, no merge | 44.29 | 16.13 | 31.01 | 69.17 | 44.29 | 11.22 | 0.598 | 1.46 |
| B tuned NMS, no merge | 65.54 | 28.47 | 47.42 | 72.13 | 65.54 | 11.65 | 0.644 | 0.76 |
| C loose + seam merge | 65.96 | 29.29 | 48.00 | 72.60 | **13.22** | — | **0.543** | 0.88 |
| D loose + **oracle** merge | **69.74** | **32.49** | **51.17** | 73.18 | 69.74 | 13.38 | 0.513 | 0.94 |
| E tuned + seam merge | 66.57 | 30.18 | 48.54 | 72.83 | 66.57 | 12.85 | 0.646 | 0.70 |

## What this says

**1. The merge criterion is nearly free but nearly useless for recognition.**
Thresholding mean seam boundary evidence recovers only **10%** of the gap to its own
oracle. Worse, the sweep is almost flat: every `tau >= 0.35` gives an identical 63.3,
meaning virtually no seam has mean B above 0.35 — the boundary head is **not calibrated
as a per-seam confidence**, so a global threshold on it carries little information.
The best `tau` found was 0.15, the *smallest* value swept, i.e. "merge almost nothing".

**2. But merging is clearly worth it for geometry.**
Every merge arm improves Dao ED (0.644 -> 0.543) and B-IoU (11.65 -> 13.22) over tuned
NMS. Arm E — tuned seeds *plus* merge — is the best non-oracle configuration on the
hold-out at **66.57 mAP@50 / 48.54 PQ / 30.18 mAP@75**, beating tuned NMS on every
metric. Small, but real and free.

**3. The decisive finding: the over-seed-then-merge framework tops out at ~70.**
Arm D is the best *any* merge rule could achieve given these basins, and it reaches
only 69.74 — a mere +4.2 over plain tuned NMS. I predicted >= 80 and was badly wrong.

## The convergence that reframes the project

Three structurally independent routes to "perfect markers" all land in the same place:

| Route | mAP@50 (eval30) |
|---|---:|
| H0 cell B — oracle seeds, predicted energy | 70.79 |
| H0 cell D — oracle seeds **and** oracle energy | 68.57 |
| H2 arm D — loose seeds + oracle merge | 68.83 |

**~70 is the ceiling of this marker-controlled watershed, and neither perfect markers
nor perfect seams can raise it.** The remaining ~30 points are not in marker selection
at all. The bottleneck that H0 identified has been closed — and a different one is now
binding.

Where it must be: SQ sits at 73 even with oracle everything, so matched instances only
reach IoU ~0.73, and mAP@75 never exceeds 32-37. Something is capping mask extent.

The suspect is the one head H0 never varied. Both H0 and H2 held the **canopy gate**
fixed at its predicted value throughout — it sets background marker 1 wherever
`canopy < 0.25`, so it hard-caps the extent of every instance regardless of how good
the seeds or the seams are. That is a gap in my own H0 design: I built a 2x2 over
seeds and energy and never questioned the third head.

## Next — H3

Full 2^3 factorial over {seeds} x {energy} x **{canopy gate}**, completing the error
budget. If the canopy gate is the new binding constraint, oracle-canopy cells will
break decisively through the ~70 wall.

## Carried forward regardless

Arm E (tuned seeds `md=160, ma=0.50` + seam merge `tau=0.15, L=30`) is now the best
known non-oracle configuration: **66.57 mAP@50, 48.54 PQ, 30.18 mAP@75, B-IoU 12.85,
ED 0.646** on 60 held-out crops.
