# H1 Analysis — The adaptive-radius idea failed; a mis-calibrated constant was the real story

**Run:** 2026-09-19 · 30 tuning crops + 60 disjoint hold-out crops (eval idx 100-159)

## Pre-registered predictions vs outcome

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Best global-radius cell reaches **mAP@50 <= 48%** | Locked grid gave 52.87; extending it gave **66.15** | **FAILED** |
| 2 | H1-b (adaptive) beats best H1-a by **>= 3 pts** | H1-b 51.20 vs H1-a 52.87 -> **-1.67** | **FAILED** |
| 3 | H1-b seed ratio lands in **[0.85, 1.20]** | 1.174 | CONFIRMED |

Two of three predictions were wrong, and wrong in the same direction: I assumed a
single global suppression radius was structurally inadequate. It is not.

## What actually happened

The pre-registered Arm-A grid topped out at its own upper corner
(`min_dist=56, min_apex=0.45`), which should have been caught before running — a grid
whose optimum sits on the boundary has not found an optimum. Extending it to
`min_dist <= 192` found a clear interior plateau:

| min_dist | min_apex | mAP@50 | PQ | seed ratio |
|---:|---:|---:|---:|---:|
| 24 (published) | 0.25 | 31.33 | 21.34 | 1.90 |
| 56 | 0.45 | 52.87 | 38.02 | 1.00 |
| 144 | 0.50 | 66.08 | 48.63 | 0.77 |
| **160** | **0.50** | **66.15** | **48.69** | 0.77 |
| 192 | 0.50 | 65.79 | 48.73 | 0.72 |

**The published `min_distance = 24 px` is roughly 6x too small for this dataset.**
BAMFORESTS eval crops average 15.6 crowns per 1024x1024 tile, i.e. ~16.8k px of canopy
per crown, i.e. an equivalent crown diameter of **~146 px**. The optimum suppression
radius (144-160 px) is almost exactly one crown diameter — which is the physically
correct setting, and what the constant should have been all along.

## Hold-out validation (the number that counts)

`min_dist`/`min_apex` were selected on the same 30 crops used to report, so 66.15 is a
selection-on-test figure. Re-scored on **60 disjoint crops** with the settings frozen:

| Arm | tune (30) | hold-out (60) | delta |
|---|---:|---:|---:|
| baseline md=24, ma=0.25 | 31.33 | 44.29 | +12.96 |
| md=56, ma=0.45 | 52.87 | 60.81 | +7.94 |
| **md=160, ma=0.50** | **66.15** | **65.54** | **-0.61** |
| H1-b adaptive | 51.20 | 57.36 | +6.16 |

The tuned setting transfers essentially unchanged (-0.61), so it is not overfit. But
the baseline scores *much* higher on the hold-out (44.3 vs 31.3): **the original 30
crops are an unusually hard subset**, and every previously published number on them —
including the 31.33 baseline and the MorphSnakes 32.99 — is pessimistic.

**The honest headline is the hold-out delta: 44.29 -> 65.54 mAP@50 (+21.2, +48%
relative), PQ 31.0 -> 47.4**, from changing two decode constants and nothing else.

## Why the adaptive radius lost

H1-b's mechanism was sound — per-instance normalisation of the surface target
(`targets_sharp.py:59-63`) does destroy scale at the peaks while leaving it in the
superlevel sets — but the estimator is noisy. `{U >= t}` components merge across
touching crowns at low `t` (over-estimating radius) and shatter at high `t`
(under-estimating badly: at `t=0.7` the seed ratio explodes to 7.8). The best cell
needed `t=0.3`, where components have already partly merged, so the "adaptive" radius
was in practice a blurred global one. A single well-chosen constant beats a noisy
per-peak estimate. **Dropped.**

## The finding that matters most: the precision/recall trap

Seed placement audit against GT (30 crops, 467 GT crowns):

| Setting | seeds | crowns hit | duplicate seeds / crown | orphan seeds / crown | crowns missed |
|---|---:|---:|---:|---:|---:|
| md=24, ma=0.25 | 860 | **83.1%** | 0.820 | 0.191 | 16.9% |
| md=112, ma=0.55 | 329 | 62.5% | **0.039** | 0.041 | **37.5%** |

Global NMS cannot win this. Loose suppression **finds 83% of all crowns** but plants
0.82 extra seeds in each one, splitting them. Tight suppression removes essentially
every duplicate (0.04/crown) but throws away 37.5% of the crowns to do it. Tuning only
slides along this frontier; it never escapes it.

The oracle (H0 cell B, 70.79) sits off the frontier because it has 100% hit rate *and*
zero duplicates. A method that keeps loose seeding's **83% hit rate** while removing
duplicates *after* flooding would be bounded far above the tuned 66 — a one-seed-per-hit-crown
configuration implies an F1@50 ceiling near 0.9, not 0.71.

## Next — H2

Over-seed deliberately (preserve the 83% hit rate), then **merge adjacent basins whose
shared seam carries weak predicted boundary evidence**. H0 cell C already showed the
boundary head is informative about geometry (SQ 67.7 -> 74.3, B-IoU 10.3 -> 24.4 with
oracle seams) even though it cannot drive recognition on its own — exactly the signal
a merge criterion needs.
