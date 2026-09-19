# H5 Analysis — Ranking information is real but not usable raw; decode-side work is exhausted

**Run:** 2026-09-19 · tuned on `eval30`, claims on `evalhold60`

## Pre-registered predictions vs outcome

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Rank relief beats raw-B control by >= +2 | **-9.44** (56.09 vs 65.54) | **FAILED (sign inverted)** |
| 2 | Percentile merge beats H2's absolute merge by >= +2 | +0.95 (66.91 vs 65.96) | **FAILED** |
| 3 | Canopy gate covers >= 30% of GT seam pixels | **32.11%** | **CONFIRMED** |

## Results (hold-out, 60 crops)

| Arm | mAP@50 | mAP@75 | PQ | SQ | B-IoU | Dao ED |
|---|---:|---:|---:|---:|---:|---:|
| A control (raw B relief, no merge) | 65.54 | 28.47 | 47.42 | 72.13 | 11.65 | 0.644 |
| B rank relief (gamma=2.0) | 56.09 | 14.85 | 38.32 | 66.93 | 6.87 | 0.733 |
| **C percentile merge (q=50)** | **66.91** | 30.45 | **48.82** | 72.78 | **12.84** | 0.638 |
| E rank relief + percentile merge | 56.82 | 16.51 | 39.30 | 67.64 | 7.66 | 0.727 |

## What this says

**1. Rank-transforming a collapsed head amplifies its noise.**
I reasoned that if B's information lives in its ordering (AUROC 0.866), restoring
dynamic range by rank should expose it. The opposite happened: -9.4 mAP@50, SQ
72.1 -> 66.9, and `gamma=1.0` (pure uniform rank) collapsed to **10.57**.

The flaw is visible in the diagnostic I already had: B's median is 0.0011 and its mean
is 0.019, so the overwhelming majority of pixels sit in a near-zero band where the
ordering is **noise, not signal**. A rank transform spreads that noise across the full
[0,1] range and manufactures ridges everywhere. AUROC is driven by the small
high-confidence tail; it does not license a monotone remap of the whole field.
`gamma=2.0` scored best of the three precisely because squaring pushes the noisy bulk
back toward zero — i.e. the best rank transform is the one that most nearly undoes
itself.

**2. Percentile merging is a small, real, free gain — and is the new best configuration.**
Replacing H2's absolute `tau` with a per-crop median seam threshold gives
**66.91 mAP@50 / 48.82 PQ / 30.45 mAP@75 / B-IoU 12.84** on the hold-out, improving on
the raw control (65.54 / 47.42) on every metric. Below the +2 bar I set, so the
prediction fails, but it is the best non-oracle result this project has produced and
costs nothing.

**3. Arm D rescues a corrected version of H3's mechanism.**
H5's protocol withdrew H3's claim that the canopy gate is the system's seam detector,
because `1 - canopy` scores AUROC 0.567 against seams. The gap audit shows why both
things are true at once:

| | GT crown-crown seam pixels | crown interior pixels |
|---|---:|---:|
| fall in predicted background (`canopy < 0.25`) | **32.1%** | 5.2% |

The gate is a poor *graded* seam detector but a strong *hard-barrier* one: a third of
all true seams are already cut by sub-threshold canopy, at 6x the rate of crown
interiors. That is the real mechanism behind H3's interaction — replacing the predicted
gate with an exact GT mask deletes those barriers, and the near-zero B field has nothing
to put in their place.

## Conclusion of the decode-side programme

Six hypotheses, five of them decode-side, and the ceiling has not moved:

| intervention | hold-out mAP@50 |
|---|---:|
| published settings (md=24, ma=0.25) | 44.29 |
| H1 tuned constants (md=160, ma=0.50) | 65.54 |
| H2 seam merge on tuned seeds | 66.57 |
| **H5 percentile merge on tuned seeds** | **66.91** |
| H2 oracle merge (framework ceiling) | 69.74 |
| H0 oracle seeds | 69.03 |
| **H3 full oracle (correct heads, same decoder)** | **97.30** |

After H1's +21.2, everything decode-side has returned **+1.4 in total**. The three
oracle routes converge at 69-70. The same unmodified decoder reaches 97.3 when the
heads are correct. **Decode-side optimisation is exhausted; the remaining ~30 points
are in the network, and specifically in the boundary head.**

## Next — H6 (first training hypothesis of the project)

Train the boundary head with a class-balanced objective so its magnitudes survive the
~6% positive rate — the diagnostic (AUROC 0.866 with mean 0.019) is exactly the
signature of a head that learned the ranking and then had its confidence squeezed out
by imbalance. Candidates: Tversky/focal loss on the seam band, or a seam-band-restricted
BCE with positive weighting. The success criterion is calibration, not just AUROC:
**mean B on true seam pixels must exceed ~0.4** (from 0.105), which would make H2's and
H4's absolute-magnitude uses viable for the first time.
