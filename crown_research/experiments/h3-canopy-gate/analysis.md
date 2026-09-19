# H3 Analysis — The heads are not separable. The decoder was never the problem.

**Run:** 2026-09-19 · full 2^3 factorial, `eval30` + `evalhold60`. Claims quoted on hold-out.

## Pre-registered predictions vs outcome

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Oracle canopy alone gains >= +8 mAP@50 | **-24.32** | **FAILED (sign inverted)** |
| 2 | Oracle seeds + oracle canopy >= 85 | **38.60** | **FAILED (badly)** |
| 3 | Full oracle >= 92 | **97.30** | **CONFIRMED** |

## The full cube (hold-out, 60 crops)

| seeds | energy | canopy | mAP@50 | mAP@75 | PQ | SQ | B-IoU | Dao ED |
|---|---|---|---:|---:|---:|---:|---:|---:|
| pred | pred | pred | 65.54 | 28.47 | 47.41 | 72.13 | 11.65 | 0.644 |
| pred | pred | **O** | 41.22 | 10.39 | 27.47 | 65.03 | 4.62 | 0.770 |
| pred | **O** | pred | 33.04 | 13.73 | 24.13 | 72.29 | 15.08 | 0.786 |
| pred | **O** | **O** | **77.96** | 74.08 | 72.51 | 92.93 | 52.19 | 0.497 |
| **O** | pred | pred | 69.03 | 29.16 | 49.84 | 72.09 | 12.17 | 0.469 |
| **O** | pred | **O** | 38.60 | 10.08 | 25.89 | 66.01 | 4.71 | 0.575 |
| **O** | **O** | pred | 40.74 | 19.32 | 30.59 | 75.49 | 16.06 | 0.561 |
| **O** | **O** | **O** | **97.30** | **93.10** | **89.69** | 92.19 | 59.20 | **0.103** |

## What this says

**1. Every single-factor oracle makes things WORSE. Only the pair helps.**
Fixing the canopy gate alone: **-24.3**. Fixing the boundary energy alone: **-32.5**.
Fixing both: **+12.4**. With oracle seeds as well: **+31.8, to 97.30**.
This is a textbook interaction, and it inverted two of my three predictions.

**2. Mechanism: the predicted canopy gate is the system's real seam detector, and the
two fields only work when they are mutually consistent.**

`flood()` sets `markers[canopy < 0.25] = 1`, so every sub-threshold pixel is a
*background seed* that a flood cannot cross. The predicted canopy mask dips between
touching crowns, and those dips have been doing the separation work all along — which
is why the boundary head B looked so uninformative in H2 (no seam had mean B above
0.35; the best merge threshold was the smallest one swept).

That explains both inversions:
- **Oracle canopy + predicted B**: the exact GT foreground removes the dips, and B has
  no ridge to replace them, so a correctly-masked blob gets partitioned essentially at
  random. SQ collapses 72.1 -> 66.0 even with one seed per true crown.
- **Oracle seams + predicted canopy**: the GT seam ring sits at the true crown edge,
  but the predicted mask extends past it. The flood halts at the ring and leaves an
  unclaimed annulus between the true edge and the predicted canopy edge, which is then
  captured arbitrarily. B-IoU rises (15.1) exactly as expected from better seams, while
  mAP@50 falls to 33.0.

Neither field is independently "the bottleneck". Their **joint consistency** is.

**3. The decoder is not the limit — it is nearly perfect.**
With all three factors correct, the same unmodified flat-basin watershed reaches
**97.30 mAP@50, 93.10 mAP@75, PQ 89.69, Dao ED 0.103**. H2 had concluded that the ~70
wall might mean the architecture had to be replaced. It does not. Every remaining point
lives in the network heads, not in the decoding algorithm.

**4. The headroom is enormous and it is all learnable.**
Current best non-oracle configuration: 66.57. Attainable with the same decoder and
correct heads: 97.30. That is **~31 points**, and unlike H0's marker gap it cannot be
reached by tuning a constant — it requires changing what the network is trained to
predict.

## Main effects (mean mAP@50 lift across the cube, hold-out)

| factor | main effect | SQ effect |
|---|---:|---:|
| seeds | +6.97 | +0.85 |
| energy | +8.67 | +14.41 |
| canopy | +11.68 | +6.04 |
| **seeds x canopy interaction** | **-6.11** | — |

Main effects alone are misleading here: they average over cells where the partner
factor is wrong. The cube's structure, not its marginals, is the result.

## Next — H4

The target is a **consistency-coupled supervision** of the boundary and canopy heads,
so that the mask edge and the ridge field agree by construction rather than by luck.
Concretely, the candidate is to train the canopy head against a foreground target that
is *seam-excluded* (crown union minus the inter-crown seam band) while the boundary
head is trained on that same seam band, so the two share a single definition of where a
crown ends. This is a training change, not a decode change, and it is the first
hypothesis in this project that requires a training run.

Before spending GPU time, a cheap decode-side probe is worth running: build the
watershed gate from a **combination** of the existing heads
(`gate = canopy - lambda * boundary`) and sweep `lambda`. If joint consistency is the
active ingredient, imposing it at decode time should already pay, and it costs minutes
rather than hours.
