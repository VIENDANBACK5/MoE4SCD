# H9 — Is the TestSet1 collapse scale, or appearance?

**Status:** LOCKED (pre-registration) · 2026-09-20
**Type:** CONFIRMATORY

## Motivation

TestSet1 scores COCO AP **2.98** even at its own decode optimum, against **23.85** on
the eval hold-out and **21.29** on TestSet2 — a ~7x gap. H8 attributed this to crown
scale (D = 314 px vs 211 px). The literature check showed that attribution was too
narrow: BAMFORESTS designed TestSet1 as an out-of-domain split and it differs on
**several confounded axes at once** —

| axis | train / val / TestSet2 | TestSet1 (Hain-AOI) |
|---|---|---|
| site type | managed mixed forest | **city park** |
| platform | Trinity F90+ @ 120 m AGL | **DJI Phantom 4 @ 85 m AGL** |
| sensor | (Trinity payload) | **different sensor** |
| date | 5-6 July | **3 August** |
| time of day | midday | **after sunrise to early midday** |
| mean crown diameter | 211 px | **314 px** |

A remedy aimed only at scale may therefore do nothing. This separates the scale axis
from everything else before any fix is proposed.

## Design

The network is fully convolutional, so **test-time rescaling** isolates scale cleanly
and costs no training. For a factor `s`: resize the crop by `s`, run the heads, resize
the three head outputs back to the original resolution, decode and score against the
unmodified ground truth. Scale-matching predicts `s* = D_train / D_test1 = 211/314 =
0.672`, which makes TestSet1's crowns subtend the number of pixels the model was
trained on.

Sweep `s in {0.5, 0.6, 0.672, 0.8, 1.0, 1.25}` on TestSet1.

Decode constants are re-derived per `s` rather than frozen, since rescaling by `s`
shrinks crowns and therefore the correct suppression radius: `min_dist = 280 * s`
(280 being TestSet1's own optimum from H8), `min_apex = 0.50`.

**Control (the part that makes this interpretable).** Run the identical sweep on
**TestSet2**, whose crown scale already matches training (D = 212 px). If rescaling
helps there too, the effect is "smaller input is just better for this network" and says
nothing about domain shift. If it helps only on TestSet1 and peaks near 0.672, scale is
the dominant axis.

## Predictions (committed before running)

1. **Scale is the dominant axis.** The best `s` on TestSet1 recovers **>= 50% of the
   gap** to in-domain performance, i.e. COCO AP **>= 13.4** (from 2.98, against 23.85).
2. **The optimum is where scale-matching says it should be.** Best `s` on TestSet1 lies
   within **±20%** of 0.672, i.e. in [0.54, 0.81].
3. **The control is flat.** On TestSet2, `s = 1.0` is within **1.0 COCO AP** of the best
   `s`, confirming the effect is not a generic benefit of downscaling.

If 1 fails, the collapse is **not** primarily geometric — it is photometric/structural
(city park canopy, different sensor response, low-sun illumination and long shadows) and
the remedy is appearance augmentation or domain adaptation, not multi-scale training.
That outcome would also mean the crown-diameter framing should be dropped from the
write-up entirely rather than merely qualified.

If 1 passes but 2 fails, scale matters but not through the simple diameter-matching
account, and the mechanism needs separate work.

## Outputs
- `results/h9_rescale_sweep.csv` — both splits, all `s`
- `results/h9_summary.csv`
