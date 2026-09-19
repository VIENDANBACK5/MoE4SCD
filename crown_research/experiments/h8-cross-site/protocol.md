# H8 — Does the tuned decode constant transfer, and is it really a crown-diameter rule?

**Status:** LOCKED (pre-registration) · 2026-09-19
**Type:** CONFIRMATORY · generalisation test, runs on CPU from cached heads

## Motivation

Everything established so far — the +21 F1@50 from retuning, the 69-70 marker wall, the
97.3 oracle ceiling — was measured on BAMFORESTS **eval** crops alone. The central
practical claim, that `min_distance` should be ~one crown diameter rather than 24 px,
is currently supported at one site with one tuning split.

Two things need separating:
- a **constant that happens to fit Bamberg's eval tiles** (weak, not worth reporting), vs
- a **rule** — suppression radius should track the dataset's crown scale — which happens
  to evaluate to ~160 px here (strong, and transferable to other sensors/altitudes).

BAMFORESTS ships two further held-out sets, `TestSet1` and `TestSet2`, untouched by this
project. They are the test.

## Design

For each of `test1`, `test2` (40 crops each) and `evalhold60` as the reference:

1. Measure **mean equivalent crown diameter** `D = 2*sqrt(mean_crown_area/pi)` from GT.
2. Score the **frozen** `eval`-tuned setting `(min_dist=160, min_apex=0.50)` — no
   re-tuning of any kind.
3. Score the **published** setting `(24, 0.25)` as the control.
4. Sweep `min_dist` per split to locate that split's own optimum, purely to test the
   rule (this optimum is *not* proposed as a method; it uses split labels).
5. Score a **parameter-free** variant: `min_dist = D` measured from that split's GT.

Metrics: repo F1@50 (for continuity with earlier tables) **and** true COCO AP.

## Predictions (committed before running)

1. **Transfer.** The frozen (160, 0.50) beats the published (24, 0.25) on **both** test
   splits by **>= +10 F1@50**.
2. **The rule is real.** Each split's own optimal `min_dist` lies within **±30%** of that
   split's measured crown diameter `D`.
3. **No per-site tuning needed.** The frozen 160 is within **2 F1@50** of each split's
   own swept optimum — i.e. re-tuning per site buys almost nothing.

If 1 holds but 2 fails, the constant transfers without the mechanism I claimed, and the
crown-diameter explanation must be dropped from the write-up even though the practical
recommendation survives. If 3 fails, the recommendation becomes "calibrate to crown
scale per dataset" rather than "use 160".

## Outputs
- `results/h8_split_stats.csv` — crown-size statistics per split
- `results/h8_transfer.csv` — frozen vs published vs per-split optimum vs D-rule
- `results/h8_sweep.csv` — full `min_dist` sweep per split
