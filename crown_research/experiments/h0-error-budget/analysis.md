# H0 Analysis — The error budget is almost entirely in marker selection

**Run:** 2026-09-19, 30 BAMFORESTS eval crops, `best_sharp_watershed.pth`
**Verdict:** All three pre-registered predictions CONFIRMED.

## Results

| Cell | Seeds | Energy | n_pred | mAP@50 | mAP@75 | PQ | SQ | RQ | B-IoU | OS | US | ED |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A (baseline) | pred | pred | 28.7 | 31.33 | 9.10 | 21.34 | 67.67 | 31.33 | 10.31 | 0.473 | 0.798 | 0.613 |
| **B** | **oracle** | pred | 14.9 | **70.79** | 32.20 | 51.68 | 73.05 | 70.79 | 11.92 | 0.634 | 0.864 | 0.428 |
| C | pred | **oracle** | 28.7 | 29.86 | 12.46 | 22.30 | 74.34 | 29.86 | 24.43 | 0.452 | 0.834 | 0.619 |
| D | oracle | oracle | 15.1 | 68.57 | 37.41 | 53.79 | 78.57 | 68.57 | 28.13 | 0.629 | 0.957 | 0.390 |

Mean GT crowns per crop: **15.6**. Mean predicted seeds: **28.7**.

## What this says

**1. Seeding is worth +39.5 mAP@50; the boundary energy is worth -1.5.**
Replacing predicted apices with one oracle seed per GT crown lifts mAP@50 from
31.3 -> 70.8 while leaving the *predicted* boundary head in place. Replacing the
energy field with GT seams while keeping predicted seeds changes nothing
(31.3 -> 29.9, slightly negative). The learned boundary head is not the bottleneck.

**2. The pipeline emits 1.90 seeds per true crown.**
`n_pred/n_gt = 1.904`. This is the direct mechanical cause of `OS = 0.473`: a crown
split roughly in two yields a best-matching fragment covering ~half of it. The Dao
indices were diagnosing over-seeding all along, not contour inaccuracy.

**3. Boundary refinement was optimising a 2-point problem next to a 40-point one.**
MorphSnakes bought +1.66 mAP@50. That is the right order of magnitude for what
contour geometry can contribute, and it is now clearly a second-order concern.
Its real contribution is geometric: B-IoU 10.3 -> 12.9, ED 0.613 -> 0.593.

**4. Oracle energy helps geometry, not recognition (C vs A, D vs B).**
C raises SQ 67.7 -> 74.3 and B-IoU 10.3 -> 24.4 at unchanged RQ. So the boundary head
*is* imperfect — it costs ~7 points of SQ and more than half the attainable B-IoU —
but that error is invisible in mAP@50 while RQ is the binding constraint. D scores
slightly *below* B on mAP@50 (68.6 vs 70.8) while beating it on mAP@75 (37.4 vs 32.2):
the thin GT seam ridge lets some floods leak across, trading a few matches for much
tighter masks on the ones that survive.

**5. The realistic ceiling for seed-only work is ~70 mAP@50.**
Cell B is the target: it uses only the existing network, changing nothing but which
markers are planted. Any seed-selection method is scored against 70.79.

## Methodological note (flagged, not yet acted on)

`evaluate_metrics` reports the **F1 score at a fixed IoU threshold**, not an
interpolated average precision, yet labels it `ap50`. The comparisons here are
internally consistent, but the metric must be renamed (or a true COCO AP added)
before any of these numbers go into a paper.

## Next

Cell B is an oracle. The open question is how much of that +39.5 is reachable without
GT. Two candidates, in order:
- **H1**: scale-adaptive seed suppression. The current `min_distance=24px` NMS is
  scale-blind, and the surface head U is already an EDT-like field whose value at a
  peak estimates the local crown radius — so the suppression radius can be read off
  the network's own output.
- **H2**: region-adjacency merging of over-segmented basins using mean predicted
  boundary evidence along each shared seam.

Also noted: `run_flat_basin_watershed` in the published benchmark uses plain greedy
NMS and never calls `extract_persistent_apices`, so the topological-persistence filter
that `decode.py` implements has never actually been exercised in the headline numbers.
