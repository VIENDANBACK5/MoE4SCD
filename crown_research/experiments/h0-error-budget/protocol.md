# H0 — Error-Budget Decomposition via 2x2 Oracle Ablation

**Status:** LOCKED (pre-registration) · 2026-09-19
**Type:** CONFIRMATORY

## Motivation

Panoptic Quality factorises as `PQ = SQ x RQ`. On 30 BAMFORESTS val crops the
Flat-Basin Watershed scores **SQ = 67.7%** but **RQ = 31.3%**. Mask geometry for
*matched* crowns is therefore already acceptable; the loss is in *recognition* —
crowns that are never matched at IoU>0.5 at all.

Boundary refinement (Douglas-Peucker smoothing, MorphSnakes) only moved mAP@50 from
31.33 -> 32.99 (+1.66). That is consistent with the SQ/RQ split: polishing contours
cannot fix crowns that were split or merged in the first place.

Dao indices point the same way: **OS = 0.473** (a matched prediction covers only 47%
of its GT crown) with **US = 0.797** (the prediction lies 80% inside that GT). That
signature is systematic **over-segmentation**: single crowns fragmented into ~2 basins.

## Question

Is the residual error caused by (a) **marker selection** — wrong number / placement of
watershed seeds — or (b) the **learned boundary energy field** B leaking across
touching crowns?

## Design — 2x2 factorial

Marker-controlled watershed, holding the flooding algorithm fixed
(`cv2.watershed` on a flat-basin relief), varying two factors:

| Cell | Seeds | Energy relief |
|---|---|---|
| A (baseline) | predicted apices from U | predicted boundary B |
| B | **oracle** (1 seed at EDT-max of each GT crown) | predicted boundary B |
| C | predicted apices from U | **oracle** (GT instance seams, blurred) |
| D (ceiling) | oracle | oracle |

Oracle seed = argmax of the Euclidean distance transform inside each GT instance —
guaranteed interior, one per crown. Oracle energy = GT inter-instance seam map,
Gaussian-blurred (sigma=1) to give the flood a finite-width ridge.

## Fixed factors
- Checkpoint: `DeadTrees/experiments/canopy_watershed/best_sharp_watershed.pth`
- Data: `BAMCanopyWatershedDataset(split="eval", crop_size=1024)`, first 30 crops, seed fixed by index order
- Decode params for predicted seeds: `min_dist=24.0, min_apex=0.25, min_canopy=0.25, min_area=25` (identical to published baseline)
- Metrics: `evaluate_metrics` (F1@IoU50, F1@IoU75, PQ, SQ, RQ), `compute_dao_segmentation_indices`, `compute_boundary_iou`

## Predictions (committed before running)

1. **Cell B >> Cell C.** Oracle seeds with the *predicted* energy will reach
   **mAP@50 > 55%**, while oracle energy with predicted seeds stays **< 42%**.
   Rationale: SQ is already 68%, so the flood front is landing roughly in the right
   place; what is missing is *which* basins exist.
2. Cell B will show **OS > 0.70** (fragmentation largely resolved) while Cell C leaves
   OS < 0.60.
3. The predicted-seed count will **exceed** the GT crown count (over-seeding), i.e.
   mean `n_pred / n_gt > 1.15`.

If prediction 1 is inverted (C > B), H1/H2 are dropped and the research pivots to the
boundary head (loss reweighting, seam-aware supervision).

## Recorded outputs
- `results/h0_error_budget.csv` — per-cell aggregate metrics
- `results/h0_per_crop.csv` — per-crop raw values for variance analysis
- `results/h0_seed_counts.csv` — n_pred vs n_gt per crop
