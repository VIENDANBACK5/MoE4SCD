# D3-query Step 2a: in-domain query vs NMS vs flow on SelvaMask SAME6 (written 2026-11-24, BEFORE computing)

## Why
- G3 needs an in-domain set-prediction model of comparable strength. CellSAM was zero-shot and INCONCLUSIVE.
- Already-saved, frozen predictions on the same 6 SelvaMask validation tiles (900 reference crowns, 4.5 cm) exist for
  three in-domain families:
  - **Query:** Mask2Former-R50, COCO-pretrained, fine-tuned on SelvaMask TRAIN, seeds 42/43/44.
    Source: `research/ITC_CARRIER_FORMATION_V1_20261103/evaluation/mask2former_seed{s}/<tile>/predictions.json`.
    Recorded: AP 21.2 ± 0.3, Recall50 63.4.
  - **NMS:** R5 Mask R-CNN (frozen, deterministic).
    Source: `research/method_development/R5_INSTANCE_METHOD/evaluation/<tile>/predictions.json`.
    Recorded: AP 22.9, Recall50 66.0.
  - **Flow:** TreeFlowNet PP200, non-oracle.
    Source: `research/diagnostics/D1_M0_BOTTLENECK/execution_v2/tiles/<tile>/PP200_predictions.json`.
- No inference and no training. The TEST split is not touched.

## Fixed choices
- **Operating point:** each system's saved accepted `predictions` (its own frozen operating point), scored by `score`.
  No re-thresholding.
- **GT raster:** reference RLEs painted in file order, a later one overwriting an earlier one, which is the same rule
  as for LIVECell.
- **Exclusion:** GT touching the tile border or touching invalid pixels (`valid.png == 0`, after a 1-px dilation).
- **Metric:** `absorption.image_rows` unchanged (d = 2 primary; 1 and 5 reported).

## Primary comparison: per-instance absorption rate
- Definition: absorbed / non-excluded GT. The GT and instance set are identical for every system.
- For each Mask2Former seed: paired difference vs R5, with a 95% CI from a tile-cluster bootstrap (1,000 draws).
- An instance-level McNemar test on the absorbed status is also reported.
- With 6 tiles the cluster CI is crude; this is stated as a limitation.

## Pre-declared reading
| Outcome | Reading |
|---|---|
| Mask2Former rate ≥ 0.8 × R5 in all 3 seeds | In-domain set prediction does **not** remove absorption on ITC. G3 closes on the ITC side; the gap stands |
| Mask2Former rate ≤ 0.5 × R5 in all 3 seeds | Set prediction substantially reduces absorption. Narrow the claim to the families that still fail |
| Otherwise | PARTIAL; report as is, no re-reading |

## Secondary (descriptive, no decision attached)
- Recall, share of misses, strong-stratum rate, and multiplicity of the absorbing mask (2 / 3–4 / ≥ 5 GT).
- **Mask2Former stage decomposition for every absorbed GT**, using its saved `raw_predictions` (all decoded queries,
  before window fusion and suppression):
  - (a) no raw query mask with IoU ≥ 0.5;
  - (b) a raw query mask with IoU ≥ 0.5 exists but was not accepted (score / fusion / suppression);
  - (c) accepted but lost to one-to-one matching.
- TreeFlowNet PP200 is reported alongside, as the flow family.
