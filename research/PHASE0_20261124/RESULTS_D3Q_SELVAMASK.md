# D3-query Step 2a: in-domain query vs NMS vs flow on SelvaMask SAME6 (2026-11-24)

## Setup
- Protocol written before computing: `PROTOCOL_D3Q_SELVAMASK.md`.
- Code: `code/d3q_selvamask_families.py`. Results: `results/d3q_selvamask.json`.
- Preview: `previews/selvamask_m2f_vs_r5_absorbed.png`.
- Saved frozen predictions only; no inference or training; TEST not touched.
- 900 reference crowns; 824 remain after excluding crowns at the border or in invalid areas.

## Results (d = 2; d = 1 and 5 within ±0.007)

| System (in-domain) | Recall | Absorption rate | Share of misses | Strong-stratum rate |
|---|---|---|---|---|
| Mask2Former s42 (query) | 0.643 | 0.200 | 0.561 | 0.149 |
| Mask2Former s43 | 0.621 | 0.224 | 0.590 | 0.159 |
| Mask2Former s44 | 0.637 | 0.200 | 0.552 | 0.145 |
| R5 Mask R-CNN (NMS) | 0.656 | 0.228 | 0.664 | 0.166 |
| TreeFlowNet PP200 (flow) | 0.507 | 0.226 | 0.458 | 0.166 |

## Primary comparison: Mask2Former vs R5, paired on the same instances

| Seed | Ratio | Difference [tile-bootstrap 95% CI] | Absorbed only by M2F / only by R5 | McNemar p |
|---|---|---|---|---|
| 42 | 0.88 | −0.028 [−0.071, 0.008] | 60 / 83 | 0.065 |
| 43 | 0.98 | −0.005 [−0.057, 0.041] | 66 / 70 | 0.80 |
| 44 | 0.88 | −0.028 [−0.060, −0.009] | 56 / 79 | 0.058 |

**Pre-declared reading: SET_PREDICTION_DOES_NOT_REMOVE_ABSORPTION.** All 3 seeds have ratio ≥ 0.8.
- In-domain Mask2Former, at the same strength as Mask R-CNN (recall 0.62–0.64 vs 0.66), absorbs about 20–22% of all
  reference crowns. That is 55–59% of its misses.
- There may be a small reduction, of about 12% relative in 2 of 3 seeds, but no CI excludes 0 except seed 44's.
- Absorption overlaps strongly between families: 105 of R5's 188 absorbed crowns are also absorbed by Mask2Former s42.
- The preview (eye, coarse) shows both models merging the same pair in 7 of 8 crops.

## Secondary
**Multiplicity: number of GT crowns covered by the absorbing prediction.**

| System | 2 crowns | 3–4 crowns | ≥ 5 crowns |
|---|---|---|---|
| Mask2Former | 36–46% | 35–42% | 18–22% |
| R5 | 28% | 38% | 34% |
| TreeFlowNet | 46% | 42% | 12% |

**Mask2Former stage decomposition of its absorbed crowns.** Uses the saved raw decoded queries from all 4 windows,
before fusion and suppression.

| Stage | Share (3 seeds) |
|---|---|
| No raw query mask reaches IoU ≥ 0.5 | **89–91%** (147/165, 167/184, 149/165) |
| A good raw query existed but was not accepted | 9–11% |
| Lost in one-to-one matching | 0 |

## What this establishes
- **[V]** On ITC, in-domain set prediction absorbs as much as NMS detection and flow grouping at comparable strength.
  Hungarian training of every instance does not remove the failure.
- **[V]** In this in-domain query model, absorption arises mostly where **no decoded query produces an adequate mask for
  the crown** (candidate formation / localisation). Selection matters less, and matching not at all.
  - In zero-shot CellSAM on LIVECell, selection mattered as much as localisation (46% / 46%).
  - The in-domain result shifts the weight to candidate formation.
- This weakens D4(b) ("formation outside the loss"): a model whose loss does see every instance still absorbs.
  - What all three families share is that no candidate is formed for one of two touching crowns, whatever the loss.
- **Limits:**
  - 6 tiles and 3 source orthomosaics; the cluster CIs are crude.
  - SelvaMask tiny-crown inter-annotator agreement is low (mRF1 12.7–20.1).
  - The R5 and TFN comparators are single runs.
  - "No raw IoU ≥ 0.5" does not prove the model lacks an internal representation.
