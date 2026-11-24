# Visible but Unseparated: Why Instance Fields Lose Small Tree Crowns Next to Large Ones

**Working draft v1, 2026-11-23. Empirical / mechanistic paper. No new segmentation method is claimed.**
It extends "story B" in `crown_segmentation_research/HOAN_THANH_DU_AN/08_DAN_Y_BAI_BAO_VA_BAO_CAO_NCKH.md` with the
2026-11-19 → 23 evidence. Every number cites a result file (paths are relative to `research/` unless stated).
`[TODO]` marks parts not yet written or not yet supported.

---

## Status of each claim
| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | Small-crown (< 2 m²) identity failures in bottom-up field models are concentrated where a small crown touches a larger one | Error decomposition; neighbour-ratio strata | supported, validation, 3 seeds |
| C2 | They are not a generalisation gap: the same rate occurs on training images | EXP-030 T1 | supported, 1 seed |
| C3 | They are shared across architectures (TFN, Mask R-CNN, SAM given the crown centre) | F1, EXP-030 T2, EXP-032 | supported, descriptive |
| C4 | They are physical, not only an annotation artefact | T4 (TLS reference) | supported, evaluation only |
| C5 | The missing information is the per-pixel membership field (+ separation), not existence, size, count or foreground | EXP-029 oracle ladder, EXP-017, G0 | supported (oracle) |
| C6 | For the merged (A) crowns, the separation is visible to a human almost as often as for crowns the models get right | EXP-033 blind audit | supported, one rater |
| C7 | Four loss-level interventions (pixel reweighting, crown reweighting, decision-aligned hinges) do not fix the merges; two of them cost about 1.2 AP | EXP-028b, 031b, 034, 036b + EXP-035 noise | supported, 1 seed each vs 3-seed noise |
| C8 | Per-crown canopy balancing (M3) moves crowns through the canopy gate (L −15 val, −102 train on every seed), but they fail downstream (A +7.7, M +8.7, OK unchanged); no confirmed recall/AP gain | EXP-037, 3 paired seeds | supported (negative) |

---

## Abstract (draft)
Bottom-up instance fields (flow-to-centre plus signed distance) are attractive for individual tree crown segmentation
from RGB drone imagery: they need no anchors and scale to dense canopies. On QuebecTrees they remain about 0.6 AP below
Mask R-CNN, and the gap concentrates on small crowns.

We isolate the failure. Small crowns (< 2 m²) touching a larger crown are absorbed or never form their own attractor:
- the failure rate on training images (0.34) is as high as on validation (0.28);
- Mask R-CNN misses the same crowns, and also misses crowns against a TLS-derived physical reference.

An oracle information ladder shows what fixes them:
- injecting the true membership field plus separation around the failed crowns recovers 99% of them;
- a correct centre point, the true size, the true count or the true foreground recovers almost none;
- a correct centre point alone recovers 0 of 210.

A blind visual audit shows the merged crowns are visible to a human almost as often as crowns the models get right
(ratio 0.91).

Four loss-level interventions, compared against measured seed noise (AP SD 0.15), do not remove the merges:
- per-crown reweighting;
- a cannot-link hinge on the decoder's saddle statistic;
- a decoder-in-the-loop hinge.

Per-crown canopy balancing does pass more small crowns through the foreground gate on every seed (3 paired seeds), but
they are then merged or poorly delineated, and recall does not rise beyond noise.

We conclude that small-crown identity is a *visible-yet-unseparated* failure: the evidence is in the image, but neither
dense fields nor query models learn to turn it into separate instances. Loss reweighting is the wrong lever.

---

## 1. Introduction `[TODO prose; skeleton]`
- Why small crowns matter: counts, regeneration, mortality. 73% of test crowns are < 9 m² (`docs/TUYEN_BO_*`, to
  verify).
- Bottom-up fields: Cellpose / Omnipose-style flow + SDT. TreeFlowNet adds saddle merging and rescoring (section 3).
- Gap: TFN 34.22 ± 0.30 vs Mask R-CNN 34.82 ± 0.67 test AP; difference +0.59 [0.15; 1.02]
  (`crown_segmentation_research/clean_benchmark/results/master_table_test.json`).
- Question: *where* exactly, *why*, and *what information* would fix it.
- Contributions:
  1. A protocol that isolates a failure mode down to the missing information: stage decomposition → oracle
     information ladder → evidence tests (train fit, cross-architecture, physical reference, blind human audit).
  2. The finding that small-crown merges are visible yet unseparated, shared across architectures and present on
     training images.
  3. A negative-result map of four loss-level interventions with seed-noise controls, showing why the decision is not
     reachable through pixel-level losses.

## 2. Related work (only works read in full; notes in `LITERATURE_FULLTEXT_20261119/READING_NOTES_*`)

**Bottom-up instance fields.**
- Cellpose and Omnipose regress a flow towards instance centres (or along distance-field gradients) and form instances
  as the sinks of an advection.
  - Instance existence is decided by the dynamics, not supervised directly.
- FG-TreeSeg applies Cellpose-SAM flows zero-shot to tree crowns. It reports that a global diameter parameter trades
  separation against aggregation, and that extreme canopy fusion stays ambiguous (AP50 only).
- TreeFlowNet (in-house, unpublished) belongs to this family. It adds an SDT head and merges basins by single linkage
  on saddle passes.

**Top-down and prompted models.**
- Mask2Former states that small objects are a weakness. Mask DINO selects queries from dense encoder scores.
- SelvaMask (detector → fine-tuned SAM) reports the best Tiny (< 9 m²) mAP of 9.2 on tropical crowns. Its
  inter-annotator mRF1 for Tiny crowns is only 12.7–20.1, a human-consensus ceiling.
- SelvaBox shows that resolution and multi-resolution training are first-order factors for crown detection.
- Teng et al. (2025), on the same QuebecTrees date, show that a photogrammetric DSM adds no mAP as an input (32.44 vs
  32.37) and that a DSM-only model reaches 9.2 mAP: crowns hardly stand out in the DSM of this closed canopy. This
  agrees with our F2.

**Over-segment and merge.**
- TreeCoG over-segments with a learned edge detector and learns merges with a GCN supervised by GT majority mapping.
- TreeCoG and FG-TreeSeg report AP, but neither isolates which decision fails for small crowns.

**Decision-aligned training.**
- MALIS trains affinities on the maximin edges that cause splits and merges. Its constrained version (Funke et al.)
  strongly reduces both errors on 3D EM neurons.
- Our saddle decoder is a maximin partitioner on min(SDT), so methods 2–3 are MALIS-type losses on the decoder's own
  statistic.
- Unlike the EM results, neither variant helped here (Section 7). The decisive merges pass through flow-defined basin
  territories, not only through SDT edges.

**Reference uncertainty.**
- RandCrowns quantifies annotator disagreement and proposes a tolerant metric.
- Allen et al. (2025) argue, with TLS references, that visual canopy breaks do not always correspond to trees in closed
  canopy. Their manual and TLS sites differ, so this is not a controlled same-image comparison.
- Pesonen et al. (2026) show models can agree with manual outlines better than with their coarse training labels.
- Our blind audit (Section 6) adds a within-dataset, group-blind measurement of how often small labelled crowns are
  visibly separable at all.

`[TODO before submission: read in full and add Slot-Attention-family pruning (AdaSlot, PSA, MetaSlot; notes exist),
StarDist, Deep Watershed, HoVer-Net, Steier et al. 2024, BAMFORESTS, Detectree2, DeepForest, focal / instance-balanced
losses, constrained MALIS follow-ups (Mutex Watershed).]`

## 3. Setup
- **Data:** QuebecTrees (CanopyRS split), 3 cm GSD.
  - Validation: 139 tiles of 1,666 px with 50% overlap, so all crown statistics are deduplicated by world centroid.
  - The test set was used once for the benchmark only; every analysis here is on validation or train.
- **Model:** TreeFlowNet, Swin-T + FPN P2.
  - Heads: flow, SDT (v2 targets, margin 256), canopy.
  - Decoder: gate canopy > 0.30 & SDT > −0.9 → advection (157 steps) → candidate sinks → assignment (50 px) → saddle
    merge (pass = max over boundary of min SDT, τ = 0.3).
  - All decoder settings are frozen (seed-42 log).
- **Matched baseline M0':** retrained under an integrity harness (`crown_segmentation_research/research_integrity/a06_hardening_v2`).
  It enforces frozen configs, hash-registered inputs, a preflight gate and no resume.
  - 3 seeds: AP 37.87 ± 0.15, R < 2 m² 0.377 ± 0.005 (`ITC_THREE_TRACKS_V1_20261120/track3_formation/RESULTS_EXP035_036.md`).
- **Stages (per GT crown):**
  - L: canopy gate covers < 50% of the crown;
  - H: no candidate or basin owned by the crown;
  - A: hypothesis exists, but best IoU < 0.5 (merged or split);
  - M: best IoU 0.5–0.75;
  - OK: best IoU ≥ 0.75.
  - Code: `ITC_INSTANCE_FORMATION_V1_20261120/code/error_sources.py`.

## 4. Where the failure is
- **Table 1: error stages by size** (TFN s42, 4,301 unique crowns; `ITC_INSTANCE_FORMATION_V1_20261120/error_sources.json`).
  - All crowns: M 24%, A 12%, L 8%, H 2.3%.
  - < 2 m²: L 28%, H 12%, A 14%, M 26%.
- **Neighbour context** (M0', 3 seeds; `ITC_INFORMATION_LADDER_V1_20261121/results/stages_*`):
  - 215 of 387 small crowns touch a crown ≥ 4× larger;
  - they hold about 68% of H+A failures (73 / 80 / 77 of 110 / 112 / 117).
- **Training images** (EXP-030 T1): H+A rate 0.336 on train vs 0.283 on validation. The model does not fit these crowns
  even when it has seen them.
- **Across architectures:**
  - 86% of identity errors < 1 m² are shared by TFN and Mask R-CNN (`ITC_IDENTITY_ERROR_ORIGIN_V1_20261119`).
  - Mask R-CNN hits 9% of TFN-H+A crowns vs 40% of TFN-correct crowns; H crowns 1/48 (`ITC_INFORMATION_LADDER_V1_20261121/RESULTS_EXP030.md`).
- **Physical reference** (MillionTrees TLS, evaluation only): absorption of < 2 m² vs ≥ 9 m² crowns: TFN 0.165 vs 0.029,
  Mask R-CNN 0.097 vs 0.011 (`ITC_THREE_TRACKS_V1_20261120/track1_tls/RESULTS_T4.md`).

## 5. What information fixes it: oracle information ladder (EXP-029)
- Method: replace one predicted quantity with its stride-4 ground truth only within 8 px of the 210 failed crowns; keep
  the frozen decoder.
  - Implementation checks: the baseline reproduces the decomposition exactly; whole-tile GT fields recover 0.99.
- **Table 2** (`ITC_INFORMATION_LADDER_V1_20261121/RESULTS.md`):

| Injected | Information | Recovered (IoU ≥ 0.5) | L / H / A |
|---|---|---|---|
| centre point | existence | **0.000** | 0 / 0 / 0 |
| canopy | foreground | 0.219 | 0.31 / 0 / 0.24 |
| SDT | separation | 0.019 | 0 / 0 / 0.07 |
| flow | membership | 0.152 | 0 / 0.40 / 0.22 |
| SDT + flow | | 0.457 | 0 / **1.00** / **0.83** |
| canopy + flow | | 0.490 | **0.54** / 0.42 / 0.47 |
| all three | | 0.986 | 1.00 / 1.00 / 0.95 |

- Size (EXP-017: +0.56 AP ceiling) and count (G0: +1.72 AP, no small-crown recall gain) are also insufficient.

## 6. Is the information in the image?
- **Colour contrast** at the boundary with the larger neighbour does not separate failed from correct crowns (AUC 0.49;
  EXP-030 T3).
- **SAM with the true centre** (smallest of three masks): hits 0.26 of H+A vs 0.39 of correct crowns, whereas TFN given
  the same point recovers 0 (`RESULTS_EXP032.md`).
- **Blind visual audit** (`ITC_BLIND_VISUAL_AUDIT_V1_20261122/RESULTS_EXP033.md`):
  - 150 crops, half failures, half size-matched controls, shown in random order with no group labels.
  - P(separate crown visible) = 0.47 for failures vs 0.63 for controls; ratio 0.75 [0.55; 1.00].
  - Merged (A) crowns: 0.57, ratio **0.91**.
  - No-attractor (H) crowns: 0.37, ratio 0.59.
  - Crowns < 1 m²: 0.22 vs 0.50.
- **Reading:**
  - Merges are visible yet unseparated.
  - H crowns and crowns < 1 m² are partly evidence-limited.
  - About a third of small *control* labels are not visibly separable either, which bounds RGB-only performance on
    this label set.

## 7. Why loss-level fixes do not work
**Table 3:** one seed each, against M0' mean ± SD over 3 seeds (`RESULTS_EXP035_036.md`, `RESULTS_EXP028.md`,
`RESULTS_EXP031.md`, `RESULTS_EXP034.md`).

| Intervention | Targets | AP | R < 2 m² | val H | val A | train A |
|---|---|---|---|---|---|---|
| M0' (3 seeds) | — | 37.87 ± 0.15 | 0.377 ± 0.005 | 51 ± 2 | 62 ± 2 | 420 ± 6 |

(M0' val L 104 ± 6; M3 s42 val L **85**: M3 moves crowns out of the canopy gate, but they then fail at H/A.)

| M3: per-crown canopy + Lagrange (3 seeds, paired Δ) | L | ΔAP −0.04 | ΔR +0.010 (+0.025/+0.012/−0.006) | ΔH −1.3 | ΔA **+7.7** | Δ train A +40 (ΔL −15 val, −102 train) |
| crown-weighted flow/SDT | H | 36.78 | 0.427 | **35** | 62 | 470 |
| GT-boundary saddle hinge | A | 36.68 | 0.411 | 51 | 56 | 450 |
| decoder-in-the-loop hinge | A + splits | 34.05 | 0.377 | 48 | **85** | 536 |

- Crown reweighting creates attractors (H −30%) but the new hypotheses are then merged (A unchanged).
- The GT-boundary hinge is satisfied on train (violations 2.0% → 0.8%), yet train merges rise.
  - Diagnostic: 98% of A crowns own a basin; only 43% have a high GT-boundary pass; only 18% a high direct pass to the
    absorber.
  - So merges travel through intermediate basins in background gaps.
- The decoder-in-the-loop hinge reduces wrong merges on training crops (11 → 3 per crop) but destabilises the field
  (AP 30.65–34.05 across snapshots).
- Interpretation: the decision that fails is made at the level of basin topology, through flow-defined territories and
  single-linkage chains. Pixel- and pair-level losses on SDT do not reach it.

## 8. Discussion and limits `[TODO prose]`
- Visible-yet-unseparated suggests the next method must represent object-level evidence (OCAAD-style), not reweight
  losses.
- Limits:
  - one benchmark;
  - one rater;
  - one seed per intervention;
  - six spatial blocks on validation;
  - visibility in RGB is not biological identity;
  - the TLS domain differs.

## 9. Before submission (work items)
1. ~~Finish EXP-037~~ done (NOT CONFIRMED; Table 3 filled). Remaining interventions (methods 1–3) are still one seed each; decide whether 3 seeds are needed for the paper.
2. Second rater, or a second blinded pass, for EXP-033.
3. Spatial-block CIs where possible; report n of blocks.
4. Same-backbone fairness for the benchmark (Swin-T vs R50) from dossier WP1, or state it as a limit.
5. A second dataset for C1–C3: at least a rerun of the stage decomposition on another dataset, or on the TLS set.
6. Related work, read in full (project rule).

## Figures (existing files)
- F1: stage decomposition by size and neighbour ratio. `[to plot from stages_*.json]`
- F2: oracle ladder panels: `ITC_INFORMATION_LADDER_V1_20261121/previews/ladder_{L,H,A}.png`.
- F3: blind-audit examples. `[to pick after deciding consent/anonymisation of crops]`
- F4: SAM whole-canopy choice: `ITC_INFORMATION_LADDER_V1_20261121/previews/exp032_sam_whole_canopy_choice.jpg`.
- F5: intervention previews: `ITC_THREE_TRACKS_V1_20261120/track3_formation/previews/exp0{28b,31b,34,36b}_val*.png`.
