# Phase 0 synthesis (2026-11-24). Status: **PHASE 0 COMPLETE for ITC scope. TRACK A: PROBLEM LOCKED · MECHANISM OPEN · NOVELTY NOT ESTABLISHED. Phase 1 (mechanism invention): Prototype 1 done — M2 LEARNABLE_NOT_BETTER (read the corrections at the top) (see ../PHASE1_20261124/RESULTS_M2_PROTOTYPE.md)** (owner decision 2026-11-24; previously REVISE).

Evidence labels:
- [V] verified on data in this repo;
- [L] read in full in the literature;
- [H] hypothesis;
- [P] proposal.

## 1. Literature landscape matrix (D1) chỗ nào ?

| Family | Representative (read in full) | Mechanism that forms instances | Stated limitation / relation to absorption |
|---|---|---|---|
| Affinity + graph partition | MALIS 2009; Funke 2020 [L] | Learned affinities → connected components / agglomeration | Pair-counting loss: merging a small object costs \|A\|·\|B\|, which favours sacrificing small objects |
| Consequence-weighted pixel loss | SEAL 2018 [L] | Superpixel leakage weights a boundary BCE | Credit sits on the GT boundary (failed for us, method 2) |
| Embedding + clustering | Kong & Fowlkes 2018; Neven 2019 [L] | Unrolled mean-shift; per-instance σ and Lovász | Per-instance normalisation; no absorption analysis |
| Flow dynamics | Cellpose 2021; Omnipose 2022 (+ code); DINO-Nucleus 2026 [L] | Flow-to-centre + Euler grouping; DINO-Nucleus trains through integration | Endpoint loss targets ANY centre; no absorption analysis |
| Field reshaping | FlowSeg 2026 [L] | Stop-gradient repulsive target | Attractor gradient pathologies |
| Detection + mask (NMS) | Mask R-CNN; LIVECell CenterMask2 [L] | Box/centre proposals + NMS + per-box mask | LIVECell: anchor-based models break down in over-confluence |
| Query / set prediction | CellSAM, Nature Methods 2025 [L] | Anchor-DETR boxes (3,500 queries, NMS-free) prompt SAM; output = argmax label map | Motivated by NMS failure in dense packing; **absorption not measured**; LIVECell only zero-shot |
| Query + inter-instance repulsion | MDQE, CVPR 2023 [L] | Matched queries; BCE/Dice-inter loss pushes each mask off nearby GT (box-IoU > 0.1) | **Prior art for P1.** Acts on matched queries only; +2.2 mAP on OVIS; no merge metric |
| Learned merge/split correction | Zung et al., NeurIPS 2017 [L] | Error detector (shape cues) + mask-pruning corrector on a supervoxel graph | **Prior art for P2.** 707/944 errors fixed; cannot fix errors inside a supervoxel |
| Centre-guided clump splitting | Split and Expand, 2022 [L] | Predicted centres in a clump → weighted GMM partition; LRP expands missed centres | **Prior art for P2/P3.** Needs reliable centres; AJI +0.02 to +0.05 |
| Overlap-aware mask loss | "Overlap Loss", ICIP 2023 | — | **NOT READ** (cited by an external review; no full text) |
| Foundation flow | Cellpose-SAM 2025 | — | **NOT READ** (the stored file is an HTML error page) |

**Closest prior art for any absorption-targeted method:**
- CellSAM-style NMS-free set prediction;
- MDQE-style inter-instance repulsion;
- Zung-style learned merge detection and correction;
- Split-and-Expand-style centre-guided splitting;
- DINO-Nucleus-style trained dynamics.

None of them reports an absorption / merge rate on a common metric, so whether they solve the problem is unknown, not
known-yes. What IS known is that the ingredients of P1–P3 already exist (see §5).

## 2. Dataset suitability (D2) [V]

| | QuebecTrees val | LIVECell val |
|---|---|---|
| Instances (non-border, de-duplicated) | 3,671 crowns | 160,540 cells |
| Strong-asymmetry (r ≥ 4) | 931 (25%) | 12,479 (7.8%); 8,307 after removing slivers |
| Annotation slivers (inscribed radius < 3 px) in the strong stratum | 0% | **33%** |
| Exhaustive? | No, by design (Cloutier 2023) | No, by design (Edlund 2021, p. 1044) |
| Licence | Research use | CC BY-NC 4.0 |

Both datasets support studying absorption. LIVECell needs a sliver filter; the thickness ≥ 3 px rule is POST-HOC and is
reported as a sensitivity analysis.

## 3. Baseline × failure matrix (D3) [V]

Absorption event = the instance is not matched one-to-one, and one prediction covers ≥ 50% of it and ≥ 50% of a touching
GT neighbour. d = 2 px; the conclusions hold at 1 and 5 px.

| Dataset | Model (family) | Recall | Absorption share of misses | Recall if ALL absorptions were hits | … if STRONG-stratum ones were | Asymmetry effect, standardised for size × density [95% CI] |
|---|---|---|---|---|---|---|
| QuebecTrees | Mask R-CNN ×3 seeds (NMS) | 0.71–0.72 | 0.31–0.44 | +0.087 to +0.125 | +0.036 to +0.048 | −0.006 to +0.013, all CIs ∋ 0 |
| QuebecTrees | TFN-c5 ×3 (flow + SDT) | 0.71–0.73 | 0.31–0.32 | +0.086 to +0.089 | +0.032 to +0.033 | −0.004 to +0.018, ∋ 0 |
| QuebecTrees | TFN-M0' ×3 | 0.71–0.72 | 0.27–0.29 | +0.075 to +0.084 | +0.030 to +0.032 | −0.003 to +0.031, ∋ 0 |
| LIVECell | Cellpose3 livecell_cp3 (flow) | 0.62 / 0.64* | 0.22 / 0.20* | +0.085 / +0.075* | +0.022 / **+0.010*** | +0.18 [0.14, 0.21] / **+0.12 [0.10, 0.13]*** |
| LIVECell | CenterMask2 (LIVECell authors) | 0.79 / 0.81* | 0.34 / 0.30* | +0.071 / +0.055* | +0.024 / **+0.008*** | +0.25 [0.16, 0.35] / **+0.11 [0.09, 0.12]*** |

\* After removing slivers.

Where the absorbed instances sit, by asymmetry stratum, after the sliver filter:
- LIVECell: balanced 3,960–5,547, moderate 3,387–4,416, strong 1,290–1,634.
- QuebecTrees: roughly a third in each stratum.

## 4. Research-gap statement (D4)
**(a) Observation [V].**
- Absorption (two touching GT instances covered by one prediction) is a major error for every model family tested
  (NMS-detection, flow + SDT, flow-only) on both datasets: 20–44% of all misses, worth +5.5 to +12.5 recall points.
- It is driven mainly by absolute size and crowding. Neighbour-size asymmetry adds a modest, dataset-dependent effect:
  - none on QuebecTrees;
  - +0.11 to +0.12 on LIVECell after removing annotation slivers, cell-type dependent, and reversed in BT474.
  - Asymmetry-specific absorption is worth only about 1 recall point on LIVECell and about 3–5 on QuebecTrees.

**(b) Mechanism hypothesis [H], revised 2026-11-24 after reading MDQE / Zung / CellSAM.** WEAKENED by Step 2a: in-domain Mask2Former, whose loss does see every instance, still absorbs at 0.88–0.98 × Mask R-CNN's rate. See D1.1 for the formation-stage statement that replaces it.
- The earlier wording ("nothing in the objectives penalises one prediction claiming two instances") was **wrong**.
  Any per-mask BCE/Dice loss already penalises the pixels of B inside the mask matched to A, and MDQE up-weights exactly
  those pixels.
- Revised hypothesis: what differs between families is whether the **formation step** (which instances exist) is
  inside the training loss.
  - **NMS detection:** the final NMS is not in the loss, so a proposal for B suppressed by A's mask is never penalised.
  - **Flow / SDT grouping:** Euler integration, basin merging and mask assembly are not in the loss. Cellpose and TFN
    train per-pixel fields only.
  - **Set prediction (DETR / Mask2Former / CellFinder):** Hungarian matching gives every GT a query during training, so
    a missing B is penalised directly.
- Prediction:
  - If the hypothesis holds, a set-prediction model of comparable strength absorbs less.
  - If it absorbs as much, "formation outside the loss" is not the explanation, and the gap is in representation or
    evidence.

**(c) Why the nearest works may not cover it [H, untested].**
- CellSAM removes NMS but still needs a query per instance. Its inference also thresholds boxes and takes an argmax
  over masks, and either step can lose an instance. Its absorption rate is unmeasured.
- MDQE's repulsion acts only on matched queries; it cannot create a hypothesis that was never formed.
- Zung-style correction assumes the candidate representation still contains the cut.
- DINO-Nucleus rewards trajectories ending at ANY centre, so absorption into a neighbour's centre is not penalised.

**(d) Assumption a new method must change [P], narrowed.**
- "A learnable one-vs-two test trained on absorption events" is NOT new on its own; Zung 2017 is the precedent.
- What remains open is how to **form a new, correct instance hypothesis when the existing predictions or grouping
  representation have already collapsed two touching objects**.
- Whether that open question is real depends on G3 and on the decomposition in §6.

**Revision of the provisional target.** The working target "small-to-large absorption" is NOT supported as the defining
problem: asymmetry is not the main driver, and its share of the error is small. The evidence supports the more general
target: **absorption of touching instances in crowded scenes**, with small size as the main risk factor.

**(e) Working research question (owner-approved wording, 2026-11-24; NOT a novelty statement).**
- **Empirical finding.** Across NMS-based detection, flow grouping and in-domain set prediction, touching-instance
  absorption remains a substantial error on the evaluated ITC benchmark. In Mask2Former, most absorbed instances lack an
  adequate decoded mask candidate, despite instance-level matching during training.
- **Research hypothesis.** Existing formation mechanisms may fail to retain or recover distinct object hypotheses when
  adjacent instances produce ambiguous and competing visual evidence.
- **Methodological question.** Can a learnable instance-formation mechanism jointly revise object hypotheses and their
  pixel assignments under structural competition, rather than relying solely on initial candidates, independent object
  predictions or fixed grouping decisions?
- **Novelty status: open.** The mechanism must be distinguished experimentally and algorithmically from:
  - recurrent detection;
  - iterative refinement;
  - learned merge correction;
  - query re-selection;
  - birth–death inference.
- **Not proven:** "no adequate decoded mask" does NOT imply "no object information in the features".

**Lock table (owner, 2026-11-24).**

| Item | Status |
|---|---|
| Touching-instance absorption on ITC | LOCKED |
| SelvaMask / QuebecTrees + current evaluation protocol | LOCKED for exploratory Track A |
| LIVECell external validation | kept; cross-domain claim not locked |
| Small-to-large as the main cause | DROPPED |
| Missing decoded candidate as a key failure stage | kept (as measured) |
| Missing object representation in features | NOT proven |
| Multi-pass instance generation as novelty | NO |
| Joint hypothesis–membership revision | research direction |
| Method for full training | NOT chosen |

## 5. Candidate method principles (D5) [P]
**P1. Exclusive-coverage set prediction (query family).**
- Each GT instance must own a query, and an objective term penalises any prediction whose mask covers ≥ 50% of two GT
  instances. This is the absorption event, made differentiable through soft mask coverage.
- Novelty risk: **VERY HIGH**. MDQE (CVPR 2023) already adds an inter-instance repulsion term to BCE/Dice, and per-mask
  losses already penalise covering a neighbour. P1 survives only as a baseline, not as a contribution.
- Decisive experiment, before any invention: measure absorption of a query-based model (Mask2Former, or CellSAM if weights
  are usable) with the D3 metric. If query models absorb as much as NMS / flow models, set prediction alone does not fix it
  and P1 is justified; if they absorb much less, the gap closes.

**P2. Split-or-merge verification on candidate merges.**
- A second-stage head looks at each predicted instance and asks "does this contain two objects?"; it is trained on
  absorption events mined from the base model's own training predictions.
- Unlike EXP-016's pass threshold or CCF's fixed hierarchy, the decision is learned from the error itself and is
  model-agnostic, so it can be applied to NMS, flow and query outputs.
- Novelty risk: **HIGH**.
  - Zung et al. 2017 already learns merge/split detection (precision and recall > 90%) and correction (707/944 errors
    fixed).
  - Split and Expand already splits clumps from predicted centres.
  - A new P2 would need a hypothesis-forming mechanism for cases where no candidate cut or centre exists.
- Decisive experiment: an oracle splitter. Replace every absorbed prediction by its GT split and measure the recall/AP
  ceiling on both datasets. Upper bound from D3: +5.5 to +12.5 recall points.

**P3. Local-count-consistent formation.**
- Predict a density (count) map; formation must produce as many instances in a region as the integral says.
- Novelty risk: MEDIUM.
- Caution: EXP-029 found that injecting correct existence points into the TFN decoder recovered 0/210, because the flow
  carried pixels to the neighbour. A count constraint therefore has to act on formation, not on seeds.
- Novelty risk: HIGH (Split and Expand uses centre count to set k).
- Decisive experiment, concretised:
  - Inside each absorbing predicted mask, partition the pixels geometrically using the oracle GT centres (Voronoi and
    weighted GMM, Split-and-Expand style), with no flow.
  - Measure how many absorbed instances reach IoU ≥ 0.5.
  - This differs from EXP-029, which injected points into the flow decoder.
  - High recovery → count/centres suffice and the problem is centre detection (known territory). Low recovery →
    membership, not count, is the bottleneck.
  - It runs on CPU using existing predictions.

## 6. Track A decision (D6): **REVISE (not locked)**

| Gate | Status |
|---|---|
| G1 Problem validity | MET. Absorption is defined independently of TFN and measured across 5 model families × 2 datasets |
| G2 Evidence | MET for "absorption in crowded touching scenes", including outside ITC. NOT met for "small-to-large" as the defining variable |
| G3 Literature gap | **ITC side: the pre-registered test is complete** (RESULTS_D3Q_SELVAMASK.md). It answered only "does in-domain set prediction remove absorption?", and the answer is no. It does NOT establish the literature gap or novelty; see D1.1 (`D1_1_MECHANISM_GAP_ANALYSIS.md`).
  - In-domain Mask2Former (3 seeds) absorbs at 0.88–0.98 × the rate of Mask R-CNN at comparable recall (0.62–0.64 vs 0.66); TFN absorbs at the same rate but with lower recall (0.507), so it is not strength-matched.
  - 89–91% of Mask2Former's absorptions have no raw query with mask IoU ≥ 0.5.

  **LIVECell side:** zero-shot only, inconclusive (below). Earlier status: CellSAM zero-shot on LIVECell val (RESULTS_D3_QUERY_CELLSAM.md) is **INCONCLUSIVE** under the pre-declared rule: the absorption rate is higher (0.124 vs 0.071), but recall is 0.30 and the absorption is colony-level (median 6 cells per mask). Its decomposition at the output stage:
  - 46% have no box with IoU ≥ 0.5 (localisation);
  - 46% have a box below the threshold (selection);
  - 8% are assignment errors.

  With oracle GT-box prompts the rate falls to 0.047. This is zero-shot evidence about failure stages, not a general mechanism. Step 1 (threshold sweep): lowering the box threshold to 0.05 rescues only 20% of absorptions and doubles the absorption rate (0.124 → 0.260), because the extra candidates include colony masks. Score calibration does not fix it. An in-domain query model is still needed. Previously: requires the absorption rate of a query-based model. MDQE, Zung 2017 and Split-and-Expand show that P1–P3 ingredients are prior art. Cellpose-SAM and Overlap Loss are unread |
| G4 Mechanism opportunity | P1–P3 are downgraded to baselines. A new mechanism must address hypothesis formation after collapse |
| G5 Evaluation feasibility | MET. Metric, matching, strata, sliver filter, CIs and two datasets with released baselines are fixed |

Unresolved limits:
- Both datasets are non-exhaustive by design.
- The sliver filter is post-hoc.
- Size bins are coarse on QuebecTrees.
- LIVECell results come from one checkpoint per family.

**Status of the steps (updated 2026-11-24; do not repeat completed work):**

| Step | Status |
|---|---|
| CellSAM zero-shot on LIVECell val, with decomposition and GT-box oracle | DONE, INCONCLUSIVE (`RESULTS_D3_QUERY_CELLSAM.md`) |
| CellSAM threshold sweep (Step 1) | DONE; score calibration does not fix absorption |
| CenterMask2 multiplicity check | DONE; pairwise, median 2 |
| In-domain query vs NMS vs flow on SelvaMask SAME6 (Step 2a) | DONE; set prediction does not remove absorption (`RESULTS_D3Q_SELVAMASK.md`) |
| D1.1 mechanism-level gap analysis | DONE (`D1_1_MECHANISM_GAP_ANALYSIS.md`); the draft gap statement awaits owner review |
| In-domain query model on LIVECell (Mask2Former training) | NOT DONE; optional, only if a cross-domain claim is wanted |
| Split-style oracle-centre partition of absorbed masks (P3 decisive test) | NOT DONE |

**Next:** owner review of the D1.1 gap statement, then a decision on locking the problem and evaluation, then mechanism
proposals.
