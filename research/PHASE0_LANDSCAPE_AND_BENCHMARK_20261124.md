# Phase 0: research landscape and benchmark selection. STATUS: PROVISIONAL, PHASE 0 IN PROGRESS. Opened 2026-11-24; revised the same day after review.

Phase 0 must answer with evidence:
1. Does small-to-large absorption exist beyond TFN and QuebecTrees?
2. How do strong current methods fail on it, and at which step?
3. Which assumption must a new method change?

It is a targeted survey with decision gates (G1–G5), NOT an open-ended audit.

## Model-agnostic absorption (defined for EVERY instance; revised)
**Matching**
- One-to-one, COCO-style: greedy by score, IoU ≥ 0.5.
- s is a *hit* if it is matched.

**Neighbours**
- N(s) = the GT instances within d px of s.
- d = 2 px is primary. Sensitivity is reported at d = 1 and d = 5 px, because a pixel means different physical sizes
  across datasets. The conclusion must hold at all three.

**Absorption event**
- M(s) = 1 iff s is not a hit AND there exist b ∈ N(s) and a prediction P with |P∩s|/|s| ≥ 0.5 and |P∩b|/|b| ≥ 0.5.

**Strata by neighbour asymmetry**, r(s) = max_{b∈N(s)} |b| / |s|:

| Stratum | Condition |
|---|---|
| balanced touching | r < 2 (this includes s being the larger one) |
| moderate | 2 ≤ r < 4 |
| strong (at-risk) | r ≥ 4 (the stage_eval ratio, fixed before EXP-031) |
| non-touching | N(s) = ∅; control for misses only, since M is 0 by definition |

**Comparison**
- P(M = 1 | strong) vs P(M = 1 | balanced), stratified by absolute instance size (dataset-specific quantile bins) and by
  local density (count of instances within 3 diameters).
- The comparison is per dataset, never pooled across datasets.

TFN's H/A stays a TFN-internal diagnostic.

## D1. Literature (two tiers)
**Landscape tier**
- Each family (grouping / affinity, embedding, flow dynamics, object-centric / query, foundation cell models) has at least
  one representative read in full.
- Already read: MALIS, Funke, SEAL, Kong 2018, Neven 2019, Cellpose, Omnipose (+ code), DINO-Nucleus, FlowSeg.

**Novelty-critical tier**
- The works closest to whatever mechanism is proposed, including their successors and recent work, are read in full.
- Stop rule: the nearest works are identified, and the difference from them can be stated as a concrete mechanism.
- Queue: LIVECell (dataset and baselines), CellSAM (2025), Cellpose-SAM, one query-based instance model.

## D2. Dataset suitability (annotations only, no GPU)
- Report for each dataset:
  - instance-size distribution;
  - counts per stratum;
  - r distribution;
  - annotation protocol (exhaustive or not, ignore regions);
  - annotation-consistency evidence (inter-annotator data if published);
  - licence and access.
- No hard rejection thresholds. ≥ 1,000 strong-stratum instances is a PRIORITY level, not a requirement.
- The deciding question: is the count large enough that a baseline's strong-stratum absorption rate gets a 95% CI half-width
  ≤ 0.05?
- Candidates: QuebecTrees (reference), LIVECell, CVPPP (check the licence first), PanNuke.

## D3. Baseline × failure matrix (effect-size report, no single hard gate)
- For each dataset × baseline report:
  - M rate per stratum, with Wilson CIs and absolute counts;
  - the strong/balanced difference and ratio within size and density strata;
  - the share of all missed instances that are absorption events;
  - the recall and AP that would be recovered if strong-stratum absorptions were hits (upper bound on consequence).
- Baselines:
  - at least one task baseline (Mask R-CNN or query-based);
  - at least one mechanism baseline (Cellpose-family or TFN).
  Released checkpoints are preferred.
- If several model families fail, that shows the problem is general. It does NOT show that one mechanism fixes all of
  them; failure reasons are analysed per family.

## D4. Research-gap statement
Keep separate:
- (a) the observation: what fails, where, how much;
- (b) the mechanism hypothesis: why;
- (c) why the nearest D1 works do not remove it;
- (d) which assumption a new method changes.

## Decision gates (revised)
- **G1 Problem validity.** The failure is defined independently of TFN and measured on annotations with a clear protocol.
- **G2 Evidence.** The failure is empirically meaningful (D3 effect sizes) on at least one strong baseline. Evidence
  outside ITC is required for a core-CV claim.
- **G3 Literature gap.** A concrete limitation that the nearest methods do not satisfactorily resolve under the tested
  conditions.
- **G4 Mechanism opportunity.** At least one principled change to representation, objective or inference that does not
  repeat a closed recipe.
- **G5 Evaluation feasibility.** Baselines, metrics, training budget, label treatment and split are fixed before the main
  method is evaluated.

Decision:
- **G1–G5 met:** LOCK the problem and evaluation protocol. Architecture and formulation stay open.
- **Evidence only within ITC:** lock an ITC-specific contract instead of a core-CV claim.

## Side work (allowed; does NOT decide the lock)
- EXP-044 (field-space SDT test) answers an internal TFN question. A negative result does not close Track A; a positive
  result does not lock it.
- Formulation work.

## Not allowed until LOCK
- Multi-arm training campaigns.
- Novelty claims.
- SOTA comparisons.
