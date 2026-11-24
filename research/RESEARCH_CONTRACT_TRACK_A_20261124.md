# Research contract, Track A (core CV method). STATUS: PROVISIONAL, PHASE 0 IN PROGRESS (2026-11-24). LOCK only via gates G1–G5 in PHASE0_LANDSCAPE_AND_BENCHMARK_20261124.md.

Purpose: fix ONE scientific target, ONE supervision source and ONE evaluation, so that method results stop being
re-interpreted through dataset or definition questions. Each line traces to a result file.

## Scientific target
Instance formation for touching, visually similar objects of very different sizes. Specifically:
**small-to-large absorption** (stages H and A of crowns < 2 m²) under the TreeFlowNet decoder family.

## In scope (Track A)
- Problem E only: the learning and grouping mechanism.
- Supervision: the existing visible-instance annotations of QuebecTrees (CanopyRS split), used as they are.
- Failure decomposition, with the evidence it rests on:
  - **H, no own basin:** a flow-target problem. Direct flow supervision removes H in field space (EXP-042/043, L_P: H → ≈ 0).
    Method 1 removes about 30% of H in training (EXP-035/036).
  - **A, absorbed:** decided at the SDT pass merge.
    - 98% of A crowns have their own basin; only 18% have a direct own↔absorber pass ≥ τ (EXP-034 diagnostic).
    - Every flow-only change converts H into A (EXP-042/043).
  - **Large-crown splits:** the known cost of size-equal weighting (EXP-041: about 680 extra FP-in on crowns ≥ 9 m²).

## Out of scope (Track B, separate project, not required for a Track A claim)
- Annotation completeness:
  - QuebecTrees is non-exhaustive by design (Cloutier 2023 p. 8);
  - about 54% of strong outside-GT sinks are trees (EXP-040);
  - the effect on AP is about 2 points for every arm, not differential (EXP-041).
- Boundary ambiguity and inter-annotator variation.
- Amodal or physical crowns (would need TLS/LiDAR).
- "Can RGB-only ITC be solved at all?"

## Evaluation (fixed)
- Primary: validation AP (COCOeval segm, maxDets 400, size bins) and R < 2 m², with the frozen scorer eval_exp028.score.
- Structural measures:
  - H and A counts of crowns < 2 m² (EXP-030 stage definitions);
  - large-crown splits (EXP-042 definition);
  - FP-in / FP-out (EXP-041).
- Label treatment (fixed; Track A does not study completeness, but does not pretend it away):
  - Training: unannotated pixels are treated as background, as in all baselines. This is a stated limitation, since
    QuebecTrees has no reliable ignore map. An ignore-aware variant counts as a separate method arm, not a fix to the
    baseline.
  - Evaluation: standard AP is primary. AP_ign (FP-out removed, EXP-041) is a sensitivity analysis only; it is not a
    ground truth and not a replacement.
- Noise floor: 3 seeds; M0' SD is AP 0.15, R < 2 m² 0.005, A 2 (EXP-035).
- Test set: one final run only, after the method is frozen on validation.

## Success condition (method claim)
Against M0' at equal backbone and budget, on 3 seeds:
- A down ≥ 25%;
- H not up;
- large-crown splits not up;
- ΔAP ≥ −0.3, and AP_ign not down;
- R < 2 m² up by more than 3 SD.

The gain must not simply move errors into over-segmentation or false positives.

## Process rule (pre-training test matched to the method type)

| Method type | Test before network training |
|---|---|
| Changes flow / SDT under the existing decoder | Field-space test (EXP-042 harness) |
| Changes the grouping operator | Controlled grouping test on fixed fields / GT-perturbed fields |
| Changes the instance representation | Toy learnability test plus a small network pilot |
| Joint representation–formation | End-to-end learnability and gradient-path test |

The purpose is to spend about 1 hour of GPU, not GPU-days, finding credit or objective failures. An old harness is never
used to reject an architecture it cannot represent.

## Closed directions (do not reopen without new evidence)
- Pixel re-weighting alone (method 1: helps H, costs AP).
- GT-boundary SDT hinge (method 2).
- Decoder-in-loop on SDT arg-max pixels in training (method 3, as implemented).
- Flow-supported gate (EXP-039).
- Partial-label explanation of the small-crown failure (EXP-041).
- Trajectory-credit losses through advection (EXP-042/043; credit collapses at attractors).

## Next experiment within the contract
- EXP-044: the A lever in field space.
  - Start from the flow after L_P (H ≈ 0, basins exist).
  - Optimise SDT only; compare method 3's component-level objective with a pixel SDT loss at equal budget.
- Reading:
  - if it passes, the objective is right and training was the problem;
  - if it fails, A needs a different separation signal.

## Open items
- **A second domain for a CV method paper.** BAMFORESTS lacks the small-crown regime (only 6 interior crowns < 2 m² in 40
  tiles). A cell dataset with small/large touching objects is a candidate. It has not been examined.
- The external review cites MDPI / PMC papers on ITC annotation quality. NOT read yet; read them in full before citing.
