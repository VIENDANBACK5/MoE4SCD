# One finite frozen diagnostic — design only

Status: PLANNED_NOT_RUN. No arm below has been executed and no transform has been used to select a favorable policy. This is a candidate diagnostic for the active original-method search, not a new carrier/training study.

## Question and limits

H_encoding: with identical RGB information and physical support, the trained carrier's raster-dependent inference loses previously recoverable reference instances.

H_deleted_information_only: changes require a noninjective observation transformation; the information-preserving contrasts do not exhibit practically important losses.

These are narrower than the user's broad H1/H2. A positive lossless contrast can rule out deleted input information as the explanation for that contrast, but cannot choose between encoder access, receptive-field effects, decoder geometry, numerical sampling or insufficient augmentation. A negative result at one factor cannot prove global scale robustness. Losses after degradation alone do not prove actual loss of visual identifiability.

## Four inputs, same scene support

Let I be a fixed eligible native1024×1024 TRAIN RGB observation, U2 exact2× nearest-neighbor replication, D2 fixed nonoverlapping2×2 mean integration. Retain sufficient precision for D2 and do not introduce an extra quantization difference between paired branches.

| Arm | Input | Dimensions | Information relation | Primary contrast |
|---|---|---:|---|---|
| A | I | 1024² | Original observation | Reference |
| B | U2 I | 2048² | Exactly equivalent to A | A↔B: lossless raster re-encoding |
| C | D2 I | 512² | Detail deleted relative to A | Coarse observation |
| D | U2 D2 I | 1024² | Exactly equivalent to C | C↔D: encoding conditional on the same degraded observation |

A↔D retains numerical grid dimensions and physical support, but changes observation detail. It is not an isolated realistic sensor/GSD intervention. B's receptive field in physical units and compute differ from A; those are explicit parts of the network's encoding response, not silently matched computation. The four-arm design is not described as a fully orthogonal2×2 with equal dimensions.

Physical FOV is fixed in all four arms. A later FOV intervention would be a separate registered factor; it is not bundled into "GSD effect." No extra tiles, higher query count, view voting, fusion or post-hoc score selection.

## Carrier and cohort before execution

Candidate principal carrier: archived original mask-only DINO seed42, frozen, exactly300queries, raw logits through the existing carrier.py path. Its inspected preprocessing normalizes and pads, without forced resize. Actual tensor shape/pixel relations must be verified before accepting a contrast.

Do not add R5 as if its native pipeline were equivalent: its inspected transform has min_size=max_size1024 and would resize these inputs. Disabling that is a separate model/inference-policy intervention.

If this diagnostic proceeds, choose a bounded metadata-only TRAIN cohort from the established eligible manifest, spanning the three recorded sources, before new predictions. Do not select by previous success, scale response or SAME6. Exact image IDs, number of windows, fixed mask materialization, resource limits and runtime budget must be committed first. No such new execution cohort is claimed selected in this research gate. Archived SAME6/BAM remain exposed, not final confirmation.

## Required control of geometry and validity

Use known pixel-edge coordinates to map each view back to the canonical source support. Nearest-neighbor replication has an exact left inverse on RGB; coordinate mapping of predicted masks is separately defined. Do not call mask pullback recovery of image detail.

For raw geometry comparisons, retain every query's logits, class score, query/site lineage and native logit-grid dimensions. Record whether changing input size creates more encoder sites at the fixed query budget. This is an endogenous mechanism observation, not a new query-selection experiment.

Common analysis must not apply a30-pixel floor independently in four different grids. Map predictions to the canonical coordinate domain first and use one fixed canonical-area rule, score definition, threshold and mask-IoU suppression policy. Native per-arm pixel-floor results may be secondary, clearly separated. Analyze raw masks before score/area/NMS so final changes cannot be mislabeled formation.

Downsampling alpha requires an explicit footprint-validity rule. Any common-support restriction must be declared, with changed pixels/reference denominators reported; never silently remove ambiguous or boundary references. Preserve independent overlapping masks and original source-crown area versus crop-fragment distinctions. GT/reference masks enter diagnostics only after RGB predictions are frozen; never routing, initialization or branch choice.

## Outcomes

Primary: gross new/lost reference qualifications in lossless pairs, raw availability50/75 and one-to-one final TP50/75 after canonical mapping. Report overall and original-source-small where valid, per source, raw-versus-final transitions, FP, duplicate proxies and score distributions. A mere change in mask count is insufficient.

Use the same observations across arms; views/crops/seeds are repeated measurements, not independent trees/sites. A set-to-set mask association is descriptive and never a replacement for actual reference matching. Inspect successes and losses by a fixed rule, without using them to choose scale factors/thresholds.

No H_encoding claim if differences are explained by auto-resize, coordinate/pasting error, validity change, unmatched postprocessing units or lossy quantization. Preprocessing/kernel checks are engineering prerequisites, not the outcome.

## Stopping and original-method entry

If lossless contrasts show little relevant formation/retention change, do not train an observability module. If only lossy contrasts regress, preserve ambiguity between removed detail and distribution effects; do not force count consistency.

If lossless contrasts show a substantial reproducible cohort of real reference losses, freeze that finding and choose one discriminating internal intervention against competing causes. ScaleMatch/DINO-MS or a faithful mechanism adaptation must be considered where actually relevant; do not jump from scale sensitivity to novel consistency loss. An information-equivalent failure need not be a previously unknown failure.

The practical method target remains net source-small recovery with AP/overall recall and FP constraints, not minimizing disagreement alone. No training, hyperparameter sweep, scale-routing policy or new architecture is queued by this design.
