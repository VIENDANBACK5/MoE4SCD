# Frozen object-conditioned readout: bounded TRAIN diagnostic

Status: LOCK BEFORE EXECUTION. This is an oracle capability measurement, not a method, deployable prediction, new witness, query-recycling experiment or independent validation.

## Question and separation from closed studies

The current carrier produces masks by sigmoid(e^T F(x)). Previous regional mean/similarity probes did not directly measure this full readout family. Query-birth training changed decoder parameters; it did not optimize a free final mask embedding on a fixed pixel field. Here all carrier parameters, pixel features, query decoder and native predictions remain frozen. Only disposable per-reference oracle embedding vectors are fitted, using authorized TRAIN masks. No SAME6, TEST or protected review.

This test asks whether currently missed TRAIN references have an **explicit attainable mask in the frozen linear-readout family**. Finding one is positive conditional capacity evidence. Failure of this bounded optimizer is inconclusive about family capacity, RGB sufficiency or biological identity. It does not justify a new architecture by itself.

## Fixed cohort and carrier

Use seed42 completed mask-only Mask DINO. Verify the archived final checkpoint SHA against its existing status. Use the same official implementation and native mask-feature materialization as archived inference; FP32 forward, no image resize. Exactly three authorized eligible TRAIN windows, one per source: lexicographically first window ID with valid_fraction >= 0.95. Selection occurs from the inherited 854-window manifest before new predictions. Preserve every nonempty original independent target membership; report original source area and crop fragment area separately. Selected IDs and counts are frozen in SELECTED_TRAIN_WINDOWS.json (137/77/84 crop targets). Three windows are a bounded diagnostic, not geographic/statistical support.

## Mathematical object

For each reference target i with valid positive set P_i and valid nonmember set N_i, optimize only e_i in R^256. No extra bias or spatial coordinates. Bilinear mask-logit upsampling commutes with the linear readout; use the exact registered align_corners=False interpolation and zero-logit threshold. Independent reference memberships may overlap and are fitted independently, never projected to exclusive ownership.

Fix sampled pixel centers once, seed42: 4096 positive and 4096 negative points per target, sampled uniformly within each set (replacement only when needed). Invalid pixels are excluded. Outside one target means nonmembership in that published reference, not verified biological background. Empty positive/negative domains are logged and excluded from fitting, without removing them from count accounting.

Sampled objective:

    L_i(e) = mean_{x in sampled P_i} softplus(-e^T F_up(x))/2
           + mean_{x in sampled N_i} softplus( e^T F_up(x))/2.

This is a convex sampled logistic objective in e, not a surrogate proved to optimize mask IoU/AP. No regularizer or extra carrier loss. Unbounded separable logistic optima and finite optimizer convergence limit negative interpretations. Gradients reach e only; F and all original parameters are detached/frozen. Native query embeddings are read after the existing mask MLP, not decoder content vectors.

Two fixed initializations: zero and the final native query embedding with highest raw IoU to that TRAIN reference (GT-conditioned diagnostic start). Adam, lr=0.01, 300 updates per embedding, no scheduler; batch at most 16 references per start. No coefficient/width/threshold search. Report both starts and choose a per-target representative by lower final sampled TRAIN objective solely as an explicitly privileged diagnostic. This selection cannot enter inference.

## Measurements and outputs

1. Verify captured native embeddings dotted with F reproduce native low-resolution final logits within FP32 tolerance (max absolute <=1e-4 or relative <=1e-5); interpolation/grid-sampling checks must agree within these tolerances. If not, stop before fitting and fix an engineering mapping error once.
2. Best native raw IoU for each reference, all 300 final masks, independent of score/NMS.
3. Oracle mask IoU50/75 on the complete valid native-resolution target, including original-small and cropped-fragment distinction; gross new/lost qualifications for each start.
4. Actual gradient finiteness, first/final sampled loss, embedding norms, parameter-freeze checks, elapsed time/peak memory.
5. Save fitted embeddings, optimizer states, source/checkpoint/config hashes, full output RLE masks, reference-level tables and logs. Every oracle output is labeled GT_CONDITIONED_NOT_DEPLOYABLE. No AP, final deployable recall or inference improvement claim is made.

There are at most three carrier forwards, one per window. Head fitting is bounded at two starts x 300 updates per target. No carrier training, no validation tuning, no new localization/selection mechanism. GPU is shared and not reserved; stop cleanly on OOM and retain logs. Resource ceiling: 20 minutes wall time, 8 GiB allocated GPU memory; no automatic run expansion.

## Predeclared interpretations and stopping

- Qualifying optimized masks for native-raw-missing targets establish attainable conditional readout capacity for those targets. They do not show how RGB-only queries can retrieve those masks or avoid collateral losses.
- Lack of improvement is inconclusive among sampling/optimization/linear-readout/feature-resolution/reference alternatives. Do not label it RGB impossibility or feature insufficiency.
- This diagnostic never opens a decoder, embedding loss, refiner or selector by default. Record results and perform an outer-loop decision against prior art and the independent-reference constraints.
