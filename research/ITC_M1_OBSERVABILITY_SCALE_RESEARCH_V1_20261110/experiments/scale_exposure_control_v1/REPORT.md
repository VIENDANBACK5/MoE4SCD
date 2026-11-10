# Conventional lossless-scale exposure: executed matched control

This is an established augmentation control on the completed Mask DINO, not an original method. The parent M1pilot showed lossless encoding sensitivity, but did not identify training versus architecture. These new runs keep original trainable scope, mask-only matching, losses,300queries and source supervision. No backbone/head freezing, new module or new loss.

## Executed contract

Same seed42parent,796eligible TRAIN windows after excluding three entire diagnostic tiles from ADDITIONAL updates, one full pass,796microbatches/398AdamW updates per arm, D4draws/targets/order/LR matched. The parent had already trained on those diagnostic tiles; this is not independent carrier validation or source holdout. S0 always sees native1024; S1 sees native1024on even draws and exact2×RGB/mask/validity replication on odd draws. Tiny/overlapping targets do not disappear, and no GT is used at inference.

Optimizer states reset identically; fixed LR1e-5nonbackbone/1e-6backbone after20-update warmup, weight decay.05 and clipnorm.01. Common standard activation checkpointing makes full-scope2048training feasible. One strictbf16pointwise gradient assertion failed initially; the record is retained. A preregistered engineering correction then verified fp32gradient parity(relativeL2≈3.9e-7) and measured bf16recompute rounding(relativeL2≈.00208). Scientific weights reload the original parent, never engineering-probe weights. No model-quality outcome informed that numerical correction.

Only optimizer updates and source-data exposure are matched. S1 processes2.5×as many raster pixels because half its inputs have4×area; FLOPs and wall time are not matched. Thus any benefit belongs to this conventional augmentation/computation treatment, not scale supervision isolated from compute or a novel mechanism.

## Actual final-mask outcomes

All24newRGB-only predictions were frozen before new metrics. A=native1024,B=lossless2048replication,C=512mean-degraded,D=replicatedC1024, with identical canonical mask coordinates, common validity and final selection.

| Training | Input | Raw50 | Raw75 | Final TP50 | FP50 | AP | Small TP50/133 |
|---|---|---:|---:|---:|---:|---:|---:|
| S0 | A | 279 | 248 | 276 | 163 | 62.601 | 118 |
| S0 | B | 132 | 51 | 114 | 329 | 6.396 | 41 |
| S0 | C | 177 | 93 | 159 | 153 | 19.268 | 42 |
| S0 | D | 212 | 137 | 205 | 206 | 31.430 | 78 |
| S1 | A | 277 | 240 | 276 | 222 | 61.641 | 116 |
| S1 | B | 241 | 163 | 236 | 325 | 38.573 | 88 |
| S1 | C | 172 | 91 | 160 | 169 | 20.688 | 43 |
| S1 | D | 245 | 175 | 239 | 231 | 41.392 | 96 |

S1minusS0gross changes on identical reference IDs:

| Input | New final | Lost final | Net TP50 | New/lost small | New/lost raw50 |
|---|---:|---:|---:|---:|---:|
| A | 2 | 2 | +0 | 0/2 | 1/3 |
| B | 123 | 1 | +122 | 47/0 | 111/2 |
| C | 13 | 12 | +1 | 6/5 | 9/14 |
| D | 39 | 5 | +34 | 20/2 | 39/6 |

Registered relevance screen: S0A/Braw50gap=147; S1gap=36; S1nativeAfinalTP50delta=0; screen_pass=True. This narrow screen is not statistical significance, universal sufficiency, deployment improvement or original-method entry. If S0gap<=0, it is inapplicable rather than a failed scale hypothesis.

## Scientific decision from these actual results

The registered narrow screen passes: A/Braw50gap shrinks by75.51%. On replicated inputB, finalTP50 rises114→236(123new/1lost); 107of the123new final matches had no qualifying final raw mask in S0B. At equal443outputs,TP50 rises114→233. Thus the effect is not only accepting more predictions. BfinalTP50 improves on each diagnostic source(ZF2+56,TBS+40,BCI+26), with only one crop per source.

NativeAis not a practical improvement: TP50remains276(2new/2lost), but smallTP50falls118→116, APchanges-0.960, AP75changes-2.673, and FP50rises163→222(+36.20%). Both lost references are source-small; both new references are outside that stratum. NativeAPdeclines on all three diagnostic sources. At equal439nativeoutputs, S1has275TPversus276in S0. The narrow gate intentionally does not certify acceptableAP/FP; its pass cannot override these regressions.

The treatment is pair-specific, not demonstrated general scale invariance. On the untrained C/Dpair, raw50discrepancy increases35→73 and finalTP50discrepancy46→79. S1Bstill trails its own nativeAby36raw qualifications and40final matches. These are counterexamples to a claim that the augmentation completely solves raster dependence.

Decision: **ORDINARY_SCALE_EXPOSURE_REDUCES_ENCODING_GAP_WITH_NATIVE_TRADEOFF_ORIGINAL_METHOD_NOT_ESTABLISHED**. Training-scale exposure with additional raster computation can substantially change the frozen pilot phenomenon. Therefore that pilot cannot establish the necessity of an original observability-conditioned mechanism. This does not isolate exposure from extra compute, certify an architectural cause, explain BAM, or refute every possible M1mechanism. The tested one-pass recipe is closed; S1is a diagnostic checkpoint, not a replacement release for RR_MASK_INIT/nativeDINO. No extra epochs, threshold rescue, V2or new loss is queued.

All values refer to298TRAIN reference occurrences/133source-small annotation occurrences, not biological trees or independent sites. AP is0–100 on these fixed crops, never directly compared with full-tile SAME6/BAM. Source-small uses the inherited original published-tile raw annotation area; crop fragments and source truncation remain explicit.

See SOURCE_RESULTS.csv and SAME_COUNT.csv for source dependence and GT-free equal-output-count controls. These controls do not turn gains into pure geometry causality or remove loss of previously correct instances. No favorable threshold, seed, epoch or branch is selected after scoring.

## Interpretation boundary

The experiment tests whether an ordinary scale-exposure treatment changes the phenomenon. A successful known treatment argues against using this engineered fixed-size failure as evidence that an original observability architecture is required. A failed finite one-pass treatment does not prove ordinary scale learning, all ScaleMatch/DINO-MS variants or the architecture incapable.

Training input dimensions and apparent object pixel size change jointly under replication. Matching updates/data does not isolate every optimization pathway. The tiny exposed diagnostic cohort and single paired seed cannot establish independent geographic efficacy or reliable biological identities. A native-loss/FP trade-off must not be hidden by the reduction in A/Bgap.

The original method-development goal remains unachieved. No new module, loss, threshold sweep or automatic longer schedule is justified solely by these results.

## Engineering / runtime / artifacts

- S0:398committed updates,796microbatches,295.5s,peak allocatedGPU4173.2MiB; full model/optimizer/RNG state saved. First/last20update mean losses 27.060/26.051; these are optimization traces, not segmentation proof.
- S1:398committed updates,796microbatches,571.3s,peak allocatedGPU12832.9MiB; full model/optimizer/RNG state saved. First/last20update mean losses 65.789/38.632; these are optimization traces, not segmentation proof.

Verified 7200raw/3463final RLE masks, source/RNG schedule, full trainable-state changes and unchanged original checkpoint. Both optimizer states are retained; final.pt hard-links the completed last state to avoid duplicate storage. GPU shared, not reserved. No TEST/protected review/SAME6evaluation.

[Protocol](protocol.md), [numerical correction](ENGINEERING_PARITY_ADDENDUM.md), [full-mask results](results/METRICS.csv), [new/lost references](results/RECOVERY_AND_LOSS.csv), [same-count](results/SAME_COUNT.csv), [source effects](results/SOURCE_RESULTS.csv), [verification](VERIFICATION.json), [fixed qualitative figure](figures/FIXED_CONTROL_COMPARISON.png).
