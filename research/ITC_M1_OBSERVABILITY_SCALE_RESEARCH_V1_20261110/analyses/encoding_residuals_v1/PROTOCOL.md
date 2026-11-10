# Post-hoc residual analysis of the completed scale-exposure control

This is one finite analysis of saved predictions, not another training/inference experiment or a search for a favorable operating point. The aggregate outcomes are already exposed. These definitions are recorded before computing the new joint residual tables.

Question: does the remaining encoding difference consist only of near-IoU50 changes or clipped reference fragments, or are there larger raw-mask disagreements after conventional augmentation? The answer determines the strength of the residual phenomenon; it cannot alone identify an architectural cause or justify a new method.

Inputs: sealed S0/S1 PER_REFERENCE tables, reference RLEs, saved raw predictions and the three diagnostic TRAIN source reference files. No new RGB loading, model forward, optimizer update, SAME6, TEST, reviewer packet or label acquisition. Preserve the completed experiment and all historical decisions.

Analyze both information-equivalent pairs A→B and C→D, in both training arms, on all 298 matching reference IDs: 1,192 paired records. Also list all native-A S0→S1 new/lost final matches. Never equate query IDs across distinct encoder grids or model states.

Keep original qualification/matching thresholds .50/.75. Define raw loss: left best IoU≥.50 and right best IoU<.50. Report these disjoint bands by left best IoU: [.50,.55), [.55,.75), [.75,1]. Also report right IoU [.45,.50) among losses, and severe discordance left≥.75/right<.25. These are descriptive bands, not confidence intervals, tuning thresholds or a replacement evaluation policy.

Quantify field-of-view fragments using common-crop target pixels / original published-tile reference RLE pixels. A ≥.90 ratio and source_truncated=False is a near-whole-source-reference proxy, not proof of complete biological extent, label reliability or RGB sufficiency. Do not exclude fragments from the primary 298-reference denominator. Report the proxy subgroup separately.

For each pair's independently chosen raw-best witness, compute target-intersection, false foreground and missed-target pixels from exact RLEs. Confirm IoU against the saved table. Describe whether false foreground / target misses increase. These are differences between best available masks, not a same-query trajectory or a causal explanation of decoder behavior.

Report gross new/lost raw qualifications, final one-to-one matches, margins, size/source/fragment profiles, retained versus shared residual IDs across S0/S1 and all individual native-A gains/losses. Do not call a difference of coverage counts the number of lost references.

Decision boundaries: a robust residual supports only persistent functional dependence on encoding. Even many such cases do not prove a particular architecture/learning bottleneck; the training treatment already changed exposure and computation jointly. Do not launch another scale loss, longer training, selector, decoder or V2 from this analysis. If the artifacts lack stage information, mark it NOT_MEASURED rather than infer a cause.
