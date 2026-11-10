# Fixed-checkpoint readout-factor interchange

## Actual representation

The verified Mask DINO code ends with `outputs_mask = einsum('bqc,bchw->bqhw', mask_embed, mask_features)`. Let E_a be the final 300×256 mask-embedding matrix and F_a the 256×h_a×w_a pixel feature field for encoding a. A is native1024 RGB; B is its exact2×replication at2048. The physical support and original RGB information are identical. Neither observation uses a reference.

Define four native-grid logit fields, L_ab=E_a F_b, a,b∈{A,B}. Each field uses exactly300 mask hypotheses. Upsample logits from F_b's grid to the common1024 coordinates with the inherited bilinear align_corners=False operation, threshold at0 and clip to the same common validity mask. Independent mask overlap is preserved. Class logits, encoder site identity and box descriptors always follow E_a. No query IDs are equated across A and B.

AA and BB are factual model outputs. AB and BA intervene only at the final mask readout; no decoder is rerun with inserted queries, new boxes or a changed query count. They are diagnostic two-encoding computations, not a claimed efficient single-pass system, oracle query initializer or R5/DINO ensemble.

## Spatial mapping identity

For fixed E and a linear spatial interpolation P acting identically on each channel, P(EF)=E P(F): each output coefficient is a finite sum, so exchanging the channel sum and spatial interpolation sum leaves the result unchanged in real arithmetic. This justifies common-coordinate comparisons; it does not guarantee floating-point identity when operation order changes. Implementation keeps the original dot-product-then-interpolation order and checks native reconstruction exactly.

## Identifiability limitation and counterexample

For any invertible channel matrix C, replacing E_B by E_B C^(-1) and F_B by C F_B leaves BB unchanged. AB and BA generally change. In particular C=−I leaves BB identical while negating both cross logits. Thus native outputs do not identify a unique cross-encoding factorization. A bad cross-readout is not a proof of insufficient pixel-feature capacity or an architecture bottleneck. This is an established bilinear reparameterization property, not an original theorem.

The actual checkpoint fixes one implementation basis, so a successful interchange does establish readout compatibility in that fixed implementation. It does not establish biological identity, a minimal representation, a necessary/sufficient learning cause or robustness to all equivalent channel reparameterizations.

## Falsifiable operational questions

Twenty S1 reference occurrences previously have native best raw IoU≥.75 and replicated best raw IoU<.50. This is an exposed diagnostic subset, not an independent sample. All predictions are generated for all298references without consulting this subset.

If AB recovers≥10of these20 while BA recovers≤5, label the operational pattern `A_EMBEDDINGS_COMPATIBLE_WITH_B_FIELD`; the native B query/readout branch is a useful next place to discriminate, but selection versus decoder processing remains unresolved.

If BA recovers≥10 while AB recovers≤5, label `B_EMBEDDINGS_COMPATIBLE_WITH_A_FIELD`; the native B embeddings retain useful access under a different field, without a feature-capacity or field-only causal claim.

If both recover≥10, label `MULTIPLE_COMPENSATING_READOUTS`; if neither directional screen holds, label `NO_DOMINANT_INTERCHANGE_PATTERN`. These are practical diagnostic screens, not significance tests or original-method entry gates. Also report full-cohort new/lost masks so subset rescue cannot hide collateral losses.

Any original mechanism still requires a specific constraint distinct from prior art, a justified intervention, matched training and net actual-mask improvement. No new model, longer scale training or loss is launched automatically from this diagnostic.
