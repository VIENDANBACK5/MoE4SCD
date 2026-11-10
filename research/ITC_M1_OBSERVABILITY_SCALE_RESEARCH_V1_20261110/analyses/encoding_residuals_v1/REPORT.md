# Remaining encoding sensitivity after the conventional learning control

**POST_HOC_RESIDUAL_CHARACTERIZED_ARCHITECTURAL_CAUSE_NOT_IDENTIFIED**

This finite analysis uses only retained predictions and reference masks from the completed study. No model, annotation or operating threshold changed. All 1,192 paired records and 2,384 best-witness RLE measurements were checked. Definitions were saved in protocol commit `9ec6f2c` before the new joint tables were measured; aggregate experiment results were already exposed, so this is explicitly post-hoc.

## The net gap is not the number of lost references

For A→B, compare native 1024 RGB with its lossless 2048 replication at the same physical field of view and canonical mask coordinates. Counts refer to the same 298 exposed TRAIN occurrences, not independent biological trees or sites.

| Raw-mask change A→B | S0 native continuation | S1 scale exposure |
|---|---:|---:|
| Qualifying references lost | 147 | 38 |
| Qualifying references gained | 0 | 2 |
| Net coverage change | −147 | −36 |
| Lost with native IoU in [0.50,0.55) | 0 | 0 |
| Lost with native IoU in [0.55,0.75) | 25 | 18 |
| Lost with native IoU ≥0.75 | 122 | 20 |
| Native IoU ≥0.75 and replicated IoU <0.25 | 70 | 12 |
| Lost with replicated IoU in [0.45,0.50) | 14 | 6 |

Thus 36 was a difference of availability counts, not 36 unique residual losses. S1 has 38 losses and 2 gains. Its native-qualified masks were not initially only marginal: none of these 38 had native IoU below 0.55. Six end close to the qualification threshold; twelve show a larger ≥0.75→<0.25 discrepancy.

There are 36 residual-loss reference IDs shared by S0 and S1, 111 only in S0 and 2 only in S1. This describes reference-level overlap between separately trained predictors; it is not same-query restoration or proof that a particular internal representation was preserved.

## Crop and source profile

Keep all 298 references in the main denominator. A separate near-whole proxy retains 225 occurrences with ≥90% of the published-tile reference RLE visible under common validity and source_truncated=False. It does not establish complete biological extent or annotation reliability.

Among S1's 38 raw losses, 28 satisfy this proxy; 16 of the 20 losses starting at IoU≥0.75 satisfy it. The robust losses therefore are not all explained by clipped reference fragments.

| Source window | S1 raw losses | Native IoU≥0.75 losses | Of those, near-whole proxy |
|---|---:|---:|---:|
| ZF2 | 19 | 10 | 8 |
| TBS | 7 | 4 | 4 |
| BCI | 12 | 6 | 4 |

One window per source and one paired computational seed cannot estimate systematic-error prevalence or independent geographic generalization. There are 28 source-small raw losses and 1 source-small raw gain under A→B; source-small refers to inherited published-tile annotation area, not independently verified tree size.

## Best-mask error profiles do not identify a decoder mechanism

For S1's 38 losses, independently selecting the best raw witness in each encoding gives:

- 21 cases with increased false foreground and increased missed-target pixels.
- 14 with increased false foreground without increased missed-target pixels.
- 3 with increased missed-target pixels without increased false foreground.

These are exact RLE pixel comparisons on the common domain. They do not trace one object query: the best query and selected encoder grid can differ between inputs. In particular, 35/38 having more false foreground does not prove a shared mask-expansion cause or justify a size penalty. Such a penalty could fragment or remove other masks; it was not tested.

## Two small native losses have different margins

The native-input S0→S1 treatment comparison loses exactly two small matches, both ZF2, and gains two non-small matches. None of these four changes has matching ambiguity under the inherited matcher.

| Published reference suffix | Change | S0 best raw IoU | S1 best raw IoU | Interpretation |
|---|---|---:|---:|---|
| ZF2 ann:201 | Small loss | 0.674605 | 0.499884 | Ends immediately below IoU50; no threshold change allowed |
| ZF2 ann:655 | Small loss | 0.686502 | 0.366873 | Larger geometry/availability degradation |
| ZF2 ann:205 | Non-small gain | 0.106566 | 0.791523 | New qualifying raw geometry |
| TBS ann:252 | Non-small gain | 0.712500 | 0.627273 | Existing raw qualification becomes accepted; no new raw qualification |

Both small losses are RAW_ABSENT in S1's observed final raw pool. This locates the proximal loss before final filtering, but does not distinguish changed encoder selection from query-conditioned decoding: the required intermediate trained-state masks are NOT_MEASURED. Do not infer an internal cause from the final best-witness masks.

## The other same-information pair remains asymmetric

C is a 512 mean-degraded RGB observation; D repeats its pixels to 1024. No additional scene information is introduced by C→D. S1 has 83 new / 10 lost raw qualifications, compared with S0's 63 new / 28 lost. S1's net difference of 73 therefore does not mean 73 losses: it is predominantly an advantage of the larger encoding. This counter-control reinforces that the treatment did not establish general scale invariance.

## Research decision

OBSERVED: conventional scale exposure/computation removes much of the constructed A/B discrepancy; a nontrivial, source-spanning residual remains, including strong native masks and near-whole reference occurrences. Some native collateral loss is near the evaluation boundary, some is larger.

DERIVED: neither threshold effects nor crop clipping alone explain the whole residual. Lossless input information is unchanged by replication; the trained inference map remains encoding-dependent. These statements do not imply equivariance is required by every deployment contract.

UNVERIFIED / NOT_MEASURED: which architectural or optimization constraint causes the residual; whether more conventional training would remove it; which RGB evidence supports independent identity; whether the phenomenon explains BAM; and whether an original observability intervention would improve native small-crown recovery without collateral loss.

No original-method entry is established. This completes the bounded saved-mask analysis; do not open another scale loss, query/decoder branch, threshold tuning or longer schedule from it. The original research objective remains unachieved. A future method requires a discriminating causal intervention and an advantage over appropriate established controls, not merely the existence of residual errors. The independent-reference dependency remains separate and unresolved; no human judgments were fabricated.

## Reproduction

From the repository root, the executed command was:

```bash
cs2_venv/bin/python research/ITC_M1_OBSERVABILITY_SCALE_RESEARCH_V1_20261110/analyses/encoding_residuals_v1/analyze.py
```

The completed-output guard prevents overwriting this analysis. A new sibling copy with the same repository depth can reproduce the frozen-input calculations. `INPUT_BINDINGS.json` records all source hashes; `COMPLETE.json` states the actual scope. No GPU or carrier import is required. See `PAIRED_REFERENCES.csv`, `SUMMARY.csv`, `SOURCE_PROFILE.csv`, `NATIVE_FINAL_CHANGES.csv` and `BEST_WITNESS_ERROR_PROFILES.csv` for complete records.
