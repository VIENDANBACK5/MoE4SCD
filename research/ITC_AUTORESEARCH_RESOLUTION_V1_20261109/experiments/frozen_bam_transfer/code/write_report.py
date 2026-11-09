from pathlib import Path
import csv,json,collections,statistics as st,html
EXP=Path(__file__).resolve().parents[1]
def rows(name):
    with (EXP/'results'/name).open() as f:return list(csv.DictReader(f))
def main():
    metrics=rows('METRICS.csv');paired=rows('RR_PAIRED_RESULTS.csv');same=rows('SAME_COUNT_RESULTS.csv')
    by={r['arm']:r for r in metrics};sm={(r['arm'],r['seed']):r for r in same}
    mean=lambda key:st.mean(float(r[key]) for r in paired)
    sc_recall=st.mean(float(sm[('RR',str(s))]['Recall50'])-float(sm[('CONTROL',str(s))]['Recall50']) for s in [42,43,44])
    sc_tp=st.mean(int(sm[('RR',str(s))]['TP50'])-int(sm[('CONTROL',str(s))]['TP50']) for s in [42,43,44])
    total_new=sum(int(r['new']) for r in paired);total_lost=sum(int(r['lost']) for r in paired)
    rec=rows('RECOVERY_AND_LOSS.csv');new_without_raw=sum(r['new']=='True' and r['control_raw50']=='False' for r in rec)
    patterns=collections.Counter(r['pattern_R5_M2F_DINO'] for r in rows('CROSS_CARRIER_PATTERNS.csv'))
    common_raw=sum(r['all_raw_missing']=='True' for r in rows('CROSS_CARRIER_PATTERNS.csv'))
    cost=rows('RUNTIME.csv');seconds=sum(float(r['seconds']) for r in cost);memory=max(float(r['peak_GPU_allocated_MiB']) for r in cost)
    matrix='| System | AP | AP75 | Recall50 % | TP50 | FP50 | Raw availability50 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in metrics:matrix+='| '+r['arm']+' | '+' | '.join([f"{float(r[k]):.3f}" for k in ['AP','AP75','Recall50']])+' | '+' | '.join(r[k] for k in ['TP50','FP50','raw50'])+' |\n'
    ps='| Seed | ΔAP | ΔRecall50 pp | New / lost | Net TP50 | ΔFP50 |\n|---|---:|---:|---:|---:|---:|\n'
    for r in paired:ps+=f"| {r['seed']} | {float(r['delta_AP']):+.3f} | {float(r['delta_Recall50_pp']):+.3f} | {r['new']} / {r['lost']} | {int(r['delta_TP50']):+d} | {int(r['delta_FP50']):+d} |\n"
    text=f'''# Frozen cross-carrier transfer on native BAM TRAIN imagery

**Decision: EXTERNAL_REFERENCE_FIDELITY_MEASURED_RR_TRANSFER_LIMITED_ORIGINAL_METHOD_ENTRY_NOT_ESTABLISHED.**

This is an executed exploratory empirical comparison. It is not a novel method, biological-identity validation, an independent multi-annotator study or an unqualified geographically independent final test. All numbers below are new measurements on this fixed cohort; historical SAME6 scores are deliberately not mixed into the performance table.

## Question and scientific scope

Does the established RR_MASK_INIT benefit survive outside the recorded SelvaMask task-training data, and what happens to final reference recovery across strong frozen RGB carriers? The study compares the same source reference occurrences under matched native views. Whole-system differences do not isolate architecture, pretraining or supervision.

The source API supports BAM TRAIN in its legacy-reference track. Actual GeoTIFF headers from all1439TRAIN records were inspected without RGB/model-score sampling. Fixed hash order and >=25m footprint separation selected24tiles in two geographic components near Bamberg, Germany. The exact cohort, controls, scores, tiling and evaluator were committed before predictions. The verified task-training inventories contain SelvaMask only, but legacy project exposure to BAM and exhaustive pretrained-data overlap remain unresolved. A new tile is not automatically a pristine independent site.

Delivered spacing is1.61–1.79cm/px, verified from metric EPSG25832 raster tags. All systems receive9native1024views per2048tile, with no resizing. Their physical view support is approximately16.5–18.4m; the inherited 30pixel floor represents a different physical area than at4.5cm. Forest, sensor, scale, annotation and source conditions change together. None of the following differences isolates GSD as a cause.

RGB and alpha alone enter inference. Original source masks/overlaps/splits are unchanged. All216image-arm prediction files (1944forwards) were frozen and hashed before scoring. No carrier training, score calibration, new selector, threshold search, TEST or protected review access occurred.

## Actual final-mask results

There are **1106nonempty alpha-valid published annotation occurrences**, not1106independent biological trees. AP uses0–100units and common maxDets[1,10,100,400]. Recall is separate all-accepted maximum one-to-one matching. R5 has its native returned RoI pool; query models expose their final query masks. Those raw pools are not architecture-equivalent.

{matrix}

R5 and the original M2F/DINO contextual comparisons use seed42. Matched continuation/RR use all three paired seeds. Do not select a favorable seed or compare one seed with a three-seed mean as an equal replication experiment. R5's higher Recall50 here comes with substantially more reference-relative false positives; it is not unconditional superiority.

## RR versus the matched continuation

{ps}

Mean paired differences: **Recall50{mean('delta_Recall50_pp'):+.3f}pp, AP{mean('delta_AP'):+.3f}, AP75{mean('delta_AP75'):+.3f}, netTP50{mean('delta_TP50'):+.3f}, FP50{mean('delta_FP50'):+.3f}** per seed. With the prespecified GT-free same-output-count control, the mean difference is **{sc_recall:+.3f}pp Recall50 / {sc_tp:+.3f}TP50**. This is a weak, budget-dependent composition signal, not a demonstrated practically large gain or statistical significance.

There are {total_new}new and {total_lost}lost matched reference–seed occurrences. Only **{new_without_raw}/{total_new}new matches lack a qualifying control final-decoder raw mask**; the other50already have one in that observed pool. New matches originate from8raw-absent,30filtered and20suppressed control cases. Lost matches comprise4RR-raw-absent,19filtered and20suppressed cases. No matching-membership ambiguity was found in these paired tables. The mean raw availability difference is+12references, but its seed-specific directions vary.

These observations separate candidate formation from final retention. They do not show that every gain is ranking-only: predictor geometry, scores and decoder context differ after matched training. Conversely, improved reference-wise raw coverage is not simultaneous deployment improvement.

## Original-source-small outcome is unavailable

COCO contains crop-level annotation IDs/masks without verified full-source crown identity/area. Therefore **original-source-small Recall50 and the +2pp small-crown gate are NOT_ESTIMABLE**. The488tile-mask-area references below9m² are a descriptive proxy, not a replacement. Their boundary-contact/alpha-clipping flags and proxy metrics are retained in the CSV but are not promoted into the primary small-crown claim. Absence of this estimand is not a measured negative result for small-crown generalization.

The existing evaluator's all-area AP/recall are reused unchanged. Its hard-coded4.5cm physical-size and2pixel boundary fields are withheld. Actual header pixel area is used only in explicitly named tile-mask proxies.

## Common reference failures and support

Seed42 R5/M2F/DINO have **{patterns['000']}/1106common final omissions** in this reference convention. **{common_raw}/1106** lack a qualifying raw mask in every registered observed pool. The patterns, in R5/M2F/DINO order, are `{dict(sorted(patterns.items()))}`. These are not strict all-seed/all-annotator stable-reference failures. The independently authored reliable-reference denominator is unavailable; R_common and R_all of the separate design cannot be estimated by pretending all1106are reference-stable.

A secondary support diagnostic was recorded before reference joins, after inference had begun. Each predicted mask is confined to one view V, so IoU(M,Y)<=|Y intersect V|/|Y|. The maximum over views is a privileged geometric upper bound, not a learned/deployable mask. Every reference has bound>=0.75; no actual raw IoU violates it. Thus **hard view-containment exclusion explains none of these IoU50/75 failures**. Context insufficiency, scale-dependent features and other causes remain possible. A permissive bound is not evidence that RGB identity or mask learnability is sufficient.

## Exploratory frame-edge diagnostic

The two fixed metadata-selected figures show outputs aligned with artificial view boundaries. After viewing them, one post-hoc descriptive proxy was defined and evaluated on all outputs, without altering predictions. For original DINO,3073/4768unmatched predictions touch an internal view edge; so do166/495currently matched predictions. Edge contact alone is not instance redundancy, and these numbers are not the net effect of deleting those masks: fallback predictions might change matching.

A strict reference-conditioned fragment proxy finds111DINO unmatched masks: they touch an internal edge, do not qualify any reference, mostly overlap a single reference crossing that view, cover10%-<50% of it and have a qualifying prediction from another view already retained. That subset does not classify the remaining edge-touching FPs. Neither the64.5%edge-contact fraction nor the2.3%strict-proxy fraction identifies the causal share of all errors. No edge filter, new NMS or fusion was deployed.

Generic cross-crop entity consistency already has direct prior art in [CropFormer / High Quality Entity Segmentation](https://openaccess.thecvf.com/content/ICCV2023/html/Qi_High_Quality_Entity_Segmentation_ICCV_2023_paper.html). Its accepted abstract identifies full-image/crop query association; a faithful implementation audit or ITC reproduction was not performed here. These figures do not justify renaming such a mechanism or opening a new architecture.

## Reference reliability and interpretation

The [source paper, Section3.5](https://elib.dlr.de/205658/1/remotesensing-16-01935-v2_opt.pdf), describes a forester-assisted labeling process using additional dates and field checking. That is not three prediction-blind independent annotations of the same single RGB input. Model disagreement with these polygons remains a valid benchmark measurement, but does not establish visually resolvable identity, biological correctness or information-theoretic RGB insufficiency. Published count/shape conventions and acquisition conditions differ from SAME6.

Two spatial groups and three computational seeds do not supply a forest-population confidence interval. The cohort is a fixed spatially separated sample, not a probability sample of all biological trees. Repeated views and seeds are not independent geography. Any future method designed using these observations must treat this cohort as exposed development evidence.

## Runtime, preservation and decision

Measured summed per-tile inference/encoding/NMS time across all nine systems is **{seconds:.3f}seconds**; this excludes checkpoint load/RGB I/O and is not total study wall time. Maximum recorded allocated GPU memory is **{memory:.3f}MiB**. GPU was shared, not reserved. Model and input bindings, all actual raw/final RLEs, hashes, logs, per-reference tables and figures are saved. No optimizer/model update was made in this experiment.

The frozen comparison is complete. It establishes limited transfer evidence and a different formation/retention signature under new conditions. It **does not validate a new robust ITC solution or establish original-method entry**. Preserve RR as the earlier source-bound coverage reference and retain the contextual carriers; do not choose a new threshold/seed or advertise deployment robustness from this cohort.

The independent-reference design remains EXECUTION_NOT_READY because its reference/independence prerequisites have not been met. The overall research goal stays active: neither an effective new mechanism nor a reliable general solution has been proved. This bounded negative/limited result is not a declaration that RGB ITC is scientifically exhausted.

Reproduction: see `REPRODUCIBILITY.md`; immutable inference freeze: `PREDICTIONS_FROZEN.json`; primary results: `results/METRICS.csv`, `RR_PAIRED_RESULTS.csv`, `RECOVERY_AND_LOSS.csv`, `SAME_COUNT_RESULTS.csv`; secondary/proxy analyses are explicitly labeled. No follow-on model is queued from this report.
'''
    (EXP/'RESEARCH_REPORT.md').write_text(text)
    decision=dict(status='EXTERNAL_REFERENCE_FIDELITY_MEASURED_RR_TRANSFER_LIMITED_ORIGINAL_METHOD_ENTRY_NOT_ESTABLISHED',empirical_execution='COMPLETE',original_method_supported=False,robust_solution_established=False,independent_reference_stability='NOT_ESTABLISHED',original_source_small_gate='NOT_ESTIMABLE',overall_goal_complete=False,images=24,references=1106,neural_forwards=1944,new_training=False,mean_RR_delta={k:mean(k) for k in ['delta_AP','delta_AP75','delta_Recall50_pp','delta_TP50','delta_FP50']},same_count_delta_recall_pp=sc_recall,new_matches=total_new,lost_matches=total_lost,new_matches_without_control_raw=new_without_raw,common_final_omissions=patterns['000'],common_raw_missing=common_raw,TEST=False,protected_review=False)
    (EXP/'SCIENTIFIC_DECISION.json').write_text(json.dumps(decision,indent=2)+'\n')
    htmltext='<!doctype html><html lang="en"><meta charset="utf-8"><title>ITC frozen transfer results</title><style>body{max-width:1250px;margin:2em auto;padding:0 1em;font:16px/1.6 system-ui}pre{white-space:pre-wrap}img{max-width:100%;height:auto}</style><h1>ITC autoresearch — executed frozen transfer</h1><img src="figures/FIXED_CROSS_CARRIER_COMPARISON.png" alt="Fixed same-RGB cross-carrier mask outlines; published references are not independently verified identity"><img src="figures/RR_NEW_AND_LOST.png" alt="Every paired seed shows both new and lost reference matches"><pre>'+html.escape(text)+'</pre></html>'
    (EXP/'REPORT.html').write_text(htmltext)
    print(json.dumps(decision))
if __name__=='__main__':main()
