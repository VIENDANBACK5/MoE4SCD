from pathlib import Path
import csv,json,hashlib,importlib.metadata
import markdown
EXP=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def rows(n):
    with (EXP/'results'/n).open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    m=rows('METRICS.csv');by={r['arm']:r for r in m};paired=rows('PAIRED_RESULTS.csv')
    v=read(EXP/'VERIFICATION.json');f=read(EXP/'PREDICTIONS_FROZEN.json');complete=read(EXP/'results/COMPLETE.json')
    assert v['raw_masks_checked']==3600 and complete['references']==298
    table=['| Arm | Input | Raw50 | Raw75 | Final TP50 | FP50 | AP | Small TP50/133 |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    names=dict(A='I,1024²',B='U2 I,2048²',C='D2 I,512²',D='U2 D2 I,1024²')
    for r in m:
        table.append(f"| {r['arm']} | {names[r['arm']]} | {r['raw50']} | {r['raw75']} | {r['TP50']} | {r['FP50']} | {float(r['AP']):.3f} | {r['source_small_TP50']} |")
    differences=['| Contrast | New/lost raw50 | New/lost final | Net final TP50 | New/lost small final |',
                 '|---|---:|---:|---:|---:|']
    for r in paired:
        differences.append(f"| {r['contrast']} | {r['new_raw']}/{r['lost_raw']} | {r['new_final']}/{r['lost_final']} | {int(r['new_final'])-int(r['lost_final']):+d} | {r['new_small_final']}/{r['lost_small_final']} |")
    report="""# Frozen M1 encoding pilot: actual results

**Decision: LOSSLESS_ENCODING_SENSITIVITY_OBSERVED_CAUSE_AND_ORIGINAL_METHOD_NOT_ESTABLISHED.**

This is a registered frozen TRAIN diagnostic, not a new model, a deployment improvement, independent generalization, or verified biological identity. No closed query/decoder/selection recipe is reopened.

## What executed

Original mask-only Mask DINO seed42, exactly300queries, three previously exposed TRAIN windows (one/source),298reference occurrences. Twelve actual forwards, no parameter update. Protocol/cohort/inference code committed in b5dc597 before predictions. Post-freeze scoring code committed in d0a6846 before reference measurements. No SAME6/TEST/protected review access. Original source annotation area is distinguished from cropped target fragment area.

A is native1024²RGB. B repeats each pixel into2×2, giving2048²without deleting information. C applies valid-normalized2×2block averaging, giving512². D exactly repeats C back to1024². All retain the same physical scene footprint. The RGB left inverses and normalized tensors actually entering the backbone are exact for A/B and C/D; no forced resize. Computation, encoder-site density and physical receptive field differ with grid size. Thus information is held equal within each pair, not every internal model condition.

All masks are materialized directly on the same1024²canonical grid before threshold, score/area acceptance and NMS. Common validity removes216/153/551original-valid pixels from the three windows; no reference becomes empty and all298IDs remain. Native A reproduces all900legacy query masks exactly after common clipping, with zero class-probability difference. The saved RLEs, input checks and NMS have been recomputed independently.

## Primary observed results

"""
    report+='\n'.join(table)
    report+="""

AP is0–100 on these three TRAIN windows only, with maxDets[1,10,100,400]; final matching uses all accepted outputs. The133source-small occurrences follow the inherited original published-tile raw-area definition, not uncropped biological crown size or the current1024crop area. None of these values is directly compared with full-tile SAME6/BAM AP.

"""
    report+='\n'.join(differences)
    report+="""

Both lossless pairs pass the preregistered finite relevance screen (at least5raw qualification losses in at least2sources); losses occur in all3sources. This is not statistical significance or method efficacy.

For A→B,165of180final losses lack a qualifying B raw mask,12fail acceptance and3fail suppression. Raw IoU75 availability also falls247→53, so the observation is not merely a tiny fluctuation around IoU0.5. B retains2new final matches, and180old matches disappear. There is no net recovery improvement.

For C→D,70final matches are new and27are lost; the net gain is43, including net32small matches(41new/9lost). Of the70gains,63lack a C qualifying raw mask,5previously fail acceptance and2suppression. Raw availability gains65andloses26. This is a known nearest-neighbor input re-encoding, not a learned original method; it has collateral losses.

At equal final counts, A/B each retain411outputs with276versus98TP50. C/D each retain308outputs with163versus194TP50. Thus those contrasts do not vanish at equal counts, although equal count does not isolate score versus geometry causality.

## What the intervention identifies

OBSERVED: the same RGB information can produce substantially different raw and final reference recovery in this frozen carrier when its raster encoding changes. Exact input checks exclude deleted RGB information as the explanation for the A/B and C/D differences. Most A/B losses already occur in the observed final raw pool.

OBSERVED: native1024encoding is much better than2048replication in this setup;1024replication of the coarse512observation is better overall than feeding512directly. This is a dependence on operating scale/encoding, not a theorem that greater resolution is worse.

DERIVED: under this exact synthetic replication, B can be mapped back to A by a known left inverse. A has already been executed. A novel observability module is therefore not necessary to recover the native operating point in this engineered case. A universal resizer for arbitrary genuinely high-resolution images is a different, untested deployment claim.

UNVERIFIED: the cause inside the network. Encoder sampling/access, learned object-size priors, physical receptive fields, decoder processing and optimization/training exposure remain alternatives. Holding RGB information equal does not hold all feature distributions equal.

Crucially, the completed carrier was fine-tuned on fixed1024native windows. B and C are out-of-operating-size interventions; an artificial distribution shift can degrade performance without proving an architectural defect. D has native dimensions but degraded image statistics. This pilot does not isolate a fundamental constraint, explain BAM's shift, establish visual identifiability or prove a deficiency requiring a new mechanism.

## Consequence for original-method research

M1 now has a concrete, large measured functional phenomenon on a strong carrier. The hypothesis that all changes require deleted input information fails for these exact lossless comparisons. The stronger hypothesis that observability-conditioned learning is the missing original mechanism is NOT_ESTABLISHED.

The necessary next discrimination is against ordinary input canonicalization and training-scale exposure. A known scale-treatment control could explain the phenomenon; novelty cannot come from adding consistency or restoring native dimensions. No new training is automatically opened here. First specify the one learning/architectural constraint that survives those alternatives and its closest prior control. Do not repeat the closed representation, query selection or decoder recipes under a scale label.

The current12-forward diagnostic is complete. The original goal is still active/unachieved. No new loss, module, threshold, seed search or V2 is queued by this report.

## Artifacts, engineering and cost

Actual RGB-only predictions and input arrays are saved. Full dense mask logits are NOT_STORED under the preregistered disk adaptation; all300canonical RLEs/class logits/scores, boxes, selected sites and grid dimensions are retained. Probability-field reanalysis would require an explicitly counted rerun.

The first analysis process wrote all results and COMPLETE.json, then failed only at printing a numpy.int64console value. Traceback remains in analysis.log; the serializer was corrected without rescoring or rerunning the model. Independent verification establishes output integrity, rather than relying on that marker.

"""
    report+=f"Measured sum of forward/tracing/materialization/NMS sections: {v['measured_sum_forward_materialization_seconds']:.3f}s; inference script total including setup/I/O: {f['seconds']:.3f}s. Neither is total research wall time. Peak allocated GPU {v['peak_GPU_allocated_MiB']:.1f}MiB on a shared RTX5880Ada. All12forwards/3600raw and1573final masks checked. Resource contracts were met.\n\n"
    report+="[Fixed-rule qualitative comparison](figures/FIXED_ENCODING_COMPARISON.png). Blue dashed is the published reference; orange is a GT-conditioned raw-best mask selected only after prediction freeze. The displayed crops are for diagnosis, not model inputs; reference fragments at window boundaries remain fragments. No independent human identity judgment is implied.\n\n"
    report+="[Protocol](protocol.md), [freeze](PREDICTIONS_FROZEN.json), [verification](VERIFICATION.json), [metrics](results/METRICS.csv), [source effects](results/SOURCE_RESULTS.csv), [per-reference changes](results/RECOVERY_AND_LOSS.csv), [reproduction](REPRODUCIBILITY.md).\n"
    (EXP/'REPORT.md').write_text(report)
    body=markdown.markdown(report,extensions=['tables','fenced_code'])
    (EXP/'REPORT.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>M1 frozen encoding diagnostic</title><style>body{max-width:1050px;margin:32px auto;padding:0 20px;font:17px/1.6 Georgia;color:#17212b}h1,h2{font-family:system-ui}table{border-collapse:collapse;font:14px system-ui;width:100%}td,th{padding:8px;border:1px solid #ccd3da}a{color:#175a90}</style>'+body+'</html>')
    (EXP/'SCIENTIFIC_DECISION.json').write_text(json.dumps(dict(
        status='LOSSLESS_ENCODING_SENSITIVITY_OBSERVED_CAUSE_AND_ORIGINAL_METHOD_NOT_ESTABLISHED',
        experiment_complete=True,forwards=12,new_training=False,original_method=False,goal_achieved=False,
        lossless_input_equivalence=True,raw_formation_difference_observed=True,
        artificial_operating_size_shift=True,training_vs_architecture_isolated=False,
        ordinary_canonicalization_is_prior_control=True,independent_generalization=False),indent=2)+'\n')
    env={k:importlib.metadata.version(k) for k in ['torch','torchvision','numpy','scipy','Pillow','pycocotools','matplotlib']}
    (EXP/'ENVIRONMENT.json').write_text(json.dumps(env,indent=2)+'\n')
    print(json.dumps(dict(report_written=True,actual_forwards=12,original_method=False)))
if __name__=='__main__':main()
