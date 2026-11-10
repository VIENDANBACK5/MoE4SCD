from common import *
import numpy as np,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycocotools import mask as mu
import markdown
from matplotlib.backends.backend_pdf import PdfPages
import textwrap
def rle(r):return dict(size=r['size'],counts=r['counts'].encode())
def main():
    ver=read(EXP/'VERIFICATION.json');decision=read(EXP/'results/DECISION.json')
    metrics=rows(EXP/'results/METRICS.csv');changes=rows(EXP/'results/CHANGE_SUMMARY.csv');same=rows(EXP/'results/SAME_COUNT.csv')
    subset=rows(EXP/'results/S1_STRONG_RESIDUAL_SUBSET.csv');cohort=read(PILOT/'COHORT.json')
    gt=gz(CONTROL/'evaluations/S0/results/REFERENCE_MASKS.json.gz')
    per=rows(EXP/'results/PER_REFERENCE.csv');ind={(r['training_arm'],r['readout'],r['reference_id']):r for r in per}
    # Fixed rule added before figure generation: first lexical pre-existing strong residual per source.
    chosen=[sorted([r for r in subset if r['source']==s],key=lambda r:r['reference_id'])[0] for s in sorted({r['source'] for r in subset})]
    (EXP/'figures').mkdir(exist_ok=True);manifest=[]
    with plt.rc_context({'font.size':9,'axes.titlesize':9}):
        fig,axs=plt.subplots(3,5,figsize=(14,9),layout='constrained')
        for row,z in enumerate(chosen):
            meta=ind['S1','AA',z['reference_id']];i=next(i for i,r in enumerate(cohort) if r['id']==meta['window_id'])
            j=next(j for j,m in enumerate(gt['metadata'][i]) if m['reference_id']==z['reference_id'])
            yr=rle(gt['references'][i][j]['rle']);y=mu.decode(yr).astype(bool)
            rgb=np.load(PILOT/f"{cohort[i]['id']}_inputs.npz")['RGB'];x0,y0,w,h=mu.toBbox(yr)
            pad=60;x1=min(1024,int(x0+w)+pad);y1=min(1024,int(y0+h)+pad);x0=max(0,int(x0)-pad);y0=max(0,int(y0)-pad)
            witnesses={}
            for col in range(5):
                ax=axs[row,col];ax.imshow(rgb,interpolation='nearest');ax.contour(y,levels=[.5],colors=['#0072B2'],linestyles='dashed',linewidths=1.3)
                ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.set_xticks([]);ax.set_yticks([])
                if col==0:ax.set_title(f"Source {row+1}, fixed residual\nPublished reference\nIdentity not independently verified")
                else:
                    code=['AA','AB','BA','BB'][col-1];d=gz(EXP/f"predictions/S1/{cohort[i]['id']}_{code}.json.gz")
                    a=mu.iou([rle(p['rle']) for p in d['raw_predictions']],[yr],[0])[:,0];q=int(a.argmax());mask=mu.decode(rle(d['raw_predictions'][q]['rle']))
                    if mask.any():ax.contour(mask,levels=[.5],colors=['#D55E00'],linewidths=1.1)
                    ax.set_title(f"{code}: E_{code[0]} × F_{code[1]}\nraw-best q{q}, IoU {a[q]:.3f}")
                    witnesses[code]=dict(query=q,IoU=float(a[q]))
            manifest.append(dict(reference_id=z['reference_id'],selection='First lexical pre-existing strong S1 residual per source; not selected on interchange outcome',
                    crop_xyxy=[x0,y0,x1,y1],raw_best_witnesses=witnesses))
        fig.suptitle('Frozen readout-factor interchange: same RGB support\nBlue dashed = reference; orange = GT-conditioned raw-best witness after freeze',fontsize=12)
        fig.savefig(EXP/'figures/FIXED_RESIDUALS.png',dpi=160,bbox_inches='tight');plt.close(fig)
    write(EXP/'figures/MANIFEST.json',dict(selection=manifest,GT_conditioned_diagnostic=True,GT_inference=False))
    table=['| Model | Readout E/F | Raw50 | Raw75 | Final TP50 | FP50 | AP | AP75 | Small TP50/133 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in metrics:table.append(f"| {r['training_arm']} | {r['readout']} | {r['raw50']} | {r['raw75']} | {r['TP50']} | {r['FP50']} | {float(r['AP']):.3f} | {float(r['AP75']):.3f} | {r['source_small_TP50']} |")
    ch=['| Model | Contrast | New/lost raw50 | New/lost final50 | New/lost small |','|---|---|---:|---:|---:|']
    for r in changes:ch.append(f"| {r['training_arm']} | {r['contrast']} | {r['new_raw']}/{r['lost_raw']} | {r['new_final']}/{r['lost_final']} | {r['new_small']}/{r['lost_small']} |")
    freeze=read(EXP/'PREDICTIONS_FROZEN.json')
    report='''# Frozen readout-factor interchange: executed diagnostic

**READOUT_INTERCHANGE_NOT_A_USEFUL_RECOVERY_POLICY_CAUSE_NOT_IDENTIFIED**

The user resumed autonomous research. The previous external-reference block was too broad as a stop for all scientific work: it blocks the independent-reference study, but not a frozen published-reference inference diagnostic. This experiment does not obtain independent annotators or resolve biological identity. The original-method objective remains unachieved.

## Actual experiment

Two completed seed42 Mask DINO checkpoints: S0 matched native continuation and S1 conventional scale exposure. Neither is retrained here. Same three previously exposed TRAIN windows, 298 reference occurrences / 133 source-small, exact RGB/validity and native300-query budget. A is native1024; B is exact pixel replication to2048 over the same physical support. Query/site IDs follow their own embedding source; they are not aligned across encoder grids.

The factual logits are E_A F_A and E_B F_B. Passive hooks capture the final mask embeddings E and pixel feature field F. AB uses E_A with F_B; BA uses E_B with F_A. These are final-readout interventions, not query insertion or decoder restoration. Class logits and box/site descriptors follow E. All independent masks use the same canonical materialization, score, area and NMS rules.

Twelve actual frozen carrier forwards produced24pools /7200raw RLE masks, frozen before new reference joins. Native factor products exactly reproduce model logits. AA/BB raw masks, scores, accepted/final IDs exactly reproduce the prior experiment. A paired sign change of E and F preserves native logits exactly, implementing the mathematical gauge control. No optimizer update, TEST/protected review/SAME6evaluation or new annotation occurred.

## Actual final-mask outcomes

AP values are0–100 on these fixed crops, not comparable with full-tile SAME6/BAM. Raw availability, simultaneous matching and score-ranked AP are distinct; see full CSVs for all fields.

'''
    report+='\n'.join(table)+'\n\nGross changes (all reference IDs):\n\n'+'\n'.join(ch)
    report+=f"\n\n## Registered strong-residual screen\n\nThe previously observed20S1references have AArawIoU≥.75 and BBrawIoU<.50. ABrecovers{decision['AB_recovered']}/20; BArecovers{decision['BA_recovered']}/20. Registered operational pattern: **{decision['operational_pattern']}**. No dominant factor screen passes. This exposed subset is not an independent sample, and these counts refer to raw availability, not deployable TP gains.\n"
    report+='''
In S1, BA adds9raw qualifications but loses121 relative to BB; it adds11final matches but loses122. AB adds3raw /4final matches but loses153raw /152final. This directly fails the requirement to recover instances without losing many others. The swaps are not useful inference policies and will not be released as improvements.

The loss occurs before final scoring for many references: raw availability itself collapses. Keeping original factual query scores does not restore the missing raw masks. SAME_COUNT.csv reports all fixed count controls; these comparisons do not hide gross losses or convert two-encoding interventions into efficient single-pass inference.

At matched output counts in S1: BB→AB changes206→88TP at269outputs; BB→BA changes226→125TP at437outputs. Thus the failure cannot be explained only by lower prediction counts.

## Exploratory same-query check

After the primary screen was known, a separately documented saved-mask analysis compared each crossed raw-best witness with the exact same source query in its factual pool. No forward or policy changed. In BA's7qualifying strong-residual witnesses, all7same-query factual BBmasks are belowIoU50; only1of7source-query boxes has boxIoU50. Six of the7crossed witnesses survive final suppression; survival alone is not one-to-one matching. In AB's2witnesses,1same-query factual AAmask and1source-query box qualify atIoU50.

The conditional existence of a usable Bembedding under F_Adoes not prove that this query already described the same localized crown under native B. The readout's geometric/identity association can change when its field changes; a poor box also is not proof that a mask cannot qualify. These observations prevent calling the7cases evidence that query localization was already solved. See EXPLORATORY_SAME_QUERY_NOTE.md and results/EXPLORATORY_SAME_QUERY.csv; this supplementary analysis does not alter the registered decision.

## Interpretation and mathematical limit

OBSERVED: both native E/F pairs work far better than their interchanged counterparts. Some individual masks can be produced by a crossed pair, but aggregate geometry and final recovery regress substantially in both frozen checkpoints.

DERIVED: a good embedding from one encoding is not automatically compatible with the field from another. Neither raw-mask rescue on a subset nor poor mixed-state performance identifies an independently deficient component. For any invertible channel transform C, E_B C^(-1) and C F_B leave native BB unchanged while generally changing both swaps. With C=−I, both cross logits reverse sign. This valid established bilinear property is a counterexample to treating interchange failure as a negative feature-capacity certificate; it is not a new theorem or segmentation mechanism.

UNVERIFIED: which trained upstream operation causes the native A/B discrepancy, whether a channel basis change or a different type of representation change explains mixed-state incompatibility, whether an inference-valid single-carrier intervention helps, and any independently justified novelty. No Procrustes alignment, new consistency loss, decoder, embedding head or longer training is automatically opened.

The methodology overlaps established activation/interchange interventions. [Mask DINO](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Mask_DINO_Towards_a_Unified_Transformer-Based_Framework_for_Object_Detection_CVPR_2023_paper.html) supplies the actual bilinear readout. [Zhang and Nanda, ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html) studies intervention-method/metric sensitivity in language models; its results are not claimed as crown evidence or a reproduced vision baseline. Mechanistic localization remains conditional on the intervention and its potentially untrained hybrid states.

## Scientific decision and limits

Close this fixed interchange recipe as an unsuccessful recovery policy and an inconclusive decomposition of the native failure cause. Retain the strong factual carriers and conventional RR baseline. This does not establish RGB insufficiency, a frozen-feature ceiling, a universal decoder limitation or exhaustion of the ITC problem. It also does not justify a new method by renaming feature/query alignment.

One paired seed, one crop per three sources, parent-seen TRAIN annotations and an artificially re-encoded input. No independent geographic/reference-stability evidence. Source-small uses inherited original published-tile annotation area; crop fragments and overlapping memberships are preserved. Displayed best masks are chosen with GT only after predictions are frozen, for diagnostics, never inference.

## Reproduction / artifacts

'''
    report+=f"Actual inference/extraction/readout/parity-check walltime {freeze['seconds']:.3f}s; peak allocated GPU {freeze['peak_GPU_MiB']:.1f}MiB; peak hostRSS {freeze['peak_RSS_GiB']:.3f}GiB. This includes diagnostic hashing/parity overhead and is not single-pass deployment latency. Verification checked {ver['raw_masks']}raw and{ver['final_masks']}final masks, both score policies, validity, provenance, matching, the same20-reference subset and unchanged checkpoints.\n"
    report+='''
Commands from repository root (executed in sequence):

```bash
cs2_venv/bin/python research/ITC_READOUT_FACTOR_INTERCHANGE_V1_20261110/code/predict.py
cs2_venv/bin/python research/ITC_READOUT_FACTOR_INTERCHANGE_V1_20261110/code/analyze.py
cs2_venv/bin/python research/ITC_READOUT_FACTOR_INTERCHANGE_V1_20261110/code/verify.py
cs2_venv/bin/python research/ITC_READOUT_FACTOR_INTERCHANGE_V1_20261110/code/same_query.py
cs2_venv/bin/python research/ITC_READOUT_FACTOR_INTERCHANGE_V1_20261110/code/report.py
```

Prediction/analysis completion guards refuse overwriting completed outputs. For reproduction, use a fresh sibling directory preserving relative repository depth and bound asset access; do not delete seals. Full feature fields/dense logits are NOT_STORED under the preregistered disk contract. Final E matrices, all independent RLEs, query/site/class/box provenance, latent/input/checkpoint/code hashes and logs are retained. No trained checkpoint is claimed because none was trained here.

[Protocol](protocol.md), [math and counterexample](MATHEMATICAL_SPECIFICATION.md), [complete metrics](results/METRICS.csv), [individual new/lost](results/RECOVERY_AND_LOSS.csv), [same-count](results/SAME_COUNT.csv), [original-score control](results/ORIGINAL_SCORE_CONTROL.csv), [source](results/SOURCE_RESULTS.csv), [size](results/SIZE_RESULTS.csv), [verification](VERIFICATION.json), [fixed diagnostic figure](figures/FIXED_RESIDUALS.png).
'''
    (EXP/'REPORT.md').write_text(report)
    (EXP/'REPORT.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Readout-factor interchange</title><style>body{max-width:1200px;margin:30px auto;padding:20px;font:17px/1.6 Georgia}table{font:14px system-ui;border-collapse:collapse;width:100%}th,td{border:1px solid #ccd3da;padding:7px}img{max-width:100%}</style>'+markdown.markdown(report,extensions=['tables'])+'<p><img src="figures/FIXED_RESIDUALS.png" alt="Fixed diagnostic masks"></p></html>')
    with PdfPages(EXP/'PROGRESS_REPORT.pdf') as pdf:
        fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off')
        ax.text(.02,.97,'Readout-factor interchange: actual research outcome',fontsize=18,weight='bold',va='top')
        ax.text(.02,.87,textwrap.fill('Question: when lossless RGB re-encoding changes a mask, can its final query embedding or pixel feature field be exchanged separately to restore useful instance formation?',105),fontsize=12,va='top')
        ax.text(.02,.73,'Frozen S1; 298 exposed TRAIN references, 133 source-small; 300 masks per pool',fontsize=11,va='top')
        rrs=[r for r in metrics if r['training_arm']=='S1']
        data=[[r['readout'],r['raw50'],r['TP50'],r['FP50'],f"{float(r['AP']):.3f}",r['source_small_TP50']] for r in rrs]
        tab=ax.table(cellText=data,colLabels=['E/F','Raw50','Final TP50','FP50','AP','Small TP50'],bbox=[.02,.38,.96,.29],cellLoc='center');tab.auto_set_font_size(False);tab.set_fontsize(11)
        ax.text(.02,.29,textwrap.fill('Strong residual screen: AB rescues 2/20, BA rescues 7/20. Neither directional screen passes. BA adds 11 final matches but loses 122; at equal 437 outputs, TP falls 226 to 125.',105),fontsize=12,va='top')
        ax.text(.02,.13,textwrap.fill('Decision: no useful recovery policy or original method established. A failed hybrid does not identify a deficient feature/query component. No new training, threshold tuning or protected access.',105),fontsize=12,va='top')
        pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
        fig,ax=plt.subplots(figsize=(14,9.5));ax.imshow(plt.imread(EXP/'figures/FIXED_RESIDUALS.png'));ax.axis('off')
        pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
        fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off')
        paragraphs=[('Interpretation boundary','Native logits are E F. For invertible C, replacing a pair by E C^-1 and C F preserves its native mask while changing crossed readouts. This established bilinear property prevents treating hybrid failure as a negative feature-capacity certificate.'),
                    ('Actual verification','Twelve frozen forwards; 24 pools; 7,200 raw and 2,996 final independent RLE masks checked. Native logits, masks, scores and final IDs reproduce exactly. Class/box/site provenance, both score policies, validity, NMS and matching verified.'),
                    ('Same-query exploration','Among seven BA raw rescues, none of the same-query native BB masks qualifies; only one source-query box reaches box IoU50. Six crossed witnesses survive suppression, which is not itself one-to-one recovery. This is a post-hoc diagnostic, not GT-based inference.'),
                    ('Scope','One paired seed, three parent-seen TRAIN crops. Artificial encoding change, possible untrained hybrid states, no independent geographic/reference validation. Biological identity and an architecture-level cause remain unestablished. Independent annotator availability blocks only that separate study, not every scientific diagnostic.')]
        yy=.96
        for title,body in paragraphs:
            ax.text(.02,yy,title,fontsize=14,weight='bold',va='top');ax.text(.02,yy-.055,textwrap.fill(body,108),fontsize=11,va='top');yy-=.23
        pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
    print(json.dumps(dict(report=True,pattern=decision['operational_pattern'],original_method=False)))
if __name__=='__main__':main()
