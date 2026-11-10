from shared import *
import csv,gzip,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycocotools import mask as mu
import markdown
PILOT=EXP.parent/'frozen_encoding_v1'
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def gz(p):
    with gzip.open(p,'rt') as f:return json.load(f)
def rle(r):return dict(size=r['size'],counts=r['counts'].encode())
def main():
    ver=read(EXP/'VERIFICATION.json');decision=read(EXP/'results/DECISION.json')
    metrics=rows(EXP/'results/METRICS.csv');changes=rows(EXP/'results/CHANGE_SUMMARY.csv')
    checks=read(PILOT/'figures/MANIFEST.json')['selection'];cohort=read(PILOT/'COHORT.json')
    gt=gz(EXP/'evaluations/S0/results/REFERENCE_MASKS.json.gz')
    with plt.rc_context({'font.size':9,'axes.titlesize':10}):
        fig,axs=plt.subplots(3,5,figsize=(14,8.5),layout='constrained')
        figmanifest=[]
        for row,z in enumerate(checks):
            i,j=z['window'],z['reference_index'];rgb=np.load(PILOT/f"{cohort[i]['id']}_inputs.npz")['RGB']
            y=mu.decode(rle(gt['references'][i][j]['rle'])).astype(bool)
            x0,y0,x1,y1=z['display_crop_xyxy']
            for col in range(5):
                ax=axs[row,col];ax.imshow(rgb,interpolation='nearest')
                ax.contour(y,levels=[.5],colors=['#0072B2'],linestyles='dashed',linewidths=1.4)
                ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.set_xticks([]);ax.set_yticks([])
                if col==0:ax.set_title(f"Fixed parent-pilot case, source{row+1}\nPublished reference\nIdentity not independently verified",fontsize=9)
                else:
                    arm,obs=[('S0','A'),('S0','B'),('S1','A'),('S1','B')][col-1]
                    p=gz(EXP/'evaluations'/arm/'predictions'/f"{cohort[i]['id']}_{obs}.json.gz")
                    a=mu.iou([rle(q['rle']) for q in p['raw_predictions']],[rle(gt['references'][i][j]['rle'])],[0])[:,0]
                    q=int(a.argmax());mask=mu.decode(rle(p['raw_predictions'][q]['rle']))
                    if mask.any():ax.contour(mask,levels=[.5],colors=['#D55E00'],linewidths=1.1)
                    ax.set_title(f"{arm}/{obs}: raw-best q{q}\nIoU{a[q]:.3f}")
            figmanifest.append(dict(reference_id=z['reference_id'],crop=z['display_crop_xyxy'],selection='Same fixed case as parent pilot, not selected on control outcome'))
        fig.suptitle('Ordinary scale exposure: fixed parent-pilot failures\nBlue dashed=reference; orange=GT-conditioned raw-best witness, not deployed mask choice',fontsize=12)
        (EXP/'figures').mkdir(exist_ok=True);fig.savefig(EXP/'figures/FIXED_CONTROL_COMPARISON.png',dpi=160,bbox_inches='tight');plt.close(fig)
    write(EXP/'figures/MANIFEST.json',dict(cases=figmanifest,GT_conditioned_diagnostic=True,biological_identity_verified=False))
    table=['| Training | Input | Raw50 | Raw75 | Final TP50 | FP50 | AP | Small TP50/133 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in metrics:table.append(f"| {r['training_arm']} | {r['arm']} | {r['raw50']} | {r['raw75']} | {r['TP50']} | {r['FP50']} | {float(r['AP']):.3f} | {r['source_small_TP50']} |")
    ch=['| Input | New final | Lost final | Net TP50 | New/lost small | New/lost raw50 |','|---|---:|---:|---:|---:|---:|']
    for r in changes:ch.append(f"| {r['observation']} | {r['new_final']} | {r['lost_final']} | {int(r['new_final'])-int(r['lost_final']):+d} | {r['new_small']}/{r['lost_small']} | {r['new_raw']}/{r['lost_raw']} |")
    losses={}
    for a in ['S0','S1']:
        rr=rows(EXP/'logs'/a/'TRAINING_LOG.csv')
        losses[a]=dict(first20_mean=float(np.mean([float(z['mean_loss']) for z in rr[:20]])),
                       last20_mean=float(np.mean([float(z['mean_loss']) for z in rr[-20:]])))
    statuses={a:read(EXP/'logs'/a/'STATUS.json') for a in ['S0','S1']}
    report="""# Conventional lossless-scale exposure: executed matched control

This is an established augmentation control on the completed Mask DINO, not an original method. The parent M1pilot showed lossless encoding sensitivity, but did not identify training versus architecture. These new runs keep original trainable scope, mask-only matching, losses,300queries and source supervision. No backbone/head freezing, new module or new loss.

## Executed contract

Same seed42parent,796eligible TRAIN windows after excluding three entire diagnostic tiles from ADDITIONAL updates, one full pass,796microbatches/398AdamW updates per arm, D4draws/targets/order/LR matched. The parent had already trained on those diagnostic tiles; this is not independent carrier validation or source holdout. S0 always sees native1024; S1 sees native1024on even draws and exact2×RGB/mask/validity replication on odd draws. Tiny/overlapping targets do not disappear, and no GT is used at inference.

Optimizer states reset identically; fixed LR1e-5nonbackbone/1e-6backbone after20-update warmup, weight decay.05 and clipnorm.01. Common standard activation checkpointing makes full-scope2048training feasible. One strictbf16pointwise gradient assertion failed initially; the record is retained. A preregistered engineering correction then verified fp32gradient parity(relativeL2≈3.9e-7) and measured bf16recompute rounding(relativeL2≈.00208). Scientific weights reload the original parent, never engineering-probe weights. No model-quality outcome informed that numerical correction.

Only optimizer updates and source-data exposure are matched. S1 processes2.5×as many raster pixels because half its inputs have4×area; FLOPs and wall time are not matched. Thus any benefit belongs to this conventional augmentation/computation treatment, not scale supervision isolated from compute or a novel mechanism.

## Actual final-mask outcomes

All24newRGB-only predictions were frozen before new metrics. A=native1024,B=lossless2048replication,C=512mean-degraded,D=replicatedC1024, with identical canonical mask coordinates, common validity and final selection.

"""
    report+='\n'.join(table)+'\n\nS1minusS0gross changes on identical reference IDs:\n\n'+'\n'.join(ch)
    report+=f"\n\nRegistered relevance screen: S0A/Braw50gap={decision['gap_S0']}; S1gap={decision['gap_S1']}; S1nativeAfinalTP50delta={decision['native_TP50_delta_S1_minus_S0']}; screen_pass={decision['registered_sufficiency_screen']}. This narrow screen is not statistical significance, universal sufficiency, deployment improvement or original-method entry. If S0gap<=0, it is inapplicable rather than a failed scale hypothesis.\n"
    by={(r['training_arm'],r['arm']):r for r in metrics}
    s0a,s1a=by['S0','A'],by['S1','A']
    equal=rows(EXP/'results/SAME_COUNT.csv')
    eq={(r['training_arm'],r['observation']):r for r in equal}
    recovered=rows(EXP/'results/RECOVERY_AND_LOSS.csv')
    formed=sum(r['observation']=='B' and r['new_final']=='True' and r['S0_failure']=='RAW_ABSENT' for r in recovered)
    reduction=100*(decision['gap_S0']-decision['gap_S1'])/decision['gap_S0'] if decision['gap_S0']>0 else float('nan')
    report+='\n## Scientific decision from these actual results\n\n'
    report+=f"The registered narrow screen passes: A/Braw50gap shrinks by{reduction:.2f}%. On replicated inputB, finalTP50 rises114→236(123new/1lost); {formed}of the123new final matches had no qualifying final raw mask in S0B. At equal443outputs,TP50 rises114→233. Thus the effect is not only accepting more predictions. BfinalTP50 improves on each diagnostic source(ZF2+56,TBS+40,BCI+26), with only one crop per source.\n\n"
    report+=f"NativeAis not a practical improvement: TP50remains276(2new/2lost), but smallTP50falls118→116, APchanges{float(s1a['AP'])-float(s0a['AP']):+.3f}, AP75changes{float(s1a['AP75'])-float(s0a['AP75']):+.3f}, and FP50rises163→222(+36.20%). Both lost references are source-small; both new references are outside that stratum. NativeAPdeclines on all three diagnostic sources. At equal439nativeoutputs, S1has275TPversus276in S0. The narrow gate intentionally does not certify acceptableAP/FP; its pass cannot override these regressions.\n\n"
    report+="The treatment is pair-specific, not demonstrated general scale invariance. On the untrained C/Dpair, raw50discrepancy increases35→73 and finalTP50discrepancy46→79. S1Bstill trails its own nativeAby36raw qualifications and40final matches. These are counterexamples to a claim that the augmentation completely solves raster dependence.\n\n"
    report+="Decision: **ORDINARY_SCALE_EXPOSURE_REDUCES_ENCODING_GAP_WITH_NATIVE_TRADEOFF_ORIGINAL_METHOD_NOT_ESTABLISHED**. Training-scale exposure with additional raster computation can substantially change the frozen pilot phenomenon. Therefore that pilot cannot establish the necessity of an original observability-conditioned mechanism. This does not isolate exposure from extra compute, certify an architectural cause, explain BAM, or refute every possible M1mechanism. The tested one-pass recipe is closed; S1is a diagnostic checkpoint, not a replacement release for RR_MASK_INIT/nativeDINO. No extra epochs, threshold rescue, V2or new loss is queued.\n"
    report+="""
All values refer to298TRAIN reference occurrences/133source-small annotation occurrences, not biological trees or independent sites. AP is0–100 on these fixed crops, never directly compared with full-tile SAME6/BAM. Source-small uses the inherited original published-tile raw annotation area; crop fragments and source truncation remain explicit.

See SOURCE_RESULTS.csv and SAME_COUNT.csv for source dependence and GT-free equal-output-count controls. These controls do not turn gains into pure geometry causality or remove loss of previously correct instances. No favorable threshold, seed, epoch or branch is selected after scoring.

## Interpretation boundary

The experiment tests whether an ordinary scale-exposure treatment changes the phenomenon. A successful known treatment argues against using this engineered fixed-size failure as evidence that an original observability architecture is required. A failed finite one-pass treatment does not prove ordinary scale learning, all ScaleMatch/DINO-MS variants or the architecture incapable.

Training input dimensions and apparent object pixel size change jointly under replication. Matching updates/data does not isolate every optimization pathway. The tiny exposed diagnostic cohort and single paired seed cannot establish independent geographic efficacy or reliable biological identities. A native-loss/FP trade-off must not be hidden by the reduction in A/Bgap.

The original method-development goal remains unachieved. No new module, loss, threshold sweep or automatic longer schedule is justified solely by these results.

## Engineering / runtime / artifacts

"""
    for a,s in statuses.items():
        report+=f"- {a}:398committed updates,796microbatches,{s['elapsed_seconds']:.1f}s,peak allocatedGPU{s['peak_GPU_MiB']:.1f}MiB; full model/optimizer/RNG state saved. First/last20update mean losses {losses[a]['first20_mean']:.3f}/{losses[a]['last20_mean']:.3f}; these are optimization traces, not segmentation proof.\n"
    report+=f"\nVerified {ver['raw_masks_checked']}raw/{ver['final_masks_checked']}final RLE masks, source/RNG schedule, full trainable-state changes and unchanged original checkpoint. Both optimizer states are retained; final.pt hard-links the completed last state to avoid duplicate storage. GPU shared, not reserved. No TEST/protected review/SAME6evaluation.\n"
    report+="\n[Protocol](protocol.md), [numerical correction](ENGINEERING_PARITY_ADDENDUM.md), [full-mask results](results/METRICS.csv), [new/lost references](results/RECOVERY_AND_LOSS.csv), [same-count](results/SAME_COUNT.csv), [source effects](results/SOURCE_RESULTS.csv), [verification](VERIFICATION.json), [fixed qualitative figure](figures/FIXED_CONTROL_COMPARISON.png).\n"
    (EXP/'REPORT.md').write_text(report)
    (EXP/'REPORT.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Conventional scale exposure control</title><style>body{max-width:1100px;margin:32px auto;padding:20px;font:17px/1.6 Georgia}table{font:14px system-ui;border-collapse:collapse;width:100%}td,th{border:1px solid #ccd3da;padding:7px}</style>'+markdown.markdown(report,extensions=['tables'])+'</html>')
    write(EXP/'LOSS_SUMMARY.json',losses)
    print(json.dumps(dict(report=True,screen=decision['registered_sufficiency_screen'],novel_method=False)))
if __name__=='__main__':main()
