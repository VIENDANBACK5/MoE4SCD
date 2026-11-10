"""Reference join after complete prediction freeze. No fitting."""
from common import *
import importlib.util,collections,time
import numpy as np
spec=importlib.util.spec_from_file_location('common_evaluator',REPO/'research/ITC_AUTORESEARCH_RESOLUTION_V1_20261109/experiments/frozen_bam_transfer/code/analyze.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
def main():
    start=time.monotonic();freeze=read(EXP/'PREDICTIONS_FROZEN.json')
    assert freeze['forwards']==12 and len(freeze['sha256'])==24
    assert not (EXP/'results/COMPLETE.json').exists()
    for p,h in freeze['sha256'].items():assert sha(EXP/p)==h
    out=EXP/'results';out.mkdir(exist_ok=True)
    gt=gz(CONTROL/'evaluations/S0/results/REFERENCE_MASKS.json.gz');cohort=read(PILOT/'COHORT.json')
    assert sum(map(len,gt['references']))==298
    metrics=[];source=[];sizes=[];perref=[];pools={};rawsets={};matsets={};scorecontrol=[]
    for arm in ['S0','S1']:
        for code in ['AA','AB','BA','BB']:
            finals=[];counter=[];prs=[];rawset=[];matset=[];raw50=raw75=rawmatched=duplicates=0
            for i,row in enumerate(cohort):
                d=gz(EXP/f"predictions/{arm}/{row['id']}_{code}.json.gz");raw=d['raw_predictions']
                assert len(raw)==300 and not d['GT_inference'] and not d['reference_masks_read']
                ps=[raw[q] for q in d['final_indices']];pre=[raw[q] for q in d['accepted_indices']]
                finals.append(ps);counter.append([dict(raw[q],score=raw[q]['native_query_score']) for q in d['final_indices_native_score']])
                a=base.ious(raw,gt['references'][i]);b=base.ious(ps,gt['references'][i]);c=base.ious(pre,gt['references'][i])
                best=a.max(0);fb=b.max(0) if len(b) else np.zeros(len(best));pb=c.max(0) if len(c) else np.zeros(len(best))
                m,amb,assignment=base.match(b,.5);m75,_,_=base.match(b,.75);rm,_,_=base.match(a,.5)
                rawset.append(set(np.flatnonzero(best>=.5)));matset.append(m);raw50+=int((best>=.5).sum());raw75+=int((best>=.75).sum());rawmatched+=len(rm)
                duplicates+=int(sum(assignment[k]<0 and b[k].max(initial=0)>=.5 for k in range(len(ps))))
                for j,meta in enumerate(gt['metadata'][i]):
                    fail='MATCHED50' if j in m else 'RAW_ABSENT' if best[j]<.5 else 'FILTERED' if pb[j]<.5 else 'SUPPRESSED' if fb[j]<.5 else 'ONE_TO_ONE_CONFLICT'
                    r=dict(training_arm=arm,readout=code,**meta,raw_best_iou=float(best[j]),raw_best_query=int(a[:,j].argmax()),
                           raw50=bool(best[j]>=.5),raw75=bool(best[j]>=.75),matched50=j in m,matched75=j in m75,
                           matching_ambiguous=j in amb,final_best_iou=float(fb[j]),failure=fail)
                    prs.append(r);perref.append(r)
            pools[arm,code]=finals;rawsets[arm,code]=rawset;matsets[arm,code]=matset
            small=[r for r in prs if r['source_annotation_small']]
            metrics.append(dict(training_arm=arm,readout=code,raw50=raw50,raw75=raw75,raw_one_to_one_TP50=rawmatched,
                   source_small_N=len(small),source_small_raw50=sum(r['raw50'] for r in small),source_small_TP50=sum(r['matched50'] for r in small),
                   source_small_Recall50=100*sum(r['matched50'] for r in small)/len(small),duplicate_unmatched=duplicates,**base.measure(gt['images'],gt['references'],finals)))
            scorecontrol.append(dict(training_arm=arm,readout=code,score_policy='FIXED_FACTUAL_QUERY_COMPOSITE_SCORE',**base.measure(gt['images'],gt['references'],counter)))
            for i,row in enumerate(cohort):
                part=[r for r in prs if r['window_id']==row['id']]
                source.append(dict(training_arm=arm,readout=code,source=row['source'],raw50=sum(r['raw50'] for r in part),raw75=sum(r['raw75'] for r in part),
                     source_small_TP50=sum(r['source_annotation_small'] and r['matched50'] for r in part),**base.measure([gt['images'][i]],[gt['references'][i]],[finals[i]])))
            for lo,hi in [(0,1),(1,4),(4,9),(9,float('inf'))]:
                part=[r for r in prs if lo<=r['source_annotation_area_m2']<hi]
                sizes.append(dict(training_arm=arm,readout=code,area_min=lo,area_max=None if np.isinf(hi) else hi,N=len(part),
                    raw50=sum(r['raw50'] for r in part),raw75=sum(r['raw75'] for r in part),TP50=sum(r['matched50'] for r in part)))
            print('SCORED',arm,code,flush=True)
    index={(r['training_arm'],r['readout'],r['reference_id']):r for r in perref}
    changes=[];summaries=[];same=[]
    for arm in ['S0','S1']:
        for left,right in [('BB','AB'),('BB','BA'),('AA','AB'),('AA','BA'),('AA','BB')]:
            part=[]
            for (aa,code,rid),a in index.items():
                if aa!=arm or code!=left:continue
                b=index[arm,right,rid]
                r=dict(training_arm=arm,contrast=left+'_to_'+right,reference_id=rid,source=a['source'],source_small=a['source_annotation_small'],
                     new_raw=not a['raw50'] and b['raw50'],lost_raw=a['raw50'] and not b['raw50'],
                     new_final=not a['matched50'] and b['matched50'],lost_final=a['matched50'] and not b['matched50'],
                     before_raw_iou=a['raw_best_iou'],after_raw_iou=b['raw_best_iou'],before_failure=a['failure'],after_failure=b['failure'],
                     matching_ambiguous=a['matching_ambiguous'] or b['matching_ambiguous'])
                changes.append(r);part.append(r)
            summaries.append(dict(training_arm=arm,contrast=left+'_to_'+right,new_raw=sum(r['new_raw'] for r in part),lost_raw=sum(r['lost_raw'] for r in part),
                  new_final=sum(r['new_final'] for r in part),lost_final=sum(r['lost_final'] for r in part),
                  new_small=sum(r['new_final'] and r['source_small'] for r in part),lost_small=sum(r['lost_final'] and r['source_small'] for r in part)))
            pp={left:[],right:[]}
            for x,y in zip(pools[arm,left],pools[arm,right]):
                n=min(len(x),len(y));pp[left].append(x[:n]);pp[right].append(y[:n])
            for code in [left,right]:same.append(dict(training_arm=arm,contrast=left+'_to_'+right,readout=code,**base.measure(gt['images'],gt['references'],pp[code])))
    # The known exposed20-reference subset is selected only after all GT-free pools are frozen.
    subset=[]
    for (arm,code,rid),a in index.items():
        if arm!='S1' or code!='AA' or not a['raw75'] or index[arm,'BB',rid]['raw50']:continue
        subset.append(dict(reference_id=rid,source=a['source'],source_small=a['source_annotation_small'],
                  **{z+'_raw_iou':index[arm,z,rid]['raw_best_iou'] for z in ['AA','AB','BA','BB']},
                  AB_recovered=index[arm,'AB',rid]['raw50'],BA_recovered=index[arm,'BA',rid]['raw50']))
    assert len(subset)==20
    x=sum(r['AB_recovered'] for r in subset);y=sum(r['BA_recovered'] for r in subset)
    label='A_EMBEDDINGS_COMPATIBLE_WITH_B_FIELD' if x>=10 and y<=5 else 'B_EMBEDDINGS_COMPATIBLE_WITH_A_FIELD' if y>=10 and x<=5 else 'MULTIPLE_COMPENSATING_READOUTS' if x>=10 and y>=10 else 'NO_DOMINANT_INTERCHANGE_PATTERN'
    for name,rs in [('METRICS',metrics),('PER_REFERENCE',perref),('SOURCE_RESULTS',source),('SIZE_RESULTS',sizes),('RECOVERY_AND_LOSS',changes),
                    ('CHANGE_SUMMARY',summaries),('SAME_COUNT',same),('ORIGINAL_SCORE_CONTROL',scorecontrol),('S1_STRONG_RESIDUAL_SUBSET',subset)]:csvwrite(out/f'{name}.csv',rs)
    write(out/'DECISION.json',dict(status='EXECUTED_FIXED_CHECKPOINT_READOUT_INTERCHANGE',S1_strong_residual_N=20,AB_recovered=x,BA_recovered=y,
           operational_pattern=label,architecture_cause_identified=False,feature_capacity_certificate=False,novel_method=False,
           independent_generalization=False,biological_identity_verified=False,original_goal_complete=False))
    write(out/'COMPLETE.json',dict(forwards=12,readout_pools=24,raw_masks=7200,reference_occurrences=298,per_reference_rows=len(perref),new_training=False,
          seconds=time.monotonic()-start,freeze_sha256=sha(EXP/'PREDICTIONS_FROZEN.json')))
    print(json.dumps(dict(pattern=label,AB=x,BA=y)),flush=True)
if __name__=='__main__':main()
