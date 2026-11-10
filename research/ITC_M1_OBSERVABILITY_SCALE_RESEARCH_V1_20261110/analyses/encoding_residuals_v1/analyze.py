"""Finite post-hoc analysis; saved masks only, no carrier import or execution."""
from pathlib import Path
import csv,json,gzip,hashlib,collections
import numpy as np
from pycocotools import mask as mu

HERE=Path(__file__).resolve().parent
M1=HERE.parents[1]
CONTROL=M1/'experiments/scale_exposure_control_v1'
PILOT=M1/'experiments/frozen_encoding_v1'
REPO=M1.parents[1]
OLD=REPO/'research/ITC_MASK_DINO_FORMATION_V1_20261104'

def read(p):return json.loads(Path(p).read_text())
def gz(p):
    with gzip.open(p,'rt') as f:return json.load(f)
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rle(r):return dict(size=r['size'],counts=r['counts'].encode() if isinstance(r['counts'],str) else r['counts'])
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def table(p,rr):
    keys=list(dict.fromkeys(k for r in rr for k in r))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rr)
def truth(x):return x=='True'

def main():
    assert not (HERE/'COMPLETE.json').exists(),'Completed analysis is immutable'
    assert read(CONTROL/'FINAL_VERIFICATION.json')['raw_masks']==7200
    bindings={}
    def bind(p):bindings[str(p.relative_to(REPO))]=sha(p)
    bind(CONTROL/'FINAL_ARTIFACTS.json');bind(HERE/'PROTOCOL.md');bind(HERE/'analyze.py')
    cohort=read(PILOT/'COHORT.json');source_area={};references={}
    gt=gz(CONTROL/'evaluations/S0/results/REFERENCE_MASKS.json.gz')
    for i,row in enumerate(cohort):
        path=OLD/'train_cache'/row['tile']/'references.json';bind(path)
        for r in read(path)['references']:source_area[r['id']]=int(mu.area(rle(r['rle'])))
        for r,m in zip(gt['references'][i],gt['metadata'][i]):references[m['reference_id']]=rle(r['rle'])
    data={};predictions={};witness={}
    for arm in ['S0','S1']:
        path=CONTROL/f'evaluations/{arm}/results/PER_REFERENCE.csv';bind(path)
        data[arm]={(r['arm'],r['reference_id']):r for r in rows(path)}
        assert len(data[arm])==1192
        for row in cohort:
            for obs in 'ABCD':
                path=CONTROL/f"evaluations/{arm}/predictions/{row['id']}_{obs}.json.gz";bind(path)
                predictions[arm,obs,row['id']]=gz(path)
        for (obs,rid),r in data[arm].items():
            p=predictions[arm,obs,r['window_id']]['raw_predictions'][int(r['raw_best_query'])]
            target=references[rid];mask=rle(p['rle'])
            tp=int(mu.area(mu.merge([mask,target],intersect=True)))
            fp=int(mu.area(mask))-tp;fn=int(mu.area(target))-tp
            iou=tp/max(tp+fp+fn,1)
            assert abs(iou-float(r['raw_best_iou']))<1e-12
            witness[arm,obs,rid]=dict(tp=tp,fp=fp,fn=fn)
    paired=[]
    for arm in ['S0','S1']:
        for left,right in [('A','B'),('C','D')]:
            ids=sorted(rid for obs,rid in data[arm] if obs==left)
            assert len(ids)==298
            for rid in ids:
                a,b=data[arm][left,rid],data[arm][right,rid]
                x,y=float(a['raw_best_iou']),float(b['raw_best_iou'])
                loss=x>=.5 and y<.5;gain=x<.5 and y>=.5
                ratio=int(a['common_crop_pixels'])/max(source_area[rid],1)
                nearwhole=ratio>=.9 and a['source_truncated']=='False'
                u,v=witness[arm,left,rid],witness[arm,right,rid]
                category=('FP_AND_FN_INCREASE' if v['fp']>u['fp'] and v['fn']>u['fn'] else
                          'FP_INCREASE_WITHOUT_FN_INCREASE' if v['fp']>u['fp'] else
                          'FN_INCREASE_WITHOUT_FP_INCREASE' if v['fn']>u['fn'] else 'NEITHER_INCREASES')
                paired.append(dict(training_arm=arm,pair=left+right,reference_id=rid,source=a['source'],window_id=a['window_id'],
                    source_small=truth(a['source_annotation_small']),source_truncated=a['source_truncated'],
                    common_crop_pixels=a['common_crop_pixels'],published_tile_RLE_pixels=source_area[rid],crop_fraction=ratio,
                    nearwhole_source_proxy=nearwhole,left_raw_iou=x,right_raw_iou=y,raw_lost=loss,raw_new=gain,
                    lost_left_50_55=loss and x<.55,lost_left_55_75=loss and .55<=x<.75,
                    robust_lost=loss and x>=.75,severe_lost=loss and x>=.75 and y<.25,
                    lost_right_45_50=loss and y>=.45,
                    new_final=not truth(a['matched50']) and truth(b['matched50']),
                    lost_final=truth(a['matched50']) and not truth(b['matched50']),
                    right_failure=b['failure'],left_query=a['raw_best_query'],right_query=b['raw_best_query'],
                    delta_false_foreground=v['fp']-u['fp'],delta_missed_target=v['fn']-u['fn'],error_profile=category))
    assert len(paired)==1192
    table(HERE/'PAIRED_REFERENCES.csv',paired)
    summaries=[];profiles=[]
    for arm in ['S0','S1']:
        for pair in ['AB','CD']:
            group=[r for r in paired if r['training_arm']==arm and r['pair']==pair]
            for label,part in [('all',group),('source_small',[r for r in group if r['source_small']]),
                               ('nearwhole_source_proxy',[r for r in group if r['nearwhole_source_proxy']])]:
                z=dict(training_arm=arm,pair=pair,subgroup=label,reference_N=len(part))
                for key in ['raw_new','raw_lost','lost_left_50_55','lost_left_55_75','robust_lost','severe_lost','lost_right_45_50','new_final','lost_final']:
                    z[key]=sum(r[key] for r in part)
                z['coverage_count_delta']=z['raw_new']-z['raw_lost'];summaries.append(z)
            for source in sorted({r['source'] for r in group}):
                part=[r for r in group if r['source']==source]
                profiles.append(dict(training_arm=arm,pair=pair,source=source,N=len(part),
                       raw_new=sum(r['raw_new'] for r in part),raw_lost=sum(r['raw_lost'] for r in part),
                       robust_lost=sum(r['robust_lost'] for r in part),
                       robust_nearwhole=sum(r['robust_lost'] and r['nearwhole_source_proxy'] for r in part)))
    table(HERE/'SUMMARY.csv',summaries);table(HERE/'SOURCE_PROFILE.csv',profiles)
    native=[]
    for (obs,rid),a in data['S0'].items():
        if obs!='A':continue
        b=data['S1'][obs,rid]
        if a['matched50']==b['matched50']:continue
        native.append(dict(reference_id=rid,source=a['source'],source_small=a['source_annotation_small'],
              change='LOST' if truth(a['matched50']) else 'NEW',
              S0_raw_iou=a['raw_best_iou'],S1_raw_iou=b['raw_best_iou'],S0_final_iou=a['final_best_iou'],S1_final_iou=b['final_best_iou'],
              S0_failure=a['failure'],S1_failure=b['failure'],
              matching_ambiguous=truth(a['matching_ambiguous']) or truth(b['matching_ambiguous'])))
    table(HERE/'NATIVE_FINAL_CHANGES.csv',native)
    stability={}
    for pair in ['AB','CD']:
        sets={a:{r['reference_id'] for r in paired if r['training_arm']==a and r['pair']==pair and r['raw_lost']} for a in ['S0','S1']}
        stability[pair]=dict(shared_raw_losses=len(sets['S0']&sets['S1']),S0_only_raw_losses=len(sets['S0']-sets['S1']),S1_only_raw_losses=len(sets['S1']-sets['S0']))
    dump(HERE/'RESIDUAL_ID_STABILITY.json',stability)
    errors=[]
    for arm in ['S0','S1']:
        for pair in ['AB','CD']:
            part=[r for r in paired if r['training_arm']==arm and r['pair']==pair and r['raw_lost']]
            for name,n in collections.Counter(r['error_profile'] for r in part).items():
                errors.append(dict(training_arm=arm,pair=pair,raw_loss_best_witness_profile=name,count=n))
    table(HERE/'BEST_WITNESS_ERROR_PROFILES.csv',errors)
    dump(HERE/'INPUT_BINDINGS.json',bindings)
    dump(HERE/'COMPLETE.json',dict(status='EXECUTED_POST_HOC_SAVED_MASK_ANALYSIS',paired_records=len(paired),
         raw_witnesses_RLE_checked=len(witness),native_final_changes=len(native),new_carrier_forwards=0,new_training_updates=0,
         new_reference_annotations=0,thresholds_changed=False,query_trajectory_measured=False,architectural_cause_identified=False,
         independent_reference_stability=False,original_method_established=False,overall_goal_complete=False))
    print(json.dumps(dict(summary=summaries,stability=stability,native_changes=native),indent=2))
if __name__=='__main__':main()
