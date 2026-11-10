"""Registered control analysis; reuse unchanged pilot evaluator."""
from shared import *
import csv,importlib.util,collections
PILOT=EXP.parent/'frozen_encoding_v1'
spec=importlib.util.spec_from_file_location('pilot_analysis',PILOT/'code/analyze.py')
pa=importlib.util.module_from_spec(spec);spec.loader.exec_module(pa)
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def main():
    frozen=read(EXP/'EVALUATIONS_FROZEN.json');assert frozen['forwards']==24
    metrics=[];references={}
    for arm in ['S0','S1']:
        pa.EXP=EXP/'evaluations'/arm;pa.main()
        metrics += [dict(training_arm=arm,**r) for r in rows(pa.EXP/'results/METRICS.csv')]
        references[arm]=rows(pa.EXP/'results/PER_REFERENCE.csv')
    out=EXP/'results';out.mkdir(exist_ok=True)
    pa.csvwrite(out/'METRICS.csv',metrics)
    index={a:{(r['arm'],r['reference_id']):r for r in rr} for a,rr in references.items()}
    changes=[]
    for obs in 'ABCD':
        for key,a in index['S0'].items():
            if key[0]!=obs:continue
            b=index['S1'][key];old=a['matched50']=='True';new=b['matched50']=='True'
            changes.append(dict(observation=obs,reference_id=a['reference_id'],source=a['source'],source_small=a['source_annotation_small'],
               new_final=new and not old,lost_final=old and not new,new_raw=b['raw50']=='True' and a['raw50']=='False',
               lost_raw=a['raw50']=='True' and b['raw50']=='False',S0_failure=a['failure'],S1_failure=b['failure']))
    pa.csvwrite(out/'RECOVERY_AND_LOSS.csv',changes)
    # Equal-count operating controls, using prediction counts and original scores only.
    gt=pa.gz(EXP/'evaluations/S0/results/REFERENCE_MASKS.json.gz')
    cohort=read(PILOT/'COHORT.json');same=[];sources=[]
    for arm in ['S0','S1']:
        sources += [dict(training_arm=arm,**r) for r in rows(EXP/'evaluations'/arm/'results/SOURCE_RESULTS.csv')]
    pa.csvwrite(out/'SOURCE_RESULTS.csv',sources)
    for obs in 'ABCD':
        pools={a:[] for a in ['S0','S1']}
        for row in cohort:
            d={a:pa.gz(EXP/'evaluations'/a/'predictions'/f"{row['id']}_{obs}.json.gz") for a in pools}
            n=min(len(x['final_indices']) for x in d.values())
            for a,x in d.items():pools[a].append([x['raw_predictions'][q] for q in x['final_indices'][:n]])
        for a,p in pools.items():same.append(dict(observation=obs,training_arm=a,**pa.base.measure(gt['images'],gt['references'],p)))
    pa.csvwrite(out/'SAME_COUNT.csv',same)
    change_summary=[]
    for obs in 'ABCD':
        rr=[r for r in changes if r['observation']==obs]
        change_summary.append(dict(observation=obs,new_final=sum(r['new_final'] for r in rr),lost_final=sum(r['lost_final'] for r in rr),
               new_small=sum(r['new_final'] and r['source_small']=='True' for r in rr),
               lost_small=sum(r['lost_final'] and r['source_small']=='True' for r in rr),
               new_raw=sum(r['new_raw'] for r in rr),lost_raw=sum(r['lost_raw'] for r in rr)))
    pa.csvwrite(out/'CHANGE_SUMMARY.csv',change_summary)
    by={(r['training_arm'],r['arm']):r for r in metrics}
    gap0=int(by['S0','A']['raw50'])-int(by['S0','B']['raw50'])
    gap1=int(by['S1','A']['raw50'])-int(by['S1','B']['raw50'])
    native_delta=int(by['S1','A']['TP50'])-int(by['S0','A']['TP50'])
    gate=gap0>0 and gap1<=.5*gap0 and native_delta>=-3
    write(out/'DECISION.json',dict(stage='EXECUTED_ORDINARY_SCALE_EXPOSURE_CONTROL',gap_S0=gap0,gap_S1=gap1,
        native_TP50_delta_S1_minus_S0=native_delta,registered_sufficiency_screen=gate,
        screen_applicable=gap0>0,novel_method=False,independent_generalization=False,overall_goal_complete=False))
    print(json.dumps(dict(gap_S0=gap0,gap_S1=gap1,native_delta=native_delta,screen=gate)))
if __name__=='__main__':main()
