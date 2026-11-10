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
