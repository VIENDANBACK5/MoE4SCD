"""Checkpoint, schedule and actual RLE verification; no model-quality fitting."""
from shared import *
import csv,gzip,collections
from pycocotools import mask as mu
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
PILOT=EXP.parent/'frozen_encoding_v1'
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def gz(p):
    with gzip.open(p,'rt') as f:return json.load(f)
def rle(r):return dict(size=r['size'],counts=r['counts'].encode())
def main():
    gate=read(EXP/'PROBE2_COMPLETE.json');assert 'PASSED' in gate['stage']
    assert (EXP/'PROBE_FAILURE.json').exists() and gate['original_failure_preserved']
    plan=read(EXP/'PLAN.json');tr=read(EXP/'TRAIN_WINDOWS.json');assert len(tr)==796
    logs={};checkpoint_records={}
    parent=torch.load(OLD/'checkpoints/mask_only_seed42/final.pt',map_location='cpu',weights_only=False)['model']
    for arm in ['S0','S1']:
        status=read(EXP/'logs'/arm/'STATUS.json');assert status['stage']=='COMPLETED_FIXED_SCHEDULE'
        cp=EXP/'checkpoints'/arm/'final.pt';assert sha(cp)==status['checkpoint_sha256']
        ck=torch.load(cp,map_location='cpu',weights_only=False)
        assert ck['updates']==398 and ck['next_microbatch']==796 and ck['parent_sha256']==plan['checkpoint_sha256']
        assert ck['protocol_sha256']==sha(EXP/'protocol.md') and ck['schedule_sha256']==sha(EXP/'TRAIN_WINDOWS.json')
        assert ck['model'].keys()==parent.keys() and all(ck['model'][n].shape==x.shape for n,x in parent.items())
        assert len(ck['optimizer']['state'])==393
        changed=collections.Counter()
        for n,x in parent.items():
            y=ck['model'][n];assert torch.isfinite(y).all()
            if not torch.equal(x,y):
                key='backbone' if n.startswith('backbone.') else 'pixel_decoder' if '.pixel_decoder.' in n else 'predictor'
                changed[key]+=1
        assert all(changed[k]>0 for k in ['backbone','pixel_decoder','predictor'])
        checkpoint_records[arm]=dict(sha256=status['checkpoint_sha256'],updates=398,optimizer_states=393,
                trainable_parameters=ck['trainable_parameters'],changed_tensors=dict(changed),bytes=cp.stat().st_size)
        logs[arm]=rows(EXP/'logs'/arm/'TRAINING_LOG.csv');assert len(logs[arm])==398
        for u,r in enumerate(logs[arm]):
            assert int(r['update'])==u+1 and int(r['microbatches'])==2*(u+1)
            assert r['windows']=='|'.join(x['id'] for x in tr[2*u:2*u+2])
            assert r['scales']==('1|1' if arm=='S0' else '1|2')
            assert np.isfinite(float(r['mean_loss'])) and np.isfinite(float(r['gradient_norm']))
        del ck
    for a,b in zip(logs['S0'],logs['S1']):
        for k in ['windows','rotations','flips','target_counts','lr']:assert a[k]==b[k],k
    del parent
    frozen=read(EXP/'EVALUATIONS_FROZEN.json');assert frozen['forwards']==24
    table={(r['training_arm'],r['arm']):r for r in rows(EXP/'results/METRICS.csv')}
    count=finals=0
    for arm in ['S0','S1']:
        folder=EXP/'evaluations'/arm;f=read(folder/'PREDICTIONS_FROZEN.json')
        assert sha(folder/'PREDICTIONS_FROZEN.json')==frozen['arms'][arm]['predictions_freeze_sha256']
        gt=gz(folder/'results/REFERENCE_MASKS.json.gz')
        stats={o:dict(raw50=0,raw75=0,tp50=0,tp75=0,smalltp=0,n=0) for o in 'ABCD'}
        for i,row in enumerate(read(PILOT/'COHORT.json')):
            inp=np.load(PILOT/f"{row['id']}_inputs.npz");v=inp['common_valid'];vr=mu.encode(np.asfortranarray(v.astype(np.uint8)))
            for obs in 'ABCD':
                rel=f"predictions/{row['id']}_{obs}.json.gz";assert sha(folder/rel)==f['sha256'][rel]
                d=gz(folder/rel);assert not d['GT_inference'] and not d['reference_masks_read']
                assert d['checkpoint_sha256']==checkpoint_records[arm]['sha256']
                raw=d['raw_predictions'];assert len(raw)==300 and [p['query_id'] for p in raw]==list(range(300))
                rr=[rle(p['rle']) for p in raw]
                areas=mu.area(rr);assert np.array_equal(areas,[p['area'] for p in raw])
                for r,area in zip(rr,areas):assert int(mu.area(mu.merge([r,vr],intersect=True)))==int(area)
                pre=[q for q,p in enumerate(raw) if p['score']>=.05 and p['area']>=30];assert pre==d['accepted_indices']
                keep=[]
                for q in sorted(pre,key=lambda q:(-raw[q]['score'],q)):
                    if keep and mu.iou([rr[q]],[rr[k] for k in keep],[0]*len(keep)).max()>.5:continue
                    keep.append(q)
                assert keep==d['final_indices'] and set(keep)|set(d['suppressed_indices'])==set(pre)
                a=np.array(mu.iou(rr,[rle(x['rle']) for x in gt['references'][i]],[0]*len(gt['references'][i])))
                b=a[keep];s=stats[obs];s['raw50']+=int((a.max(0)>=.5).sum());s['raw75']+=int((a.max(0)>=.75).sum())
                m=maximum_bipartite_matching(csr_matrix(b>=.5),perm_type='column');m75=maximum_bipartite_matching(csr_matrix(b>=.75),perm_type='column')
                matched={int(j) for j in m if j>=0};s['tp50']+=len(matched);s['tp75']+=int((m75>=0).sum());s['n']+=len(keep)
                s['smalltp']+=sum(j in matched and z['source_annotation_area_m2']<9 for j,z in enumerate(gt['metadata'][i]))
                count+=300;finals+=len(keep)
        for obs,s in stats.items():
            r=table[arm,obs]
            for a,b in [('raw50','raw50'),('raw75','raw75'),('tp50','TP50'),('n','predictions'),('smalltp','source_small_TP50')]:assert s[a]==int(r[b]),(arm,obs,a)
            assert int(r['GT'])==298 and s['n']-s['tp50']==int(r['FP50'])
    assert count==7200 and sha(OLD/'checkpoints/mask_only_seed42/final.pt')==plan['checkpoint_sha256']
    write(EXP/'VERIFICATION.json',dict(stage='VERIFIED_TWO_FULL_SCOPE_MATCHED_CONTROLS',scientific_updates_per_arm=398,
         scientific_microbatches_per_arm=796,training_dataset_windows=796,paired_data_augmentation_and_lr_exact=True,
         original_trainable_scope=True,probe_weights_not_used=True,checkpoints=checkpoint_records,
         evaluation_forwards=24,raw_masks_checked=count,final_masks_checked=finals,
         geometry_validity_NMS_and_matching_recomputed=True,original_checkpoint_unchanged=True,
         engineering_initial_failure_preserved=True,novel_method=False,independent_generalization=False,goal_complete=False))
    print(json.dumps(dict(verified=True,raw_masks=count,final_masks=finals)))
if __name__=='__main__':main()
