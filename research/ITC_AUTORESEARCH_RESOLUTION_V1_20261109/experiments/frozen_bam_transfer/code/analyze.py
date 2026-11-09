"""Reference scoring strictly after all fixed RGB predictions are frozen."""
from pathlib import Path
import os,sys,json,gzip,csv,hashlib,collections,time
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
EXP=Path(__file__).resolve().parents[1];REPO=EXP.parents[3];R=REPO/'research'
sys.path.insert(0,str(R/'ITC_MASK_DINO_FORMATION_V1_20261104'))
import bootstrap
sys.path.insert(0,str(R/'D0_SELVA_M0_GPU_CACHE_V2_20261031/scripts'))
import fast_evaluator as fe
from pycocotools import mask as mu
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
import numpy as np
from treecrown_data import TreeCrownData
from treecrown_data.masks import pack,unpack

def read(p):return json.loads(Path(p).read_text())
def rgzip(p):
    with gzip.open(p,'rt') as f:return json.load(f)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def ious(ps,rs):
    return np.asarray(mu.iou([unpack(z['rle']) for z in ps],[unpack(z['rle']) for z in rs],[0]*len(rs))) if ps and rs else np.zeros((len(ps),len(rs)))
def match(a,t):
    m=maximum_bipartite_matching(csr_matrix(a>=t),perm_type='column') if a.size else np.full(len(a),-1)
    matched={int(j) for j in m if j>=0};unmatched=set(range(a.shape[1]))-matched
    # Alternating paths from unmatched references identify ambiguous membership.
    reachable=set(unmatched);queue=list(unmatched);ambiguous=set()
    while queue:
        ref=queue.pop()
        for p in np.flatnonzero(a[:,ref]>=t):
            nxt=int(m[p])
            if nxt>=0:
                ambiguous.update([ref,nxt])
                if nxt not in reachable:reachable.add(nxt);queue.append(nxt)
    return matched,ambiguous,m
KEEP={'AP','AP50','AP75','COCO_AR','maxDets','GT','predictions','tiles','Recall50','Recall75','TP50','FP50','FN50','count_MAE','count_bias','significant_merges','significant_splits','accepted_mask_availability50'}
def measure(images,refs,ps):
    # Original all-area evaluator is unchanged. Its legacy 4.5cm size/boundary fields are withheld.
    return {k:v for k,v in fe.measure(images,refs,ps).items() if k in KEEP}

def main():
    start=time.monotonic();freeze=read(EXP/'PREDICTIONS_FROZEN.json')
    assert freeze['stage']=='ALL_RGB_ONLY_PREDICTIONS_FROZEN_BEFORE_REFERENCES'
    input_hashes={}
    for p,h in freeze['sha256'].items():
        assert sha(EXP/p)==h,p
        d=rgzip(EXP/p);key=d['source_image_id'];value=(d['RGB_sha256'],d['valid_sha256'])
        if key in input_hashes:assert input_hashes[key]==value,'Arms did not receive identical input pixels'
        input_hashes[key]=value
    assert len(freeze['sha256'])==216 and freeze['forwards']==1944
    cohort=read(EXP/'metadata/SELECTED_COHORT.json')['selected'];results=EXP/'results';results.mkdir(exist_ok=True)
    assert not (results/'COMPLETE.json').exists(),'No overwrite of completed analysis'
    D=TreeCrownData.from_root('/home/chung/RS/data/TreeCrown-MultiSource-V1')
    view=D.open(source='bamforests',split='train',track='legacy_reference',access_role='development')
    ind={r['image_id']:i for i,r in enumerate(view.rows)};images=[];refs=[];refmeta=[];empty=[]
    for i,row in enumerate(cohort):
        s=view[ind[row['image_id']]]
        assert input_hashes[row['image_id']]==(hashlib.sha256(s['rgb'].tobytes()).hexdigest(),hashlib.sha256(s['valid_mask'].tobytes()).hexdigest())
        v=mu.encode(np.asfortranarray(s['valid_mask'].astype(np.uint8)))
        ts=[];meta=[]
        for aid,r0,record in zip(s['original_instance_ids'],s['instance_masks'].rles,s['annotation_records']):
            original=unpack(r0);rr=mu.merge([original,v],intersect=True);area=int(mu.area(rr))
            if not area:empty.append(dict(tile=i,reference_id=aid));continue
            bx,by,bw,bh=mu.toBbox(rr)
            ts.append(dict(rle=pack(rr),original_id=aid))
            meta.append(dict(reference_id=aid,tile_mask_area_m2=area*row['gx_m']*row['gy_m'],
                original_source_crown_area_m2=None,original_size_known=False,
                tile_boundary_contact=bool(bx<=0 or by<=0 or bx+bw>=s['width'] or by+bh>=s['height']),
                alpha_clipped=area<int(mu.area(original)),raw_pixels=int(mu.area(original)),valid_pixels=area))
        images.append(dict(tile=f'b{i:03d}',height=s['height'],width=s['width'],spatial_group=row['spatial_group']))
        refs.append(ts);refmeta.append(meta)
    write(results/'REFERENCE_METADATA.json',dict(original_source_area_available=False,references=refmeta,empty=empty,independent_annotation_stability='UNVERIFIED'))
    arms=[f'{a}_seed{s}' for a,s in freeze['arms']]
    metrics=[];source=[];tile_metrics=[];perref=[];sizes=[];cost=[];preds={};matsets={};ambsets={};rawsets={}
    for arm in arms:
        finals=[];armsets=[];armamb=[];armraw=[]
        for i,row in enumerate(cohort):
            d=rgzip(EXP/'predictions'/arm/f'b{i:03d}.json.gz')
            assert d['RGB_only'] and not d['GT_inference'] and not d['annotations_read']
            assert d['source_image_id']==row['image_id']
            raw=d['raw_predictions'];pre=[raw[j] for j in d['pre_fusion_indices']];ps=[raw[j] for j in d['final_indices']];finals.append(ps)
            a=ious(raw,refs[i]);b=ious(ps,refs[i]);c=ious(pre,refs[i]);best=a.max(0) if len(a) else np.zeros(len(refs[i]));fb=b.max(0) if len(b) else np.zeros(len(refs[i]));pb=c.max(0) if len(c) else np.zeros(len(refs[i]))
            matched,amb,m=match(b,.5);m75,_,_=match(b,.75);armsets.append(matched);armamb.append(amb);armraw.append(set(np.flatnonzero(best>=.5).tolist()))
            tm=measure([images[i]],[refs[i]],[ps]);tile_metrics.append(dict(arm=arm,tile=i,spatial_group=row['spatial_group'],raw50=int((best>=.5).sum()),raw75=int((best>=.75).sum()),matching_ambiguous_references=len(amb),duplicate_unmatched=int(sum(m[j]<0 and b[j].max(initial=0)>=.5 for j in range(len(ps)))),**tm))
            cost.append(dict(arm=arm,tile=i,seconds=d['seconds'],peak_GPU_allocated_MiB=d['peak_GPU_allocated_MiB'],peak_RSS_GiB=d['peak_RSS_GiB'],forwards=d['forwards']))
            for j,t in enumerate(refmeta[i]):
                fail='MATCHED50' if j in matched else ('RAW_ABSENT' if best[j]<.5 else ('FILTERED' if pb[j]<.5 else ('SUPPRESSED' if fb[j]<.5 else ('MATCHING_AMBIGUOUS' if j in amb else 'ONE_TO_ONE_CONFLICT'))))
                perref.append(dict(arm=arm,tile=i,spatial_group=row['spatial_group'],**t,raw_best_iou=float(best[j]),final_best_iou=float(fb[j]),matched50=j in matched,matched75=j in m75,matching_ambiguous=j in amb,failure=fail))
            print('SCORED',arm,i,flush=True)
        preds[arm]=finals;matsets[arm]=armsets;ambsets[arm]=armamb;rawsets[arm]=armraw
        metrics.append(dict(arm=arm,raw50=sum(len(v) for v in armraw),original_source_small_Recall50=None,**measure(images,refs,finals)))
        for group in sorted({r['spatial_group'] for r in cohort}):
            ix=[i for i,r in enumerate(cohort) if r['spatial_group']==group]
            source.append(dict(arm=arm,spatial_group=group,**measure([images[i] for i in ix],[refs[i] for i in ix],[finals[i] for i in ix])))
        ar=[r for r in perref if r['arm']==arm]
        for lo,hi in [(0,1),(1,4),(4,9),(9,float('inf'))]:
            for boundary in [False,True]:
                subset=[r for r in ar if lo<=r['tile_mask_area_m2']<hi and r['tile_boundary_contact']==boundary]
                sizes.append(dict(arm=arm,size_semantics='TILE_MASK_AREA_PROXY_NOT_ORIGINAL_SOURCE_CROWN',area_min=lo,area_max=None if np.isinf(hi) else hi,boundary_contact=boundary,n=len(subset),TP50=sum(r['matched50'] for r in subset),Recall50=100*sum(r['matched50'] for r in subset)/len(subset) if subset else None))
    paired=[];recovery=[];samecount=[]
    by={r['arm']:r for r in metrics}
    for seed in [42,43,44]:
        rr=f'rr_seed{seed}';ctrl=f'control_seed{seed}';new=lost=rawnew=rawlost=uncertain=0
        for i,row in enumerate(cohort):
            for j,t in enumerate(refmeta[i]):
                old=j in matsets[ctrl][i];now=j in matsets[rr][i];amb=j in (ambsets[ctrl][i]|ambsets[rr][i]);rn=j in rawsets[rr][i];ro=j in rawsets[ctrl][i]
                new+=now and not old;lost+=old and not now;rawnew+=rn and not ro;rawlost+=ro and not rn;uncertain+=amb
                recovery.append(dict(seed=seed,tile=i,spatial_group=row['spatial_group'],**t,control_matched50=old,RR_matched50=now,new=now and not old,lost=old and not now,matching_ambiguous=amb,control_raw50=ro,RR_raw50=rn))
        paired.append(dict(seed=seed,delta_AP=by[rr]['AP']-by[ctrl]['AP'],delta_AP75=by[rr]['AP75']-by[ctrl]['AP75'],delta_Recall50_pp=by[rr]['Recall50']-by[ctrl]['Recall50'],delta_TP50=by[rr]['TP50']-by[ctrl]['TP50'],delta_FP50=by[rr]['FP50']-by[ctrl]['FP50'],new=new,lost=lost,new_raw50=rawnew,lost_raw50=rawlost,matching_ambiguous_references=uncertain,original_source_small_gate='NOT_ESTIMABLE'))
        rps=[];cps=[]
        for rp,cp in zip(preds[rr],preds[ctrl]):
            n=min(len(rp),len(cp));rps.append(rp[:n]);cps.append(cp[:n])
        for label,pp in [('RR',rps),('CONTROL',cps)]:samecount.append(dict(seed=seed,arm=label,**measure(images,refs,pp)))
    patterns=[]
    for i,row in enumerate(cohort):
        for j,t in enumerate(refmeta[i]):
            ar=['r5_seed42','m2f_seed42','dino_seed42'];bits=''.join('1' if j in matsets[a][i] else '0' for a in ar)
            patterns.append(dict(tile=i,spatial_group=row['spatial_group'],**t,pattern_R5_M2F_DINO=bits,all_raw_missing=all(j not in rawsets[a][i] for a in ar),any_matching_ambiguity=any(j in ambsets[a][i] for a in ar),reference_stability='UNVERIFIED'))
    for name,rows in [('METRICS',metrics),('SPATIAL_GROUP_RESULTS',source),('TILE_RESULTS',tile_metrics),('PER_REFERENCE',perref),('TILE_AREA_PROXY_RESULTS',sizes),('RUNTIME',cost),('RR_PAIRED_RESULTS',paired),('RECOVERY_AND_LOSS',recovery),('SAME_COUNT_RESULTS',samecount),('CROSS_CARRIER_PATTERNS',patterns)]:csvwrite(results/f'{name}.csv',rows)
    write(results/'COMPLETE.json',dict(stage='EXECUTED_EXTERNAL_TRAIN_REFERENCE_FIDELITY',images=len(images),references=sum(map(len,refs)),arms=arms,forwards=1944,TEST=False,protected_review=False,new_training=False,original_source_small_gate='NOT_ESTIMABLE',independent_human_reference_stability='NOT_ESTABLISHED',unqualified_independent_generalization='NOT_ESTABLISHED',analysis_seconds=time.monotonic()-start,freeze_sha256=sha(EXP/'PREDICTIONS_FROZEN.json')))
    print(json.dumps(dict(metrics=metrics,paired=paired)),flush=True)
if __name__=='__main__':main()
