"""Post-freeze reference diagnostics; no prediction policy fitting."""
from pathlib import Path
import sys,json,gzip,csv,hashlib,importlib.util,collections,time
import numpy as np
from pycocotools import mask as mu
EXP=Path(__file__).resolve().parents[1];REPO=EXP.parents[3]
OLD=REPO/'research/ITC_MASK_DINO_FORMATION_V1_20261104'
BAM=REPO/'research/ITC_AUTORESEARCH_RESOLUTION_V1_20261109/experiments/frozen_bam_transfer'
spec=importlib.util.spec_from_file_location('sealed_eval_helpers',BAM/'code/analyze.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from treecrown_data.masks import crop_mask,pack,unpack
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def gz(p):
    with gzip.open(p,'rt') as f:return json.load(f)
def csvwrite(path,rs):
    ks=list(dict.fromkeys(k for r in rs for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=ks);w.writeheader();w.writerows(rs)
def write(path,d):Path(path).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def iou(ps,refs):
    return np.array(mu.iou([unpack(r['rle']) for r in ps],[unpack(r['rle']) for r in refs],[0]*len(refs))) if ps else np.zeros((0,len(refs)))

def main():
    start=time.monotonic();out=EXP/'results';out.mkdir(exist_ok=True)
    assert not (out/'COMPLETE.json').exists(),'Do not overwrite completed scientific results'
    freeze=read(EXP/'PREDICTIONS_FROZEN.json');assert freeze['forwards']==12 and len(freeze['sha256'])==12
    for rel,h in freeze['sha256'].items():assert sha(EXP/rel)==h
    assert all(x['actual_backbone_tensor_left_inverse_exact'] for x in freeze['exact_left_inverse_proofs'])
    rows=read(EXP/'COHORT.json');binds={x['window_id']:x for x in read(EXP/'INHERITED_CACHE_BINDINGS.json')['window_bindings']}
    images=[];refs=[];meta=[];domain=[]
    for i,row in enumerate(rows):
        folder=OLD/'train_cache'/row['tile'];assert sha(folder/'references.json')==binds[row['id']]['reference_sha256']
        data=np.load(EXP/f"{row['id']}_inputs.npz");v=data['common_valid'];ov=data['original_valid']
        y,x,h,w=row['window_yxhw'];rr=[];mm=[]
        for r in read(folder/'references.json')['references']:
            m=crop_mask(r['rle'],(y,x,h,w)).astype(bool)
            if not m.any():continue
            orig=m&ov;common=m&v;area=float(r['raw_area'])*.045**2
            rr.append(dict(rle=pack(mu.encode(np.asfortranarray(common.astype(np.uint8)))),original_id=r['id']))
            mm.append(dict(reference_id=r['id'],source=row['source'],window_id=row['id'],
                           source_annotation_area_m2=area,source_annotation_small=area<9,
                           source_truncated=r.get('source_truncated','UNKNOWN'),
                           original_crop_valid_pixels=int(orig.sum()),common_crop_pixels=int(common.sum()),
                           empty_common_support=not common.any()))
        assert len(rr)==row['instances'];refs.append(rr);meta.append(mm)
        images.append(dict(tile=row['id'],height=1024,width=1024))
        domain.append(dict(window_id=row['id'],original_valid_pixels=int(ov.sum()),common_valid_pixels=int(v.sum()),
                           removed_original_valid_pixels=int((ov&~v).sum()),references=len(rr),
                           empty_common_support_references=sum(m['empty_common_support'] for m in mm)))
    metrics=[];source=[];perref=[];preds={};sets={};rawsets={};repro=[]
    for arm in 'ABCD':
        finals=[];allsets=[];allraw=[];nraw=nraw75=rawmatched=0
        for i,row in enumerate(rows):
            d=gz(EXP/'predictions'/f"{row['id']}_{arm}.json.gz");raw=d['raw_predictions']
            assert len(raw)==300 and not d['GT_inference'] and not d['reference_masks_read']
            ps=[raw[q] for q in d['final_indices']];pre=[raw[q] for q in d['accepted_indices']];finals.append(ps)
            a=iou(raw,refs[i]);b=iou(ps,refs[i]);c=iou(pre,refs[i])
            best=a.max(0);fb=b.max(0) if len(b) else np.zeros(len(refs[i]));pb=c.max(0) if len(c) else np.zeros(len(refs[i]))
            m,amb,_=base.match(b,.5);m75,_,_=base.match(b,.75);rm,_,_=base.match(a,.5)
            allsets.append(m);allraw.append(set(np.flatnonzero(best>=.5)));nraw+=int((best>=.5).sum());nraw75+=int((best>=.75).sum());rawmatched+=len(rm)
            for j,r in enumerate(meta[i]):
                fail='MATCHED50' if j in m else 'RAW_ABSENT' if best[j]<.5 else 'FILTERED' if pb[j]<.5 else 'SUPPRESSED' if fb[j]<.5 else 'ONE_TO_ONE_CONFLICT'
                perref.append(dict(arm=arm,**r,raw_best_iou=float(best[j]),raw_best_query=int(a[:,j].argmax()),
                                   raw50=best[j]>=.5,raw75=best[j]>=.75,matched50=j in m,matched75=j in m75,
                                   matching_ambiguous=j in amb,final_best_iou=float(fb[j]),failure=fail))
            if arm=='A':
                legacy=REPO/'research/ITC_AUTORESEARCH_RESOLUTION_V1_20261109/experiments/frozen_readout_capacity/results'/f"{row['id']}_native_raw_masks.json"
                previous=read(legacy);v=np.load(EXP/f"{row['id']}_inputs.npz")['common_valid'];difference=0;queries_different=0
                for q,p in enumerate(raw):
                    old=mu.decode(dict(size=previous['raw_masks'][q]['size'],counts=previous['raw_masks'][q]['counts'].encode())).astype(bool)&v
                    new=mu.decode(unpack(p['rle'])).astype(bool);z=int((old!=new).sum());difference+=z;queries_different+=z>0
                repro.append(dict(window_id=row['id'],legacy_native_pixel_differences=difference,queries_different=queries_different,
                                  max_class_probability_delta=float(np.max(np.abs(np.array(previous['class_probability'])-np.array([x['class_probability'] for x in raw]))))))
        preds[arm]=finals;sets[arm]=allsets;rawsets[arm]=allraw
        prs=[r for r in perref if r['arm']==arm];small=[r for r in prs if r['source_annotation_small']]
        metrics.append(dict(arm=arm,raw50=nraw,raw75=nraw75,raw_one_to_one_TP50=rawmatched,
                            source_small_N=len(small),source_small_raw50=sum(r['raw50'] for r in small),
                            source_small_TP50=sum(r['matched50'] for r in small),
                            source_small_Recall50=100*sum(r['matched50'] for r in small)/len(small),
                            **base.measure(images,refs,finals)))
        for i,row in enumerate(rows):
            subset=[r for r in prs if r['window_id']==row['id']]
            source.append(dict(arm=arm,source=row['source'],raw50=sum(r['raw50'] for r in subset),
                               raw75=sum(r['raw75'] for r in subset),
                               source_small_N=sum(r['source_annotation_small'] for r in subset),
                               source_small_TP50=sum(r['source_annotation_small'] and r['matched50'] for r in subset),
                               **base.measure([images[i]],[refs[i]],[finals[i]])))
    index={(r['arm'],r['reference_id']):r for r in perref}
    changes=[];paired=[];same=[]
    for left,right in [('A','B'),('C','D'),('A','D')]:
        change=[]
        for i,rr in enumerate(meta):
            for r in rr:
                a=index[left,r['reference_id']];b=index[right,r['reference_id']]
                change.append(dict(contrast=f'{left}_to_{right}',**r,new_raw=not a['raw50'] and b['raw50'],
                     lost_raw=a['raw50'] and not b['raw50'],new_final=not a['matched50'] and b['matched50'],
                     lost_final=a['matched50'] and not b['matched50'],before_raw_iou=a['raw_best_iou'],
                     after_raw_iou=b['raw_best_iou'],before_failure=a['failure'],after_failure=b['failure'],
                     matching_ambiguous=a['matching_ambiguous'] or b['matching_ambiguous']))
        changes+=change
        paired.append(dict(contrast=f'{left}_to_{right}',new_raw=sum(r['new_raw'] for r in change),
              lost_raw=sum(r['lost_raw'] for r in change),new_final=sum(r['new_final'] for r in change),
              lost_final=sum(r['lost_final'] for r in change),
              new_small_final=sum(r['new_final'] and r['source_annotation_small'] for r in change),
              lost_small_final=sum(r['lost_final'] and r['source_annotation_small'] for r in change),
              sources_with_raw_losses=len({r['source'] for r in change if r['lost_raw']}),
              ambiguous_change_membership=sum(r['matching_ambiguous'] and (r['new_final'] or r['lost_final']) for r in change)))
        pp={left:[],right:[]}
        for aa,bb in zip(preds[left],preds[right]):
            n=min(len(aa),len(bb));pp[left].append(aa[:n]);pp[right].append(bb[:n])
        for arm in [left,right]:same.append(dict(contrast=f'{left}_to_{right}',arm=arm,**base.measure(images,refs,pp[arm])))
    for name,rs in [('METRICS',metrics),('SOURCE_RESULTS',source),('PER_REFERENCE',perref),('RECOVERY_AND_LOSS',changes),
                    ('PAIRED_RESULTS',paired),('SAME_COUNT',same),('VALID_DOMAIN',domain),('NATIVE_A_REPRODUCTION',repro)]:csvwrite(out/f'{name}.csv',rs)
    with gzip.open(out/'REFERENCE_MASKS.json.gz','wt') as f:json.dump(dict(images=images,references=refs,metadata=meta),f)
    screen={r['contrast']:r['lost_raw']>=5 and r['sources_with_raw_losses']>=2 for r in paired if r['contrast']!='A_to_D'}
    summary=dict(status='EXECUTED_FROZEN_TRAIN_ENCODING_DIAGNOSTIC',forwards=12,windows=3,
                 references=sum(map(len,refs)),source_annotation_small=sum(r['source_annotation_small'] for mm in meta for r in mm),
                 empty_common_support=sum(r['empty_common_support_references'] for r in domain),
                 lossless_relevance_screen=screen,information_deletion_explanation_for_lossless_contrasts='EXCLUDED_BY_EXACT_LEFT_INVERSE',
                 architecture_cause_isolated=False,new_method_supported=False,biological_identity_verified=False,
                 independent_generalization=False,new_training=False,TEST=False,overall_goal_complete=False,
                 frozen_predictions_sha256=sha(EXP/'PREDICTIONS_FROZEN.json'),analysis_seconds=time.monotonic()-start)
    write(out/'COMPLETE.json',summary);print(json.dumps(dict(summary=summary,metrics=metrics,paired=paired),default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))

if __name__=='__main__':main()
