from common import *
import numpy as np,collections,math
from pycocotools import mask as mu
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
def rle(r):return dict(size=r['size'],counts=r['counts'].encode())
def select(raw,key):
    pre=[i for i,p in enumerate(raw) if p[key]>=.05 and p['area']>=30];keep=[];supp=[]
    for i in sorted(pre,key=lambda i:(-raw[i][key],i)):
        if keep and mu.iou([rle(raw[i]['rle'])],[rle(raw[j]['rle']) for j in keep],[0]*len(keep)).max()>.5:supp.append(i)
        else:keep.append(i)
    return pre,keep,supp
def main():
    frozen=read(EXP/'PREDICTIONS_FROZEN.json');assert frozen['forwards']==12 and len(frozen['sha256'])==24
    assert all(z['native_logits_exact'] and z['native_raw_and_final_reproduced'] and z['paired_sign_gauge_native_logits_exact'] for z in frozen['native_checks'])
    assert len(frozen['native_checks'])==12
    cohort=read(PILOT/'COHORT.json');gt=gz(CONTROL/'evaluations/S0/results/REFERENCE_MASKS.json.gz')
    metrics={(r['training_arm'],r['readout']):r for r in rows(EXP/'results/METRICS.csv')};counts=finals=0
    for arm in ['S0','S1']:
        agg={code:collections.Counter() for code in ['AA','AB','BA','BB']}
        for i,row in enumerate(cohort):
            allp={code:gz(EXP/f"predictions/{arm}/{row['id']}_{code}.json.gz") for code in agg}
            v=np.load(PILOT/f"{row['id']}_inputs.npz")['common_valid'];valid=mu.encode(np.asfortranarray(v.astype(np.uint8)))
            emb=np.load(EXP/f"embeddings/{arm}/{row['id']}.npz")
            for code,d in allp.items():
                p=EXP/f"predictions/{arm}/{row['id']}_{code}.json.gz";assert sha(p)==frozen['sha256'][str(p.relative_to(EXP))]
                assert not d['GT_inference'] and not d['reference_masks_read'] and not d['parameters_updated']
                left,right=code;native=allp[left+left];field=allp[right+right];raw=d['raw_predictions'];assert len(raw)==300
                assert d['embedding_sha256']==native['embedding_sha256'] and d['pixel_field_sha256']==field['pixel_field_sha256']
                assert hashlib.sha256(np.ascontiguousarray(emb['E_'+left]).tobytes()).hexdigest()==d['embedding_sha256']
                assert d['mask_logit_grid_hw']==field['mask_logit_grid_hw']
                for j,z in enumerate(raw):
                    source=native['raw_predictions'][j]
                    for k in ['query_id','encoder_site','class_logit','class_probability','predicted_box_cxcywh']:assert z[k]==source[k]
                    assert z['native_query_score']==source['score']
                    assert 0<=z['mask_quality']<=1.000001 and np.isclose(z['score'],z['class_probability']*z['mask_quality'],atol=1e-7)
                rr=[rle(z['rle']) for z in raw];aa=mu.area(rr);assert np.array_equal(aa,[z['area'] for z in raw])
                for z,ar in zip(rr,aa):assert int(mu.area(mu.merge([z,valid],intersect=True)))==int(ar)
                for key,suffix in [('score',''),('native_query_score','_native_score')]:
                    pre,keep,supp=select(raw,key)
                    assert pre==d['accepted_indices'+suffix] and keep==d['final_indices'+suffix] and supp==d['suppressed_indices'+suffix]
                if left==right:
                    previous=gz(CONTROL/f"evaluations/{arm}/predictions/{row['id']}_{left}.json.gz")
                    for z,y in zip(raw,previous['raw_predictions']):assert z['rle']==y['rle'] and z['score']==y['score']
                    assert d['final_indices']==previous['final_indices']
                a=np.asarray(mu.iou(rr,[rle(z['rle']) for z in gt['references'][i]],[0]*len(gt['references'][i])))
                b=a[d['final_indices']];m=maximum_bipartite_matching(csr_matrix(b>=.5),perm_type='column');m75=maximum_bipartite_matching(csr_matrix(b>=.75),perm_type='column')
                matched={int(x) for x in m if x>=0};s=agg[code]
                s['raw50']+=int((a.max(0)>=.5).sum());s['raw75']+=int((a.max(0)>=.75).sum());s['tp50']+=len(matched);s['tp75']+=int((m75>=0).sum());s['n']+=len(b)
                s['small']+=sum(j in matched and z['source_annotation_area_m2']<9 for j,z in enumerate(gt['metadata'][i]))
                counts+=300;finals+=len(b)
        for code,s in agg.items():
            r=metrics[arm,code]
            for k,v in [('raw50','raw50'),('raw75','raw75'),('tp50','TP50'),('n','predictions'),('small','source_small_TP50')]:assert s[k]==int(r[v]),(arm,code,k)
            assert s['n']-s['tp50']==int(r['FP50']) and int(r['GT'])==298 and int(r['source_small_N'])==133
            assert np.isclose(s['tp75']/298*100,float(r['Recall75']))
    assert counts==7200
    decision=read(EXP/'results/DECISION.json');subset=rows(EXP/'results/S1_STRONG_RESIDUAL_SUBSET.csv');assert len(subset)==20
    previous=rows(REPO/'research/ITC_M1_OBSERVABILITY_SCALE_RESEARCH_V1_20261110/analyses/encoding_residuals_v1/PAIRED_REFERENCES.csv')
    old={r['reference_id'] for r in previous if r['training_arm']=='S1' and r['pair']=='AB' and r['robust_lost']=='True'}
    assert old=={r['reference_id'] for r in subset}
    assert sum(r['AB_recovered']=='True' for r in subset)==decision['AB_recovered']
    assert sum(r['BA_recovered']=='True' for r in subset)==decision['BA_recovered']
    for arm,h in read(EXP/'BINDINGS.json')['checkpoints'].items():assert sha(CONTROL/f'checkpoints/{arm}/final.pt')==h
    write(EXP/'VERIFICATION.json',dict(stage='VERIFIED_FACTUAL_REPRODUCTION_AND_INTERCHANGE_MASKS',forwards=12,pools=24,raw_masks=counts,final_masks=finals,
         native_logit_reconstruction_exact=True,paired_sign_gauge_native_control_exact=True,native_raw_final_scores_exact=True,
         independent_RLE_validity_NMS_matching_checked=True,embedding_and_score_provenance_checked=True,strong_subset_exactly_previous20=True,
         checkpoints_unchanged=True,new_training=False,GT_inference=False,original_method=False,goal_complete=False))
    print(json.dumps(dict(verified=True,raw_masks=counts,final_masks=finals)))
if __name__=='__main__':main()
