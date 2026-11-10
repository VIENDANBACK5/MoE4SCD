"""Independent saved-mask checks and the preregistered qualitative sample."""
from pathlib import Path
import json,gzip,csv,hashlib
import numpy as np
from pycocotools import mask as mu
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
EXP=Path(__file__).resolve().parents[1];REPO=EXP.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def gz(p):
    with gzip.open(p,'rt') as f:return json.load(f)
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def rle(r):return dict(size=r['size'],counts=r['counts'].encode())
def matches(a,t):
    z=maximum_bipartite_matching(csr_matrix(a>=t),perm_type='column')
    return {int(i) for i in z if i>=0}
def main():
    freeze=read(EXP/'PREDICTIONS_FROZEN.json');cohort=read(EXP/'COHORT.json')
    gt=gz(EXP/'results/REFERENCE_MASKS.json.gz');by={r['arm']:r for r in rows(EXP/'results/METRICS.csv')}
    actual={a:dict(raw50=0,raw75=0,tp50=0,tp75=0,predictions=0,smalltp=0) for a in 'ABCD'}
    preds={};count=0;peak=0;seconds=0
    for i,row in enumerate(cohort):
        inp=np.load(EXP/f"{row['id']}_inputs.npz");rgb=inp['RGB'].astype(np.float32);coarse=inp['coarse_RGB']
        rep=lambda a:np.repeat(np.repeat(a,2,0),2,1)
        arrays=dict(A=rgb,B=rep(rgb),C=coarse,D=rep(coarse));v=inp['common_valid'];vr=mu.encode(np.asfortranarray(v.astype(np.uint8)))
        assert np.array_equal(arrays['B'][::2,::2],rgb) and np.array_equal(arrays['D'][::2,::2],coarse)
        assert np.array_equal(rep(inp['coarse_valid']),v)
        for arm in 'ABCD':
            name=f"predictions/{row['id']}_{arm}.json.gz";assert sha(EXP/name)==freeze['sha256'][name]
            d=gz(EXP/name);raw=d['raw_predictions'];assert [r['query_id'] for r in raw]==list(range(300))
            assert d['RGB_input_sha256']==ah(arrays[arm]) and d['common_valid_sha256']==ah(v)
            assert d['protocol_sha256']==sha(EXP/'protocol.md') and d['cohort_sha256']==sha(EXP/'COHORT.json')
            rr=[rle(r['rle']) for r in raw];areas=mu.area(rr)
            assert np.array_equal(areas,[r['area'] for r in raw])
            for j,r in enumerate(rr):assert int(mu.area(mu.merge([r,vr],intersect=True)))==int(areas[j])
            pre=[j for j,r in enumerate(raw) if r['score']>=.05 and r['area']>=30]
            assert pre==d['accepted_indices']
            kept=d['final_indices'];supp=d['suppressed_indices'];assert not set(kept)&set(supp) and set(kept)|set(supp)==set(pre)
            assert len(set(kept))==len(kept)
            # Reproduce the exact GT-free NMS, rather than merely check counts.
            recalculated=[]
            for j in sorted(pre,key=lambda q:(-raw[q]['score'],q)):
                if recalculated and mu.iou([rr[j]],[rr[k] for k in recalculated],[0]*len(recalculated)).max()>.5:continue
                recalculated.append(j)
            assert recalculated==kept
            a=np.array(mu.iou(rr,[rle(r['rle']) for r in gt['references'][i]],[0]*len(gt['references'][i])))
            b=a[kept];m=matches(b,.5);m75=matches(b,.75)
            stats=actual[arm];stats['raw50']+=int((a.max(0)>=.5).sum());stats['raw75']+=int((a.max(0)>=.75).sum())
            stats['tp50']+=len(m);stats['tp75']+=len(m75);stats['predictions']+=len(kept)
            stats['smalltp']+=sum(j in m and z['source_annotation_area_m2']<9 for j,z in enumerate(gt['metadata'][i]))
            preds[i,arm]=(d,a)
            count+=300;peak=max(peak,d['peak_GPU_allocated_MiB']);seconds+=d['seconds']
    for arm,s in actual.items():
        b=by[arm]
        for k,col in [('raw50','raw50'),('raw75','raw75'),('tp50','TP50'),('predictions','predictions'),('smalltp','source_small_TP50')]:assert s[k]==int(b[col]),(arm,k)
        assert s['predictions']-s['tp50']==int(b['FP50'])
        assert abs(100*s['tp75']/298-float(b['Recall75']))<1e-10
    # Fixed source-wise selection was specified before outcome scoring.
    allrefs=rows(EXP/'results/PER_REFERENCE.csv');selected=[]
    for i,row in enumerate(cohort):
        aa=[r for r in allrefs if r['arm']=='A' and r['window_id']==row['id']]
        bb={r['reference_id']:r for r in allrefs if r['arm']=='B' and r['window_id']==row['id']}
        eligible=[r for r in aa if r['raw50']=='True' and bb[r['reference_id']]['raw50']=='False'];rule='A_raw50_B_not_raw50'
        if not eligible:eligible=[r for r in aa if r['raw50']=='True'];rule='fallback_A_raw50'
        r=min(eligible,key=lambda z:hashlib.sha256(z['reference_id'].encode()).hexdigest())
        j=next(j for j,z in enumerate(gt['metadata'][i]) if z['reference_id']==r['reference_id'])
        selected.append(dict(window=i,reference_index=j,reference_id=r['reference_id'],selection_rule=rule))
    figures=EXP/'figures';figures.mkdir(exist_ok=True)
    with plt.rc_context({'font.size':9,'axes.titlesize':10,'figure.facecolor':'white'}):
        fig,axes=plt.subplots(3,5,figsize=(14,8.5),layout='constrained')
        for n,z in enumerate(selected):
            i,j=z['window'],z['reference_index'];row=cohort[i];inp=np.load(EXP/f"{row['id']}_inputs.npz")
            target=mu.decode(rle(gt['references'][i][j]['rle'])).astype(bool);ys,xs=np.where(target)
            pad=max(32,int(max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)*.5))
            y0,y1=max(0,ys.min()-pad),min(1024,ys.max()+pad+1);x0,x1=max(0,xs.min()-pad),min(1024,xs.max()+pad+1)
            for k in range(5):
                ax=axes[n,k];ax.imshow(inp['RGB'],interpolation='nearest')
                ax.contour(target,levels=[.5],colors=['#0072B2'],linewidths=1.4,linestyles='dashed')
                ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.set_xticks([]);ax.set_yticks([])
                if k==0:
                    area=gt['metadata'][i][j]['source_annotation_area_m2']
                    ax.set_title(f"TRAIN source {n+1}: RGB/reference\nsource annotation area {area:.3f} m²")
                else:
                    arm='ABCD'[k-1];d,a=preds[i,arm];q=int(a[:,j].argmax());mask=mu.decode(rle(d['raw_predictions'][q]['rle']))
                    if mask.any():ax.contour(mask,levels=[.5],colors=['#D55E00'],linewidths=1.1)
                    ax.set_title(f"{arm}: raw-best q{q}, IoU {a[q,j]:.3f}\ninput {d['input_hw'][0]}²; same canonical crop")
            z['display_crop_xyxy']=[int(x0),int(y0),int(x1),int(y1)]
        fig.suptitle('Frozen TRAIN diagnostic: information-equivalent encodings can lose qualifying masks\nBlue dashed = published reference; orange solid = GT-conditioned raw-best witness, not deployed selection',fontsize=12)
        fig.savefig(figures/'FIXED_ENCODING_COMPARISON.png',dpi=160);plt.close(fig)
    (figures/'MANIFEST.json').write_text(json.dumps(dict(selection=selected,GT_conditioned_diagnostic=True,RGB_adjustments='NONE',display_crops_only_not_inference=True,independent_human_review=False),indent=2)+'\n')
    old=read(EXP/'INHERITED_CACHE_BINDINGS.json')
    assert sha(REPO/'research/ITC_MASK_DINO_FORMATION_V1_20261104/checkpoints/mask_only_seed42/final.pt')==old['sha256']
    result=dict(status='VERIFIED_EXECUTED_FROZEN_ENCODING_DIAGNOSTIC',forwards=12,raw_masks_checked=count,
                final_masks_checked=sum(s['predictions'] for s in actual.values()),reference_denominator=298,
                exact_RGB_reencoding_reconstructed=True,normalized_backbone_equivalence_runtime_assertions=True,
                canonical_validity_and_NMS_recomputed=True,raw_IoUs_and_final_matching_recomputed=True,
                checkpoint_unchanged=True,peak_GPU_allocated_MiB=peak,measured_sum_forward_materialization_seconds=seconds,
                dense_mask_logits_archived=False,analysis_console_serialization_error_preserved=True,
                new_training=False,new_method=False,independent_generalization=False,goal_complete=False)
    (EXP/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
