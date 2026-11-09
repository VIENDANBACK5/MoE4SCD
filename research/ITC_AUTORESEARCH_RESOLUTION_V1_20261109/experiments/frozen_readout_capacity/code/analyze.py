"""Recompute oracle pixel errors and separate pool capacities from saved RLEs."""
from pathlib import Path
import sys, json, csv
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from pycocotools import mask as mu
sys.path.insert(0, '/home/chung/RS/data/TreeCrown-MultiSource-V1')
from treecrown_data.masks import crop_mask

EXP=Path(__file__).resolve().parents[1]; OUT=EXP/'results'
OLD=EXP.parents[3]/'research/ITC_MASK_DINO_FORMATION_V1_20261104'
def byt(d): return dict(size=d['size'], counts=d['counts'].encode() if isinstance(d['counts'], str) else d['counts'])
def dec(d): return mu.decode(byt(d)).astype(bool)
def enc(m): return mu.encode(np.asfortranarray(m, dtype=np.uint8))

def main():
    rows=list(csv.DictReader((OUT/'PER_REFERENCE.csv').open()))
    extra=[]; pools=[]
    for w in json.loads((EXP/'SELECTED_TRAIN_WINDOWS.json').read_text()):
        rr=[x for x in rows if x['window']==w['id']]
        refs={x['id']:x for x in json.loads((OLD/'train_cache'/w['tile']/'references.json').read_text())['references']}
        yy,xx,hh,ww=w['window_yxhw']
        v=np.asarray(np.load(OLD/'train_cache'/w['tile']/'valid.npy',mmap_mode='r')[yy:yy+hh,xx:xx+ww]).astype(bool)
        native=json.loads((OUT/f"{w['id']}_native_raw_masks.json").read_text())['raw_masks']
        oracle=json.loads((OUT/f"{w['id']}_oracle_masks.json").read_text())['chosen_by_sampled_training_loss']
        targets=[]
        for i,x in enumerate(rr):
            target=crop_mask(refs[x['reference_id']]['rle'],w['window_yxhw']).astype(bool)&v
            targets.append(enc(target))
            d={'window':w['id'],'reference_id':x['reference_id'],'gt_valid_pixels':int(target.sum()),'valid_pixels':int(v.sum())}
            for name,m in [('native',dec(native[int(x['baseline_best_query'])])),('oracle',dec(oracle[i]))]:
                assert not (m&~v).any()
                tp=int((m&target).sum());fp=int((m&~target).sum());fn=int((~m&target).sum());iou=tp/max(1,tp+fp+fn)
                expected=float(x['baseline_raw_best_iou'] if name=='native' else x['chosen_oracle_iou'])
                assert abs(iou-expected)<1e-6
                d.update({f'{name}_tp_pixels':tp,f'{name}_fp_pixels':fp,f'{name}_fn_pixels':fn,f'{name}_iou':iou})
            d.update(loss50=d['native_iou']>=.5 and d['oracle_iou']<.5,new50=d['native_iou']<.5 and d['oracle_iou']>=.5,
                     oracle_fp_increase=d['oracle_fp_pixels']-d['native_fp_pixels'],oracle_fn_increase=d['oracle_fn_pixels']-d['native_fn_pixels'])
            extra.append(d)
        z={'window':w['id'],'references':len(targets),'oracle_pool_GT_conditioned':True}
        for name,pool in [('native',native),('oracle',oracle)]:
            iou=mu.iou([byt(x) for x in pool],targets,[0]*len(targets))
            np.savez_compressed(OUT/f"{w['id']}_{name}_iou_matrix.npz",iou=iou)
            for t in [.5,.75]:
                z[f'{name}_reference_availability_{t}']=int((iou.max(0)>=t).sum())
                match=maximum_bipartite_matching(csr_matrix(iou>=t),perm_type='column')
                z[f'{name}_one_to_one_{t}']=int((match>=0).sum())
        pools.append(z)
    with (OUT/'FULL_PIXEL_ERROR_DECOMPOSITION.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(extra[0]));wr.writeheader();wr.writerows(extra)
    lost=[d for d in extra if d['loss50']];missing=[d for d in extra if d['native_iou']<.5]
    summary=dict(targets=len(extra),native_raw50=sum(d['native_iou']>=.5 for d in extra),oracle_target50=sum(d['oracle_iou']>=.5 for d in extra),
                 new50=sum(d['new50'] for d in extra),lost50=len(lost),lost_with_FP_increase=sum(d['oracle_fp_increase']>0 for d in lost),
                 lost_with_FN_increase=sum(d['oracle_fn_increase']>0 for d in lost),
                 lost_with_both_increasing=sum(d['oracle_fp_increase']>0 and d['oracle_fn_increase']>0 for d in lost),
                 native_missing=len(missing),native_missing_with_oracle50=sum(d['oracle_iou']>=.5 for d in missing),
                 RLE_IoU_recomputation_verified=True,empirical_result_is_deployment=False)
    (OUT/'ANALYSIS_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'ORACLE_POOL_DIAGNOSTICS.json').write_text(json.dumps(pools,indent=2)+'\n')
    print(json.dumps(summary))
if __name__=='__main__': main()
