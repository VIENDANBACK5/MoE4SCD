"""POST-HOC EXPLORATORY: quantify frame-edge fragments seen in fixed figures.

No new model, filtering policy, threshold selection or output modification.
The proxy definition below was set after inspecting the fixed two-tile figure.
It is reference-conditioned and does not establish biological identity.
"""
from pathlib import Path
import sys,json,gzip,csv
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from pycocotools import mask as mu
sys.path.insert(0,'/home/chung/RS/data/TreeCrown-MultiSource-V1')
from treecrown_data.masks import unpack
EXP=Path(__file__).resolve().parents[1]
def rgzip(p):
    with gzip.open(p,'rt') as f:return json.load(f)
def main():
    refs=rgzip(EXP/'results/TRAIN_REFERENCE_MASKS.json.gz')['references']
    freeze=json.loads((EXP/'PREDICTIONS_FROZEN.json').read_text());summary=[];details=[]
    for kind,seed in freeze['arms']:
        arm=f'{kind}_seed{seed}'
        for i,ts in enumerate(refs):
            d=rgzip(EXP/'predictions'/arm/f'b{i:03d}.json.gz');ps=[d['raw_predictions'][j] for j in d['final_indices']]
            pp=[unpack(p['rle']) for p in ps];rr=[unpack(t['rle']) for t in ts]
            iou=np.asarray(mu.iou(pp,rr,[0]*len(rr))) if pp and rr else np.zeros((len(pp),len(rr)))
            match=maximum_bipartite_matching(csr_matrix(iou>=.5),perm_type='column') if iou.size else np.full(len(pp),-1)
            owners={int(g):p for p,g in enumerate(match) if g>=0}
            pa=np.asarray([mu.area(p) for p in pp],float);ta=np.asarray([mu.area(t) for t in rr],float)
            inter=np.rint(iou*(pa[:,None]+ta[None,:])/(1+iou))
            pbox=[mu.toBbox(p) for p in pp];tbox=[mu.toBbox(t) for t in rr]
            counts=dict(arm=arm,tile=i,accepted=len(ps),TP=int((match>=0).sum()),FP=int((match<0).sum()),TP_touch_internal_edge=0,FP_touch_internal_edge=0,reference_fragment_proxy=0,proxy_reference_ambiguous=0)
            for p,z in enumerate(ps):
                vy,vx=z['window_yx'];x,y,w,h=pbox[p]
                edge=bool((vx>0 and x==vx) or (vy>0 and y==vy) or (vx+1024<2048 and x+w==vx+1024) or (vy+1024<2048 and y+h==vy+1024))
                if edge:counts['TP_touch_internal_edge' if match[p]>=0 else 'FP_touch_internal_edge']+=1
                candidates=[]
                if edge and match[p]<0 and iou[p].max(initial=0)<.5:
                    for g in owners:
                        q=owners[g];gx,gy,gw,gh=tbox[g]
                        outside=gx<vx or gy<vy or gx+gw>vx+1024 or gy+gh>vy+1024
                        if outside and ps[q]['window_yx']!=z['window_yx'] and inter[p,g]>=.8*pa[p] and .1*ta[g]<=inter[p,g]<.5*ta[g]:candidates.append(g)
                proxy=len(candidates)==1
                counts['reference_fragment_proxy']+=proxy;counts['proxy_reference_ambiguous']+=len(candidates)>1
                details.append(dict(arm=arm,tile=i,candidate_id=z['candidate_id'],matched_reference_index=int(match[p]),FP_reference_relative=bool(match[p]<0),touch_internal_view_edge=edge,fragment_proxy=proxy,proxy_ambiguous=len(candidates)>1,proxy_reference_id=ts[candidates[0]]['original_id'] if proxy else None,score=z['score']))
            summary.append(counts)
    for name,rows in [('VIEW_EDGE_BY_TILE',summary),('VIEW_EDGE_PER_PREDICTION',details)]:
        with (EXP/f'results/{name}.csv').open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    pooled=[]
    for arm in sorted({q['arm'] for q in summary}):
        a=[q for q in summary if q['arm']==arm]
        pooled.append(dict(arm=arm,**{k:sum(q[k] for q in a) for k in a[0] if k not in ['arm','tile']}))
    (EXP/'results/VIEW_EDGE_SUMMARY.json').write_text(json.dumps(dict(status='POST_HOC_EXPLORATORY_REFERENCE_CONDITIONED',definition='Unmatched mask touches an internal view edge; no IoU50 to any reference; exactly one already recovered reference from another view crosses this view and covers>=80% of prediction but prediction covers10%-<50% of reference.',biological_identity_established=False,any_predictions_changed=False,results=pooled),indent=2)+'\n')
    print(json.dumps(pooled))
if __name__=='__main__':main()
