"""Score-separated, denominator-explicit diagnostics on original instance masks."""
import contextlib
import io
import hashlib
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mu
from .core import binary, encode, require, size_bin

def ratio(n,d):return float(n/d) if d else None

def iou_matrix(gt,pred):
    allm=list(gt)+list(pred)
    if allm:
        shape=np.asarray(allm[0]).shape
        allm=[binary(m,shape) for m in allm]
    return np.array([[ratio(int((g&p).sum()),int((g|p).sum())) or 0. for p in allm[len(gt):]] for g in allm[:len(gt)]],float).reshape(len(gt),len(pred))

def match_matrix(iou,threshold=.5):
    iou=np.asarray(iou,float);require(iou.ndim==2 and np.isfinite(iou).all() and ((iou>=0)&(iou<=1)).all(),"Invalid IoU matrix")
    require(0<threshold<=1,"Positive IoU threshold required")
    if not min(iou.shape):return []
    eligible=iou>=threshold
    # Bonus > maximum possible total IoU difference: exact lexicographic objective.
    weights=np.where(eligible,min(iou.shape)+1+iou,0.)
    rows,cols=linear_sum_assignment(-weights)
    return [(int(i),int(j),float(iou[i,j])) for i,j in zip(rows,cols) if eligible[i,j]]

def recall(gt,pred,gt_areas,threshold=.5):
    require(len(gt)==len(gt_areas),"GT area alignment mismatch")
    matches=match_matrix(iou_matrix(gt,pred),threshold);hit={i for i,j,v in matches}
    bins={}
    for b in ('lt1','1to2','2to4','4to9','ge9'):
        ids=[i for i,a in enumerate(gt_areas) if size_bin(a)==b]
        bins[b]={'n_gt':len(ids),'tp':len(hit&set(ids)),'recall':ratio(len(hit&set(ids)),len(ids))}
    return {'tp':len(matches),'n_gt':len(gt),'n_pred':len(pred),'recall':ratio(len(matches),len(gt)),
            'precision':ratio(len(matches),len(pred)),'bins':bins,'matches':matches}

def grouping(cover,pairs,same,scores=None,threshold=None,max_dets=None,stable_ids=None):
    cover=np.asarray(cover);require(cover.ndim==2 and np.isin(cover,[0,1]).all(),"Binary mask/anchor cover matrix required")
    raw=np.asarray(pairs)
    require(raw.size==0 or (raw.ndim==2 and raw.shape[1]==2 and np.isfinite(raw).all() and np.equal(raw,np.rint(raw)).all()),"Pair indices must be integral Nx2, without rounding")
    pairs=raw.astype(int).reshape(-1,2);same=np.asarray(same)
    require(np.isin(same,[0,1]).all(),"Pair class labels must be binary");same=same.astype(bool)
    require(len(pairs)==len(same),"Pair labels misaligned")
    require(not len(pairs) or (pairs.min()>=0 and pairs.max()<cover.shape[1]),"Anchor index outside cache")
    idx=np.arange(len(cover));structural=threshold is None
    if structural:require(max_dets is None,"Structural metric cannot use score-based maxDets")
    else:
        scores=np.asarray(scores,float);require(scores.shape==(len(cover),) and np.isfinite(scores).all(),"Score alignment/nonfinite score")
        require(np.isfinite(threshold),"Finite E2E threshold required")
        ids=list(range(len(cover))) if stable_ids is None else list(stable_ids)
        require(len(ids)==len(cover) and len(set(ids))==len(ids),"Unique stable prediction IDs required")
        idx=np.array(sorted(idx,key=lambda i:(-scores[i],hashlib.sha256(cover[i].tobytes()).hexdigest(),str(ids[i]))),int)
        if max_dets is not None:require(max_dets>0,"Positive maxDets required");idx=idx[:max_dets]
        idx=idx[scores[idx]>=threshold]
    sub=cover[idx].astype(bool);count=sub.sum(0);assigned=np.full(cover.shape[1],-1,int)
    hit=count==1 if structural else count>0
    if len(sub):assigned[hit]=np.argmax(sub[:,hit],axis=0)
    both=(assigned[pairs[:,0]]>=0)&(assigned[pairs[:,1]]>=0)
    correct=both&((assigned[pairs[:,0]]==assigned[pairs[:,1]])==same)
    result={'stage_policy':'structural_unique_cover' if structural else 'E2E_highest_score',
            'n_pairs':len(pairs),'n_covered':int(both.sum()),'pair_coverage':ratio(int(both.sum()),len(pairs)),
            'pair_abstention':ratio(int((~both).sum()),len(pairs)),
            'anchor_overlap':ratio(int((count>1).sum()),len(count)),'anchor_unassigned':ratio(int((count==0).sum()),len(count))}
    acc=[]
    for label,name in ((same,'same'),(~same,'different')):
        n=int(label.sum());covered=int((label&both).sum());ok=int((label&correct).sum())
        result[name]={'n':n,'covered':covered,'correct':ok,'unconditional':ratio(ok,n),'conditional':ratio(ok,covered)}
        acc.append(ratio(ok,n))
    result['balanced']=sum(acc)/2 if all(x is not None for x in acc) else None
    result['balanced_NA_reason']=None if result['balanced'] is not None else 'one_or_both_pair_classes_absent'
    return result

def core_mask(original,unique_valid,mode):
    m=binary(original);u=binary(unique_valid,m.shape)
    d=ndi.distance_transform_edt(np.pad(m,1))[1:-1,1:-1]
    require(mode in ('r1','r4','adaptive'),"Unknown anchor core")
    radius={'r1':1,'r4':4}.get(mode,min(4,int(.25*d[u].max())) if u.any() else 0)
    return u&(d>radius),radius

def candidate_strata(candidates_yx,gt,reference_id,reference_stage):
    require(bool(reference_id) and reference_stage in (0,1,2),"Fixed reference/stage required for C0/C1")
    raw=np.asarray(candidates_yx)
    require(raw.size==0 or (raw.ndim==2 and raw.shape[1]==2 and np.isfinite(raw).all() and np.equal(raw,np.rint(raw)).all()),"Candidate coordinates must be integral Nx2, without rounding")
    points=raw.astype(int).reshape(-1,2);result=[]
    for m in gt:
        m=binary(m);require(not len(points) or ((points>=0).all() and (points[:,0]<m.shape[0]).all() and (points[:,1]<m.shape[1]).all()),"Candidate outside image")
        result.append(bool(m[points[:,0],points[:,1]].any()) if len(points) else False)
    return {'reference_id':reference_id,'reference_stage':reference_stage,'C1':result,'definition':'candidate pixel inside original GT; not biological detection'}

def conditioned_recall(hit,strata,expected_reference_id=None):
    if expected_reference_id is not None:require(strata['reference_id']==expected_reference_id,"C0/C1 reference changed across compared heads")
    hit=np.asarray(hit,bool);c=np.asarray(strata['C1'],bool);require(hit.shape==c.shape,"Strata/GT mismatch")
    return {name:{'n':int(v.sum()),'tp':int((v&hit).sum()),'recall':ratio(int((v&hit).sum()),int(v.sum()))} for name,v in [('C0',~c),('C1',c)]}

def proposal_coverage(gt,proposals,purity_includes_background):
    all_masks=list(gt)+list(proposals)
    if not all_masks:return {'all_proposal_coverage':[],'owner_linked_coverage':[],'purity':[],'ambiguous_owner_proposals':0}
    shape=np.asarray(all_masks[0]).shape;gt=[binary(g,shape) for g in gt];proposals=[binary(p,shape) for p in proposals]
    foreground=np.logical_or.reduce(gt) if gt else np.zeros(shape,bool)
    all_union=np.logical_or.reduce(proposals) if proposals else np.zeros(shape,bool)
    intersections=np.array([[int((g&p).sum()) for p in proposals] for g in gt],int).reshape(len(gt),len(proposals))
    owners=[];purity=[]
    for j,p in enumerate(proposals):
        values=intersections[:,j];maximum=int(values.max()) if len(values) else 0
        hits=np.flatnonzero(values==maximum) if maximum else []
        owners.append(int(hits[0]) if len(hits)==1 else None)
        denominator=int(p.sum()) if purity_includes_background else int((p&foreground).sum())
        purity.append(ratio(maximum,denominator))
    linked=[]
    for i,g in enumerate(gt):
        members=[p for j,p in enumerate(proposals) if owners[j]==i]
        union=np.logical_or.reduce(members) if members else np.zeros(shape,bool)
        linked.append(ratio(int((g&union).sum()),int(g.sum())))
    return {'all_proposal_coverage':[ratio(int((g&all_union).sum()),int(g.sum())) for g in gt],
        'owner_linked_coverage':linked,'purity':purity,'purity_includes_background':purity_includes_background,
        'ambiguous_owner_proposals':sum(x is None for x in owners),'owner_rule':'unique largest original-GT overlap; ties/uncovered abstain; not biological identity'}

def accuracy_coverage_curve(cover,pairs,same,scores,thresholds,max_dets,stable_ids=None):
    require(thresholds and len(set(thresholds))==len(thresholds),"Frozen distinct score thresholds required")
    return [{'threshold':float(t),'grouping':grouping(cover,pairs,same,scores,t,max_dets,stable_ids)} for t in thresholds]

def validate_pair_cache(gt,annotation_ids,valid,anchors,anchor_GT_ids,pairs,same,core_mode):
    from .core import supports,canonical_hash
    valid=binary(valid);s=supports(gt,valid,annotation_ids);anchors=np.asarray(anchors)
    require(anchors.ndim==2 and anchors.shape[1]==2 and np.isfinite(anchors).all() and np.equal(anchors,np.rint(anchors)).all(),"Integral anchor coordinates required")
    anchors=anchors.astype(int);require(len(anchors)==len(anchor_GT_ids),"Anchor GT identity length mismatch")
    domains=[core_mask(m,u,core_mode)[0] for m,u in zip(gt,s['unique_valid'])]
    for (y,x),aid in zip(anchors,anchor_GT_ids):
        require(aid in annotation_ids and 0<=y<valid.shape[0] and 0<=x<valid.shape[1],"Anchor source ID/coordinate invalid")
        require(bool(domains[annotation_ids.index(aid)][y,x]),"Anchor violates original unique-valid core")
    raw=np.asarray(pairs);require(raw.size==0 or (raw.ndim==2 and raw.shape[1]==2 and np.isfinite(raw).all() and np.equal(raw,np.rint(raw)).all()),"Cache pair indices must be integral Nx2")
    require(np.isin(same,[0,1]).all(),"Cache pair labels must be binary")
    pair=raw.astype(int).reshape(-1,2)
    require(not len(pair) or (pair.min()>=0 and pair.max()<len(anchors)),"Pair cache references missing anchors")
    expected=[anchor_GT_ids[a]==anchor_GT_ids[b] for a,b in pair]
    require(np.array_equal(expected,np.asarray(same,bool)),"Same/different labels disagree with original GT IDs")
    return {'status':'PASS','core_mode':core_mode,'n_GT':len(gt),'eligible_GT':sum(d.any() for d in domains),
        'excluded_GT_ids':[aid for aid,d in zip(annotation_ids,domains) if not d.any()],
        'n_anchors':len(anchors),'n_pairs':len(pair),'same_pairs':sum(expected),'different_pairs':len(pair)-sum(expected),
        'anchor_cache_hash':canonical_hash({'anchors':anchors.tolist(),'anchor_GT_ids':list(anchor_GT_ids),'pairs':pair.tolist(),'same':expected,'core_mode':core_mode}),
        'scope':'Native annotation relations and reference geometry only; not independent biological identity'}

def overlap_diagnostics(gt,pred,min_gt_coverage):
    require(0<min_gt_coverage<=1,"Explicit overlap significance required")
    cover=np.array([[int((binary(g)&binary(p,np.asarray(g).shape)).sum())/max(int(np.asarray(g).sum()),1) for p in pred] for g in gt]).reshape(len(gt),len(pred))
    edges=cover>=min_gt_coverage
    return {'definition':'graph counts at explicit fraction-of-GT overlap; not causal error decomposition','threshold':min_gt_coverage,
            'merge_predictions':int((edges.sum(0)>1).sum()),'split_GT':int((edges.sum(1)>1).sum()),
            'duplicate_extra_predictions':sum(max(int(n)-1,0) for n in edges.sum(1))}

def count_statistics(gt_counts,pred_counts,area_certificate):
    g=np.asarray(gt_counts,int);p=np.asarray(pred_counts,int)
    require(g.ndim==p.ndim==1 and g.shape==p.shape and (g>=0).all() and (p>=0).all(),"Invalid counts")
    require(isinstance(area_certificate,dict) and area_certificate.get('method')=='exact_affine_cell_union' and area_certificate.get('deduplicated') is True,"Verified union-area certificate required; tile sums are not accepted")
    valid_union_area_m2=area_certificate['valid_union_area_m2']
    require(np.isfinite(valid_union_area_m2) and valid_union_area_m2>0,"Deduplicated valid area required")
    diff=p-g
    return {'n_scenes':len(g),'count_MAE':float(abs(diff).mean()) if len(g) else None,'signed_bias':float(diff.mean()) if len(g) else None,
            'total_signed_bias_per_ha':float(diff.sum())/(valid_union_area_m2/10000),'valid_union_area_m2':valid_union_area_m2}

def boundary_f1(pred,gt,tolerance_px=None,affine=None,tolerance_m=None):
    p=binary(pred);g=binary(gt,p.shape)
    physical=tolerance_m is not None
    require((physical and tolerance_px is None and affine is not None and np.isfinite(tolerance_m) and tolerance_m>=0) or
            (not physical and tolerance_px is not None and np.isfinite(tolerance_px) and tolerance_px>=0),"Specify one finite boundary tolerance with declared units")
    pb=p&~ndi.binary_erosion(p,border_value=0);gb=g&~ndi.binary_erosion(g,border_value=0)
    npix,ng=int(pb.sum()),int(gb.sum())
    if physical:
        from scipy.spatial import cKDTree
        from .core import pixel_area
        pixel_area(affine);a,b,c,d,e,f=affine
        def coords(mask):
            y,x=np.where(mask);return np.stack([a*(x+.5)+b*(y+.5)+c,d*(x+.5)+e*(y+.5)+f],axis=1)
        pc,gc=coords(pb),coords(gb)
        pp=ratio(int((cKDTree(gc).query(pc)[0]<=tolerance_m+1e-12).sum()) if ng else 0,npix)
        rr=ratio(int((cKDTree(pc).query(gc)[0]<=tolerance_m+1e-12).sum()) if npix else 0,ng)
    else:
        pp=ratio(int((pb&(ndi.distance_transform_edt(~gb)<=tolerance_px)).sum()) if ng else 0,npix)
        rr=ratio(int((gb&(ndi.distance_transform_edt(~pb)<=tolerance_px)).sum()) if npix else 0,ng)
    f=None if pp is None and rr is None else (2*(pp or 0)*(rr or 0)/((pp or 0)+(rr or 0)) if (pp or 0)+(rr or 0) else 0.)
    return {'precision':pp,'recall':rr,'F1':f,'pred_boundary_pixels':npix,'gt_boundary_pixels':ng,'tolerance_px':tolerance_px,'tolerance_m':tolerance_m}

def coco_ap(gt,pred,scores,max_dets=400,affine=None,area_range_m2=None):
    """Real COCO evaluator; used in synthetic tests only during hardening. Scale 0..1."""
    require(len(pred)==len(scores) and np.isfinite(scores).all(),"Prediction/score mismatch")
    if not gt:return {'AP':None,'reason':'no_GT','n_pred':len(pred)}
    if not pred:return {'AP':0.,'reason':None,'n_pred':0}
    shape=np.asarray(gt[0]).shape
    from .core import pixel_area
    scale=pixel_area(affine) if affine is not None else 1.
    require(area_range_m2 is None or affine is not None,"Physical AP bins require a pixel affine")
    with contextlib.redirect_stdout(io.StringIO()):
        coco=COCO();coco.dataset={'info':{},'images':[{'id':1,'height':shape[0],'width':shape[1]}],
          'categories':[{'id':1,'name':'tree'}],'annotations':[{'id':i+1,'image_id':1,'category_id':1,'segmentation':encode(binary(m,shape)),
          'area':int(np.asarray(m).sum())*scale,'bbox':mu.toBbox(encode(m)).tolist(),'iscrowd':0} for i,m in enumerate(gt)]};coco.createIndex()
        dt=coco.loadRes([{'image_id':1,'category_id':1,'segmentation':encode(binary(m,shape)),'score':float(s)} for m,s in zip(pred,scores)])
        for ann in dt.dataset['annotations']:ann['area']*=scale
        ev=COCOeval(coco,dt,'segm');ev.params.maxDets=[1,10,max_dets]
        if area_range_m2 is not None:
            lo,hi=area_range_m2;require(0<=lo<hi,"Invalid physical area range")
            # COCO uses an inclusive upper area boundary; nextafter implements registered [lo,hi).
            ev.params.areaRng=[[lo,float(np.nextafter(hi,-np.inf))]];ev.params.areaRngLbl=['physical_registered']
        ev.evaluate();ev.accumulate()
        values=ev.eval['precision'][:,:,:,0,2];values=values[values>=0]
    return {'AP':float(values.mean()) if len(values) else None,'reason':None,'n_pred':len(pred)}
