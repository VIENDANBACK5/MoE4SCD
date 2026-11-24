"""Deterministic crop selection and complete/partial accounting, independent of model."""
import numpy as np
from scipy import ndimage as ndi
from .core import binary, check_capacity, pixel_area, require, rng_for

CATEGORIES=('background','one_complete','two_to_four_complete','crowded','small','multilobed','touching','partial')

def record_crop(masks,annotation_ids,valid,window,affine,source,queries=None,ignore_partial=False):
    valid=binary(valid);masks=[binary(m,valid.shape) for m in masks]
    require(len(annotation_ids)==len(masks) and len(set(annotation_ids))==len(masks),"Annotation IDs misaligned")
    y,x,h,w=window;require(min(y,x)>=0 and min(h,w)>0 and y+h<=valid.shape[0] and x+w<=valid.shape[1],"Crop outside image")
    targets=[];complete=[];partial=[];ignored=[];invalid=[];ambiguous=[]
    for aid,m in zip(annotation_ids,masks):
        crop=m[y:y+h,x:x+w];n=int(crop.sum())
        if not n:continue
        full=n==int(m.sum());edge=bool(m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any())
        (ambiguous if full and edge else complete if full else partial).append(aid)
        if (crop&~valid[y:y+h,x:x+w]).any():invalid.append(aid)
        if ignore_partial and not full:ignored.append(aid)
        else:targets.append({'annotation_id':aid,'mask':crop.copy(),'complete':full and not edge,'complete_native_mask':full,'source_edge_ambiguous':edge,'parent_source':source})
    if queries is not None:check_capacity(len(targets),queries)
    px=pixel_area(affine)
    categories=[]
    if not complete and not partial and not ambiguous:categories.append('background')
    if len(complete)==1:categories.append('one_complete')
    if 2<=len(complete)<=4:categories.append('two_to_four_complete')
    if len(complete)+len(partial)+len(ambiguous)>4:categories.append('crowded')
    if partial:categories.append('partial')
    if any(int(m.sum())*px<1 and aid in {t['annotation_id'] for t in targets} for aid,m in zip(annotation_ids,masks)):categories.append('small')
    # Tags are observable shape proxies, not taxonomy/biological truths.
    for t in targets:
        m=t['mask'];d=ndi.distance_transform_edt(m)
        peaks=(d==ndi.maximum_filter(d,size=3))&(d>1)
        if ndi.label(peaks)[1]>1 and 'multilobed' not in categories:categories.append('multilobed')
    if any((ndi.binary_dilation(a['mask'])&b['mask']).any() for i,a in enumerate(targets) for b in targets[:i]):categories.append('touching')
    return targets,{'source':source,'window_yxhw':list(window),'original_annotation_ids':list(annotation_ids),
        'complete_annotation_ids':complete,'partial_annotation_ids':partial,'ignored_annotation_ids':ignored,
        'visible_fragments':len(partial),'visible_target_count':len(targets),'query_capacity_requirement':len(targets),
        'complete_count':len(complete),'partial_count':len(partial),'ignored_count':len(ignored),'ambiguous_target_count':len(ambiguous),
        'ambiguous_annotation_ids':ambiguous,'ambiguity_status':'fully contained native mask touches source edge; full crown unknown; overlapping reference retained','outside_valid_annotation_ids':invalid,
        'physical_crop_area_m2':h*w*px,'physical_axis_lengths_m':[h*float(np.hypot(affine[1],affine[4])),w*float(np.hypot(affine[0],affine[3]))],
        'pixel_area_m2':px,'valid_fraction':float(valid[y:y+h,x:x+w].mean()),'categories':categories,
        'partial_policy':'ignore_explicitly_accounted' if ignore_partial else 'visible_fragment_retains_original_identity'}

def choose_window(valid,crop_shape,master_seed,epoch,split,tile,crop_index,min_valid,attempts=32):
    valid=binary(valid);h,w=crop_shape;H,W=valid.shape
    require(0<h<=H and 0<w<=W and 0<=min_valid<=1,"Crop cannot fit physical field of view / invalid support threshold")
    rng=rng_for(master_seed,epoch,split,tile,crop_index)
    best=None
    for _ in range(attempts):
        y=int(rng.integers(H-h+1));x=int(rng.integers(W-w+1));fraction=float(valid[y:y+h,x:x+w].mean())
        if best is None or fraction>best[0]:best=(fraction,y,x)
        if fraction>=min_valid:return (y,x,h,w)
    raise ValueError(f"No crop meets declared valid fraction {min_valid}; best={best[0] if best else None}; no silent fallback")

def crown_centered_window(mask,crop_shape,margin):
    mask=binary(mask);ys,xs=np.where(mask);require(len(ys)>0,"Empty crown")
    h,w=crop_shape;H,W=mask.shape
    if ys.ptp()+1+2*margin>h or xs.ptp()+1+2*margin>w or h>H or w>W:
        return {'status':'DOES_NOT_FIT','required_shape':[int(ys.ptp()+1+2*margin),int(xs.ptp()+1+2*margin)]}
    y=int(np.clip((int(ys.min())+int(ys.max())-h+1)//2,0,H-h));x=int(np.clip((int(xs.min())+int(xs.max())-w+1)//2,0,W-w))
    return {'status':'PASS','window':(y,x,h,w)}

def synchronized_transform(image,masks,k=0,flip_x=False,flip_y=False,new_shape=None):
    image=np.asarray(image);ms=[binary(m,image.shape[:2]) for m in masks]
    def geom(a):
        if flip_x:a=a[:,::-1]
        if flip_y:a=a[::-1]
        return np.ascontiguousarray(np.rot90(a,k))
    image=geom(image);ms=[geom(m) for m in ms]
    if new_shape is not None:
        from PIL import Image
        nh,nw=new_shape;require(min(nh,nw)>0,"Positive resize shape required")
        image=np.array(Image.fromarray(image).resize((nw,nh),resample=Image.Resampling.BILINEAR))
        ms=[np.array(Image.fromarray(m).resize((nw,nh),resample=Image.Resampling.NEAREST)).astype(bool) for m in ms]
    return image,ms

def plan_crops(masks,annotation_ids,valid,affine,crop_shape,quotas,master_seed,margin,random_candidates,queries=None):
    """Explicit category quotas, with missing strata/capacity reported rather than hidden."""
    require(set(quotas)==set(CATEGORIES) and all(isinstance(n,int) and n>=0 for n in quotas.values()),"All category quotas must be explicit nonnegative integers")
    valid=binary(valid);h,w=crop_shape;H,W=valid.shape
    require(0<h<=H and 0<w<=W,"Declared crop does not fit source; alternative FOV needed")
    windows=set();does_not_fit=[]
    for aid,m in zip(annotation_ids,masks):
        candidate=crown_centered_window(m,crop_shape,margin)
        if candidate['status']=='PASS':windows.add(candidate['window'])
        else:does_not_fit.append({'annotation_id':aid,**candidate})
    rng=rng_for(master_seed,0,'train','planner',0)
    for _ in range(random_candidates):windows.add((int(rng.integers(H-h+1)),int(rng.integers(W-w+1)),h,w))
    candidates=[];overflow=[]
    for window in sorted(windows):
        _,record=record_crop(masks,annotation_ids,valid,window,affine,'category_planner')
        if queries is not None and record['query_capacity_requirement']>queries:overflow.append(record)
        candidates.append(record)
    selected=[];counts={k:0 for k in CATEGORIES}
    for record in candidates:
        if any(counts[k]<quotas[k] for k in record['categories']):
            selected.append(record)
            for k in record['categories']:counts[k]+=1
    missing={k:quotas[k]-counts[k] for k in quotas if counts[k]<quotas[k]}
    selected_overflow=[r for r in selected if queries is not None and r['query_capacity_requirement']>queries]
    return {'status':'FAIL' if missing or selected_overflow else 'PASS','selected':selected,'category_counts':counts,'unmet_quotas':missing,
        'unfittable_native_crowns':does_not_fit,'capacity_overflow_candidates':len(overflow),'selected_capacity_overflows':len(selected_overflow),
        'category_semantics':'overlapping tags, quotas are minima, no assumption of disjoint strata; small uses original area; no biological completeness claim'}
