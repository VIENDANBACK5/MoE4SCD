"""Explicit native→proposal→accepted→scene lineage; policies supplied by caller."""
from dataclasses import dataclass,replace
import hashlib
import numpy as np
from .core import binary, canonical_hash, pixel_area, require

@dataclass(frozen=True)
class Instance:
    id:str
    mask:np.ndarray
    score:float
    native_id:str
    parents:tuple
    stage:int
    tile_id:str
    active:bool=True

def validate(instances,shape,stage,native_registry):
    ids=[x.id for x in instances];require(len(ids)==len(set(ids)),"Duplicate instance IDs in declared scope")
    for x in instances:
        binary(x.mask,shape);require(bool(x.mask.any()),"Empty instance mask")
        require(x.stage==stage and bool(x.id) and bool(x.tile_id),"Stage/scope missing")
        require(np.isfinite(x.score) and x.native_id in native_registry,"Missing native provenance/nonfinite score")
        source=native_registry[x.native_id]
        require(bool(source.get('input_sha256')) and bool(source.get('config_sha256')) and source.get('stage')==0,"Incomplete Stage0 lineage")
        require(len(x.parents)>0,"Missing parent IDs")
        if stage>=2:require(x.active,"Inactive query cannot be accepted")
    return instances

def materialize_q(mask_logits,objectness,no_object,tile_id,native_id,materialization_threshold,mask_logit_threshold):
    logits=np.asarray(mask_logits,float);obj=np.asarray(objectness,float);no=np.asarray(no_object,float)
    require(logits.ndim==3 and obj.shape==no.shape==(len(logits),),"Q logits/objectness/no-object mismatch")
    require(np.isfinite(logits).all() and np.isfinite(obj).all() and np.isfinite(no).all(),"Nonfinite query output")
    require(((obj>=0)&(obj<=1)&(no>=0)&(no<=1)).all(),"Query probabilities outside [0,1]")
    require(0<=materialization_threshold<=1 and np.isfinite(mask_logit_threshold),"Explicit Q materialization policy required")
    result=[]
    for i in range(len(logits)):
        m=logits[i]>mask_logit_threshold
        if obj[i]>=materialization_threshold and obj[i]>no[i] and m.any():
            result.append(Instance(f'{tile_id}:q{i}',m,float(obj[i]),native_id,(f'{native_id}:slot{i}',),1,tile_id))
    return result

def accept(proposals,valid,affine,threshold,min_area_m2,duplicate_policy,invalid_support_policy,native_registry):
    valid=binary(valid);validate(proposals,valid.shape,1,native_registry);px=pixel_area(affine)
    require(np.isfinite(threshold) and np.isfinite(min_area_m2) and min_area_m2>=0,"Explicit selection threshold/area required")
    require(duplicate_policy in ('keep','exact_geometry_highest_score'),"Undeclared duplicate policy")
    require(invalid_support_policy in ('keep_original','intersect_valid','reject_any_invalid'),"Invalid-support policy missing")
    out=[];seen=set()
    key=lambda x:(-x.score,hashlib.sha256(x.mask.tobytes()).hexdigest(),x.id)
    for x in sorted(proposals,key=key):
        if not x.active or x.score<threshold:continue
        m=x.mask.copy()
        if invalid_support_policy=='reject_any_invalid' and (m&~valid).any():continue
        if invalid_support_policy=='intersect_valid':m&=valid
        if not (m&valid).any() or int(m.sum())*px<min_area_m2:continue
        digest=hashlib.sha256(m.tobytes()).hexdigest()
        if duplicate_policy=='exact_geometry_highest_score' and digest in seen:continue
        seen.add(digest);out.append(replace(x,id=f'{x.id}:accepted',mask=m,parents=(x.id,),stage=2))
    validate(out,valid.shape,2,native_registry);return out

def stitch(tile_sets,scene_id,scene_shape,duplicate_iou,merge_policy,native_registry,scene_affine=None):
    """Same-grid translation only. Ambiguous disjoint edge fragments remain separate, flagged."""
    require(bool(scene_id) and 0<duplicate_iou<=1,"Explicit scene/association threshold required")
    require(merge_policy in ('union','highest_score'),"Explicit scene geometry policy required")
    require(scene_affine is not None,"Scene affine required for global coordinate validation")
    pixel_area(scene_affine)
    placed=[]
    for tile in tile_sets:
        require(tile.get('crs')==tile_sets[0].get('crs') and tile.get('crs'),"Scene CRS mismatch")
        y,x=tile['origin_yx'];require(isinstance(y,int) and isinstance(x,int) and min(y,x)>=0,"Integer same-grid origin required")
        a,b,c,d,e,f=scene_affine
        expected=(a,b,c+a*x+b*y,d,e,f+d*x+e*y)
        require('affine' in tile and np.allclose(tile['affine'],expected,rtol=0,atol=1e-8),"Tile transform/origin/GSD inconsistent with scene grid")
        shape=tile['shape'];validate(tile['instances'],shape,2,native_registry)
        require(y+shape[0]<=scene_shape[0] and x+shape[1]<=scene_shape[1],"Tile outside declared scene boundary")
        for inst in tile['instances']:
            m=np.zeros(scene_shape,bool);m[y:y+shape[0],x:x+shape[1]]=inst.mask
            placed.append((inst,m))
    placed.sort(key=lambda p:(hashlib.sha256(p[1].tobytes()).hexdigest(),p[0].tile_id,p[0].id))
    parent=list(range(len(placed)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,(a,ma) in enumerate(placed):
        for j,(b,mb) in enumerate(placed[:i]):
            if a.tile_id!=b.tile_id and (ma&mb).sum()/max(int((ma|mb).sum()),1)>=duplicate_iou:parent[root(i)]=root(j)
    result=[]
    for k in sorted({root(i) for i in range(len(placed))}):
        members=[placed[i] for i in range(len(placed)) if root(i)==k]
        best=sorted(members,key=lambda p:(-p[0].score,hashlib.sha256(p[1].tobytes()).hexdigest(),p[0].id))[0]
        mask=np.logical_or.reduce([m for _,m in members]) if merge_policy=='union' else best[1]
        sid=canonical_hash([scene_id,list(scene_shape),hashlib.sha256(mask.tobytes()).hexdigest()])[:24]
        result.append({'id':sid,'stage':3,'scene_id':scene_id,'mask':mask,'score':best[0].score,
                       'parents':sorted(p.id for p,m in members),'native_ids':sorted({p.native_id for p,m in members}),
                       'edge_fragments_policy':'disjoint/low-overlap fragments retained separately; no inferred biological association',
                       'association_policy':{'duplicate_iou':duplicate_iou,'merge_policy':merge_policy}})
    require(len({x['id'] for x in result})==len(result),"Scene IDs collide; unresolved duplicate clusters")
    return result
