"""Validated reference primitives and safe artifact I/O shared by runtime and tests."""
from __future__ import annotations
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import numpy as np
from pycocotools import mask as mu

REPO = Path(__file__).resolve().parents[3]
VERSION = "2.0.0"

class IntegrityError(ValueError):
    pass

def require(condition, message):
    if not condition:
        raise IntegrityError(message)

def sha256(path):
    path=Path(path);require(path.is_file(),f"Missing source: {path}")
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def write_new(path, value):
    """Exclusive creation: never overwrite prior scientific evidence."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def source_key(path):
    """Stable repository alias while checking the actual target against registered read roots."""
    path=Path(path)
    logical=Path(os.path.abspath(REPO/path if not path.is_absolute() else path));physical=logical.resolve()
    if physical.is_relative_to(REPO):return str(physical.relative_to(REPO))
    policy=Path(__file__).resolve().parent/'READ_ROOTS.json'
    aliases=json.loads(policy.read_text())['aliases'] if policy.is_file() else {}
    for alias,target in sorted(aliases.items(),key=lambda kv:-len(kv[0])):
        root=Path(target)
        if physical.is_relative_to(root):
            candidate=Path(alias)/physical.relative_to(root)
            require((REPO/candidate).resolve()==physical,"Source alias was redirected")
            return str(candidate)
    raise IntegrityError(f"Source outside registered physical read roots: {physical}")

def verify_manifest(manifest, root=REPO, required_paths=None):
    require(isinstance(manifest,dict) and bool(manifest),"Empty/missing source manifest")
    if required_paths is not None:require(set(required_paths)<=set(manifest),"Incomplete source manifest")
    verified=[];root=Path(root).resolve()
    for rel,record in manifest.items():
        path=(root/rel).resolve()
        if root==REPO:require(source_key(path)==str(Path(rel)),f"Noncanonical source alias: {rel}")
        else:require(path.is_relative_to(root),f"Source outside declared root: {path}")
        require(isinstance(record,dict),f"Missing digest record: {rel}")
        digest=record.get('sha256')
        require(isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),f"Invalid digest: {rel}")
        require(sha256(path)==digest,f"Source hash changed: {rel}")
        verified.append(str(path))
    return verified

def binary(mask, shape=None):
    mask=np.asarray(mask)
    require(mask.ndim==2 and (shape is None or mask.shape==tuple(shape)),"Mask coordinate dimensions mismatch")
    require(np.isfinite(mask).all() and np.isin(mask,(0,1)).all(),"Mask must be finite binary")
    return mask.astype(bool,copy=False)

def rle(r, shape=None):
    require(isinstance(r,dict) and set(('size','counts'))<=r.keys(),"RLE fields missing")
    size=r['size'];require(len(size)==2 and all(isinstance(v,(int,np.integer)) and v>0 for v in size),"Invalid RLE dimensions")
    if shape is not None:require(list(size)==list(shape),"RLE/image shape mismatch")
    raw=r['counts']
    if isinstance(raw,list):require(all(isinstance(v,(int,np.integer)) and not isinstance(v,bool) and v>=0 for v in raw),"RLE counts must be nonnegative integers")
    else:
        require(isinstance(raw,(str,bytes)),"Unsupported RLE counts type")
        encoded=raw.encode() if isinstance(raw,str) else raw
        require(all(48<=c<=111 for c in encoded),"Invalid compressed RLE alphabet")
    # Reuse the repository's existing compressed-count parser; validate its output before C extension use.
    parse=importlib.import_module('crown_segmentation_research.diagnosis_2026-10-11.fgq_revision_2026-10-13.grouping').rle_counts
    try:counts=parse(r)
    except Exception as exc:raise IntegrityError(f"Invalid RLE encoding: {exc}") from exc
    require(len(counts)>0 and (counts>=0).all() and int(counts.sum())==math.prod(size),"RLE runs exceed/miss image support")
    if isinstance(r['counts'],list):return mu.frPyObjects(r,*size)
    return {'size':list(size),'counts':r['counts'].encode() if isinstance(r['counts'],str) else r['counts']}

def encode(mask):return mu.encode(np.asfortranarray(binary(mask).astype(np.uint8)))

def pixel_area(affine):
    values=np.asarray(affine,float);require(values.shape==(6,) and np.isfinite(values).all(),"Finite six-coefficient affine required")
    a,b,_,d,e,_=values;det=abs(a*e-b*d)
    require(det>0,"Degenerate pixel transform")
    return float(det)

def metric_crs(crs):
    from pyproj import CRS
    try:parsed=CRS.from_user_input(crs)
    except Exception as exc:raise IntegrityError(f"Unknown CRS: {crs}") from exc
    require(parsed.is_projected and len(parsed.axis_info)>=2 and all(np.isclose(axis.unit_conversion_factor,1.,rtol=0,atol=1e-12) for axis in parsed.axis_info[:2]),"Physical metrics require projected metre axes; degrees/feet need an explicit transform")
    return parsed.to_string()

def affine_from_bounds(bounds, shape):
    xmin,ymin,xmax,ymax=map(float,bounds);h,w=shape
    require(h>0 and w>0 and xmax>xmin and ymax>ymin,"Invalid georeferenced bounds")
    out=((xmax-xmin)/w,0.,xmin,0.,-(ymax-ymin)/h,ymax);pixel_area(out);return out

def resized_affine(affine, old_shape, new_shape):
    pixel_area(affine);oh,ow=old_shape;nh,nw=new_shape
    require(min(oh,ow,nh,nw)>0,"Positive resize dimensions required")
    a,b,c,d,e,f=affine
    return (a*ow/nw,b*oh/nh,c,d*ow/nw,e*oh/nh,f)

def area(r, affine):return int(mu.area(rle(r)))*pixel_area(affine)

def size_bin(value):
    require(np.isfinite(value) and value>=0,"Nonnegative finite physical area required")
    return ('lt1','1to2','2to4','4to9','ge9')[int(np.searchsorted([1.,2.,4.,9.],value,side='right'))]

def supports(masks, valid, ids, policy='legacy_first_wins'):
    valid=binary(valid);masks=[binary(m,valid.shape) for m in masks]
    require(len(ids)==len(masks) and len(set(ids))==len(ids),"Stable distinct annotation IDs required")
    require(policy in ('legacy_first_wins','stable_id_first_wins','area_ascending_experimental'),"Undeclared ownership policy")
    coverage=np.zeros(valid.shape,np.int32)
    for mask in masks:coverage+=mask
    order=list(range(len(masks)))
    if policy=='stable_id_first_wins':order.sort(key=lambda i:ids[i])
    if policy=='area_ascending_experimental':order.sort(key=lambda i:(int(masks[i].sum()),ids[i]))
    owner=np.full(valid.shape,-1,np.int64)
    for i in order:owner[masks[i]&(owner<0)]=i
    priority=[owner==i for i in range(len(masks))]
    return {'original':masks,'valid':[m&valid for m in masks],
            'unique_original':[m&(coverage==1) for m in masks],
            'unique_valid':[m&(coverage==1)&valid for m in masks],
            'priority':priority,'priority_valid':[m&valid for m in priority],
            'owner_index':owner,'annotation_ids':list(ids),'policy':policy}

def rng_for(master_seed, epoch, split, tile, crop, fixed=False):
    require(isinstance(epoch,int) and epoch>=0,"Epoch must be nonnegative")
    effective=0 if fixed or split!='train' else epoch
    digest=canonical_hash([int(master_seed),effective,str(split),str(tile),int(crop)])
    return np.random.default_rng(int(digest[:16],16))

def check_capacity(count, queries):
    require(isinstance(count,int) and count>=0 and isinstance(queries,int) and queries>0,"Invalid target/query count")
    require(count<=queries,f"Query overflow: {count} visible targets > {queries}; no silent target truncation")

def check_access(role, purpose):
    allowed={'training':{'TRAIN'},'preprocessing_fit':{'TRAIN'},'model_selection':{'DEVELOPMENT'},
             'threshold_selection':{'DEVELOPMENT','CALIBRATION'},'calibration':{'CALIBRATION'},
             'evaluation':{'DEVELOPMENT','CONFIRMATORY_TEST'},'integrity_scan':{'TRAIN','DEVELOPMENT','CALIBRATION','CONFIRMATORY_TEST','LEGACY_TEST'}}
    require(purpose in allowed and role in allowed[purpose],f"Forbidden dataset role {role} for {purpose}")
