"""Footprint policies and exact same-grid valid support checks; no implicit erosion."""
import itertools
import numpy as np
from .core import binary,pixel_area,require

def footprint(tile):
    from shapely.geometry import Polygon
    a,b,c,d,e,f=tile['affine'];h,w=tile['shape']
    pixel_area(tile['affine'])
    return Polygon([(a*x+b*y+c,d*x+e*y+f) for x,y in [(0,0),(w,0),(w,h),(0,h)]])

def validate_split(tiles,policy,mode):
    require(policy.get('version') and policy.get('support') in ('full_footprint','valid_pixels'),"Versioned spatial support policy required")
    buffer=policy.get('buffer_m');require(isinstance(buffer,(int,float)) and np.isfinite(buffer) and buffer>=0,"Explicit nonnegative spatial buffer required")
    if mode=='confirmatory':require(bool(policy.get('buffer_justification')) and policy.get('approved') is True,"Confirmatory buffer justification/approval missing")
    edges=[];unverified=[]
    for a,b in itertools.combinations(tiles,2):
        if a['role']==b['role']:continue
        if a['crs']!=b['crs']:unverified.append([a['id'],b['id'],'CRS mismatch']);continue
        pa,pb=footprint(a),footprint(b);distance=float(pa.distance(pb));overlap=float(pa.intersection(pb).area)
        shared=None
        if policy['support']=='valid_pixels':
            va=binary(a['valid'],a['shape']);vb=binary(b['valid'],b['shape'])
            aa,bb=np.array(a['affine']),np.array(b['affine'])
            linear=np.array([[aa[0],aa[1]],[aa[3],aa[4]]]);offset=np.linalg.solve(linear,np.array([bb[2]-aa[2],bb[5]-aa[5]]))
            if not np.allclose(aa[[0,1,3,4]],bb[[0,1,3,4]],rtol=0,atol=1e-9) or not np.allclose(offset,np.rint(offset),rtol=0,atol=1e-5):
                if overlap>0 or distance<buffer:unverified.append([a['id'],b['id'],'Cross-grid valid support requires exact ground intersection scan'])
                continue
            x,y=np.rint(offset).astype(int);H,W=va.shape;h,w=vb.shape
            ya,yb=max(0,y),min(H,y+h);xa,xb=max(0,x),min(W,x+w)
            shared=int((va[ya:yb,xa:xb]&vb[ya-y:yb-y,xa-x:xb-x]).sum()) if ya<yb and xa<xb else 0
            # Full footprints are a conservative proxy for buffered distance. Do not certify precise valid-boundary buffer when nodata could change it.
            if buffer>0 and distance<buffer and not overlap:unverified.append([a['id'],b['id'],'Valid-support buffer distance not established; use full_footprint conservative policy'])
        violates_overlap=(shared>0 if shared is not None else overlap>0)
        same_source=bool(a.get('mosaic_id')) and a.get('mosaic_id')==b.get('mosaic_id')
        if violates_overlap or distance<buffer:edges.append({'a':a['id'],'b':b['id'],'footprint_overlap_m2':overlap,
            'same_mosaic_metadata':same_source,'aligned_valid_ground_cells':shared,
            'shared_source_grid_pixels':shared if same_source else None,'footprint_distance_m':distance,
            'RGB_byte_identity':'not inferred from ground/grid intersection'})
    allow=mode=='exploratory' and policy.get('allow_development_overlap') is True
    require(not allow or bool(policy.get('exception_justification')),"Exploratory spatial exception needs justification")
    return {'status':'UNKNOWN' if unverified else ('WARN' if edges and allow else 'FAIL' if edges else 'PASS'),
            'edges':edges,'unverified':unverified,'policy':policy,'note':'No claim of spatial independence follows solely from pixel-disjointness.'}

def valid_union_area(tiles,max_rectangles):
    """Exact union of affine cell footprints; row strips reduce rectangular-mask cost."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from .core import canonical_hash,metric_crs
    require(tiles and max_rectangles>0,"Valid footprints and explicit complexity budget required")
    polygons=[];sources=[];crs=tiles[0]['crs'];metric_crs(crs)
    for tile in tiles:
        require(tile['crs']==crs,"Union requires common projected CRS")
        valid=binary(tile['valid']);a,b,c,d,e,f=tile['affine'];pixel_area(tile['affine']);active={}
        def close(run,y0,y1):
            x0,x1=run;points=[(a*x+b*y+c,d*x+e*y+f) for x,y in ((x0,y0),(x1,y0),(x1,y1),(x0,y1))]
            polygons.append(Polygon(points));require(len(polygons)<=max_rectangles,"Exact union exceeds declared budget; do not substitute summed tile area")
        for y,row in enumerate(valid):
            delta=np.diff(np.pad(row.astype(np.int8),1));runs=set(zip(np.where(delta==1)[0].tolist(),np.where(delta==-1)[0].tolist()))
            for run in list(active):
                if run not in runs:close(run,active.pop(run),y)
            for run in runs:active.setdefault(run,y)
        for run,start in active.items():close(run,start,len(valid))
        sources.append(canonical_hash({'affine':tile['affine'],'shape':valid.shape,'valid_bytes_hash':__import__('hashlib').sha256(valid.tobytes()).hexdigest()}))
    geometry=unary_union(polygons)
    return {'method':'exact_affine_cell_union','deduplicated':True,'valid_union_area_m2':float(geometry.area),'crs':crs,
            'input_hashes':sources,'geometry_wkb_sha256':__import__('hashlib').sha256(geometry.wkb).hexdigest(),
            'claim_limit':'Geometric ground support only; biological GT association and spatial independence not established.'}
