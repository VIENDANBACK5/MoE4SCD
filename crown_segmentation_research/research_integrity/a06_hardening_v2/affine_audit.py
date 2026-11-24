"""Bind runtime reference to actual raw affine, rather than reconstructing it from rounded bounds."""
import csv
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from pycocotools import mask as mu
from .core import REPO,metric_crs,pixel_area,rle,sha256,size_bin,write_new

HERE=Path(__file__).resolve().parent
PREP=REPO/'data/itc_benchmarks/canopyrs_quebectrees/prepared'

def main():
    scan=HERE/'source_scan';source=json.loads((scan/'SOURCE_VERIFICATION_BOUND.json').read_text())
    registry=json.loads((scan/'provenance_registry.json').read_text());rawhashes=json.loads((scan/'raw_source_hashes.json').read_text())
    bypath={t['prepared_path']:t for t in registry['tiles']};native={};errors=[];changes=[]
    for split in ('train','validation','test'):
        coco=json.loads((PREP/f'{split}_coco.json').read_text());byid={i['id']:i for i in coco['images']}
        for p in sorted(PREP.parent.glob(f'{split}-*.parquet')):
            assert sha256(p)==rawhashes[str(p.relative_to(REPO))]['sha256']
            for batch in pq.ParquetFile(p).iter_batches(batch_size=32,columns=['tile_name','tile_metadata']):
                for raw in batch.to_pylist():
                    meta=raw['tile_metadata'];path=str(PREP/split/(Path(raw['tile_name']).stem+'.npz'))
                    transform=meta['transform'];assert len(transform)==9 and np.allclose(transform[6:],[0,0,1])
                    affine=transform[:6];metric_crs(meta['crs']);px=pixel_area(affine)
                    h,w=meta['height'],meta['width'];a,b,c,d,e,f=affine
                    corners=np.array([(a*x+b*y+c,d*x+e*y+f) for x,y in ((0,0),(w,0),(w,h),(0,h))])
                    bounds=[corners[:,0].min(),corners[:,1].min(),corners[:,0].max(),corners[:,1].max()]
                    if not np.allclose(bounds,meta['bounds'],rtol=0,atol=1e-8):errors.append(path)
                    native[str(Path(path).relative_to(REPO))]={'affine':affine,'crs':meta['crs'],'shape':[h,w],
                        'evidence':'raw Parquet tile_metadata.transform; no date/site inference','raw_source':str(p.relative_to(REPO))}
                    t=bypath[path];t['bounds_reconstructed_affine']=t['affine'];t['affine']=affine;t['gsd']=[float(np.hypot(a,d)),float(np.hypot(b,e))]
                    t['affine_evidence']='raw Parquet tile_metadata.transform'
        for ann in coco['annotations']:
            im=byid[ann['image_id']];path=str((PREP/split/(Path(im['file_name']).stem+'.npz')).relative_to(REPO))
            px=pixel_area(native[path]['affine']);canonical=int(mu.area(rle(ann['segmentation'])))*px;metadata=float(ann['area'])*px
            if size_bin(canonical)!=size_bin(metadata):changes.append({'split':split,'annotation_id':ann['id'],'canonical_area_m2':canonical,'metadata_area_m2':metadata,'canonical_bin':size_bin(canonical),'metadata_bin':size_bin(metadata)})
    source['native_images']=native;source['raw_source_inputs']=rawhashes
    source['six_support_verification_path']=str(scan/'SIX_SUPPORTS_VERIFICATION.json')
    source['actual_affine_status']='PASS' if not errors else 'FAIL';source['actual_affine_errors']=errors
    source['status']='PASS' if source['status']=='PASS' and not errors and json.loads((scan/'SIX_SUPPORTS_VERIFICATION.json').read_text())['status']=='PASS' else 'FAIL'
    write_new(scan/'SOURCE_VERIFICATION_RUNTIME.json',source);write_new(HERE/'PROVENANCE_REGISTRY.json',registry)
    with (scan/'AREA_EXACT_AFFINE_CHANGES.csv').open('x',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(changes[0]));writer.writeheader();writer.writerows(changes)
    write_new(scan/'AFFINE_VERIFICATION.json',{'status':'PASS' if not errors else 'FAIL','tiles':len(native),'bounds_transform_mismatches':errors,
        'area_bin_changes':len(changes),'definition':'Native RLE count times determinant of actual Parquet affine; original metadata retained',
        'original_coco_modified':False})

if __name__=='__main__':main()
