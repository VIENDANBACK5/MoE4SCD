"""Fresh read-only reference scan, reusing A00 spatial/RLE machinery in a new destination."""
import collections
import csv
import importlib.util
import io
import json
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from .core import REPO, affine_from_bounds, area, canonical_hash, require, rle, sha256, size_bin, write_new

BASE=Path(__file__).resolve().parent
OLD=BASE.parents[1]/'execution_plan_audit_2026-10-15'
PREP=REPO/'data/itc_benchmarks/canopyrs_quebectrees/prepared'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def main():
    dest=BASE/'source_scan';dest.mkdir(exist_ok=False);(dest/'registry').mkdir()
    audit=load('audit',OLD/'audit.py');audit.OUT=dest
    # The existing data scan reads original COCO + all valid/priority masks; no prediction input.
    audit.data_audit()
    border=load('v2_border',OLD/'border_audit.py');border.main()
    cross=load('v2_cross_grid',OLD/'cross_grid_audit.py');cross.main()
    issues=[];bins=[];summaries=[];registry=[];raw_hashes={}
    for split in ('train','validation','test'):
        coco=json.loads((PREP/f'{split}_coco.json').read_text());ims={i['file_name']:i for i in coco['images']}
        require(len(ims)==len(coco['images']),"Duplicate image filenames")
        require(len({i['id'] for i in coco['images']})==len(ims),"Duplicate image IDs")
        require(len({a['id'] for a in coco['annotations']})==len(coco['annotations']),"Duplicate annotation IDs")
        imageids={i['id'] for i in ims.values()};by=collections.defaultdict(list)
        for a in coco['annotations']:
            require(a['image_id'] in imageids,"Orphan annotation");by[a['image_id']].append(a)
        nraw=0;errors=0
        for shard in sorted(PREP.parent.glob(f'{split}-*.parquet')):
            raw_hashes[str(shard.relative_to(REPO))]={'sha256':sha256(shard),'bytes':shard.stat().st_size}
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=1):
                for source in batch.to_pylist():
                    im=ims[source['tile_name']];nraw+=1
                    path=PREP/split/f"{Path(im['file_name']).stem}.npz"
                    with np.load(path) as z:
                        rgb=z['image'];valid=z['valid'];label=z['label']
                    raw=np.array(Image.open(io.BytesIO(source['image']['bytes'])))
                    source_valid=raw[...,3]>0 if raw.shape[-1]==4 else raw[...,:3].sum(-1)>0
                    checks={'raw_RGB_alignment':np.array_equal(raw[...,:3].astype(np.uint8),rgb),
                            'raw_valid_alignment':np.array_equal(source_valid,valid),
                            'image_valid_label_dimensions':rgb.shape[:2]==valid.shape==label.shape==(im['height'],im['width']),
                            'binary_valid_mask':bool(np.isin(valid,(0,1)).all()),
                            'nonnegative_integer_owner':bool(np.issubdtype(label.dtype,np.integer) and label.min()>=0),
                            'raw_bounds_alignment':np.allclose(im['bounds'],source['tile_metadata']['bounds'],rtol=0,atol=1e-8)}
                    aa=by[im['id']];source_aa=source['annotations'];require(len(aa)==len(source_aa['segmentation']),"Raw/prepared annotation count differs")
                    transform=affine_from_bounds(im['bounds'],valid.shape)
                    for j,ann in enumerate(aa):
                        rr=rle(ann['segmentation'],valid.shape)
                        rawseg=source_aa['segmentation'][j]
                        checks['raw_annotation_'+str(j)]=(canonical_hash(ann['segmentation'])==canonical_hash(rawseg)
                            and ann['area']==float(source_aa['area'][j]) and np.array_equal(ann['bbox'],source_aa['bbox'][j]))
                        canonical_area=area(rr,transform);native_area=float(ann['area'])*abs(transform[0]*transform[4])
                        if size_bin(canonical_area)!=size_bin(native_area):bins.append({'split':split,'annotation_id':ann['id'],'RLE_area_m2':canonical_area,'native_area_m2':native_area,'RLE_bin':size_bin(canonical_area),'native_bin':size_bin(native_area)})
                        if not np.isclose(float(ann['area']),float(audit.mu.area(rr)),atol=1e-6,rtol=0):issues.append({'split':split,'annotation_id':ann['id'],'flag':'NATIVE_RLE_AREA_DIFFERENCE','severity':'INFO','action':'preserve both'})
                        if not np.allclose(ann['bbox'],audit.mu.toBbox(rr),atol=1e-6,rtol=0):issues.append({'split':split,'annotation_id':ann['id'],'flag':'NATIVE_RLE_BBOX_DIFFERENCE','severity':'INFO','action':'preserve both'})
                    failed=[k for k,v in checks.items() if not v];errors+=len(failed)
                    registry.append({'tile_key':f"{split}:{im['id']}",'split':split,'role':{'train':'TRAIN','validation':'DEVELOPMENT','test':'LEGACY_TEST'}[split],
                        'scene_id':im['raster_name'],'mosaic_id':im['raster_name'],'site_id':None,'survey_id':None,'acquisition_time':None,
                        'source_path':str(shard),'source_sha256':raw_hashes[str(shard.relative_to(REPO))]['sha256'],
                        'prepared_path':str(path),'prepared_sha256':sha256(path),'crs':im['crs'],'affine':transform,'gsd':[abs(transform[0]),abs(transform[4])],
                        'valid_mask_sha256':canonical_hash({'shape':valid.shape,'bytes_sha256':__import__('hashlib').sha256(valid.tobytes()).hexdigest()}),
                        'valid_footprint_representation':'prepared NPZ valid mask; bounds are enclosing rectangle','annotation_ids':[a['id'] for a in aa],
                        'lineage':['Parquet native image/RLE','prepare_canopyrs.py','prepared RGB/valid/first-wins priority'],
                        'annotation_policy':'dataset_native_RLE; upstream polygon protocol unknown','license':None,'provenance_status':'prepared/raw correspondence verified; survey/license UNKNOWN','failed_checks':failed})
                    if nraw%25==0:print('raw source verified',split,nraw,flush=True)
        require(nraw==len(ims),"Raw shard coverage differs from prepared tile count")
        summaries.append({'split':split,'images':len(ims),'annotations':len(coco['annotations']),'raw_tiles_compared':nraw,'critical_mismatches':errors})
    audit.csvout(dest/'annotation_flags.csv',issues);audit.csvout(dest/'area_bin_changes.csv',bins)
    write_new(dest/'provenance_registry.json',{'version':'2.0.0','tiles':registry})
    write_new(dest/'raw_source_hashes.json',raw_hashes)
    write_new(dest/'SOURCE_VERIFICATION.json',{'status':'PASS' if not any(s['critical_mismatches'] for s in summaries) else 'FAIL',
        'splits':summaries,'bin_changes':len(bins),'flags':len(issues),'original_vector_polygon_roundtrip':'UNKNOWN: raw source supplies RLE, not pre-rasterization vectors',
        'biological_reference_validation':'UNKNOWN','mode':'read-only metadata/reference integrity; no model predictions/performance'})
    require(not any(s['critical_mismatches'] for s in summaries),"Raw/prepared correspondence mismatch")

if __name__=='__main__':main()
