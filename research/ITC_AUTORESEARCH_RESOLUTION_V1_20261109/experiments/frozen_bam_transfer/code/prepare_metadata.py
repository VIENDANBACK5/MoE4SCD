"""TRAIN-only TIFF headers; no RGB decode, labels, TEST members or model scores."""
from pathlib import Path
import sys,json,csv,zipfile,hashlib,math
import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

EXP=Path(__file__).resolve().parents[1]
sys.path.insert(0,'/home/chung/RS/data/TreeCrown-MultiSource-V1')
from treecrown_data import TreeCrownData

def main():
    D=TreeCrownData.from_root('/home/chung/RS/data/TreeCrown-MultiSource-V1')
    view=D.open(source='bamforests',split='train',track='legacy_reference',access_role='inference')
    dst=EXP/'metadata';dst.mkdir(parents=True,exist_ok=True)
    assert not (dst/'SELECTED_COHORT.json').exists()
    records=[]
    assert len({r['container'] for r in view.rows})==1
    archive=D.resolve('bamforests',view.rows[0]['container'])
    with zipfile.ZipFile(archive) as z:
        for row in view.rows:
            assert row['split']=='train' and 'train2023' in Path(row['member']).parts
            assert 'test' not in row['member'].lower()
            with z.open(row['member']) as f:
                im=Image.open(f);tags=im.tag_v2
                assert im.size==(2048,2048) and im.mode=='RGBA'
                assert 34264 not in tags,'Do not silently reinterpret rotated coordinates'
                gx,gy,_=tags[33550];tie=tags[33922];keys=tags[34735]
                keymap={keys[k]:keys[k+3] for k in range(4,len(keys),4) if keys[k+1]==0}
                assert keymap[3072]==25832 and keymap[3076]==9001
                assert gx>0 and gy>0 and math.isclose(gx,gy,rel_tol=1e-6)
                x0=tie[3]-tie[0]*gx;y1=tie[4]+tie[1]*gy
            records.append(dict(image_id=row['image_id'],source_image_id=row['source_image_id'],
                member=row['member'],split='train',width=2048,height=2048,crs='EPSG:25832',
                gx_m=gx,gy_m=gy,xmin=x0,ymin=y1-2048*gy,xmax=x0+2048*gx,ymax=y1,
                center_x=x0+1024*gx,center_y=y1-1024*gy,zip_CRC=z.getinfo(row['member']).CRC,
                source_row=row,header_only=True))
    xy=np.array([[r['center_x'],r['center_y']] for r in records])
    distance=np.linalg.norm(xy[:,None,:]-xy[None,:,:],axis=2)
    n,group=connected_components(csr_matrix(distance<=2500),directed=False)
    assert 2<=n<=3,('Unexpected sampling-frame grouping; stop before predictions',n)
    selected=[];stats=[]
    for g in range(n):
        pool=[r for r,k in zip(records,group) if k==g]
        pool.sort(key=lambda r:hashlib.sha256(('20261109:'+r['image_id']).encode()).hexdigest())
        chosen=[]
        for row in pool:
            def gap(q):
                dx=max(row['xmin']-q['xmax'],q['xmin']-row['xmax'],0)
                dy=max(row['ymin']-q['ymax'],q['ymin']-row['ymax'],0)
                return math.hypot(dx,dy)
            if all(gap(q)>=25 for q in chosen):
                chosen.append(dict(row,spatial_group=f'geo_component_{g:02d}'))
            if len(chosen)==12:break
        assert len(chosen)==12,('Insufficient spatial support for fixed frame',g,len(chosen))
        selected.extend(chosen)
        stats.append(dict(spatial_group=f'geo_component_{g:02d}',train_tiles=len(pool),selected=len(chosen),
                          filename_prefixes=sorted({r['source_image_id'].split('_')[0] for r in pool}),
                          gx_range_m=[min(r['gx_m'] for r in pool),max(r['gx_m'] for r in pool)]))
    plain=[{k:v for k,v in r.items() if k!='source_row'} for r in records]
    with (dst/'TRAIN_HEADER_FRAME.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(plain[0]));wr.writeheader();wr.writerows(plain)
    cohort=dict(status='METADATA_SELECTED_BEFORE_PREDICTION',seed=20261109,spatial_group_radius_m=2500,
                minimum_between_footprint_gap_m=25,groups=stats,selected=selected,
                source_annotation_area_available=False,independent_annotators_available=False,
                independent_geographic_generalization_established=False,
                permitted_role='TRAIN-only external reference-fidelity evaluation; not a new reference-stability study')
    (dst/'SELECTED_COHORT.json').write_text(json.dumps(cohort,indent=2)+'\n')
    print(json.dumps(dict(header_records=len(records),selected=len(selected),groups=stats,RGB_pixels_decoded=False,annotations_read=False)))
if __name__=='__main__':main()
