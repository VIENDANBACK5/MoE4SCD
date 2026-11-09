"""Privileged one-view support upper bounds; no new prediction or inference policy."""
from pathlib import Path
import sys,json,gzip,csv
import numpy as np
from pycocotools import mask as mu
sys.path.insert(0,'/home/chung/RS/data/TreeCrown-MultiSource-V1')
from treecrown_data.masks import pack,unpack
EXP=Path(__file__).resolve().parents[1]
def main():
    assert (EXP/'results/COMPLETE.json').exists()
    with gzip.open(EXP/'results/TRAIN_REFERENCE_MASKS.json.gz','rt') as f:d=json.load(f)
    with (EXP/'results/PER_REFERENCE.csv').open() as f:rr=list(csv.DictReader(f))
    with (EXP/'results/CROSS_CARRIER_PATTERNS.csv').open() as f:patterns=list(csv.DictReader(f))
    per={}
    for q in rr:per.setdefault((int(q['tile']),q['reference_id']),[]).append(q)
    pattern={(int(q['tile']),q['reference_id']):q for q in patterns}
    out=[];rects=[]
    for y in [0,753,1024]:
        for x in [0,753,1024]:
            a=np.zeros((2048,2048),np.uint8);a[y:y+1024,x:x+1024]=1
            rects.append((y,x,mu.encode(np.asfortranarray(a))))
    for i,targets in enumerate(d['references']):
        for t in targets:
            ref=unpack(t['rle']);area=int(mu.area(ref));assert area>0
            fractions=[int(mu.area(mu.merge([ref,r],intersect=True)))/area for y,x,r in rects]
            u=max(fractions);best=int(np.argmax(fractions));records=per[(i,t['original_id'])]
            assert all(float(q['raw_best_iou'])<=u+1e-6 for q in records),'An actual mask violates the view-support bound'
            pat=pattern[(i,t['original_id'])]
            out.append(dict(tile=i,reference_id=t['original_id'],spatial_group=pat['spatial_group'],valid_target_pixels=area,
                U=u,best_view_y=rects[best][0],best_view_x=rects[best][1],view_support_impossible50=u<.5,
                view_support_impossible75=u<.75,all_three_raw_missing=pat['all_raw_missing']=='True',
                final_pattern_R5_M2F_DINO=pat['pattern_R5_M2F_DINO'],
                RGB_learnability_established=False,original_source_area_known=False))
    with (EXP/'results/VIEW_SUPPORT_BOUNDS.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(out[0]));wr.writeheader();wr.writerows(out)
    groups=sorted({q['spatial_group'] for q in out});summary=[]
    for g in ['ALL']+groups:
        a=[q for q in out if g=='ALL' or q['spatial_group']==g]
        summary.append(dict(group=g,references=len(a),support_impossible50=sum(q['view_support_impossible50'] for q in a),
            support_impossible75=sum(q['view_support_impossible75'] for q in a),
            all_three_raw_missing=sum(q['all_three_raw_missing'] for q in a),
            raw_common_missing_with_U50=sum(q['all_three_raw_missing'] and q['view_support_impossible50'] for q in a),
            final_common_omission=sum(q['final_pattern_R5_M2F_DINO']=='000' for q in a)))
    (EXP/'results/VIEW_SUPPORT_SUMMARY.json').write_text(json.dumps(dict(status='REGISTERED_LATE_DIAGNOSTIC_NOT_DEPLOYABLE',all_raw_masks_respect_support_bound=True,groups=summary),indent=2)+'\n')
    print(json.dumps(summary))
if __name__=='__main__':main()
