"""All six supports + exact legacy first-wins replay on every native annotation."""
import collections
import csv
import importlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from .core import REPO,write_new

HERE=Path(__file__).resolve().parent
PREP=REPO/'data/itc_benchmarks/canopyrs_quebectrees/prepared'
AUDIT=importlib.import_module('crown_segmentation_research.execution_plan_audit_2026-10-15.audit')

def inspect(task):
    split,im,anns=task;path=PREP/split/f"{Path(im['file_name']).stem}.npz"
    with np.load(path) as z:valid=z['valid'].astype(bool);label=z['label']
    h,w=valid.shape;events=np.zeros(h*w+1,np.int32);runs=[]
    for ann in anns:
        s,e=AUDIT.mask_runs(ann['segmentation']);runs.append((s,e));np.add.at(events,s,1);np.add.at(events,e,-1)
    coverage=np.cumsum(events[:-1]);del events
    unique=np.r_[0,np.cumsum(coverage==1,dtype=np.int32)]
    uv=np.r_[0,np.cumsum((coverage==1)&valid.ravel(order='F'),dtype=np.int32)]
    vp=AUDIT.prefix(valid)
    counts=np.bincount(label.ravel(),minlength=len(anns)+1);cv=np.bincount(label[valid],minlength=len(anns)+1)
    owner=np.zeros(h*w,np.int32);rows=[]
    for j,(ann,(s,e)) in enumerate(zip(anns,runs),1):
        for lo,hi in zip(s,e):
            sub=owner[lo:hi];sub[sub==0]=j
        original=int((e-s).sum());v=int((vp[e]-vp[s]).sum());u=int((unique[e]-unique[s]).sum());uvalid=int((uv[e]-uv[s]).sum())
        flags=[]
        if original-v:flags.append('OUTSIDE_VALID_SUPPORT')
        if u==0:flags.append('EMPTY_UNIQUE_ORIGINAL')
        if uvalid==0:flags.append('EMPTY_UNIQUE_VALID')
        if counts[j]==0:flags.append('EMPTY_PRIORITY_ORIGINAL')
        if cv[j]==0:flags.append('EMPTY_PRIORITY_VALID')
        if original-u:flags.append('OVERLAPPING_GT')
        rows.append({'split':split,'image_id':im['id'],'annotation_id':ann['id'],'original_pixels':original,'valid_pixels':v,
            'unique_original_pixels':u,'unique_valid_pixels':uvalid,'priority_pixels':int(counts[j]),'priority_valid_pixels':int(cv[j]),
            'flags':'|'.join(flags),'primary_matching_reference':'original_RLE_retained','priority_policy':'legacy_first_wins_in_original_annotation_order'})
    return rows,{'split':split,'image_id':im['id'],'priority_replay_mismatch_pixels':int((owner.reshape((h,w),order='F')!=label).sum())}

def main():
    tasks=[]
    for split in ('train','validation','test'):
        gt=json.loads((PREP/f'{split}_coco.json').read_text());by=collections.defaultdict(list)
        for a in gt['annotations']:by[a['image_id']].append(a)
        tasks.extend((split,i,by[i['id']]) for i in gt['images'])
    rows=[];tiles=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for n,(rr,t) in enumerate(pool.map(inspect,tasks),1):
            rows.extend(rr);tiles.append(t)
            if n%50==0:print('six-support tiles',n,flush=True)
    with (HERE/'source_scan/SIX_SUPPORTS.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for split in ('train','validation','test'):
        rr=[r for r in rows if r['split']==split]
        summary.append({'split':split,'annotations':len(rr),'outside_valid':sum(r['original_pixels']>r['valid_pixels'] for r in rr),
            'unique_original_empty':sum(r['unique_original_pixels']==0 for r in rr),'unique_valid_empty':sum(r['unique_valid_pixels']==0 for r in rr),
            'priority_empty':sum(r['priority_pixels']==0 for r in rr),'priority_valid_empty':sum(r['priority_valid_pixels']==0 for r in rr)})
    mismatch=sum(t['priority_replay_mismatch_pixels'] for t in tiles)
    write_new(HERE/'source_scan/SIX_SUPPORTS_VERIFICATION.json',{'status':'PASS' if not mismatch else 'FAIL','tiles':len(tiles),'annotations':len(rows),
        'priority_replay_mismatch_pixels':mismatch,'summary':summary,'per_tile_replay':tiles})
    if mismatch:raise RuntimeError('Prepared priority differs from exact original first-wins replay')

if __name__=='__main__':main()
