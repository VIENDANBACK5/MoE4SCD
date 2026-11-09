"""Actual-output checks, not a green proxy for original-method completion."""
from pathlib import Path
import sys,json,gzip,csv,hashlib,time
import numpy as np
from pycocotools import mask as mu
sys.path.insert(0,'/home/chung/RS/data/TreeCrown-MultiSource-V1')
from treecrown_data import TreeCrownData
from treecrown_data.masks import unpack
EXP=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for q in iter(lambda:f.read(2**20),b''):h.update(q)
    return h.hexdigest()
def bind_paths(d):
    out={}
    if not isinstance(d,dict):return out
    for key in ['path','checkpoint']:
        if isinstance(d.get(key),str) and d.get('sha256'):out[d[key]]=d['sha256']
    if d.get('predictor_checkpoint') and d.get('predictor_sha256'):out[d['predictor_checkpoint']]=d['predictor_sha256']
    for v in d.values():
        if isinstance(v,dict):out.update(bind_paths(v))
    return out
def main():
    start=time.monotonic();freeze=read(EXP/'PREDICTIONS_FROZEN.json');cohort=read(EXP/'metadata/SELECTED_COHORT.json')['selected']
    D=TreeCrownData.from_root('/home/chung/RS/data/TreeCrown-MultiSource-V1')
    view=D.open(source='bamforests',split='train',track='legacy_reference',access_role='inference')
    ind={r['image_id']:i for i,r in enumerate(view.rows)};valids={};inputs={}
    for row in cohort:
        s=view[ind[row['image_id']]];assert s['instance_masks'] is None and not D._coco
        valids[row['image_id']]=mu.encode(np.asfortranarray(s['valid_mask'].astype(np.uint8)))
        inputs[row['image_id']]=(hashlib.sha256(s['rgb'].tobytes()).hexdigest(),hashlib.sha256(s['valid_mask'].tobytes()).hexdigest())
    checkpoints={};rawcount=finalcount=forwards=0
    for rel,h in freeze['sha256'].items():
        p=EXP/rel;assert sha(p)==h
        with gzip.open(p,'rt') as f:d=json.load(f)
        assert d['RGB_only'] and not d['GT_inference'] and not d['annotations_read'] and not d['resize_applied']
        assert (d['RGB_sha256'],d['valid_sha256'])==inputs[d['source_image_id']]
        rr=[unpack(q['rle']) for q in d['raw_predictions']]
        ar=mu.area(rr);bb=mu.toBbox(rr)
        origins=np.array([q['window_yx'] for q in d['raw_predictions']]);xy=origins[:,::-1]
        assert np.all(ar==[q['area'] for q in d['raw_predictions']]) and np.all(ar>0)
        assert np.all(bb[:,:2]>=xy) and np.all(bb[:,:2]+bb[:,2:]<=xy+1024)
        coverage=mu.iou(rr,[valids[d['source_image_id']]],[1])[:,0]
        assert np.allclose(coverage,1.,atol=1e-12),'Saved mask leaks into invalid alpha'
        pre=[j for j,q in enumerate(d['raw_predictions']) if q['score']>=.05 and q['area']>=30]
        assert pre==d['pre_fusion_indices']
        keep=set(d['final_indices']);removed=set(d['suppressed_indices'])
        assert not keep&removed and keep|removed==set(pre)
        checkpoints.update(bind_paths(d['checkpoint']))
        rawcount+=len(rr);finalcount+=len(keep);forwards+=d['forwards']
    assert forwards==1944 and len(freeze['sha256'])==216 and not D._coco
    for p,h in checkpoints.items():assert sha(p)==h,('Historical checkpoint changed',p)
    with (EXP/'results/METRICS.csv').open() as f:metrics=list(csv.DictReader(f))
    for r in metrics:
        n=int(r['GT']);tp=int(r['TP50']);fp=int(r['FP50']);pred=int(r['predictions'])
        assert n==1106 and pred==tp+fp and int(r['FN50'])==n-tp
        assert abs(float(r['Recall50'])-100*tp/n)<1e-10
        assert not r['original_source_small_Recall50']
    assert read(EXP/'results/VIEW_SUPPORT_SUMMARY.json')['all_raw_masks_respect_support_bound']
    assert all(p.exists() for p in [EXP/'figures/FIXED_CROSS_CARRIER_COMPARISON.png',EXP/'RESEARCH_REPORT.md',EXP/'SCIENTIFIC_DECISION.json'])
    files=[p for p in EXP.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='FINAL_VERIFICATION.json']
    total=sum(p.stat().st_size for p in files);assert total<2*2**30
    out=dict(status='VERIFIED_BOUNDED_EXTERNAL_REFERENCE_FIDELITY_STUDY',raw_masks_checked=rawcount,final_masks_checked=finalcount,images_per_arm=24,systems=9,forwards=forwards,source_split='TRAIN_ONLY',all_masks_alpha_valid=True,all_masks_confined_to_declared_view=True,all_input_pixels_identical_across_arms=True,source_checkpoints_unchanged=checkpoints,original_small_metric='NOT_ESTIMABLE',independent_annotation_stability='NOT_ESTABLISHED',original_method_supported=False,overall_goal_complete=False,bytes=total,verification_seconds=time.monotonic()-start,files={str(p.relative_to(EXP)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
    (EXP/'FINAL_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['files','source_checkpoints_unchanged']}))
if __name__=='__main__':main()
