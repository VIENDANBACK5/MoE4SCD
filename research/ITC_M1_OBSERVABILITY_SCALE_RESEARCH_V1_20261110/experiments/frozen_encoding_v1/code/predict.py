"""Frozen RGB-only encoding contrasts; no reference-mask loader."""
from pathlib import Path
import os,sys,json,gzip,hashlib,time,resource,shutil,traceback
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
EXP=Path(__file__).resolve().parents[1];REPO=EXP.parents[3]
OLD=REPO/'research/ITC_MASK_DINO_FORMATION_V1_20261104'
sys.path.insert(0,str(OLD));import bootstrap
import numpy as np,torch
import torch.nn.functional as F
from carrier import build_carrier,predict_logits
from pycocotools import mask as mu
sys.path.insert(0,str(REPO/'research/ITC_RR_BASELINE_RELEASE_V1_20261109/code'))
from rr_component import install
START=time.monotonic()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.suffix=='.gz':
        with gzip.open(p,'wt') as f:json.dump(d,f,allow_nan=False)
    else:p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def guard():
    assert time.monotonic()-START<1200,'20minute bound'
    assert shutil.disk_usage(EXP).free>=5*2**30,'5GiB free-disk floor'
    assert torch.cuda.max_memory_allocated()<=8*2**30,'8GiB GPU bound'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<8*2**20,'8GiB RSS bound'
    assert sum(p.stat().st_size for p in EXP.rglob('*') if p.is_file())<192*2**20,'192MiB artifact bound'
def pack(m):
    r=mu.encode(np.asfortranarray(m.astype(np.uint8)))
    return dict(size=r['size'],counts=r['counts'].decode())
def select(raw):
    pre=[i for i,p in enumerate(raw) if p['score']>=.05 and p['area']>=30]
    keep=[];suppressed=[]
    for i in sorted(pre,key=lambda i:(-raw[i]['score'],i)):
        if keep and mu.iou([dict(size=raw[i]['rle']['size'],counts=raw[i]['rle']['counts'].encode())],
                           [dict(size=raw[j]['rle']['size'],counts=raw[j]['rle']['counts'].encode()) for j in keep],[0]*len(keep)).max()>.5:suppressed.append(i)
        else:keep.append(i)
    return pre,keep,suppressed

@torch.no_grad()
def main():
    assert not (EXP/'PREDICTIONS_FROZEN.json').exists(),'Never overwrite a complete run'
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.18)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    guard()
    inherited=read(EXP/'INHERITED_CACHE_BINDINGS.json')
    checkpoint=OLD/'checkpoints/mask_only_seed42/final.pt'
    assert sha(checkpoint)==inherited['sha256'] and sha(OLD/'carrier.py')==inherited['carrier_sha256']
    rows=read(EXP/'COHORT.json');canonical={r['id']:r for r in read(OLD/'TRAIN_WINDOWS.json')}
    assert len(rows)==3 and len({r['source'] for r in rows})==3
    assert all(r==canonical[r['id']] and r['eligible'] and not r['shared_cells'] and r['image_id'].startswith('selvamask::train:') for r in rows)
    model,_,_=build_carrier(seed=42,pretrained=False)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True);del ck
    model.eval().requires_grad_(False);install(model.sem_seg_head.predictor,'native')
    cap={}
    def before_backbone(module,args):cap['tensor']=args[0].detach().cpu()
    hook=model.backbone.register_forward_pre_hook(before_backbone)
    binds={r['window_id']:r for r in inherited['window_bindings']}
    hashes={};proofs=[];forward_count=0
    for row in rows:
        guard();rid=row['id'];folder=OLD/'train_cache'/row['tile'];b=binds[rid]
        assert sha(folder/'rgb.npy')==b['rgb_sha256'] and sha(folder/'valid.npy')==b['valid_sha256']
        y,x,h,w=row['window_yxhw'];assert (h,w)==(1024,1024)
        rgb=np.array(np.load(folder/'rgb.npy',mmap_mode='r')[y:y+h,x:x+w])
        valid=np.array(np.load(folder/'valid.npy',mmap_mode='r')[y:y+h,x:x+w]).astype(bool)
        image=rgb.astype(np.float32)
        count=valid.reshape(512,2,512,2).sum((1,3))
        total=(image*valid[:,:,None]).reshape(512,2,512,2,3).sum((1,3))
        coarse=(total/np.maximum(count,1)[:,:,None]).astype(np.float32)
        cv=count==4;common=np.repeat(np.repeat(cv,2,0),2,1)
        rep=lambda a:np.repeat(np.repeat(a,2,0),2,1)
        arrays=dict(A=image,B=rep(image),C=coarse,D=rep(coarse))
        assert np.array_equal(arrays['B'][::2,::2],arrays['A']) and np.array_equal(arrays['D'][::2,::2],arrays['C'])
        np.savez_compressed(EXP/f'{rid}_inputs.npz',RGB=rgb,original_valid=valid,common_valid=common,coarse_RGB=coarse,coarse_valid=cv)
        tensors={}
        for arm,a in arrays.items():
            guard();dest=EXP/'predictions'/f'{rid}_{arm}.json.gz';assert not dest.exists(),'Partial attempt requires explicit audit, not silent rerun'
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t0=time.monotonic()
            t=torch.from_numpy(a.transpose(2,0,1).copy()).cuda()
            torch.manual_seed(42);out=predict_logits(model,[t]);forward_count+=1
            norm=cap.pop('tensor');assert tuple(norm.shape)==(1,3,a.shape[0],a.shape[1])
            tensors[arm]=norm
            logits=out['pred_logits'][0].float();assert logits.shape==(300,1) and torch.isfinite(logits).all()
            cls=logits.sigmoid()[:,0];mask_logits=out['pred_masks'][0]
            trace=model.sem_seg_head.predictor.site_trace
            assert len(trace['indices'][0])==300
            vv=torch.from_numpy(common).cuda();raw=[]
            for j in range(0,300,8):
                z=F.interpolate(mask_logits[j:j+8][None].float(),(1024,1024),mode='bilinear',align_corners=False)[0]
                assert torch.isfinite(z).all()
                mm=(z>0)&vv;areas=mm.flatten(1).sum(1)
                quality=(z.sigmoid()*mm).flatten(1).sum(1)/(areas+1e-6)
                for k in range(len(mm)):
                    q=j+k
                    raw.append(dict(query_id=q,encoder_site=trace['indices'][0][q],class_logit=float(logits[q,0]),
                                    class_probability=float(cls[q]),mask_quality=float(quality[k]),score=float(cls[q]*quality[k]),
                                    area=int(areas[k]),rle=pack(mm[k].cpu().numpy()),
                                    predicted_box_cxcywh=out['pred_boxes'][0,q].float().cpu().tolist()))
            pre,kept,supp=select(raw);torch.cuda.synchronize()
            payload=dict(window_id=rid,source=row['source'],arm=arm,raw_predictions=raw,accepted_indices=pre,final_indices=kept,
                         suppressed_indices=supp,input_hw=list(a.shape[:2]),canonical_hw=[1024,1024],
                         mask_logit_grid_hw=list(mask_logits.shape[-2:]),encoder_trace=trace,
                         RGB_input_sha256=ah(a),backbone_normalized_input_sha256=ah(norm.numpy()),
                         common_valid_sha256=ah(common),original_valid_pixels=int(valid.sum()),common_valid_pixels=int(common.sum()),
                         removed_original_valid_pixels=int((valid&~common).sum()),checkpoint_sha256=inherited['sha256'],
                         protocol_sha256=sha(EXP/'protocol.md'),cohort_sha256=sha(EXP/'COHORT.json'),
                         RGB_only=True,GT_inference=False,reference_masks_read=False,queries=300,parameters_updated=False,
                         dense_mask_logits_stored=False,seconds=time.monotonic()-t0,
                         peak_GPU_allocated_MiB=torch.cuda.max_memory_allocated()/2**20)
            write(dest,payload);hashes[str(dest.relative_to(EXP))]=sha(dest)
            print(json.dumps(dict(window=rid,arm=arm,input_hw=payload['input_hw'],raw=300,final=len(kept),
                                 seconds=payload['seconds'],peakMiB=payload['peak_GPU_allocated_MiB'])),flush=True)
            del out,t,mask_logits,z,mm
        assert torch.equal(tensors['B'][:,:,::2,::2],tensors['A'])
        assert torch.equal(tensors['D'][:,:,::2,::2],tensors['C'])
        proofs.append(dict(window_id=rid,RGB_left_inverse_exact=True,actual_backbone_tensor_left_inverse_exact=True,
                           tensor_shapes={k:list(v.shape) for k,v in tensors.items()}))
        del tensors
    hook.remove();assert forward_count==12 and all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert sha(checkpoint)==inherited['sha256']
    write(EXP/'PREDICTIONS_FROZEN.json',dict(stage='ALL_RGB_ONLY_PREDICTIONS_FROZEN_BEFORE_NEW_REFERENCE_MEASUREMENTS',
        windows=3,forwards=12,sha256=hashes,exact_left_inverse_proofs=proofs,checkpoint_sha256=inherited['sha256'],
        protocol_sha256=sha(EXP/'protocol.md'),code_sha256=sha(__file__),weights_unchanged=True,
        new_training=False,TEST=False,SAME6=False,protected_review=False,seconds=time.monotonic()-START))
    guard();print('COMPLETE: frozen12forwards',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        (EXP/'attempts').mkdir(exist_ok=True);(EXP/'attempts'/f'{time.time_ns()}.txt').write_text(traceback.format_exc());raise
