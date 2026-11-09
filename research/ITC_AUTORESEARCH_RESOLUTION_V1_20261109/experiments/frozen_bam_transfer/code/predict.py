"""Frozen RGB/alpha-only BAM TRAIN inference. Labels are withheld by source API."""
from pathlib import Path
import os,sys,json,gzip,hashlib,time,argparse,shutil,resource,importlib.util,traceback
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='4'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
EXP=Path(__file__).resolve().parents[1]; REPO=EXP.parents[3]
R=REPO/'research'; DINO=R/'ITC_MASK_DINO_FORMATION_V1_20261104'
M2F=R/'ITC_CARRIER_FORMATION_V1_20261103'; RELEASE=R/'ITC_RR_BASELINE_RELEASE_V1_20261109'
SEL=R/'ITC_ENCODER_SELECTION_CONTROLS_V1_20261108'
sys.path.insert(0,'/home/chung/RS/data/TreeCrown-MultiSource-V1')
import torch,numpy as np
import torch.nn.functional as F
from pycocotools import mask as mu
from treecrown_data import TreeCrownData
from treecrown_data.masks import pack,unpack

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp')
    if p.name.endswith('.gz'):
        with gzip.open(q,'wt',encoding='utf-8') as f:json.dump(d,f,allow_nan=False)
    else:q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    q.replace(p)
def paste(mask,y,x,H,W):
    h,w=mask.shape;ix=np.flatnonzero(mask.T)
    if not len(ix):counts=[H*W]
    else:
        z=(ix//h+x)*H+(ix%h+y);beg=z[np.r_[True,np.diff(z)>1]];end=z[np.r_[np.diff(z)>1,True]]+1
        counts=np.empty(2*len(beg)+1,dtype=np.int64);counts[0]=beg[0];counts[2:-1:2]=beg[1:]-end[:-1];counts[1::2]=end-beg;counts[-1]=H*W-end[-1]
        if counts[-1]==0:counts=counts[:-1]
        counts=counts.tolist()
    return pack(mu.frPyObjects(dict(size=[H,W],counts=counts),H,W))
def select(raw):
    pre=[i for i,p in enumerate(raw) if p['score']>=.05 and p['area']>=30]
    keep=[];removed=[]
    for i in sorted(pre,key=lambda i:(-raw[i]['score'],raw[i]['candidate_id'])):
        if keep and np.max(mu.iou([unpack(raw[i]['rle'])],[unpack(raw[k]['rle']) for k in keep],[0]*len(keep)))>.5:removed.append(i)
        else:keep.append(i)
    return pre,keep,removed

def build(kind,seed):
    if kind=='m2f':
        sys.path.insert(0,str(M2F));import bootstrap
        from carrier import build_carrier,predict_logits
        cp=M2F/f'checkpoints/fp32_uniform_seed{seed}/final.pt'
        expected=read(M2F/f'logs/fp32_uniform_seed{seed}/STATUS.json')['checkpoint_sha256']
        assert sha(cp)==expected
        m,_,_=build_carrier(seed=seed,pretrained=False)
        ck=torch.load(cp,map_location='cpu',weights_only=False);m.load_state_dict(ck['model'],strict=True);del ck
        binding=dict(arm='M2F_FP32_UNIFORM',seed=seed,path=str(cp),sha256=expected,queries=192)
        query_predict=predict_logits
    elif kind=='r5':
        p=R/'method_development/R5_INSTANCE_METHOD/model.py'
        spec=importlib.util.spec_from_file_location('archived_r5_model',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        m=mod.build();cp=p.parent/'checkpoints/final.pt';expected=read(p.parent/'TRAINING_STATUS.json')['checkpoint_sha256'];assert sha(cp)==expected
        ck=torch.load(cp,map_location='cpu',weights_only=False);m.load_state_dict(ck['model'],strict=True);del ck;m.cuda()
        binding=dict(arm='R5_NATIVE_MATCHED_VIEWS',seed=42,path=str(cp),sha256=expected,returned_ROI_limit=256)
        query_predict=None
    else:
        sys.path.insert(0,str(RELEASE/'code'));from release import load as load_rr
        from rr_component import install
        if kind=='rr':m,binding=load_rr(seed)
        elif kind=='control':
            sys.path.insert(0,str(SEL/'code'));from shared import load_arm
            m,binding=load_arm(seed,'NATIVE_MASK_INIT')
        else:
            sys.path.insert(0,str(DINO));import bootstrap
            from carrier import build_carrier
            cp=DINO/f'checkpoints/mask_only_seed{seed}/final.pt';expected=read(DINO/f'logs/mask_only_seed{seed}/STATUS.json')['checkpoint_sha256'];assert sha(cp)==expected
            m,_,_=build_carrier(seed=seed,pretrained=False);ck=torch.load(cp,map_location='cpu',weights_only=False);m.load_state_dict(ck['model'],strict=True);del ck
            install(m.sem_seg_head.predictor,'native')
            binding=dict(arm='DINO_ORIGINAL_MASK_ONLY',seed=seed,path=str(cp),sha256=expected,queries=300)
        from carrier import predict_logits
        query_predict=predict_logits
    m.eval().requires_grad_(False)
    assert all(not p.requires_grad for p in m.parameters())
    return m,binding,query_predict

def guard(start):
    assert time.time()-start<2700,'45 minute experiment deadline'
    assert shutil.disk_usage(EXP).free>=20*2**30,'Registered20GiBdiskreserve'
    assert torch.cuda.max_memory_allocated()<8*2**30,'8GiBallocatedGPUlimit'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<8*2**30/1024,'8GiBhostRSSlimit'

@torch.no_grad()
def predict_tile(model,qp,kind,rgb,valid):
    H,W=valid.shape;assert (H,W)==(2048,2048)
    raw=[];traces=[]
    for y in [0,753,1024]:
        for x in [0,753,1024]:
            image=np.ascontiguousarray(rgb[y:y+1024,x:x+1024].transpose(2,0,1))
            vv=valid[y:y+1024,x:x+1024];t=torch.from_numpy(image).float().cuda()
            if kind=='r5':
                out=model([t/255.])[0]
                assert out['masks'].shape[-2:]==(1024,1024)
                for j in range(len(out['scores'])):
                    mask=(out['masks'][j,0]>.5).cpu().numpy()&vv
                    if mask.any():raw.append(dict(candidate_id=f'{y}_{x}_roi{j}',query_id=j,window_yx=[y,x],score=float(out['scores'][j]),area=int(mask.sum()),rle=paste(mask,y,x,H,W),predicted_box_xyxy=(out['boxes'][j].cpu().numpy()+np.array([x,y,x,y])).tolist()))
            else:
                out=qp(model,[t]);score=out['pred_logits'][0].float()
                cls=score.softmax(-1)[:,0] if kind=='m2f' else score.sigmoid()[:,0]
                assert len(cls)==(192 if kind=='m2f' else 300)
                trace=getattr(model.sem_seg_head.predictor,'site_trace',None)
                if trace is not None:traces.append(dict(window_yx=[y,x],selected_sites=trace['indices'][0]))
                v=torch.from_numpy(vv.copy()).cuda()
                for j in range(0,len(cls),16):
                    z=F.interpolate(out['pred_masks'][:,j:j+16].float(),(1024,1024),mode='bilinear',align_corners=False)[0]
                    mm=(z>0)&v;areas=mm.flatten(1).sum(1);scores=cls[j:j+16]*(z.sigmoid()*mm).flatten(1).sum(1)/(areas+1e-6)
                    for k in range(len(mm)):
                        if not int(areas[k]):continue
                        q=j+k;d=dict(candidate_id=f'{y}_{x}_query{q}',query_id=q,window_yx=[y,x],score=float(scores[k]),class_probability=float(cls[q]),area=int(areas[k]),rle=paste(mm[k].cpu().numpy(),y,x,H,W))
                        if trace is not None:d['encoder_site']=trace['indices'][0][q]
                        if 'pred_boxes' in out:d['predicted_box_cxcywh']=(out['pred_boxes'][0,q].float().cpu().numpy()*1024+np.array([x,y,0,0])).tolist()
                        raw.append(d)
            del out,t
    pre,kept,removed=select(raw)
    return dict(raw_predictions=raw,pre_fusion_indices=pre,final_indices=kept,suppressed_indices=removed,
                views=traces,forwards=9,input_hw=[H,W],RGB_only=True,GT_inference=False,resize_applied=False,independent_masks=True)

def main(kind,seed):
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.18)
    torch.backends.cudnn.benchmark=False
    start=float(os.environ.get('BAM_STUDY_START',time.time()));guard(start)
    cohort=read(EXP/'metadata/SELECTED_COHORT.json');rows=cohort['selected']
    arm=f'{kind}_seed{seed}';folder=EXP/'predictions'/arm;folder.mkdir(parents=True,exist_ok=True)
    model,binding,qp=build(kind,seed)
    D=TreeCrownData.from_root('/home/chung/RS/data/TreeCrown-MultiSource-V1')
    view=D.open(source='bamforests',split='train',track='legacy_reference',access_role='inference')
    indices={r['image_id']:i for i,r in enumerate(view.rows)}
    protocol_hash=sha(EXP/'protocol.md');cohort_hash=sha(EXP/'metadata/SELECTED_COHORT.json')
    for i,row in enumerate(rows):
        guard(start);dest=folder/f'b{i:03d}.json.gz'
        if dest.exists():
            with gzip.open(dest,'rt') as f:old=json.load(f)
            assert old['protocol_sha256']==protocol_hash and old['cohort_sha256']==cohort_hash and old['checkpoint']==binding and old['source_image_id']==row['image_id']
            continue
        s=view[indices[row['image_id']]]
        assert s['instance_masks'] is None and s['annotation_records'] is None and s['original_instance_ids'] is None and not D._coco
        assert s['provenance']['label_access']=='WITHHELD_FOR_INFERENCE'
        rgb=s['rgb'];valid=s['valid_mask'];rgbhash=hashlib.sha256(rgb.tobytes()).hexdigest();vhash=hashlib.sha256(valid.tobytes()).hexdigest()
        torch.cuda.synchronize();t0=time.monotonic();torch.cuda.reset_peak_memory_stats()
        out=predict_tile(model,qp,kind,rgb,valid);torch.cuda.synchronize()
        out.update(arm=arm,source_image_id=row['image_id'],spatial_group=row['spatial_group'],gx_m=row['gx_m'],gy_m=row['gy_m'],checkpoint=binding,protocol_sha256=protocol_hash,cohort_sha256=cohort_hash,RGB_sha256=rgbhash,valid_sha256=vhash,seconds=time.monotonic()-t0,peak_GPU_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,annotations_read=False)
        assert all(p.grad is None for p in model.parameters()) and not D._coco
        write(dest,out)
        print(json.dumps(dict(arm=arm,tile=i,raw=len(out['raw_predictions']),final=len(out['final_indices']),seconds=out['seconds'])),flush=True)
    write(folder/'COMPLETE.json',dict(arm=arm,tiles=len(rows),forwards=len(rows)*9,checkpoint=binding,protocol_sha256=protocol_hash,cohort_sha256=cohort_hash,RGB_only=True,annotations_read=False,TEST=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',required=True,choices=['rr','control','dino','r5','m2f']);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    try:main(a.kind,a.seed)
    except Exception:
        q=EXP/'attempts';q.mkdir(exist_ok=True);(q/f'{a.kind}_{a.seed}_{time.time_ns()}.txt').write_text(traceback.format_exc());raise
