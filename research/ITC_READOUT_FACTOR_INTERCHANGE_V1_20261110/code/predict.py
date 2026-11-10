"""Passive extraction and fixed final-readout swaps. No reference loader."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
from common import *
import sys,time,resource,shutil,traceback,copy,importlib.util
sys.path.insert(0,str(OLD));import bootstrap
import torch,numpy as np
import torch.nn.functional as F
from carrier import build_carrier,predict_logits
spec=importlib.util.spec_from_file_location('native_materializer',PILOT/'code/predict.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
START=time.monotonic()
def guard():
    assert time.monotonic()-START<1200
    assert shutil.disk_usage(EXP).free>=4*2**30
    assert torch.cuda.max_memory_allocated()<=8*2**30
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<8*2**20
    assert sum(q.stat().st_size for q in EXP.rglob('*') if q.is_file())<=96*2**20
def ah(t):return core.ah(t.detach().cpu().contiguous().numpy())
def materialize(e,field,logits,boxes,trace,valid,source_raw=None):
    dense=torch.einsum('bqc,bchw->bqhw',e,field)[0]
    assert torch.isfinite(dense).all()
    cls=logits.sigmoid()[:,0];raw=[];vv=torch.from_numpy(valid).cuda()
    for j in range(0,300,8):
        z=F.interpolate(dense[j:j+8][None].float(),(1024,1024),mode='bilinear',align_corners=False)[0]
        mm=(z>0)&vv;area=mm.flatten(1).sum(1)
        quality=(z.sigmoid()*mm).flatten(1).sum(1)/(area+1e-6)
        for k in range(len(mm)):
            q=j+k;s=float(cls[q]*quality[k])
            raw.append(dict(query_id=q,encoder_site=trace['indices'][0][q],class_logit=float(logits[q,0]),
                       class_probability=float(cls[q]),mask_quality=float(quality[k]),score=s,
                       native_query_score=s if source_raw is None else source_raw[q]['score'],
                       area=int(area[k]),rle=core.pack(mm[k].cpu().numpy()),predicted_box_cxcywh=boxes[q].tolist()))
    pre,keep,supp=core.select(raw)
    alt=[dict(q,score=q['native_query_score']) for q in raw];pre0,keep0,supp0=core.select(alt)
    return dict(raw_predictions=raw,accepted_indices=pre,final_indices=keep,suppressed_indices=supp,
                accepted_indices_native_score=pre0,final_indices_native_score=keep0,suppressed_indices_native_score=supp0)

@torch.no_grad()
def main():
    assert not (EXP/'PREDICTIONS_FROZEN.json').exists()
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.18)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    binding=read(EXP/'BINDINGS.json');cohort=read(PILOT/'COHORT.json');hashes={};checks=[];forwards=0
    for arm in ['S0','S1']:
        checkpoint=CONTROL/f'checkpoints/{arm}/final.pt';assert sha(checkpoint)==binding['checkpoints'][arm]
        model,_,_=build_carrier(seed=42,pretrained=False)
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True);del ck
        model.eval().requires_grad_(False);core.install(model.sem_seg_head.predictor,'native');cap={}
        def features_hook(module,args):cap['field']=args[1].detach()
        def embedding_hook(module,args,out):cap['embedding']=out.detach();cap['embedding_calls']=cap.get('embedding_calls',0)+1
        hf=model.sem_seg_head.predictor.register_forward_pre_hook(features_hook)
        he=model.sem_seg_head.predictor.mask_embed.register_forward_hook(embedding_hook)
        for row in cohort:
            guard();path=PILOT/f"{row['id']}_inputs.npz";inp=np.load(path);valid=inp['common_valid'];image=inp['RGB'].astype(np.float32)
            arrays={'A':image,'B':np.repeat(np.repeat(image,2,0),2,1)};states={};outputs={}
            for obs,a in arrays.items():
                cap.clear();torch.manual_seed(42);t0=time.monotonic();t=torch.from_numpy(a.transpose(2,0,1).copy()).cuda()
                out=predict_logits(model,[t]);forwards+=1
                e=cap['embedding'];field=cap['field'];assert tuple(e.shape)==(1,300,256) and field.shape[:2]==(1,256)
                assert torch.isfinite(e).all() and torch.isfinite(field).all()
                reconstructed=torch.einsum('bqc,bchw->bqhw',e,field)
                assert torch.equal(reconstructed,out['pred_masks']),'Native reconstruction must be exact'
                # Exact paired sign change is the declared algebraic gauge control.
                assert torch.equal(torch.einsum('bqc,bchw->bqhw',-e,-field),reconstructed)
                del reconstructed
                logits=out['pred_logits'][0].float();boxes=out['pred_boxes'][0].float().cpu();trace=copy.deepcopy(model.sem_seg_head.predictor.site_trace)
                payload=materialize(e,field,logits,boxes,trace,valid)
                previous=gz(CONTROL/f"evaluations/{arm}/predictions/{row['id']}_{obs}.json.gz")
                assert previous['RGB_input_sha256']==core.ah(a) and previous['common_valid_sha256']==core.ah(valid)
                for p,q in zip(payload['raw_predictions'],previous['raw_predictions']):
                    for key in ['query_id','encoder_site','rle','area','score','class_logit','class_probability','mask_quality','predicted_box_cxcywh']:
                        assert p[key]==q[key],(arm,row['id'],obs,key)
                for key in ['accepted_indices','final_indices','suppressed_indices']:assert payload[key]==previous[key]
                states[obs]=dict(e=e,field=field,logits=logits,boxes=boxes,trace=trace,
                                  e_sha=ah(e),field_sha=ah(field),rgb_sha=core.ah(a),field_shape=list(field.shape),
                                  seconds=time.monotonic()-t0,embedding_calls=cap['embedding_calls'])
                outputs[obs+obs]=payload
                checks.append(dict(arm=arm,window_id=row['id'],observation=obs,native_logits_exact=True,native_raw_and_final_reproduced=True,
                            paired_sign_gauge_native_logits_exact=True,RGB_input_sha256=states[obs]['rgb_sha'],
                            embedding_shape=list(e.shape),field_shape=list(field.shape),embedding_sha256=states[obs]['e_sha'],field_sha256=states[obs]['field_sha']))
                del out,t;guard()
            for left,right in [('A','B'),('B','A')]:
                s=states[left];outputs[left+right]=materialize(s['e'],states[right]['field'],s['logits'],s['boxes'],s['trace'],valid,outputs[left+left]['raw_predictions'])
            (EXP/'embeddings'/arm).mkdir(parents=True,exist_ok=True)
            np.savez_compressed(EXP/f"embeddings/{arm}/{row['id']}.npz",E_A=states['A']['e'].cpu().numpy(),E_B=states['B']['e'].cpu().numpy())
            for code,payload in outputs.items():
                left,right=code;s=states[left]
                payload.update(dict(training_arm=arm,arm=code,window_id=row['id'],source=row['source'],embedding_source=left,field_source=right,
                      checkpoint_sha256=binding['checkpoints'][arm],RGB_only=True,GT_inference=False,reference_masks_read=False,
                      queries=300,parameters_updated=False,canonical_hw=[1024,1024],mask_logit_grid_hw=states[right]['field_shape'][-2:],
                      encoder_trace=s['trace'],common_valid_sha256=core.ah(valid),input_file_sha256=sha(path),
                      embedding_source_RGB_sha256=s['rgb_sha'],field_source_RGB_sha256=states[right]['rgb_sha'],
                      embedding_sha256=s['e_sha'],pixel_field_sha256=states[right]['field_sha'],
                      dense_pixel_field_stored=False,dense_mask_logits_stored=False,protocol_sha256=sha(EXP/'protocol.md')))
                dest=EXP/f"predictions/{arm}/{row['id']}_{code}.json.gz";write(dest,payload);hashes[str(dest.relative_to(EXP))]=sha(dest)
                print(arm,row['id'],code,len(payload['final_indices']),flush=True)
            del states,outputs,cap,inp,e,field,logits,boxes,trace,s,payload;cap={};torch.cuda.empty_cache();guard()
        hf.remove();he.remove()
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        assert sha(checkpoint)==binding['checkpoints'][arm]
        del model;torch.cuda.empty_cache()
    assert forwards==12 and len(hashes)==24
    write(EXP/'PREDICTIONS_FROZEN.json',dict(stage='ALL_INTERCHANGE_PREDICTIONS_FROZEN_BEFORE_NEW_REFERENCE_JOIN',forwards=12,pools=24,queries_per_pool=300,
        sha256=hashes,native_checks=checks,parameters_updated=False,reference_masks_read=False,GT_inference=False,
        seconds=time.monotonic()-START,peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20,peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        code_sha256=sha(__file__),common_code_sha256=sha(EXP/'code/common.py'),protocol_sha256=sha(EXP/'protocol.md')))
    guard()
if __name__=='__main__':
    try:main()
    except Exception:
        write(EXP/f'attempts/{time.time_ns()}.json',dict(status='FAILED_PRESERVE_BEFORE_CORRECTION',traceback=traceback.format_exc()));raise
