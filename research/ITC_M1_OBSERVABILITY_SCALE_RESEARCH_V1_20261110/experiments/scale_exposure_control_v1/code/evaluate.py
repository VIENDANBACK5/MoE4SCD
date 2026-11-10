"""Freeze both trained controls on the exact registered diagnostic inputs."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
from shared import *
import gzip,time,importlib.util,shutil
import torch.nn.functional as F
PILOT=EXP.parent/'frozen_encoding_v1'
spec=importlib.util.spec_from_file_location('frozen_materializer',PILOT/'code/predict.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
def main():
    assert not (EXP/'EVALUATIONS_FROZEN.json').exists()
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.18)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cohort=read(PILOT/'COHORT.json');records={};forwards=0
    for arm in ['S0','S1']:
        status=read(EXP/'logs'/arm/'STATUS.json');assert status['stage']=='COMPLETED_FIXED_SCHEDULE' and status['updates']==398
        cp=EXP/'checkpoints'/arm/'final.pt';assert sha(cp)==status['checkpoint_sha256']
        model=load();ck=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True);del ck
        model.eval().requires_grad_(False);core.install(model.sem_seg_head.predictor,'native')
        outdir=EXP/'evaluations'/arm;(outdir/'predictions').mkdir(parents=True,exist_ok=True)
        for name in ['COHORT.json','INHERITED_CACHE_BINDINGS.json']:shutil.copyfile(PILOT/name,outdir/name)
        # Reuse immutable exact inputs without duplicated payload.
        for row in cohort:
            src=PILOT/f"{row['id']}_inputs.npz";dst=outdir/src.name
            if not dst.exists():os.link(src,dst)
        capt={}
        def hook(module,args):capt['tensor']=args[0].detach().cpu()
        h=model.backbone.register_forward_pre_hook(hook);hashes={};proof=[]
        for row in cohort:
            inp=np.load(PILOT/f"{row['id']}_inputs.npz");im=inp['RGB'].astype(np.float32);c=inp['coarse_RGB'];v=inp['common_valid']
            rep=lambda a:np.repeat(np.repeat(a,2,0),2,1)
            inputs=dict(A=im,B=rep(im),C=c,D=rep(c));norms={}
            for obs,a in inputs.items():
                assert shutil.disk_usage(EXP).free>4*2**30
                t0=time.monotonic();t=torch.from_numpy(a.transpose(2,0,1).copy()).cuda()
                seed(42)
                with torch.no_grad():
                    out=core.predict_logits(model,[t]);forwards+=1;norms[obs]=capt.pop('tensor')
                    logits=out['pred_logits'][0].float();cls=logits.sigmoid()[:,0];vv=torch.from_numpy(v).cuda();raw=[]
                    for j in range(0,300,8):
                        z=F.interpolate(out['pred_masks'][0,j:j+8][None].float(),(1024,1024),mode='bilinear',align_corners=False)[0]
                        mm=(z>0)&vv;area=mm.flatten(1).sum(1);quality=(z.sigmoid()*mm).flatten(1).sum(1)/(area+1e-6)
                        for k in range(len(mm)):
                            q=j+k;raw.append(dict(query_id=q,encoder_site=model.sem_seg_head.predictor.site_trace['indices'][0][q],
                                class_logit=float(logits[q,0]),class_probability=float(cls[q]),mask_quality=float(quality[k]),
                                score=float(cls[q]*quality[k]),area=int(area[k]),rle=core.pack(mm[k].cpu().numpy()),
                                predicted_box_cxcywh=out['pred_boxes'][0,q].float().cpu().tolist()))
                    pre,keep,supp=core.select(raw)
                dest=outdir/'predictions'/f"{row['id']}_{obs}.json.gz"
                data=dict(window_id=row['id'],source=row['source'],arm=obs,training_arm=arm,raw_predictions=raw,
                     accepted_indices=pre,final_indices=keep,suppressed_indices=supp,
                     RGB_only=True,GT_inference=False,reference_masks_read=False,queries=300,
                     input_hw=list(a.shape[:2]),canonical_hw=[1024,1024],checkpoint_sha256=status['checkpoint_sha256'],
                     protocol_sha256=sha(EXP/'protocol.md'),seconds=time.monotonic()-t0)
                with gzip.open(dest,'wt') as f:json.dump(data,f,allow_nan=False)
                hashes[str(dest.relative_to(outdir))]=sha(dest);print(arm,row['id'],obs,len(keep),flush=True)
                del out,t,z,mm
            assert torch.equal(norms['B'][:,:,::2,::2],norms['A']) and torch.equal(norms['D'][:,:,::2,::2],norms['C'])
            proof.append(dict(window_id=row['id'],actual_backbone_tensor_left_inverse_exact=True))
        h.remove();del model;torch.cuda.empty_cache()
        core.write(outdir/'PREDICTIONS_FROZEN.json',dict(forwards=12,sha256=hashes,exact_left_inverse_proofs=proof,checkpoint_sha256=status['checkpoint_sha256']))
        records[arm]=dict(checkpoint_sha256=status['checkpoint_sha256'],predictions_freeze_sha256=sha(outdir/'PREDICTIONS_FROZEN.json'))
    assert forwards==24
    write(EXP/'EVALUATIONS_FROZEN.json',dict(stage='BOTH_CONTROL_PREDICTIONS_FROZEN_BEFORE_NEW_METRICS',forwards=24,arms=records,GT_inference=False))
if __name__=='__main__':main()
