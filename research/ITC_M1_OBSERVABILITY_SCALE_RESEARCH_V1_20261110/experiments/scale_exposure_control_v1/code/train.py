"""Full original trainable scope; one fixed matched continuation arm."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
from shared import *
import argparse,time,shutil,resource,csv,traceback
from torch.utils.data import Dataset,DataLoader
class Samples(Dataset):
    def __init__(self,rows,start):self.rows=rows;self.start=start
    def __len__(self):return len(self.rows)-self.start
    def __getitem__(self,i):
        k=i+self.start;r=self.rows[k];draw=420000+k
        rng=np.random.default_rng(draw);im,t,ids=load_window(r,augment=True,rng=rng)
        g=np.random.default_rng(draw)
        return im,t,dict(index=k,window_id=r['id'],source=r['source'],targets=len(ids),rotation=int(g.integers(4)),flip=bool(g.integers(2)))
def first(b):return b[0]
def append(p,r):
    exists=p.exists()
    with p.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(r))
        if not exists:w.writeheader()
        w.writerow(r)
def physical_bytes():
    seen=set();n=0
    for p in EXP.rglob('*'):
        if p.is_file():
            st=p.stat();key=(st.st_dev,st.st_ino)
            if key not in seen:seen.add(key);n+=st.st_size
    return n
def main(arm,resume=False):
    plan=read(EXP/'PLAN.json');probe=read(EXP/'PROBE2_COMPLETE.json');assert 'PASSED' in probe['stage']
    assert probe['protocol_sha256']==sha(EXP/'protocol.md')
    rows=read(EXP/'TRAIN_WINDOWS.json');assert len(rows)==796
    original={r['id']:r for r in read(OLD/'TRAIN_WINDOWS.json')}
    assert all(r==original[r['id']] and r['tile'] not in plan['excluded_from_additional_updates_tiles'] for r in rows)
    folder=EXP/'checkpoints'/arm;folder.mkdir(parents=True,exist_ok=True)
    logs=EXP/'logs'/arm;logs.mkdir(parents=True,exist_ok=True)
    assert not (folder/'final.pt').exists(),'Completed arm immutable'
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.42);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    seed(42);model=load();modules=activation_checkpointing(model);opt=optimizer(model)
    index0=updates=0;prior=0.
    if resume:
        ck=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        assert ck['protocol_sha256']==sha(EXP/'protocol.md') and ck['arm']==arm
        model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer'])
        index0=ck['next_microbatch'];updates=ck['updates'];prior=ck['elapsed_seconds']
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);del ck
    else:assert not (folder/'last.pt').exists(),'Explicit resume required'
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    write(logs/'INITIALIZATION.json',dict(arm=arm,parent_sha256=plan['checkpoint_sha256'],trainable_parameters=trainable,
          checkpointed_modules=modules,protocol_sha256=sha(EXP/'protocol.md'),schedule_sha256=sha(EXP/'TRAIN_WINDOWS.json'),
          code_sha256=sha(__file__),shared_sha256=sha(EXP/'code/shared.py'),probe_weights_used=False))
    loader=DataLoader(Samples(rows,index0),batch_size=1,num_workers=2,collate_fn=first,pin_memory=True,prefetch_factor=2)
    start=time.monotonic();opt.zero_grad(set_to_none=True);model.train();group=[]
    checkpoint_bytes=probe['model_tensor_bytes']+probe['optimizer_tensor_bytes']+8*2**20
    def guard():
        assert prior+time.monotonic()-start<2700,'45minute arm bound'
        assert torch.cuda.max_memory_allocated()<=20*2**30
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<16*2**20
        assert shutil.disk_usage(EXP).free>4*2**30
        assert physical_bytes()<2*2**30
    def save(next_index):
        guard();assert shutil.disk_usage(EXP).free-checkpoint_bytes>4*2**30,'Atomic save disk bound'
        tmp=folder/'last.tmp'
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all(),next_microbatch=next_index,updates=updates,
                   elapsed_seconds=prior+time.monotonic()-start,arm=arm,protocol_sha256=sha(EXP/'protocol.md'),
                   schedule_sha256=sha(EXP/'TRAIN_WINDOWS.json'),parent_sha256=plan['checkpoint_sha256'],
                   trainable_parameters=trainable,selection='FIXED_FINAL_NO_VAL'),tmp)
        tmp.replace(folder/'last.pt');guard()
    try:
        for im,t,meta in loader:
            guard();k=meta['index'];seed(4200000+k);factor=.1+.9*min(updates/20.,1.)
            for g in opt.param_groups:g['lr']=g['initial_lr']*factor
            im=im.cuda(non_blocking=True);t={a:b.cuda(non_blocking=True) for a,b in t.items()}
            scale=2 if arm=='S1' and k%2 else 1
            if scale==2:im,t=replicate(im,t)
            val,parts=loss_backward(model,im,t,divisor=2.)
            group.append(dict(**meta,scale=scale,loss=val,parts=parts));del im,t
            if k%2:
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.01,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True);updates+=1
                record=dict(arm=arm,update=updates,microbatches=k+1,mean_loss=float(np.mean([z['loss'] for z in group])),
                     gradient_norm=float(norm),lr=1e-5*factor,seconds=prior+time.monotonic()-start,
                     windows='|'.join(z['window_id'] for z in group),scales='|'.join(str(z['scale']) for z in group),
                     rotations='|'.join(str(z['rotation']) for z in group),flips='|'.join(str(z['flip']) for z in group),
                     target_counts='|'.join(str(z['targets']) for z in group),
                     **{a:float(np.mean([z['parts'][a] for z in group])) for a in group[0]['parts']})
                append(logs/'TRAINING_LOG.csv',record);group=[]
                if updates%20==0:
                    st=dict(stage='TRAINING',arm=arm,updates=updates,total_updates=398,microbatches=k+1,
                            elapsed_seconds=record['seconds'],loss=record['mean_loss'],pid=os.getpid(),
                            peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20)
                    write(logs/'STATUS.json',st);print(json.dumps(st),flush=True)
                if updates%100==0 or updates==398:save(k+1)
        assert updates==398 and not group
        os.link(folder/'last.pt',folder/'final.pt')
        write(logs/'STATUS.json',dict(stage='COMPLETED_FIXED_SCHEDULE',arm=arm,updates=updates,microbatches=796,
              elapsed_seconds=prior+time.monotonic()-start,checkpoint_sha256=sha(folder/'final.pt'),
              checkpoint_bytes=(folder/'final.pt').stat().st_size,optimizer_state_saved=True,
              peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20,TEST=False,VAL_selection=False,probe_weights_used=False))
        print('COMPLETED',arm,updates,flush=True)
    except Exception:
        write(logs/'FAILURE.json',dict(stage='FAILED_NO_AUTOMATIC_PROTOCOL_CHANGE',arm=arm,updates=updates,
              last_verified_checkpoint=str(folder/'last.pt') if (folder/'last.pt').exists() else None,traceback=traceback.format_exc()))
        raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['S0','S1'],required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();main(a.arm,a.resume)
