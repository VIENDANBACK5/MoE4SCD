"""One full-scope numerical/memory gate, no evaluation/model selection."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
from shared import *
import time,traceback,shutil
def main():
    assert not (EXP/'PROBE_COMPLETE.json').exists()
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.42);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert shutil.disk_usage(EXP).free>4*2**30
    plan=read(EXP/'PLAN.json');row=read(EXP/'TRAIN_WINDOWS.json')[0];assert row['id']==plan['probe_window']
    model=load();model.train();image,target,_=load_window(row,device='cuda',augment=False)
    start=time.monotonic();seed(742);loss0,_=loss_backward(model,image,target)
    gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    selected=activation_checkpointing(model);seed(742);loss1,_=loss_backward(model,image,target)
    assert abs(loss0-loss1)<=1e-5*max(1.,abs(loss0)),(loss0,loss1)
    maximum=0.;checked=0
    for n,p in model.named_parameters():
        if p.grad is None:assert n not in gradients;continue
        a=gradients[n];b=p.grad.detach().cpu()
        assert torch.allclose(a,b,atol=1e-6,rtol=1e-3),n
        maximum=max(maximum,float((a-b).abs().max()));checked+=1
    del gradients;model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    image,target=replicate(image,target);assert image.shape[-2:]==(2048,2048)
    torch.cuda.reset_peak_memory_stats();seed(742);opt=optimizer(model)
    first=next((n,p) for n,p in model.named_parameters() if p.requires_grad)
    before=first[1].detach().clone();loss2,parts=loss_backward(model,image,target)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.01,error_if_nonfinite=True);opt.step()
    changed=float((first[1]-before).abs().max());assert changed>0
    peak=torch.cuda.max_memory_allocated()/2**30;assert peak<=20
    modelbytes=sum(v.numel()*v.element_size() for v in model.state_dict().values() if torch.is_tensor(v))
    optbytes=sum(v.numel()*v.element_size() for d in opt.state.values() for v in d.values() if torch.is_tensor(v))
    out=dict(stage='ENGINEERING_GATE_PASSED',loss_native_uncheckpointed=loss0,loss_native_checkpointed=loss1,
             gradient_tensors_compared=checked,max_absolute_gradient_delta=maximum,checkpointed_modules=selected,
             replicated_training_loss=loss2,gradient_norm=float(norm),first_parameter_name=first[0],first_parameter_max_update=changed,
             peak_GPU_GiB=peak,full_model_tensor_bytes=modelbytes,optimizer_tensor_bytes=optbytes,
             projected_three_checkpoint_peak_bytes=3*(modelbytes+optbytes),seconds=time.monotonic()-start,
             engineering_forwards=3,engineering_updates=1,weights_discarded_not_used_for_scientific_arms=True,
             new_method=False,protocol_sha256=sha(EXP/'protocol.md'))
    write(EXP/'PROBE_COMPLETE.json',out);print(json.dumps(out))
if __name__=='__main__':
    try:main()
    except Exception:
        write(EXP/'PROBE_FAILURE.json',dict(stage='ENGINEERING_GATE_FAILED_NO_AUTOMATIC_SCOPE_CHANGE',traceback=traceback.format_exc()))
        raise
