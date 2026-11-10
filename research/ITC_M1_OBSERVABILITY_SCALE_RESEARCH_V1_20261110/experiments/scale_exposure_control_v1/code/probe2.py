"""One bounded numerical correction; not scientific training."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
from shared import *
import traceback,time,math,shutil
def grads(model,image,target,precision):
    model.zero_grad(set_to_none=True);seed(742)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
        loss,parts,out=training_loss(model,image,target)
    val=float(loss.detach());del out,parts;assert math.isfinite(val)
    loss.backward();del loss
    return val,{n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
def compare(a,b):
    assert a.keys()==b.keys()
    diffsq=baseq=0.;cos=[];bad=[];maxabs=0.
    for n,x in a.items():
        y=b[n];d=x-y;dif=float(d.double().square().sum());norm=float(x.double().square().sum())
        diffsq+=dif;baseq+=norm;maxabs=max(maxabs,float(d.abs().max()))
        if norm>1e-12 and float(y.double().square().sum())>1e-12:
            cos.append(float(torch.dot(x.flatten().double(),y.flatten().double())/(x.double().norm()*y.double().norm())))
        if not torch.allclose(x,y,atol=1e-6,rtol=1e-3):bad.append(n)
    return dict(gradient_tensors=len(a),relative_L2=math.sqrt(diffsq/max(baseq,1e-30)),
                minimum_nontrivial_tensor_cosine=min(cos,default=1.),max_absolute_delta=maxabs,
                pointwise_tolerance_failure_count=len(bad),first_pointwise_failures=bad[:5])
def main():
    assert (EXP/'PROBE_FAILURE.json').exists() and not (EXP/'PROBE2_COMPLETE.json').exists()
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.42)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.monotonic();row=read(EXP/'TRAIN_WINDOWS.json')[0];results={}
    for precision in ['fp32','bf16']:
        model=load();model.train();image,target,_=load_window(row,device='cuda',augment=False)
        v0,g0=grads(model,image,target,precision);model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        modules=activation_checkpointing(model)
        v1,g1=grads(model,image,target,precision);result=dict(loss_plain=v0,loss_checkpointed=v1,**compare(g0,g1));results[precision]=result
        write(EXP/'PROBE2_NUMERICAL_PROGRESS.json',results);print(precision,json.dumps(result),flush=True)
        assert abs(v0-v1)<=1e-5*max(abs(v0),1.)
        if precision=='fp32':assert result['pointwise_tolerance_failure_count']==0,result
        else:assert result['relative_L2']<=.02 and result['minimum_nontrivial_tensor_cosine']>=.99,result
        del model,image,target,g0,g1;torch.cuda.empty_cache()
    model=load();model.train();modules=activation_checkpointing(model)
    image,target,_=load_window(row,device='cuda',augment=False);image,target=replicate(image,target)
    opt=optimizer(model);torch.cuda.reset_peak_memory_stats();seed(742)
    n,p=next((n,p) for n,p in model.named_parameters() if p.requires_grad);before=p.detach().clone()
    loss,parts=loss_backward(model,image,target);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.01,error_if_nonfinite=True);opt.step()
    change=float((p-before).abs().max());assert change>0
    peak=torch.cuda.max_memory_allocated()/2**30;assert peak<=20
    mb=sum(v.numel()*v.element_size() for v in model.state_dict().values() if torch.is_tensor(v))
    ob=sum(v.numel()*v.element_size() for z in opt.state.values() for v in z.values() if torch.is_tensor(v))
    assert shutil.disk_usage(EXP).free-3*(mb+ob+8*2**20)>4*2**30,'Not enough for atomic full-state saves'
    out=dict(stage='ENGINEERING_GATE_PASSED_AFTER_DOCUMENTED_NUMERICAL_CORRECTION',numerical=results,
             checkpointed_modules=modules,replicated_training_loss=loss,gradient_norm=float(norm),
             first_parameter=n,parameter_max_update=change,peak_GPU_GiB=peak,model_tensor_bytes=mb,optimizer_tensor_bytes=ob,
             projected_three_checkpoint_peak_bytes=3*(mb+ob+8*2**20),seconds=time.monotonic()-start,
             engineering_forwards=5,engineering_updates=1,probe_weights_discarded=True,
             original_failure_preserved=True,protocol_sha256=sha(EXP/'protocol.md'),
             addendum_sha256=sha(EXP/'ENGINEERING_PARITY_ADDENDUM.md'))
    write(EXP/'PROBE2_COMPLETE.json',out);print(json.dumps(out),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        write(EXP/'PROBE2_FAILURE.json',dict(stage='SECOND_ENGINEERING_GATE_FAILED_STOP_NO_SCOPE_CHANGE',traceback=traceback.format_exc()))
        raise
