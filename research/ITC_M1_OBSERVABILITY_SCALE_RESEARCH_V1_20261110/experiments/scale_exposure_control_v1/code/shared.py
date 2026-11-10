from pathlib import Path
import sys,json,hashlib,types,random
import numpy as np
EXP=Path(__file__).resolve().parents[1];REPO=EXP.parents[3]
OLD=REPO/'research/ITC_MASK_DINO_FORMATION_V1_20261104'
sys.path.insert(0,str(OLD));import bootstrap
import torch
from torch.utils.checkpoint import checkpoint
from carrier import build_carrier,training_loss
from train_data import load_window
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for q in iter(lambda:f.read(2**20),b''):h.update(q)
    return h.hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp')
    q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)
def seed(x):
    torch.manual_seed(x);torch.cuda.manual_seed_all(x);np.random.seed(x);random.seed(x)
def load():
    plan=read(EXP/'PLAN.json');cp=OLD/'checkpoints/mask_only_seed42/final.pt'
    assert sha(cp)==plan['checkpoint_sha256']
    model,prov,cfg=build_carrier(seed=42,pretrained=False)
    state=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True);del state
    model.criterion.matcher.cost_box=0.;model.criterion.matcher.cost_giou=0.
    return model
def optimizer(model):
    groups=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        wd=0. if p.ndim==1 or any(k in name for k in ['query_embed','query_feat','level_embed','relative_position_bias_table']) else .05
        lr=1e-6 if name.startswith('backbone.') else 1e-5
        groups.append(dict(params=[p],lr=lr,initial_lr=lr,weight_decay=wd))
    return torch.optim.AdamW(groups)
def activation_checkpointing(model):
    selected=[]
    for name,module in model.named_modules():
        pieces=name.split('.')
        backbone=len(pieces)==2 and pieces[0]=='backbone' and pieces[1] in {'res2','res3','res4','res5'}
        encoder=name.startswith('sem_seg_head.pixel_decoder.transformer.encoder.layers.') and len(pieces)==6 and pieces[-1].isdigit()
        decoder=name.startswith('sem_seg_head.predictor.decoder.layers.') and len(pieces)==5 and pieces[-1].isdigit()
        if not (backbone or encoder or decoder):continue
        original=module.forward
        def factory(fn):
            def forward(self,*args,**kwargs):
                if self.training and torch.is_grad_enabled():return checkpoint(fn,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
                return fn(*args,**kwargs)
            return forward
        module.forward=types.MethodType(factory(original),module);selected.append(name)
    assert selected and any(x.startswith('backbone.') for x in selected)
    assert any('.encoder.layers.' in x for x in selected) and any('.decoder.layers.' in x for x in selected)
    return selected
def replicate(image,target):
    image=image.repeat_interleave(2,-2).repeat_interleave(2,-1)
    target={k:v.repeat_interleave(2,-2).repeat_interleave(2,-1) if k in {'masks','valid'} else v for k,v in target.items()}
    return image,target
def loss_backward(model,image,target,divisor=1.):
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts,out=training_loss(model,image,target)
    assert torch.isfinite(loss)
    logs={k:float(v.detach()) for k,v in parts.items()};value=float(loss.detach());del parts,out
    (loss/divisor).backward();del loss
    return value,logs
