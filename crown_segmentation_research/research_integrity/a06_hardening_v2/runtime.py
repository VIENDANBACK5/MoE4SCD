"""Mandatory production entrypoint gate before data/model access or artifact writes."""
import json
import sys
from pathlib import Path
from .core import REPO,canonical_hash,require,sha256,source_key,write_new
from .preflight import Checker

def add_arguments(parser):
    parser.add_argument('--integrity-config',required=True,help='Hardening v2 JSON recipe; required before model/data access')
    parser.add_argument('--integrity-dry-run',action='store_true',help='Validate only; no model load, data iteration, fitting or output experiment')

def validate_runtime_spec(cfg,method=None):
    if cfg['operation'] in ('integrity_scan','synthetic_smoke'):return {'status':'NOT_APPLICABLE','scope':'no model runtime'}
    operation=cfg['operation'];cli=cfg['runtime'].get('cli')
    require(isinstance(cli,dict),"Frozen CLI map required")
    entry=cfg['runtime'].get('entrypoint')
    supported={'crown_segmentation_research/clean_benchmark/train_treeflow.py':('training','F'),
               'crown_segmentation_research/clean_benchmark/train_treeflow_m3.py':('training','F'),  # EXP-028 M3, user-approved 2026-11-20
               'crown_segmentation_research/clean_benchmark/train_maskrcnn.py':('training','MASKRCNN'),
               'crown_segmentation_research/clean_benchmark/evaluate.py':('evaluation',{'treeflow':'F','maskrcnn':'MASKRCNN'}.get(cli.get('model')))}
    require(entry in supported and supported[entry][0]==operation,"Unregistered runtime recipe/operation")
    selected=supported[entry][1];require(selected in cfg['methods'] and (method is None or method==selected),"Runtime model binding differs from declared comparison")
    method=selected
    features={'F':['P2'],'MASKRCNN':['P2','P3','P4','P5','pool']}
    require(cfg['comparison']['feature_levels'].get(method)==features[method],"Declared feature access differs from implemented head (Mask R-CNN RPN uses all levels)")
    require(cfg['comparison']['TTA'] is False,"No TTA adapter registered")
    if 'out' in cli:require(Path(cli['out']).resolve()==Path(cfg['output_dir']).resolve(),"Trainer output differs from protected destination")
    if operation=='training':
        require(isinstance(cfg['runtime'].get('deterministic_algorithms'),bool),"Explicit model determinism policy required")
        require(0<=cli['seed']<2**32,"Runtime global RNG seed outside NumPy-compatible range")
        require(all(isinstance(cli.get(k),int) and cli[k]>0 for k in ('epochs','batch','accum')),"Invalid training epoch/batch/accumulation budget")
        require(cli.get('workers',0)>=0,"Negative worker count")
        require(cli.get('seed')==cfg['sampling']['master_seed'],"Runtime seed/sampling seed mismatch")
        require(cli.get('crop',1024)==cfg['sampling']['crop_shape'][0]==cfg['sampling']['crop_shape'][1],"Runtime crop/FOV mismatch")
        require(cfg['sampling']['partial_policy']=='visible_fragment_retains_original_identity',"This production adapter does not implement an ignore-partial policy")
        require(cfg['runtime'].get('sampler_adapter') in ('random_with_original_RLE_accounting','planned_native_crowns_v2'),"Unsupported production sampler recipe")
        expected={'name':'AdamW' if method=='F' else 'SGD','lr':cli['lr'],'weight_decay':1e-4}
        if method=='MASKRCNN':expected['momentum']=.9
        require(cfg['comparison']['optimizer'].get(method)==expected,"Declared optimizer differs from implemented training adapter")
        require(cfg['comparison']['input_channels']==3 and cfg['comparison']['input_resolution']==cfg['sampling']['crop_shape'],"RGB/input-resolution contract differs from runtime")
        require(cfg['comparison']['train_distribution']==cfg['runtime']['sampler_adapter'],"Declared train distribution differs from runtime adapter")
        actual_augmentation={'flip_x_probability':.5,'flip_y_probability':.5,'rot90':[0,1,2,3],'brightness_scale':[.9,1.1],'brightness_shift':[-.05,.05]}
        require(cfg['comparison']['augmentations'].get(method)==actual_augmentation,"Declared augmentation differs from runtime")
        expected_supervision={'reference':'legacy_first_wins_priority','target_version':cli['targets']} if method=='F' else {'reference':'original_overlapping_RLE','partial_policy':'visible_fragment_retains_original_identity'}
        require(cfg['comparison']['supervision_by_method'].get(method)==expected_supervision,"Supervision differs from actual loader targets")
    require(not cli.get('resume',False),"Resume into an existing historical destination is prohibited; use a new versioned fork recipe")
    if operation=='evaluation':require(cli.get('split')=='validation',"Historical test performance evaluation forbidden")
    for k in ('ckpt',):
        if k in cli:require(source_key(cli[k]) in cfg['inputs'],"Runtime checkpoint not hash registered")
    if 'tag' in cli:
        import re
        require(isinstance(cli['tag'],str) and re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}',cli['tag']),"Artifact tag can escape protected destination")
    return {'runtime_method':method,'entrypoint':entry,'frozen_CLI':cli}

def gate(args,operation,method,entrypoint):
    from .preflight import load_config
    cfg=load_config(args.integrity_config)
    require(cfg['operation']==operation and method in cfg['methods'],"Config cannot authorize this operation/model")
    require(cfg['runtime'].get('entrypoint')==str(Path(entrypoint).resolve().relative_to(REPO)),"Entrypoint/config mismatch")
    cli={k:v for k,v in vars(args).items() if not k.startswith('integrity_')}
    require(cfg['runtime'].get('cli')==cli,"CLI differs from frozen experiment configuration")
    validate_runtime_spec(cfg,method)
    report=Checker(args.integrity_config,cfg['mode']).run()
    require(report['overall_status']=='PASS',"Preflight rejected: "+json.dumps([r for r in report['checks'] if r['status'] not in ('PASS','NOT_APPLICABLE')]))
    if args.integrity_dry_run:
        print(json.dumps({'dry_run':'PASS','model_loaded':False,'experiment_executed':False,'preflight':report},indent=2));raise SystemExit(0)
    if operation=='training':
        import torch,os
        deterministic=cfg['runtime']['deterministic_algorithms']
        if deterministic:
            require(not torch.cuda.is_initialized() or os.environ.get('CUBLAS_WORKSPACE_CONFIG') in (':4096:8',':16:8'),"CUDA already initialized without deterministic workspace policy")
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.benchmark=False
    hardware=configure_cuda_budget(cfg) if operation in ('training','evaluation') else None
    dest=Path(cfg['output_dir'])
    if 'out' in cli:require(Path(cli['out']).resolve()==dest.resolve(),"Trainer output differs from protected destination")
    dest.mkdir(parents=True,exist_ok=False)
    write_new(dest/'PREFLIGHT.json',report);write_new(dest/'CONFIG.json',cfg)
    write_new(dest/'RUN_PROVENANCE.json',{'config_hash':canonical_hash(cfg),'entrypoint_sha256':sha256(entrypoint),'version':'2.0.0','hardware':hardware,'claims':'exploratory' if cfg['mode']=='exploratory' else 'subject to frozen confirmatory protocol'})
    return cfg

def configure_cuda_budget(cfg):
    import torch,os
    require(cfg['comparison']['GPU']=='cuda:0' and torch.cuda.is_available(),"Registered production adapters require available logical cuda:0")
    require(torch.cuda.current_device()==0,"Runtime CUDA device differs from frozen recipe")
    props=torch.cuda.get_device_properties(0);limit=cfg['comparison']['VRAM_GB']
    require(isinstance(limit,(int,float)) and 0<limit*1024**3<=props.total_memory,"Invalid/unavailable GPU allocator budget")
    fraction=limit*1024**3/props.total_memory;torch.cuda.set_per_process_memory_fraction(fraction,0)
    return {'logical_device':'cuda:0','name':props.name,'total_memory_bytes':props.total_memory,'allocator_limit_GB':limit,
            'allocator_fraction':fraction,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch_cuda_version':torch.version.cuda,
            'limit_scope':'PyTorch allocator only, not driver/external allocations; model-level deterministic reproducibility not verified by this task'}

def initialize_encoder(model,cfg,method):
    """Load only the explicitly registered local encoder; never download model weights."""
    import torch
    source=cfg['comparison']['encoder_checkpoint']
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    state=checkpoint.get('model_state_dict',checkpoint)
    state={k.removeprefix('backbone.'):v for k,v in state.items() if k.startswith('backbone.')}
    require(bool(state),"Encoder checkpoint has no declared backbone.* weights")
    model.backbone.load_state_dict(state,strict=True)
    frozen=cfg['study_type']=='controlled_frozen_features'
    for p in model.backbone.parameters():p.requires_grad_(not frozen)
    count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    require(cfg['comparison']['trainable_parameters'][method]==count,"Measured trainable parameter count differs from frozen comparison")
    write_new(Path(cfg['output_dir'])/'TRAINABLE_PARAMETERS.json',{'method':method,'count':count,'encoder_frozen':frozen,'encoder_sha256':sha256(source)})
    return frozen

def accumulation_divisor(step,n_batches,accum):
    require(1<=step<=n_batches and accum>0,"Invalid accumulation schedule")
    start=((step-1)//accum)*accum
    return min(accum,n_batches-start)

def check_budget(cfg,n_batches,epochs,accum):
    import math
    steps=epochs*math.ceil(n_batches/accum)
    require(steps==cfg['comparison']['updates'],"Runtime optimizer budget differs from registered updates")
    return steps

def log_crop_records(out,records):
    # Append-only within a fresh authorized run, never a historical report.
    with (Path(out)/'CROP_RECORDS.jsonl').open('a') as f:
        for record in records:
            value=json.loads(record) if isinstance(record,str) else record
            f.write(json.dumps(value,allow_nan=False)+'\n')

def reference_metadata(cfg):
    metadata={}
    for dataset in cfg['datasets']:
        report=json.loads(Path(dataset['source_verification']).read_text())
        require(report.get('actual_affine_status')=='PASS',"Native affine integrity not verified")
        metadata.update(report['native_images'])
    return metadata

def sample_plan_for(cfg,split):
    if cfg['runtime'].get('sampler_adapter')!='planned_native_crowns_v2':return None
    plan=json.loads(Path(cfg['sampling']['plan']).read_text())
    records=[r for r in plan['records'] if r['split']==split]
    require(records,f"Frozen crop plan has no records for {split}")
    return records

def reject_legacy_entrypoint(path):
    raise SystemExit(f"INTEGRITY GATE: archival entrypoint {Path(path).name} is disabled for unguarded execution. Original source is preserved in a06_hardening_v2/ORIGINAL_IMPLEMENTATION.zip. Use an audited versioned recipe; do not overwrite historical outputs or rerun oracles/test benchmarks.")

def store_native_output(cfg,arrays):
    import numpy as np
    if '_stage0_path' not in cfg:return # library calls remain pure unless a gated run supplies an artifact sink
    path=Path(cfg['_stage0_path']);path.parent.mkdir(parents=True,exist_ok=True)
    require(all(np.isfinite(a).all() for a in arrays.values()),"Nonfinite model-native output")
    with path.open('xb') as f:np.savez_compressed(f,**arrays)
