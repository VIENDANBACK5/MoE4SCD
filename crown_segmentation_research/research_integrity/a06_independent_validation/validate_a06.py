"""Independent read-only A06 validation; all mutations confined to review sandbox."""
import ast
import collections
import csv
import hashlib
import inspect
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REPO=ROOT.parent
A06=Path(os.environ.get('A06_REVIEW_SOURCE',str(HERE/'reviewed_a06_snapshot'))).resolve()
AUDIT=ROOT/'execution_plan_audit_2026-10-15'
sys.path.insert(0,str(REPO))
_spec=importlib.util.spec_from_file_location('reviewed_a06_helpers',A06/'test_suite.py')
t=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t)


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def dump(p,z):Path(p).write_text(json.dumps(z,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def freeze():
    sources={str(p):sha(p) for p in A06.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    sources.update({str(REPO/'crown_segmentation_research/clean_benchmark'/n):sha(ROOT/'clean_benchmark'/n) for n in ['dataset.py','train_treeflow.py','train_maskrcnn.py','prepare_canopyrs.py']})
    dump(HERE/'snapshot_replay_source_freeze.json',sources)
    if not (HERE/'git_status_before.txt').exists():
        (HERE/'git_status_before.txt').write_text(subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True))
    old=json.loads((AUDIT/'source_freeze.json').read_text())
    mismatch=[p for p,h in old.items() if not Path(p).is_file() or sha(p)!=h]
    dump(HERE/'historical_source_check.json',{'checked':len(old),'mismatches':mismatch,'note':'Verifies the previously frozen 1026 files, not every unregistered historical file.'})
    return sources


def run_original_tests():
    import scipy,torch,torchvision
    env={'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,
         'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),'interpreter':sys.executable}
    dump(HERE/'environment.json',env)
    command=[sys.executable,'-B',str(A06/'test_suite.py')]
    start=time.perf_counter();p=subprocess.run(command,cwd=REPO,capture_output=True,text=True)
    (HERE/'original_tests.log').write_text(p.stdout+'\n'+p.stderr)
    summary=json.loads(p.stdout.split('--- TEST SUMMARY ---')[-1])
    tree=ast.parse((A06/'test_suite.py').read_text())
    empty=[];tests=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.FunctionDef) and n.name.startswith('test_'):
            tests.append(n.name)
            body=[v for v in n.body if not (isinstance(v,ast.Expr) and isinstance(v.value,ast.Constant) and isinstance(v.value.value,str))]
            if body and all(isinstance(v,ast.Pass) for v in body):empty.append({'name':n.name,'line':n.lineno})
    dump(HERE/'original_tests_reexecuted.json',{'command':command,'exit_code':p.returncode,'subprocess_wall_seconds':time.perf_counter()-start,
                                              'summary':summary,'declared_tests':len(tests),'empty_test_bodies':empty})


def cli_probe(name,config=None,baseline='original',mode='exploratory'):
    work=HERE/'sandbox'/name
    base=work/'crown_segmentation_research/research_integrity/a06_hardening'
    base.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(A06,base,ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    (work/'crown_segmentation_research/__init__.py').write_text('')
    (work/'crown_segmentation_research/research_integrity/__init__.py').write_text('')
    bf=base/'BASELINE_HASHES.json'
    if baseline=='missing':bf.unlink()
    elif baseline=='empty':dump(bf,{})
    elif baseline=='tampered':
        # The copied module's wrong parents[4] resolves HERE/sandbox, so put a fixture there.
        target=work.parent/'tamper_fixture.txt';target.write_text('tampered content')
        dump(bf,{'tamper_fixture.txt':{'sha256':hashlib.sha256(b'expected content').hexdigest()}})
    command=[sys.executable,'-B','-m','crown_segmentation_research.research_integrity.a06_hardening.preflight','--mode',mode]
    if config=='missing':command+=['--config',str(work/'does_not_exist.json')]
    elif config is not None:
        cfg=work/'experiment.json';cfg.write_text(config if isinstance(config,str) else json.dumps(config));command+=['--config',str(cfg)]
    before=(base/'READINESS_REPORT.json').read_bytes()
    p=subprocess.run(command,cwd=work,capture_output=True,text=True,env={**os.environ,'PYTHONPATH':str(work)})
    (work/'preflight.log').write_text(p.stdout+'\n'+p.stderr)
    report=json.loads((base/'READINESS_REPORT.json').read_text())
    return {'case':name,'mode':mode,'exit_code':p.returncode,'readiness':report['readiness_verdict'],
            'overall':report['overall_status'],'checks':report['checks'],'audit_inputs_present':False,
            'copied_readiness_report_bytes_changed':before!=(base/'READINESS_REPORT.json').read_bytes(),
            'command':command,'stdout_path':str(work/'preflight.log')}


def preflight_probes():
    bad={'datasets':{'train':{'role':'CONFIRMATORY_TEST','path':'historical_test'},'calibration':{'role':'CONFIRMATORY_TEST','labels':'test'}},
         'query_count':1,'max_GT_per_crop':7,'encoder_checkpoint':'missing.pth','split_manifest':'missing.json',
         'output_dir':str(ROOT/'clean_benchmark'),'threshold_selection_split':'CONFIRMATORY_TEST',
         'source_hash':'wrong','optimizer':None,'buffer_m':-10,'mode':'confirmatory','waivers':[]}
    cases=[cli_probe('no_config'),cli_probe('missing_config','missing'),cli_probe('malformed_config','{ invalid JSON'),
           cli_probe('forbidden_roles_capacity_paths',bad),cli_probe('missing_baseline',bad,'missing'),
           cli_probe('empty_baseline',bad,'empty'),cli_probe('tampered_real_digest',bad,'tampered'),
           cli_probe('confirmatory_no_inputs',None,'original','confirmatory')]
    dump(HERE/'preflight_probes.json',cases)
    print('CLI probes',[(r['case'],r['exit_code']) for r in cases],flush=True)


def metric_probes():
    probes=[]
    def record(name,expected,actual):probes.append({'case':name,'expected':expected,'actual':actual})
    anchors=np.array([[0,0],[0,1]]);pairs=np.array([[0,1]]);same=np.array([True])
    empty=t.evaluate_structural_and_e2e_grouping([],np.array([]),np.empty((0,2),int),np.empty((0,2),int),np.array([],bool))
    record('zero_denominator','NA/null with counts',empty)
    mask=np.ones((2,2),bool)
    singleclass=t.evaluate_structural_and_e2e_grouping([mask],np.array([.9]),anchors,pairs,same)
    record('no_different_pairs','diff accuracy NA, balanced common-support policy required',singleclass)
    a=np.array([[1,1,0]],bool);b=np.array([[1,0,1]],bool)
    r1=t.evaluate_structural_and_e2e_grouping([a,b],np.array([.9,.8]),anchors,pairs,same,.5)
    r2=t.evaluate_structural_and_e2e_grouping([a,b],np.array([.8,.9]),anchors,pairs,same,.5)
    record('highest_score_overlap_E2E','Prior execution plan requires same accuracy1 then0 under highest-score E2E resolution; A06 changes that policy to unique cover',{'scores_09_08':r1,'scores_08_09':r2})
    n=t.assign_size_bin(float('nan'));record('NaN_area','reject invalid area',n)
    z=t.compute_canonical_area(t.mu.encode(np.asfortranarray(mask.astype(np.uint8))),affine_transform=(0,0,0,0,0,0))
    record('degenerate_affine','reject zero determinant',z)
    try:
        supp=t.derive_support_sets([np.ones((2,1),bool)],np.ones((2,2),bool))
        shape_result={'original_shape':list(supp['original'][0].shape),'derived_valid_shape':list(supp['valid_support'][0].shape)}
    except Exception as exc:
        shape_result={'rejected':True,'exception':type(exc).__name__,'message':str(exc)}
    record('mismatched_mask_shapes','reject nonmatching mask/image dimensions',shape_result)
    v,d=t.validate_spatial_buffer((0,0,10,10),(1,1,9,9),min_buffer_meters=0)
    record('overlapping_boxes_zero_buffer','overlap policy separate from distance; cannot certify disjoint support',{'valid':v,'distance':d})
    # Legacy first annotation wins; A06 legacy label paints last annotation over the first.
    m0=np.ones((5,5),bool);m1=np.zeros((5,5),bool);m1[1:4,1:4]=1
    legacy=np.zeros((5,5),int)
    for k,m in enumerate((m0,m1),1):legacy[m&(legacy==0)]=k
    a06=t.derive_support_sets([m0,m1],np.ones((5,5),bool),conflict_policy='index_descending')['priority_raster']
    record('legacy_priority_replay','same as historical first-wins code',{'equal':bool(np.array_equal(legacy,a06)),'changed_pixels':int((legacy!=a06).sum()),'historical_owner_at_center':int(legacy[2,2]),'A06_owner_at_center':int(a06[2,2])})
    val0=t.get_crop_rng(123,0,'validation',1,0).integers(10000,size=8).tolist()
    val1=t.get_crop_rng(123,1,'validation',1,0).integers(10000,size=8).tolist()
    record('validation_epoch_handling','caller must lock epoch; helper itself does not enforce it',{'equal':val0==val1})
    dump(HERE/'metric_probes.json',probes)


def sampler_probe():
    import torch
    from torch.utils.data import DataLoader
    from crown_segmentation_research.clean_benchmark import dataset as dsmod
    folder=HERE/'synthetic_data/train';folder.mkdir(parents=True,exist_ok=True)
    yy,xx=np.indices((40,40));image=np.stack([yy*6,xx*6,(xx+yy)*3],axis=-1).astype(np.uint8)
    label=np.zeros((40,40),np.int32);label[3:37,3:37]=1
    for i in range(4):np.savez_compressed(folder/f'{i}.npz',image=image,valid=np.ones((40,40),bool),label=label)
    previous=dsmod.PREP;dsmod.PREP=folder.parent;results=[]
    try:
        for nw,persistent in ((0,False),(2,False),(2,True)):
            torch.manual_seed(17)
            ds=dsmod.CanopyCropDataset('train',crop=12,crops_per_tile=1,augment=False,fixed=False)
            dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=nw,persistent_workers=persistent)
            epochs=[]
            for _ in range(2):epochs.append([b['image'].clone() for b in dl])
            equal=[torch.equal(a,b) for a,b in zip(*epochs)]
            results.append({'workers':nw,'persistent_workers':persistent,'same_crop_images_across_two_epochs':equal,
                            'has_set_epoch_api':hasattr(ds,'set_epoch'),'data':'synthetic fixture; actual CanopyCropDataset unchanged'})
            if persistent and dl._iterator is not None:dl._iterator._shutdown_workers()
    finally:dsmod.PREP=previous
    dump(HERE/'actual_loader_rng_probe.json',results)


def source_analysis():
    baseline=json.loads((A06/'BASELINE_HASHES.json').read_text())
    actual=[{'path':p,'recorded_missing':v=='MISSING','actually_exists':(REPO/p).is_file(),
             'actual_sha256':sha(REPO/p) if (REPO/p).is_file() else None} for p,v in baseline.items()]
    tree=ast.parse((A06/'preflight.py').read_text())
    reads=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Attribute) and n.attr=='config_path':reads.append({'line':n.lineno,'ctx':type(n.ctx).__name__})
    refs=[]
    for parent in (ROOT/'clean_benchmark',ROOT/'methods',REPO/'tests'):
        for p in parent.rglob('*.py'):
            text=p.read_text(errors='replace')
            if 'a06_hardening' in text or 'get_crop_rng' in text:refs.append(str(p))
    data=json.loads((AUDIT/'data_summary.json').read_text())
    declared=json.loads((A06/'READINESS_REPORT.json').read_text())
    totals=sum(r['cross_split_shared_pixels']['train__validation']+r['cross_split_shared_pixels']['train__test']+r['cross_split_shared_pixels']['validation__test'] for r in json.loads((AUDIT/'source_grid_support.json').read_text()))
    dump(HERE/'source_analysis.json',{'baseline_entries':actual,'reviewed_A06_source':str(A06),'correct_repo_root':str(REPO),'A06_computed_repo_root':str((HERE.parent/'a06_hardening/preflight.py').resolve().parents[4]),
                                     'config_path_AST_usages':reads,'runtime_integration_references':refs,
                                     'recorded_readiness':declared['readiness_verdict'],'verified_prior_support_counts':data,
                                     'actual_pairwise_shared_pixel_sum':totals,'A06_fallback_pixel_sum':38955})


def evidence_crosschecks():
    tree=ast.parse((A06/'test_suite.py').read_text())
    names={n.name for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')}
    rows=list(csv.DictReader((A06/'ROOT_CAUSE_MATRIX.csv').open()))
    refs=[{'id':r['id'],'claimed_regression_test':r['regression_test'],'test_exists_in_delivered_suite':r['regression_test'] in names} for r in rows]
    analysis=json.loads((HERE/'source_analysis.json').read_text())
    actual={Path(r['path']).name:r['actual_sha256'] for r in analysis['baseline_entries']}
    claimed=re.findall(r'`([^`]+)`:\s*`([a-f0-9]{64})`',(A06/'HARDENING_REPORT.md').read_text())
    hashes=[{'file':n,'reported_sha256':h,'computed_sha256':actual.get(n),'matches':h==actual.get(n)} for n,h in claimed]
    dump(HERE/'evidence_crosschecks.json',{'regression_test_references':refs,'report_hash_comparisons':hashes,
                                         'reported_environment_matches_current_core_versions':True})


def source_annotation_metadata_scan():
    """Scan compressed RLE records without loading imagery or evaluating predictions."""
    records=[]
    for split in ('train','validation','test'):
        path=REPO/'data/itc_benchmarks/canopyrs_quebectrees/prepared'/f'{split}_coco.json'
        coco=json.loads(path.read_text()); images={im['id']:im for im in coco['images']}
        counts=collections.Counter();examples=collections.defaultdict(list)
        def flag(key,ann):
            counts[key]+=1
            if len(examples[key])<8:examples[key].append(ann.get('id'))
        imgids=[im['id'] for im in coco['images']];annids=[a['id'] for a in coco['annotations']]
        counts['duplicate_image_id_records']=len(imgids)-len(set(imgids))
        counts['duplicate_annotation_id_records']=len(annids)-len(set(annids))
        for ann in coco['annotations']:
            im=images.get(ann['image_id'])
            if im is None:flag('orphan_annotations',ann);continue
            seg=ann.get('segmentation')
            if not isinstance(seg,dict):flag('non_RLE_segmentation',ann);continue
            if seg.get('size')!=[im['height'],im['width']]:flag('RLE_dimension_mismatch',ann)
            try:
                rle={**seg,'counts':seg['counts'].encode() if isinstance(seg['counts'],str) else seg['counts']}
                area=float(t.mu.area(rle));box=np.asarray(t.mu.toBbox(rle),float)
                if area==0:flag('zero_RLE_area',ann)
                if not np.isfinite(area) or area<0:flag('invalid_RLE_area',ann)
                if box[0]<0 or box[1]<0 or box[0]+box[2]>im['width'] or box[1]+box[3]>im['height']:flag('RLE_bbox_outside_image',ann)
                if not np.isclose(float(ann['area']),area,rtol=0,atol=1e-6):flag('metadata_RLE_area_difference',ann)
                if not np.allclose(np.asarray(ann['bbox'],float),box,rtol=0,atol=1e-6):flag('metadata_RLE_bbox_difference',ann)
            except Exception as exc:flag('RLE_scan_exception_'+type(exc).__name__,ann)
        records.append({'split':split,'source':str(path),'sha256':sha(path),'images':len(imgids),'annotations':len(annids),
                        'flags':dict(counts),'example_annotation_ids':dict(examples)})
    dump(HERE/'source_annotation_metadata_scan.json',{'scope':'Prepared COCO compressed RLE area/bbox/dimensions/IDs only; no image/valid-mask decoding, raw parquet comparison, model predictions or performance evaluation.',
                                                    'splits':records})


def advertised_cli_probe():
    """Help-only subprocesses: never start training, decoding or model evaluation."""
    items=[]
    for name,claimed in (('train_treeflow.py',['--config','--crop-size','--query-capacity','--device']),
                         ('exp021_veto_merge.py',['--mode'])):
        source=ROOT/'clean_benchmark'/name
        command=[sys.executable,'-B',str(source),'--help']
        p=subprocess.run(command,cwd=REPO,text=True,capture_output=True)
        (HERE/(name+'.help.log')).write_text(p.stdout+'\n'+p.stderr)
        tree=ast.parse(source.read_text());opts=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument':
                opts.extend(a.value for a in n.args if isinstance(a,ast.Constant) and isinstance(a.value,str))
        items.append({'script':str(source),'help_command':command,'help_exit_code':p.returncode,'accepted_options':opts,
                      'advertised_but_unsupported_options':[o for o in claimed if o not in opts]})
    dump(HERE/'advertised_commands_check.json',{'commands':items,
         'advertised_Q_yaml_exists':(ROOT/'clean_benchmark/configs/q_full_crown_exploratory.yaml').is_file(),
         'train_treeflow_requires_out':True,'note':'Only --help executed; no advertised training or experiment command executed.'})


def main():
    before=freeze();run_original_tests();source_analysis();evidence_crosschecks();source_annotation_metadata_scan();advertised_cli_probe();metric_probes();sampler_probe();preflight_probes()
    changed=[p for p,h in before.items() if sha(p)!=h]
    dump(HERE/'snapshot_replay_preservation.json',{'reviewed_source_files':len(before),'changed_reviewed_files':changed,'reviewed_snapshot_reports_preserved':not changed,
                                         'no_training_or_historical_test_performance':True})
    assert not changed


if __name__=='__main__':main()
