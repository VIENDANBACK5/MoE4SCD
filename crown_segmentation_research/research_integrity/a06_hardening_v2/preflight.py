"""Fail-closed config/data/access/output gates; JSON config, versioned evidence only."""
from __future__ import annotations
import argparse
import datetime
import importlib.metadata
import json
import platform
import sys
import numpy as np
from pathlib import Path
from .core import REPO,IntegrityError,canonical_hash,check_access,check_capacity,require,sha256,source_key,verify_manifest,write_new
from .spatial import validate_split

HERE=Path(__file__).resolve().parent
REQUIRED=('version','mode','operation','study_type','methods','inputs','required_inputs','datasets','output_dir','spatial_policy',
          'sampling','selection','calibration','comparison','runtime','measurement')
OPERATIONS=('integrity_scan','synthetic_smoke','training','evaluation','calibration')
METHODS=('F','G0','G1_merge','G_repartition','Q','MASKRCNN')

def critical_sources():
    paths=list(HERE.glob('*.py'))+list((REPO/'crown_segmentation_research/clean_benchmark').glob('*.py'))+list((REPO/'crown_segmentation_research/methods/tree_flow').glob('*.py'))
    paths+=[REPO/'crown_segmentation_research/diagnosis_2026-10-11/fgq_revision_2026-10-13/grouping.py',HERE/'ENTRYPOINT_REGISTRY.json',HERE/'ADDITIONAL_ENTRYPOINT_REGISTRY.json',HERE/'HISTORICAL_RUNNER_REGISTRY.json',HERE/'READ_ROOTS.json']
    for name in ('ENTRYPOINT_REGISTRY.json','ADDITIONAL_ENTRYPOINT_REGISTRY.json','HISTORICAL_RUNNER_REGISTRY.json'):
        paths += [REPO/row['path'] for row in json.loads((HERE/name).read_text())]
    return sorted({str(p.relative_to(REPO)) for p in paths})

def load_config(path):
    require(path is not None and Path(path).is_file(),"Experiment config missing")
    try:cfg=json.loads(Path(path).read_text())
    except (ValueError,UnicodeError) as exc:raise IntegrityError(f"Invalid JSON config: {exc}") from exc
    require(isinstance(cfg,dict) and all(k in cfg for k in REQUIRED),"Configuration incomplete")
    require(cfg['version']=='2.0.0' and cfg['mode'] in ('exploratory','confirmatory') and cfg['operation'] in OPERATIONS,"Unsupported config version/mode/operation")
    require(cfg['study_type'] in ('infrastructure','controlled_frozen_features','practical_system'),"Unknown study type")
    require(isinstance(cfg['methods'],list) and cfg['methods'] and set(cfg['methods'])<=set(METHODS),"Explicit method variants required")
    require(len(cfg['methods'])==len(set(cfg['methods'])),"Duplicate method declarations")
    require(isinstance(cfg['required_inputs'],list) and cfg['required_inputs'],"Nonempty required input list")
    return cfg

def environment():
    versions={}
    for name in ('numpy','scipy','torch','torchvision','pycocotools','Pillow','pyarrow','shapely','pyproj'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='MISSING'
    return {'python':platform.python_version(),'executable':sys.executable,'platform':platform.platform(),'packages':versions}

def waiver_valid(waiver,requirement_id,scope):
    fields=('requirement_id','justification','scope','approver','date','expiry','review_trigger','effect_on_claims')
    require(isinstance(waiver,dict) and all(waiver.get(k) for k in fields),"Incomplete waiver")
    require(waiver['requirement_id']==requirement_id and waiver['scope']==scope,"Waiver scope mismatch")
    require(datetime.date.fromisoformat(waiver['date'])<=datetime.date.today()<=datetime.date.fromisoformat(waiver['expiry']),"Waiver expired/not yet valid")
    return True

def output_safety(path):
    p=Path(path).resolve();allowed=HERE/'runs'
    require(p.is_relative_to(allowed) and p!=allowed,"Output must be a new run below hardening_v2/runs; historical destinations forbidden")
    require(not p.exists(),"Output already exists; no historical overwrite/resume implicit authorization")
    return p

class Checker:
    def __init__(self,path,mode):self.path=path;self.mode=mode;self.results=[];self.cfg=None
    def check(self,key,fn,critical=True):
        try:
            value=fn();status=value.get('status','PASS') if isinstance(value,dict) else 'PASS'
            require(status in ('PASS','FAIL','WARN','UNKNOWN','NOT_APPLICABLE'),"Unknown check status")
            self.results.append({'id':key,'critical':critical,'status':status,'evidence':value})
        except Exception as exc:self.results.append({'id':key,'critical':critical,'status':'FAIL','reason':str(exc),'exception':type(exc).__name__})
    def config(self):
        self.cfg=load_config(self.path);require(self.cfg['mode']==self.mode,"CLI/config mode mismatch")
        return {'config_sha256':sha256(self.path),'semantic_hash':canonical_hash(self.cfg)}
    def inputs(self):
        cfg=self.cfg
        required=set(critical_sources())|set(cfg['required_inputs'])
        return {'verified_paths':verify_manifest(cfg['inputs'],required_paths=required)}
    def data(self):
        cfg=self.cfg;operation=cfg['operation'];datasets=cfg['datasets']
        require(isinstance(datasets,list),"Dataset registry list required")
        if operation in ('training','evaluation','calibration'):require(datasets,"No datasets registered for runtime operation")
        for d in datasets:
            require(all(k in d for k in ('id','role','purpose','annotation_path','tile_paths','source_verification')),"Dataset contract incomplete")
            check_access(d['role'],d['purpose'])
            expected={'training':{'TRAIN':'training','DEVELOPMENT':'model_selection'},'evaluation':{'DEVELOPMENT':'evaluation','CONFIRMATORY_TEST':'evaluation'},'calibration':{'CALIBRATION':'calibration'}}
            if operation in expected:require(expected[operation].get(d['role'])==d['purpose'],"Dataset purpose does not match runtime operation")
            if operation!='integrity_scan':require(d['role']!='LEGACY_TEST',"Historical test model-performance access forbidden")
            for path in [d['annotation_path'],d['source_verification']]+d['tile_paths']:
                require(source_key(path) in cfg['inputs'],"Dataset source not hash registered")
            verification=json.loads(Path(d['source_verification']).read_text())
            require(verification.get('status')=='PASS',"Annotation/source verification not PASS")
            require(verification.get('actual_affine_status')=='PASS',"Actual source affine not verified")
            for key,record in verification.get('raw_source_inputs',{}).items():
                require(key in cfg['inputs'] and cfg['inputs'][key]['sha256']==record['sha256'],"Raw source provenance hash not registered or changed")
            # Identity check against the frozen native split, independent of user-supplied role/name.
            baseline=json.loads((HERE/'BASELINE_HASHES.json').read_text())['files']
            protected={v.get('sha256') for p,v in baseline.items() if '/prepared/test/' in p or p.endswith('/prepared/test_coco.json')}
            if operation!='integrity_scan':
                require(all(cfg['inputs'][source_key(p)]['sha256'] not in protected for p in [d['annotation_path']]+d['tile_paths']),"Historical test content cannot be relabelled or copied into an allowed role")
            verified=verification.get('verified_inputs',{})
            for p in [d['annotation_path']]+d['tile_paths']:
                key=source_key(p)
                require(verified.get(key)==cfg['inputs'][key]['sha256'],"Source-scan attestation stale or not bound to dataset input")
            if d['role']=='CONFIRMATORY_TEST':
                require(d.get('unexamined') is True and d.get('independence_verified') is True and d.get('provenance_verified') is True,"Confirmatory provenance/selection independence UNKNOWN")
                require(d['id'] not in ('test','historical_test','legacy_test'),"Cannot rename/reuse historical holdout")
            if operation=='training':
                nominal=np.asarray(cfg['sampling']['pixel_affine'],float)
                for p in d['tile_paths']:
                    actual=np.asarray(verification['native_images'][source_key(p)]['affine'],float)
                    require(np.allclose(nominal[[0,1,3,4]],actual[[0,1,3,4]],rtol=0,atol=1e-10),"Actual GSD/rotation differs from frozen crop geometry")
                h,w=cfg['sampling']['crop_shape']
                physical=[h*float(np.hypot(nominal[1],nominal[4])),w*float(np.hypot(nominal[0],nominal[3]))]
                require(np.allclose(cfg['comparison']['physical_FOV_m'],physical,rtol=0,atol=1e-9),"Registered physical FOV differs from pixels/affine")
        roles={d['role'] for d in datasets}
        if operation=='training':require({'TRAIN','DEVELOPMENT'}<=roles,"Training/model selection roles not registered")
        if operation=='calibration':require('CALIBRATION' in roles,"Calibration labels not registered")
        if operation in ('training','evaluation'):
            prepared=cfg['runtime'].get('prepared_root');require(prepared and Path(prepared).is_dir(),"Runtime prepared root missing")
            for d in datasets:
                split='train' if d['role']=='TRAIN' else 'validation'
                expected=Path(prepared)/f'{split}_coco.json'
                require(Path(d['annotation_path']).resolve()==expected.resolve(),"Runtime annotation binding differs from registered data")
                coco=json.loads(expected.read_text())
                native={str((Path(prepared)/split/(Path(im['file_name']).stem+'.npz')).resolve()) for im in coco['images']}
                require(native=={str(Path(p).resolve()) for p in d['tile_paths']}=={str(p.resolve()) for p in (Path(prepared)/split).glob('*.npz')},"Runtime tile enumeration differs from frozen manifest; no silent exclusion/extra tile")
        return {'dataset_roles':[{'id':d['id'],'role':d['role'],'purpose':d['purpose']} for d in datasets]}
    def spatial(self):
        cfg=self.cfg;policy=cfg['spatial_policy']
        if cfg['operation']=='synthetic_smoke':return {'status':'NOT_APPLICABLE','scope':'synthetic no dataset labels'}
        tiles=[]
        from .core import affine_from_bounds
        for item in policy.get('tiles',[]):
            require(all(k in item for k in ('npz_path','annotation_path','image_id','role')),"Spatial record must reference actual registered data")
            for p in (item['npz_path'],item['annotation_path']):require(source_key(p) in cfg['inputs'],"Spatial input not hash registered")
            coco=json.loads(Path(item['annotation_path']).read_text());im=next((im for im in coco['images'] if im['id']==item['image_id']),None)
            require(im is not None,"Unknown spatial image ID")
            with np.load(item['npz_path']) as z:valid=z['valid'].astype(bool)
            require(valid.shape==(im['height'],im['width']),"Spatial valid mask misaligned")
            from .core import metric_crs
            metric_crs(im['crs'])
            native={}
            for d in cfg['datasets']:native.update(json.loads(Path(d['source_verification']).read_text()).get('native_images',{}))
            meta=native.get(source_key(item['npz_path']))
            require(meta is not None and meta['shape']==list(valid.shape),"Actual source transform missing/misaligned")
            tiles.append({'id':str(item['npz_path']),'shape':list(valid.shape),'affine':meta['affine'],'crs':im['crs'],'role':item['role'],'valid':valid,'mosaic_id':im.get('raster_name')})
        require(tiles,"No proposed spatial tile registry supplied")
        require({str(Path(t['npz_path']).resolve()) for t in policy['tiles']}=={str(Path(p).resolve()) for d in cfg['datasets'] for p in d['tile_paths']},"Spatial registry does not cover actual dataset tile inputs")
        out=validate_split(tiles,policy,self.mode)
        if out['status']=='WARN':
            waiver_valid(cfg.get('spatial_waiver'),'SPATIAL_DEVELOPMENT_OVERLAP',cfg['operation'])
            out['status']='PASS';out['waived_exploratory_overlap']=True;out['claims']='exploratory only, no spatial independence'
        return out
    def sampling(self):
        cfg=self.cfg;s=cfg['sampling']
        if cfg['operation'] in ('integrity_scan','evaluation','calibration'):return {'status':'NOT_APPLICABLE','reason':'no training sampler execution'}
        require(all(k in s for k in ('master_seed','crop_shape','pixel_affine','min_valid','partial_policy','categories','queries','max_targets','rng_version')),"Sampling recipe incomplete")
        require(s['rng_version']=='epoch_shared_v2' and s['partial_policy'] in ('visible_fragment_retains_original_identity','ignore_explicitly_accounted'),"Undeclared RNG/fragment policy")
        require(set(s['categories'])=={'background','one_complete','two_to_four_complete','crowded','small','multilobed','touching','partial'},"Crop categories incomplete")
        from .core import pixel_area
        pixel_area(s['pixel_affine']);require(len(s['crop_shape'])==2 and min(s['crop_shape'])>0 and 0<=s['min_valid']<=1,"Invalid crop geometry/support")
        if 'Q' in cfg['methods']:check_capacity(s['max_targets'],s['queries'])
        if cfg['operation']=='training' and cfg['runtime'].get('sampler_adapter')=='planned_native_crowns_v2':
            path=s.get('plan');require(path and source_key(path) in cfg['inputs'],"Frozen sampling plan not hash registered")
            plan=json.loads(Path(path).read_text());require(plan.get('status')=='PASS' and plan.get('records'),"Sampling plan incomplete or infeasible")
            require(plan.get('unmet_quotas')=={} and plan.get('selected_capacity_overflows')==0,"Sampling plan hides missing categories/capacity overflow")
            for record in plan['records']:
                require(all(k in record for k in ('tile_name','split','window_yxhw','annotation_ids','visible_target_count','categories')),"Crop plan record incomplete")
                require(record['visible_target_count']<=s['max_targets'],"Sampling plan exceeds registered target capacity")
                if 'Q' in cfg['methods']:check_capacity(record['visible_target_count'],s['queries'])
            from .sampling import validate_frozen_plan
            from .runtime import reference_metadata
            validate_frozen_plan(plan,cfg['runtime']['prepared_root'],reference_metadata(cfg),s['max_targets'],s['min_valid'],s['queries'] if 'Q' in cfg['methods'] else None)
        require(isinstance(s['master_seed'],int),"Integer master seed required")
        return {'sampling_policy':s}
    def selection(self):
        s=self.cfg['selection'];require(all(k in s for k in ('stage','threshold','threshold_source_role','min_area_m2','duplicate_policy','invalid_support_policy','E2E_overlap','structural_overlap','max_dets')),"Output selection contract incomplete")
        require(s['stage']==2 and s['E2E_overlap']=='highest_score' and s['structural_overlap']=='unique_cover',"Unapproved output/grouping policy change")
        require(s['threshold_source_role'] in ('TRAIN','DEVELOPMENT','CALIBRATION','PREREGISTERED'),"Forbidden threshold labels")
        require(isinstance(s['threshold'],(int,float)) and 0<=s['threshold']<=1 and s['min_area_m2']>=0 and s['max_dets']>0,"Invalid numeric selection policy")
        require(s['duplicate_policy'] in ('keep','exact_geometry_highest_score') and s['invalid_support_policy'] in ('keep_original','intersect_valid','reject_any_invalid'),"Selection policy unknown")
        return s
    def calibration(self):
        c=self.cfg['calibration'];require(isinstance(c,dict) and 'enabled' in c,"Calibration policy missing")
        if not c['enabled']:return {'status':'NOT_APPLICABLE','reason':'No calibration or fitting permitted by this recipe'}
        require(all(c.get(k) for k in ('dataset_id','event','fold_manifest','policy')),"Calibration event/provenance incomplete")
        require(any(d['id']==c['dataset_id'] and d['role']=='CALIBRATION' for d in self.cfg['datasets']),"Calibration dataset role forbidden")
        require(source_key(c['fold_manifest']) in self.cfg['inputs'],"Fold manifest not hash registered")
        folds=json.loads(Path(c['fold_manifest']).read_text());require(folds.get('status')=='PASS' and folds.get('spatial_policy')==c['policy'],"Calibration spatial evidence missing/mismatch")
        from .calibration import buffered_folds
        require(buffered_folds(folds['tiles'],folds['assignment'],c['policy'])['status']=='PASS',"Calibration fold evidence fails independent spatial replay")
        return c
    def comparison(self):
        cfg=self.cfg
        if cfg['study_type']=='infrastructure':require(cfg['operation'] in ('integrity_scan','synthetic_smoke'),"Infrastructure config cannot authorize model runs");return {'status':'NOT_APPLICABLE','scope':'infrastructure only'}
        c=cfg['comparison'];keys=('encoder_checkpoint','input_channels','feature_levels','input_resolution','physical_FOV_m','train_distribution','augmentations','supervision_by_method','optimizer','updates','seeds','decoder_depth','trainable_parameters','GPU','VRAM_GB','TTA','postprocessing','evaluator_version','recipe_by_method')
        require(all(k in c and c[k] is not None for k in keys),"Comparison capacity/budget/supervision recipe incomplete")
        checkpoint=source_key(c['encoder_checkpoint']);require(checkpoint in cfg['inputs'],"Encoder not hash registered")
        require(c['updates']>0 and c['seeds'] and all(type(x) is int and 0<=x<2**32 for x in c['seeds']) and len(set(c['seeds']))==len(c['seeds']),"Training budget/seeds invalid or duplicated")
        if cfg['study_type']=='controlled_frozen_features':require(c.get('encoder_frozen') is True,"Controlled feature probe must freeze encoder")
        for method in cfg['methods']:
            require(method in c['supervision_by_method'] and method in c['trainable_parameters'],"Per-method supervision/capacity omitted")
            recipe=c['recipe_by_method'].get(method)
            require(recipe and Path(recipe).is_file(),f"Executable recipe missing for {method}")
            # train_treeflow_m3.py: EXP-028 M3 variant of F (dossier HOAN_THANH_DU_AN/12), user-approved 2026-11-20
            supported={'F':{REPO/'crown_segmentation_research/clean_benchmark/train_treeflow.py',REPO/'crown_segmentation_research/clean_benchmark/train_treeflow_m3.py'},
                       'MASKRCNN':{REPO/'crown_segmentation_research/clean_benchmark/train_maskrcnn.py'}}
            require(method in supported and Path(recipe).resolve() in supported[method],f"No registered executable benchmark recipe for {method}; do not substitute an F trainer for Q/G")
        measurement=cfg['measurement']
        require(measurement.get('reference')=='original_COCO_RLE' and measurement.get('structural_stage')==1 and measurement.get('E2E_stage')==2,"Measurement reference/stage not frozen")
        require(measurement.get('primary_endpoints') and measurement.get('core_modes')==['r1','r4','adaptive'],"Endpoints/core policy missing")
        for key in ('pair_cache','candidate_reference_manifest'):
            p=measurement.get(key)
            require(p and source_key(p) in cfg['inputs'],f"Fixed {key} not hash registered")
        strata=json.loads(Path(measurement['candidate_reference_manifest']).read_text())
        require(strata.get('reference_model_sha256') and strata.get('reference_stage') is not None and strata.get('reference_support')=='original_COCO_RLE',"C0/C1 reference is not explicit and fixed across heads")
        return c
    def tests(self):
        import unittest
        from . import test_primitives,test_runtime
        import io
        suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in (test_primitives,test_runtime)])
        result=unittest.TextTestRunner(stream=io.StringIO(),verbosity=0).run(suite)
        require(result.testsRun>0 and result.wasSuccessful() and not result.skipped,"Metric/core regression suite failed or skipped")
        return {'run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped)}
    def runtime_spec(self):
        from .runtime import validate_runtime_spec
        return validate_runtime_spec(self.cfg)
    def run(self):
        self.check('CONFIG',self.config)
        if self.cfg is not None and self.results[0]['status']=='PASS':
            for name,fn in [('HASH',self.inputs),('DATA',self.data),('SPATIAL',self.spatial),('SAMPLING',self.sampling),('OUTPUT',self.selection),('CALIBRATION',self.calibration),('COMPARISON',self.comparison),('RUNTIME_SPEC',self.runtime_spec),('DESTINATION',lambda: {'new_output':str(output_safety(self.cfg['output_dir']))}),('TESTS',self.tests)]:
                previous={r['id']:r['status'] for r in self.results}
                prerequisite='HASH' if name=='DATA' else 'DATA' if name in ('SPATIAL','SAMPLING','CALIBRATION') else None
                if prerequisite and previous.get(prerequisite)!='PASS':self.results.append({'id':name,'critical':True,'status':'UNKNOWN','reason':f'Prerequisite {prerequisite} not PASS; source not accessed'})
                else:self.check(name,fn)
        good=bool(self.results) and all(r['status'] in ('PASS','NOT_APPLICABLE') for r in self.results)
        independent=bool(self.cfg) and any(d.get('role')=='CONFIRMATORY_TEST' and d.get('independence_verified') for d in self.cfg.get('datasets',[]))
        if self.mode=='confirmatory' and independent:
            def protocol():
                p=self.cfg.get('confirmatory_protocol')
                require(p and source_key(p) in self.cfg['inputs'],"Confirmatory protocol not hash registered")
                doc=json.loads(Path(p).read_text())
                require(all(doc.get(k) for k in ('approved_by','date','primary_endpoints','margins','model_selection_frozen','calibration_frozen','holdout_registry')),"Confirmatory protocol/approval incomplete")
                return {'protocol_sha256':sha256(p)}
            self.check('CONFIRMATORY_PROTOCOL',protocol)
            good=good and self.results[-1]['status']=='PASS'
        if self.mode=='confirmatory' and not independent:self.results.append({'id':'EXTERNAL_HOLDOUT','critical':True,'status':'FAIL','reason':'EXTERNAL_HOLDOUT_REQUIRED'});good=False
        return {'version':'2.0.0','overall_status':'PASS' if good else 'FAIL','mode':self.mode,'operation':self.cfg.get('operation') if self.cfg else None,
                'checks':self.results,'environment':environment(),'config_path':str(self.path),
                'scope':'This gate validates the supplied operation; synthetic/integrity PASS does not authorize F/G/Q benchmark',
                'EXTERNAL_HOLDOUT_REQUIRED':not independent}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--mode',required=True,choices=['exploratory','confirmatory']);ap.add_argument('--report-dir')
    args=ap.parse_args();report=Checker(args.config,args.mode).run()
    if args.report_dir:
        dest=Path(args.report_dir)
        try:dest.mkdir(parents=True,exist_ok=False);write_new(dest/'READINESS_REPORT.json',report)
        except OSError as exc:report['overall_status']='FAIL';report['report_write_error']=str(exc)
    print(json.dumps(report,indent=2,allow_nan=False));return 0 if report['overall_status']=='PASS' else 1

if __name__=='__main__':sys.exit(main())
