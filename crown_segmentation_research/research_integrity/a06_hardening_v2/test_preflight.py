"""Positive/negative operation gates and real CLI non-execution tests."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from .core import REPO,sha256
from .preflight import Checker,HERE,critical_sources,load_config,waiver_valid

def fixture_config():
    source=HERE/'core.py';key=str(source.relative_to(REPO))
    return {'version':'2.0.0','mode':'exploratory','operation':'synthetic_smoke','study_type':'infrastructure','methods':['Q'],
        'inputs':{key:{'sha256':sha256(REPO/key)} for key in critical_sources()},'required_inputs':critical_sources(),'datasets':[],
        'output_dir':str(HERE/'runs'/('synthetic-'+uuid.uuid4().hex)),
        'spatial_policy':{'version':'fixture-only','support':'full_footprint','buffer_m':0},
        'sampling':{'master_seed':5,'crop_shape':[12,12],'pixel_affine':[.03,0,0,0,-.03,0],'min_valid':1,
                    'partial_policy':'visible_fragment_retains_original_identity','categories':['background','one_complete','two_to_four_complete','crowded','small','multilobed','touching','partial'],
                    'queries':4,'max_targets':4,'rng_version':'epoch_shared_v2'},
        'selection':{'stage':2,'threshold':.5,'threshold_source_role':'PREREGISTERED','min_area_m2':0,
                    'duplicate_policy':'keep','invalid_support_policy':'keep_original','E2E_overlap':'highest_score','structural_overlap':'unique_cover','max_dets':400},
        'calibration':{'enabled':False},'comparison':{},'runtime':{'entrypoint':None,'cli':{}},'measurement':{'scope':'synthetic only; no biological/benchmark endpoint selected'}}

class GateTests(unittest.TestCase):
    def check_cfg(self,cfg,mode='exploratory'):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'config.json';p.write_text(json.dumps(cfg));return Checker(p,mode).run()
    def test_positive_complete_synthetic_operation(self):
        r=self.check_cfg(fixture_config());self.assertEqual(r['overall_status'],'PASS',r)
        self.assertTrue(next(x for x in r['checks'] if x['id']=='TESTS')['evidence']['run']>0)
    def test_missing_config(self):self.assertEqual(Checker('/not/existing/config.json','exploratory').run()['overall_status'],'FAIL')
    def test_malformed_config(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bad';p.write_text('{ invalid JSON')
            self.assertEqual(Checker(p,'exploratory').run()['overall_status'],'FAIL')
    def test_empty_and_incomplete_configs(self):
        for cfg in ({},{'mode':'exploratory'}):self.assertEqual(self.check_cfg(cfg)['overall_status'],'FAIL')
    def test_empty_missing_incomplete_and_bad_hash(self):
        c=fixture_config();key=c['required_inputs'][0]
        for manifest in ({},{key:'MISSING'},{key:{'sha256':'0'*64}}):
            cfg=copy.deepcopy(c);cfg['inputs']=manifest;self.assertEqual(self.check_cfg(cfg)['overall_status'],'FAIL')
    def test_query_capacity_rejected(self):
        c=fixture_config();c['sampling']['max_targets']=7;self.assertEqual(self.check_cfg(c)['overall_status'],'FAIL')
    def test_unknown_sampler_policy_rejected(self):
        c=fixture_config();c['sampling']['rng_version']='helper_not_integrated';self.assertEqual(self.check_cfg(c)['overall_status'],'FAIL')
    def test_threshold_on_test_forbidden(self):
        c=fixture_config();c['selection']['threshold_source_role']='CONFIRMATORY_TEST';self.assertEqual(self.check_cfg(c)['overall_status'],'FAIL')
    def test_calibration_test_label_rejected(self):
        c=fixture_config();c['calibration']={'enabled':True,'dataset_id':'test','event':'IoU>=.5','fold_manifest':'bad','policy':{}}
        self.assertEqual(self.check_cfg(c)['overall_status'],'FAIL')
    def test_role_test_training_rejected(self):
        c=fixture_config();c['datasets']=[{'id':'x','role':'CONFIRMATORY_TEST','purpose':'training','annotation_path':'bad','tile_paths':[],'source_verification':'bad'}]
        self.assertEqual(self.check_cfg(c)['overall_status'],'FAIL')
    def test_historical_destination_rejected(self):
        c=fixture_config();c['output_dir']=str(REPO/'crown_segmentation_research/clean_benchmark/results')
        self.assertEqual(self.check_cfg(c)['overall_status'],'FAIL')
    def test_existing_destination_rejected(self):
        c=fixture_config();c['output_dir']=str(HERE/'runs');self.assertEqual(self.check_cfg(c)['overall_status'],'FAIL')
    def test_mode_mismatch_and_confirmatory_without_holdout(self):
        self.assertEqual(self.check_cfg(fixture_config(),'confirmatory')['overall_status'],'FAIL')
        c=fixture_config();c['mode']='confirmatory';r=self.check_cfg(c,'confirmatory')
        self.assertEqual(r['overall_status'],'FAIL');self.assertTrue(r['EXTERNAL_HOLDOUT_REQUIRED'])
    def test_raw_query_is_not_accepted_stage(self):
        c=fixture_config();c['selection']['stage']=0;self.assertEqual(self.check_cfg(c)['overall_status'],'FAIL')
    def test_incomplete_comparison_does_not_become_ready(self):
        c=fixture_config();c['study_type']='controlled_frozen_features';self.assertEqual(self.check_cfg(c)['overall_status'],'FAIL')
    def test_unknown_spatial_source_is_not_PASS(self):
        c=fixture_config();c['operation']='integrity_scan';self.assertEqual(self.check_cfg(c)['overall_status'],'FAIL')
    def test_partial_waiver_is_invalid(self):
        with self.assertRaises(ValueError):waiver_valid({'requirement_id':'X'},'X','training')
    def test_runtime_tag_path_escape_rejected_without_model_load(self):
        from .runtime import validate_runtime_spec
        c=fixture_config();c['operation']='evaluation';c['methods']=['F']
        c['comparison']={'feature_levels':{'F':['P2']},'TTA':False}
        c['runtime']={'entrypoint':'crown_segmentation_research/clean_benchmark/evaluate.py','cli':{'model':'treeflow','split':'validation','tag':'../../history'}}
        with self.assertRaises(ValueError):validate_runtime_spec(c)
    def test_real_preflight_CLI_positive_and_negative(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'cfg.json';p.write_text(json.dumps(fixture_config()))
            command=[sys.executable,'-B','-m','crown_segmentation_research.research_integrity.a06_hardening_v2.preflight','--config',str(p),'--mode','exploratory']
            run=subprocess.run(command,cwd=REPO,capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            p.write_text('{ invalid JSON');run=subprocess.run(command,cwd=REPO,capture_output=True,text=True);self.assertNotEqual(run.returncode,0)
    def test_active_entrypoints_require_gate_before_checkpoint_or_data(self):
        cases=[('train_treeflow.py',['--out','/not/a/run']),('train_maskrcnn.py',['--out','/not/a/run']),
               ('evaluate.py',['--model','treeflow','--ckpt','/not/a/checkpoint','--tag','x'])]
        for script,args in cases:
            command=[sys.executable,'-B',str(REPO/'crown_segmentation_research/clean_benchmark'/script)]+args
            run=subprocess.run(command,cwd=REPO,capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0);self.assertIn('integrity-config',run.stderr)
            self.assertNotIn('FileNotFoundError',run.stderr)
    def test_legacy_oracle_and_test_scorer_blocked_before_run(self):
        for script in ('oracle_seed_nms.py','exp021_veto_merge.py','seed_scorer_test.py'):
            run=subprocess.run([sys.executable,'-B',str(REPO/'crown_segmentation_research/clean_benchmark'/script)],cwd=REPO,capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0);self.assertIn('INTEGRITY GATE',run.stdout+run.stderr)
    def test_all_registered_entrypoint_gates_exist(self):
        for registry in ('ENTRYPOINT_REGISTRY.json','ADDITIONAL_ENTRYPOINT_REGISTRY.json','HISTORICAL_RUNNER_REGISTRY.json'):
            for item in json.loads((HERE/registry).read_text()):
                text=(REPO/item['path']).read_text()
                if item['status']=='LEGACY_BLOCKED':self.assertIn('reject_legacy_entrypoint(__file__)',text,item['path'])
                if item['status']=='ACTIVE_GATED':self.assertIn('gate(args,',text,item['path'])

if __name__=='__main__':unittest.main()
