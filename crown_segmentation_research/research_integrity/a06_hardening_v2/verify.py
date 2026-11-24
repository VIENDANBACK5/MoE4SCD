"""Execute local synthetic tests and verify preservation; never run a research benchmark."""
import argparse
import contextlib
import datetime
import importlib.util
import io
import json
import subprocess
import sys
import time
import unittest
import zipfile
from pathlib import Path
from .core import REPO,require,sha256,write_new
from .preflight import HERE,environment

class RecordedResult(unittest.TextTestResult):
    def startTest(self,test):super().startTest(test);self.started=time.perf_counter()
    def record(self,test,status,detail=None):
        self.records.append({'id':test.id(),'status':status,'seconds':time.perf_counter()-self.started,'detail':detail})
    def addSuccess(self,test):super().addSuccess(test);self.record(test,'PASS')
    def addFailure(self,test,err):super().addFailure(test,err);self.record(test,'FAIL',self._exc_info_to_string(err,test))
    def addError(self,test,err):super().addError(test,err);self.record(test,'ERROR',self._exc_info_to_string(err,test))
    def addSkip(self,test,reason):super().addSkip(test,reason);self.record(test,'SKIP',reason)
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.records=[]

def preservation():
    baseline=json.loads((HERE/'BASELINE_HASHES.json').read_text())['files']
    additional=json.loads((HERE/'ADDITIONAL_BASELINE_HASHES.json').read_text())
    additional.update(json.loads((HERE/'HISTORICAL_RUNNER_BASELINE.json').read_text()))
    for p,v in additional.items():
        if p in baseline:require(baseline[p]['sha256']==v['sha256'],"Pre-patch source baseline disagrees")
        baseline[p]=v
    allowed=json.loads((HERE/'AUTHORIZED_SOURCE_CHANGES.json').read_text())
    modified=[];unchanged=0;bad=[]
    for name,record in baseline.items():
        path=Path(name)
        digest=sha256(path) if path.is_file() else None
        if digest==record.get('sha256'):unchanged+=1
        elif name in allowed and allowed[name]['before_sha256']==record['sha256'] and allowed[name]['after_sha256']==digest:modified.append(name)
        else:bad.append({'path':name,'expected':record.get('sha256'),'actual':digest})
    archived=0
    for filename in ('ORIGINAL_IMPLEMENTATION.zip','ADDITIONAL_ENTRYPOINT_ORIGINALS.zip','HISTORICAL_RUNNER_ORIGINALS.zip'):
        with zipfile.ZipFile(HERE/filename) as z:
            for n in z.namelist():
                path=str(REPO/n);digest=__import__('hashlib').sha256(z.read(n)).hexdigest()
                require(path in baseline and digest==baseline[path]['sha256'],"Archived source differs from before-state")
                archived+=1
    raw=json.loads((HERE/'source_scan/raw_source_hashes.json').read_text())
    raw_bad=[p for p,r in raw.items() if sha256(REPO/p)!=r['sha256']]
    # Check the previous independent review and its byte-exact snapshot; do not overwrite it.
    previous=HERE.parent/'a06_independent_validation'
    old=json.loads((previous/'REVIEW_ARTIFACT_HASHES.json').read_text())['files']
    review_bad=[n for n,r in old.items() if sha256(previous/n)!=r['sha256']]
    frozen=json.loads((previous/'review_source_freeze.json').read_text());snapshot_bad=[]
    for n,digest in frozen.items():
        if '/a06_hardening/' in n and sha256(previous/'reviewed_a06_snapshot'/Path(n).name)!=digest:snapshot_bad.append(n)
    return {'status':'PASS' if not (bad or raw_bad or review_bad or snapshot_bad) else 'FAIL','frozen_paths':len(baseline),
        'unchanged_registered_files':unchanged,'authorized_implementation_changes':modified,'unauthorized_changes':bad,
        'archived_source_files_verified':archived,'raw_Parquet_checked':len(raw),'raw_Parquet_mismatches':raw_bad,
        'prior_review_mismatches':review_bad,'A06_snapshot_mismatches':snapshot_bad,
        'historical_data_labels_checkpoints_predictions_reports_preserved':not (bad or raw_bad or review_bad or snapshot_bad),
        'inherited_missing_A06_original_files':'23 source files already absent before this task; preserved snapshot retained, not restored over worktree'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--synthetic-only',action='store_true');args=ap.parse_args()
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    from . import test_primitives,test_runtime,test_preflight
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in (test_primitives,test_runtime,test_preflight)])
    text=io.StringIO();start=time.perf_counter()
    result=unittest.TextTestRunner(stream=text,verbosity=2,resultclass=RecordedResult).run(suite)
    (dest/'UNIT_INTEGRATION.log').write_text(text.getvalue())
    related=[]
    # Established source tests use temporary synthetic data/checkpoints, never real model predictions.
    spec=importlib.util.spec_from_file_location('related_clean_tests',REPO/'tests/test_clean_benchmark.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    related_log=io.StringIO()
    with contextlib.redirect_stdout(related_log),contextlib.redirect_stderr(related_log):
        for name,fn in vars(mod).items():
            if name.startswith('test_') and callable(fn):
                t=time.perf_counter()
                try:fn();related.append({'id':'tests/test_clean_benchmark.py::'+name,'status':'PASS','seconds':time.perf_counter()-t})
                except Exception as exc:related.append({'id':name,'status':'FAIL','exception':repr(exc),'seconds':time.perf_counter()-t})
    (dest/'RELATED_TESTS.log').write_text(related_log.getvalue())
    ok=result.wasSuccessful() and not result.skipped and all(r['status']=='PASS' for r in related)
    report={'status':'PASS' if ok else 'FAIL','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
            'cases':result.records,'related_tests':related,'unique_test_total':result.testsRun+len(related),
            'seconds':time.perf_counter()-start,'environment':environment(),'command':sys.argv,
            'not_executed':['full training','oracle experiments','historical test model-performance evaluation','calibration fitting','hyperparameter search'],
            'nested_preflight_selftests':'Repeated executions of the same primitives/loader tests are not counted as additional unique tests.'}
    write_new(dest/'TEST_RESULTS.json',report)
    if not args.synthetic_only:
        keep=preservation();write_new(dest/'PRESERVATION_CHECK.json',keep);ok=ok and keep['status']=='PASS'
    print(json.dumps({'status':'PASS' if ok else 'FAIL','unique_tests':report['unique_test_total'],'output':str(dest)}));return 0 if ok else 1

if __name__=='__main__':sys.exit(main())
