"""Read-only independent COCO rescoring; writes exclusively to this audit directory.

Run from repository root using cs2_venv/bin/python -B.
No training, inference, tuning, or modification of original results.
"""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import time
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'clean_benchmark/results'
OUT = Path(__file__).resolve().parent
PREP = ROOT.parent / 'data/itc_benchmarks/canopyrs_quebectrees/prepared'
CLASSES = {'Tiny': (0, 9), 'Small': (9, 25), 'Medium': (25, 49), 'Large': (49, 100), 'Giant': (100, 1e5)}

def score_one(task):
    name, seed, row = task
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(PREP / "test_coco.json"))
    ids = {im["id"]: im for im in coco.dataset["images"]}
    start = time.time()
    f = RES / row['file']
    raw = f.read_bytes()
    dets = json.loads(raw)
    bad = []
    for i, d in enumerate(dets):
        if d['image_id'] not in ids or d['category_id'] != 1 or not np.isfinite(d['score']):
            bad.append(i)
        elif d['segmentation']['size'] != [ids[d['image_id']]['height'], ids[d['image_id']]['width']]:
            bad.append(i)
    with contextlib.redirect_stdout(io.StringIO()):
        e = COCOeval(coco, coco.loadRes(dets), 'segm')
        e.params.maxDets = [1, 10, 300, 400]
        e.params.areaRng = [[0, 1e10]] + [[lo / .03**2, hi / .03**2] for lo, hi in CLASSES.values()]
        e.params.areaRngLbl = ['all'] + list(CLASSES)
        e.evaluate()
        e.accumulate()
    p = e.eval['precision']
    def ap(x):
        v = x[x > -1]
        return float(v.mean() * 100) if len(v) else None
    m = {'AP': ap(p[:, :, :, 0, 3]), 'AP50': ap(p[np.isclose(e.params.iouThrs, .5), :, :, 0, 3]),
         'AP75': ap(p[np.isclose(e.params.iouThrs, .75), :, :, 0, 3]), 'AP_maxdet300': ap(p[:, :, :, 0, 2])}
    m.update({f'AP_{c}': ap(p[:, :, :, i+1, 3]) for i,c in enumerate(CLASSES)})
    delta = {k: round(m[k], 2) - row[k] for k in m if m[k] is not None and row[k] is not None}
    rec = {'method': name, 'seed': seed, 'file': row['file'], 'sha256': hashlib.sha256(raw).hexdigest(),
           'n_pred': len(dets), 'invalid_detections': bad, 'metrics': m, 'delta_vs_table_rounded': delta,
           'seconds': round(time.time()-start, 2)}
    return rec

def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    table = json.load(open(RES / 'master_table_test.json'))
    output = OUT / 'recomputed_metrics.json'
    report = json.load(open(output)) if output.exists() else {'protocol': 'Independent COCOeval on saved masks; maxDets 300/400, original GT', 'rows': [], 'methods': {}}
    finished = {(r['method'], r['seed']) for r in report['rows']}
    tasks = [(name, seed, row) for name, method in table['methods'].items() for seed, row in method['seeds'].items() if (name,seed) not in finished]
    print('Resuming',len(finished),'completed;',len(tasks),'remaining; 6 CPU processes',flush=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(score_one,t) for t in tasks]):
            rec = future.result()
            report['rows'].append(rec)
            json.dump(report, open(output, 'w'), indent=2)
            print(rec['method'],rec['seed'],'AP',round(rec['metrics']['AP'],6),'max delta',max(abs(v) for v in rec['delta_vs_table_rounded'].values()),'seconds',rec['seconds'],flush=True)
    for name in table['methods']:
        metrics = [r['metrics'] for r in report['rows'] if r['method']==name]
        report['methods'][name] = {k: {'mean': float(np.mean([m[k] for m in metrics if m[k] is not None])), 'std': float(np.std([m[k] for m in metrics if m[k] is not None],ddof=1))} for k in metrics[0]}
    report['completed'] = len(report['rows']) == sum(len(m['seeds']) for m in table['methods'].values())
    json.dump(report,open(output,'w'),indent=2)

if __name__ == '__main__':
    main()
