"""Small CPU counterexamples and audit checks; original files are read-only."""
from pathlib import Path
import ast
import contextlib
import collections
import hashlib
import importlib
import io
import json
import sys
import tempfile
import numpy as np
import torch
from torch.utils.data import DataLoader
from pycocotools import mask as mu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
OUT = Path(__file__).resolve().parent
RES = ROOT / 'clean_benchmark/results'
PREP = ROOT.parent / 'data/itc_benchmarks/canopyrs_quebectrees/prepared'

def main():
    report = {}
    # Extract only the pure metric function, avoiding legacy module's environment manipulation.
    path = ROOT / 'methods/tree_flow/hamiltonian_watershed.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate_metrics')
    ns = {'np': np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    gt_map = np.zeros((64,64),np.int32); gt_map[2:18,2:18] = 2
    pred = gt_map.copy(); pred[40:50,40:50] = 3
    insts = [{'id':2,'area':256,'score':.9},{'id':3,'area':100,'score':.1}]
    old = ns['evaluate_metrics'](pred,insts,gt_map)
    def rle(m):
        r = mu.encode(np.asfortranarray(m.astype(np.uint8)))
        return {'size':r['size'],'counts':r['counts'].decode()}
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(); coco.dataset = {'images':[{'id':1,'height':64,'width':64}], 'categories':[{'id':1,'name':'tree'}],
        'annotations':[{'id':1,'image_id':1,'category_id':1,'segmentation':rle(gt_map==2),'area':256,'bbox':[2,2,16,16],'iscrowd':0}]}
        coco.createIndex()
        dets = [{'image_id':1,'category_id':1,'segmentation':rle(pred==i['id']),'score':i['score']} for i in insts]
        e = COCOeval(coco,coco.loadRes(dets),'segm'); e.evaluate(); e.accumulate(); e.summarize()
    report['legacy_F1_called_AP'] = {'reported_AP50':old['ap50'],'true_COCO_AP50':float(e.stats[1]*100), 'actual_F1_50':2/3*100}
    # Verify custom bootstrap accumulation including repeated tiles against official COCOeval.
    bg = importlib.import_module('crown_segmentation_research.clean_benchmark.bootstrap_gaps')
    recs = []
    for item in e.evalImgs[:len(e.params.imgIds)]:
        recs.append((np.asarray(item['dtScores']),item['dtMatches']>0,item['dtIgnore'].astype(bool),int((~np.asarray(item['gtIgnore'],bool)).sum()),0,0))
    report['bootstrap_toy'] = {'official_AP':float(e.stats[0]*100),'custom_AP':bg.ap_from(recs,[0],e.params.recThrs),
                              'custom_duplicated_tile_AP':bg.ap_from(recs,[0,0],e.params.recThrs)}
    # Same i in the same persistent worker receives the same seed after each epoch.
    D = importlib.import_module('crown_segmentation_research.clean_benchmark.dataset')
    M = importlib.import_module('crown_segmentation_research.clean_benchmark.train_maskrcnn')
    with tempfile.TemporaryDirectory(dir='/tmp') as td:
        temp = Path(td); (temp/'train').mkdir()
        rng = np.random.default_rng(17)
        lab = np.zeros((96,96),np.int32); lab[10:40,12:50]=1; lab[52:87,49:87]=2
        np.savez(temp/'train/toy.npz',image=rng.integers(0,256,(96,96,3),dtype=np.uint8),valid=np.ones((96,96),bool),label=lab)
        old_d,old_m = D.PREP,M.PREP; D.PREP = M.PREP = temp
        try:
            checks = {}
            for name,ds in [('TreeFlowNet',D.CanopyCropDataset('train',crop=64,crops_per_tile=4,augment=True)),
                            ('MaskRCNN',M.CanopyInstanceCrops('train',crop=64,crops_per_tile=4,augment=True))]:
                dl = DataLoader(ds,batch_size=1,num_workers=2,persistent_workers=True,shuffle=False)
                def collect():
                    return [hashlib.sha256((b['image'] if name=='TreeFlowNet' else b[0]).numpy().tobytes()).hexdigest() for b in dl]
                a,b = collect(),collect()
                checks[name] = {'same_images_across_epochs':sum(x==y for x,y in zip(a,b)),'n_samples':len(a),'epoch1':a,'epoch2':b}
                del dl
            report['persistent_worker_crops'] = checks
        finally:
            D.PREP,M.PREP=old_d,old_m
    # Inputs and labels: metadata, area, unique IDs and sample prepared mask consistency.
    splits = {}
    for split in ['train','validation','test']:
        obj = json.load(open(PREP/f'{split}_coco.json'))
        area_bad=[]
        for a in obj['annotations']:
            if abs(float(mu.area(a['segmentation']))-a['area'])>.1: area_bad.append(a['id'])
        samples=[]
        for im in [obj['images'][0],obj['images'][len(obj['images'])//2],obj['images'][-1]]:
            z = np.load(PREP/split/(Path(im['file_name']).stem+'.npz'))
            anns = [a for a in obj['annotations'] if a['image_id']==im['id']]
            lbl = np.zeros(z['label'].shape,np.int32)
            for k,a in enumerate(anns,1):
                m = mu.decode(a['segmentation']).astype(bool); lbl[m & (lbl==0)] = k
            samples.append({'image_id':im['id'],'label_matches_RLE_priority':bool(np.array_equal(lbl,z['label'])),
                            'invalid_rgb_nonzero_pixels':int(np.count_nonzero(z['image'][~z['valid']].sum(-1))),
                            'labels_outside_valid':int(np.count_nonzero(z['label'][~z['valid']]))})
        splits[split] = {'n_images':len(obj['images']),'n_annotations':len(obj['annotations']), 'area_mismatches':area_bad,
                         'unique_image_ids':len({im['id'] for im in obj['images']}),'unique_annotation_ids':len({a['id'] for a in obj['annotations']}),
                         'iscrowd':dict(collections.Counter(a['iscrowd'] for a in obj['annotations'])),
                         'rasters':dict(collections.Counter(im['raster_name'] for im in obj['images'])), 'samples':samples}
    report['dataset_checks'] = splits
    # Selections: winning configuration, actual file, and frozen test config.
    selections=[]
    for f in sorted(RES.glob('*_val_selection.json')):
        s=json.load(open(f)); chosen=s['runs'][s['chosen_tag']]
        t=RES/(f.name.replace('_val_selection.json','_test.json'))
        test=json.load(open(t)) if t.exists() else None
        config_mismatch = {k:[v,test['cfg'].get(k)] for k,v in s['chosen'].items() if test and v!=test['cfg'].get(k)}
        selections.append({'file':f.name,'ckpt_exists':(ROOT.parent/s['ckpt']).exists(),'n_trials':len(s['runs']),
                           'chosen_is_best_reported_AP':chosen['AP']==max(r['AP'] for r in s['runs'].values()),
                           'test_cfg_mismatches':config_mismatch,'test_exists':t.exists()})
    report['selections']=selections
    # Recompute the supposedly blind visual study from the CSV, never using it as human annotation validation.
    import csv
    from sklearn.metrics import cohen_kappa_score
    keys=json.load(open(ROOT/'blind_visual_exp019/_KEY/key.json'))
    answers={r[0]:r[1].strip() for r in list(csv.reader(open(ROOT/'blind_visual_exp019/answers_agent.csv')))[1:]}
    report['blind_visual_recalculation']={'n':len(keys),'accuracy':float(np.mean([keys[k]['outcome']==answers[k] for k in keys])),
       'kappa':float(cohen_kappa_score([keys[k]['outcome'] for k in keys],[answers[k] for k in keys]))}
    json.dump(report,open(OUT/'probe_results.json','w'),indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ['selections','dataset_checks']},indent=2),flush=True)

if __name__=='__main__':
    main()
