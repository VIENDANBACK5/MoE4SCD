"""TRAIN-only privileged readout diagnostic; never deployment predictions."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'
from pathlib import Path
import sys, time, json, hashlib, csv, traceback
import numpy as np

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[3]
OLD = REPO / 'research/ITC_MASK_DINO_FORMATION_V1_20261104'
sys.path.insert(0, str(OLD))
import bootstrap
import torch
import torch.nn.functional as F
from carrier import build_carrier, features
from train_data import load_window
from pycocotools import mask as mu

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
RESULTS = EXP / 'results'
START = time.monotonic()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''): h.update(chunk)
    return h.hexdigest()

def emit(**record):
    record['elapsed_s'] = round(time.monotonic() - START, 3)
    print(json.dumps(record), flush=True)
    with (RESULTS / 'events.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')

def guard():
    if time.monotonic() - START > 1200: raise RuntimeError('Registered 20 minute wall limit')
    if torch.cuda.max_memory_allocated() > 8 * 2**30:
        raise RuntimeError('Registered 8 GiB allocated memory limit')

def pack(mask):
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    r['counts'] = r['counts'].decode('ascii')
    return r

@torch.no_grad()
def evaluate(embeddings, field, valid, targets, save_masks=False):
    """Full native-pixel IoU; outputs are independent, valid-clipped masks."""
    matrix = []; encoded = []
    tf = targets.flatten(1).float()
    ta = tf.sum(1)
    for j in range(0, len(embeddings), 16):
        logits = torch.einsum('qc,chw->qhw', embeddings[j:j+16], field)
        logits = F.interpolate(logits[None], valid.shape, mode='bilinear', align_corners=False)[0]
        masks = (logits > 0) & valid
        mf = masks.flatten(1).float()
        inter = mf @ tf.T
        union = mf.sum(1)[:, None] + ta[None] - inter
        matrix.append((inter / union.clamp_min(1)).cpu())
        if save_masks: encoded.extend(pack(m.cpu().numpy()) for m in masks)
        guard()
    return torch.cat(matrix), encoded

def main():
    assert not (RESULTS / 'COMPLETE.json').exists(), 'Do not overwrite a completed run'
    assert torch.cuda.is_available()
    checkpoint = OLD / 'checkpoints/mask_only_seed42/final.pt'
    expected = json.loads((OLD / 'logs/mask_only_seed42/STATUS.json').read_text())['checkpoint_sha256']
    actual = sha(checkpoint); assert actual == expected
    rows = json.loads((EXP / 'SELECTED_TRAIN_WINDOWS.json').read_text())
    canonical = {r['id']: r for r in json.loads((OLD / 'TRAIN_WINDOWS.json').read_text())}
    assert len(rows) == 3 and len({r['source'] for r in rows}) == 3
    assert all(r == canonical[r['id']] and r['eligible'] and r['shared_cells'] == 0 for r in rows)
    assert all(r['image_id'].startswith('selvamask::train:') for r in rows)
    model, provenance, _ = build_carrier(seed=42, pretrained=False)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'], strict=True); del ck
    model.eval().requires_grad_(False)
    captured = []
    hook = model.sem_seg_head.predictor.mask_embed.register_forward_hook(
        lambda module, args, output: captured.append(output.detach()))
    all_rows = []; summaries = []
    binding = {'checkpoint': str(checkpoint), 'sha256': actual,
               'code_sha256': sha(__file__), 'protocol_sha256': sha(EXP/'protocol.md'),
               'manifest_sha256': sha(OLD/'TRAIN_WINDOWS.json'),
               'train_loader_sha256': sha(OLD/'train_data.py'),
               'carrier_sha256': sha(OLD/'carrier.py'),
               'RGB_only_native_forward': True, 'oracle_fit_uses_GT': True,
               'deployment_prediction': False, 'window_bindings': []}
    for row in rows:
        guard(); captured.clear(); torch.manual_seed(42)
        im, target, ids = load_window(row, device='cuda', augment=False)
        assert im.shape[-2:] == (1024, 1024)
        valid = target['valid'].bool()
        gt = target['masks'].bool() & valid
        with torch.no_grad():
            image_features = features(model, [im])
            field, _, multi = model.sem_seg_head.pixel_decoder.forward_features(image_features, None)
            out, _ = model.sem_seg_head.predictor(multi, field, None)
            field = field[0].detach().float(); native = captured[-1][0].float()
            logits = torch.einsum('qc,chw->qhw', native, field)
            error = float((logits - out['pred_masks'][0].float()).abs().max())
            assert torch.allclose(logits, out['pred_masks'][0].float(), atol=1e-4, rtol=1e-5)
            baseline, native_rles = evaluate(native, field, valid, gt, save_masks=True)
            best, best_query = baseline.max(0)
            native_payload={'RGB_only':True,'GT_used_for_prediction':False,
                            'query_ids':list(range(len(native))),
                            'class_probability':out['pred_logits'][0].sigmoid()[:,0].cpu().tolist(),
                            'raw_masks':native_rles}
            (RESULTS/f"{row['id']}_native_raw_masks.json").write_text(json.dumps(native_payload)+'\n')
            torch.save({'mask_features':field.cpu(),'native_mask_embeddings':native.cpu(),
                        'native_iou_matrix_GT_diagnostic':baseline,'GT_conditioned_features':False},
                       RESULTS/f"{row['id']}_frozen_fields.pt")
        del image_features, multi, out, logits, im
        refs = json.loads((OLD/'train_cache'/row['tile']/'references.json').read_text())['references']
        lookup = {r['id']: r for r in refs}
        bind = {'window_id': row['id'], 'rgb_sha256': sha(OLD/'train_cache'/row['tile']/'rgb.npy'),
                'valid_sha256': sha(OLD/'train_cache'/row['tile']/'valid.npy'),
                'reference_sha256': sha(OLD/'train_cache'/row['tile']/'references.json')}
        binding['window_bindings'].append(bind)
        fitted = {}; states = {}; losses = {}
        for start in ('native_best', 'zero'):
            fits = []; before = []; after = []
            for j in range(0, len(gt), 16):
                count = min(16, len(gt)-j); points = []; eligible = []
                for k in range(j, j+count):
                    pos = torch.nonzero(gt[k].flatten(), as_tuple=False)[:, 0].cpu().numpy()
                    neg = torch.nonzero((valid & ~gt[k]).flatten(), as_tuple=False)[:, 0].cpu().numpy()
                    assert len(pos) and len(neg), 'Empty target domain; retain attempt and stop'
                    seed = int(hashlib.sha256(f"42:{row['id']}:{ids[k]}".encode()).hexdigest()[:8], 16)
                    rng = np.random.default_rng(seed)
                    pp = rng.choice(pos, 4096, replace=len(pos)<4096)
                    nn = rng.choice(neg, 4096, replace=len(neg)<4096)
                    idx = torch.tensor(np.concatenate([pp, nn]), device='cuda')
                    yy = torch.div(idx, 1024, rounding_mode='floor'); xx = idx % 1024
                    points.append(torch.stack([2*(xx+.5)/1024-1, 2*(yy+.5)/1024-1], -1))
                grid = torch.stack(points).float()
                sampled = F.grid_sample(field[None], grid[None], mode='bilinear',
                                        padding_mode='border', align_corners=False)[0].permute(1,2,0).detach()
                with torch.no_grad():
                    test_e = native[best_query[j:j+count].to('cuda')]
                    sampled_logit = torch.einsum('bpc,bc->bp', sampled, test_e)
                    dense_test = F.interpolate(torch.einsum('bc,chw->bhw', test_e, field)[None],
                                               (1024,1024),mode='bilinear',align_corners=False)[0]
                    # Check the first target's grid against the same native-pixel materialization.
                    xy = ((grid[0]+1)*512-.5).round().long().clamp(0,1023)
                    dense_values = dense_test[0,xy[:,1],xy[:,0]]
                    assert torch.allclose(sampled_logit[0], dense_values,atol=1e-4,rtol=1e-5)
                initial = test_e.clone() if start=='native_best' else torch.zeros_like(test_e)
                e = torch.nn.Parameter(initial); opt = torch.optim.Adam([e], lr=.01)
                first = None
                for step in range(301):
                    score = torch.einsum('bpc,bc->bp', sampled, e)
                    loss_i = .5*F.softplus(-score[:,:4096]).mean(1)+.5*F.softplus(score[:,4096:]).mean(1)
                    if first is None: first = loss_i.detach().cpu()
                    assert torch.isfinite(loss_i).all()
                    if step == 300: break
                    opt.zero_grad(set_to_none=True); loss_i.sum().backward()
                    assert e.grad is not None and torch.isfinite(e.grad).all()
                    opt.step()
                    if step in (0,99,199,299): guard()
                fits.append(e.detach().clone());before.extend(first.tolist());after.extend(loss_i.detach().cpu().tolist())
                states[f'{start}_{j}'] = opt.state_dict()
                emit(window=row['id'],start=start,targets_done=j+count,targets=len(gt),
                     first_loss=float(first.mean()),final_loss=float(loss_i.mean()))
                del sampled, score, loss_i, dense_test, sampled_logit, e, opt
            fitted[start] = torch.cat(fits);losses[start] = {'first':before,'last':after}
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        metrics = {}; masks_out = {}
        for start, e in fitted.items():
            mat, masks = evaluate(e, field, valid, gt, save_masks=True)
            metrics[start] = mat.diag().tolist(); masks_out[start] = masks
        selected_start = np.argmin(np.stack([losses[s]['last'] for s in ('native_best','zero')]),axis=0)
        chosen = torch.stack([fitted[('native_best','zero')[int(s)]][i] for i,s in enumerate(selected_start)])
        chosen_mat, chosen_rles = evaluate(chosen, field, valid, gt, save_masks=True)
        per_target = chosen_mat.diag()
        for i, rid in enumerate(ids):
            ref = lookup[rid];area = float(ref['raw_area'])*.045**2
            all_rows.append({'window':row['id'],'source':row['source'],'reference_id':rid,
                'source_annotation_area_m2':area,'source_truncated':ref.get('source_truncated','UNKNOWN'),
                'crop_valid_area_m2':float(gt[i].sum())*.045**2,
                'baseline_raw_best_iou':float(best[i]),'baseline_best_query':int(best_query[i]),
                'native_start_oracle_iou':metrics['native_best'][i],'zero_start_oracle_iou':metrics['zero'][i],
                'chosen_start':('native_best','zero')[int(selected_start[i])],
                'chosen_oracle_iou':float(per_target[i]),
                'chosen_sampled_loss':losses[('native_best','zero')[int(selected_start[i])]]['last'][i],
                'oracle_GT_conditioned_not_deployable':True})
        torch.save({'embeddings':{k:v.cpu() for k,v in fitted.items()},'optimizer_states':states,
                    'ids':ids,'losses':losses,'binding':bind,'GT_conditioned_not_deployable':True},
                   RESULTS/f"{row['id']}_fit.pt")
        payload={'GT_conditioned_not_deployable':True,'valid_clipped':True,'reference_ids':ids,
                 'starts':masks_out,'chosen_by_sampled_training_loss':chosen_rles}
        (RESULTS/f"{row['id']}_oracle_masks.json").write_text(json.dumps(payload)+'\n')
        summary={'window':row['id'],'targets':len(gt),'baseline_raw50':int((best>=.5).sum()),
                 'baseline_raw75':int((best>=.75).sum()),
                 'oracle_target50':int((per_target>=.5).sum()),'oracle_target75':int((per_target>=.75).sum()),
                 'new_target50':int(((best<.5)&(per_target>=.5)).sum()),
                 'lost_target50':int(((best>=.5)&(per_target<.5)).sum()),
                 'native_dot_product_max_error':error}
        summaries.append(summary);emit(**summary)
        del field,native,gt,target,baseline,fitted,states,chosen,chosen_mat
        torch.cuda.empty_cache()
    hook.remove()
    with (RESULTS/'PER_REFERENCE.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    (RESULTS/'BINDINGS.json').write_text(json.dumps(binding,indent=2)+'\n')
    final={'status':'EXECUTED_TRAIN_ORACLE_NOT_DEPLOYABLE','windows':summaries,
           'carrier_forwards':len(rows),'carrier_parameters_updated':False,
           'SAME6_accessed':False,'TEST_accessed':False,'protected_review_accessed':False,
           'elapsed_s':time.monotonic()-START,'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
           'original_checkpoint_still_matches':sha(checkpoint)==actual}
    (RESULTS/'COMPLETE.json').write_text(json.dumps(final,indent=2)+'\n');emit(status=final['status'])

if __name__ == '__main__':
    try: main()
    except Exception:
        (RESULTS/'FAILED.txt').write_text(traceback.format_exc());raise
