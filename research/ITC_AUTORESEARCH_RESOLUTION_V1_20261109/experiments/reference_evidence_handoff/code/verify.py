"""Pixel/blinding verification and temporary engineering-only parser fixtures."""
from pathlib import Path
import sys, json, hashlib, tempfile, copy, zipfile, importlib.metadata
import numpy as np
from PIL import Image
from pycocotools import mask as mu
from import_response import convert, FLAGS

EXP = Path(__file__).resolve().parents[1]
PACKET = EXP / 'packet'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, x): Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False) + '\n')
def reject(fn):
    try: fn()
    except (AssertionError, ValueError): return True
    raise AssertionError('Invalid fixture accepted')

def parser_checks():
    # These labels are deliberately synthetic, in a temporary directory only.
    # They test storage semantics, not annotation accuracy or human agreement.
    with tempfile.TemporaryDirectory(prefix='itc_engineering_only_') as d:
        p = Path(d); (p/'valid').mkdir(); (p/'annotations').mkdir(); (p/'images').mkdir()
        valid = np.ones((32, 32), np.uint8)*255; valid[:3, :3] = 0
        Image.fromarray(valid).save(p/'valid/f.png')
        rgb = np.zeros((32,32,3),np.uint8); Image.fromarray(rgb).save(p/'images/f.png')
        write(p/'CASE_MANIFEST.json', dict(schema='itc_rgb_packet_v1', cases=[dict(case_id='f', width=32, height=32, validity='valid/f.png', image='images/f.png',
              RGB_pixels_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), valid_bool_sha256=hashlib.sha256((valid>0).tobytes()).hexdigest())]))
        status = dict(schema='itc_blind_status_v1', annotator_id='SYNTHETIC_ENGINEERING_FIXTURE',
                      declarations=dict(independently_completed=True, used_only_supplied_RGB_and_validity=True,
                      saw_predictions_or_published_polygons=False, prior_exposure_to_cases=False),
                      cases=[dict(case_id='f', state='COMPLETED', exhaustive_visible_crowns=True, count_lower=None, count_upper=None, notes='ENGINEERING ONLY')])
        flags = {k: False for k in FLAGS}
        def shape(g, pts, label='crown'):
            return dict(label=label, group_id=g, points=pts, flags=flags.copy(), shape_type='polygon')
        ann = dict(imageHeight=32, imageWidth=32, imagePath='../images/f.png', imageData=None, shapes=[
            shape(1, [[0,0],[20,0],[20,20],[0,20]]),
            shape(1, [[24,2],[28,2],[28,6],[24,6]]),
            shape(2, [[10,10],[30,10],[30,30],[10,30]])])
        write(p/'RESPONSE_STATUS.json', status); write(p/'annotations/f.json', ann)
        r = convert(p); ins = r['cases'][0]['instances']
        assert len(ins)==2 and ins[0]['polygon_components']==2
        a,b = [mu.decode(dict(size=x['rle']['size'], counts=x['rle']['counts'].encode())).astype(bool) for x in ins]
        assert (a & b).sum()==100 and not a[valid==0].any() and ins[0]['clipped_invalid_pixels']==9
        assert not r['cases'][0]['unmarked_pixels_are_verified_background']
        rgb[0,0,0]=255; Image.fromarray(rgb).save(p/'images/f.png'); reject(lambda:convert(p))
        rgb[0,0,0]=0; Image.fromarray(rgb).save(p/'images/f.png')
        bad = copy.deepcopy(ann); bad['shapes'][1]['flags']['identity_uncertain'] = True
        write(p/'annotations/f.json', bad); reject(lambda: convert(p))
        write(p/'annotations/f.json', ann)
        bad = copy.deepcopy(status); bad['cases'][0]['state']='NOT_STARTED'
        write(p/'RESPONSE_STATUS.json', bad); reject(lambda: convert(p))
        bad = copy.deepcopy(status); bad['declarations']['saw_predictions_or_published_polygons']=True
        write(p/'RESPONSE_STATUS.json', bad); reject(lambda: convert(p))
        status['cases'][0]['state']='INSUFFICIENT_EVIDENCE'; status['cases'][0]['exhaustive_visible_crowns']=False
        ann['shapes']=[shape(None, [[2,2],[15,2],[15,15],[2,15]], 'ambiguous_region')]
        write(p/'RESPONSE_STATUS.json', status); write(p/'annotations/f.json', ann)
        result=convert(p)['cases'][0]; assert len(result['instances'])==0 and len(result['ambiguous_regions'])==1
        status['cases'][0].update(state='NO_VISIBLE_CROWNS', exhaustive_visible_crowns=True)
        ann['shapes']=[]; write(p/'RESPONSE_STATUS.json', status); write(p/'annotations/f.json', ann)
        assert len(convert(p)['cases'][0]['instances'])==0
        ann['shapes']=[shape(1, [[1,1],[20,20],[1,20],[15,1]])]
        status['cases'][0]['state']='COMPLETED'; write(p/'RESPONSE_STATUS.json', status); write(p/'annotations/f.json', ann)
        reject(lambda: convert(p))
    return dict(temporary_synthetic_fixtures=True, overlap_preserved=True, same_instance_components_united=True,
                invalid_pixels_clipped=True, conflicting_instance_flags_rejected=True, unstarted_rejected=True,
                unblinded_rejected=True, ambiguous_only_preserved=True, explicit_no_crowns_preserved=True,
                self_crossing_rejected=True, changed_RGB_rejected=True, human_evidence=False)

def main():
    public=json.loads((PACKET/'CASE_MANIFEST.json').read_text()); private=json.loads((EXP/'private/SOURCE_BINDINGS.json').read_text())
    assert len(public['cases'])==len(private['selected'])==6
    assert [x['cohort_position'] for x in private['selected']]==[0,4,8,12,16,20]
    counts={g:sum(x['spatial_group']==g for x in private['selected']) for g in ['geo_component_00','geo_component_01']}
    assert set(counts.values())=={3}
    bindings={x['case_id']:x for x in private['selected']}
    for case in public['cases']:
        for folder,mode in [('images','RGB'),('valid','L')]:
            im=Image.open(PACKET/folder/f"{case['case_id']}.png")
            assert im.size==(2048,2048) and im.mode==mode and im.info=={}
            arr=np.array(im); arr=(arr>0) if folder=='valid' else arr
            key='valid_bool_sha256' if folder=='valid' else 'RGB_pixels_sha256'
            assert hashlib.sha256(arr.tobytes()).hexdigest()==case[key]==bindings[case['case_id']][key]
        ann=json.loads((PACKET/'annotations'/f"{case['case_id']}.json").read_text())
        assert ann['shapes']==[] and ann['imageData'] is None
    status=json.loads((PACKET/'RESPONSE_STATUS.json').read_text())
    assert not status['annotator_id'] and all(x['state']=='NOT_STARTED' for x in status['cases'])
    assert all(v is None or v=='' for v in status['declarations'].values())
    assert reject(lambda:convert(PACKET))
    forbidden=[x[k] for x in private['selected'] for k in ['image_id','source_image_id','original_member']]
    forbidden += ['R5','DINO','Mask2Former','IoU','TP50','FP50','predictions/','SOURCE_BINDINGS']
    for file in PACKET.rglob('*'):
        if file.is_file() and file.suffix in {'.json','.md'}:
            text=file.read_text()
            assert not any(s in text for s in forbidden), f'Rater-facing metadata leakage in {file.name}'
    engineering=parser_checks()
    hashes={str(p.relative_to(PACKET)):sha(p) for p in sorted(PACKET.rglob('*')) if p.is_file()}
    write(EXP/'PACKET_FILE_HASHES.json', hashes)
    archive=EXP/'RGB_BLIND_PACKET.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for rel in hashes:
            info=zipfile.ZipInfo('RGB_BLIND_PACKET/'+rel, date_time=(2026,11,10,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o100644<<16
            z.writestr(info,(PACKET/rel).read_bytes())
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==len(hashes)
        assert all(hashlib.sha256(z.read('RGB_BLIND_PACKET/'+p)).hexdigest()==h for p,h in hashes.items())
    result=dict(status='VERIFIED_PREPARED_NOT_COLLECTED', images=6, native_hw=[2048,2048],
                per_geographic_component=counts, decoded_RGB_and_validity_match_frozen_inputs=True,
                PNG_metadata_empty=True, rater_facing_source_and_model_metadata_absent=True,
                blank_templates_not_counted_as_zero=True, engineering_checks=engineering,
                actual_annotators=0, actual_responses=0, independent_reference_stability='NOT_ESTABLISHED',
                new_model_forwards=0, TEST_access=False, protected_review_access=False,
                packet_files=len(hashes), archive_bytes=archive.stat().st_size, archive_sha256=sha(archive),
                dependencies={x:importlib.metadata.version(x) for x in ['numpy','Pillow','pycocotools','shapely']})
    write(EXP/'VERIFICATION.json',result); print(json.dumps(result))

if __name__=='__main__':main()
