"""Validate completed LabelMe responses; preserve independent masks/uncertainty.

No default blank file is treated as a human annotation. No source/GT join here.
"""
from pathlib import Path
import json, argparse, gzip, hashlib
import numpy as np
from PIL import Image
from pycocotools import mask as mu

FLAGS = ['presence_uncertain', 'identity_uncertain', 'boundary_uncertain', 'truncated_by_image_edge']
STATES = {'NOT_STARTED', 'COMPLETED', 'INSUFFICIENT_EVIDENCE', 'NO_VISIBLE_CROWNS'}

def read(p): return json.loads(Path(p).read_text())
def pack(mask):
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return dict(size=r['size'], counts=r['counts'].decode('ascii'))

def convert(packet):
    packet = Path(packet)
    manifest = read(packet / 'CASE_MANIFEST.json'); response = read(packet / 'RESPONSE_STATUS.json')
    assert manifest['schema'] == 'itc_rgb_packet_v1' and response['schema'] == 'itc_blind_status_v1'
    assert isinstance(response['annotator_id'], str) and response['annotator_id'].strip(), 'No actual annotator response'
    dec = response['declarations']
    assert dec['independently_completed'] is True and dec['used_only_supplied_RGB_and_validity'] is True
    assert dec['saw_predictions_or_published_polygons'] is False, 'Blinding conditions not met'
    assert type(dec['prior_exposure_to_cases']) is bool, 'Prior exposure must be explicitly declared'
    cases = {c['case_id']: c for c in manifest['cases']}
    records = {c['case_id']: c for c in response['cases']}
    assert len(cases) == len(manifest['cases']) == len(records) == len(response['cases']) and cases.keys() == records.keys()
    out = []
    for case_id, case in cases.items():
        status = records[case_id]; state = status['state']; assert state in STATES
        assert state != 'NOT_STARTED', 'Incomplete packet; blank templates are missing data'
        assert type(status['exhaustive_visible_crowns']) is bool
        if state in {'COMPLETED', 'NO_VISIBLE_CROWNS'}: assert status['exhaustive_visible_crowns'] is True
        lo, hi = status['count_lower'], status['count_upper']
        assert all(x is None or (type(x) is int and x >= 0) for x in [lo, hi])
        assert lo is None or hi is None or lo <= hi
        ann = read(packet / 'annotations' / f'{case_id}.json')
        H, W = case['height'], case['width']
        assert (ann['imageHeight'], ann['imageWidth']) == (H, W) and ann['imageData'] is None
        assert ann['imagePath'] == f'../images/{case_id}.png'
        rgb = np.array(Image.open(packet / case['image']))
        valid = np.array(Image.open(packet / case['validity'])) > 0; assert valid.shape == (H, W)
        assert rgb.shape == (H, W, 3) and rgb.dtype == np.uint8
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == case['RGB_pixels_sha256'], 'RGB changed in returned packet'
        assert hashlib.sha256(valid.tobytes()).hexdigest() == case['valid_bool_sha256'], 'Validity changed in returned packet'
        groups, ambiguous = {}, []
        for shape in ann['shapes']:
            assert shape.get('shape_type', 'polygon') == 'polygon'
            label = shape['label']; assert label in {'crown', 'ambiguous_region'}
            points = np.asarray(shape['points'], dtype=float)
            assert points.ndim == 2 and points.shape[1] == 2 and len(points) >= 3 and np.isfinite(points).all()
            assert (points >= 0).all() and (points[:, 0] <= W).all() and (points[:, 1] <= H).all()
            assert len(np.unique(points, axis=0)) >= 3
            twice_area = abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1)))
            assert twice_area > 0, 'Degenerate polygon'
            # Reject self crossings rather than silently repair annotator geometry.
            from shapely.geometry import Polygon
            assert Polygon(points).is_valid, 'Self-intersecting polygon needs explicit correction'
            mask = mu.decode(mu.frPyObjects([points.flatten().tolist()], H, W))[:, :, 0].astype(bool)
            clipped = mask & valid
            assert clipped.any(), 'Polygon has no valid pixel support'
            if label == 'ambiguous_region':
                ambiguous.append(dict(rle=pack(clipped), clipped_invalid_pixels=int((mask & ~valid).sum())))
                continue
            gid = shape.get('group_id'); assert type(gid) is int and gid >= 0, 'Each crown needs an instance group_id'
            flags = {k: shape.get('flags', {}).get(k, False) for k in FLAGS}
            assert all(type(x) is bool for x in flags.values())
            if gid not in groups: groups[gid] = dict(mask=np.zeros((H, W), bool), flags=flags, clipped=0, components=0)
            group = groups[gid]; assert group['flags'] == flags, 'Conflicting annotations within one identity'
            group['mask'] |= clipped; group['clipped'] += int((mask & ~valid).sum()); group['components'] += 1
        if state == 'NO_VISIBLE_CROWNS':
            assert not groups and not ambiguous
            assert lo in {None, 0} and hi in {None, 0}, 'Count contradicts NO_VISIBLE_CROWNS'
        if state == 'COMPLETED': assert groups or ambiguous, 'Explicit NO_VISIBLE_CROWNS required for empty completed image'
        instances = [dict(instance_id=str(gid), rle=pack(g['mask']), area=int(g['mask'].sum()),
                          flags=g['flags'], polygon_components=g['components'], clipped_invalid_pixels=g['clipped'])
                     for gid, g in sorted(groups.items())]
        out.append(dict(case_id=case_id, state=state, RGB_pixels_sha256=case['RGB_pixels_sha256'],
                        valid_bool_sha256=case['valid_bool_sha256'], exhaustive_visible_crowns=status['exhaustive_visible_crowns'],
                        count_lower=lo, count_upper=hi, notes=status['notes'], instances=instances, ambiguous_regions=ambiguous,
                        unmarked_pixels_are_verified_background=False))
    return dict(schema='itc_independent_response_masks_v1', annotator_id=response['annotator_id'],
                declarations=dec, cases=out, independent_masks=True,
                biological_identity_verified=False, independent_geography_established=False,
                qualification_for_stability_analysis='PENDING_HUMAN_PROVENANCE_AND_PROTOCOL_REVIEW')

def main():
    p = argparse.ArgumentParser(); p.add_argument('--packet', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); result = convert(a.packet)
    assert not a.output.exists(), 'Do not overwrite a response'
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.output, 'wt') as f: json.dump(result, f, allow_nan=False)
    print(json.dumps(dict(status='IMPORTED_NOT_YET_ADJUDICATED', cases=len(result['cases']), annotator_id=result['annotator_id'])))

if __name__ == '__main__': main()
