"""Export only RGB/validity from the already fixed TRAIN inference cohort."""
from pathlib import Path
import sys, json, gzip, hashlib, shutil
import numpy as np
from PIL import Image

EXP = Path(__file__).resolve().parents[1]
PREV = EXP.parent / 'frozen_bam_transfer'
DATA = Path('/home/chung/RS/data/TreeCrown-MultiSource-V1')
sys.path.insert(0, str(DATA))
from treecrown_data import TreeCrownData

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def main():
    assert shutil.disk_usage(EXP).free > 20 * 2**30
    packet = EXP / 'packet'
    # Never overwrite an annotation someone may have started.
    assert not (packet / 'CASE_MANIFEST.json').exists(), 'Existing packet must remain immutable'
    for name in ['images', 'valid', 'annotations']:
        (packet / name).mkdir(parents=True, exist_ok=True)
    cohort = json.loads((PREV / 'metadata/SELECTED_COHORT.json').read_text())
    data = TreeCrownData.from_root(DATA)
    view = data.open(source='bamforests', split='train', track='legacy_reference', access_role='inference')
    indices = {r['image_id']: i for i, r in enumerate(view.rows)}
    public, private, statuses = [], [], []
    for position in [0, 4, 8, 12, 16, 20]:
        row = cohort['selected'][position]
        case_id = 'rgb_' + hashlib.sha256(('reference-evidence-handoff-v1:' + row['image_id']).encode()).hexdigest()[:10]
        sample = view[indices[row['image_id']]]
        assert sample['annotation_records'] is None and sample['instance_masks'] is None and not data._coco
        assert sample['provenance']['label_access'] == 'WITHHELD_FOR_INFERENCE'
        rgb, valid = sample['rgb'], sample['valid_mask']
        assert rgb.shape == (2048, 2048, 3) and rgb.dtype == np.uint8 and valid.shape == (2048, 2048)
        with gzip.open(PREV / f'predictions/dino_seed42/b{position:03d}.json.gz', 'rt') as f:
            # Read only prior input bindings, not predictions or reference metrics.
            frozen = json.load(f)
        rh = hashlib.sha256(rgb.tobytes()).hexdigest()
        vh = hashlib.sha256(valid.tobytes()).hexdigest()
        assert frozen['source_image_id'] == row['image_id']
        assert (rh, vh) == (frozen['RGB_sha256'], frozen['valid_sha256'])
        Image.fromarray(rgb).save(packet / 'images' / f'{case_id}.png')
        Image.fromarray(valid.astype(np.uint8) * 255).save(packet / 'valid' / f'{case_id}.png')
        write(packet / 'annotations' / f'{case_id}.json', dict(version='5.5.0', flags={}, shapes=[],
              imagePath=f'../images/{case_id}.png', imageData=None, imageHeight=2048, imageWidth=2048))
        public.append(dict(case_id=case_id, image=f'images/{case_id}.png', validity=f'valid/{case_id}.png',
                           width=2048, height=2048, gsd_x_cm=100*row['gx_m'], gsd_y_cm=100*row['gy_m'],
                           RGB_pixels_sha256=rh, valid_bool_sha256=vh))
        private.append(dict(case_id=case_id, cohort_position=position, image_id=row['image_id'],
                            source_image_id=row['source_image_id'], spatial_group=row['spatial_group'],
                            original_member=row['member'], RGB_pixels_sha256=rh, valid_bool_sha256=vh))
        statuses.append(dict(case_id=case_id, state='NOT_STARTED', exhaustive_visible_crowns=None,
                             count_lower=None, count_upper=None, notes=''))
    public.sort(key=lambda x: x['case_id']); statuses.sort(key=lambda x: x['case_id'])
    write(packet / 'CASE_MANIFEST.json', dict(schema='itc_rgb_packet_v1', coordinate_system='image pixels: x right, y down', cases=public))
    write(packet / 'RESPONSE_STATUS.json', dict(schema='itc_blind_status_v1', annotator_id='',
          declarations=dict(independently_completed=None, used_only_supplied_RGB_and_validity=None,
                            saw_predictions_or_published_polygons=None, prior_exposure_to_cases=None,
                            prior_exposure_details=''), cases=statuses))
    shutil.copyfile(DATA / 'provenance/license_documents/bam_readme.txt', packet / 'SOURCE_LICENSE.txt')
    write(EXP / 'private/SOURCE_BINDINGS.json', dict(status='PREPARED_NOT_COLLECTED', cohort_sha256=sha(PREV / 'metadata/SELECTED_COHORT.json'),
          protocol_sha256=sha(EXP / 'protocol.md'), annotations_read=False, new_model_forwards=0, selected=private))
    print(json.dumps(dict(status='PREPARED_NOT_COLLECTED', cases=len(public), RGB_pixels_unchanged=True,
                         annotations_read=False, new_model_forwards=0)))

if __name__ == '__main__':
    main()
