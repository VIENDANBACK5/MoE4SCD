"""Read original Parquet annotation columns; verify prepared COCO without writing source data."""
import json
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2] / 'data/itc_benchmarks/canopyrs_quebectrees'
OUT = Path(__file__).resolve().parent
report = {}
for split in ['train', 'validation', 'test']:
    prepared = json.load(open(ROOT / 'prepared' / f'{split}_coco.json'))
    filenames = {im['id']: im['file_name'] for im in prepared['images']}
    by_file = {name: [] for name in filenames.values()}
    for ann in prepared['annotations']:
        by_file[filenames[ann['image_id']]].append(ann)
    errors, visited, count = [], [], 0
    for shard in sorted(ROOT.glob(f'{split}-*.parquet')):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=4, columns=['tile_name', 'annotations']):
            for row in batch.to_pylist():
                name, source = row['tile_name'], row['annotations']
                visited.append(name)
                anns = by_file.get(name, [])
                if len(anns) != len(source['segmentation']):
                    errors.append({'tile': name, 'problem': 'count'})
                    continue
                for k, ann in enumerate(anns):
                    for field in ['segmentation', 'area', 'bbox', 'iscrowd']:
                        if ann[field] != source[field][k]:
                            errors.append({'tile': name, 'annotation': k, 'field': field})
                    count += 1
    report[split] = {'checked_tiles': len(visited), 'checked_annotations': count,
                     'missing_tiles': sorted(set(by_file) - set(visited)),
                     'duplicate_source_tiles': len(visited) - len(set(visited)), 'errors': errors}
    print(split, count, 'annotations;', len(errors), 'errors', flush=True)
json.dump(report, open(OUT / 'source_annotation_checks.json', 'w'), indent=2)
