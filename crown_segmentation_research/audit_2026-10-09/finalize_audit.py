"""Assemble audit summary, freeze selected inputs, verify original files, finish report."""
from pathlib import Path
import collections
import hashlib
import json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
REPO = ROOT.parent
RES = ROOT / 'clean_benchmark/results'

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()

scores = json.load(open(OUT / 'recomputed_metrics.json'))
assert scores['completed'] and len(scores['rows']) == 30
assert all(not r['invalid_detections'] for r in scores['rows'])
assert all(abs(d) < 1e-9 for r in scores['rows'] for d in r['delta_vs_table_rounded'].values())
inventory = json.load(open(OUT / 'inventory.json'))
changes = []
hash_checked = 0
for entry in inventory:
    f = ROOT / entry['path']
    if not f.exists():
        changes.append({'path': entry['path'], 'change': 'missing'})
        continue
    stat = f.stat()
    if stat.st_size != entry['size'] or stat.st_mtime_ns != entry['mtime_ns']:
        changes.append({'path': entry['path'], 'change': 'size_or_mtime'})
    if entry.get('sha256'):
        hash_checked += 1
        if sha256(f) != entry['sha256']:
            changes.append({'path': entry['path'], 'change': 'sha256'})

checkpoint_paths = set()
for f in RES.glob('*_val_selection.json'):
    checkpoint_paths.add(REPO / json.load(open(f))['ckpt'])
manifest = []
for f in sorted(checkpoint_paths):
    manifest.append({'path': str(f.relative_to(REPO)), 'size': f.stat().st_size, 'sha256': sha256(f)})
for split in ['train', 'validation', 'test']:
    f = REPO / 'data/itc_benchmarks/canopyrs_quebectrees/prepared' / f'{split}_coco.json'
    manifest.append({'path': str(f.relative_to(REPO)), 'size': f.stat().st_size, 'sha256': sha256(f)})
json.dump(manifest, open(OUT / 'input_checkpoint_manifest.json', 'w'), indent=2)

table = json.load(open(RES / 'master_table_test.json'))
lines = ['| Cấu hình | AP độc lập, mean ± SD | AP từng seed 42 / 43 / 44 |', '|---|---:|---|']
rounding = []
for name, method in table['methods'].items():
    metrics = scores['methods'][name]['AP']
    values = sorted((r for r in scores['rows'] if r['method'] == name), key=lambda r: int(r['seed']))
    lines.append(f"| {name} | {metrics['mean']:.2f} ± {metrics['std']:.2f} | " + ' / '.join(f"{r['metrics']['AP']:.6f}" for r in values) + ' |')
    rounding.append({'method': name, 'raw_mean_AP': metrics['mean'],
                     'mean_of_rounded_seed_AP': sum(method['seeds'][r['seed']]['AP'] for r in values) / len(values)})
report = (OUT / 'BAO_CAO_KIEM_TOAN.md').read_text()
report = report.replace('<!-- VERIFIED_RESULTS_TABLE -->',
    '**Hoàn tất 30/30 bộ dự đoán: 270 giá trị metric khớp bảng tới hai chữ số thập phân; không có detection sai ID ảnh/lớp, sai kích thước RLE hoặc score không hữu hạn theo các kiểm tra đã thực hiện.**\n\n' + '\n'.join(lines) + '\n\n'
    'SD dùng ddof=1; mean và SD ở bảng audit tính từ AP chưa làm tròn. EXP-023 có mean 33,745083… AP: làm tròn từ số đầy đủ là **33,75**, giống bảng hiện tại; sai khác làm tròn cần đọc theo từng cấu hình, không mặc định mọi dòng sai.\n\n'
    'Kiểm tra cuối đối với ' + str(len(inventory)) + ' tệp gốc: ' + str(len(changes)) + ' thay đổi size/mtime/hash phát hiện; ' + str(hash_checked) + ' tệp văn bản có hash được đối chiếu. ')
(OUT / 'BAO_CAO_KIEM_TOAN.md').write_text(report)
summary = {'date': '2026-10-09', 'scope': 'crown_segmentation_research',
           'original_files': len(inventory), 'original_bytes': sum(x['size'] for x in inventory),
           'original_changes': changes, 'original_text_hashes_verified': hash_checked,
           'rescored_runs': len(scores['rows']), 'rescored_metrics_per_run': 9,
           'all_metric_deltas_at_2dp_zero': True, 'invalid_detection_counts': 0,
           'metrics': scores['methods'], 'rounding': rounding,
           'source_annotation_checks': json.load(open(OUT / 'source_annotation_checks.json')),
           'selected_unique_checkpoints_hashed': len(checkpoint_paths),
           'pytest': {'passed': 39, 'warnings': 1, 'seconds': 48.45},
           'limitations': ['No training or full checkpoint-to-prediction inference rerun',
                           'No independent human ground-truth adjudication',
                           'Not all auxiliary experiments or 2000 bootstrap replicates re-executed',
                           'No immutable historical provenance; current hashes cannot prove past lineage']}
json.dump(summary, open(OUT / 'audit_summary.json', 'w'), indent=2)
print(json.dumps({'runs': len(scores['rows']), 'original_changes': changes,
                  'hashes_verified': hash_checked, 'checkpoints_hashed': len(checkpoint_paths),
                  'raw_AP': {n: v['AP'] for n, v in scores['methods'].items()}}, indent=2))
