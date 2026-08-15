# DeadTrees — báo cáo đúng 3 bước evaluator → suppression → classification

Ngày chốt: 2026-08-03

## Kết luận ngắn

Ba bước đã chạy xong, không thêm dataset/model/generator branch.

1. High-recall giải quyết phần lớn miss nhưng chuyển lỗi chính sang split/duplicate và background proposals.
2. Mask NMS + containment suppression loại 3.207 proposal và giảm split, nhưng chỉ cải thiện high-recall raw; chưa vượt original baseline về global segmentation F1 hoặc Boundary F1.
3. Shape+RGB tốt hơn Shape-only trên cả original baseline và suppressed masks. Tuy nhiên suppressed-mask classification chưa vượt original baseline classification do proposal universe lớn và khó hơn nhiều.

## Bước 1 — evaluator segmentation đã chốt

Cùng 100 ảnh, 1.987 GT instances sau lọc 20 px, prediction filter 50 px, Hungarian matching và Boundary F1 tolerance 3 px.

| Metric | Baseline | High-recall raw | Suppressed |
|---|---:|---:|---:|
| Predictions | 5.052 | 17.233 | 14.026 |
| Mean best IoU / GT | 0,1364 | **0,2957** | 0,2852 |
| Recall @ IoU 0,50 | 0,1374 | **0,3045** | 0,2889 |
| Object F1 @ IoU 0,50 | **0,0776** | 0,0630 | 0,0717 |
| Boundary F1 | 0,3672 | **0,4294** | 0,4161 |
| Miss rate | 0,4841 | **0,1344** | 0,1384 |
| Split rate | **0,1585** | 0,5445 | 0,4217 |
| Merge rate | 0,0111 | 0,0094 | **0,0087** |

### Lỗi chính là gì?

- Original baseline: **miss object** là lỗi lớn nhất; 48,4% GT không có overlap đáng kể.
- High-recall: miss giảm còn 13,4%, matched boundary tốt hơn, nhưng split tăng lên 54,5% và có 17.233 predictions.
- Merge rate luôn quanh 1%, nên merge nhiều GT vào một proposal không phải lỗi chính.
- Sau high-recall, lỗi chính là **split/duplicate/non-duplicate fragments**, cộng với background/live-tree proposals; boundary của các cặp match không phải bottleneck số một.

Per-site table đầy đủ nằm trong `per_site_metrics.csv`. Suppression tăng F1 so với high-recall raw trên cả 5 site:

| Site | High-recall F1@0,50 | Suppressed F1@0,50 |
|---|---:|---:|
| 3889 | 0,0751 | **0,0885** |
| 3968 | 0,0478 | **0,0540** |
| 5650 | 0,0610 | **0,0699** |
| 5653 | 0,0686 | **0,0721** |
| 5737 | 0,0477 | **0,0550** |

Improvement nhất quán theo site, nhưng global suppressed F1 vẫn thấp hơn original baseline 0,0776.

## 10 failure overlays

Đã cố định 5 ca split nặng và 5 ca miss nặng. Mỗi ảnh có 5 panel:

```text
RGB | GT | Baseline | High-recall | Suppressed
```

Split cases:

- `dataset_3889_r00003_c00048`;
- `dataset_3889_r00006_c00050`;
- `dataset_3889_r00006_c00051`;
- `dataset_3889_r00014_c00009`;
- `dataset_3968_r00042_c00048`.

Miss cases:

- `dataset_5650_r00048_c00029`;
- `dataset_5650_r00068_c00027`;
- `dataset_5650_r00072_c00017`;
- `dataset_5737_r00030_c00009`;
- `dataset_5737_r00041_c00033`.

Quan sát:

- một số GT có nhiều mask gần trùng/lồng nên NMS giảm được;
- nhiều split còn lại là fragment hoặc proposal phủ các phần khác nhau, pairwise IoU không đủ cao để suppress;
- high-recall sinh nhiều mask trên live canopy, road, shadow và thân cây dài không thuộc deadwood GT;
- các GT nhỏ/màu nhạt ở site 5650/5737 vẫn dễ bị bỏ qua.

## Bước 2 — post-processing cố định

Không grid search. Rule duy nhất:

```text
area >= 50 px
quality = predicted_iou × stability
mask IoU NMS >= 0,70
containment >= 0,90 khi min_area/max_area >= 0,50
```

Area-ratio guard tránh xóa object nhỏ chỉ vì nó nằm trong một proposal lớn.

Kết quả suppression:

- input: 17.264 masks;
- dưới 50 px: 31;
- mask-IoU NMS: 2.280;
- containment: 927;
- retained: 14.026.

So với high-recall raw:

- Object F1@0,50: 0,0630 → 0,0717;
- split rate: 0,5445 → 0,4217;
- recall@0,50: 0,3045 → 0,2889;
- Boundary F1: 0,4294 → 0,4161.

Kết luận: duplicate/nesting là một phần lỗi, nhưng NMS đơn giản không giải quyết hết fragmentation. Không nên gọi suppressed masks là segmentation thắng baseline; đây là một positive-but-insufficient post-processing result.

## Bước 3 — classification ablation

Cùng five-site outer LOSO và inner-LOSO threshold selection. Chỉ chạy Shape và Shape+RGB.

| Mask | Features | Macro F1 | Macro PR-AUC | Pooled F1 |
|---|---|---:|---:|---:|
| Original raw | Shape | 0,1909 | 0,1385 | 0,2031 |
| Original raw | Shape+RGB | **0,3153** | **0,3942** | **0,3314** |
| Suppressed | Shape | 0,1732 | 0,1201 | 0,1851 |
| Suppressed | Shape+RGB | **0,2341** | **0,3216** | **0,2874** |

### Câu hỏi 1: RGB có giúp hơn shape-only không?

**Có.**

- Original raw: +0,1244 macro F1, +0,2558 PR-AUC.
- Suppressed: +0,0609 macro F1, +0,2015 PR-AUC.

Kết quả theo site vẫn biến động: Shape+RGB không thắng Shape ở mọi fold, nhất là site 3968. Vì vậy phải báo macro mean ± std và từng site, không chỉ pooled score.

### Câu hỏi 2: mask tốt hơn có giúp classification tốt hơn không?

Trong high-recall chain, suppression cải thiện cả segmentation F1 và classification:

- high-recall Shape F1 0,1384 → suppressed Shape 0,1732;
- high-recall Shape+RGB F1 0,2137 → suppressed Shape+RGB 0,2341.

Nhưng so với original raw masks, suppressed classification vẫn thấp hơn. Hai universe cũng khác độ khó:

- original: 423 positives / 3.947 clean negatives;
- suppressed: 923 positives / 11.077 clean negatives.

Kết luận đúng mức: **suppression tốt hơn high-recall raw và giúp downstream classification trong cùng high-recall chain, nhưng chưa đủ để khẳng định suppressed masks tốt hơn original baseline trên toàn bộ pipeline.**

## Quyết định sau ba bước

- Giữ high-recall generator vì nó giảm miss rất mạnh.
- Giữ suppression module như baseline đơn giản có bằng chứng, không coi là final refinement.
- Không thêm LoRA/MoE/RemoteCLIP/dataset/change detection/caption/grid search.
- Lỗi còn lại cần giải quyết là fragment/non-duplicate split và background proposals; mask NMS mạnh hơn sẽ làm mất recall nên không tự ý hạ threshold tiếp.

## Artifacts

- Segmentation evaluator: `deadtrees_pipeline/evaluate_raw_sam2.py`;
- Suppression: `deadtrees_pipeline/suppress_masks.py`;
- Comparison/overlays: `deadtrees_pipeline/segmentation_report.py`;
- Classification: `deadtrees_pipeline/classify_objects.py`;
- Per-site metrics: `DeadTrees/experiments/segmentation_comparison_v1/per_site_metrics.csv`;
- Failure cases: `DeadTrees/experiments/segmentation_comparison_v1/failure_cases.csv`;
- 10 overlays: `DeadTrees/experiments/segmentation_comparison_v1/overlays/`;
- Classification comparison: `DeadTrees/experiments/segmentation_comparison_v1/classification_comparison.csv`;
- Per-site classification: `DeadTrees/experiments/segmentation_comparison_v1/classification_per_site.csv`;
- Suppression manifest: `DeadTrees/sam2_masks_improved_v1/manifest.json`.

Validation: 9 unit tests pass. Các warning pandas/skimage là version/deprecation warning, không làm thay đổi output.
