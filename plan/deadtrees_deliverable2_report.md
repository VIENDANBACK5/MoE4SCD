# DeadTrees Deliverable 2 — Controlled object classification ablation

Ngày chạy: 2026-08-02

## 1. Mục tiêu

Kiểm tra hai câu hỏi trên cùng dữ liệu và không trộn site:

- RGB/radiometric statistics có bổ sung thông tin cho shape-only hay không?
- Nếu proposal classifier tốt hơn, kết quả detection end-to-end có tự động tốt hơn hay không?

Kết quả bên dưới không dùng random train/test split. Mọi prediction được báo đều là out-of-fold theo site.

## 2. Protocol

### Object universe và nhãn

Raw SAM2 sau lọc diện tích 50 px có 5.052 proposal:

| Nhãn | Định nghĩa | Số lượng |
|---|---|---:|
| Positive | Hungarian match với GT tại IoU >= 0,25 | 423 |
| Clean negative | `intersection/GT < 0,10` và `intersection/pred < 0,10` với mọi GT | 3.947 |
| Ambiguous | Các proposal còn lại | 682 |

Ambiguous không được dùng để fit hoặc tính classifier F1/PR-AUC, nhưng vẫn được dự đoán và giữ trong đánh giá end-to-end.

GT upper bound sử dụng 1.987 GT polygon positives và cùng 3.947 clean negatives. Đây là diagnostic upper bound, không phải phép so sánh trên cùng positive universe với raw SAM2.

Phân bố raw labels theo site:

| Site | Positive | Clean negative | Ambiguous |
|---|---:|---:|---:|
| 3889 | 223 | 1.537 | 434 |
| 3968 | 40 | 841 | 40 |
| 5650 | 34 | 450 | 43 |
| 5653 | 76 | 522 | 102 |
| 5737 | 50 | 597 | 63 |

### Features

- Shape: 12 features về area, perimeter, aspect ratio, extent, compactness, solidity, eccentricity, axes, connected components, holes và tile border.
- RGB: mean, standard deviation và percentiles 10/50/90 cho ba kênh, tổng 15 features.
- Shape+RGB: 27 features.

GeoTIFF có hai kiểu container (`uint8` và `float32`) nhưng cùng thang RGB 0–255. Pipeline kiểm tra finite/range và chuẩn hóa về [0,1], không stretch riêng từng ảnh.

### Cross-validation

- Outer split: leave-one-`dataset_id`-out, 5 geographical folds.
- Classifier: Random Forest, 300 trees, `balanced_subsample`, `min_samples_leaf=2`, seed 42.
- Threshold: với từng outer fold, chạy inner leave-one-training-site-out trên bốn training sites; chọn threshold tối đa inner-OOF F1 trên grid 0,05–0,95.
- Outer test site không được dùng để chọn model hoặc threshold.
- Báo macro mean ± sample standard deviation qua 5 sites và pooled OOF.

## 3. Kết quả classification

### Macro qua 5 held-out sites

| Segmentation / features | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|
| Raw SAM2 / Shape | 0,1909 ± 0,0473 | 0,1385 ± 0,0762 | 0,6211 ± 0,0596 |
| Raw SAM2 / RGB | 0,1797 ± 0,1475 | 0,2533 ± 0,1256 | 0,7913 ± 0,1239 |
| Raw SAM2 / Shape+RGB | **0,3153 ± 0,1827** | **0,3942 ± 0,1902** | **0,8136 ± 0,1282** |
| GT masks / Shape+RGB | **0,6782 ± 0,1063** | **0,7906 ± 0,1201** | **0,8839 ± 0,0753** |

So với Shape-only, Shape+RGB tăng:

- macro F1: **+0,1244**;
- macro PR-AUC: **+0,2558**;
- macro ROC-AUC: **+0,1925**.

RGB-only có PR-AUC cao hơn Shape-only nhưng F1 thấp hơn nhẹ do operating threshold chuyển giao kém giữa site. Kết luận hợp lý là radiometric information cải thiện khả năng ranking, còn shape và RGB bổ sung cho nhau để tạo F1 tốt nhất.

### Pooled OOF

| Cấu hình | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| Raw / Shape | 0,1246 | 0,5485 | 0,2031 | 0,1281 |
| Raw / RGB | 0,1613 | 0,3452 | 0,2199 | 0,1993 |
| Raw / Shape+RGB | **0,3272** | 0,3357 | **0,3314** | **0,3131** |
| GT upper bound / Shape+RGB | 0,6404 | **0,6990** | **0,6684** | **0,7333** |

### Shape+RGB theo site

| Held-out site | Threshold học từ 4 training sites | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|
| 3889 | 0,42 | 0,2047 | 0,3767 | 0,7317 |
| 3968 | 0,32 | 0,0896 | 0,0801 | 0,6282 |
| 5650 | 0,28 | 0,3830 | 0,4698 | 0,8786 |
| 5653 | 0,22 | 0,5714 | 0,5822 | 0,9143 |
| 5737 | 0,25 | 0,3276 | 0,4624 | 0,9150 |

Site 3968 là failure domain rõ ràng. Nó có ít positive, nhiều ảnh không có GT, và ranking Shape+RGB gần như không tốt hơn random nhiều. Không được chỉ báo pooled score rồi che biến động này.

### Feature importance

Top Random Forest importances của Shape+RGB chủ yếu là:

1. percentile 90 kênh B;
2. percentile 90 kênh R;
3. median B;
4. mean B;
5. standard deviation B.

Đây chỉ là model importance, không phải causal importance. Tuy vậy nó củng cố kết luận rằng phân bố màu bên trong object chứa tín hiệu mà shape-only không có.

## 4. GT-mask upper bound chẩn đoán bottleneck

Macro F1 tăng từ 0,3153 trên raw proposals lên 0,6782 với GT positives. Macro PR-AUC tăng từ 0,3942 lên 0,7906.

Khoảng cách lớn này cho thấy feature/classifier không phải bottleneck duy nhất. Chất lượng và độ phủ của segmentation đang giới hạn mạnh classification:

- raw positives chỉ có 423 proposal match tại IoU 0,25;
- trong khi evaluator có 1.987 GT instances;
- gần một nửa GT không có proposal overlap đáng kể;
- object nhỏ nhất gần như bị SAM2 bỏ qua.

## 5. End-to-end trên cùng 100 ảnh

Classifier OOF được dùng để lọc toàn bộ raw proposal, sau đó Hungarian detection được tính lại. Không chỉ đánh giá các object sống sót sau filter.

### IoU 0,50

| Cấu hình | Số prediction | Precision | Recall | Object F1 |
|---|---:|---:|---:|---:|
| Raw unfiltered | 5.052 | 0,0540 | **0,1374** | **0,0776** |
| Shape filter | 2.148 | 0,0694 | 0,0750 | 0,0721 |
| RGB filter | 1.043 | 0,0863 | 0,0453 | 0,0594 |
| Shape+RGB filter | 517 | **0,1741** | 0,0453 | 0,0719 |

Shape+RGB loại 89,8% proposal và tăng precision hơn ba lần, nhưng cũng loại quá nhiều true proposal. Object F1@0.50 giảm 0,0057 so với raw.

Tại IoU 0,25, Shape+RGB F1 là 0,1150 so với raw 0,1202. Tại IoU 0,75 nó tăng rất nhẹ 0,0008, không đủ để xem là improvement thực tế.

Kết luận trung thực:

> Radiometric features improve proposal-level deadwood classification, especially when combined with shape. However, filtering current raw SAM2 proposals does not improve end-to-end object F1 because proposal recall is already too low and the classifier further removes true objects.

## 6. Trả lời research questions hiện tại

- **RQ2:** Có bằng chứng ủng hộ. Shape+RGB tốt hơn Shape-only về macro F1, PR-AUC và ROC-AUC trên leave-one-site-out.
- **RQ3:** Có bằng chứng rằng segmentation ảnh hưởng mạnh tới classification vì GT upper bound cao hơn raw rất nhiều.
- **End-to-end:** Chưa thắng baseline. Classifier cải thiện precision nhưng mất recall, nên không được kết luận hệ thống tổng thể đã tốt hơn.
- **RQ1:** Chưa được kiểm tra bằng một refinement/generator mới; đây là bước tiếp theo.

## 7. Quyết định bước tiếp theo

Không tiếp tục tối ưu classifier trên raw proposals hiện tại. Bước có giá trị nhất là tăng proposal recall của SAM2, đặc biệt cho object <= 375 px:

1. tạo cấu hình high-recall định trước: `points_per_side=64`, `min_mask_region_area=20`, hạ vừa phải SAM predicted-IoU/stability thresholds;
2. chạy ra thư mục mới, không ghi đè `DeadTrees/sam2_masks`;
3. đánh giá bằng cùng evaluator và cùng 100 ảnh;
4. chỉ nếu proposal recall@0.25/@0.50 tăng thì mới chạy lại object ablation;
5. mọi lựa chọn threshold sau đó phải dùng train/validation sites, không nhìn outer test site.

Việc sinh lại SAM2 Hiera-L nên chạy trên GPU. Kiểm tra ban đầu trong Codex sandbox không thấy `/dev/nvidia*`, nhưng kiểm tra ngoài sandbox xác nhận máy có RTX 5880 Ada 49 GB. Generation được chạy bằng `cs2_venv` do môi trường Anaconda mặc định có binary mismatch giữa PyTorch và torchvision.

Generator đã được chuẩn bị với cấu hình định trước và thư mục output tách biệt:

```bash
python -m deadtrees_pipeline.generate_sam2_variant --device cuda
python -m deadtrees_pipeline.evaluate_raw_sam2 \
  --sam-dir DeadTrees/sam2_masks_high_recall_v1 \
  --output DeadTrees/experiments/high_recall_v1
```

Mặc định high-recall v1 dùng grid 64, predicted-IoU threshold 0,75, stability threshold 0,85 và minimum region 20 px. Script từ chối ghi vào thư mục raw baseline và từ chối chạy CPU trừ khi người dùng chủ động truyền `--allow-cpu`. Lượt chạy hoàn chỉnh và kết quả tiếp theo được ghi trong `deadtrees_deliverable3_high_recall_report.md`.

## 8. Artifacts tái lập

- `deadtrees_pipeline/classify_objects.py`;
- `deadtrees_pipeline/evaluate_classifier_filter.py`;
- `deadtrees_pipeline/generate_sam2_variant.py`;
- `DeadTrees/experiments/classification_v1/object_features.csv`;
- `DeadTrees/experiments/classification_v1/fold_metrics.csv`;
- `DeadTrees/experiments/classification_v1/oof_predictions.csv`;
- `DeadTrees/experiments/classification_v1/summary.json`;
- `DeadTrees/experiments/classification_v1/end_to_end_metrics.json`.

Kiểm tra integrity: 7.039 object rows, không trùng `sample_id`, không có feature NaN; mỗi experiment có đúng một OOF prediction cho mỗi sample trong universe; 6 unit tests pass.
