# DeadTrees — quyết định triển khai sau raw SAM2 baseline

Ngày chốt: 2026-08-02

## 1. Phạm vi nghiên cứu được giữ lại

- **RQ1 — segmentation:** refinement/prompting dựa trên hình học và RGB có làm mask SAM2 tốt hơn không?
- **RQ2 — representation:** trên cùng tập mask, RGB có giúp phân loại deadwood tốt hơn shape-only không?
- **RQ3 — downstream:** mask tốt hơn có làm kết quả phân loại/end-to-end tốt hơn không?

DeadTrees là bộ kiểm chứng có polygon thủ công. SECOND/SCD chỉ được quay lại sau khi ba câu hỏi trên có kết quả đáng tin.

## 2. Những phần tạm đóng băng

Không xóa code hoặc dữ liệu, nhưng không tiếp tục sửa/chạy các nhánh sau trong giai đoạn hiện tại:

- `SECOND-OC/`, ChangeStar/SECOND, reasoner, Token-MoE và các script train SCD;
- các checkpoint/output của SECOND;
- `spectral_mask_refine.py`: threshold chưa được chọn trong fold; K-means theo RGB có thể tạo các mảnh không liên thông; output score bị thay bằng 1;
- `spectral_edge_prompt.py`: mới là heuristic, chưa có baseline/fold protocol;
- các evaluator cũ dựa trên union mask/connected components;
- `classify_deadwood.py` ở dạng hiện tại: random split từng proposal gây leakage giữa ảnh/site, GT bị union, label chỉ dựa trên phần prediction được phủ, thiếu PR-AUC và GT-mask upper bound;
- mở rộng kiến trúc Transformer/MoE hoặc tải thêm dataset.

Các file trên chỉ là tài liệu tham khảo; không dùng số liệu của chúng trong báo cáo chính.

## 3. Những phần được giữ và tái sử dụng

- 100 GeoTIFF DeadTrees hiện có;
- GeoPackage polygon gốc và `source_fid`;
- 100 file proposal SAM2 raw hiện có;
- SAM2 checkpoint/repository để chạy lại generator khi cần;
- ý tưởng feature hình học và RGB trong code cũ, nhưng phải trích xuất lại theo protocol mới.

Không cần tải dataset mới ở Deliverable 1–2. Không cần cài thêm package để chạy evaluator hiện tại. Hai warning phiên bản `numexpr`/`bottleneck` không ảnh hưởng kết quả và chưa cần sửa môi trường.

## 4. Deliverable 1 — đã hoàn thành

### Ground truth

- 100 ảnh, 2.060 polygon instance trước lọc diện tích;
- 92 ảnh có GT và 8 ảnh âm hoàn toàn;
- giữ geometry Polygon/MultiPolygon, hole, `source_fid`, `instance_id`, site và tile;
- evaluator rasterize riêng từng polygon trong CRS/transform gốc của ảnh;
- ngưỡng báo cáo hiện tại loại 73 GT nhỏ hơn 20 px, còn 1.987 GT.

### Raw SAM2 baseline

Protocol: prediction tối thiểu 50 px, Hungarian one-to-one tại IoU 0.25/0.50/0.75, split/merge tính trên overlap matrix trước Hungarian, Boundary F1 tolerance 3 px.

| Chỉ số | Kết quả |
|---|---:|
| Object F1 @ IoU 0.50 | 0,0776 |
| Precision @ IoU 0.50 | 0,0540 |
| Recall @ IoU 0.50 | 0,1374 |
| Mean best IoU / GT | 0,1364 |
| Mean IoU của cặp đã match @0.50 | 0,6769 |
| Boundary F1 của cặp đã match | 0,3672 |
| ASSD | 6,05 px |
| HD95 | 15,46 px |
| Split rate | 15,85% |
| Merge rate / prediction | 1,11% |
| GT không có overlap đáng kể | 48,41% |
| Prediction không liên quan GT | 84,70% |
| AP50 dùng SAM stability score | 0,0080 |

AP thấp cho thấy stability score không phải confidence deadwood. Nó không thể thay classifier.

### Failure taxonomy

| Nhóm GT | Số lượng | Tỷ lệ |
|---|---:|---:|
| Có proposal IoU >= 0,50 | 273 | 13,74% |
| Partial IoU 0,25–0,50 | 150 | 7,55% |
| Chỉ overlap yếu | 602 | 30,30% |
| Miss gần như hoàn toàn | 962 | 48,41% |

Theo kích thước, quartile nhỏ nhất (<= 374,5 px) chỉ có recall@0.50 = **0,80%**. Hai quartile lớn hơn đạt 26,21% và 17,91%. Vấn đề đầu tiên là proposal recall cho vật thể nhỏ, không phải chỉ là làm mượt boundary.

Theo site, recall@0.50 dao động từ 7,28% (`5650`) đến 18,04% (`3889`), nên mọi kết quả tiếp theo phải báo riêng từng site và pooled out-of-fold.

## 5. Quyết định kỹ thuật từ baseline

Hai bottleneck độc lập phải được xử lý riêng:

1. **Miss/GT nhỏ:** hậu xử lý mask hiện có không thể tạo lại 962 GT đã mất. Cần thử generator/prompting có recall cao hơn trước.
2. **False proposal/live vegetation:** hình ảnh kiểm tra cho thấy SAM2 segment nhiều cây sống và vùng nền. Đây là nhiệm vụ của classifier/filter, không nên cố giải quyết toàn bộ bằng boundary refinement.

Vì vậy chưa chạy `spectral_mask_refine.py`. Thứ tự tiếp theo là classification ablation có kiểm soát, sau đó mới chạy một generator/refinement ablation dựa trên lỗi.

## 6. Deliverable 2 — phần cần code tiếp theo

Tạo một object table duy nhất từ raw proposal:

- `stem`, `dataset_id`, proposal ID, score, area;
- best IoU, Hungarian match và quan hệ split/merge với GT;
- shape features;
- RGB features;
- nhãn train rõ ràng.

Quy tắc nhãn đề xuất:

- positive: proposal được Hungarian match với GT tại IoU >= 0,25;
- clean negative: proposal không overlap đáng kể với bất kỳ GT nào;
- ambiguous: proposal overlap nhưng không đạt IoU 0,25; không dùng để fit classifier nhưng vẫn giữ trong đánh giá end-to-end.

Không random split theo mask. Dùng 5 leave-one-site-out folds (`dataset_id`), báo:

- F1 và PR-AUC từng site;
- macro mean ± std;
- pooled out-of-fold prediction;
- số positive/negative/ambiguous trong từng fold;
- cùng negative universe cho Shape, RGB và Shape+RGB;
- Random Forest seed cố định và class weight; mọi sampling/tuning chỉ dùng train/validation sites.

Ablation bắt buộc:

1. raw SAM2 + Shape;
2. raw SAM2 + RGB;
3. raw SAM2 + Shape+RGB;
4. GT mask + Shape+RGB upper bound.

Tiêu chí hoàn thành: có một bảng OOF duy nhất, per-site table và file prediction để audit; không khẳng định improvement chỉ từ một random split.

## 7. Deliverable 3 — thí nghiệm segmentation có bằng chứng

Do lỗi nhỏ/miss chiếm ưu thế, candidate đầu tiên nên là **high-recall SAM2 generation**, không phải RGB K-means split/merge:

- tăng `points_per_side` từ 32 lên 64;
- giảm `min_mask_region_area` từ 100 xuống khoảng 20;
- thử mức `pred_iou_thresh`/`stability_score_thresh` thấp hơn;
- giữ nguyên raw RGB normalization để so sánh công bằng.

Không quét threshold rồi báo kết quả trên cùng 100 ảnh. Hoặc dùng các cấu hình định trước, hoặc chọn cấu hình trên train/validation sites trong mỗi outer fold.

Chỉ khi overlap matrix cho thấy split/merge/boundary là bottleneck còn lại mới thêm một refinement. Mọi raw/refined comparison phải giữ cùng 100 ảnh và báo cả số proposal, matched GT, full-set recall và end-to-end FN.

## 8. Lệnh tái lập Deliverable 1

```bash
python -m deadtrees_pipeline.gt_instances --overwrite
pytest -q tests/test_deadtrees_metrics.py
python -m deadtrees_pipeline.evaluate_raw_sam2 --visualizations 10
```

Output chuẩn:

- `DeadTrees/instances_gt/instances.gpkg`;
- `DeadTrees/instances_gt/instances.summary.json`;
- `DeadTrees/experiments/raw_sam2_v1/summary.json`;
- `DeadTrees/experiments/raw_sam2_v1/per_image.csv`;
- `DeadTrees/experiments/raw_sam2_v1/per_gt.csv`;
- `DeadTrees/experiments/raw_sam2_v1/overlays/`.

## 9. Việc làm ngay

Viết Deliverable 2 trên evaluator mới. Không tải thêm dataset, không train lại SECOND, không sửa MoE, và chưa tune refinement trên toàn bộ 100 ảnh.
