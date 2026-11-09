# Chu kỳ 2 — đã đánh giá thật trên BAM, chưa có lời giải mới được chứng minh

**24 tiles TRAIN, hai nhóm địa lý ở Đức, 1.106 reference occurrences, chín hệ frozen, 1.944 forwards.** Cohort được chọn bằng GeoTIFF headers và khoảng cách footprints trước predictions. Tất cả outputs được đóng băng trước khi chấm reference; không training, tuning SAME6, TEST hay protected review.

Đây là phép đo external reference fidelity. Manifest training của các carriers chỉ ghi SelvaMask, nhưng lịch sử dùng BAM trong toàn dự án và mọi pretraining exposure chưa được loại trừ. Hai nhóm địa lý trong cùng vùng Bamberg không phải nhiều khu rừng độc lập trên toàn thế giới.

## Kết quả chính

RR so với đúng matched continuation, trung bình ba paired seeds:

| Đại lượng | RR − control |
|---|---:|
| Recall50 | +0,452 pp |
| AP | −0,041 |
| AP75 | −0,035 |
| TP50 | +5,0/seed |
| FP50 | +64,7/seed |
| Recall50 khi cùng output count | +0,332 pp |

Ba seeds có net TP lần lượt **−2, +9, +8**. Tổng cộng **58 matches mới và 43 matches mất**, không có matching-membership ambiguity trong các bảng paired này. Đây là reference–seed occurrences, không phải số cây sinh học độc lập.

Chỉ **8/58 matches mới** không có qualifying mask trong final-decoder raw pool của control. Phần còn lại gồm 30 trường hợp control lọc bỏ và 20 bị NMS bỏ. Như vậy, tín hiệu mới trên cohort này không đủ để nói RR chủ yếu tạo ra những cá thể trước đó chưa có hypothesis. Cũng không được suy ra mọi lợi ích chỉ do ranking: geometry, scores và decoder context sau training đều có thể thay đổi.

**Không đánh giá được original-source-small Recall50:** COCO của BAM không cung cấp full-source crown area/identity đáng tin cậy cho từng cropped occurrence. Các thống kê diện tích theo tile được giữ làm proxy riêng; không dùng chúng để thay thế hay tuyên bố đạt/trượt gate +2 pp source-small.

## Những lỗi đã đo được

Với R5/M2F/DINO seed42 trên cùng references:

- **447/1.106** reference occurrences cùng không được final-match.
- **320/1.106** không có qualifying raw mask trong cả ba observed pools đã đăng ký. R5 pool ở đây là returned RoI masks, không phải toàn bộ RPN/mask hypothesis space.
- Không reference nào có one-view support upper bound dưới IoU75. Do đó, không thể giải thích các trường hợp thiếu mask bằng riêng sự bất khả thi của geometry do view containment. Điều này không chứng minh context, features hay RGB identity đủ thông tin.

Các con số trên là disagreement với published references. Chúng chưa phải common failure trên ba bộ nhãn độc lập ổn định, nên chưa được đưa vào R_common/R_all của nghiên cứu reference-stability.

## Quan sát về view boundaries

Hình cố định cho thấy nhiều prediction outlines bám vào đường biên view. Phân tích **post-hoc exploratory**, không thay predictions, cho biết ở DINO:

- 3.073/4.768 unmatched predictions chạm internal view edge.
- 166/495 predictions đang được match cũng chạm các edges đó.
- Proxy fragment nghiêm ngặt chỉ xác định 111 unmatched masks thuộc tình huống reference-conditioned đã định nghĩa.

Không thể đồng nhất edge contact với instance redundancy. Những số này cũng không cho biết net TP sẽ mất bao nhiêu nếu xóa masks, vì matching có thể dùng fallback predictions khác. Không có edge filter hoặc NMS mới được triển khai. CropFormer đã có prior art trực tiếp về liên kết entities qua full image/crops; chưa có căn cứ mở một kiến trúc mới từ hình minh họa này.

## Quyết định khoa học

`EXTERNAL_REFERENCE_FIDELITY_MEASURED_RR_TRANSFER_LIMITED_ORIGINAL_METHOD_ENTRY_NOT_ESTABLISHED`

Đã có bằng chứng mới về giới hạn chuyển giao của một baseline dương, không chỉ thêm một proposal. Nhưng chưa có validated robust solution hoặc nguyên nhân đủ cô lập để phát triển original method.

Theo paper nguồn, BAM labeling có dùng thêm các thời điểm ảnh và kiểm tra thực địa. Đây không phải ba người independently delineate cùng một RGB một cách prediction-blind. Nghiên cứu reference-stability đã thiết kế vì thế vẫn chưa đủ điều kiện thực nghiệm.

Giữ RR làm coverage reference trong phạm vi đã chứng minh; giữ R5/M2F/DINO và toàn bộ kết quả mới. Không chọn lại seed, threshold hoặc module để làm đẹp cohort này. Goal tổng thể còn active; không có GPU training/inference job đang chạy sau khi chu kỳ này hoàn tất.

[Báo cáo đầy đủ và bảng chín hệ](../experiments/frozen_bam_transfer/RESEARCH_REPORT.md) · [Hình so sánh cố định](../experiments/frozen_bam_transfer/figures/FIXED_CROSS_CARRIER_COMPARISON.png) · [Bảng recovery/loss](../experiments/frozen_bam_transfer/results/RECOVERY_AND_LOSS.csv) · [Commands và provenance](../experiments/frozen_bam_transfer/REPRODUCIBILITY.md).
