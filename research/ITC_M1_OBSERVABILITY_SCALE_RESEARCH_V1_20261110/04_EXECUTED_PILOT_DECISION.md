# M1 pilot — đã thực nghiệm, chưa xác lập method mới

**LOSSLESS_ENCODING_SENSITIVITY_OBSERVED_CAUSE_AND_ORIGINAL_METHOD_NOT_ESTABLISHED**

Đã thực hiện12forwards với frozen Mask DINO mask-only seed42,300queries, ba TRAIN windows đã exposed,298reference occurrences. Bốn arms A/B/C/D giữ cùng physical footprint. Đã kiểm tra chính xác phép nghịch trái trên RGB và tensor chuẩn hóa thực sự đi vào backbone. Không train/fine-tune carrier, không SAME6/TEST/protected review.

| Arm | Input | Raw IoU50 | Final TP50 | Small TP50/133 | FP50 |
|---|---|---:|---:|---:|---:|
| A | Native1024² | 278 | 276 | 116 | 158 |
| B | Lặp pixel2048², cùng thông tin A | 113 | 98 | 36 | 321 |
| C | Giảm mẫu512² | 174 | 163 | 46 | 145 |
| D | Lặp C về1024², cùng thông tin C | 213 | 206 | 78 | 206 |

A→B có2final matches mới/180mất;167raw qualifications mất/2mới. Trong180final losses,165đã thiếu qualifying raw mask của B. C→D có70final matches mới/27mất, tăng ròng43; small có41mới/9mất. Vì vậy resize có thể giúp hoặc làm hại, và vẫn gây collateral losses.

Ở equal-output-count: A/B cùng411outputs đạt276/98TP50; C/D cùng308outputs đạt163/194TP50. Native A cũng tái lập đúng900query masks và class probabilities của probe cũ sau cùng common-valid clipping. Các thay đổi không biến mất khi chỉ kiểm soát số predictions.

Điều đã chứng minh trong thiết lập này: mất thông tin RGB đầu vào không giải thích được khác biệt trong hai cặp lossless. Điều chưa chứng minh: một lỗi kiến trúc, representation hay query selection cụ thể; causal explanation của BAM; biological identity; hoặc một phương pháp mới có hiệu quả.

Điểm giới hạn đặc biệt quan trọng: carrier fine-tune với1024native windows. B/C thay kích thước operating input; tạo ra distribution shift nhân tạo. Thực nghiệm này không bác bỏ khả năng conventional scale augmentation hoặc input canonicalization giải quyết hiện tượng. Với đúng phép lặp pixel B, một phép nghịch trái đưa input trở về A vốn đã chạy được; đây không phải cơ chế mới. Không dùng kết quả lớn để tự động hợp thức hóa observability loss.

Đã kiểm tra lại3600raw/1573final masks, NMS, validity, IoU, matching và checkpoint. Có lỗi JSON console sau khi bảng phân tích ghi xong; đã giữ traceback và sửa serializer, không chạy lại model hoặc đổi thí nghiệm. Full dense mask logits không lưu theo storage adaptation đã khóa; RLEs/scores/site traces có lưu.

[Báo cáo đầy đủ](experiments/frozen_encoding_v1/REPORT.md), [fixed-rule figure](experiments/frozen_encoding_v1/figures/FIXED_ENCODING_COMPARISON.png), [verification](experiments/frozen_encoding_v1/VERIFICATION.json), [protocol](experiments/frozen_encoding_v1/protocol.md).

Quyết định: đã có một phenomenon đủ lớn cho nghiên cứu cơ chế, chưa có original-method entry. Không mở module, loss hoặc training mới từ riêng kết quả này. Goal tổng thể tiếp tục ACTIVE.
