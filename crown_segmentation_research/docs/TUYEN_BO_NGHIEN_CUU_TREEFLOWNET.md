# Tuyên bố nghiên cứu: TreeFlowNet trên CanopyRS/QuebecTrees

*Bản nháp ngày 02/10/2026 để thảo luận với thầy. Mọi số đo nằm trong `clean_benchmark/results/` và `docs/KIEN_TRUC_TREEFLOWNET_CLEAN_TOAN_HOC_VA_DIEM_YEU.md`. Nhãn: **[đo]** có số; **[giả thuyết]** chưa có số. Seed 42/43/44 là ba lần huấn luyện độc lập của cùng cấu hình; mọi kết quả test chạy một lần, chọn cấu hình chỉ trên validation.*

## 1. Câu hỏi

Phân đoạn từng tán cây (instance segmentation) trên ảnh RGB drone 3 cm. Trên CanopyRS/QuebecTrees (test: 22.018 tán, 72.6% dưới 9 m², chỉ 4 tán trên 100 m²), một phương pháp bottom-up dựa trên trường (flow + khoảng cách có dấu) hiện thua Mask R-CNN cùng điều kiện. **Vì sao thua, và thua ở đâu?**

## 2. Khẳng định hiện có thể bảo vệ (chỉ phần có số đo)

1. **Khoảng cách tới Mask R-CNN [đo].** Test AP[.5:.95], maxDets 400: Mask R-CNN R50 34.82 ± 0.67 (3 seed); TreeFlowNet Swin-T 30.17 (30.43 / 29.96 / 30.13); R50 28.81 (một seed). Hai bên cùng chỉ lật/xoay 90°, cùng crop 1024, cùng khởi tạo ImageNet.
2. **Khoảng cách không đều theo kích thước [đo, một seed, test].** Recall IoU 0.5: TreeFlowNet Swin-T thắng Mask R-CNN ở 4–9 m² (0.828 so với 0.775), thua ở dưới 2 m² (1–2 m²: 0.276 so với 0.496) và ở 25–100 m² (0.834 so với 0.956).
3. **Phần lớn lỗi nằm ở luật chọn seed [đo, validation, 3 seed].** Giữ nguyên checkpoint và mọi cấu hình, chỉ đổi luật seed sang luật dùng GT: AP validation 33.65 lên 37.96 (+4.31, +4.50/+4.30 từng seed); Mask R-CNN cùng validation 38.21 (một seed). Đây là cận trên, không phải phương pháp.
4. **Lỗi crown nhỏ có hai nguồn [đo, validation, một seed].** Với crown 1–2 m²: khoảng 38% không có peak nào trong crown (lỗi của trường), khoảng 21 điểm phần trăm có peak nhưng bị NMS bán kính cố định loại (lỗi giải mã). Trên test, 88–93% crown dưới 4 m² chạm biên một crown khác (≤ 3 px) và khoảng một nửa số crown chạm biên bị mất là bị nuốt vào crown kề; Mask R-CNN có tỉ lệ nuốt tương đương.
5. **Bộ chấm seed học được sửa được một phần [đo, test, 3 seed].** Thay NMS cố định bằng bộ chấm (15 đặc trưng, huấn luyện trên seed của validation, không dùng GT lúc suy luận): AP test 30.17 lên 31.85 (+1.68; +1.62/+1.78/+1.65) với ngưỡng τ = 0.3. Lưu ý τ = 0.3 chọn trên validation, còn τ = 0.5 chốt từ đầu chỉ +0.5 (30.66). Còn khoảng 3.0 AP tới Mask R-CNN.

## 3. Khẳng định chưa được phép nói

- "TreeFlowNet bắt kịp hoặc vượt Mask R-CNN": chưa; hiện kém 3.0 AP sau khi sửa luật seed.
- "Nguyên nhân gốc là thang kích thước cố định": **[giả thuyết]**. Bằng chứng nhất quán nhưng chưa tách khỏi các nguyên nhân khác (trọng số loss theo diện tích, độ phân giải stride 4, bộ lọc diện tích).
- "Bottom-up thua vì bản chất paradigm": chỉ một bộ dữ liệu, một họ mô hình. Không có thí nghiệm ngoài miền.
- So với số công bố của SelvaMask (27.6 mAP trên QuebecTrees, SelvaBox → SAM3): SelvaMask Bảng 4 đánh dấu số này là huấn luyện trên miền đích (✗, SelvaBox được huấn luyện trên QuebecTrees), **không phải zero-shot**. Nhưng giao thức mAP mức tile của họ khác và chưa chạy lại trên đúng test của mình, nên chưa so trực tiếp được. (Tôi đã từng ghi nhầm đây là zero-shot; đã sửa.)

## 4. Bằng chứng còn thiếu

| Thiếu | Vì sao cần | Chi phí |
|---|---|---|
| Phân tích recall theo kích thước cho bộ chấm seed | Biết nó sửa ở bin nào | CPU, nhanh |
| Pipeline thế giới trên đúng test của mình: SelvaBox → SAM2/SAM3 (suy luận), Cellpose-SAM zero-shot | Trả lời "đứng ở đâu" bằng số cùng giao thức; kiểm tra tập train của SelvaBox có chồng với test của mình không (SelvaBox được huấn luyện trên QuebecTrees) | Suy luận GPU, không train |
| Sửa trường: loss cân theo số cây, augmentation theo thang (thêm cho cả Mask R-CNN) | Kiểm chứng nhân quả cho nhánh "không có peak" (khoảng 38%) | Train lại, nhiều giờ |
| Mask R-CNN backbone Swin-T | Loại trừ khoảng cách do backbone | Train |
| Ngoài miền (BAMFORESTS TestSet-1/2) | Xem prior theo mét có chuyển được không | Suy luận + giải mã |
| Kiểm định thống kê (bootstrap theo tile) | Khoảng tin cậy cho các chênh +1.7 AP | CPU |

## 5. Ba hướng đóng góp, mỗi hướng cần khác nhau

- **A. Kỹ thuật:** TreeFlowNet bắt kịp Mask R-CNN. Cần: sửa trường (loss, thang) + bộ chấm seed. Rủi ro: kể cả thắng vẫn có thể dưới pipeline SAM; tính mới yếu (flow: Cellpose, FG-TreeSeg; seed học: EmbedSeg, InstanSeg).
- **B. Phân tích cơ chế:** vì sao bottom-up mất crown nhỏ chen chúc (tỉ lệ lỗi trường:giải mã khoảng 2:1, nuốt vào hàng xóm, phụ thuộc thang). Phần lớn bằng chứng đã có; cần thêm một thí nghiệm sửa trường và so sánh ngoài miền để thành nhân quả.
- **C. Hybrid:** seed bottom-up học được làm prompt cho SAM. Cần pipeline mới và SAM; chỉ nên làm sau khi có bảng xếp hạng ở mục 4.

## 6. Câu hỏi muốn hỏi thầy

1. Chọn A, B hay C (hay B trước, rồi A/C tuỳ kết quả)?
2. Nhãn chỉ là tập con crown thật và ranh giới giữa tán cùng màu mờ: thầy muốn đưa "phổ dẫn đường biên" (OBIA) vào ở đâu: đặc trưng cho bộ chấm seed, hay loss ở biên?
3. Chấp nhận báo cáo AP theo kích thước và kiểu lỗi (matched/merged/split/missed) làm đóng góp chính, thay vì chỉ AP tổng?
