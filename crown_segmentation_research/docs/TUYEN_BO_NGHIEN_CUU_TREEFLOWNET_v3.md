# Tuyên bố nghiên cứu (v3): phương pháp trường từ dưới lên cho tách tán cây — thua ở đâu, vì sao, sửa được bao nhiêu, và cần gì tiếp

*Bản 05/10/2026 cho buổi họp cổng giai đoạn 3. Bản này thay v2 (04/10, giữ nguyên để đối chiếu).*

- Mọi số đều từ file trong `clean_benchmark/results/`, bảng tổng hợp ở `HOAN_THANH_DU_AN/07_BANG_BIEU_SO_SANH.md` (sinh tự động), hình ở `HOAN_THANH_DU_AN/hinh/`.
- Nhãn: [đo] có số; [toán] suy ra; [giả thuyết] chưa có số.
- Quy trình: 3 seed huấn luyện; chọn mọi thứ trên validation; test chạy một lần cho mỗi cấu hình; CI 95% bằng bootstrap ghép cặp theo tile.

## 1. Câu hỏi và dữ liệu

- **Bài toán:** tách từng tán cây trên ảnh RGB drone 3 cm.
- **Dữ liệu:** CanopyRS/QuebecTrees, split cố định do SelvaBox phát hành.
  - Train: 148 tile, vùng Z1 và Z2.
  - Validation: 139 tile, chỉ vùng Z1.
  - Test: 168 tile (22 018 tán), trong đó **138 tile ở vùng Z3**, vùng không có trong train.
  - Test có 72.6% số tán dưới 9 m², train chỉ có 49.2%.
- **Rò rỉ:** không có. Vùng test chỉ trùng train ở viền 1 px (EXP-015). Mọi mô hình chỉ khởi tạo từ ImageNet. [đo]
- **Câu hỏi:** TreeFlowNet (flow + khoảng cách tới biên + điểm hút) thua Mask R-CNN cùng điều kiện. Thua ở đâu, vì cơ chế nào, và sửa đúng cơ chế thì được bao nhiêu?

## 2. Kết quả có số [đo]

| Cấu hình (test, 3 seed) | Backbone | AP | Δ ghép cặp [CI 95%] |
|---|---|---|---|
| TreeFlowNet, bộ giải mã gốc | Swin-T | 30.18 ± 0.23 | — |
| + gộp theo đèo (EXP-016) | Swin-T | 32.58 ± 0.22 | +2.41 [2.06, 2.80] |
| + chấm điểm lại theo chất lượng mask (EXP-023) | Swin-T | 33.75 ± 0.32 | +1.16 [1.01, 1.29] |
| + nới ngưỡng SDT của cổng (EXP-025) | Swin-T | **34.22 ± 0.30** | +0.47 [0.31, 0.56] |
| Mask R-CNN | R50 | 34.82 ± 0.67 | — |
| Mask R-CNN + chấm lại (EXP-024) | R50 | 34.80 ± 0.58 | −0.02 [−0.11, 0.09] |

- Ba cách sửa, **không train lại mạng**, cộng **+4.04 [3.60, 4.43]**.
- Khoảng cách tới Mask R-CNN (cả hai cùng chấm lại) đi từ 4.65 xuống **0.59 [0.15, 1.02]**.

**Cơ chế đã đo:**

1. **Thua ở hai đầu kích thước** (EXP-013, hình F2):
   - tán < 2 m² bị **gộp** vào tán bên cạnh (+0.098 trên khoảng cách 0.133);
   - tán ≥ 9 m² bị **chia** (+0.036);
   - ở 4–9 m², TreeFlowNet **hơn** Mask R-CNN.
2. **Hai họ mô hình có điểm yếu ngược nhau** (thế giới tổng hợp 2 × 2, EXP-010/011, hình F3):
   - hình không lồi làm TreeFlowNet mất khoảng 31.5 AP, Mask R-CNN chỉ khoảng 7.6;
   - mất biên làm Mask R-CNN giảm nhiều hơn TreeFlowNet.
3. **Phát hiện mới, EXP-026** (hình F4): cho chính bộ giải mã cuối đọc **trường đáp án** thay vì trường mạng dự đoán (validation):
   - AP đạt 91.8–95.7, so với khoảng 37 trên trường dự đoán;
   - không chia tán ≥ 9 m² nào (so với 0.037);
   - không gộp tán nhỏ nào (so với khoảng 0.20).

   ⇒ Bộ giải mã và định nghĩa target **không còn là nút thắt**. Phần mất còn lại nằm ở **chất lượng trường mạng học được**.

## 3. Tính mới: nói thẳng (chi tiết: `HOAN_THANH_DU_AN/14_TONG_QUAN_TAI_LIEU_VA_TINH_MOI.md`)

- **Biểu diễn không mới** (đã đọc toàn văn Omnipose, Nat Methods 2022; Deep Watershed Transform, CVPR 2017):
  - Flow = gradient đơn vị của khoảng cách tới biên chính là flow của Omnipose và vector hướng của Deep Watershed.
  - Điểm khác: mình chuẩn hoá khoảng cách theo từng tán.
- **Gộp theo đèo** là họ hàng của "cắt năng lượng ở một mức" (Deep Watershed). Phần riêng là lập luận về thang và số đo.
  - Omnipose có cách riêng (Euler giảm dần + DBSCAN). **Phải so trực tiếp** trước khi gọi đèo là đóng góp; đây là việc CPU, có thể làm ngay.
- **Đóng góp bảo vệ được:**
  - phân tích cơ chế lỗi của phương pháp trường trên benchmark cây cố định, có can thiệp đăng ký trước;
  - phép đo "trần bộ giải mã trên trường đáp án" (EXP-026). Phép đo này dùng được cho mọi phương pháp trường, kể cả trên tế bào.

## 4. Điều chưa được nói

- **"Bằng hoặc hơn Mask R-CNN":** chưa, còn thua 0.59 [0.15, 1.02].
- **So sánh chưa công bằng về backbone:** TreeFlowNet dùng Swin-T, Mask R-CNN dùng R50. Lệch này có lợi cho TreeFlowNet; TreeFlowNet R50 bộ giải mã gốc chỉ được 28.80.
  - Cũng chưa cùng ngân sách dò tham số: TreeFlowNet được chọn nhiều tham số giải mã trên validation hơn.
- **"Hơn SelvaMask (27.6)":** chưa xác nhận cùng giao thức. Họ chấm tối đa 300 tán mỗi tile, ghi GSD 1.9 cm, và bộ phát hiện của họ đã thấy QuebecTrees.
- **Tổng quát:** mới có một bộ dữ liệu, một site.

## 5. Kế hoạch (chi tiết: `HOAN_THANH_DU_AN/03_KE_HOACH_CONG_VIEC.md`, `12_…`, `17_…`)

| Giai đoạn | Việc | Cần train? |
|---|---|---|
| Ngay (CPU, suy luận) | So với giải mã kiểu Omnipose; TreeFlowNet R50 với đủ ba cách sửa (công bằng backbone); đối chiếu giao thức SelvaMask; trung bình trường qua lật/xoay (TTA) | Không |
| Baseline | Mask R-CNN Swin-T; Mask2Former Swin-T (3 seed mỗi mô hình, khoảng 30–40 giờ GPU) | Có (baseline) |
| Phương pháp (sau khi thầy đồng ý) | Loss "hiểu bộ giải mã": đẩy đèo trong tán lên trên τ, đèo giữa tán xuống dưới τ; đầu dự đoán cỡ tán; loss ràng buộc cho tán nhỏ (phương pháp nhân tử). Mỗi cái: thế giới tổng hợp → 1 seed → 3 seed | Có (khoảng 20 giờ GPU mỗi ứng viên) |
| Ngoài miền | BAMFORESTS (đã có dữ liệu), zero-shot và fine-tune | Có |

## 6. Cần thầy quyết định

1. **Hướng bài:**
   - (B) phân tích cơ chế + sửa đúng cơ chế, nhắm tạp chí viễn thám Q1 (ISPRS JPRS / RSE / TGRS / JAG). Đã có khoảng 70–80% bằng chứng.
   - (A) phương pháp mới cho hội nghị A*. Chỉ khả thi nếu phương pháp mới thắng rõ baseline **cùng backbone** trên ≥ 2–3 bộ dữ liệu.
2. **Nơi nộp và hạn nộp.**
3. **Có cho train** target / loss / đầu ra mới không (mục 5, hàng "Phương pháp")?
4. **Có cho commit code** để tái lập không?
5. **Kết quả chính:** dùng AP tổng hay báo kèm AP theo kích thước và kiểu lỗi (gộp / chia / bỏ sót)?
