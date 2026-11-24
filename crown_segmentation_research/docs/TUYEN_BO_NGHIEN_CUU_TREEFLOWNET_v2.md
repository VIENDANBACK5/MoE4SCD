# Tuyên bố nghiên cứu (v2): vì sao phương pháp trường "từ dưới lên" mất tán nhỏ và chia tán lớn, và sửa đúng cơ chế

*Bản nháp 04/10/2026 để họp thầy (cổng giai đoạn 3). Thay bản 02/10 (`TUYEN_BO_NGHIEN_CUU_TREEFLOWNET.md`, giữ nguyên để đối chiếu). Mọi số có thẻ thí nghiệm trong `target/NHAT_KY_THI_NGHIEM.md` và file trong `clean_benchmark/results/`. Nhãn: [đo] có số; [giả thuyết] chưa có số. Seed 42/43/44 = ba lần huấn luyện độc lập; chọn mọi tham số trên validation, test chạy một lần; khoảng tin cậy = bootstrap ghép cặp theo tile (2000 lần).*

## 1. Câu hỏi

Tách từng tán cây trên ảnh RGB drone 3 cm (CanopyRS/QuebecTrees; test 22 018 tán, 73% dưới 9 m²). TreeFlowNet (flow + khoảng cách tới biên + điểm hút) thua Mask R-CNN cùng điều kiện. **Thua ở đâu, vì cơ chế nào, và sửa đúng cơ chế đó thì được bao nhiêu?**

## 2. Khẳng định có số

1. **Khoảng cách là thật** [đo]: test AP Mask R-CNN 34.82 so với TreeFlowNet 30.17; chênh 4.65 [4.11, 5.15]. Không do lệch biên vài cm (nới dung sai biên làm chênh rộng ra, EXP-004) và không do rò rỉ dữ liệu (EXP-015: vùng test chỉ trùng train ở viền 1 px).
2. **Thua ở hai đầu kích thước, hơn ở giữa** [đo, EXP-013]: tán < 2 m² mất vì bị **gộp** vào tán bên cạnh (0.098 của 0.133); tán ≥ 9 m² mất vì bị **chia**; ở 4–9 m² TreeFlowNet hơn Mask R-CNN 0.040 recall. Biên mờ làm khó mọi mô hình như nhau (EXP-008/008b).
3. **Chia tán là điểm yếu riêng của biểu diễn trường** [đo, can thiệp trên dữ liệu tổng hợp, EXP-010/011]: tán không lồi làm TreeFlowNet mất 31.5 AP (2.2–2.4 điểm hút mỗi tán), Mask R-CNN chỉ mất 7.6 và gần như không chia. Trên dữ liệu thật: tán càng không lồi TreeFlowNet càng bị chia hơn Mask R-CNN (EXP-012, nhất quán).
4. **Gốc chung của hai lỗi** [code, toán]: target chuẩn hoá khoảng cách theo từng tán (tâm tán nào cũng = 1) và flow có độ dài 1, nên mạng không mang thông tin cỡ tán; bộ giải mã phải dùng một bán kính cố định 1.5 m, quá lớn với tán nhỏ (bán kính 0.56 m ở 1 m²) và quá nhỏ với tán lớn nhiều thuỳ.
5. **Sửa đúng cơ chế, không train lại, được một nửa khoảng cách** [đo, EXP-016]: thay bán kính cố định bằng tiêu chí "đèo" (hai vùng hút là hai tán khi khoảng cách-tới-biên giữa chúng tụt về gần 0): test **30.17 → 32.58, +2.41 [2.06, 2.80]**; recall 1–2 m² +0.16; tỉ lệ chia ≥ 9 m² giảm một nửa; **Mask R-CNN − TreeFlowNet: 4.65 → 2.24 [1.74, 2.63]**. Thu 66% trần "biết đúng liên kết seed" (validation 33.65 → 36.50, trần 37.96).
6. **Chấm điểm mask theo chất lượng, cũng không train mạng** [đo, EXP-022/023]: điểm chọn tay xếp hạng mask kém (xếp hạng hoàn hảo cho 46.7 so với 36.5 trên validation). Một bộ hồi quy nhỏ dự đoán IoU của mask (có flow error kiểu Cellpose) học trên validation: test **32.58 → 33.74, +1.16 [1.01, 1.29]**; tổng so với ban đầu **+3.57 [3.20, 3.96]**; **Mask R-CNN − TreeFlowNet: 4.65 → 1.08 [0.61, 1.46]**. Lợi ích test chỉ khoảng 58% lợi ích validation.

## 3. Chưa được phép nói

- "Bắt kịp Mask R-CNN": chưa; còn 1.08 AP [0.61, 1.46] (một seed không còn khác 0, hai seed còn).
- Về công bằng (EXP-024): áp cùng cách chấm điểm lại cho Mask R-CNN bằng đầu ra của chính nó (mask + điểm) không giúp nó trên test (−0.02 [−0.11, 0.09]); khoảng cách công bằng **1.06 [0.62, 1.42]**. Khoảng 90% lợi ích chấm lại của TreeFlowNet đến từ đặc trưng trường (đỉnh SDT, flow error), thứ Mask R-CNN không có. Chưa thử Mask Scoring R-CNN thật (đầu chấm điểm học trong mạng), cần train.
- Cách sửa có giá: tán 4–9 / ≥ 9 m² biên yếu bị gộp nhiều hơn (recall 4–9 m² −0.033). Sửa tiếp bằng tín hiệu sẵn có đã cạn (EXP-020/021: tốt nhất +0.07; phủ quyết đổi gộp nhầm thành chia nhầm), cần thông tin cỡ tán.
- Khoảng một nửa tán nhỏ bị mất vì đầu "có phải cây" coi là nền (EXP-006/007); tiêu chí đèo không chạm tới phần này.
- Một bộ dữ liệu, một site, một ngày bay; chưa có ngoài miền.

## 4. Đóng góp đề xuất (hướng B, không dùng SAM: Chung đã chọn)

Phân tích cơ chế thất bại của phương pháp trường ở hai đầu kích thước **và** một cách sửa đúng cơ chế: thông tin "cùng / khác tán" lấy từ đèo của trường khoảng cách (không cần thang cố định), bổ sung bằng **thông tin cỡ tán** cho tán lớn (EXP-017: biết đúng cỡ tán giúp tán lớn, đèo giúp tán nhỏ; hai cái bổ sung nhau, trần oracle 37.06).

## 5. Bằng chứng còn thiếu (cần train → cần thầy đồng ý)

| Thiếu | Kiểm cái gì | Chi phí |
|---|---|---|
| Đầu dự đoán cỡ tán (khoảng cách tuyệt đối, log mét) + kết hợp với đèo | Sửa gộp nhầm tán lớn; nhân quả cho "thiếu thông tin cỡ" | Train lại TreeFlowNet, 3 seed |
| Loss canopy cân theo tán (mỗi tán đóng góp như nhau) | Nửa tán nhỏ bị coi là nền | Train lại, 3 seed |
| Ngoài miền (BAMFORESTS) | Tổng quát | Suy luận |
| Gán nhãn lại mù (bộ EXP-009 đã sẵn, cần 3 người) | Tán nhỏ bị mất: ảnh không đủ thông tin hay mô hình chưa học | Người gán |

## 6. Câu hỏi cho thầy

1. Đồng ý hướng B (cơ chế + sửa đúng cơ chế), không SAM?
2. Cho phép train lại với target / đầu ra mới (đầu cỡ tán, loss canopy) ngay bây giờ?
3. Hội nghị / tạp chí mục tiêu và hạn nộp?
4. Báo AP theo kích thước và kiểu lỗi (gộp / chia / bỏ sót) làm kết quả chính, không chỉ AP tổng?
