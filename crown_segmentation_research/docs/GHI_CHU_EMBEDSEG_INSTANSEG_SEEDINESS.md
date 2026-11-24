# Ghi chú: seediness và bán kính học được (EmbedSeg, InstanSeg)

Nguồn: EmbedSeg (Lalit et al., MedIA 81:102523, 2022; PDF `paper/EmbedSeg.pdf`, mục 3 và 3.1, đọc từ `paper/texts/EmbedSeg.txt`);
InstanSeg (arXiv 2408.15954, đọc qua bản HTML, chỉ phần phương pháp; chưa đọc kỹ phần thực nghiệm).

## EmbedSeg
- Xác suất thuộc đối tượng k: φ_k(e_i) = exp(−‖(e_i − C_k)ᵀ Σ_k⁻¹ (e_i − C_k)‖ / 2), gán pixel nếu φ_k > 0.5.
- **Băng thông học được:** mỗi pixel dự đoán σ_i; σ_k của đối tượng = trung bình σ_i trên các pixel của đối tượng. Đối tượng lớn học σ lớn. Loss làm mượt L_var = mean ‖σ_i − σ_k‖².
- **Seediness:** mỗi pixel dự đoán s_i, giám sát bằng chính φ_k(e_i) cho pixel tiền cảnh và 0 cho nền: L_seed = mean[w_fg·1(i∈S_k)‖s_i − φ_k(e_i)‖² + w_bg·1(i∉S_fg)‖s_i‖²]. Nghĩa là s_i cao khi pixel nằm gần tâm cụm của đối tượng của nó, kể cả không nằm đúng tâm hình học.
- Điểm tâm là **medoid** (pixel của đối tượng có khoảng cách trung bình nhỏ nhất tới các pixel còn lại), vì centroid/geometric median có thể nằm ngoài đối tượng.
- Suy luận: lấy S_fg = {s_i > 0.5}; chọn pixel có s lớn nhất (> s_min = 0.9); gom các pixel của S_fg có φ > 0.5 theo (e_seed, σ_seed); bỏ chúng khỏi S_fg; lặp. Không có NMS bán kính cố định: **bán kính nằm trong σ_seed học được**.
- Ghép tile: ghép IoU hai chiều trong vùng chồng, ngưỡng 0.5.

## InstanSeg
- Seed map S giám sát bằng L1 tới khoảng cách tương đối tới biên gần nhất; seed lấy ở cực đại cục bộ của S (cả lúc train và lúc suy luận).
- Với mỗi seed k: Q_ij − Q_k = (P_ij + O_ij) − (P_k + O_{u_k}); MLP một lớp ẩn nhận (Q_ij − Q_k, E_ij) và cho ra logit Φ_ij^k; loss Lovász-hinge theo từng instance, K tối đa 50 seed mỗi crop khi train.
- Instance lớn có thể có nhiều seed: mạng được train để dự đoán trọn instance bất kể seed ở đâu; các mask trùng nhau (IoU cao) được gộp bằng hợp. Nhãn cuối: argmax_k Φ^k nếu ≥ 0, ngược lại nền.
- Siêu tham số công bố: lr 1e-3, batch 3, crop 256, 500 epoch, khởi động 10 epoch bằng BCE+Dice. Dữ liệu là tế bào, không phải ảnh rừng.

## Hệ quả cho bộ chấm seed của TreeFlowNet
1. Seediness của EmbedSeg là một **đầu hồi quy thưa/mềm giám sát bằng φ_k**, không phải bộ phân loại sau cùng. Với TreeFlowNet, centroid head cũng là đầu kiểu này và bỏ nó lại tăng 2.99 AP, nên không nên lặp lại đúng thiết kế đó. (Nguyên nhân chưa biết, chưa tách được.)
2. Điều khác biệt đáng thử là **nhãn từ ghép với GT** (seed nào cho mask khớp một crown thật) và **bán kính/độ phủ học được**, đúng tinh thần bước 1–2 của lộ trình.
3. Cả hai bài đều dùng seed làm điểm vào và gộp trùng lặp, thay vì NMS bán kính cố định. Điều này khớp với kết quả oracle (gộp quan trọng hơn chia).
4. Những công thức trên thuộc ảnh tế bào; chưa có bằng chứng chúng chuyển sang tán cây.
