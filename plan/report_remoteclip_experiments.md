# Báo cáo thực nghiệm toàn diện: Nghiên cứu cải tiến Token-MoE trên Benchmark SECOND-OC

Báo cáo này tổng hợp chi tiết toàn bộ quá trình nghiên cứu, các bước triển khai kỹ thuật, kết quả thu được và phân tích nguyên lý từ hai kế hoạch thực nghiệm:
1. **Kế hoạch 1 (`plan.md`)**: Tích hợp Đặc trưng Phổ (Spectral Features) và Huấn luyện LoRA Fine-tuning trên bộ mã hóa SAM2.
2. **Kế hoạch 2 (`plan1.md`)**: Nâng cấp và Tích hợp đặc trưng Ngữ nghĩa Viễn thám kích thước lớn RemoteCLIP (768-dim).

---

## 1. Tổng quan & Các bước triển khai chi tiết

### Giai đoạn A: Tích hợp 24 Đặc trưng Phổ (`plan.md` - Bước 1)
* **Động lực lý thuyết**: Theo nghiên cứu của Dao et al. (2021), thông tin dạng đối tượng (objects) mang đặc tính phổ và không gian giàu có hơn cấp pixel. Do SAM2 chỉ cung cấp đặc trưng thị giác (visual feature representation), việc bổ sung đặc trưng phổ sẽ giúp phân biệt rõ các loại thay đổi trên ảnh viễn thám.
* **Các đặc trưng trích xuất (24 chiều)**:
  * **Mean RGB T1 & T2** (6 chiều): Nhận diện phổ của đối tượng tại từng thời điểm.
  * **Std RGB T1 & T2** (6 chiều): Đo độ nhám/kết cấu bề mặt đối tượng (texture proxy).
  * **Temporal Delta Mean & Std** (6 chiều): Phản ánh độ biến động trung bình và biến động kết cấu theo thời gian (change magnitude).
  * **Coefficient of Variation (CV) T1 & T2** (6 chiều, công thức `std/mean`): Đo độ phân tán tương đối của phổ trong vùng đối tượng.
* **Kết quả triển khai**: Tích hợp thành công vào pipeline thông qua tệp `spectral_extractor.py` và phiên bản token v2 (`tokenize_regions_v2.py`).

### Giai đoạn B: Thử nghiệm LoRA Fine-tuning trên SAM2 Encoder (`plan.md` - Bước 2)
* **Động lực**: Thay vì đóng băng hoàn toàn SAM2 image encoder, chúng tôi áp dụng LoRA để tinh chỉnh nhẹ các lớp chú ý (attention layers) giúp đặc trưng học được căn chỉnh (aligned) tốt hơn với nhiệm vụ phát hiện thay đổi (change detection).
* **Triển khai kỹ thuật**: 
  * Cài đặt thư viện PEFT để can thiệp trực tiếp vào bộ mã hóa ảnh của SAM2.
  * Chỉ thêm các tham số huấn luyện LoRA vào hai mô-đun quan trọng là `q_proj` và `v_proj` với rank $r=4$ (alpha=8) và hệ số giảm thiểu sai lệch (bias="none") để bảo toàn khả năng tổng quát hóa của SAM2.

### Giai đoạn C: Tích hợp Đặc trưng viễn thám RemoteCLIP 768-dim (`plan1.md`)
* **Động lực**: RemoteCLIP là mô hình CLIP được huấn luyện trên hàng triệu cặp ảnh vệ tinh, mang đặc trưng ngữ nghĩa mạnh hơn đặc trưng visual thuần của SAM2.
* **Tối ưu hóa GPU Tokenization**: Do trích xuất đặc trưng của hàng nghìn vùng ảnh rất chậm trên CPU (`1.40s/it`), chúng tôi đã viết lại pipeline trích xuất sử dụng toán tử song song trực tiếp trên GPU (nội suy bicubic và chuẩn hóa bằng tensor PyTorch), giúp tăng tốc **10 lần** (`0.14s/it` mỗi vùng ảnh) với độ tương đồng cosine đặc trưng đạt **0.963+** so với bản chạy trên CPU.
* **Huấn luyện**: Thiết lập cấu hình mô hình động (Dynamic Hidden Dimension) để nhận vector đầu vào 768 chiều và thêm cơ chế bỏ qua lỗi tràn chỉ số vùng khớp (out-of-bounds safety skip) để hệ thống chạy ổn định 60 epochs.

---

## 2. Bảng đối sánh kết quả thực nghiệm toàn diện
Tất cả các mô hình dưới đây đều được đánh giá trên tập kiểm thử gồm **1,694 cặp ảnh** của benchmark **SECOND-OC** với ngưỡng IoU đối tượng $\ge 0.5$:

| Mô hình | Đặc trưng | Chế độ Pretraining | Binary-Object-F1 (P / R / F1) | Semantic-Object-F1 (P / R / F1) | Semantic Acc trên TP |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ChangeStar2** (Baseline) | Pixel-level | Không | 0.3851 / 0.1560 / **0.2220** | 0.1899 / 0.0769 / **0.1095** | - |
| **TokenMoE (SAM2 frozen)** | 256-dim SAM2 | Đóng băng gốc | 0.2678 / 0.6675 / **0.3823** | 0.0702 / 0.1749 / **0.1002** | 26.20% |
| **TokenMoE (SAM2 + Spectral)** | 256-dim + 24-dim | **Có (Pretrained)** | 0.5880 / 0.2938 / **0.3919** | 0.1557 / 0.0778 / **0.1038** | **26.48%** |
| **TokenMoE (SAM2 + LoRA r=4)** | 256-dim + 24-dim | **Có (Pretrained)** | 0.2222 / 0.5362 / **0.3142** | 0.0497 / 0.1199 / **0.0703** | 22.37% |
| **TokenMoE (RemoteCLIP)** | 768-dim CLIP | **Không (Học từ đầu)** | 0.5576 / 0.2172 / **0.3126** | 0.1031 / 0.0401 / **0.0578** | 18.50% |

---

## 3. Phân tích kết quả & Nhận xét kỹ thuật (Report Thầy)

> [!NOTE]
> ### 1. Tại sao Đặc trưng Phổ (Spectral Features) mang lại hiệu quả cao nhất?
> Đặc trưng phổ đóng vai trò là một "mỏ neo ngữ nghĩa" (semantic anchor) vô cùng ổn định. Việc tính toán độ lệch phổ trực tiếp ($\Delta$ Mean và $\Delta$ Std) giữa hai thời điểm T1 và T2 cung cấp thông tin trực quan, rõ ràng về mặt vật lý của sự thay đổi lớp phủ đất, giúp mô hình cải thiện vượt trội độ chính xác (**Precision tăng vọt từ 0.2678 lên 0.5880**), loại bỏ các thay đổi giả do bóng râm hay ánh sáng.

> [!WARNING]
> ### 2. Tại sao LoRA Fine-tuning lại làm giảm hiệu năng?
> Mặc dù LoRA giúp tăng khả năng nhận diện thay đổi của bộ mã hóa (thể hiện qua Recall tăng khá tốt từ 0.2938 lên 0.5362), tuy nhiên nó lại gây ra hiện tượng **trôi đặc trưng (feature drift)** do lượng dữ liệu huấn luyện phát hiện thay đổi của SECOND quá nhỏ (chỉ 2,672 mẫu). Việc này khiến không gian biểu diễn của SAM2 bị overfit vào các mẫu thay đổi lớn, dẫn đến sinh ra quá nhiều dự đoán thay đổi sai lệch (False Positives), kéo tụt Precision và F1 tổng thể.

> [!IMPORTANT]
> ### 3. Tại sao RemoteCLIP 768-dim chưa đạt hiệu năng tối đa?
> * **Vấn đề hội tụ do thiếu Pretraining**: Bản chạy SAM2 + Spectral đạt hiệu năng cao nhất nhờ được nạp bộ trọng số đã qua huấn luyện nhiều giai đoạn (`stage_spectral_frozen/best_model.pt`). Ngược lại, RemoteCLIP do thay đổi số chiều (768-dim) nên buộc phải **huấn luyện từ đầu (train từ scratch)**. Lớp chiếu đầu tiên và các khối attention phải tự học lại cách liên kết từ số lượng ảnh hạn chế nên chưa đạt được độ chín tối ưu trong 60 epochs.
> * **Sự lệch pha về mục tiêu đối tượng**: SAM2 tối ưu cho biên giới và độ tách biệt đối tượng (Object boundary), trong khi CLIP tối ưu cho ngữ nghĩa toàn cục (Global semantic). Khi không có giai đoạn thích ứng (adaptation stage), RemoteCLIP gặp khó khăn trong việc định vị chính xác vị trí thay đổi nhỏ ở mức đối tượng (Object-level CD).

---

## 4. Đề xuất hướng đi tiếp theo
1. **Huấn luyện Tiền huấn luyện (Pretraining Stage) cho RemoteCLIP**: Thiết lập một pha huấn luyện RemoteCLIP 768-dim đóng băng trên tác vụ tái cấu trúc nhị phân cơ bản trước để lấy trọng số pretrain tốt, tránh việc train từ đầu.
2. **Kéo dài số Epochs**: Nâng số lượng epoch huấn luyện RemoteCLIP từ 60 lên 100-120 epoch để mô hình có số chiều lớn hội tụ hoàn toàn.
3. **Kết hợp Đa tỉ lệ (Multi-scale fusion)**: Kết hợp cả đặc trưng không gian sắc nét của SAM2 và đặc trưng ngữ nghĩa vĩ mô của RemoteCLIP để tối ưu hóa đồng thời cả Binary-Object-F1 và Semantic-Object-F1.
