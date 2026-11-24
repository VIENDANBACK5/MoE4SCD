# Hazard Tree Instance Detection via Deep Neural Network Models

- **Tên bài báo:** HAZARD TREE INSTANCE DETECTION VIA DEEP NEURAL NETWORK MODELS
- **Tác giả:** Pranav Prakash, Ankur Saxena
- **Đơn vị:** AiDash (Bangalore, India)
- **Hội nghị:** IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2025)
- **DOI:** [10.1109/IGARSS55030.2025.11243161](https://doi.org/10.1109/IGARSS55030.2025.11243161)
- **Phân loại phương pháp:** Foundation Model-based Semantic Segmentation (DINOv2 U-Net) kết hợp Instance Post-Processing (TTA & Morphological Opening)

---

## 1. Bài toán & Bối cảnh Nghiên cứu

- **Nhiệm vụ (Task):** Phát hiện từng cá thể cây nguy hiểm / cây chết đứng (Hazard Tree & Standing Dead Tree Instance Detection) nằm gần hành lang lưới điện nhằm ngăn ngừa nguy cơ gãy đổ gây chập cháy, mất điện trong mùa giông bão.
- **Hệ sinh thái:** Đa dạng địa hình trên quy mô toàn cầu, bao gồm rừng rậm (Forests), đồng cỏ/savannah (Grasslands), đồi núi (Hilly) và khu vực đô thị (Urban) trải dài qua 4 quốc gia: Hoa Kỳ, Canada, Bỉ và Australia.
- **Dữ liệu đầu vào & Độ phân giải:**
  - **Ảnh vệ tinh siêu phân giải (VHR Satellite):** Vệ tinh Pléiades Neo, độ phân giải **30 cm**, dữ liệu phản xạ bề mặt 16-bit đã hiệu chỉnh khí quyển.
  - **Ảnh hàng không (Aerial Photography):** Cảm biến Hexagon và Vexcel, độ phân giải siêu cao **7 – 15 cm**, dữ liệu ảnh 8-bit.
- **Quy mô tập dữ liệu:**
  - Tổng cộng 26,860 đa giác (polygons) gán nhãn chi tiết: Rừng (13,352), Đồng cỏ (9,896), Đô thị (3,612).
  - Tập huấn luyện chỉ dùng ảnh tại Mỹ; kiểm thử ngoại suy (zero-shot transfer) trên Canada, Bỉ, Australia và bộ dữ liệu mở NAIP của Cheng et al. (Nature Comms, 2024).

---

## 2. Chi tiết Phương pháp Phân đoạn (Segmentation Methodology)

```
       [ Ảnh VHR: Satellite 30cm (16-bit) / Aerial 7-15cm (8-bit) ]
                                    │
    Pre-processing: Percentile-based Normalization (1st - 99th percentile)
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │   DINOv2 Encoder (ViT Pre-trained Foundation Model)    │
       │   Trích xuất đặc trưng từ 4 Transformer Blocks cuối    │
       └────────────────────────────┬───────────────────────────┘
                                    │ Skip Connections
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │   ResNet Decoder Blocks                                │
       │   (Conv 3x3 + BatchNorm + ReLU + Bilinear Upsampling)  │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
       Loss: L_total = alpha * L_FCE + (1 - alpha) * L_WFT
         - L_FCE: Focal Cross Entropy (cân bằng mẫu khó/dễ)
         - L_WFT: Weighted Focal Tversky Loss (Đặc biệt: gamma < 1)
         - Class Weighting: w_c = 1 / log(1.02 + N_c / N_p)
                                    │
                                    ▼
       Post-Processing & Instance Extraction:
         - Test-Time Augmentation (TTA: Scale 0.8x, 1.2x, Rotate 90°)
         - Phép mở nhị phân (Binary Opening Morphological Operation)
         - Trích xuất polygon cá thể cây chết (Vectorization)
```

### 2.1. Kiến trúc Mô hình: DINOv2-based U-Net
- **Encoder:** Sử dụng mô hình nền tảng thị giác **DINOv2** (Oquab et al., 2023) tự giám sát (self-supervised). Để tạo kiến trúc đa mức kiểu U-Net, tác giả trích xuất biểu diễn đặc trưng từ **4 khối Transformer cuối cùng** của DINOv2, chuyển đổi thành các feature map không gian có độ phân giải phân cấp để đưa vào các đường kết nối tắt (skip connections).
- **Decoder:** Gồm các khối tích chập ResNet (Conv block: Conv 3x3 $\to$ Batch Normalization $\to$ ReLU) kết hợp với các tầng nội suy tăng kích thước (upsampling).
- **Lý do chọn DINOv2:** Khả năng tổng quát hóa vượt bậc so với ResNet50 truyền thống khi chuyển giao giữa các cảm biến (satellite 30cm vs aerial 7-15cm) và địa hình khác biệt.

### 2.2. Tiền xử lý (Pre-processing)
- Đối với ảnh hàng không 8-bit: Chia cho 255 đưa về đoạn $[0, 1]$.
- Đối với ảnh vệ tinh 16-bit có phân phối đuôi dài (heavy-tailed distribution): Áp dụng **chuẩn hóa phân vị (Percentile-based normalization)** từ phân vị 1% đến 99% để loại bỏ triệt để các pixel nhiễu cực sáng hoặc cực tối.

### 2.3. Thiết kế Hàm mất mát Đột phá (Loss Functions)
Hàm mất mát tổng thể kết hợp giữa Focal Cross Entropy ($L_{FCE}$) và Weighted Focal Tversky Loss ($L_{WFT}$):

$$\mathcal{L}_{total} = \alpha \mathcal{L}_{FCE} + (1 - \alpha) \mathcal{L}_{WFT}$$

Trong đó, hàm $L_{WFT}$ được định nghĩa:

$$\mathcal{L}_{WFT} = \left( \sum_c w_c \cdot \left( 1 - \frac{\sum (y \cdot \hat{y} + \epsilon)}{\sum ((1 - F_\beta) \cdot y + F_\beta \cdot \hat{y} + \epsilon)} \right) \right)^\gamma$$

- **Trọng số lớp (Class weighting):** Áp dụng công thức ENet (Paszke et al., 2016):
  $$w_c = \frac{1}{\log(1.02 + N_c / N_p)}$$
  giúp kiểm soát sự mất cân bằng cực lớn giữa nền đất/rừng xanh và cây chết (chỉ chiếm $< 1\%$ diện tích).
- **Phát hiện quan trọng về siêu tham số $\gamma$ (Gamma):**
  - Trong lý thuyết Focal Loss thông thường, người ta thường đặt $\gamma > 1$ (như $\gamma = 2.0$) để phạt nặng các mẫu khó.
  - Tuy nhiên, qua phân tích fANOVA và Optuna, tác giả phát hiện: **Đặt $\gamma < 1$ (cụ thể $\gamma = 0.75 - 0.90$) mới đem lại kết quả tốt nhất trên ảnh viễn thám!**
  - **Lý giải vật lý:** Trên ảnh viễn thám, dữ liệu nhãn luôn có độ nhiễu và mơ hồ (ví dụ bóng đổ của cành cây khô, nhãn bị sót). Nếu đặt $\gamma > 1$, mô hình sẽ tập trung phạt cực mạnh vào các điểm nhiễu/nhãn sai này. Đặt $\gamma < 1$ giúp triệt tiêu ảnh hưởng của các ngoại lai có loss cao vô lý, giúp mô hình hội tụ ổn định.
  - Đặt $F_\beta > 0.6$ ưu tiên Recall để bắt được cả các cành cây chết nhỏ nhất.

### 2.4. Hậu xử lý (Post-processing)
1. **Test-Time Augmentation (TTA):** Suy luận trung bình qua các biến thể: ảnh gốc, co giãn $0.8\times$, $1.2\times$, và xoay $90^\circ$.
2. **Binary Opening (Phép mở hình thái học):** Loại bỏ các điểm pixel cô lập rải rác và cắt đứt các cầu nối mỏng giữa hai cây chết gần nhau.
3. **Vector hóa & Đánh giá cá thể:** Đa giác hóa và tính True Positive khi IoU với GT $\ge 10\%$ (do tán cây chết không có lá, diện tích pixel rất nhỏ).

---

## 3. Kết quả Thực nghiệm & Số liệu Định lượng

### 3.1. So sánh Backbone trên các độ phân giải khác nhau (F1-score @ IoU 10%)

| Mô hình (Backbone) | Ảnh vệ tinh Pléiades Neo (30 cm) | Ảnh hàng không Hexagon/Vexcel (7 – 15 cm) |
|---|:---:|:---:|
| **ResNet-50 v2** | 62.0 | 82.0 |
| **SatlasPretrain (Remote Sensing Foundation Model)** | 63.0 | 85.0 |
| **DINOv2 (Proposed)** | **65.0** | **91.0** |

### 3.2. Khả năng khái quát hóa đa địa hình của DINOv2 U-Net

| Hệ sinh thái / Địa hình | F1-score |
|---|:---:|
| **Rừng rậm (Forest)** | **90.0** |
| **Đô thị (Urban)** | **88.0** |
| **Đồng cỏ / Đất trống (Grassland)** | **94.0** |

### 3.3. Đánh giá ngoại suy trên tập dữ liệu mở NAIP (Cheng et al., Nature Comms 2024)
- Sai số tuyệt đối trung bình (MAE số lượng cây/ha): **6.1** cây/ha (so với tác giả gốc là 2.27).
- Độ lệch ước lượng (Underestimation bias): **9%** (so với 3.61%). Mô hình giữ được độ chính xác cạnh tranh cao dù tập NAIP gán nhãn cả các cây chỉ có kích thước 2–3 pixel.

---

## 4. Phân tích Điểm yếu & Lỗ hổng (Failure Modes)

1. **Sự nhầm lẫn giữa bóng đổ và khung xương cây (Shadow Ambiguity):** Bóng đổ của cành cây khô trên mặt đường hoặc mái nhà có hình thái và màu xám tương tự như thân cây chết, dẫn đến việc mô hình segment nhầm bóng râm thành cây.
2. **Sự bất đồng giữa các chuyên gia gán nhãn:** Độ trùng lặp giữa các annotator người thật chỉ đạt **40%**, cho thấy cây suy thoái/chết đứng có ranh giới vô cùng chủ quan và khó phân định rõ ràng trên ảnh viễn thám.

---

## 5. Giá trị Tham khảo cho TreeFlowNet & Bài báo IGARSS 2027

> **Đã sửa 02/10/2026:** mục này từng chứa các khẳng định ưu việt của TreeFlowNet không có bằng chứng. Các số của bài gốc ở mục 1–4 đã được đối chiếu với toàn văn; mục 5 chỉ nêu điều bài cho thấy, tách khỏi giả thuyết. (Số đo của chính dự án: trên test CanopyRS, seed 42–44, Mask R-CNN R50 34.82 ± 0.67 AP, TreeFlowNet C5 R50 28.81 AP (một seed), TreeFlowNet Swin-T khoảng 30 AP; xem `docs/KIEN_TRUC_TREEFLOWNET_CLEAN_TOAN_HOC_VA_DIEM_YEU.md` mục 5 và 9. Các giả thuyết dưới đây chưa được kiểm chứng trên dữ liệu của mình.)

- **Focal γ < 1:** bài thấy γ < 1 cho kết quả tốt hơn trong thí nghiệm của họ (tập hazard tree, nhãn nhiễu). Việc suy ra rằng điều này giúp với nhãn thiếu của QuebecTrees là **giả thuyết**, chưa thử. Lưu ý TreeFlowNet chỉ dùng focal ở đầu centroid (C0–C4), còn C5 không có đầu này.
- **Encoder nền tảng:** DINOv2 đạt F1 91 so với ResNet-50 82 ở ảnh 7–15 cm và 65 so với 62 ở ảnh 30 cm. Thử backbone nền tảng cho TreeFlowNet là hướng hợp lý; Swin-T đã cho +1.37 AP so với R50 ở C5, DINOv2/SAM chưa thử.
- **Chuẩn hoá phân vị 1%–99%** cho ảnh vệ tinh 16-bit là kỹ thuật tiền xử lý cần nếu mở rộng sang dữ liệu đó.
