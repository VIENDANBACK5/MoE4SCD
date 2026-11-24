# Individual Tree Crown Delineation Based on Deep Learning for Arid Areas Using High-Resolution Satellite Imagery

- **Tên bài báo:** INDIVIDUAL TREE CROWN DELINEATION BASED ON DEEP LEARNING FOR ARID AREAS USING HIGH-RESOLUTION SATELLITE IMAGERY
- **Tác giả:** Jinzhuang Shi (1,2), Hui Li (1,3,5*), Yuanyuan Lin (4), Linhai Jing (1,3*), Kongwen Zhang (6)
- **Đơn vị:** (1) Aerospace Information Research Institute, CAS, Beijing; (2) Xinjiang University; (3) CBAS, Beijing; (4) China University of Geosciences; (5) Hainan Aerospace Information Research Institute; (6) University of the Fraser Valley, Canada
- **Hội nghị:** IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2024, pp. 5231-5234)
- **DOI:** [10.1109/IGARSS53475.2024.10641289](https://doi.org/10.1109/IGARSS53475.2024.10641289)
- **Phân loại phương pháp:** Two-stage Pipeline (Mask R-CNN Instance Delineation $\to$ CNN Patch-based Species Classification)

---

## 1. Bài toán & Bối cảnh Nghiên cứu

- **Nhiệm vụ (Task):** Phân đoạn ranh giới cá thể cây (ITC delineation) và phân loại loài cây (species classification) tại vùng sinh thái khô hạn đặc thù nhằm phục vụ đánh giá sinh khối và giám sát chương trình chống sa mạc hóa.
- **Hệ sinh thái:** Vùng sa mạc khô hạn và bán khô hạn Mu Us Desert (khu vực Wushen Banner, Ordos, Nội Mông, Trung Quốc; độ cao 1300–1400m). Các loài cây chính: Liễu (Willow), Dương (Populus), Thông (Pine), Salix và Caragana.
- **Dữ liệu đầu vào & Độ phân giải:**
  - Cảnh vệ tinh **WorldView-3** chụp ngày 16/07/2023.
  - Ảnh đa phổ gốc (8 bands): Độ phân giải 1.24 m.
  - Ảnh toàn sắc (Panchromatic): Độ phân giải 0.31 m.
  - **Thuật toán Pan-sharpening:** Sử dụng phương pháp RMI (Reduce Misalignment Impact - Li et al., 2018) để hòa trộn dải panchromatic 0.31 m với 8 dải đa phổ, tạo ra ảnh đa phổ 8 kênh siêu nét **0.31 m/pixel**.
- **Quy mô tập mẫu:**
  - 5 vùng huấn luyện: 2,856 cây gán nhãn thủ công. Cắt thành các patch $64 \times 64$ pixels (độ chồng lấn 50%). Tăng cường dữ liệu (lật ngang, lật dọc, xoay 90°, 180°, 270°) tạo thành 15,875 patch nhãn.
  - 2 vùng kiểm thử độc lập (Test Region 1 & Test Region 2): 2,167 cây (chủ yếu là liễu và dương).

---

## 2. Chi tiết Phương pháp Phân đoạn (Segmentation Methodology)

Quy trình tách biệt thành 2 công đoạn nối tiếp (Two-stage decoupled framework):

```
       [ Ảnh WorldView-3 8-band (1.24m) + Panchromatic (0.31m) ]
                                   │
                      Hòa trộn Pan-sharpening (RMI)
                                   │
                     [ Ảnh đa phổ 8 băng tần 0.31m ]
                                   │
                Cắt lát mẫu 64x64 px (50% overlap) + Augmentation
                                   │
                                   ▼
        ┌────────────────────────────────────────────────────────┐
        │   GIAI ĐOẠN 1: Phân đoạn ranh giới cá thể (Mask R-CNN) │
        │   - Backbone: ResNet-50 (bài không nêu FPN)            │
        │   - Chi tiết Mask R-CNN: bài không mô tả               │
        │   - Phân chia: 70% Train, 30% Validation               │
        └──────────────────────────┬─────────────────────────────┘
                                   │
                     [ Đa giác ranh giới từng cây ]
                                   │
                                   ▼
        ┌────────────────────────────────────────────────────────┐
        │   GIAI ĐOẠN 2: Phân loại loài cây (Patch-level CNN)    │
        │   - Trích xuất ảnh crop từng tán cây theo mask/box     │
        │   - Thử nghiệm 3 kiến trúc: ResNet-18, GoogLeNet,      │
        │     DenseNet-40                                        │
        │   - Huấn luyện: 500 epochs, batch 64, Early Stopping   │
        └──────────────────────────┬─────────────────────────────┘
                                   │
                                   ▼
            [ Bản đồ hoàn chỉnh: Vị trí, Viền tán, Loài cây ]
```

### 2.1. Giai đoạn 1: Phân đoạn cá thể cây bằng Mask R-CNN
- **Kiến trúc:** Mask R-CNN với backbone **ResNet-50** (bài chỉ nêu ResNet-50 là backbone; **không nêu FPN** hay chi tiết khác).
- **Tiền xử lý & Cắt patch:** Sử dụng kích thước patch cực nhỏ **$64 \times 64$ pixels** với **50% overlap**. Bài **không nêu lý do** chọn 64×64 (phần giải thích trước đây về kích thước tán và nền cát là suy đoán, đã bỏ).
- **Huấn luyện:** Huấn luyện trên 15,875 patch đã tăng cường hình học. Tỉ lệ train:val = 7:3.

### 2.2. Giai đoạn 2: Phân loại loài cây (Species Classification)
- Từ kết quả mask của Giai đoạn 1, hệ thống cắt các ô vuông chứa từng tán cây cá thể.
- Huấn luyện 3 kiến trúc CNN kinh điển để so sánh:
  1. **ResNet-18:** Mạng dư sâu 18 tầng với các khối residual block chống suy biến gradient.
  2. **GoogLeNet (Inception-v1):** Khối Inception đa kích thước kernel tích hợp song song (1x1, 3x3, 5x5).
  3. **DenseNet-40:** Mạng kết nối dày đặc, nối tất cả feature map từ các tầng trước vào tầng sau.
- **Tham số huấn luyện:** Batch size = 64, tối đa 500 epochs, dừng sớm (early stopping) nếu sau 10 vòng độ chính xác không cải thiện quá 0.001.

---

## 3. Kết quả Thực nghiệm & Số liệu Định lượng

### 3.1. Hiệu năng phân đoạn cá thể của Mask R-CNN (ITC Delineation)

| Vùng kiểm thử | Precision (P) | Recall (R) | F1-score | Đặc điểm địa hình thực địa |
|---|:---:|:---:|:---:|---|
| **Region 1** | **0.426** (42.6%) | **0.846** (84.6%) | **0.675** (67.5%) | (bài không mô tả đặc điểm vùng) |
| **Region 2** | **0.789** (78.9%) | **0.999** (99.9%) | **0.882** (88.2%) | (bài không mô tả đặc điểm vùng) |

*Điều bài nêu:* recall 84.6% và 99.9% ở hai vùng; bài gọi Precision là "accuracy of ITC delineation" (42.6% và 78.9%) và ghi độ chính xác huấn luyện của Mask R-CNN là 35%. Bài **không giải thích** vì sao Precision ở Region 1 thấp; các giải thích trước đây (bụi cỏ khô, bóng cồn cát, RPN sinh nhiều anchor) là suy đoán và đã bỏ.

### 3.2. Hiệu năng phân loại loài cây của các mạng CNN (Giai đoạn 2)

| Mô hình phân loại | Độ chính xác Train | Test Region 1 (OA / Kappa) | Test Region 2 (OA / Kappa) |
|---|:---:|:---:|:---:|
| **DenseNet-40** | 98.55% | 95.18% / 89.24% | 94.20% / 88.38% |
| **GoogLeNet** | 97.89% | **96.01%** / **93.48%** | **96.39%** / **92.77%** |
| **ResNet-18** | **98.88%** | 91.89% / 84.78% | 92.55% / 85.13% |

*Điều bài nêu:* ResNet-18 có độ chính xác huấn luyện cao nhất (98.88%), GoogLeNet có độ chính xác kiểm thử cao nhất (OA 96.01% và 96.39%); mọi mô hình đều > 90%. Lưu ý bài tự mâu thuẫn: abstract viết ResNet-18 đạt độ chính xác phân loại cao nhất và dùng ResNet-18 cho Hình 5, còn kết luận viết GoogLeNet cao nhất khi kiểm thử. Bài không giải thích nguyên nhân (giải thích về khối Inception trước đây là suy đoán). Mẫu loài: liễu 828, dương 894, thông 1601; tổng 3.323 tán, sau tăng cường 16.615 mẫu.

---

## 4. Hạn chế

**Bài tự nêu:** không có mục hạn chế riêng; kết luận chỉ nói kiểm chứng tiềm năng phân loại loài ở vùng khô hạn.

**Quan sát từ số liệu (của người đọc, không phải của bài):**
- Precision phân đoạn ở Region 1 chỉ 42.6% (Region 2: 78.9%) và độ chính xác huấn luyện của Mask R-CNN chỉ 35%; bài không phân tích nguyên nhân.
- Pipeline hai giai đoạn: lỗi phân đoạn truyền sang phân loại (suy luận hợp lý, bài không đo).
- Số vùng kiểm thử rất ít (2 vùng, 2.167 cây), không có seed lặp.

## 5. Giá trị Tham khảo cho TreeFlowNet & Bài báo IGARSS 2027

> **Đã sửa 02/10/2026:** mục này từng chứa các khẳng định ưu việt của TreeFlowNet không có bằng chứng. Các số của bài gốc ở mục 1–4 đã được đối chiếu với toàn văn; mục 5 chỉ nêu điều bài cho thấy, tách khỏi giả thuyết. (Số đo của chính dự án: trên test CanopyRS, seed 42–44, Mask R-CNN R50 34.82 ± 0.67 AP, TreeFlowNet C5 R50 28.81 AP (một seed), TreeFlowNet Swin-T khoảng 30 AP; xem `docs/KIEN_TRUC_TREEFLOWNET_CLEAN_TOAN_HOC_VA_DIEM_YEU.md` mục 5 và 9. Các giả thuyết dưới đây chưa được kiểm chứng trên dữ liệu của mình.)

- **Hành vi của Mask R-CNN trong bài:** Recall cao (84.6%–99.9%) nhưng Precision thấp ở vùng thưa (42.6% ở Region 1).
- **Giả thuyết, chưa kiểm chứng:** TreeFlowNet có thể ít sinh dương tính giả hơn nhờ canopy gate. Chưa có thí nghiệm trên dữ liệu thưa như bài này; các số của mình đều trên rừng dày (QuebecTrees).
- **Pan-sharpening RMI:** chỉ liên quan nếu dùng ảnh vệ tinh 8 băng tần.
