# Individual Tree Crown Based Tree Species Classification from Very High-Resolution UAV Images

- **Tên bài báo:** INDIVIDUAL TREE CROWN BASED TREE SPECIES CLASSIFICATION FROM VERY HIGH-RESOLUTION UAV IMAGES
- **Tác giả:** Mareike Weishaupt (1), Wen Fan (2), Jonas Troles (3), Jiaojiao Tian (1,2)
- **Đơn vị:** (1) Remote Sensing Technology Institute (IMF), DLR, Germany; (2) Technical University of Munich (TUM), Germany; (3) University of Bamberg, Germany
- **Hội nghị:** IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2025, Brisbane, pp. 7328-7332)
- **DOI:** [10.1109/IGARSS55030.2025.11243070](https://doi.org/10.1109/IGARSS55030.2025.11243070)
- **Phân loại phương pháp:** Object-level Consistency Loss (ITC Loss) kết hợp Geometry-guided Post-Processing (Majority Voting inside ITC Polygons)

---

## 1. Bài toán & Bối cảnh Nghiên cứu

- **Nhiệm vụ (Task):** Phân loại loài cây (Tree Species - TS classification) ở cấp độ từng cá thể tán cây (ITC) từ ảnh UAV siêu phân giải trong rừng hỗn giao ôn đới dày đặc.
- **Hệ sinh thái:** Rừng tự nhiên tại Bamberg, Đức (BAMFORESTS dataset, diện tích 105 ha).
- **Dữ liệu đầu vào & Độ phân giải:**
  - Ảnh trực giao UAV RGB siêu phân giải: **GSD ~2 cm/pixel**.
  - Dữ liệu vector đa giác ranh giới từng cây riêng lẻ (ITC shapefiles/JSON).
- **Quy mô dữ liệu & Phân chia (Split):**
  - Tổng số cây có nhãn: **27,160 cá thể cây**.
  - 10 lớp phân loại: Thông (Pine 36.2%), Dẻ gai (Beech 23.1%), Sồi (Oak 23.3%), Vân sam (Spruce 5.5%), Larch (2.7%), Douglas Fir (1.1%), Fir (1.2%), Cây khác (6.8%), Cây chết (1.8%), Nền đất (Background).
  - Phân chia:
    - **Train:** 17,212 cây (Stadtwald, Tretzendorf 1 & 2).
    - **Val:** 4,390 cây.
    - **Test-2 (In-distribution):** 3,580 cây (cùng vùng rừng với tập Train).
    - **Test-1 (Out-of-distribution / Unseen):** 1,978 cây tại công viên Hain (chủ yếu là rừng lá rộng rụng lá, phân phối loài khác hoàn toàn với tập Train).

---

## 2. Chi tiết Phương pháp Phân đoạn (Segmentation Methodology)

Vấn đề cốt lõi của phân đoạn ngữ nghĩa (Semantic Segmentation như U-Net) là: mô hình dự đoán độc lập từng pixel. Khi tán cây đan cài vào nhau hoặc cành lá không đồng nhất, trên cùng một cây mô hình lại dự đoán lốm đốm 2-3 loài khác nhau, làm rách nát hình thái tán cây.

Nhóm tác giả đề xuất hai giải pháp hiệp đồng: **ITC Loss** (can thiệp trong quá trình train) và **ITC Post-processing** (can thiệp sau khi dự đoán).

```
  [ Ảnh UAV RGB 512x512 ]                [ Vector Ranh giới Tán cây ITC (GT) ]
            │                                             │
            ▼                                             │
      ┌───────────┐                                       │
      │   U-Net   │                                       │
      └─────┬─────┘                                       │
            │ Dự đoán pixel-level                         │
            ▼                                             │
     [ Softmax Map ] ◄────────────────────────────────────┘
            │               │
            │               ▼
            │     [ Phép nhân Mask ITC & Bỏ phiếu Đa số (Majority Voting) ]
            │     - Nhóm các pixel thuộc cùng 1 polygon cây
            │     - Gán toàn bộ cây theo loài chiếm số pixel nhiều nhất
            │               │
            │               ▼
            │     [ Bản đồ đồng nhất cấp cây (Voting Output) ]
            │               │
            ▼               ▼
      [ Cross-Entropy ] [ ITC Loss: Tỉ lệ pixel dự đoán sai / Tổng pixel ]
            │               │
            └───────┬───────┘
                    ▼
     Tổng Loss: L = alpha * L_CE + beta * L_ITC
                    │
            Backpropagation
                    │
                    ▼
     [ Suy luận trên Patch 2048x2048 ]
                    │
                    ▼
     [ ITC Postprocessing: Ghép mask ITC dự đoán & Bầu chọn đa số ]
                    │
                    ▼
     [ Bản đồ phân đoạn loài cây đồng nhất, sạch nhiễu viền ]
```

### 2.1. Chiến lược Tiling & Cắt mảnh đa quy mô
- **Giai đoạn huấn luyện ngữ nghĩa:** Cắt patch kích thước **$512 \times 512$ pixels** với độ chồng lấn **50% overlap**. Patch nhỏ giúp mô hình tập trung học cấu trúc vân lá, kết cấu vỏ cây và đặc trưng quang phổ chi tiết.
- **Giai đoạn phân đoạn hình học (ITC delineation):** Thực hiện tốt nhất trên patch kích thước lớn **$2048 \times 2048$ pixels** để bao quát trọn vẹn toàn bộ chu vi tán cây mà không bị cắt cụt ở biên cạnh. Các patch nhỏ 512x512 sau khi dự đoán được ghép lại thành khổ 2048x2048 trước khi hậu xử lý.

### 2.2. Hàm mất mát cấp độ cá thể: ITC Loss Function
Để ép mô hình phải tôn trọng tính toàn vẹn của một cá thể cây, tác giả nhân ma trận dự đoán với đa giác nhãn ITC và áp dụng cơ chế bỏ phiếu đa số (majority voting). Sau đó, tính tỷ lệ sai lệch trên toàn bộ cây:

$$\mathcal{L}_{ITC} = \frac{\text{Số pixel dự đoán sai trên các cây}}{\text{Tổng số pixel dự đoán}}$$

Hàm mất mát tổng hợp tích hợp vào lan truyền ngược:

$$\mathcal{L}_{total} = \alpha \mathcal{L}_{CE} + \beta \mathcal{L}_{ITC}$$

- $\mathcal{L}_{CE}$: Categorical Cross-Entropy Loss truyền thống ở mức pixel.
- $\mathcal{L}_{ITC}$: Thúc đẩy tính nhất quán nội tại trong từng tán cây, loại bỏ hiện tượng phân mảnh loài trên cùng một tán.

### 2.3. Hậu xử lý dựa trên hình học tán cây (ITC-Based Postprocessing)
- Sử dụng kết quả phân đoạn hình học ITC (dạng đa giác nhãn hoặc từ mô hình phân đoạn biên dạng như Mask R-CNN/Tian et al., 2025).
- Với mỗi đa giác tán cây riêng biệt: Thống kê số lượng pixel của từng loài bên trong đa giác đó. Loài nào có số lượng pixel áp đảo (majority voting) sẽ được gán cho 100% diện tích của cây đó.
- Kết quả: Xóa bỏ hoàn toàn nhiễu lấm chấm (salt-and-pepper noise), tạo ra các đường viền tán cây sắc nét, hoàn hảo về mặt sinh thái học.

### 2.4. Cấu hình Huấn luyện
- Mạng cơ sở: U-Net chuẩn.
- Optimizer: **AdamW** với weight decay $0.011$.
- Batch size: 10.
- Số epoch: 30 epochs (có cơ chế Early Stopping chống overfitting).

---

## 3. Kết quả Thực nghiệm & Số liệu Định lượng (mIoU %)

| Cấu hình mô hình | Test-2 (In-distribution) | Test-1 (OOD Hain Park) |
|---|:---:|:---:|
| **U-Net Baseline (Chỉ dùng CE Loss)** | 42.03% | 15.46% |
| **U-Net + ITC Loss ($\mathcal{L}_{CE} + \mathcal{L}_{ITC}$)** | 42.79% (+0.76%) | 17.33% (+1.87%) |
| **U-Net + ITC Loss + ITC Postprocessing** | **44.39%** (**+2.36%**) | **20.93%** (**+5.47%**) |

### Chi tiết trên các loài cây chính (Test-2):
- **Dẻ gai (Beech):** 32.83% (Baseline) $\to$ **38.58%** (Đề xuất) — Tăng **+5.75% mIoU**.
- **Sồi (Oak):** 33.24% (Baseline) $\to$ **37.62%** (Đề xuất) — Tăng **+4.38% mIoU**.
- **Thông (Pine):** 50.22% (Baseline) $\to$ **50.96%** (Đề xuất).

---

## 4. Phân tích Điểm yếu & Lỗ hổng (Failure Modes)

1. **Sự sụp đổ ở các loài thiểu số (Minority Class Collapse):** Các loài có tỷ lệ mẫu $< 5\%$ trong tập dữ liệu (như Fir 1.1%, Douglas Fir 1.2%) có mIoU gần như bằng 0% (0.00% – 0.27%). (Việc majority voting "nuốt chửng" pixel thiểu số là suy đoán của người đọc, bài không chứng minh.)
2. **Sự phụ thuộc vào chất lượng đa giác hình học:** Nếu mô hình phân đoạn hình thái cây ban đầu gộp nhầm 2 cây khác loài vào làm 1 polygon, majority voting sẽ gán cả polygon về một loài (suy luận của người đọc, không có trong bài).
3. **Độ lệch miền dữ liệu (Domain Gap):** Trên tập kiểm thử Test-1 (rừng công viên khác loài), mIoU cơ sở tụt dốc thảm hại từ 42% xuống 15%, cho thấy mô hình phụ thuộc rất lớn vào phân phối loài cục bộ của vùng bay chụp.

---

## 5. Giá trị Tham khảo cho TreeFlowNet & Bài báo IGARSS 2027

> **Đã sửa 02/10/2026:** mục này từng chứa các khẳng định ưu việt của TreeFlowNet không có bằng chứng. Các số của bài gốc ở mục 1–4 đã được đối chiếu với toàn văn; mục 5 chỉ nêu điều bài cho thấy, tách khỏi giả thuyết. (Số đo của chính dự án: trên test CanopyRS, seed 42–44, Mask R-CNN R50 34.82 ± 0.67 AP, TreeFlowNet C5 R50 28.81 AP (một seed), TreeFlowNet Swin-T khoảng 30 AP; xem `docs/KIEN_TRUC_TREEFLOWNET_CLEAN_TOAN_HOC_VA_DIEM_YEU.md` mục 5 và 9. Các giả thuyết dưới đây chưa được kiểm chứng trên dữ liệu của mình.)

- **Loss mức cá thể:** ý tưởng "ITC loss" của bài có thể gợi ý một số hạng phạt cấp crown cho TreeFlowNet. Đây là **ý tưởng chưa thử**; tài liệu kiến trúc đã nêu loss của TreeFlowNet thuần cục bộ theo pixel và chưa có số hạng cấp cây.
- **Kích thước patch:** bài thấy 512×512 tốt cho phân đoạn ngữ nghĩa và 2048×2048 tốt cho phân đoạn ITC với bài toán loài cây. Điều này **không** chứng minh crop 1024 là "cân bằng lý tưởng" cho TreeFlowNet: thí nghiệm của mình cho thấy target margin 256 px không cải thiện (+0.04 AP) và chưa so các kích thước crop khác.
- **Majority voting** trên mask là cách hợp lý để mở rộng sang loài cây, nhưng chưa thử.
