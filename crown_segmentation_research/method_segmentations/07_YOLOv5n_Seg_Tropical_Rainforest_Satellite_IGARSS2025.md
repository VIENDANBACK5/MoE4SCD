# Individual Tree Crown Segmentation Based on YOLOv5n-Seg Using High-Resolution Satellite Images of Tropical Rainforests

- **Tên bài báo:** INDIVIDUAL TREE CROWN SEGMENTATION BASED ON YOLOV5N-SEG USING HIGH-RESOLUTION SATELLITE IMAGES OF TROPICAL RAINFORESTS
- **Tác giả:** Jinzhuang Shi (1), Hui Li (2*), Junmao Hua (1), Linhai Jing (3*), Kongwen Zhang (4)
- **Đơn vị:** (1) Xinjiang University; (2) Hainan Aerospace Information Research Institute, CAS; (3) China University of Geosciences, Beijing; (4) University of the Fraser Valley, Canada
- **Hội nghị:** IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2025, pp. 4181-4184)
- **DOI:** [10.1109/IGARSS55030.2025.11243801](https://doi.org/10.1109/IGARSS55030.2025.11243801)
- **Phân loại phương pháp:** Lightweight Single-stage Instance Segmentation (YOLOv5n-seg) vs Heavyweight Two-stage Instance Segmentation (Mask R-CNN)

---

## 1. Bài toán & Bối cảnh Nghiên cứu

- **Nhiệm vụ (Task):** Phân đoạn cá thể tán cây (ITC instance segmentation) trong rừng mưa nhiệt đới rậm rạp từ ảnh vệ tinh quang học độ phân giải cao, tập trung giải quyết bài toán cân bằng giữa độ chính xác và chi phí tài nguyên tính toán (model complexity & inference latency).
- **Hệ sinh thái:** Rừng mưa nhiệt đới thứ sinh đất thấp và vùng núi thuộc Công viên Quốc gia Rừng mưa nhiệt đới Hải Nam (khu vực núi Wuzhishan, tỉnh Hải Nam, Trung Quốc; tọa độ 18°38'–19°02'N, 109°19'–109°44'E). Các loài ưu thế: Dacrydium pierrei, Syzygium araiocladum, Heritiera parvifolia, Mận xanh và Amesiodendron chinense.
- **Đặc trưng hệ sinh thái:** Rừng tán kín đa tầng, mật độ cực dày, độ che phủ tán lớn, bóng đổ phức tạp và tán cây lồng vào nhau nghiêm trọng.
- **Dữ liệu đầu vào & Độ phân giải:**
  - Ảnh vệ tinh quang học RGB thu thập từ nền tảng **Google Earth**.
  - Độ phân giải không gian: **~0.5 m (50 cm/pixel)**.
  - Thời gian ghi hình: 06/02/2021.
- **Quy mô tập dữ liệu & Gán nhãn:**
  - 9 ảnh phụ (subimages) chứa tổng cộng **~12,500 đa giác tán cây** được gán nhãn thủ công bằng phần mềm mã nguồn mở **LabelMe**.
  - Cắt thành các patch kích thước $640 \times 640$ pixels.
  - Tăng cường dữ liệu (lật ngang, xoay 90°, 180°) tạo thành 704 patch hoàn chỉnh:
    - **Tập Train:** 563 patch (từ 8 subimages).
    - **Tập Test:** 136 patch (từ 1 subimage độc lập).

---

## 2. Chi tiết Phương pháp Phân đoạn (Segmentation Methodology)

Nghiên cứu đối đầu trực tiếp giữa mô hình đơn tầng siêu nhẹ (**YOLOv5n-seg**) và mô hình 2 tầng chuẩn mực (**Mask R-CNN**).

```
                      [ Ảnh vệ tinh Google Earth RGB 0.5m ]
                                       │
                      Cắt lát patch 640x640 px + Augmentation
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │                 KIẾN TRÚC YOLOV5N-SEG                       │
        │                                                             │
        │ 1. BACKBONE (Trích xuất đặc trưng):                         │
        │    - Focus Module (lát cắt không gian giảm tải tính toán)    │
        │    - CSPNet (Cross Stage Partial Networks)                  │
        │    - SPP (Spatial Pyramid Pooling đa quy mô)                │
        │                                                             │
        │ 2. NECK (Trộn đặc trưng đa mức):                            │
        │    - FPN (Top-down) kết hợp PANet (Bottom-up)               │
        │                                                             │
        │ 3. SEGMENTATION HEAD:                                       │
        │    - Nhánh Detection: BBox + Class + Objectness Score       │
        │    - Nhánh Prototyping: Sinh k prototype masks              │
        │    - Nhánh Mask Coefficients: Dự đoán trọng số tuyến tính   │
        │      Mask = Sigmoid(Prototypes x Coefficients)              │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                 Huấn luyện: Batch 16, lr = 0.01, 400 Epochs
                                       │
                                       ▼
             [ Dự đoán mặt nạ cá thể từng cây mượt mà, siêu nhẹ ]
```

### 2.1. Cấu trúc Mô hình YOLOv5n-seg (Nano Version)
Mô hình kế thừa nguyên lý phân đoạn một giai đoạn (single-stage instance segmentation kiểu YOLACT):
1. **Backbone:**
   - **Focus Module:** Cắt lát không gian (space-to-depth slice) đưa ảnh $640 \times 640 \times 3$ thành $320 \times 320 \times 12$ trước khi qua lớp tích chập đầu tiên, giảm triệt để số lượng phép tính FLOPs mà không làm mất thông tin điểm ảnh.
   - **CSPNet:** Tách dòng gradient thành hai nhánh (một nhánh qua khối bottleneck dư, một nhánh nối tắt) giúp tăng tốc độ truyền đạo hàm và tiết kiệm bộ nhớ GPU.
   - **SPP (Spatial Pyramid Pooling):** Kết hợp các phép pooling kích thước $5 \times 5, 9 \times 9, 13 \times 13$ để mở rộng trường tiếp nhận (receptive field), nắm bắt ngữ cảnh tán rừng lớn.
2. **Neck:** Cấu trúc kết hợp kim tự tháp hai chiều **FPN + PANet**, cho phép đặc trưng ngữ nghĩa mức cao truyền từ trên xuống (top-down) và đặc trưng vị trí không gian chính xác truyền từ dưới lên (bottom-up).
3. **Head (Instance Segmentation Module):**
   - Không dùng RoIAlign cắt đặc trưng như Mask R-CNN.
   - Đầu Proto sinh ra một tập hợp 32 mặt nạ nguyên mẫu (*prototype masks*) trên toàn bộ ảnh.
   - Với mỗi bounding box được phát hiện, đầu Head dự đoán một vector hệ số mặt nạ (*mask coefficients*).
   - Mặt nạ cuối cùng của từng cây được tạo ra bằng phép nhân ma trận giữa hệ số và các prototype, sau đó cắt xén theo bounding box.

### 2.2. Mô hình Đối chứng: Mask R-CNN
- Two-stage framework chuẩn mực: Backbone ResNet-50 + FPN, đề xuất vùng RoI bằng RPN, trích xuất đặc trưng vùng qua RoIAlign kích thước $14 \times 14$, và nhánh Mask Head gồm 4 lớp Conv $3 \times 3$ sinh ra mask nhị phân $28 \times 28$.

### 2.3. Thiết lập Huấn luyện (Training Setup)
- Cả hai mô hình đều được huấn luyện trong cùng điều kiện thực nghiệm:
  - Batch size: 16.
  - Learning rate ban đầu: $0.01$.
  - Số lượng epoch: **400 epochs**.
  - Phần cứng: GPU NVIDIA.

---

## 3. Kết quả Thực nghiệm & Số liệu Định lượng So sánh

### 3.1. Bảng so sánh toàn diện giữa Mask R-CNN và YOLOv5n-seg

| Mô hình | Precision (%) | Recall (%) | F1-score (%) | mAP@50 (%) | mAP@50-95 (%) | Số tham số (Params) | Độ phức tạp (GFLOPS) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Mask R-CNN (ResNet-50)** | 86.1 | 77.5 | 81.6 | 85.2 | 45.8 | 43.2 M | 76.2 |
| **YOLOv5n-seg (Đề xuất)** | **87.0** | **79.3** | **83.2** | **90.3** | **57.6** | **2.0 M** | **7.1** |
| **Mức độ chênh lệch** | *+0.9%* | *+1.8%* | *+1.6%* | **+5.1%** | **+11.8%** | **Nhẹ hơn 21.6 lần** | **Nhanh hơn 10.7 lần** |

### 3.2. Phân tích số liệu mAP
- Ở ngưỡng dễ **mAP@50**: YOLOv5n-seg vượt Mask R-CNN 5.1 điểm (90.3% vs 85.2%).
- Ở dải ngưỡng khắt khe **mAP@50-95**: YOLOv5n-seg áp đảo hoàn toàn với mức chênh lệch lên tới **+11.8 điểm** (57.6% so với 45.8%). Điều này chứng minh chất lượng đường bao đa giác của YOLOv5n-seg khít với tán cây thực tế hơn nhiều so với việc nội suy từ mask phân giải thấp $28 \times 28$ của Mask R-CNN.

---

## 4. Phân tích Điểm yếu & Lỗ hổng (Failure Modes)

1. **Phụ thuộc vào kích thước Bounding Box trong nhánh Prototype Mask:** *Suy luận của người đọc, không phải kết luận của bài:* mask YOLO-seg được cắt theo bounding box nên hai tán chồng nhau có thể bị lẹm sang nhau; bài không phân tích điều này.
2. **Kích thước mẫu ảnh kiểm thử nhỏ:** Tập test chỉ gồm 136 patch (sau tăng cường) từ 1 trong 9 subimage (8 subimage cho huấn luyện, khoảng 12.500 tán tổng) trong cùng một cảnh Google Earth ngày 06/02/2021. Mô hình chưa được kiểm thử ngoại suy (cross-scene / cross-sensor evaluation) sang các mùa khác hoặc các khu rừng khác ngoài đảo Hải Nam.

---

## 5. Giá trị Tham khảo cho TreeFlowNet & Bài báo IGARSS 2027

> **Đã sửa 02/10/2026:** mục này từng chứa các khẳng định ưu việt của TreeFlowNet không có bằng chứng. Các số của bài gốc ở mục 1–4 đã được đối chiếu với toàn văn; mục 5 chỉ nêu điều bài cho thấy, tách khỏi giả thuyết. (Số đo của chính dự án: trên test CanopyRS, seed 42–44, Mask R-CNN R50 34.82 ± 0.67 AP, TreeFlowNet C5 R50 28.81 AP (một seed), TreeFlowNet Swin-T khoảng 30 AP; xem `docs/KIEN_TRUC_TREEFLOWNET_CLEAN_TOAN_HOC_VA_DIEM_YEU.md` mục 5 và 9. Các giả thuyết dưới đây chưa được kiểm chứng trên dữ liệu của mình.)

- **Số liệu của bài:** YOLOv5n-seg mAP50 90.3% và mAP50-95 57.6% so với Mask R-CNN 85.2% và 45.8%; 2.0M so với 43.2M tham số (nhẹ hơn 21.6 lần); 7.1 so với 76.2 GFLOPs. Chỉ trên 136 patch cục bộ một khu vực, nên không suy ra được cho dữ liệu khác.
- **Giải thích của mình (không phải của bài):** mask 28×28 của Mask R-CNN có thể làm mờ biên tán lớn. Mình chưa đo chất lượng biên của Mask R-CNN theo kích thước; số đo hiện có cho thấy Mask R-CNN trên CanopyRS mạnh hơn TreeFlowNet (34.82 so với 28.81 AP ở R50) nên giả thuyết này chưa giải thích được dữ liệu của mình.
- **Vị thế của TreeFlowNet:** không có bằng chứng để nói TreeFlowNet vượt Mask R-CNN hay YOLO-seg ở rừng có độ che phủ > 80%. Hiện nó kém Mask R-CNN ở test.
