# Deep Learning-Based Individual Tree Segmentation Based on Airborne LiDAR and Very High-Resolution Aerial Images

- **Tên bài báo:** DEEP LEARNING-BASED INDIVIDUAL TREE SEGMENTATION BASED ON AIRBORNE LIDAR AND VERY HIGH-RESOLUTION AERIAL IMAGES
- **Tác giả:** Yingtan Li, Zhanmang Liao*, Jie Wang, Yiru Zhang, Jialin Hou, Lei Wu, Rui Chen
- **Đơn vị:** University of Electronic Science and Technology of China (UESTC) & China West Normal University
- **Hội nghị:** IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2025)
- **DOI:** [10.1109/IGARSS55030.2025.11242445](https://doi.org/10.1109/IGARSS55030.2025.11242445)
- **Phân loại phương pháp:** Weakly-supervised / Pseudo-label Generation (LiDAR Point Cloud DBSCAN + Concave Hull) kết hợp Single-stage Instance Segmentation (YOLOv11-seg)

---

## 1. Bài toán & Bối cảnh Nghiên cứu

- **Nhiệm vụ (Task):** Phân đoạn cá thể tán cây (Individual tree instance segmentation) trên ảnh quang học hàng không độ phân giải cao (VHR) bằng cách tự động sinh nhãn huấn luyện (pseudo-labels) từ dữ liệu LiDAR trên không mà không cần vẽ tay thủ công.
- **Hệ sinh thái:** Rừng tự nhiên tại California, Hoa Kỳ (Khu vực Carr Hirz Delta Fires 2019, diện tích kiểm thử không kiểm chứng được trong văn bản bài (README cũ ghi 13 km²)).
- **Dữ liệu đầu vào:**
  - **LiDAR trên không (USGS NGP Foundation LiDAR Spec 1.3):** Độ phân giải 0.5 m, NAD 1983 2011 UTM Zone 10N, thu thập thu-đông 2019 (17 tiles).
  - **Ảnh hàng không NAIP RGB:** Độ phân giải không gian **0.6 m (60 cm)**, chụp năm 2020 trong mùa sinh trưởng thực vật, tải qua Google Earth Engine (GEE).
- **Thách thức cốt lõi:**
  - Nhãn thủ công tốn kém và không khả thi trên diện rộng.
  - Sử dụng thuật toán truyền thống (Watershed) trên CHM sinh nhãn huấn luyện bị lỗi chẻ nhỏ tán cây (over-segmentation) nghiêm trọng, làm mô hình Deep Learning học sai quy luật.

---

## 2. Chi tiết Phương pháp Phân đoạn (Segmentation Methodology)

Quy trình gồm 2 giai đoạn: (1) Xây dựng thuật toán sinh nhãn giả sạch từ LiDAR point cloud, và (2) Huấn luyện mô hình instance segmentation YOLOv11m-seg trên ảnh quang học NAIP.

```
                      [ Dữ liệu LiDAR Point Cloud (0.5m) ]
                                      │
               Thuật toán phát triển vùng (Region Growing - Li et al.)
                                      │
                        Các cụm điểm Tree ID sơ bộ
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
    [ Lọc nhiễu DBSCAN ]                             [ Lọc ngoại vi 95% ]
  Loại bỏ các điểm cô lập                       Loại bỏ điểm cách tâm > 95%
  nằm ngoài tán cây chính                       bán kính cụm (tán chạm nhau)
              └───────────────────────┬───────────────────────┘
                                      ▼
                        [ Đa giác Bao lõm Concave Hull ]
                       Tạo viền mặt cắt thẳng đứng chính xác
                                      ▼
                 [ Bộ nhãn giả chất lượng cao (F1 = 0.87) ]
                                      │
                                      ▼
      [ Cắt ảnh NAIP RGB 0.6m thành Patch 256x256 + Data Augmentation ]
                                      │
                                      ▼
               [ Huấn luyện YOLOv11m-seg (Single-stage Instance Seg) ]
                  - Backbone: C3k2 + SPPF
                  - Neck: FPN + PANet
                  - Segment Head: Prototype Masks + Mask Coefficients
                                      ▼
                [ Dự đoán Instance Mask trên ảnh quang học VHR ]
```

### 2.1. Giai đoạn 1: Sinh nhãn giả từ LiDAR (LiDAR Sample Generation)
Tác giả so sánh hai phương pháp sinh nhãn từ LiDAR:

#### Phương pháp A: Watershed trên mô hình chiều cao tán (CHM Watershed)
- Tạo CHM độ phân giải 0.5 m từ LiDAR.
- Lọc Gauss với $\sigma = 1.9$, chiều cao cây tối thiểu $3\,\text{m}$.
- Chạy thuật toán Watershed dựa trên gradient mức xám để tìm đường viền đóng kín.
- Loại bỏ các polygon có diện tích $< 2.5\,\text{m}^2$.
- *Nhược điểm:* Over-segmentation cực kỳ nặng (Precision chỉ 0.47).

#### Phương pháp B: Region Growing trên đám mây điểm + Lọc nhiễu DBSCAN + Concave Hull (Cải tiến của tác giả)
1. **Phân đoạn sơ bộ:** Dùng thuật toán phát triển vùng dựa trên đám mây điểm (Li et al., 2012) để gán Tree ID sơ bộ cho từng cụm điểm.
2. **Lọc nhiễu bằng DBSCAN:** Áp dụng DBSCAN cho từng Tree ID để cô lập và loại bỏ các điểm rời rạc nằm ngoài tán cây chính.
3. **Lọc tán lấn biên (Centroid Distance Filtering):** Tính toán tâm hình học (centroid) của cụm điểm dày đặc. Loại bỏ toàn bộ các điểm nằm ngoài bán kính 95% khoảng cách từ tâm, vì các điểm này thuộc về cành nhánh của cây lân cận đan cài vào.
4. **Tạo đa giác bằng Concave Hull (Bao lõm):** Thay vì dùng Convex Hull (bao lồi làm phồng tán và dính vào nhau), tác giả dùng **Concave Hull** để viền ôm sát các hốc lá và cấu trúc thực tế của tán cây theo mặt phẳng chiếu đứng, tạo thành file vector Polygon nhãn.

### 2.2. Giai đoạn 2: Huấn luyện Instance Segmentation (YOLOv11-seg)
- **Kiến trúc:** YOLOv11 bản trung bình (**YOLOv11m-seg**, phát hành 09/2024).
- **Phân chia Patch & Augmentation:**
  - Ảnh NAIP 0.6 m được cắt thành các patch kích thước nhỏ **$256 \times 256$ pixels** (tổng cộng 119 patch lớn).
  - Tăng cường dữ liệu (Data Augmentation): Thêm nhiễu Gauss, thay đổi độ sáng, tịnh tiến (translation), lật gương (mirroring), và biến đổi điểm ngẫu nhiên.
  - Phân chia: 70% Train, 20% Validation, 10% Test.
- **Môi trường thực thi:** Python 3.11.4, PyTorch, chạy trên GPU NVIDIA RTX 3090.

---

## 3. Kết quả Thực nghiệm & Số liệu Định lượng

### 3.1. Đánh giá chất lượng nhãn sinh từ LiDAR (So với GT vẽ tay chuyên gia)

| Phương pháp sinh nhãn từ LiDAR | Precision (P) | Recall (R) | F1-score (F) |
|---|:---:|:---:|:---:|
| **Watershed trên CHM (Truyền thống)** | 0.47 | **0.92** | 0.62 |
| **Point Cloud Region Growing + DBSCAN + Concave Hull (Đề xuất)** | **0.89** | 0.85 | **0.87** |

*Nhận xét:* Watershed có Recall cao (0.92) nhưng Precision rất thấp (0.47) do hiện tượng chẻ nhỏ 1 cây thành nhiều mảnh. Thuật toán đề xuất cải thiện Precision thêm **+42%**, đưa F1 lên 0.87.

### 3.2. Hiệu năng phân đoạn của YOLOv11m-seg trên ảnh quang học NAIP 0.6m

| Nguồn nhãn huấn luyện (Sample Source) | Precision (P) | Recall (R) | F1-score | mAP@50 |
|---|:---:|:---:|:---:|:---:|
| **Train bằng nhãn CHM Watershed** | 0.777 | 0.653 | 0.711 | 0.720 |
| **Train bằng nhãn Point Cloud DBSCAN (Đề xuất)** | **0.918** | **0.781** | **0.846** | **0.856** |

---

## 4. Phân tích Điểm yếu & Lỗ hổng (Failure Modes)

1. **Độ nhạy cực cao của Deep Learning với chất lượng nhãn:** Kết quả chứng minh nếu nhãn pseudo-label bị over-segmentation (Watershed), mô hình deep learning sẽ học sai và tụt mAP@50 từ 85.6% xuống 72.0% (giảm 13.6 điểm).
2. **Góc nghiêng quang sai máy ảnh (Camera Visual Angle Distortion):** Ảnh hàng không NAIP ở góc chụp nghiêng làm tán cây bị lệch so với hình chiếu thẳng đứng của LiDAR, dẫn đến nhãn bị trượt vài pixel so với bóng cây thực trên ảnh RGB.
3. **Lỗi hình hộp chữ nhật ở cây nhỏ (Small Crowns Artifact):** Với các tán cây quá nhỏ (chỉ chiếm vài pixel trên ảnh 0.6 m), YOLOv11-seg có xu hướng dự đoán ra các **mặt nạ hình chữ nhật vuông vức** thay vì vẽ được đường viền tự nhiên cong tròn của tán cây.

---

## 5. Giá trị Tham khảo cho TreeFlowNet & Bài báo IGARSS 2027

> **Đã sửa 02/10/2026:** mục này từng chứa các khẳng định ưu việt của TreeFlowNet không có bằng chứng. Các số của bài gốc ở mục 1–4 đã được đối chiếu với toàn văn; mục 5 chỉ nêu điều bài cho thấy, tách khỏi giả thuyết. (Số đo của chính dự án: trên test CanopyRS, seed 42–44, Mask R-CNN R50 34.82 ± 0.67 AP, TreeFlowNet C5 R50 28.81 AP (một seed), TreeFlowNet Swin-T khoảng 30 AP; xem `docs/KIEN_TRUC_TREEFLOWNET_CLEAN_TOAN_HOC_VA_DIEM_YEU.md` mục 5 và 9. Các giả thuyết dưới đây chưa được kiểm chứng trên dữ liệu của mình.)

- **Over-segmentation của Watershed trên CHM:** bài cho Watershed P = 0.47, R = 0.92, F = 0.62, so với region growing trên point cloud F = 0.87 (Bảng I). Đây là bằng chứng cho nhãn giả từ CHM watershed, không phải bằng chứng về TreeFlowNet.
- **YOLOv11-seg trên cây nhỏ:** bài ghi nhận mô hình xuất vùng hình chữ nhật thay vì biên chính xác ở cây rất nhỏ. Giả thuyết (chưa kiểm chứng): dự đoán dày đặc có thể giữ hình dạng tốt hơn. Số đo của mình cho thấy TreeFlowNet có recall thấp hơn Mask R-CNN ở crown dưới 2 m², nên chưa có cơ sở nói nó tốt hơn ở lớp này.
- **Concave Hull** thay bounding box là bài học hợp lý cho việc sinh nhãn đa giác.
- Lưu ý: abstract bài ghi mAP@50 = 0.836, Bảng II ghi 0.856 cho nguồn nhãn region growing; hai số trong cùng bài không khớp.
