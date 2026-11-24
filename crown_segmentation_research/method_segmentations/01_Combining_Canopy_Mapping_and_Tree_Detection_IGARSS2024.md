# Combining Canopy Mapping and Tree Detection for Urban Tree Heritage Monitoring from VHR Images

- **Tên bài báo:** COMBINING CANOPY MAPPING AND TREE DETECTION FOR URBAN TREE HERITAGE MONITORING FROM VHR IMAGES
- **Tác giả:** R. Budin, B. Beguet, C. Rozo, N. Debonnaire, V. Lafon
- **Đơn vị:** i-Sea, France (hợp tác với Bordeaux Métropole, Biogeco, IMB)
- **Hội nghị:** IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2024)
- **DOI:** [10.1109/IGARSS53475.2024.10642956](https://doi.org/10.1109/IGARSS53475.2024.10642956)
- **Phân loại phương pháp:** Hybrid (Semantic Canopy Segmentation + Bounding-box Object Detection + Geometric Heuristic Fusion)

---

## 1. Bài toán & Bối cảnh Nghiên cứu

- **Nhiệm vụ (Task):** Giám sát di sản cây xanh đô thị (Urban tree heritage monitoring) trên quy mô diện rộng (~580 km² thuộc vùng đô thị Bordeaux, Pháp). Bài toán bao gồm hai mục tiêu song song:
  1. Lập bản đồ mảng tán rừng/cây dày đặc (canopy coverage) liên tục theo thời gian.
  2. Định vị và đếm từng cá thể cây riêng lẻ (đặc biệt là cây non mới trồng trong chương trình tái thiết rừng đô thị).
- **Hệ sinh thái:** Rừng đô thị (Urban forest), công viên, dải cây ven đường và các mảng rừng nhỏ xen kẽ công trình kiến trúc.
- **Dữ liệu đầu vào & Độ phân giải:**
  - **Ảnh vệ tinh đa thời gian Pléiades:** Độ phân giải không gian **0.5 m (50 cm)**, chụp trong khoảng thời gian 07/2021 – 02/2022 (3 mosaic).
  - **Ảnh hàng không RGB trực giao (Aerial photography):** Độ phân giải siêu cao **0.1 m (10 cm)**, thu nhận mùa hè 2020.
  - **Mô hình độ cao số (DEM/CHM) từ LiDAR:** Mật độ 15 điểm/m², độ phân giải raster nội suy **0.5 m (50 cm)**, thu nhận hè 2020.
- **Bộ nhãn tham chiếu:**
  - Nhãn phân loại không gian kẽ (interstitial spaces): ~15,000 polygon phân bố toàn đô thị.
  - Nhãn phát hiện cây đơn lẻ: ~50,000 bounding box vẽ tay trên 40 ROI; 8 ROI (~13,000 box) làm tập validation PITVD.
  - Tập kiểm chứng độc lập của chính quyền địa phương (`EC_ARBRES_P`): ~60,000 cây đơn lẻ đo đạc thực địa.

---

## 2. Chi tiết Phương pháp Phân đoạn (Segmentation Methodology)

Tác giả chỉ ra giới hạn mang tính hệ thống của hai trường phái độc lập:
- *Canopy mapping (pixel-level)* tốt trên mảng rừng lớn nhưng bỏ sót hoặc làm mờ cây đơn lẻ/cây non.
- *Tree detection (bounding box deep learning)* phát hiện tốt cây đơn lẻ ngoài đường phố nhưng sụp đổ hoàn toàn trong mảng rừng dày (không tách được tán cây lồng vào nhau).

Do đó, nhóm tác giả đề xuất một pipeline kết hợp hai pha (2-stage hybrid fusion pipeline).

```
                             [ Ảnh vệ tinh Pléiades 50cm ]
                                          │
                         Radiometric Indices & Stacking
                                          │
                      Random Forest (Chuỗi xử lý Biocoast)
                                          │
                      [ Mặt nạ Canopy (Woody Vegetation) ]
                                          │
    [ Ảnh Aerial RGB 10cm ]               │ 
              │                           │
  RetinaNet (DeepForest)                  │
              │                           │
      BBox cá thể cây                     │
      + LiDAR Height                      │
              │                           │
              └───────────────┬───────────┘
                              ▼
           [ Bộ lọc không gian & Quy tắc hợp nhất ]
            - Lọc bbox theo chiều cao & diện tích
            - Buffer (+3m/-2m) & diện tích >250m²
            - Loại bỏ bbox lọt hoàn toàn trong canopy
                              ▼
    [ Bản đồ tích hợp: Mảng tán rừng lớn + Từng cây riêng lẻ ]
```

### 2.1. Nhánh 1: Lập bản đồ tán rừng (Canopy Mapping via Biocoast)
- **Mô hình:** Random Forest Classifier (Breiman, 2001) trong chuỗi xử lý *Biocoast* của i-Sea.
- **Đặc trưng:** Tính toán các chỉ số quang sai (radiometric indices: NDVI, v.v.) từ ảnh Pléiades 50 cm đa thời kỳ, ghép chồng thành một tensor đa kênh (band stacking).
- **Hậu xử lý:** Hợp nhất tất cả các polygon thuộc lớp "Woody Vegetation" (thực vật thân gỗ).

### 2.2. Nhánh 2: Phát hiện cây cá thể (Individual Tree Detection via RetinaNet)
- **Mô hình:** Kế thừa framework *DeepForest* (Weinstein et al., 2019) dựa trên kiến trúc **RetinaNet** với **Focal Loss** (Lin et al., 2017).
- **Chiến lược Active Learning:** Huấn luyện lặp 3 vòng: Train $\to$ Dự đoán vùng chưa gán nhãn $\to$ Bổ sung mẫu khó $\to$ Retrain.
- **Trích xuất thông số sinh học từ LiDAR:**
  - **Chiều cao cây:** Lấy phân vị 90 (90th percentile) của giá trị độ cao các pixel nằm trong bounding box.
  - **Diện tích tán (Crown Area):** Tính tổng diện tích các pixel bên trong bounding box có độ cao lớn hơn chiều cao trung bình của box đó.

### 2.3. Quy tắc hợp nhất không gian (Spatial Fusion & Post-processing Rules)

#### (A) Xử lý mặt nạ Canopy:
1. Lọc polygon canopy giao cắt với bounding box cây nếu:
   - Diện tích polygon canopy < Diện tích tán trích xuất từ bounding box.
   - Diện tích polygon canopy < Tổng diện tích tán của tất cả các bounding box giao cắt với nó.
2. Áp dụng phép nở-co hình thái học (Morphological Buffering): **Buffer +3 m** rồi **Buffer -2 m** để làm mịn viền răng cưa quanh các khóm cây dày.
3. Loại bỏ các polygon canopy có diện tích $< 250\,\text{m}^2$ (loại bỏ nhiễu vụn không đại diện cho cụm rừng).

#### (B) Xử lý nhãn cây cá thể (Bounding Boxes):
Áp dụng 4 quy tắc lọc thực nghiệm loại bỏ False Positives và cụm tán bị gộp:
1. Loại bỏ cây có chiều cao trích xuất $> 30\,\text{m}$.
2. Loại bỏ cây có chiều cao trích xuất $< 0.5\,\text{m}$.
3. Loại bỏ cây có diện tích tán $> 500\,\text{m}^2$ (quá lớn so với 1 cá thể, thực chất là một cụm cây bị gộp).
4. Nếu cây nằm đè lên polygon canopy và diện tích tán của nó nhỏ hơn diện tích polygon canopy $\implies$ loại bỏ box (để polygon canopy đại diện cho khu vực đó, tránh trùng lặp).

---

## 3. Kết quả Thực nghiệm & Số liệu Định lượng

### 3.1. Kết quả phân loại Canopy (Random Forest trên Pléiades)
- Độ chính xác tổng thể (Overall Accuracy): **87.9%**.
- F-score lớp cây rụng lá (Deciduous): **0.76**.
- F-score lớp cây thường xanh (Evergreen): **0.82**.

### 3.2. Hiệu năng so sánh trên các tập kiểm chứng độc lập

| Phương pháp | Tập kiểm chứng nội bộ PITVD (~13k cây) | Tập thực địa toàn thành phố EC_ARBRES_P (~60k cây) |
|---|:---:|:---:|
| **Chỉ dùng Canopy Mapping** | 76% | 54% |
| **Chỉ dùng Tree Detection (DeepForest/RetinaNet)** | 84% | 58% |
| **Phương pháp kết hợp (Proposed Combination)** | **95%** | **85%** |

---

## 4. Phân tích Điểm yếu & Lỗ hổng (Failure Modes)

1. **Thất bại của Bounding Box trong rừng rậm:** RetinaNet/DeepForest đạt 84% trên cây ngoài đường phố nhưng tụt xuống **58%** trên tập cây rừng tự nhiên `EC_ARBRES_P`. Tác giả thừa nhận mô hình hoàn toàn bất lực trong việc tách các tán cây nằm sát hoặc đan cài vào nhau trong mảng rừng rậm (*incapable of exhaustively detaching crowns within dense stands*).
2. **Lệch pha thời gian (Temporal Discrepancy):** Ảnh Pléiades (2021-2022) và ảnh Aerial/LiDAR (2020) lệch nhau 2 năm, gây ra sai số do cây bị đốn hạ hoặc phát triển thêm.
3. **Mô hình hình học đơn giản:** Phương pháp chỉ dùng quy tắc ngưỡng cứng (hard-coded thresholds: 250 m², 500 m², chiều cao 0.5–30 m), không có cơ chế end-to-end học biểu diễn đường biên thực tế của tán cây.

---

## 5. Giá trị Tham khảo cho TreeFlowNet & Bài báo IGARSS 2027

> **Đã sửa 02/10/2026:** mục này từng chứa các khẳng định ưu việt của TreeFlowNet không có bằng chứng. Các số của bài gốc ở mục 1–4 đã được đối chiếu với toàn văn; mục 5 chỉ nêu điều bài cho thấy, tách khỏi giả thuyết. (Số đo của chính dự án: trên test CanopyRS, seed 42–44, Mask R-CNN R50 34.82 ± 0.67 AP, TreeFlowNet C5 R50 28.81 AP (một seed), TreeFlowNet Swin-T khoảng 30 AP; xem `docs/KIEN_TRUC_TREEFLOWNET_CLEAN_TOAN_HOC_VA_DIEM_YEU.md` mục 5 và 9. Các giả thuyết dưới đây chưa được kiểm chứng trên dữ liệu của mình.)

- **Điều bài báo này thực sự cho thấy:** detector hộp (RetinaNet/DeepForest) đạt P = 85% trên tập cây đô thị nhưng chỉ 58% trên tập cây độc lập EC_ARBRES_P; ghép với bản đồ tán (random forest) đưa P toàn thành phố từ 58% lên 85%. Bài chỉ nêu RetinaNet/DeepForest, không nêu Faster R-CNN.
- **Giả thuyết, chưa kiểm chứng:** đầu canopy kết hợp flow có thể giúp trong tán dày. Trong dữ liệu của mình, lỗi chính của TreeFlowNet ở crown nhỏ không phải là bounding box mà là crown chạm biên bị nuốt vào crown kề (khoảng một nửa crown nhỏ bị mất; Mask R-CNN có tỉ lệ tương đương), nên không thể nói TreeFlowNet đã "giải quyết triệt để" điểm nghẽn này.
- **Kỹ thuật hậu xử lý:** buffer (+3 m, −2 m) và lọc diện tích là bước vector hoá đáng tham khảo khi xuất bản đồ diện rộng.
