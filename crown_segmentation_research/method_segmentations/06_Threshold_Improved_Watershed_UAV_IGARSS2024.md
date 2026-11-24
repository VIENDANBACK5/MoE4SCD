# Individual Tree Crown Extraction from High Resolution Image Based on a Threshold-Improved Watershed Algorithm

- **Tên bài báo:** INDIVIDUAL TREE CROWN EXTRACTION FROM HIGH RESOLUTION IMAGE BASED ON A THRESHOLD-IMPROVED WATERSHED ALGORITHM
- **Tác giả:** Ziyi Wang*, Hao Tang
- **Đơn vị:** School of Resources and Environment, University of Electronic Science and Technology of China (UESTC), Chengdu, China
- **Hội nghị:** IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2024, pp. 9521-9524)
- **DOI:** [10.1109/IGARSS53475.2024.10642760](https://doi.org/10.1109/IGARSS53475.2024.10642760)
- **Phân loại phương pháp:** Advanced Morphology & Threshold-Controlled Watershed (Traditional Computer Vision / Non-Deep Learning)

---

## 1. Bài toán & Bối cảnh Nghiên cứu

- **Nhiệm vụ (Task):** Tách chiết ranh giới cá thể tán cây (ITC extraction) từ ảnh trực giao UAV độ phân giải siêu cao phục vụ mô hình hóa 3D cảnh quan khuôn viên trường đại học (campus 3D modeling) và giám sát sinh trưởng cây xanh.
- **Hệ sinh thái:** Khuôn viên đô thị (Campus landscape, UESTC Qingshuihe Campus), gồm công trình, đường, bãi cỏ và cây (bài không nêu loài cây).
- **Dữ liệu đầu vào & Độ phân giải:**
  - Thiết bị bay: **DJI Phantom 4 Pro**, độ cao bay $100\,\text{m}$.
  - Dữ liệu: Ảnh trực giao kỹ thuật số (Digital Orthophoto Map - DOM) hiệu chỉnh quang sai góc nghiêng.
  - Phân tích trên 2 khu vực mẫu:
    - **data_a:** Vùng mật độ cây dày đặc (167 cây thực địa).
    - **data_b:** Vùng mật độ cây phân tán, thưa thớt (182 cây thực địa).
- **Thách thức cốt lõi của Watershed truyền thống:**
  - **Lỗi nhận diện 2 đỉnh (hoặc hơn) trên 1 cây:** bài nêu việc tìm cực đại địa phương làm sinh nhiều ngọn trên một cây và gây phân đoạn thừa; bài không giải thích nguyên nhân vật lý (phần "cành nhánh lá rộng" trước đây là suy đoán).
  - **Phân đoạn thừa:** bài nêu thuật toán watershed không có chức năng nhận biết diện tích nên sẽ phân đoạn cả khoảng hở giữa tán và các đối tượng khác.

---

## 2. Chi tiết Phương pháp Phân đoạn (Segmentation Methodology)

Nhóm tác giả đề xuất quy trình cải tiến 3 bước chặt chẽ kết hợp giữa **Chỉ số thực vật khả kiến**, **Ngưỡng khoảng cách cực đại địa phương** và **Ngưỡng diện tích tán cây**:

```
                       [ Ảnh UAV DOM RGB ]
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │   BƯỚC 1: Tách chiết Cụm cây (Tree Clusters) │
         │   - Tính chỉ số thực vật khả kiến VI         │
         │   - Phân ngưỡng độ sáng tách thảm cỏ/cây     │
         └──────────────────────┬───────────────────────┘
                                │ Mặt nạ cụm cây nhị phân
                                ▼
         ┌──────────────────────────────────────────────┐
         │   BƯỚC 2: Tách ngọn cây & Ranh giới nền      │
         │   - Xử lý mở/đóng tái cấu trúc hình thái     │
         │   - Tìm cực đại địa phương (Local Maxima)    │
         │   - [CẢI TIẾN 1]: Lọc ngưỡng khoảng cách d   │
         │     Gộp 2 đỉnh gần nhau trên cùng 1 tán cây  │
         │   - Biến đổi khoảng cách tìm đường viền nền  │
         └──────────────────────┬───────────────────────┘
                                │ Markers: Ngọn cây (Foreground) & Nền (Background)
                                ▼
         ┌──────────────────────────────────────────────┐
         │   BƯỚC 3: Phân đoạn Watershed & Lọc diện tích│
         │   - Áp đặt marker lên bản đồ gradient xám    │
         │   - Chạy Watershed Segmentation              │
         │   - [CẢI TIẾN 2]: Lọc ngưỡng diện tích Area  │
         │     Xóa các mảnh quá nhỏ hoặc quá lớn        │
         │   - Giao cắt với mặt nạ cụm cây ban đầu      │
         └──────────────────────┬───────────────────────┘
                                │
                                ▼
         [ Đa giác ranh giới từng cây hoàn chỉnh (F1: 86.67% và 88.77%) ]
```

### 2.1. Bước 1: Tách chiết cụm cây (Tree Cluster Extraction)
Để loại bỏ các đối tượng phi thực vật (đường giao thông, mái nhà, sân bê tông), tác giả sử dụng **Chỉ số thực vật dải ánh sáng nhìn thấy cải tiến (Visible-band Vegetation Index - VI)**:

$$VI = (2 \times G' - R' - B') - (1.4 \times R' - G')$$

Trong đó $R', G', B'$ là các giá trị chuẩn hóa độ chói:
$$R' = \frac{R}{R+G+B}, \quad G' = \frac{G}{R+G+B}, \quad B' = \frac{B}{R+G+B}$$

- **Tách cây khỏi bãi cỏ:** bài nêu độ sáng của cỏ và tán cây khác nhau và dùng ngưỡng độ sáng phù hợp để tách; bài không nêu thêm lý do.

### 2.2. Bước 2: Tách ngọn cây (Foreground) và Ranh giới nền (Background)
1. **Làm mịn hình thái học:** Chuyển đổi mặt nạ cụm cây sang mức xám, tạo phần tử cấu trúc tròn (`strel`) và áp dụng phương pháp *mở bằng tái cấu trúc* (open-based reconstruction) và *đóng bằng tái cấu trúc* (closed-based reconstruction) để khử nhiễu gân lá.
2. **Tìm ngọn cây bằng Local Maxima:** Dựa trên giả định đỉnh tán cây có độ phản xạ ánh sáng cao nhất.
3. **Cải tiến bằng Ngưỡng Khoảng Cách (Distance-Threshold Algorithm):**
   - Tính ma trận khoảng cách ngắn nhất giữa các vùng liên thông cực đại địa phương.
   - Nếu khoảng cách giữa 2 vùng nhỏ hơn ngưỡng trung bình $\implies$ so sánh đặc trưng độ sáng và màu sắc của 2 vùng đó.
   - Giữ lại vùng có đặc trưng gần với đỉnh tán cây nhất và loại bỏ điểm cực đại phụ.
   - Theo bài, bước này "cải thiện" lỗi hai ngọn trên một cây (không phải khắc phục triệt để: N3 còn 6 và 10 cây).
4. **Tách đường ranh giới nền:** Áp dụng biến đổi khoảng cách (distance transform) trên ma trận nhị phân để tạo các đường phân cách đáy thung lũng giữa các cụm cây.

### 2.3. Bước 3: Phân đoạn Watershed có kiểm soát diện tích
1. Đưa các điểm ngọn cây (foreground markers) và đường nền (background markers) làm các điểm cực tiểu nhân tạo trên bản đồ gradient mức xám.
2. Chạy giải thuật Watershed đánh dấu (Marker-controlled Watershed).
3. **Cải tiến bằng Ngưỡng Diện Tích (Area Thresholding):**
   - Đặt ngưỡng diện tích tán phù hợp, xóa các vùng quá lớn và quá nhỏ (bài không nêu giá trị ngưỡng, cũng không nêu liên hệ với loài cây hay trường hợp hai cây dính liền).
   - Dùng lại mặt nạ cụm cây ban đầu để cắt tỉa các phần viền thừa.

---

## 3. Kết quả Thực nghiệm & Số liệu Định lượng

Tác giả phân loại kết quả thành 6 trạng thái: Hoàn hảo ($N_1$), Gần đúng ($N_2$), Chẻ nhỏ/Over-seg ($N_3$), Gộp tán/Under-seg ($N_4$), Sai lệch/Incorrect ($N_5$), Bỏ sót/Missed ($N_6$).
- Độ chính xác thống kê: $A_c = \frac{N_1+N_2}{N_d} \times 100\%$
- Recall: $Re = \frac{N_1+N_2}{N_r} \times 100\%$
- Tỉ lệ khớp hoàn hảo (Perfect Match): $R_p = \frac{N_1}{N_d} \times 100\%$
- Lỗi chẻ nhỏ (Over-segmentation Error): $E_o = \frac{N_3+N_5}{N_d} \times 100\%$
- Lỗi gộp tán (Under-segmentation Error): $E_u = \frac{N_4+N_6}{N_d} \times 100\%$

### Bảng đối chứng trước và sau khi áp dụng thuật toán cải tiến ngưỡng:

| Thước đo đánh giá | Khu vực dày (data_a) - Gốc | Khu vực dày (data_a) - Cải tiến | Khu vực thưa (data_b) - Gốc | Khu vực thưa (data_b) - Cải tiến |
|---|:---:|:---:|:---:|:---:|
| **Số cây khớp hoàn hảo ($N_1$)** | 82 | **101** (+23%) | 88 | **101** (+15%) |
| **Số cây bị chẻ nhỏ ($N_3$)** | 15 | **6** (-60%) | 21 | **10** (-52%) |
| **Độ chính xác $A_c$ (%)** | 79.77% | **87.73%** (+7.96%) | 77.14% | **86.46%** (+9.32%) |
| **Độ thu hồi $Re$ (%)** | 82.63% | **85.63%** (+3.00%) | 89.01% | **91.21%** (+2.20%) |
| **F1-score (%)** | 81.18% | **86.67%** (+5.49%) | 82.65% | **88.77%** (+6.12%) |
| **Tỉ lệ khớp hoàn hảo $R_p$ (%)** | 47.40% | **61.96%** (+14.56%) | 41.90% | **52.60%** (+10.70%) |
| **Lỗi chẻ nhỏ $E_o$ (%)** | 13.87% | **6.13%** (-7.74%) | 19.05% | **8.85%** (-10.20%) |
| **Lỗi gộp tán $E_u$ (%)** | 6.36% | 6.75% | 6.19% | 7.29% |

---

## 4. Hạn chế

**Điều từ số liệu của bài:**
- Lỗi gộp tán $E_u$ **tăng nhẹ** sau cải tiến (data_a 6.36% → 6.75%, data_b 6.19% → 7.29%); bài không nhận xét điều này.
- Hai vùng, 167 và 182 cây, một khuôn viên, không có seed hay kiểm chứng ngoài vùng.
- Bài nêu "các tham số được điều chỉnh theo hiệu quả trích xuất thực tế" (mục 2.1), tức tham số chọn thủ công theo từng vùng.

**Suy luận của người đọc, bài không nêu:** việc chuyển sang vùng khác có thể cần chỉnh lại các ngưỡng; ảnh RGB 2D khó tách hai tán màu giống nhau (số lượng "5–6 siêu tham số" ở README cũ là ước lượng, không có trong bài).

## 5. Giá trị Tham khảo cho TreeFlowNet & Bài báo IGARSS 2027

> **Đã sửa 02/10/2026:** mục này từng chứa các khẳng định ưu việt của TreeFlowNet không có bằng chứng. Các số của bài gốc ở mục 1–4 đã được đối chiếu với toàn văn; mục 5 chỉ nêu điều bài cho thấy, tách khỏi giả thuyết. (Số đo của chính dự án: trên test CanopyRS, seed 42–44, Mask R-CNN R50 34.82 ± 0.67 AP, TreeFlowNet C5 R50 28.81 AP (một seed), TreeFlowNet Swin-T khoảng 30 AP; xem `docs/KIEN_TRUC_TREEFLOWNET_CLEAN_TOAN_HOC_VA_DIEM_YEU.md` mục 5 và 9. Các giả thuyết dưới đây chưa được kiểm chứng trên dữ liệu của mình.)

- **Watershed cải tiến:** F1 86.67% (từ 81.18%), over-segmentation giảm từ 13.87% xuống 6.13%, nhưng phải đặt nhiều ngưỡng thủ công.
- **So với TreeFlowNet:** flow đưa pixel về điểm hút, nhưng giải mã của TreeFlowNet cũng có ngưỡng chọn trên validation (`seed_m`, `area_m2`, ngưỡng canopy/SDT, độ dài lan truyền). Oracle seed NMS cho thấy luật seed vẫn là nút thắt: với seed chọn bằng GT, validation AP tăng từ 33.65 lên 37.96 (trung bình 3 seed). Vì vậy không thể nói flow "triệt tiêu" lỗi nhiều đỉnh.
- **Bộ lỗi N1…N6** (hoàn hảo, chẻ nhỏ, gộp tán…) hữu ích cho phần phân tích lỗi; mình đã có cách phân loại tương tự (matched/merged/split/missed) trong `error_breakdown.py`.
