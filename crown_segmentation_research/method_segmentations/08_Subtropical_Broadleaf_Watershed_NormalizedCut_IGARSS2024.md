# Individual Tree Crown Segmentation in Subtropical Broadleaf Forests Using UAV-Based Ultrahigh-Resolution RGB Data

- **Tên bài báo:** INDIVIDUAL TREE CROWN SEGMENTATION IN SUBTROPICAL BROADLEAF FORESTS USING UAV-BASED ULTRAHIGH-RESOLUTION RGB DATA
- **Tác giả:** Ruoning Zhu (1), Guoqi Chai (2), Xin Tian (3*)
- **Đơn vị:** (1) The Third Construction Co., Ltd of China Construction First Group; (2) College of Forestry, Beijing Forestry University; (3) Institute of Forest Resource Information Techniques, Chinese Academy of Forestry (CAF), Beijing, China
- **Hội nghị:** IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2024, pp. 3097-3100)
- **DOI:** [10.1109/IGARSS53475.2024.10642737](https://doi.org/10.1109/IGARSS53475.2024.10642737)
- **Phân loại phương pháp:** Graph-Theoretic Clustering & Watershed Fusion (DSM Watershed Over-segmentation + Spectral-Texture-Intensity Normalized Cut)

---

## 1. Bài toán & Bối cảnh Nghiên cứu

- **Nhiệm vụ (Task):** Phân đoạn ranh giới từng cá thể tán cây (ITC segmentation) trong rừng lá rộng cận nhiệt đới rậm rạp, tập trung giải quyết bài toán tán cây có nhiều ngọn/nhiều đỉnh sinh trưởng (*multi-tree apex crowns*).
- **Hệ sinh thái:** rừng lá rộng cận nhiệt đới tại Rừng Gaofeng, Nam Ninh, Quảng Tây (22°49′–23°5′N, 108°7′–108°38′E; mưa 1,300 mm/năm, nhiệt độ TB 21.7 °C). Bài chỉ nêu một **khu rừng Castanopsis hystrix được chọn từ vùng nghiên cứu để kiểm chứng**; các mô tả "thuần loài", "tự nhiên" trước đây không có trong bài.
- **Dữ liệu đầu vào & Độ phân giải:**
  - Thiết bị bay: UAV **DJI M300 RTK**, camera ghi trong bài là "Zenith P1" (45 megapixel, tiêu cự 35 mm; có thể là Zenmuse P1 viết sai).
  - Độ cao bay: $200\,\text{m}$ (tháng 12/2022).
  - Xử lý quang trắc: Phần mềm Agisoft Metashape Professional v2.0 tạo ra hai lớp dữ liệu siêu nét:
    - **DOM (Digital Orthophoto Map RGB):** Độ phân giải **0.06 m (6 cm/pixel)**.
    - **DSM (Digital Surface Model):** Độ phân giải **0.06 m (6 cm/pixel)**.
- **Thách thức cốt lõi:**
  - Bài nêu tán cây lá rộng cận nhiệt đới có nhiều ngọn (multiple tree apexes) nên watershed thường bị phân đoạn thừa. Con số "3 đến 5 nhánh" trước đây không có trong bài.
  - Nếu chạy Watershed trực tiếp trên DSM $\implies$ mỗi đỉnh nhánh bị coi là một cây riêng biệt, gây ra **chẻ nhỏ tán cây (over-segmentation) thảm hại**.

---

## 2. Chi tiết Phương pháp Phân đoạn (Segmentation Methodology)

Nhóm tác giả đề xuất triết lý 2 bước: **Cố tình chẻ nhỏ (Over-segmentation) có kiểm soát bằng DSM Watershed**, sau đó **Dùng thuật toán cắt đồ thị chuẩn hóa (Normalized Cut) để gom các mảnh cùng một cây lại dựa trên độ tương đồng quang phổ, kết cấu và cường độ**.

```
    [ UAV DSM 6cm ]                                     [ UAV DOM RGB 6cm ]
           │                                                     │
    Gaussian Smoothing                                           │
  (sigma = 2.2, window = 81px)                                   │
           │                                                     │
           ▼                                                     │
    [ DSM-based Watershed ]                                      │
    (Cố tình chẻ nhỏ, Recall = 1.0)                              │
           │                                                     │
    Các mảnh tán nhỏ (Crown Patches)                             │
           │                                                     │
           └───────────────────────┬─────────────────────────────┘
                                   │
                                   ▼
                 ┌───────────────────────────────────┐
                 │   XÂY DỰNG ĐỒ THỊ VÔ HƯỚNG G=(V,E) │
                 │   - Đỉnh V: Tâm của các mảnh tán  │
                 │   - Trích xuất 3 nhóm đặc trưng:  │
                 │     * Quang phổ: S(i) = VEG Index │
                 │     * Kết cấu: T(i) = GLCM Mean   │
                 │     * Cường độ sáng: I(i)         │
                 │   - Tính trọng số cạnh w(i, j)    │
                 │     bằng hàm khoảng cách Gauss    │
                 └─────────────────┬─────────────────┘
                                   │
                                   ▼
                 ┌───────────────────────────────────┐
                 │   CẮT ĐỒ THỊ CHUẨN HÓA (N-CUT)    │
                 │   - Gom cụm các mảnh có w(i,j) cao│
                 │   - Gộp các mảnh cùng một cây      │
                 └─────────────────┬─────────────────┘
                                   │
                                   ▼
                 [ Đa giác ranh giới từng cây hoàn chỉnh (F1 = 0.87) ]
```

### 2.1. Bước 1: Phân đoạn sơ bộ bằng Watershed trên DSM
- Mục tiêu: Đảm bảo **không bỏ sót cây (Zero Under-segmentation / Recall = 1.0)** bằng cách cố tình chấp nhận chẻ nhỏ tán cây thành các mảnh nhỏ.
- Lọc mịn DSM bằng bộ lọc Gauss với $\sigma = 2.2$ và cửa sổ kích thước **$81 \times 81$ pixels** (bài ghi "xấp xỉ đường kính tán trung bình"; 81 px × 0.06 m ≈ 4.9 m là số tính của người đọc).
- Chạy giải thuật Watershed tạo ra các mảnh tán sơ bộ (*crown patches*). Kết quả: Recall đạt tuyệt đối 1.0, nhưng Precision chỉ đạt 0.58 do over-segmentation.

### 2.2. Bước 2: Trích xuất đặc trưng đa chiều từ ảnh quang học DOM
Tác giả khảo sát 6 chỉ số quang phổ thực vật (ExG, ExGR, GLA, VDVI, NGRDI, VEG) và 8 đặc trưng kết cấu GLCM trên kênh Green (Mean, Entropy, Contrast, Second Moment, Homogeneity, Dissimilarity, Variance, Correlation) cùng kênh cường độ sáng Intensity $I$.
- **Lựa chọn đặc trưng tối ưu:**
  1. **Đặc trưng quang phổ $S(i)$:** Chọn chỉ số **VEG** (Vegetative Index) vì làm nổi bật cấu trúc tán lá tốt nhất.
  2. **Đặc trưng kết cấu $T(i)$:** Chọn **GLCM Mean** vì làm nổi bật ranh giới giữa hai tán cây khác nhau rõ nhất.
  3. **Đặc trưng cường độ $I(i)$:** Kênh cường độ sáng phản xạ bề mặt.
- Mọi đặc trưng đều được chuẩn hóa min-max về đoạn $[0, 1]$.

### 2.3. Bước 3: Xây dựng Đồ thị & Cắt đồ thị chuẩn hóa (Normalized Cut)
- Xây dựng đồ thị vô hướng $G = (V, E)$, trong đó mỗi đỉnh $v_i \in V$ là tâm của một mảnh tán sơ bộ, và cạnh $e_{ij} \in E$ nối giữa hai mảnh lân cận.
- **Hàm trọng số cạnh tương đồng liên mảnh ($w_{i,j}$):**

$$w_{i,j} = \exp \left( - \left[ \frac{|S(i) - S(j)|^2}{0.05 \times \max|S(i) - S(j)|} + \frac{|T(i) - T(j)|^2}{0.05 \times \max|T(i) - T(j)|} + \frac{|I(i) - I(j)|^2}{0.05 \times \max|I(i) - I(j)|} \right] \right)$$

- **Nguyên lý gom cụm (Normalized Cut Criterion - Shi & Malik, 2000):**
  - Bài nêu tiêu chí: tối đa độ tương đồng trong cùng một tán và tối thiểu giữa các tán khác nhau, dựa trên giả định quang phổ và kết cấu trong cùng một cây giống nhau hơn giữa các cây lân cận.
  - Bài **không** trình bày cách giải (giải bài toán trị riêng là của Shi & Malik 2000, không phải mô tả trong bài này); các lý giải về loài, tuổi lá, hướng nắng trước đây là suy đoán.

---

## 3. Kết quả Thực nghiệm & Số liệu Định lượng

Đánh giá trên tập kiểm thử rừng *Castanopsis hystrix Miq.* so với nhãn vẽ tay (đọc từ Hình 2c của bài: watershed TP 43, FP 31, FN 0; phương pháp đề xuất TP 40, FP 9, FN 3), tức **tập kiểm thử chỉ có 43 cây**:

| Phương pháp | Recall (R) | Precision (P) | F-score (F) | Nhận xét chi tiết |
|---|:---:|:---:|:---:|---|
| **DSM-based Watershed (Gốc)** | **1.00** | 0.58 | 0.74 | Bắt trọn vẹn mọi cây, nhưng chẻ nhỏ nát tán (P rất thấp) |
| **Watershed + Spectral-Texture-Intensity N-Cut (Đề xuất)** | 0.93 | **0.82** | **0.87** | Tăng 0.13 điểm F (0.74 → 0.87; bài viết "13%"); bài nói "giảm" over-segmentation |

- Khi áp dụng Normalized Cut, Precision tăng vọt từ **0.58 lên 0.82** (+24 điểm phần trăm), trong khi Recall vẫn giữ ở mức rất cao **0.93**.

---

## 4. Hạn chế

**Điều từ số liệu của bài:**
- Tập kiểm thử chỉ **43 cây** (từ Hình 2c), một khu rừng, không có lặp hay kiểm chứng ngoài vùng; ước lượng F = 0.87 do đó rất không chắc chắn.
- Recall giảm từ 1.00 xuống 0.93 (FN từ 0 lên 3); bài không phân tích nguyên nhân.
- Cách tính TP/FP/FN: bài ghi lẫn nhãn (chú thích nói "FP là missed detection"), nên cần thận trọng khi so với metric khác.

**Suy luận của người đọc, bài không nêu:** phương pháp cần DSM từ ảnh UAV chồng phủ (bước watershed dùng DSM); tính toán N-Cut có thể tốn bộ nhớ khi số mảnh lớn (khẳng định O(N²) trước đây là ước lượng của người viết, không có trong bài).

## 5. Giá trị Tham khảo cho TreeFlowNet & Bài báo IGARSS 2027

> **Đã sửa 02/10/2026:** mục này từng chứa các khẳng định ưu việt của TreeFlowNet không có bằng chứng. Các số của bài gốc ở mục 1–4 đã được đối chiếu với toàn văn; mục 5 chỉ nêu điều bài cho thấy, tách khỏi giả thuyết. (Số đo của chính dự án: trên test CanopyRS, seed 42–44, Mask R-CNN R50 34.82 ± 0.67 AP, TreeFlowNet C5 R50 28.81 AP (một seed), TreeFlowNet Swin-T khoảng 30 AP; xem `docs/KIEN_TRUC_TREEFLOWNET_CLEAN_TOAN_HOC_VA_DIEM_YEU.md` mục 5 và 9. Các giả thuyết dưới đây chưa được kiểm chứng trên dữ liệu của mình.)

- **Hướng "chẻ nhỏ rồi gộp":** bài (Normalized Cut sau watershed DSM) và TreeCoG (contour + GCN) cùng theo nguyên tắc này trên rừng lá rộng dày. TreeCoG dùng mạng đồ thị GCN để gộp contour.
- **Ý tưởng cho TreeFlowNet (chưa thử):** một bước gộp các bồn trũng lân cận theo độ tương đồng. Có cơ sở từ số đo: oracle `nosplit` (loại seed trùng trong cùng một crown GT) tăng validation AP khoảng +0.94 và `nomerge` +1.53, nên cả chia và gộp đều có dư địa. Nhưng bài này dùng DSM, còn mình chỉ dùng RGB.
