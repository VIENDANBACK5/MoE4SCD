# UAV Imagery to Support Individual Tree Management and Monitoring

- **Tên bài báo:** UAV IMAGERY TO SUPPORT INDIVIDUAL TREE MANAGEMENT AND MONITORING
- **Tác giả:** S. Ottoy (1,2,3), C. Metsu (2), W. De Witte (1,3), K. Van Meerbeek (2), A. De Vocht (1,3)
- **Đơn vị:** (1) Bio-Research, PXL University College, Diepenbeek, Belgium; (2) Division of Forest, Nature and Landscape, KU Leuven, Belgium; (3) Center for Environmental Sciences, Hasselt University, Belgium
- **Hội nghị:** IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2024, pp. 1113-1116)
- **DOI:** [10.1109/IGARSS53475.2024.10642937](https://doi.org/10.1109/IGARSS53475.2024.10642937)
- **Phân loại phương pháp:** Photogrammetric SfM Pipeline Benchmarking & Point Cloud-derived CHM Marker-Controlled Watershed

---

## 1. Bài toán & Bối cảnh Nghiên cứu

- **Nhiệm vụ (Task):** Đánh giá tác động của các phần mềm tái tạo cấu trúc từ chuyển động (Structure-from-Motion - SfM photogrammetry) đến độ chính xác của giải thuật phân đoạn ranh giới cá thể cây (Individual Tree Delineation - ITD) từ ảnh drone.
- **Hệ sinh thái & Đối tượng:** Rừng trồng cây thủy tùng cảnh quan (*Taxus plantation*) tại Bỉ; các hàng cây được cắt tỉa thành khối tròn đều đặn, khoảng cách cây rõ ràng, giúp cô lập tác động của sai số quang trắc khỏi sự phức tạp hình thái tự nhiên.
- **Dữ liệu đầu vào & Thông số bay chụp:**
  - Thời gian bay: Tháng 11/2021.
  - Độ cao bay: $40\,\text{m}$ so với mặt đất (AGL).
  - Góc nghiêng camera: $60^\circ$ (chụp xiên obique).
  - Độ chồng lấn ảnh: **80% forward & side overlap**.
- **Ba phần mềm SfM đưa vào thử nghiệm đối đầu:**
  1. **Pix4Dmapper:** Phần mềm thương mại chuẩn công nghiệp trong trắc địa viễn thám.
  2. **Agisoft Metashape:** Phần mềm thương mại xử lý thị giác máy tính và quang trắc 3D phổ biến.
  3. **WebODM (OpenDroneMap):** Phần mềm mã nguồn mở hoàn toàn dựa trên OpenSfM.

---

## 2. Chi tiết Phương pháp Phân đoạn (Segmentation Methodology)

```
                       [ Ảnh UAV RGB Chụp xiên 40m (80% overlap) ]
                                            │
        ┌───────────────────────────────────┼───────────────────────────────────┐
        ▼                                   ▼                                   ▼
  [ Pix4Dmapper ]                 [ Agisoft Metashape ]                     [ WebODM ]
  (Thương mại 1)                    (Thương mại 2)                       (Mã nguồn mở)
        │                                   │                                   │
        ▼                                   ▼                                   ▼
  [ Point Cloud 1 ]                 [ Point Cloud 2 ]                   [ Point Cloud 3 ]
        └───────────────────────────────────┬───────────────────────────────────┘
                                            │
                                            ▼
                        [ Trích xuất Mô hình Chiều cao Tán (CHM) ]
                                            │
                         Bộ lọc Gauss làm mịn cửa sổ 3x3 pixels
                                            │
                                            ▼
                        [ Marker-Controlled Watershed trong R ]
                          - Tìm đỉnh tán làm hạt nhân (Markers)
                          - Lan tràn lưu vực nước phân cách ranh giới
                                            │
                                            ▼
                        [ Đánh giá thông số hình học từng cây ]
                          - Chiều cao cây (Tree Height - TH)
                          - Diện tích tán (Crown Area - CA)
                          - True Positive, False Negative, False Positive
```

### 2.1. Quy trình Xử lý Đám mây điểm & Tạo CHM
1. **Tái tạo 3D Point Cloud:** Ảnh bay chụp được nạp vào 3 phần mềm với thiết lập mặc định (default processing templates).
2. **Nội suy Canopy Height Model (CHM):**
   - Phân loại điểm mặt đất (Ground points) và điểm bề mặt tán cây (Canopy points).
   - Nội suy raster CHM biểu diễn chiều cao tương đối của từng điểm so với mặt đất.
3. **Làm mịn CHM:** Áp dụng bộ lọc làm mịn cửa sổ $3 \times 3$ pixels để loại bỏ các điểm gai nhọn cục bộ do lá cây gây ra.

### 2.2. Thuật toán Phân đoạn Tán cây (ITD Algorithm)
- Triển khai trong môi trường ngôn ngữ R (gói `ForestTools` / `lidR`).
- Sử dụng thuật toán **Watershed có kiểm soát điểm đánh dấu (Marker-Controlled Watershed)**:
  - Xác định các đỉnh cây cục bộ (local maxima) làm marker đại diện cho tâm tán cây.
  - Đảo ngược ma trận khoảng cách / chiều cao CHM để đỉnh cây biến thành bồn trũng (catchment basins).
  - Cho "nước dâng lên" từ các đỉnh cây; ranh giới nơi các bồn trũng gặp nhau chính là đường phân cách giữa hai cây lân cận.

---

## 3. Kết quả Thực nghiệm & Số liệu Định lượng

Tổng số cây thực tế trên ô tiêu chuẩn (Ground Truth Total Tree Count - TTC): **237 cây**.

| Thông số / Chỉ số đánh giá | Pix4Dmapper | Agisoft Metashape | WebODM (Mã nguồn mở) |
|---|:---:|:---:|:---:|
| **Số cây phát hiện đúng (TP)** | **235** | 230 | 175 |
| **Số cây bỏ sót (FN)** | **2** | 7 | **62 (Sót rất nặng)** |
| **Số cây phát hiện sai/dư (FP)** | **2** | 3 | 18 |
| **Độ thu hồi - Recall ($r$)** | **0.99** (99%) | 0.97 (97%) | 0.74 (74%) |
| **Độ chính xác - Precision ($p$)** | **0.99** (99%) | 0.99 (99%) | 0.91 (91%) |
| **F-score ($F$)** | **0.99** | **0.98** | **0.81** |
| **Chiều cao cây trích xuất ($TH$, m)** | $3.43 \pm 0.95$ | $3.28 \pm 0.93$ | $3.22 \pm 0.99$ |
| **Diện tích tán trích xuất ($CA$, m²)** | $1.79 \pm 1.01$ | $1.30 \pm 0.79$ | $1.56 \pm 1.08$ |

---

## 4. Phân tích Nguyên nhân Thất bại & Điểm nghẽn Kỹ thuật (Failure Modes)

1. **Cơ chế nén và co tỷ lệ ảnh của WebODM (Image Downsampling):**
   - Mặc định, WebODM tự động co nhỏ kích thước ảnh đầu vào (resizing) để tiết kiệm RAM và tăng tốc tính toán trên máy chủ phổ thông.
   - Hậu quả: Đám mây điểm 3D bị suy giảm mật độ nghiêm trọng, làm mất đi các chi tiết kết cấu rãnh khe giữa các tán cây.
   - Các cây đứng sát nhau bị **gộp lại thành một khối liền mạch duy nhất (blurred cloud)**.
   - Thuật toán Watershed chỉ tìm được 1 đỉnh thay vì 2 đỉnh, dẫn đến **số lượng cây bỏ sót (False Negative) vọt lên 62 cây** (Recall tụt xuống 74%).
2. **Sai số đo đạc chiều cao tán:** Do đám mây điểm của WebODM và Metashape thưa hơn ở phần chóp nhọn của ngọn cây, chiều cao cây trích xuất bị thấp hơn trung bình $15 - 21\,\text{cm}$ so với Pix4Dmapper ($3.22\,\text{m}$ vs $3.43\,\text{m}$).

---

## 5. Giá trị Tham khảo cho TreeFlowNet & Bài báo IGARSS 2027

> **Đã sửa 02/10/2026:** mục này từng chứa các khẳng định ưu việt của TreeFlowNet không có bằng chứng. Các số của bài gốc ở mục 1–4 đã được đối chiếu với toàn văn; mục 5 chỉ nêu điều bài cho thấy, tách khỏi giả thuyết. (Số đo của chính dự án: trên test CanopyRS, seed 42–44, Mask R-CNN R50 34.82 ± 0.67 AP, TreeFlowNet C5 R50 28.81 AP (một seed), TreeFlowNet Swin-T khoảng 30 AP; xem `docs/KIEN_TRUC_TREEFLOWNET_CLEAN_TOAN_HOC_VA_DIEM_YEU.md` mục 5 và 9. Các giả thuyết dưới đây chưa được kiểm chứng trên dữ liệu của mình.)

- **Điều bài thực sự cho thấy:** trên rừng trồng Taxus cắt tỉa hình cầu, SfM bằng Pix4D cho F1 0.99 còn WebODM cho F1 0.81 (bỏ sót 62 cây), vì phần mềm làm mờ point cloud. Bài không đánh giá rừng tự nhiên tán kín và không liên quan đến QuebecTrees.
- **Giả thuyết (của mình, không phải của bài):** lỗi SfM có thể làm mờ biên tán trong QuebecTrees. Chưa kiểm tra pipeline SfM của bộ dữ liệu đó, nên chỉ coi là hướng cần xác minh khi mô tả hạn chế dữ liệu.
- Bài chỉ so các phần mềm SfM với cùng một thuật toán watershed trên CHM, không so deep learning với watershed, nên không dùng được để biện hộ "deep learning cần thiết hơn watershed".
