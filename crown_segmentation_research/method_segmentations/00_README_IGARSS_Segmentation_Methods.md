# Tổng hợp & Phân tích Chi tiết 10 Phương pháp Phân đoạn Tán cây (ITC) tại Hội nghị IGARSS

*Thư mục: `method_segmentations/`*  
*Cập nhật: 02/10/2026. Nguồn: toàn văn 10 bài IGARSS 2024/2025. Số liệu mục 1 đã đối chiếu với văn bản trích từ PDF (`paper/texts/paper_IGARSS/`); bài 05, 06, 08 đã đọc lại trực tiếp từ PDF (kể cả hình). Mục 2 và 3 đã viết lại vì bản cũ chứa kết luận quá mức và khẳng định về TreeFlowNet không có bằng chứng. Các số đọc từ hình (bài 09) chưa kiểm chứng bằng văn bản.*

---

## 1. Bảng Đối chiếu Tổng hợp 10 Bài báo IGARSS (Master Comparison Matrix)

| ID | Tên bài báo & Tác giả | Năm & Venue | Cảm biến & Độ phân giải (GSD) | Hệ sinh thái & Khu vực | Trường phái Phân đoạn (Paradigm) | Mô hình / Thuật toán cốt lõi | Metric & Kết quả tiêu biểu | Hạn chế / Điểm nghẽn chính |
|:---:|---|:---:|:---:|:---:|:---:|---|---|---|
| **01** | **Combining Canopy Mapping and Tree Detection**<br>*(Budin et al., i-Sea)* | IGARSS 2024 | Pléiades (50 cm) + Aerial (10 cm) + LiDAR (50 cm) | Rừng đô thị Bordeaux, Pháp (580 km²) | **Hybrid Fusion** (Semantic + BBox Detection) | Random Forest (Canopy) + RetinaNet (DeepForest) + Lọc hình thái | PITVD: P = **95%**<br>Toàn thành phố: P = **85%** (tăng từ 58%) | RetinaNet sụp đổ trong rừng dày (P = 58%); dựa vào các ngưỡng cứng thủ công. |
| **02** | **Deep Learning-Based Tree Segmentation with LiDAR & Aerial**<br>*(Li et al., UESTC)* | IGARSS 2025 | LiDAR (0.5 m) + NAIP RGB (0.6 m) | Rừng tự nhiên California, Mỹ (Carr Hirz Delta Fires 2019) | **Weakly-supervised / Pseudo-label** | Point Cloud Region Growing + DBSCAN + Concave Hull $\to$ YOLOv11m-seg | LiDAR Sample: F = **0.87**<br>YOLOv11: mAP@50 = **0.856**, P = **0.918** | YOLOv11-seg xuất ra mask hình hộp chữ nhật trên cây nhỏ; CHM Watershed bị over-seg nặng. |
| **03** | **Hazard Tree Instance Detection via DNN Models**<br>*(Prakash & Saxena, AiDash)* | IGARSS 2025 | Pléiades Neo (30 cm) + Aerial (7–15 cm) | Đa địa hình (Rừng, Đô thị, Đồng cỏ) tại Mỹ, Úc, Bỉ | **Foundation Model Semantic Segmentation** | DINOv2 ViT Encoder + ResNet Decoder + Weighted Focal Tversky Loss ($\gamma < 1$) | Aerial F1 = **91.0** (Satellite: 65.0)<br>Forest F1 = **90.0**, Grassland: **94.0** | Dễ nhầm bóng râm cành khô thành cây chết; chuyên gia người thật chỉ đồng thuận 40%. |
| **04** | **ITC Based Tree Species Classification from UAV**<br>*(Weishaupt et al., DLR & TUM)* | IGARSS 2025 | UAV RGB (**2 cm/px**) | Rừng hỗn giao ôn đới BAMFORESTS, Đức (105 ha, 27k cây) | **Instance-consistency Loss & Post-proc** | U-Net 512x512 + ITC Loss (tỉ lệ pixel sai cấp cây) + Majority Voting trên mask ITC | Test-2: mIoU = **44.39%** (+2.36%)<br>Test-1 (OOD): mIoU = **20.93%** (+5.47%) | Sụp đổ ở các loài thiểu số ($< 5\%$ mẫu); phụ thuộc vào chất lượng đa giác ranh giới ban đầu. |
| **05** | **ITC Delineation Based on DL for Arid Areas**<br>*(Shi et al., CAS & Xinjiang Univ)* | IGARSS 2024 | WorldView-3 Pan-sharpened (**0.31 m/px**) | Sa mạc Mu Us Desert, Nội Mông, TQ (Cây khô hạn) | **Two-stage Decoupled Pipeline** | Mask R-CNN (ResNet-50) $\to$ Patch-based CNNs (GoogLeNet, DenseNet, ResNet) | Mask R-CNN: Recall = **84.6–99.9%**<br>GoogLeNet species OA = **96.0–96.4%** | Mask R-CNN bị False Positives rất cao ở vùng thưa (P chỉ 42.6%); patch 64x64 mất ngữ cảnh. |
| **06** | **Threshold-Improved Watershed for Crown Extraction**<br>*(Wang & Tang, UESTC)* | IGARSS 2024 | UAV DOM RGB (Flight 100m) | Khuôn viên đại học UESTC (Cây đô thị) | **Traditional Computer Vision** (Advanced Watershed) | Chỉ số thực vật $VI$ + Distance-threshold local maxima + Area-threshold | data_a F1 = **86.67%** (tăng từ 81.18%)<br>Over-seg $E_o$ giảm từ 13.87% xuống **6.13%** | Tham số chọn thủ công theo từng vùng (bài nêu); $E_u$ tăng nhẹ sau cải tiến; chỉ hai vùng nhỏ. |
| **07** | **ITC Segmentation Based on YOLOv5n-Seg in Tropical Rainforest**<br>*(Shi et al., CAS & Xinjiang Univ)* | IGARSS 2025 | Google Earth Satellite RGB (**0.5 m/px**) | Rừng mưa nhiệt đới Hải Nam, TQ (Tán kín đa tầng) | **Lightweight Single-stage Instance Seg** | YOLOv5n-seg (Focus + CSPNet + SPP + FPN/PAN + Prototype Mask Head) | mAP50 = **90.3%** (Mask R-CNN: 85.2%)<br>mAP50-95 = **57.6%** (Mask R-CNN: 45.8%) | Tập test chỉ 136 patch (sau tăng cường) từ 1 trong 9 subimage; chưa kiểm tra ngoài cảnh. |
| **08** | **ITC Segmentation in Subtropical Broadleaf Forests**<br>*(Zhu, Chai, Tian; China Construction First Group, Beijing Forestry Univ., CAF)* | IGARSS 2024 | UAV DOM + DSM (**6 cm/px**) | Rừng lá rộng cận nhiệt đới Gaofeng, Quảng Tây, TQ | **Graph-Theoretic Clustering & Watershed** | DSM Watershed Over-segmentation (Recall 1.0) $\to$ Spectral-Texture-Intensity Normalized Cut | Watershed gốc: F1 = 0.74 (P = 0.58)<br>Watershed + N-Cut: F1 = **0.87** (P = **0.82**) | Tập kiểm thử chỉ 43 cây (Hình 2c); cần DSM chất lượng cao; Recall giảm 1.00 → 0.93 (bài không giải thích). |
| **09** | **More Labels or Better Labels? Historical Aerial Tree Delineation**<br>*(Ferreira et al., IRISA & DYNAFOR)* | IGARSS 2024 | Ảnh lịch sử 1942 đơn sắc Grayscale (**0.75 m/px**) | Vùng nông lâm kết hợp Tây Nam nước Pháp (172 km²) | **Data-Centric Benchmark & Attention Seg** | U-Net + MobileNetV2 + scSE Attention Block + CLAHE + Dice Loss (51 runs) | Cây đơn lẻ: Fine F1 = **54%** vs Coarse **42%**<br>Toàn tập: Fine = **70.70%** vs Coarse **69.96%** (chênh +0.74) | Nhãn mịn tốn 10–40 phút/patch so với 4–8 phút của nhãn thô, nhưng Dice tổng chỉ tăng 0.74 điểm (số F1 từng nhóm đọc từ hình, chưa kiểm chứng bằng văn bản). |
| **10** | **UAV Imagery for Individual Tree Monitoring & SfM Impact**<br>*(Ottoy et al., KU Leuven)* | IGARSS 2024 | UAV RGB 40m AGL (80% overlap) | Rừng trồng Taxus cắt tỉa hình cầu đều đặn tại Bỉ | **Photogrammetric SfM Benchmarking** | Pix4D vs Agisoft vs WebODM $\to$ CHM $\to$ Marker-controlled Watershed (R `lidR`) | Pix4D F1 = **0.99** (TP = 235, FN = 2)<br>WebODM F1 = **0.81** (TP = 175, **FN = 62**) | Bài nêu quy trình mặc định của WebODM thu nhỏ ảnh, point cloud kém sắc nét nên cây gần nhau bị gộp (FN = 62); chỉ một rừng trồng Taxus. |

---

## 2. Phân loại theo 4 Trường phái Công nghệ tại IGARSS

### Nhóm A: Học sâu Phân đoạn Cá thể (Instance Segmentation: Top-Down vs Single-Stage)
- **Đại diện:** Paper 02 (YOLOv11-seg), Paper 05 (Mask R-CNN), Paper 07 (YOLOv5n-seg).
- **Đặc trưng:**
  - **Mask R-CNN (Two-stage):** Paper 05: Recall 84.6%–99.9% nhưng Precision 42.6% ở Region 1 (một vùng, bài không giải thích nguyên nhân). Paper 07: mAP50-95 45.8% so với 57.6% của YOLOv5n-seg trên 136 patch (nguyên nhân do lưới mask $28 \times 28$ là suy đoán, bài không chứng minh).
  - **YOLO-seg (Single-stage prototype):** Nhẹ (Paper 07: 2.0M tham số, 7.1 GFLOPs so với 43.2M và 76.2 GFLOPs của Mask R-CNN), mAP50-95 cao hơn 11.8 điểm trên 136 patch vệ tinh 0.5 m. Paper 02 ghi nhận YOLOv11-seg đôi khi xuất vùng hình chữ nhật thay vì biên chính xác ở cây rất nhỏ.

### Nhóm B: Ứng dụng Mô hình Nền tảng (Foundation Models & Attention Mechanisms)
- **Đại diện:** Paper 03 (DINOv2 ViT U-Net), Paper 09 (scSE Attention U-Net).
- **Đặc trưng:**
  - **DINOv2:** Tận dụng biểu diễn tự giám sát (self-supervised visual features) giúp mô hình đạt F1 = 90–94% (rừng 90, đô thị 88, đồng cỏ 94) ở ảnh 7–15 cm; bài không nói rõ mức fine-tune.
  - **scSE Attention:** Tái hiệu chuẩn đồng thời không gian và kênh đặc trưng, (bài mô tả cơ chế; chưa thấy ablation tách riêng đóng góp của scSE trong phần đã đọc).
  - **Loss function đặc thù:** Paper 03 ghi nhận trong thí nghiệm của họ, đặt tham số focal $\gamma < 1$ cho kết quả tốt hơn $\gamma > 1$ (cơ chế giải thích là của người viết tóm tắt, không phải kết luận chắc của bài).

### Nhóm C: Giải thuật Hình thái học & Cắt Đồ thị (Morphology, Watershed & Graph Cut)
- **Đại diện:** Paper 06 (Threshold-improved Watershed), Paper 08 (Watershed + Normalized Cut), Paper 10 (SfM CHM Watershed).
- **Đặc trưng:**
  - **Nguyên lý 2 bước (Over-segment $\to$ Merge):** Paper 08 và Paper 06 cùng nêu watershed dễ bị chẻ nhỏ tán. Paper 08 xử lý bằng cách chẻ nhỏ có kiểm soát bằng DSM rồi gộp các mảnh bằng độ tương đồng quang phổ, kết cấu và cường độ (Normalized Cut). Paper 06 không gộp mà lọc các đỉnh gần nhau theo ngưỡng khoảng cách rồi lọc theo diện tích.
  - Paper 08 tăng F từ 0.74 lên 0.87 bằng Normalized Cut (trên tập kiểm thử chỉ 43 cây). Paper 06 tăng F1 từ 81.18% lên 86.67% (data_a) và từ 82.65% lên 88.77% (data_b) bằng ngưỡng khoảng cách và ngưỡng diện tích.

### Nhóm D: Dữ liệu, Nhãn giả & Sai số Quang trắc (Data-Centric, Pseudo-labels & SfM Bias)
- **Đại diện:** Paper 02 (LiDAR point cloud DBSCAN + Concave Hull), Paper 09 (More vs Better labels), Paper 10 (Ảnh hưởng của phần mềm SfM).
- **Đặc trưng:**
  - **Tạo nhãn giả từ LiDAR:** Paper 02: nhãn giả từ CHM watershed cho F = 0.62 và mô hình YOLOv11 huấn luyện trên đó có mAP@50 0.720, so với F = 0.87 và 0.856 khi nhãn lấy từ region growing trên point cloud (chênh 13.6 điểm; một khu vực, không lặp).
  - **Quy luật kích thước tán:** Paper 09: nhãn mịn chỉ nâng Dice tổng từ 69.96% lên 70.70%; theo bài, ảnh hưởng lớn hơn ở cây đơn lẻ (con số +12% F1 đọc từ hình, chưa kiểm chứng bằng văn bản).
  - **Sai số từ SfM:** Paper 10: trên một rừng trồng Taxus, WebODM (thu nhỏ ảnh theo quy trình mặc định) cho FN = 62 so với 2 của Pix4D (gấp 31 lần); bài giải thích do point cloud kém sắc nét làm cây gần nhau bị gộp.

---

## 3. Bài học cho dự án (đã viết lại, chỉ nêu điều có số đo hoặc nguồn)

### 3.1 Chuẩn mực của các bài IGARSS (nhận xét từ 10 bài này)
- Phần lớn dùng mô hình chuẩn (U-Net, Mask R-CNN, YOLO, watershed) trên một khu vực nhỏ, báo cáo F1/Accuracy hoặc mAP@50 trên tập đánh giá của chính họ, thường không có nhiều seed hay kiểm tra rò rỉ. Số liệu của họ **không so thẳng được** với AP[.5:.95] của benchmark CanopyRS.
- Không bài nào dùng QuebecTrees/CanopyRS, nên không có đối chiếu trực tiếp.

### 3.2 Số đo của chính dự án (test CanopyRS, 22.018 crown, 3 seed)
| Mô hình | AP[.5:.95] |
|---|---|
| Mask R-CNN R50 (ImageNet) | 34.82 ± 0.67 |
| TreeFlowNet C5 R50 (seed 42) | 28.81 |
| TreeFlowNet C5 Swin-T (3 seed) | 30.17 (30.43, 29.96, 30.13) |
| TreeFlowNet C5 Swin-T + bộ chấm seed học được τ = 0.3 (3 seed) | 31.85 (32.05, 31.74, 31.78); τ = 0.3 chọn trên validation, τ = 0.5 chốt từ đầu cho 30.66 |

- Khoảng cách tới Mask R-CNN: 6.0 AP với R50, 4.65 với Swin-T, khoảng 3.0 với Swin-T cộng bộ chấm seed. TreeFlowNet **chưa** vượt Mask R-CNN.
- **Cẩn thận khi so với SelvaMask 27.6 và Teng 32.44:** SelvaMask Bảng 4 ghi SelvaBox → SAM3 đạt 27.6 mAP trên QuebecTrees với dấu ✗ (huấn luyện trên miền đích; bài ghi SelvaBox đã được huấn luyện trên QuebecTrees), tức **không phải zero-shot** (ghi chú trước đây trong tài liệu này viết là OOD zero-shot đã sai). Giao thức vẫn khác (mAP mức tile, tile/overlap và quy đổi riêng của SelvaMask) nên cần chạy lại trên đúng test của mình trước khi so. 32.44 của Teng là một AOI khác.
- Kích thước tập test: 22.018 crown (đã đếm).

### 3.3 Vị thế của TreeFlowNet so với các bài IGARSS
- **So với watershed cải tiến (Paper 06, 08):** TreeFlowNet cũng có ngưỡng chọn trên validation và luật seed NMS cố định. Oracle seed (dùng GT, chỉ trên validation) cho AP 37.96 so với 33.65 của luật cố định, gần Mask R-CNN (38.21), tức luật seed là nút thắt lớn; bộ chấm học được lấy khoảng 40% phần đó trên test. Không có "loại bỏ hoàn toàn siêu tham số thủ công".
- **So với YOLO-seg / Mask R-CNN (Paper 02, 05, 07):** chưa có bằng chứng TreeFlowNet vượt các mô hình này; trên CanopyRS nó kém Mask R-CNN. Điểm mạnh đã đo: recall tốt hơn Mask R-CNN ở crown 4–9 m² (0.828 so với 0.775, một seed, test); điểm yếu: crown dưới 2 m² và 25–100 m².

### 3.4 Điều có thể dùng khi viết (đã gắn đúng nguồn)
- Nhãn đo bằng tay trong rừng dày có thể là tập con của cây thật: dẫn Allen et al. 2025 (đã có PDF trong `paper/`); Paper 09 chỉ nói về chất lượng nhãn trên ảnh lịch sử, không phải chuyện nhãn thiếu cây.
- Về sự đánh đổi giữa top-down và bottom-up, dùng số đo của mình (mục 3.3), không dùng Paper 07/02 như bằng chứng cho xếp hạng điểm tin cậy: hai bài đó không nghiên cứu vấn đề này.
