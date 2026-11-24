# More Labels or Better Labels? A Semantic Segmentation Study Case Using Historical Aerial Images for Tree Delineation

- **Tên bài báo:** MORE LABELS OR BETTER LABELS? A SEMANTIC SEGMENTATION STUDY CASE USING HISTORICAL AERIAL IMAGES FOR TREE DELINEATION
- **Tác giả:** Vitória Barbosa Ferreira (1*), David Sheeren (2), Sébastien Lefèvre (1), Stefan Lang (3)
- **Đơn vị:** (1) Université Bretagne Sud, UMR 6074 IRISA, Vannes, France; (2) Toulouse INP-ENSAT, UMR 1201 DYNAFOR, Toulouse, France; (3) Department of Geoinformatics, University of Salzburg, Austria
- **Hội nghị:** IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2024, pp. 9604-9608)
- **DOI:** [10.1109/IGARSS53475.2024.10642817](https://doi.org/10.1109/IGARSS53475.2024.10642817)
- **Phân loại phương pháp:** Empirical Data-Centric Benchmark & Attention-augmented Semantic Segmentation (U-Net + MobileNetV2 + scSE Attention)

---

## 1. Bài toán & Bối cảnh Nghiên cứu

- **Nhiệm vụ (Task):** Phân đoạn ngữ nghĩa thực vật thân gỗ (Woody vegetation segmentation: cây đơn lẻ, hàng rào cây xanh hedgerows, mảng rừng nhỏ) từ kho ảnh hàng không lịch sử đơn sắc (grayscale) chụp từ năm 1942.
- **Câu hỏi khoa học cốt lõi:** *"Nhiều nhãn hơn hay Nhãn tốt hơn?" (More labels or better labels?)* — Khi ngân sách gán nhãn có hạn, mô hình Deep Learning sẽ hưởng lợi nhiều hơn từ việc tăng số lượng nhãn thô (coarse labels, vẽ nhanh) hay đầu tư thời gian vẽ nhãn cực kỳ chi tiết ở cấp độ từng cây (fine labels, vẽ kỹ)?
- **Hệ sinh thái:** Cảnh quan nông thôn - lâm nghiệp vùng Tây Nam nước Pháp (khu bảo tồn dài hạn LTSER "Vallées et Côteaux de Gascogne", cách Toulouse 60 km về phía nam; diện tích 172 km²). Cảnh quan gồm đất canh tác, đồng cỏ và các mảng rừng tư nhân xen kẽ.
- **Dữ liệu đầu vào:**
  - Ảnh hàng không đơn sắc (Grayscale panchromatic) số hóa lịch sử năm 1942 do Viện Địa lý và Rừng Quốc gia Pháp (IGN) cung cấp.
  - Độ phân giải không gian mặt đất (GSD): **0.75 m/pixel**.
  - Không có kênh màu RGB, không có mô hình độ cao DSM đi kèm (chỉ thuần túy mức xám 1 kênh).
  - Toàn bộ vùng nghiên cứu được chia thành **3,484 patch kích thước $256 \times 256$ pixels**.

---

## 2. Chi tiết Thiết kế Thực nghiệm & Phương pháp Phân đoạn

Tác giả thiết kế một quy trình thực nghiệm nghiêm ngặt gồm **51 lượt huấn luyện mô hình độc lập** với kích thước tập dữ liệu tăng dần từ 750 patch đến 3,250 patch (bước nhảy 50 patch/lượt), chạy song song trên 2 cấp độ chất lượng nhãn:

```
                  [ 3,484 Patch ảnh lịch sử 1942 (256x256 px, 0.75m) ]
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
  [ TẬP NHÃN THÔ (COARSE) ]                                           [ TẬP NHÃN MỊN (FINE) ]
  - Gán nhãn trên lưới 512x512 px                                     - Gán nhãn trên lưới 256x256 px
  - Thời gian: 4 – 8 phút/patch                                       - Thời gian: 10 – 40 phút/patch
  - Viền mượt, gộp cụm cây, bỏ sót cây nhỏ                           - Vẽ chi tiết từng cây đơn lẻ, cành nhánh
         │                                                                   │
         └─────────────────────────────────┬─────────────────────────────────┘
                                           │
                        [ GIAO THỨC HUẤN LUYỆN 51 VÒNG ]
                    Tăng kích thước từ 750 -> 3,250 patch (bước 50)
                      Cố định Random Seed để cùng Patch ID
                                           │
                                           ▼
                   ┌──────────────────────────────────────────────┐
                   │        KIẾN TRÚC MÔ HÌNH: scSE U-NET         │
                   │  - Encoder: MobileNetV2 (Pretrained)         │
                   │  - Khối chú ý: scSE (Spatial & Channel SE)   │
                   │  - Decoder: Transposed Convolutions + Skip   │
                   │  - Tiền xử lý: Tăng tương phản cục bộ CLAHE  │
                   │  - Hàm mất mát: Dice Loss                    │
                   │  - Optimizer: Adam (lr 0.001 -> decay 0.1)   │
                   └───────────────────────┬──────────────────────┘
                                           │
                                           ▼
                   [ Đánh giá trên 225 patch Test Set độc lập ]
                     (Phân rã theo 3 loại: Cây đơn lẻ, Hàng rào, Mảng rừng)
```

### 2.1. Hai bộ dữ liệu nhãn đối đầu (Coarse vs Fine Datasets)
1. **Bộ nhãn thô (Coarse Dataset):** Gán nhãn ở quy mô cửa sổ lớn $512 \times 512$ px. Người vẽ bo viền mượt xung quanh các cụm cây, gộp nhiều cây gần nhau thành một khối, bỏ qua các cây con đơn lẻ. Tốc độ: **4 – 8 phút/patch**.
2. **Bộ nhãn mịn (Fine Dataset):** Gán nhãn tỉ mỉ ở quy mô $256 \times 256$ px (đúng độ phân giải nạp vào mạng). Vẽ chi tiết ranh giới của từng cá thể cây riêng lẻ và hàng rào cây xanh. Tốc độ: **10 – 40 phút/patch** (tốn công gấp 3 đến 5 lần).

### 2.2. Kiến trúc Mạng: scSE Attention U-Net
- **Thư viện:** `segmentation_models_pytorch` (SMP).
- **Encoder:** **MobileNetV2** siêu nhẹ với trọng số tiền huấn luyện (ImageNet pre-trained weights) giúp hội tụ nhanh trên tập dữ liệu hạn chế.
- **Khối chú ý không gian và kênh đồng thời (Concurrent Spatial and Channel Squeeze & Excitation - scSE):**
  - Khối scSE (Roy et al., 2018) hiệu chuẩn lại đồng thời các kênh đặc trưng (*channel-wise*) và không gian (*spatial-wise*).
  - Tác dụng: Báo hiệu cho mạng biết cần tập trung vào vị trí không gian nào (*where*) và kênh đặc trưng nào (*what*), đặc biệt hiệu quả để định vị các đối tượng có kích thước siêu nhỏ như cây đơn lẻ trên ảnh xám mờ.
- **Hàm mất mát:** **Dice Loss**, trực tiếp tối ưu hóa độ trùng lặp hình học, chống lại sự mất cân bằng giữa nền đất trống và diện tích cây.
- **Tăng cường tương phản CLAHE:** Ảnh hàng không 1942 bị suy giảm chất lượng theo thời gian và có độ tương phản cực kỳ thấp. Tác giả áp dụng thuật toán cân bằng biểu đồ tần suất thích ứng cục bộ có giới hạn tương phản (**CLAHE**) khi huấn luyện.

---

## 3. Kết quả Thực nghiệm & Số liệu Định lượng

### 3.1. Điểm số Dice / F1-score trên toàn bộ tập kiểm thử (225 patch độc lập)

| Cấu hình mô hình | Số lượng patch Train | Điểm Dice nhỏ nhất | Điểm Dice lớn nhất | Điểm Dice trung vị (Median) |
|---|:---:|:---:|:---:|:---:|
| **Mô hình nhãn thô (Coarse Model)** | 750 $\to$ 3,250 | 66.57% | 69.96% | **68.05%** |
| **Mô hình nhãn mịn (Fine Model)** | 750 $\to$ 3,250 | 64.78% | **72.56%** | **69.11%** |

- Khi dùng toàn bộ dữ liệu (3,250 patch): Mô hình nhãn mịn đạt **70.70%**, mô hình nhãn thô đạt **69.96%**.

### 3.2. Ngưỡng bão hòa dữ liệu (Data Saturation Point)
- Đồ thị thực nghiệm chỉ ra rằng: Khi số lượng mẫu huấn luyện vượt qua ngưỡng ~2,550 patch (số này **không tìm thấy** trong văn bản bài; bài chỉ nêu khoảng 750–3,250 patch), việc bổ sung thêm dữ liệu **hầu như không làm tăng điểm số Dice nữa** (đường cong hiệu năng đi ngang và dao động nhiễu).

### 3.3. Phân rã hiệu năng theo kích thước & cấu hình không gian cây (Per-category Performance)

| Loại cấu hình cây trên ảnh | Coarse Model F1 (%) | Fine Model F1 (%) | Nhận xét chi tiết |
|---|:---:|:---:|---|
| **Mảng rừng lớn (Mixed / Forest fragments)** | ~78% | ~79% | Chênh lệch không đáng kể (nhãn thô hay mịn đều học tốt) |
| **Hàng rào cây xanh (Hedgerows)** | ~68% | **~73%** | Nhãn mịn vượt trội (+5% F1) |
| **Cây đơn lẻ (Individual Trees)** | ~42% | **~54%** | **Nhãn mịn áp đảo hoàn toàn (+12% F1)** |

---

## 4. Phân tích Điểm yếu & Lỗ hổng (Failure Modes)

1. **Độ nhạy cực đoan của metric Dice/IoU đối với vật thể nhỏ:** Với cây đơn lẻ (chỉ chiếm vài chục pixel), chỉ cần mô hình lệch 1–2 pixel ở đường viền thì điểm Dice tụt dốc thảm hại. Bài nêu giải thích tương tự: thiếu một pixel ở vật thể nhỏ ảnh hưởng nhiều hơn nhiều so với vật thể lớn. Phần "1–2 pixel", "vài chục pixel" và nhận xét mắt người phát hiện đúng vị trí là diễn giải thêm của người viết.
2. **Sự đánh đổi về mặt chi phí gán nhãn:** Nhãn mịn tốn từ 10 đến 40 phút/patch (tổng cộng hàng trăm giờ lao động chuyên gia), nhưng chỉ giúp tăng điểm Dice tổng thể từ 69.96% lên 70.70% (+0.74% trên toàn tập). Lợi ích của nhãn mịn chỉ thể hiện rõ ràng khi người dùng quan tâm đến cây nhỏ riêng lẻ.

---

## 5. Giá trị Tham khảo cho TreeFlowNet & Bài báo IGARSS 2027

> **Đã sửa 02/10/2026:** mục này từng chứa các khẳng định ưu việt của TreeFlowNet không có bằng chứng. Các số của bài gốc ở mục 1–4 đã được đối chiếu với toàn văn; mục 5 chỉ nêu điều bài cho thấy, tách khỏi giả thuyết. (Số đo của chính dự án: trên test CanopyRS, seed 42–44, Mask R-CNN R50 34.82 ± 0.67 AP, TreeFlowNet C5 R50 28.81 AP (một seed), TreeFlowNet Swin-T khoảng 30 AP; xem `docs/KIEN_TRUC_TREEFLOWNET_CLEAN_TOAN_HOC_VA_DIEM_YEU.md` mục 5 và 9. Các giả thuyết dưới đây chưa được kiểm chứng trên dữ liệu của mình.)

- **Điều bài thực sự cho thấy:** trên ảnh lịch sử 1942 đơn sắc, nhãn mịn cho Dice 70.70% so với 69.96% của nhãn thô khi dùng đủ 3,250 patch (chênh +0.74); cây đơn lẻ được cải thiện nhiều hơn (số theo từng nhóm trong bảng trên đọc từ Hình 6, chưa kiểm chứng bằng văn bản).
- **Giả thuyết, chưa kiểm chứng:** quy luật "nhãn mịn quan trọng với cây nhỏ" có thể áp dụng cho QuebecTrees (72.6% crown test là Tiny, < 9 m²). Bài này dùng ảnh khác hẳn (đơn sắc, 0.75 m, không phải 3 cm) và phân đoạn ngữ nghĩa, không phải instance.
- **Khối scSE và CLAHE:** là module nhẹ có thể thử; chưa thử trên TreeFlowNet.
