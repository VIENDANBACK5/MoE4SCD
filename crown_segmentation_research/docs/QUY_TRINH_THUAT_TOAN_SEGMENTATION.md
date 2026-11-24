# QUY TRÌNH THUẬT TOÁN PHÂN TÁCH TÁN CÂY CHI TIẾT (STEP-BY-STEP ITC DELINEATION PIPELINE)
**Mô hình:** TreeFlowNet (End-to-End Co-Supervised Flow-SDT Network)  
**Tác vụ:** Phân tách và khoanh vùng ranh giới từng cá thể tán cây (Individual Tree Crown Delineation - ITC) từ không ảnh độ phân giải siêu cao (VHR Aerial Imagery).

---

## 1. TỔNG QUAN LUỒNG DỮ LIỆU (DATAFLOW ARCHITECTURE)

Thuật toán của TreeFlowNet không dừng lại ở việc dự đoán một mặt nạ thô, mà là một **chuỗi tích hợp vật lý - toán học 9 giai đoạn** từ ảnh quang học RGB ban đầu đến các đa giác ranh giới từng cây hoàn chỉnh:

```
[Ảnh RGB Gốc (VHR 0.05m)]
          │
          ▼
[GIAI ĐOẠN 1: SimMIM SSL Backbone + Scale-Gated FPN P2-P5]
          │
          ├──────────────────────────────────────────────────────┐
          ▼                                                      ▼
[GIAI ĐOẠN 2: 4 Đầu ra Đa nhiệm]                       [Ràng buộc Toán học Sobolev]
  ├─ Head 1: Centripetal Flow Field v(x) ◄───────────────┤ L_sobolev = ||∇v - H(SDT)||_2
  ├─ Head 2: Signed Distance Transform (SDT) ─────────────┘
  ├─ Head 3: Centroid Apex Peak Heatmap
  └─ Head 4: Canopy Foreground Gate
          │
          ▼
[GIAI ĐOẠN 3: Lọc Tiền cảnh Quang học (Foreground Gating)]
  FG = (Canopy > 0.30) ∧ (SDT > -0.15)
          │
          ▼
[GIAI ĐOẠN 4: Tích phân Số Euler - Vectorized Particle Advection]
  x_{t+1} = x_t + η · v(x_t) / ||v(x_t)||  (Lặp 25-30 bước)
          │
          ▼
[GIAI ĐOẠN 5: Tích lũy Mật độ Điểm Tụ (Sink Accumulation) & Scale-Adaptive NMS (H22)]
  Bán kính bài trừ động: r_cand = r_min + Δr · (0.65·SDT + 0.35·Centroid)
  ──► Trích xuất Hạt nhân Tâm cây Thật sự (Apex Seeds)
          │
          ▼
[GIAI ĐOẠN 6: Khắc Rãnh Thủy văn Địa hình (Hydrological Stream-Burning H27)]
  Carve gorge walls: Cost = (1 - SDT) + 0.3·(1 - Canopy) - 0.4·∇_DiZenzo + 0.6·Gorge
          │
          ▼
[GIAI ĐOẠN 7: Phân vùng Thủy văn Có hạt giống (Marker-Controlled Watershed)]
  Ranh giới phân tách tự nhiên tại các đường phân thủy
          │
          ▼
[GIAI ĐOẠN 8: Tinh chỉnh Ranh giới Quang học Siêu vi (Guided Filter / MorphSnakes)]
  Khử bậc thang raster, uốn lượn theo mép lá thực tế
          │
          ▼
[KẾT QUẢ: Tập hợp Các Đa giác Cá thể Tán cây (ITC Instance Polygons & Masks)]
```

---

## 2. CHI TIẾT TỪNG BƯỚC THUẬT TOÁN

### BƯỚC 1: TIỀN XỬ LÝ & TRÍCH XUẤT ĐẶC TRƯNG ĐA TỈ LỆ (SCALE-GATED FPN)
- **Đầu vào:** Patch ảnh RGB kích thước $H \times W \times 3$ (chuẩn hóa ImageNet mean/std).
- **Backbone:** Swin Transformer hoặc ResNet-50 được pretrain trực tiếp bằng kỹ thuật **In-Domain SimMIM Self-Supervised Learning ($H29$)** trên hàng chục nghìn patch ảnh viễn thám độ phân giải siêu cao (GSD $0.05\text{ m}$).
- **Cơ chế Scale-Adaptive Attention Gating ($H28$):**
  - Khác với FPN thông thường chỉ cộng gộp đặc trưng các tầng, nhánh FPN được tích hợp cổng chú ý theo tỉ lệ:
  - Tầng $P_2, P_3$ (độ phân giải cao: $H/4, H/8$): tập trung dẫn truyền thông tin biên cạnh sắc nét và tán cây con tầng dưới (đường kính $2 - 4\text{ m}$).
  - Tầng $P_4, P_5$ (độ phân giải thấp: $H/16, H/32$): tổng hợp ngữ cảnh không gian diện rộng của các cây đại thụ tầng vượt tán (đường kính $15 - 25\text{ m}$).
  - $\implies$ Giải quyết triệt để lỗi "nuốt chửng cây nhỏ" (Understory Drowning) mà các mô hình flow y sinh gặp phải.

---

### BƯỚC 2: DỰ ĐOÁN ĐỒNG THỜI 4 ĐẦU RA ĐA NHIỆM (MULTI-TASK HEADS)
Mạng đồng thời suy luận ra 4 trường biểu diễn tương hỗ:

1. **Nhánh 1 - Trường Vector Dòng chảy Hướng tâm (Centripetal Flow Field):**
   $$\mathbf{v}(x, y) = (v_y, v_x) \in [-1, 1]^{2 \times H \times W}$$
   Mỗi pixel dự đoán một vector 2 chiều chuẩn hóa trỏ trực tiếp về đỉnh cao nhất (chóp tán) của cá thể cây chứa pixel đó.
2. **Nhánh 2 - Hàm Khoảng cách Có dấu (Signed Distance Transform - SDT):**
   $$S(x, y) \in [-1, 1]^{1 \times H \times W}$$
   Bề mặt địa hình liên tục: đạt giá trị cực đại $+1.0$ tại tâm cây và giảm dần về $0.0$ tại ranh giới tán cây, nhận giá trị âm bên ngoài tán.
3. **Nhánh 3 - Bản đồ Xác suất Đỉnh tâm Tán (Centroid Peak Heatmap):**
   $$C(x, y) \in [0, 1]^{1 \times H \times W}$$
   Phân bố Gaussian 2D sắc nhọn tập trung tại tọa độ chóp cây cao nhất.
4. **Nhánh 4 - Mặt nạ Tán rừng Ngữ nghĩa (Canopy Semantic Mask):**
   $$K(x, y) \in [0, 1]^{1 \times H \times W}$$
   Phân loại pixel nhị phân: Tán thực vật quang học ($1$) vs Đất trống / Bóng đổ / Nền cỏ ($0$).

---

### BƯỚC 3: RÀNG BUỘC VI PHÂN SOBOLEV (SOBOLEV GRADIENT COUPLING)
Thay vì huấn luyện Flow Field và SDT độc lập (dẫn tới mâu thuẫn gradient tại các vùng tiếp giáp mép tán), TreeFlowNet áp đặt ràng buộc không gian vi phân:
$$\mathcal{L}_{\text{Sobolev}} = \|\nabla \mathbf{v} - \mathcal{H}(S)\|_2 + \lambda_{\text{cos}} (1 - \cos(\mathbf{v}, \nabla S))$$
- $\nabla \mathbf{v}$: Tensor Jacobi đạo hàm không gian của trường vector.
- $\mathcal{H}(S)$: Ma trận Hessian (đạo hàm bậc 2) của bề mặt khoảng cách SDT.
- **Ý nghĩa vật lý:** Ép trường vector phải là gradient chuẩn tắc của bề mặt thế năng khoảng cách. Nhờ đó, vector dòng chảy triệt tiêu mượt mà khi tiến về tâm và đảo chiều dốc đứng tại mép tán, loại bỏ hoàn toàn các vector nhiễu loạn ngẫu nhiên.

---

### BƯỚC 4: LỌC MẶT NẠ TIỀN CẢNH QUANG HỌC (FOREGROUND GATING)
Trước khi giải tích phân, các vùng nền đất trống, mặt đường và khoảng trống giữa các tán được loại trừ thông qua điều kiện lọc kép:
$$\mathcal{FG}(x, y) = \left(K(x, y) > \tau_{\text{canopy}}\right) \land \left(S(x, y) > \tau_{\text{sdt}}\right)$$
- $\tau_{\text{canopy}} = 0.30$: Ngưỡng tin cậy diệp lục quang học.
- $\tau_{\text{sdt}} = -0.15$: Ngưỡng khoảng cách biên (chỉ giữ các pixel nằm bên trong hoặc sát mép ngoài của tán cây).
- *Tác dụng:* Giảm $70\%$ khối lượng tính toán tích phân và triệt tiêu 100% các dương tính giả ngoài đất trống.

---

### BƯỚC 5: TÍCH PHÂN SỐ EULER ĐA HẠT (VECTORIZED EULER ADVECTION)
Toàn bộ $N$ pixel thuộc tập tiền cảnh $\mathcal{FG}$ được xem như các chất điểm (particles) chuyển động trên trường vector $\mathbf{v}$:
- Khởi tạo vị trí ban đầu: $\mathbf{x}_0 = (y_0, x_0)$ với mọi pixel $(y_0, x_0) \in \mathcal{FG}$.
- Cập nhật vị trí sau từng bước thời gian qua phép nội suy song tuyến tính (Bilinear Grid Sampling trên GPU):
  $$\mathbf{x}_{t+1} = \mathbf{x}_t + \eta \cdot \frac{\mathbf{v}(\mathbf{x}_t)}{\|\mathbf{v}(\mathbf{x}_t)\|_2 + \epsilon}$$
  với bước di chuyển $\eta = 1.0 - 1.2$ pixel và số bước lặp $T = 25 - 30$.
- **Cơ chế tụ dòng:** Dù bắt đầu ở bất kỳ vị trí nào trong tán cây, dòng vector hướng tâm sẽ cuốn tất cả các pixel tụ về một điểm cực tiểu thế năng duy nhất ở chóp tán — gọi là **Topological Sink**.

---

### BƯỚC 6: KHỬ CỰC ĐẠI ĐA TỈ LỆ THEO THƯỚC ĐO SINH HỌC (SCALE-ADAPTIVE SINK NMS $H22$)
Sau khi tích phân Euler kết thúc:
1. Tạo bản đồ mật độ điểm tụ: $\mathcal{D}(y, x) = \sum_{i} \delta(\mathbf{x}_T^{(i)} - (y, x))$.
2. Làm mịn bằng bộ lọc Gaussian 2D ($\sigma = 2.5$) để gộp các điểm tụ lân cận.
3. Nhân trọng số sinh học: $\mathcal{A}(y, x) = \mathcal{D}_{\text{smooth}} \cdot \left(1 + 2 C(y, x) + \max(0, S(y, x))\right)$.
4. **Vấn đề phân cành đa đỉnh (Multi-leader Branching):** Một cây to có thể sinh ra 2–3 đỉnh tụ dòng cục bộ.
5. **Giải pháp Scale-Adaptive Allometric NMS ($H22$):**
   Thay vì dùng bán kính triệt tiêu cố định, bán kính loại trừ $r_{\text{cand}}$ được tính thích ứng theo kích thước thực tế của từng cây:
   $$r_{\text{cand}} = r_{\min} + (r_{\max} - r_{\min}) \cdot \left(0.65 \cdot S(y, x) + 0.35 \cdot C(y, x)\right)$$
   với $r_{\min} = 16\text{ px}$ (cây con nhỏ) và $r_{\max} = 52\text{ px}$ (cây đại thụ).
   $\implies$ Triệt tiêu hoàn toàn các đỉnh giả trên cành cây lớn mà không làm mất cây nhỏ đứng cạnh!

---

### BƯỚC 7: KHẮC RÃNH THỦY VĂN ĐỊA HÌNH (HYDROLOGICAL STREAM-BURNING $H27$)
Đây là đột phá kỹ thuật then chốt giải quyết hiện tượng dính chùm tán:
- Tại tiếp điểm giữa 2 tán cây tiếp xúc, bề mặt khoảng cách SDT xuất hiện một điểm yên ngựa phẳng khiến tích phân dễ bị rò rỉ.
- TreeFlowNet thiết lập **Bề mặt Chi phí Địa hình (Topographic Cost Surface)** kết hợp cơ chế Stream-Burning:
  $$\mathcal{C}(y, x) = (1 - S(y, x)) + 0.3 \cdot (1 - K(y, x)) - 0.4 \cdot \nabla_{\text{DiZenzo}}(y, x) + 0.6 \cdot \mathcal{G}_{\text{stream}}(y, x)$$
  - $\nabla_{\text{DiZenzo}}$: Gradient tensor màu đa phổ của không ảnh quang học.
  - $\mathcal{G}_{\text{stream}}$: Hào ngăn cách nhân tạo khoét sâu tại ranh giới tiếp xúc giữa 2 cây.
  - $\implies$ Biến điểm yên ngựa thành bờ vực dốc đứng, ép các phân vùng nước không thể tràn qua nhau.

---

### BƯỚC 8: PHÂN VÙNG THỦY VĂN MARKER-CONTROLLED WATERSHED
- Đặt các hạt nhân tâm cây thu được từ Bước 6 làm các giếng thoát nước (Markers):
  $$M(y_k, x_k) = k, \quad k = 1, 2, \dots, K$$
- Thực thi giải thuật loang thủy văn Marker Watershed trên bề mặt chi phí $\mathcal{C}$ bị giới hạn bởi mặt nạ tiền cảnh $\mathcal{FG}$.
- Dòng nước dâng lên từ các hạt nhân và gặp nhau chính xác tại các rãnh phân thủy do cơ chế Stream-Burning tạo ra, phân chia không gian thành $K$ vùng độc lập $\mathcal{R}_1, \mathcal{R}_2, \dots, \mathcal{R}_K$.

---

### BƯỚC 9: TINH CHỈNH RANH GIỚI QUANG HỌC SIÊU VI (BOUNDARY REFINEMENT)
- Ranh giới ban đầu từ Watershed ở độ phân giải lưới pixel thường xuất hiện hiện tượng răng cưa bậc thang (raster staircases).
- Mỗi mặt nạ vùng $\mathcal{R}_k$ được đưa qua bộ lọc biên **Guided Filter quang học kết hợp biến dạng chủ động (Morphological Snakes)**:
  - Sử dụng ảnh RGB gốc $I$ làm ảnh dẫn đường (guide image) với bán kính $r = 4$, tham số điều hòa $\epsilon = 10^{-3}$.
  - Mặt nạ tự động co giãn và bám chặt vào các viền cong tự nhiên của phiến lá và kẽ cành.
- **Kết quả đầu ra:** Tập hợp các đối tượng cây cá thể hoàn chỉnh:
  $$\mathcal{O}_k = \left\{\text{Mask}_k, \; \text{Score}_k, \; \text{Centroid}_k, \; \text{BoundingBox}_k, \; \text{Area}_k, \; \text{PolygonContour}_k\right\}$$
  sẵn sàng xuất định dạng chuẩn GIS (GeoJSON) hoặc COCO Instance Segmentation.

---

## 3. BẢNG TỔNG HỢP CÁC THAM SỐ THUẬT TOÁN (HYPERPARAMETER SPECIFICATION)

| Tham số | Ký hiệu | Giá trị Khuyến nghị | Ý nghĩa & Vai trò Vật lý |
| :--- | :---: | :---: | :--- |
| Ngưỡng Tiền cảnh Tán | $\tau_{\text{canopy}}$ | `0.30` | Lọc bỏ đất trống, bóng râm và cỏ nền |
| Ngưỡng Khoảng cách SDT | $\tau_{\text{sdt}}$ | `-0.15` | Giữ lại các pixel thuộc hoặc sát mép ngoài tán cây |
| Bước Tích phân Euler | $\eta$ | `1.0 - 1.2` | Bước nhảy advection của từng chất điểm pixel |
| Số Bước Lặp Euler | $T$ | `25 - 30` | Đảm bảo 100% pixel hội tụ về bồn chóp tán |
| Bán kính Cực tiểu Cây con | $r_{\min}$ | `16 px` (0.8m) | Giữ lại cây con đường kính nhỏ tầng dưới |
| Bán kính Cực đại Cây lớn | $r_{\max}$ | `52 px` (2.6m) | Triệt tiêu phân cành đa đỉnh trên cây cổ thụ |
| Trọng số Khắc rãnh | $w_{\text{gorge}}$ | `0.5 - 0.6` | Độ sâu hào phân cách nhân tạo tại ranh giới chạm |
| Bán kính Lọc Dẫn đường | $r_{\text{guided}}$ | `4 px` | Bán kính làm mịn viền lá quang học |

---
*Tài liệu kỹ thuật đi kèm phục vụ buổi họp:*
- File trình chiếu LaTeX Beamer: `meeting/week1/slides_segmentation_pipeline.tex`
- Bộ ảnh trực quan hóa từng bước: `meeting/week1/images/step_by_step_segmentation_pipeline.png`
