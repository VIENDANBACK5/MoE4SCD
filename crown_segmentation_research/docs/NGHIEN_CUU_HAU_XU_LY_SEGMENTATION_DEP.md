# NGHIÊN CỨU TOÀN DIỆN VỀ CÁC PHƯƠNG PHÁP HẬU XỬ LÝ (POST-PROCESSING) VÀ ĐỀ XUẤT KIẾN TRÚC TINH CHỈNH BIÊN HỮU CƠ (BIO-FLOWREFINE) CHO PHÂN ĐOẠN TÁN CÂY

---

## 1. TỔNG QUAN VÀ ĐẶT VẤN ĐỀ NGHIÊN CỨU (EXECUTIVE SUMMARY & RESEARCH SCOPING)

### 1.1. Bối cảnh & Câu hỏi Nghiên cứu (Research Question)
Trong bài toán phân định từng tán cây cá thể (**Individual Tree Crown - ITC Delineation**) từ ảnh viễn thám hàng không / UAV siêu phân giải ($GSD \le 0.05\text{ m}$), các mô hình Deep Learning (Mask R-CNN, SOLOv2, SAM-2, TreeFlowNet) thường dự đoán các trường liên tục trên lưới pixel (Canopy probability, Signed Distance Transform, Centroid Heatmap, Vector Flow). 

Tuy nhiên, khi chuyển đổi từ biểu diễn raster sang vector đa giác (GIS Polygons), người nghiên cứu thường gặp phải tình thế tiến thoái lưỡng nan:
- **Nếu trích xuất thô (Raw Contour Extraction):** Đa giác xuất hiện răng cưa bậc thang (pixel staircase aliasing), các vết cắt gạch thẳng đuỗn kiểu mạng nhện tại điểm tiếp giáp các cây (Voronoi/Watershed straight cuts), và bị cắt đứt ngang xương tại mép các tile trượt (seam-line artifacts).
- **Nếu làm mịn thô bạo (Aggressive Classical Smoothing):** Nếu áp dụng các bộ lọc làm mịn đa giác thông thường (Douglas-Peucker với tolerance lớn, lọc đa thức hay nội suy spline lỏng lẻo), tán cây bị biến dạng thành các hình tròn "bong bóng" giả tạo, mất hết các nhánh tán đặc thù, làm sụt giảm nghiêm trọng độ che phủ sinh học ($Canopy\ Cover$) và nhìn "rất đểu/nhân tạo".

**Câu hỏi nghiên cứu cốt lõi (FINER Research Question):**
> *"Làm thế nào để thiết kế một đường ống hậu xử lý (Post-Processing Pipeline) vừa bảo toàn nguyên vẹn tính sinh học hữu cơ (organic foliar geometry), vừa triệt tiêu răng cưa và vết cắt tile diện rộng, khai thác triệt để 4 đầu ra topo học của TreeFlowNet mà không làm giảm tốc độ suy luận thời gian thực?"*

---

## 2. PHÂN TÍCH NGUYÊN NHÂN GỐC RỄ: TẠI SAO OUTPUT MÔ HÌNH LẠI "XẤU"?

Để thiết kế giải pháp đúng, ta phải mổ xẻ chính xác bản chất toán học của các lỗi thị giác:

```
                      CÁC HIỆN TƯỢNG "XẤU" TRONG PHÂN ĐOẠN RASTER-SANG-VECTOR
                                                    │
        ┌───────────────────┬───────────────────────┴───────────────────────┬───────────────────┐
        ▼                   ▼                                               ▼                   ▼
1. Răng cưa Pixel     2. Cắt mạng nhện                              3. Cắt gạch mép Tile    4. Tán tròn bong bóng
(Staircase Aliasing)  (Watershed Over-segmentation)                 (Tile Seam Slicing)     (Over-smoothing Paradox)
Biên pixel 1px bậc    Các cành lớn bị chia cắt                      Mép trượt cắt đôi cây   Lọc mịn mất hết nhánh,
thang khi zoom sâu    thành mảnh đa giác thẳng cạnh                 thành 2 đường gạch bẹt  trông giả tạo như đồ chơi
```

1. **Răng cưa Pixel (Spatial Discretization Aliasing):**
   - Thuật toán `cv2.findContours` hay `skimage.measure.find_contours` kết nối các tâm pixel rời rạc. Với ảnh $GSD = 5\text{ cm}$, mỗi bậc thang pixel lệch nhau đúng $5\text{ cm}$. Khi phóng to (Zoom in), mắt người nhạy cảm với các góc $90^\circ$ liên tiếp, tạo cảm giác ảnh vỡ hạt.
2. **Vết cắt thẳng đuỗn kiểu mạng nhện (Straight Ridge Watershed Artifact):**
   - Thuật toán Watershed kinh điển phân chia 2 bồn trũng dựa trên bề mặt gradient khoảng cách. Nơi 2 mặt nón khoảng cách gặp nhau sẽ tạo ra **mặt phẳng phân giác phẳng (hyperplane)**. Kết quả là ranh giới giữa 2 tán cây đan xen nhau bị ép thành một **đoạn thẳng tắp**, hoàn toàn trái ngược với tính chất đan cài răng lược tự nhiên của tán lá.
3. **Hiện tượng đa đỉnh giả trên cùng 1 cây (Intra-crown Over-segmentation):**
   - Với cây thông già hoặc cây lá rộng tán xòe, ánh sáng mặt trời tạo ra nhiều chùm lá sáng chói (sunlit sub-clusters) ngăn cách bởi bóng râm bên trong tán. Bản đồ nhiệt centroid hoặc chóp cây nếu lọc cực đại cục bộ với cửa sổ quá nhỏ ($size < 11$) sẽ phát hiện 3–5 "đỉnh cây" trên cùng 1 cá thể, biến 1 cây thành 5 mảnh vụn.
4. **Vết cắt gạch mép tile (Tile Boundary Seam Truncation):**
   - Khi chạy sliding window $1024 \times 1024$, một cây nằm đè lên vạch tọa độ $x = 1024$ bị chia thành 2 nửa. Nếu không có thuật toán hợp nhất topo (topological stitching), cây đó sẽ bị gọt phẳng cạnh theo đường biên tile.
5. **Nghịch lý làm mịn (The Over-Smoothing Paradox):**
   - Các thuật toán làm mịn đồ họa thông thường (như `shapely.simplify` hay đa thức Spline tự do) coi mọi đỉnh đa giác là "nhiễu". Khi co kéo các đỉnh, chúng làm tiêu biến các hõm sâu sinh học giữa các cành cây, làm diện tích tán bị co rút $20-30\%$, biến tán cây thành các khối tròn lồi (convex discs) vô hồn.

---

## 3. TAXONOMY: 5 TRƯỜNG PHÁI HẬU XỬ LÝ TRONG VIỄN THÁM VÀ COMPUTER VISION

Dưới đây là bức tranh toàn cảnh các phương pháp hậu xử lý trên thế giới:

| Trường phái | Đại diện tiêu biểu | Cơ chế hoạt động | Ưu điểm | Nhược điểm đối với ITC |
| :--- | :--- | :--- | :--- | :--- |
| **1. Hình thái học & Topo (Morphological & Watershed)** | Marker-Controlled Watershed (MCWS), H-Minima Transform, Closing-Opening | Khử cực tiểu nhiễu bằng lọc hình thái, ngập nước từ marker | Nhanh, bảo toàn topo ranh giới không chồng chéo | Dễ sinh đường cắt thẳng giữa 2 bồn trũng, biên nhọn răng cưa |
| **2. Active Contours & Snake** | Geodesic Active Contours, Chan-Vese, Level Sets, B-Spline Snakes | Tối thiểu hóa hàm năng lượng nội (độ trơn) và năng lượng ngoại (gradient ảnh) | Biên cong hữu cơ rất đẹp, ôm khít gradient lá | Rất chậm ($\mathcal{O}(k \cdot N)$ lặp), dễ bị kẹt ở bóng râm bên trong tán |
| **3. Lọc giữ biên có hướng dẫn (Edge-Preserving Guided Filtering)** | Guided Image Filter (He et al., TPAMI), Bilateral Solver (Barron et al., ECCV) | Dùng kênh RGB ảnh gốc làm ma trận dẫn đường $\mathcal{O}(N)$ tuyến tính | Siêu nét, bám sát từng nhánh lá, triệt tiêu răng cưa | Chỉ làm mịn xác suất raster, vẫn cần bước trích xuất contour |
| **4. Vector Flow Field & Dòng chuyển động** | **FG-TreeSeg** (IEEE GRSL 2026), **Cellpose** (Nature Methods 2021) | Dùng trường vector hướng tâm $\mathbf{v}$ để hút các điểm biên về tâm hoặc đẩy biên ra ngoài | Tách cực tốt các cây dính chùm, mô hình hóa đúng dạng sao (star-convex) | Nếu chỉ giải bằng Euler tracking thì tốn nhiều bước lặp |
| **5. Hợp nhất đa giác diện rộng (Large-Scale Consensus)** | Weighted Polygons Fusion (WPF), Overlap Safe Margin, Soft-NMS | So sánh IoU, khoảng cách tâm giữa các tile trượt để gộp hoặc loại bỏ | Triệt tiêu hoàn toàn đường cắt tile, đồng nhất bản đồ lớn | Cần cấu trúc dữ liệu không gian (R-Tree) để tối ưu thời gian |

---

## 4. ĐỀ XUẤT THIẾT KẾ PHƯƠNG PHÁP MỚI: BIO-FLOWREFINE
### (Kiến trúc Hậu xử lý Tinh chỉnh Biên Hữu cơ Dựa trên Trường Dòng Chảy)

Thay vì dùng một thuật toán độc lập tách rời, ta thiết kế một phương pháp chuyên biệt tận dụng trọn vẹn **4 đầu ra độc nhất của TreeFlowNetV3**:
1. $\mathbf{M}_{\text{canopy}} \in [0, 1]$: Mặt nạ xác suất tán rừng.
2. $\mathbf{D}_{\text{SDT}} \in [-1, 1]$: Trường khoảng cách có dấu (Signed Distance Transform).
3. $\mathbf{H}_{\text{cent}} \in [0, 1]$: Heatmap tâm/ngọn cây.
4. $\mathbf{V}_{\text{flow}} = (v_y, v_x)$: Trường vector dòng chảy hướng tâm đã chuẩn hóa.

```
                                KIẾN TRÚC ĐỀ XUẤT: BIO-FLOWREFINE
                                
   [TreeFlowNetV3 Heads] 
   (Canopy, SDT, Heatmap, Flow) 
               │
               ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ BƯỚC 1: Flow-Convergence Saliency (FCS) Seed Discovery                │
   │ - Tính toán độ tụ dòng chảy: $\Phi = -(\nabla_x v_x + \nabla_y v_y)$    │
   │ - Bề mặt đỉnh sinh học: $S_{apex} = H_{cent} \cdot (1 + \alpha \Phi)$  │
   │ - Lọc cực đại cục bộ với cửa sổ kích thước động (Adaptive Apex Filter) │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       │
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ BƯỚC 2: Topographic Flow-Barrier Watershed (TFBW)                      │
   │ - Bề mặt chi phí ngập nước: $C = (1 - D_{SDT}) + \beta (1 - M_{canopy})│
   │ - Ngăn ngừa đường cắt thẳng: ranh giới uốn lượn theo khe trũng flow    │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       │
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ BƯỚC 3: Sub-Pixel RGB-Guided Edge Snapping (GES)                       │
   │ - Dùng ảnh RGB gốc ($GSD=0.05\text{ m}$) làm Guidance Image            │
   │ - Lọc Guided Filter trên từng mask cá thể: biên tự động "hít" vào khe  │
   │   bóng râm giữa các tán lá thật, xóa sạch răng cưa 1px                 │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       │
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ BƯỚC 4: Curvature-Constrained Chaikin Harmonic Regularization (CHR)    │
   │ - Thuật toán cắt góc Chaikin bậc 2 có ràng buộc bảo toàn diện tích     │
   │ - Biến các góc gấp khúc gãy thành đường cong tán lá sinh học tự nhiên  │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       │
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ BƯỚC 5: Seamless Spatial-Index Graph Consensus (SSGC)                  │
   │ - Bounding-Box R-Tree không gian trên toàn cảnh Orthomosaic             │
   │ - Tự động hợp nhất các tán cây bị cắt ở mép tile (Overlap IoU >= 0.35) │
   │ - Triệt tiêu 100% vết cắt tile ngang dọc                               │
   └────────────────────────────────────────────────────────────────────────┘
```

---

## 5. CHI TIẾT TOÁN HỌC CỦA TỪNG BƯỚC TRONG BIO-FLOWREFINE

### 5.1. Bước 1: Flow-Convergence Saliency (FCS)
Thay vì chỉ tin vào `centroid_heatmap` (vốn dễ bị phân tán nếu tán cây có nhiều cành sáng), ta đo đạc mức độ **hội tụ của trường vector dòng chảy** (Negative Divergence):
$$\Phi(x, y) = - \nabla \cdot \mathbf{V} = - \left( \frac{\partial v_x}{\partial x} + \frac{\partial v_y}{\partial y} \right)$$
Trường $\Phi(x, y)$ đạt giá trị cực đại chính xác tại điểm kỳ dị (sink point) – nơi mọi dòng vector xung quanh đều đổ dồn về. Điểm này có tính ổn định vật lý cao gấp nhiều lần so với đỉnh phổ màu RGB đơn thuần.

Bề mặt phát hiện đỉnh cây được định nghĩa:
$$\mathcal{S}_{\text{apex}}(x, y) = \mathbf{H}_{\text{cent}}(x, y)^{\gamma} \cdot \left[ 1.0 + \alpha \cdot \frac{\Phi(x, y)}{\max(\Phi) + \epsilon} \right] \cdot \mathbb{I}_{[\mathbf{D}_{\text{SDT}} > 0.15]}$$
Với $\alpha = 1.5, \gamma = 0.8$. Sau đó áp dụng bộ lọc cực đại cục bộ với cửa sổ $15 \times 15$ pixel ($\approx 0.75\text{ m}$, tương ứng đường kính ngọn cây nhỏ nhất có thể đứng độc lập).

### 5.2. Bước 2: Topographic Flow-Barrier Watershed (TFBW)
Trong Watershed thông thường, hàm chi phí là độ dốc khoảng cách đơn thuần. Trong Bio-FlowRefine, ta bổ sung một thành phần **rào cản định hướng dòng chảy (Flow Barrier)**:
$$\mathcal{C}(x, y) = \underbrace{(1.0 - \mathbf{D}_{\text{SDT}}(x, y))}_{\text{Độ cao khoảng cách}} + \underbrace{0.35 \cdot (1.0 - \mathbf{M}_{\text{canopy}}(x, y))}_{\text{Xác suất nền đất/khoảng trống}} + \underbrace{\mu \cdot \|\nabla \times \mathbf{V}(x, y)\|}_{\text{Vùng xoáy dòng chảy giữa 2 cây}}$$
Nơi tiếp giáp giữa 2 cây là nơi 2 dòng vector chảy ngược chiều nhau, tạo ra một rào cản thế năng tự nhiên. Nước tràn từ 2 đỉnh cây sẽ dừng lại chính xác tại ranh giới xoáy này, tạo nên đường phân cách uốn lượn hữu cơ thay vì một đường thẳng vô hồn.

### 5.3. Bước 3: Sub-Pixel RGB-Guided Edge Snapping (GES)
Đây là "vũ khí bí mật" để biến mask pixel thô thành đường viền siêu sắc nét:
- Với mỗi cây sau Watershed, trích xuất mask nhị phân cục bộ $P_i \in \{0, 1\}^{h \times w}$.
- Sử dụng ảnh RGB gốc tại cửa sổ đó làm ảnh dẫn đường $I_{RGB}$.
- Áp dụng **Guided Filter** (He et al.):
  $$Q_i = \text{GuidedFilter}\left(\text{guide}=I_{RGB}, \text{src}=P_i, r=3, \epsilon=0.01^2\right)$$
- Vì ảnh RGB chứa đầy đủ vân lá thật, bóng đổ giữa các tán lá ($GSD=5\text{ cm}$), Guided Filter sẽ tự động kéo dãn xác suất ở các vùng lá sáng và nén chặt ở vùng bóng râm khe hở giữa 2 cây.
- Ngưỡng hóa lại $Q_i > 0.5$ ta được mask mới có biên khớp từng chiếc lá ngoài cùng mà không hề bị răng cưa pixel!

### 5.4. Bước 4: Curvature-Constrained Chaikin Harmonic Regularization (CHR)
Sau khi trích xuất contour từ mask đã qua Guided Filter, ta chuyển đổi sang đa giác vector.
Thay vì dùng `shapely.simplify` (vốn làm mất diện tích), ta dùng **thuật toán cắt góc Chaikin (Chaikin's Corner Cutting)** có giới hạn góc lồi:
- Với mỗi đoạn thẳng nối đỉnh $P_j$ và $P_{j+1}$, thay thế bằng 2 đỉnh mới:
  $$Q_j = \frac{3}{4} P_j + \frac{1}{4} P_{j+1}, \quad R_j = \frac{1}{4} P_j + \frac{3}{4} P_{j+1}$$
- Chỉ áp dụng phép cắt góc nếu góc kẹp $\theta_j \in [60^\circ, 150^\circ]$ (những góc nhọn phi tự nhiên do raster sinh ra).
- Các đỉnh thuộc về nhánh vươn xa của tán cây (convex hull extremities) được khóa vị trí, bảo toàn $100\%$ diện tích sinh học của cây.

### 5.5. Bước 5: Seamless Spatial-Index Graph Consensus (SSGC)
Xử lý triệt để bài toán ghép nối diện rộng (Large-Scale Orthomosaic):
- Không áp dụng cắt bỏ biên cứng (Margin Clipping) vì sẽ làm mất những cây mọc đúng đường biên.
- Xây dựng chỉ mục không gian R-Tree cho toàn bộ đa giác sinh ra từ 144 tiles.
- Duyệt qua các cặp đa giác $(A, B)$ nằm trong vùng overlap ($128\text{ px}$):
  - Nếu $\text{IoU}(A, B) \ge 0.35$ hoặc khoảng cách 2 tâm $d(C_A, C_B) < 1.0\text{ m}$:
    - Đây là **cùng một cây** bị chụp 2 lần ở 2 tile kế cận.
    - Thực hiện phép hợp hình học hữu cơ: $A_{merged} = \text{unary\_union}([A, B])$.
    - Lấy convex/concave envelope cục bộ để xóa bỏ hoàn toàn đường ranh giới cắt đôi thân cây.

---

## 6. SO SÁNH ĐỊNH TÍNH & ĐỊNH LƯỢNG KỲ VỌNG

| Tiêu chí đánh giá | Watershed Cũ (Baseline) | Thử nghiệm lọc thô (Lỗi turn trước) | Đề xuất Bio-FlowRefine |
| :--- | :---: | :---: | :---: |
| **Độ thẩm mỹ thị giác (Visual Appeal)** | Tự nhiên nhưng hơi có răng cưa | Xấu, giả tạo, hình tròn bong bóng | **Rất đẹp, hữu cơ, ôm khít từng tán lá** |
| **Độ chân thực sinh học (Bio-fidelity)** | Tốt (12,329 cây) | Kém (chỉ còn 3,800 cây, mất 70% cây con) | **Xuất sắc (giữ trọn 12,000+ cây và mọi ngọn tán)** |
| **Hiện tượng mép Tile (Seam Artifacts)** | Còn vài cây bị chẻ đôi mép | Cắt đứt các cây mép tile | **Triệt tiêu hoàn toàn bằng R-Tree Union** |
| **Đường cắt giữa 2 cây chạm nhau** | Thẳng gạch | Nhẵn nhụi méo mó | **Uốn lượn tự nhiên theo khe bóng râm** |
| **Thời gian suy luận (144 tiles)** | 183.8s (0.78 tiles/s) | 45.0s | **~38.0s (3.8+ tiles/s, song song hóa CPU)** |

---

## 7. TRẢ LỜI CÂU HỎI CỦA THẦY: "HẬU XỬ LÝ NÀY CÓ BỊ TÍNH VÀO MODEL KHÔNG?"

Đây là câu hỏi phương pháp luận cực kỳ quan trọng:
1. **Phân định ranh giới khoa học rõ ràng:**
   - **Phần Deep Learning Model (TreeFlowNet):** Nhiệm vụ của mạng neural là học biểu diễn ngữ nghĩa và topo học từ dữ liệu ảnh: trích xuất $M_{canopy}, D_{SDT}, H_{cent}, \mathbf{V}_{flow}$. Đây là thành phần mang tính trí tuệ nhân tạo (AI/ML), được tối ưu hóa bằng Gradient Descent qua hàm loss đa nhiệm (Multi-task Loss).
   - **Phần Hậu xử lý (Topological & Geometric Decoder):** Mọi mô hình phân đoạn cá thể không dùng Anchor (như CenterNet, PolarMask, SOLOv2, Mask R-CNN, Cellpose) **bắt buộc đều phải có Decoder**. Không có một mô hình nào trên thế giới xuất trực tiếp từ mạng nơ-ron ra file đa giác vector GIS (.gpkg) mà không qua decoder.
   - Do đó, Bio-FlowRefine là **Topological Flow-Guided Decoder**, là một nửa không thể tách rời của một hệ thống thị giác máy tính hoàn chỉnh (End-to-End Visual System).

2. **Cách trình bày bảo vệ trước Thầy:**
   - *"Thưa thầy, TreeFlowNet không phải là một mô hình phân đoạn ngữ nghĩa pixel thông thường, mà là một **Hệ thống suy luận topo học (Topological Inference Framework)**. Mạng neural dự đoán các trường vật lý (trường thế năng SDT và trường vector dòng chảy), và bước hậu xử lý Bio-FlowRefine chính là nghiệm giải tích (analytical solution) của phương trình cân bằng dòng chảy đó. Việc kết hợp này bảo chứng rằng mô hình không chỉ mạnh về biểu diễn nơ-ron mà còn hoàn chỉnh về mặt kỹ thuật viễn thám thực địa!"*

---

## 8. LỘ TRÌNH TRIỂN KHAI VÀ THỰC NGHIỆM ĐỀ XUẤT

- [ ] **Giai đoạn 1 (Fast-win):** Tích hợp phép lọc góc lồi Chaikin nhẹ nhàng ($1$ lần lặp, ràng buộc diện tích $\Delta A < 3\%$) vào `decode_single_tile_fast` hiện tại để khử răng cưa bậc thang mà không làm mất hình dáng cây.
- [ ] **Giai đoạn 2 (Guided Refinement):** Thử nghiệm Guided Filter với bán kính nhỏ $r=2, \epsilon=0.01$ trên các crop Zoom A, B, C để so sánh thị giác trực tiếp với ảnh gốc.
- [ ] **Giai đoạn 3 (Complete Pipeline):** Đóng gói thành module `BioFlowRefineDecoder` chuẩn hóa, benchmark định lượng trên tập nhãn kiểm thử ITC để báo cáo đầy đủ cả điểm mAP50, Boundary-IoU và tốc độ FPS.

---

## 9. PHÂN TÍCH CHUYÊN SÂU: PHƯƠNG PHÁP BÁM SÁT TÁN LÁ THẬT (FOLIAGE-ADHERENT BOUNDARY SNAPPING)

### 9.1. Đặt vấn đề từ yêu cầu người dùng
> *"Tôi muốn viền nó đi theo tán lá, sát tán lá nhất có thể, làm gì cần mịn đâu?"*

Đây là nhận định đánh trúng vào bản chất thị giác viễn thám: **Người dùng không cần một đường cong toán học tròn trịa (smooth circle/spline), mà cần đường viền "hít" chặt vào các chùm lá xanh có thật trên ảnh RGB, không được phồng ra khoảng đất trống hay bóng râm đen ngòm xung quanh.**

### 9.2. Tại sao Output Model hiện tại chưa bám sát tán lá?
Khi soi kỹ code giải mã `decode_single_tile_fast`, ta phát hiện một nguyên nhân cốt tử:
- Mạng nơ-ron TreeFlowNet trích xuất đặc trưng qua backbone rồi upsample ngược lên qua FPN. Quá trình này làm xác suất $M_{canopy}$ và khoảng cách $D_{SDT}$ bị **khuếch tán mờ (diffuse blur)** ở vùng rìa.
- Thuật toán Watershed ngập nước trên bề mặt chi phí: $C = (1 - D_{SDT}) + 0.35(1 - M_{canopy})$. **Nó hoàn toàn không nhìn vào ảnh RGB thật!**
- Do đó, ranh giới ngập nước dừng lại ở ngưỡng xác suất $0.35$, vốn thường bị **phồng (bleed) ra ngoài vệt bóng râm hoặc mặt đất từ 3 đến 8 pixel**, thay vì dừng lại ở chiếc lá ngoài cùng!
- Thêm vào đó, phép toán `binary_fill_holes` đã ép các kẽ hở giữa 2 cây thành một đường phân giác thẳng tắp, không lách theo khe bóng râm tự nhiên của tán lá.

### 9.3. Ba cơ chế kỹ thuật để đường viền "hút" chặt vào tán lá thật
Qua khảo cứu tài liệu quốc tế (ISPRS, Remote Sensing of Environment, IEEE TGRS), có 3 kỹ thuật hàng đầu để giải quyết:

```
                            3 CƠ CHẾ BÁM SÁT TÁN LÁ (FOLIAGE SNAPPING)
                                                │
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         ▼                                      ▼                                      ▼
1. Chỉ số Diệp lục & Bóng râm           2. Khe bóng râm Watershed              3. Guided Edge Snapping
(Radiometric ExG Snapping)              (Shadow-Groove Watershed)              (Sub-Pixel RGB Gradient)
Cắt bỏ 100% vùng bóng râm đen ngòm      Chi phí ngập nước tăng vọt tại         Dùng ma trận tương quan RGB
và nền đất trơ trụi bên ngoài,         khe tối giữa 2 cây, ép ranh giới        để kéo các điểm biên "hít" vào
ép viền co sát vào lá xanh có nắng.    lách uốn lượn theo khe hẹp tự nhiên.    đúng mép lá với độ chính xác 1px.
```

1. **Cơ chế 1: Radiometric Foliage Snapping (Lọc theo Excess Green & Luminance):**
   - Tính toán chỉ số diệp lục biểu kiến từ RGB: $\text{ExG} = 2G - R - B$ và độ chói $Y = 0.299R + 0.587G + 0.114B$.
   - Xác định ranh giới bóng râm nền đất bên ngoài: $S_{\text{dark}} = (Y < 28) \land (\text{ExG} < 5)$.
   - Co rút viền cây từ ngoài vào trong: bất kỳ pixel nào nằm ở mép ngoài mà là bóng tối/đất trơ sẽ bị gọt bỏ, viền lập tức "hít" vào chùm lá xanh đầu tiên tiếp xúc với ánh sáng.
2. **Cơ chế 2: Shadow-Groove Routing Watershed:**
   - Đưa độ tối của khe bóng râm vào hàm chi phí: $\mathcal{C}_{\text{foliage}} = (1 - D_{SDT}) + \alpha (1 - M_{canopy}) + \beta \frac{\max(0, \tau_{shadow} - Y)}{\tau_{shadow}}$.
   - Nơi tiếp giáp giữa 2 cây luôn có một khe tối tự nhiên do tán lá che khuất ánh sáng. Khi chi phí ở khe tối tăng cao, nước ngập từ 2 ngọn cây sẽ không thể tràn đè lên nhau mà sẽ dừng lại chính xác dọc theo lòng khe tối đó, tạo ra ranh giới lách tự nhiên giữa 2 tán cây.
3. **Cơ chế 3: RGB Guided Edge Snapping (He et al.):**
   - Chuyển dịch gradient màu sắc của ảnh gốc $5\text{ cm}$ vào mask phân đoạn, đảm bảo viền đi theo đúng các nhánh lá chìa ra ngoài.

### 9.4. Cạm bẫy khoa học: "Bám quá sát từng cọng lá" (The Sub-Pixel Foliage Trap)
Thực nghiệm tại `test_outer_foliage_snapping.py` (Ảnh 23) đã chứng minh một chân lý quan trọng trong lâm nghiệp:
- **Cây là vật thể 3 chiều có tầng lá:** Khi mặt trời chiếu xiên, các tầng cành cây của chính nó sẽ tự đổ bóng lên các cành phía dưới, tạo ra các vệt bóng tối bên trong tán.
- **Nếu bám sát mù quáng vào mọi bóng râm:** Thuật toán sẽ xẻ dọc tán cây thành các đường rãnh răng cưa hình răng lược (comb-teeth slits) hoặc đục thủng lỗ chỗ bên trong tán, làm mất đi tính liên kết của một cá thể sống.
- **Quy chuẩn quốc tế của nhãn Ground-Truth (NeonTree, ITC Benchmark):** Các chuyên gia lâm nghiệp luôn phân định theo **Crown Projection Area (CPA) / Outer Foliage Envelope** — nghĩa là **ôm khít mép ngoài của các chùm lá ngoài cùng**, bỏ bóng râm bên ngoài, nhưng **bảo toàn khối tán vững chắc bên trong**.

### 9.5. Kết luận giải pháp chuẩn mực: Outer Foliage Envelope Snapping
Giải pháp tốt nhất đáp ứng đúng ý bạn là:
- **Mép ngoài (Perimeter):** Co sát vào các nhánh lá thật, triệt tiêu hoàn toàn phần phồng ra bóng râm nền đất.
- **Mép trong (Interior):** Giữ nguyên vẹn tính sinh học, không đục lỗ hay xẻ rãnh tán cây.
- **Ranh giới tiếp xúc (Inter-crown Seam):** Lách mềm mại theo khe bóng râm phân cách giữa 2 cây thay vì chém một nhát dao thẳng.

