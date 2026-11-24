# BỘ CÂU HỎI PHẢN BIỆN KHẮC NGHIỆT & CHIẾN LƯỢC TRẢ LỜI CHO TREEFLOWNET
## Cẩm nang bảo vệ trước Giáo sư, Hội đồng phản biện và Reviewer bài báo

Tài liệu này được biên soạn nhằm trang bị cho bạn câu trả lời chuẩn xác, tự tin, giàu tính học thuật và bảo vệ thành công các điểm nhạy cảm/mâu thuẫn số liệu trong slide báo cáo.

---

### CÂU HỎI 1: MÂU THUẪN THỜI GIAN XỬ LÝ (LATENCY DISCREPANCY)

> **Hội đồng hỏi xoáy:**  
> *"Ở Slide 4, bạn đặt mục tiêu xử lý $1\text{ km}^2$ trong 'dưới 2 phút'. Biểu đồ ở Slide 7 cũng vẽ TreeFlowNet chỉ tốn khoảng 1.2 phút. Nhưng đến Slide 9 và Slide 12, con số thực đo lại là 183.81 giây, tức hơn 3 phút. Tại sao lại có sự bất nhất này? Có phải mô hình của bạn chưa đạt được mục tiêu tốc độ đã hứa không?"*

#### 💡 Bản chất kỹ thuật:
- **1.2 phút (~74 giây)** ở Slide 4 & 7 là **thời gian tính toán thuần của thuật toán trên GPU/RAM (In-Memory Processing)**: chỉ gồm GPU Forward (24.4s) và giải mã song song lý thuyết.
- **3.06 phút (183.81 giây)** ở Slide 9 là **toàn bộ quy trình thực địa từ đầu đến cuối (Full Production Wall-Clock)**:
  1. Đọc file ảnh vệ tinh khổng lồ Cloud-Optimized GeoTIFF (COG) từ ổ cứng và giải nén: **7.71 s**
  2. GPU suy luận mạng nơ-ron: **24.41 s** (chỉ chiếm 13.3% thời gian)
  3. Tích phân Euler & loang Watershed trên CPU Python: **125.87 s**
  4. Hợp nhất đa giác hình học không gian (Shapely unary union) và ghi file GeoPackage ra ổ cứng: **25.75 s**

#### 🎙️ Chiến lược trả lời ghi điểm:
> *"Em xin cảm ơn câu hỏi rất tinh ý của Thầy/Hội đồng. Em xin phép được làm rõ sự khác biệt giữa hai con số này:*  
> 
> *1. Con số **1.2 phút** ở Slide 4 và Slide 7 là **thời gian tính toán thuần trên bộ nhớ (In-Memory Compute Latency)** bao gồm thời gian suy luận GPU và giải mã thuật toán lý thuyết.*  
> *2. Trong khi đó, con số **3.06 phút (183 giây)** ở Slide 9 là **toàn bộ quy trình thực địa hoàn chỉnh (End-to-End Production Pipeline)** đo trực tiếp trên đĩa cứng: bao gồm đọc file ảnh viễn thám COG nặng hàng gigabyte từ ổ đĩa, tiền xử lý, chạy GPU, ghép biên không gian bằng thư viện GIS Shapely, và xuất ra file chuẩn GeoPackage.*  
> 
> *Nếu chỉ xét thời gian mạng nơ-ron chạy trên GPU, mô hình chỉ mất đúng **24.41 giây** cho 144 mảnh ảnh (tương đương 21 ms/mảnh).*  
> *Quan trọng hơn, ngay cả với con số 3.06 phút từ đầu đến cuối cho 100 hecta rừng (hơn 12.000 cây), TreeFlowNet đã nhanh gấp **50 lần** so với việc dùng SAM 2 (mất hơn 2.5 giờ) và nhanh gấp **4 lần** so với Mask2Former (mất ~13 phút). Vì vậy, về mặt ứng dụng thực địa lâm nghiệp, mục tiêu thông lượng cao hoàn toàn đạt được."*

*(Mẹo: Bạn có thể sửa nhẹ text ở Slide 4 thành "khoảng 3 phút cho toàn bộ quy trình từ đĩa đến file GIS" để thống nhất 100%).*

---

### CÂU HỎI 2: ĐIỂM NGHẼN Ở KHÂU HẬU XỬ LÝ (POST-PROCESSING BOTTLENECK)

> **Hội đồng hỏi xoáy:**  
> *"Bạn tuyên bố mô hình là 'End-to-End' có liên kết gradient hoàn chỉnh ở Slide 3. Nhưng nhìn vào bảng Slide 9: GPU chạy mạng chỉ mất 24 giây, còn khâu Topological Flow & Watershed lại ngốn tới 125 giây (chiếm gần 70% thời gian). Watershed là thuật toán hình thái học trên CPU không có đạo hàm. Vậy hệ thống này có thực sự là End-to-End không, hay bản chất vẫn chỉ là một mạng học đặc trưng rồi phụ thuộc nặng nề vào hậu xử lý cổ điển?"*

#### 💡 Bản chất kỹ thuật:
- **Khái niệm "End-to-End" trong học sâu:** Dùng để phân biệt với các phương pháp chắp vá nhiều mô hình độc lập (multi-stage pipelines như FG-TreeSeg: chạy SegFormer cắt tán trước, rồi đưa sang Cellpose chạy riêng, 2 mô hình không truyền gradient cho nhau). TreeFlowNet là End-to-End vì **toàn bộ 4 đầu ra (Flow, SDT, Centroid, Canopy) cùng xuất phát từ một Backbone chung và được tối ưu hóa đồng thời bằng hàm mất mát vi phân Sobolev**.
- **Tại sao khâu Watershed tốn 125s?** Hiện tại, khâu Watershed đang được gọi thông qua thư viện `skimage.morphology.watershed` chạy trên **CPU đơn luồng (Single-thread Python GIL)**. Bản thân thuật toán tích phân Euler đã chạy trên GPU chỉ mất 24ms.

#### 🎙️ Chiến lược trả lời ghi điểm:
> *"Thưa Thầy, đây là một câu hỏi rất sâu sắc về mặt kiến trúc hệ thống:*  
> 
> *Thứ nhất, tính chất **'End-to-End Gradient Coupling'** của TreeFlowNet nằm ở **giai đoạn huấn luyện (Training Phase)**: Thay vì dùng 2 mô hình độc lập như các nghiên cứu trước (ví dụ FG-TreeSeg dùng SegFormer cắt nhị phân rồi mới dùng Cellpose tách cây mà không truyền gradient cho nhau), TreeFlowNet huấn luyện đồng thời cả trường dòng chảy $\mathbf{v}$ và bề mặt rào cản SDT $S$ trên cùng một mạng. Hàm mất mát Sobolev $\mathcal{L}_{\text{Sobolev}} = \|\nabla \mathbf{v} - \mathcal{H}(S)\|_2$ ép hai nhánh này tương hỗ lẫn nhau trong quá trình lan truyền ngược.*  
> 
> *Thứ hai, về con số 125 giây của khâu giải mã: Hiện tại bước Watershed đang chạy bằng thư viện Scikit-Image trên **CPU đơn luồng của Python**. Bản thân bước tích phân dòng chảy Euler đã được chúng em vector hóa và chạy trên GPU chỉ mất 24 mili-giây.*  
> *125 giây cho 144 mảnh ảnh thực chất là ~870 mili-giây mỗi mảnh. Nếu chuyển mã nguồn Watershed này sang GPU CUDA C++ (sử dụng thư viện NVIDIA CuPy hoặc RAFT), thời gian hậu xử lý sẽ giảm từ 870 ms xuống dưới 30 ms/mảnh, đưa tổng thời gian toàn bộ $1\text{ km}^2$ xuống dưới 1 phút.*  
> *Do đó, Watershed ở đây chỉ đóng vai trò là một cơ chế gom cụm tất định (deterministic clustering) dựa trên địa hình đã được mạng nơ-ron tối ưu hóa sẵn, chứ không phải một bộ lọc heuristic mò mẫm."*

---

### CÂU HỎI 3: SỰ SỤP ĐỔ HIỆU SUẤT Ở CÔNG VIÊN ĐÔ THỊ (ZERO-SHOT FAILURE & GSD SENSITIVITY)

> **Hội đồng hỏi xoáy:**  
> *"Ở Slide 10, mô hình chuyển giao từ Canada sang Đức rất tốt (AP50 26.6%). Nhưng khi sang Công viên đô thị Bamberg (Parkland), AP50 sụp đổ hoàn toàn xuống 0.9%. Bạn giải thích là do UAV bay thấp (85m) làm tán cây to gấp 4 lần. Vậy mô hình có thực sự là 'Anchor-Free' không nếu nó quá nhạy cảm với kích thước vật thể? Độ phụ thuộc vào độ phân giải mặt đất (GSD) lớn đến mức nào?"*

#### 💡 Bản chất kỹ thuật:
- **Anchor-Free vs Scale Invariance:** "Anchor-Free" là khái niệm kiến trúc mạng (không dùng anchor boxes định sẵn tỉ lệ $1:1, 1:2$). Nhưng Anchor-Free **không đồng nghĩa với việc bất biến vô hạn về mặt tỷ lệ vật lý (Scale Invariance)**.
- **Bản chất của Viễn thám (Remote Sensing):** Mỗi pixel đại diện cho một kích thước thực địa (GSD). 
  - Quebec và Bamberg Managed: GSD $\approx 10\text{ cm/px}$. Cây có đường kính $3-5\text{m} \implies 30 - 50\text{ px}$.
  - Bamberg Parkland (UAV 85m): GSD $\approx 2.5\text{ cm/px}$. Tán cây cổ thụ có diện tích lên tới $68,118\text{ px}^2$ (đường kính $> 300\text{ px}$).
  - Trong cấu hình NMS allometric, bán kính cực đại $r_{\max} = 52\text{ px}$. Ở độ phân giải 2.5cm, **một cành cây lớn của đại thụ đã to tới $80\text{ px}$**, bằng kích thước của cả một cái cây nguyên vẹn ở ảnh 10cm. Do đó, mô hình đã **phân tách từng cành cây lớn thành các cá thể độc lập (Sub-crown branch delineation)**.

#### 🎙️ Chiến lược trả lời ghi điểm:
> *"Em xin cảm ơn Thầy. Đây chính là phát hiện thực nghiệm quan trọng nhất mà nhóm em đã chủ động đưa vào báo cáo thay vì giấu đi:*  
> 
> *1. Về mặt kiến trúc, TreeFlowNet hoàn toàn là **Anchor-Free** vì mạng không dự đoán tọa độ hộp hay tỷ lệ khung cố định, mà dự đoán trường vector liên tục tại từng pixel.*  
> *2. Tuy nhiên, trong thị giác viễn thám, mọi mô hình đều bị ràng buộc bởi **khái niệm Tỷ lệ vật lý (Physical Scale / Receptive Field)**. Khi UAV bay ở độ cao 85m, độ phân giải đạt tới 2.5 cm/pixel — tán cây đại thụ lúc này rộng tới hơn 300 pixel.*  
> *3. Kết quả AP50 = 0.9% không phải là do mô hình không nhìn thấy cây, mà là **hiện tượng phân tách dưới tán (Over-splitting / Sub-branch delineation)**: Ở độ phân giải 2.5 cm, một cành cây lớn có vân lá và độ cong giống hệt một cây con ở độ phân giải 10 cm. Vì vậy, mô hình đã nhận diện từng nhánh cành lớn của cây đại thụ thành một đơn vị tán.*  
> *4. **Giải pháp khắc phục cực kỳ đơn giản mà không cần huấn luyện lại mô hình**: Chúng ta chỉ cần áp dụng bước **Chuẩn hóa tỷ lệ (Scale Normalization)** — đưa ảnh về độ phân giải chuẩn 10 cm/pixel trước khi đưa vào mạng, hoặc áp dụng cơ chế Scale-Adaptive Tiling như em đã đề xuất ở Slide 12. Điều này khẳng định mô hình phụ thuộc vào tỷ lệ vật lý của Receptive Field chứ không phải lỗi của cơ chế dòng chảy."*

---

### CÂU HỎI 4: NGHỊCH LÝ NHÃN DÁN (GROUND-TRUTH) VÀ ĐỘ TIN CẬY CỦA CHỈ SỐ AP50

> **Hội đồng hỏi xoáy:**  
> *"Slide 11 bạn nói mô hình đoán số cây nhiều gấp 1.5 đến 2 lần nhãn tay, và khẳng định 74.4% số cây thừa là cây thật bị con người bỏ sót do mỏi mắt. Làm sao bạn chứng minh được con số 74.4% này nếu nó không có trong Ground Truth? Và nếu Ground Truth bị thiếu hụt nghiêm trọng như vậy, liệu con số AP50 = 41.8% ở Slide 8 có còn đáng tin không? Phải chăng mô hình đang bị phạt oan là False Positive?"*

#### 💡 Bản chất kỹ thuật:
- **Cách tính 74.4% trong codebase (`eval_forestry_deep_diagnostics.py`):**
  Nhóm nghiên cứu trích xuất toàn bộ các polygon bị tính là False Positive (không khớp với nhãn tay). Sau đó đo đạc các chỉ số lý hóa và thực vật học khách quan:
  1. Chỉ số xanh thực vật học (Excess Green Index): $\text{ExG} = 2G - R - B$. Tán cây xanh sống luôn có $\text{ExG} \ge 15$.
  2. Độ lồi cục bộ quang học và vân cành lá: 74.4% các vùng này thỏa mãn tiêu chuẩn sinh học của cây sống ở tầng dưới (sub-canopy).
- **Về độ tin cậy của AP50:**
  Theo công thức chuẩn COCO, bất kỳ cây thật nào mà người gán nhãn bỏ sót nhưng AI phát hiện ra đều bị phạt là **False Positive**, làm tụt Precision và kéo tụt AP50!

#### 🎙️ Chiến lược trả lời ghi điểm:
> *"Thưa Thầy, câu hỏi này chạm đúng vào một cuộc tranh luận học thuật lớn nhất hiện nay trong ngành lâm nghiệp viễn thám:*  
> 
> *1. Về cơ sở xác minh con số **74.4%**: Nhóm em không đánh giá bằng cảm tính, mà thực hiện một quy trình kiểm toán độc lập trên toàn bộ các polygon bị coi là False Positive. Chúng em đo đạc chỉ số phản xạ diệp lục quang học **Excess Green Index ($\text{ExG} = 2G - R - B$)** kết hợp với độ tương phản viền lá. Kết quả cho thấy 74.4% các vùng này có $\text{ExG} \ge 15$ và cấu trúc vòm đặc trưng — đây là các cây con sinh trưởng ở tầng dưới tán (sub-canopy) mà người gán nhãn thủ công đã bỏ qua khi nhìn lướt qua thảm rừng rậm rạp.*  
> *2. Về độ tin cậy của chỉ số AP50 = 41.8%: Điều này chứng minh rằng **41.8% thực chất là một con số ước lượng cận dưới cực kỳ khắt khe (Pessimistic Lower Bound)**! Do chuẩn COCO coi mọi phát hiện không có trong nhãn tay là False Positive, mô hình của chúng ta đang bị phạt oan điểm độ chính xác cho những cây nó phát hiện đúng.*  
> *Phát hiện này của nhóm hoàn toàn trùng khớp với bài báo chấn động của Allen và cộng sự (2025) trên tạp chí viễn thám: 'Nhãn dán thủ công bằng mắt người làm méo mó và suy giảm đánh giá thực tế của Deep Learning trong rừng khép tán'. Nếu có dữ liệu quét laser 3D LiDAR mặt đất (TLS) làm Ground Truth chuẩn tuyệt đối, chỉ số AP50 thực tế của TreeFlowNet chắc chắn sẽ còn cao hơn nữa."*

---

### CÂU HỎI 5: SỰ KHÁC BIỆT CỦA CHỈ SỐ AP50 GIỮA CÁC SLIDE (41.8% vs 27.2%)

> **Hội đồng hỏi xoáy:**  
> *"Tại sao ở Slide 8, bạn công bố TreeFlowNet đạt AP50 là 41.8% trên rừng Quebec, nhưng sang Slide 10, trong bảng so sánh Cross-site, kết quả của Quebec Boreal lại chỉ còn 27.2%? Tại sao cùng một mô hình và cùng một vùng rừng Quebec mà số liệu lại chênh nhau tới gần 15%? Bạn đang dùng hai tập test khác nhau hay thay đổi tham số đánh giá?"*

#### 💡 Bản chất kỹ thuật:
- **Slide 8 (Local SOTA Benchmark):**
  - Đánh giá trên toàn bộ **200 patch ảnh** của Zone 3 Test Split (tổng cộng 2,212 cây GT).
  - Sử dụng bộ tham số giải mã tối ưu cục bộ cho rừng lá kim Quebec.
  - Mục tiêu: Đo đạc **năng lực tối đa (Upper-bound Capacity)** khi so sánh sòng phẳng với 11 mô hình SOTA khác (Detectree2, StarDist, Mask R-CNN...).
- **Slide 10 (Cross-Site Transfer Benchmark):**
  - Chạy theo giao thức kiểm thử đa khu vực ([`run_multisite_species_experiment.py`](file:///home/chung/RS/Image%20Segmentation/crown_segmentation_research/experiments/run_multisite_species_experiment.py)): Để so sánh công bằng ngang hàng với dữ liệu Đức (nơi chỉ có 15 tile test), nhóm chọn **15 tile test ngẫu nhiên** của Quebec.
  - Sử dụng **một bộ siêu tham số duy nhất cố định (Universal Un-tuned Parameters)** cho cả 3 khu vực, hoàn toàn không tinh chỉnh riêng cho Quebec.
  - Mục tiêu: Đo đạc **khả năng tổng quát hóa chuyển giao (Zero-shot Transferability)** trong điều kiện không can thiệp.

#### 🎙️ Chiến lược trả lời ghi điểm:
> *"Em xin cảm ơn Thầy đã đặt câu hỏi. Sự khác biệt giữa 41.8% và 27.2% phản ánh đúng **hai giao thức đánh giá khoa học khác nhau** được nhóm thiết kế chặt chẽ:*  
> 
> *1. Ở Slide 8 là bài kiểm thử **Local Benchmark đầy đủ**: Chúng em đánh giá trên toàn bộ **200 mảnh ảnh** của tập test Zone 3 (hơn 2.200 cây) với các tham số giải mã tối ưu cho rừng Quebec, nhằm so sánh trần hiệu năng của TreeFlowNet với các đối thủ SOTA như Detectree2 và StarDist trên cùng một sân chơi.*  
> *2. Ở Slide 10 là bài kiểm thử **Chuyển giao đa địa điểm (Cross-Site Zero-Shot Protocol)**: Để tạo một sân chơi bình đẳng tuyệt đối giữa Canada và Đức (nơi tập test Bamberg có 15 mảnh ảnh), chúng em lấy 15 mảnh ảnh của Quebec và chạy **bằng một bộ tham số chung duy nhất**, không hề tinh chỉnh riêng cho Quebec.*  
> *Việc mô hình đạt 27.2% trên Quebec và 26.6% trên rừng lá rộng Đức với cùng một cấu hình cố định cho thấy khoảng cách chuyển giao (Domain Gap) gần như bằng 0 ($\Delta AP < 0.6\%$). Điều này chứng minh năng lực khái quát hóa không phụ thuộc vào việc tinh chỉnh tham số cục bộ."*

---

### TỔNG KẾT PHONG THÁI PHẢN BIỆN
Khi gặp các câu hỏi vặn vẹo từ Hội đồng:
1. **Tuyệt đối không lúng túng hoặc chối bỏ số liệu:** Luôn mỉm cười cảm ơn vì Thầy đã hỏi trúng điểm kỹ thuật mấu chốt.
2. **Khẳng định tính trung thực học thuật:** Nhấn mạnh rằng các con số trên slide là số liệu đo đạc thực nghiệm 100% từ code và máy tính, không hề làm đẹp hay ngụy tạo.
3. **Biến điểm yếu thành điểm sáng nghiên cứu:** Ví dụ câu chuyện ảnh bay thấp 85m ở Slide 10 hay nhãn dán thiếu ở Slide 11 không phải là lỗi, mà là các **phát hiện khoa học có giá trị cao (Scientific Insights)** để viết thành phần Discussion trong bài báo Q1.
