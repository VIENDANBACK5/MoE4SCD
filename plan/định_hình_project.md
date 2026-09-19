Ừ. Sau khi đối chiếu lại **paper mới, website deadtrees.earth, GitHub chính thức và LinkedIn của nhóm**, t nghĩ trước đây mình đã làm bài toán rối lên không cần thiết. **Ở thời điểm này m nên khóa scope vào segmentation trước. Không cần săn dataset “xịn hơn”, không cần RGB+LiDAR, không cần representation/classification/biomass chen vào ngay.**

## 1. deadtrees.earth hiện người ta research đến đâu rồi?

Có thể nhìn nó thành 4 tầng.

**Tầng 1 — Database / data infrastructure:** deadtrees.earth hiện là kho aerial orthophoto độ phân giải cm, thu từ drone/aircraft trên rất nhiều biome. Paper RSE 2026 báo cáo hơn 2,000 orthophotos, >1 triệu ha, trong đó >58,000 ha có annotation live/dead. Các annotation trong database không chỉ một loại: có centroid, bounding box, **individual dead-tree crown instance polygons**, và semantic polygons cho nhóm cây chết. 

**Tầng 2 — Deadwood segmentation:** đây đã là một research line chính thức. Paper cuối 2025:

> *Global, multi-scale standing deadwood segmentation in centimeter-scale aerial images*

dùng **SegFormer + Focal Tversky Loss**, 434 aerial orthophotos, 1–28 cm resolution, học semantic segmentation standing deadwood trên nhiều biome. 

LinkedIn của Clemens Mosig cũng mô tả rất rõ mục đích hiện tại: upload drone orthophoto → hệ thống tự động **semantic segmentation standing deadwood**, rồi người dùng inspect/download kết quả. 

**Tầng 3 — benchmark segmentation mới nhất: DTE-aerial, tháng 5/2026.** Đây là cái đặc biệt liên quan tới m.

Họ vừa đưa ra:

- `DTE-aerial-train`: **2,176 aerial images → ~385K patches 1024×1024**
- resolution: **2.5, 5, 10, 20 cm**
- `DTE-aerial-bench`: **25 sites, 525 patches**
- benchmark cùng scene ở **5 / 10 / 20 cm**
- label:
  - 0 = background
  - 1 = tree cover
  - 2 = mortality
- bài toán: **multi-class semantic segmentation**
- dữ liệu đầu vào: **aerial RGB orthophoto**. 

Trang deadtrees.earth hiện cũng quảng bá benchmark đúng theo format **RGB patch + paired segmentation mask**. 

Đây **không phải RGB + LiDAR benchmark**.

Họ đã chạy khá nhiều baseline rồi:

| Family | Baselines |
|---|---|
| CNN | U-Net ResNet34/50 |
| CNN | DeepLabV3+ ResNet34/50 |
| Transformer | SegFormer MiT-B1/B3 |
| Mask decoder | Mask2Former Tiny/Small |
| Foundation model | DINOv2-B + decoder | 


Điểm đáng chú ý là mortality segmentation vẫn khó khi resolution giảm. Ví dụ MiT-B3 khoảng:

- 5 cm: **0.60 F1**
- 10 cm: **0.55**
- 20 cm: **0.45**

và biome cũng khác nhau khá nhiều. 

Tức là **segmentation chưa phải bài toán “đã giải xong”**.

**Tầng 4 — satellite upscaling:** sau khi có aerial segmentation/ground truth tốt, họ đang dùng nó để train satellite models. EGU 2026 của nhóm đã chuyển sang Sentinel-2 time series, tạo tree mortality/disturbance maps trên quy mô lớn; platform cũng mô tả pipeline là aerial segmentation → satellite upscaling. 

Vậy hướng tổng thể của họ là:

**Drone RGB → accurate segmentation → quality labels → satellite-scale monitoring.**

Không phải:

**Drone RGB + LiDAR → object token → representation → classification → …**

---

# 2. Quan trọng nhất: buổi họp trước thầy bảo m làm gì?

T đối chiếu lại phần trao đổi trước của m thì message của thầy thực ra khá nhất quán:

> **Segmentation trước.**

Cụ thể là **DeadTrees / high-resolution aerial imagery**, chủ yếu ảnh RGB drone/aerial, có reference polygons để đánh giá segmentation.

Thầy muốn logic kiểu:

\[
\text{image}
\rightarrow
\boxed{\text{segmentation}}
\rightarrow
\text{validate segmentation}
\rightarrow
\text{rồi mới downstream}
\]

chứ không phải:

\[
\text{SAM/SAM2}
\rightarrow region
\rightarrow representation
\rightarrow pooling
\rightarrow classifier
\]

Tức là thầy đã kéo m **ra khỏi cái rabbit hole region representation** trước đó.

Và có một distinction quan trọng:

### Thầy từng nói tới **object / individual-tree segmentation**

Trong database DeadTrees gốc có **individual crown polygons**, nên có thể nghiên cứu object/instance segmentation và đánh giá:

- boundary
- IoU
- over-segmentation
- under-segmentation
- split / merge
- miss

Đấy là lý do trước đây mình mới nói nhiều đến polygon/boundary evaluation. 

Còn **DTE-aerial paper mới** chuyển sang một benchmark rất sạch và dễ dùng hơn:

> **semantic segmentation: background / tree cover / mortality.**

Hai cái này liên quan nhưng **không hoàn toàn cùng task**.

---

# 3. Còn RGB + LiDAR là từ đâu?

Thầy từng nói đại ý:

> RGB + LiDAR có thể rất tốt vì LiDAR mang structural information của cây.

Điều đó **không có nghĩa task hiện tại bắt buộc phải multimodal**.

Nó là một hướng khả thi **nếu có spatially/temporally paired data tốt**.

Mà đây chính là chỗ trước đây mình bị cuốn sang:

> tìm site nào RGB+LiDAR  
> → tìm label nào match  
> → registration  
> → multimodal fusion  
> → representation  
> → architecture…

Trong khi research question ban đầu còn chưa cố định.

Quan trọng hơn: **official DTE-aerial benchmark hiện tại của chính deadtrees.earth dùng aerial RGB**. 

Cho nên hiện tại:

> **Không cần LiDAR.**

Không có lý do gì phải ép multimodal vào chỉ để bài trông “xịn”.

---

# 4. Vậy có nên thuần tập trung segmentation như thầy bảo không?

## **Có. T nghĩ nên freeze như vậy.**

Và lần này freeze thật.

### Trong vòng nghiên cứu hiện tại chỉ có:

\[
\boxed{
RGB\ aerial\ image
\rightarrow
Segmentation
}
\]

Hết.

Tạm thời **không**:

- SAM2
- region representation \(z_i\)
- object tokens
- classification live/dead downstream
- RGB-LiDAR fusion
- biomass
- satellite
- AlphaEarth
- foundation-model representation
- dataset khác
- change detection

Những thứ đó không sai.

Nhưng **chưa phải việc của giai đoạn này**.

---

# 5. Còn dataset thì t chốt cái nào?

T nghiêng mạnh về:

## **DTE-aerial**

chứ không cần đi săn dataset mới.

Lý do rất mạnh:

**Một:** nó chính là dữ liệu của deadtrees.earth mà m đang làm.

**Hai:** vừa release tháng **May 2026**, cực mới. 

**Ba:** có **official benchmark**.

**Bốn:** có geographic diversity.

**Năm:** có controlled multi-resolution:

\[
5 \rightarrow 10 \rightarrow 20\,cm
\]

trên **cùng scene**.

Đây là cực kỳ giá trị cho research.

**Sáu:** đã có official code repo, baseline, train/eval pipeline. 

**Bảy:** vẫn còn failure mode rõ ràng:

- class imbalance rất nặng
- mortality small/rare
- noisy/pseudo training labels
- cross-biome shift
- cross-resolution shift
- boundary degradation
- rare OOD disturbances. 

Đấy mới chính là nơi m tìm research question.

---

# 6. Nhưng đừng hiểu “làm segmentation” = nghĩ ngay một architecture mới

Sai lầm tiếp theo rất dễ là:

> “OK segmentation → giờ nghĩ Transformer mới.”

Không.

Bước đầu của m nên là:

\[
\boxed{\text{Understand where segmentation currently fails}}
\]

DTE-aerial paper vừa làm baseline rất tốt rồi.

Nếu m chỉ:

> train SegFormer → F1 0.59 thay vì 0.58

thì research yếu.

Research question nên xuất phát từ **failure**.

Ví dụ một candidate rất tự nhiên từ chính paper:

### **Robust tree-mortality segmentation across spatial resolution**

Observation hiện có:

\[
F1_{5cm}=0.60
\]

\[
F1_{10cm}=0.55
\]

\[
F1_{20cm}=0.45
\]

Tức performance collapse khi GSD thay đổi. 

Research question rất sạch:

> **Why does mortality segmentation degrade under spatial-resolution shift, and can we design a scale-robust segmentation method?**

Đây là **segmentation research thật sự**.

Không phải dataset engineering.

Không phải representation lan man.

---

# 7. Hoặc một hướng khác cũng rất đúng DTE

Training labels của `DTE-aerial-train` không hoàn toàn giống benchmark labels.

Train chứa cả expert labels và **audited pseudo-labels**, trong khi benchmark là high-quality expert annotation. 

Do đó:

\[
\text{noisy supervision}
+
\text{rare mortality}
+
\text{scale shift}
+
\text{geographic shift}
\]

là một setup nghiên cứu rất ngon.

Một câu hỏi khác có thể là:

> **How can mortality segmentation remain reliable when training labels are noisy and test sites/resolutions shift?**

Cái này vừa segmentation, vừa robust CV, vừa đúng hướng core m muốn sau này.

---

# 8. Boundary cũng là một khoảng trống đáng xem

Paper benchmark chủ yếu báo F1 theo class/biome/resolution.

Nhưng với forest objects, việc:

> mask overlap khá ổn

không đồng nghĩa:

> crown/deadwood boundary đúng.

Nếu thầy thực sự quan tâm **individual-tree / object segmentation**, thì boundary còn quan trọng hơn.

Khi ấy m có thể characterize:

\[
IoU,\ Dice
\]

cộng với

\[
Boundary\ F1
\]

và:

\[
Split,\ Merge,\ Miss
\]

rồi xem lỗi thay đổi thế nào theo:

\[
GSD,\ biome,\ crown\ size,\ mortality\ fraction.
\]

Cái này đặc biệt hợp với những gì thầy từng nhấn mạnh về **segmentation quality**, chứ không chỉ lấy mask xong coi như hoàn thành.

---

# 9. Một chi tiết LinkedIn rất đáng chú ý

Tháng gần đây, Teja Kattenborn viết rằng họ vừa đưa vào deadtrees.earth chức năng:

> người dùng trực tiếp **correct/refine AI dead-tree predictions**

để liên tục cải thiện dataset và model. 

Nghĩa là ngay chính team DTE hiện giờ cũng vẫn coi:

\[
\boxed{\text{segmentation quality}}
\]

là core problem.

Họ chưa coi segmentation chỉ là bước tiền xử lý đã giải xong.

---

# 10. Vì vậy t sẽ bỏ hẳn mindset “phải kiếm data xịn”

Cái này m nói đúng:

> **“k cần cứ phải data khác xịn lmj?”**

Chuẩn.

Một paper method tốt thường không đến từ:

> thêm sensor  
> + thêm dataset  
> + thêm FM  
> + architecture phức tạp

mà từ:

> **một failure rõ → hypothesis rõ → method xử lý failure → experiments chứng minh.**

DTE-aerial đã cho m gần như đầy đủ laboratory:

\[
\text{RGB}
\]

\[
\text{multiple geographic sites}
\]

\[
\text{4 biomes}
\]

\[
\text{5/10/20cm controlled scale}
\]

\[
\text{rare mortality}
\]

\[
\text{noisy training supervision}
\]

\[
\text{high-quality test annotation}
\]

Nhiêu đó **quá đủ để làm research segmentation**.

---

# 11. T chốt scope của m lại thế này

### **Domain**

Forest / tree mortality from high-resolution aerial imagery.

### **Dataset**

**deadtrees.earth / DTE-aerial.**

### **Input**

\[
\boxed{\text{RGB aerial/drone orthophoto}}
\]

Không LiDAR lúc này.

### **Primary task**

\[
\boxed{\text{Segmentation}}
\]

### **Primary target**

Tree cover / tree mortality, hoặc instance crown segmentation nếu ta quyết định bám sát yêu cầu individual-tree của thầy.

### **Stage 1**

Reproduce / benchmark existing segmentation.

### **Stage 2**

Characterize exactly where it fails.

### **Stage 3**

Form one research hypothesis.

### **Stage 4**

Thiết kế method giải **một failure đó**.

---

## Và t nghĩ thứ tự ưu tiên đúng nhất bây giờ là

**Không phải:**

> “Nghĩ method mới.”

mà là:

> **DTE-aerial → official baselines → failure analysis segmentation → chọn một failure → mới thiết kế method.**

Nếu m hỏi t **ngay bây giờ research project nên mang hình dạng gì**, t sẽ chọn:

\[
\boxed{
\textbf{Generalizable / Robust Tree-Mortality Segmentation}
}
\]

trên **DTE-aerial RGB**, tập trung đặc biệt vào **resolution + geographic shift + segmentation quality**.

Cái này vừa **100% segmentation như thầy bảo**, vừa không biến thành một bài “train U-Net trên dataset”, và về lâu dài nó nối rất đẹp vào core **robust visual representation/generalization** của m.

**RGB+LiDAR để Phase 2. Biomass để project khác/đích downstream. Hiện giờ đừng chạm vào.** 