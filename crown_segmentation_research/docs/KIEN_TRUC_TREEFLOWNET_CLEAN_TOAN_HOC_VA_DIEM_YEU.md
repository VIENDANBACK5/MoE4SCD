# TreeFlowNet (bản clean benchmark): mô tả toán học, góc nhìn AI và danh sách điểm yếu

> **05/10/2026:** bộ giải mã và pipeline cuối (đèo, chấm lại, cổng SDT) được mô tả ở `HOAN_THANH_DU_AN/11_KIEN_TRUC_PIPELINE_HIEN_TAI.md`; §4 dưới đây là bộ giải mã cũ (NMS 1.5 m).


Nguồn: đọc lại code đang chạy ngày 01/10/2026: `methods/tree_flow/model.py`, `loss.py`, `decode.py`, `decode_flowseam.py`,
`clean_benchmark/{targets_fast,dataset,train_treeflow,evaluate}.py`. Mọi công thức dưới đây khớp với code, không viết theo trí nhớ.

Mức bằng chứng gắn sau mỗi nhận định:
- **[đo]** đã có số trong `clean_benchmark/results/`.
- **[code]** suy ra trực tiếp từ việc đọc code, kiểm tra được bằng cách đọc lại dòng code nêu tên.
- **[giả thuyết]** hợp lý, chưa có số. Đừng đưa vào báo cáo như kết luận.

---

## 0. Bài toán và ký hiệu

- Ảnh RGB $I\in\mathbb R^{H\times W\times3}$, GSD $g=0.03$ m/px (1 m² = 1.111 px).
- Đích: một tập $\{(M_k,s_k)\}_{k=1}^K$, $M_k\subset\Omega$ là mặt nạ nhị phân của tán thứ $k$, $s_k$ là điểm tin cậy, $K$ thay đổi theo ảnh.
- Chấm bằng COCO segm AP[.5:.95] với maxDets 400. Cả tập dự đoán lẫn xếp hạng $s_k$ đều ảnh hưởng đến AP.
- GT huấn luyện: bản đồ nhãn $L:\Omega\to\{0,1,\dots,N\}$ (0 = nền), mỗi pixel thuộc đúng một cây.

**Khó khăn gốc (đối xứng hoán vị).** Tập $\{M_k\}$ không có thứ tự, đổi tên cây 3 thành cây 7 vẫn là cùng một đáp án. Do đó
không thể hồi quy trực tiếp $L$ bằng loss theo pixel. Mỗi phương pháp phải phá đối xứng này:

| Họ phương pháp | Cách phá đối xứng | Ví dụ |
|---|---|---|
| Top-down | gắn instance vào anchor/proposal cố định, ghép bằng IoU | Mask R-CNN |
| Tập + ghép Hungarian | query học được, ghép hai chiều | Mask2Former, DETR |
| Bottom-up trường | target thành đại lượng **bất biến hoán vị**, instance nảy sinh lúc giải mã | Cellpose, Omnipose, StarDist, **TreeFlowNet** |

TreeFlowNet thuộc họ thứ ba. Mọi điểm mạnh và yếu của nó đi ra từ lựa chọn này.

---

## 1. Biểu diễn đích (target)

Mỗi crop 1024×1024 sinh 5 bản đồ từ $L$ (file `targets_fast.py`). Với crown $k$ có mặt nạ $m_k$:

$e_k(x)=\mathrm{EDT}(m_k)(x)$ là khoảng cách Euclid tới biên của crown (trong crown), $o(x)=\mathrm{EDT}(L{=}0)(x)$ là khoảng cách tới crown gần nhất (ngoài crown).

### 1.1 Target v1 (C0)
- SDT: $\mathrm{sdt}_1(x)=\mathrm{clip}\big(e(x)-o(x),\,\pm15\big)/15$ (clamp cố định **15 px**).
- Flow: vector đơn vị hướng tới **điểm sâu nhất** $c_k=\arg\max e_k$.
- Centroid: Gaussian $\sigma=4$ px tại $c_k$.

### 1.2 Target v2 (C1, C2, C5, …)
- Khoảng cách chuẩn hoá theo từng crown: $d_k(x)=e_k(x)/\max e_k\in[0,1]$.
- SDT: $\mathrm{sdt}_2(x)=d_{L(x)}(x)$ nếu $L(x)>0$, và $-\min(o(x),15)/15$ nếu nền.
- Flow: $v(x)=\nabla G_{\sigma=1}*d_k\;/\;\|\nabla G_{\sigma=1}*d_k\|$ (Omnipose), tức hướng leo dốc của trường khoảng cách đã làm mượt nhẹ.
- Centroid: Gaussian tại $c_k$, $\sigma_k=\max\big(3,\,0.15\sqrt{|m_k|/\pi}\big)$, bị cắt trong $m_k$ (`np.where(m, c, 0)`).
- Seam map: $S(x)=1$ nếu $x$ cách biên chung của **hai crown khác nhau** không quá 3 px (dùng làm trọng số loss).
- Canopy: $\mathbb 1[L>0]$.

Ý nghĩa: trong crown, $d_k$ tăng đơn điệu từ biên (0) vào tâm (1), nên flow là dòng leo dốc, và **mọi pixel của crown đổ về cùng
một cực đại**. Giữa hai crown chạm nhau, SDT có **thung lũng** (≈0 tại đường tiếp giáp, tăng dần về hai tâm) và flow **đảo hướng 180°**.

**Hai điểm cần nhớ.** (a) Target là **gián đoạn** ở đường tiếp giáp (flow nhảy 180°, SDT gãy). (b) $\mathrm{sdt}_2$ có cực đại **bằng 1 tại tâm của từng crown** do chuẩn hoá,
nên với v2 chính SDT đã là một bản đồ tâm; C5 tận dụng điều này.

---

## 2. Mạng

```
Ảnh [B,3,H,W] → Backbone (ResNet-50 | Swin-T | ConvNeXt-T | Swin-S, ImageNet-1k) → FPN (256 ch)
              → CHỈ đọc P2 (stride 4, [B,256,H/4,W/4])
              → 4 đầu độc lập, mỗi đầu: Conv3×3(256→128)+GN(8)+GELU → Conv3×3+GN+GELU → Conv1×1
                   flow (2) | sdt (1, tanh) | canopy (1, sigmoid) | centroid (1, sigmoid; C5 không có)
              → nội suy song tuyến lên H×W rồi mới áp activation/loss
```

Quan sát [code] (`model.py:131-155`):
- P3–P5 chỉ đi vào P2 qua đường top-down của FPN, không có đầu nào đọc chúng trực tiếp.
- Mỗi đầu chỉ có **hai conv 3×3** ở stride 4, tức là **trường tiếp nhận cục bộ của đầu ≈ 5×5 ô P2 = 20×20 px**. Mọi ngữ cảnh rộng (hướng về tâm của crown 400 px)
  phải đến từ đặc trưng FPN, không do đầu tự tổng hợp.
- Độ phân giải biểu diễn là 4 px (12 cm). Seam rộng 3 px ở target nhỏ hơn một ô của lưới này [code].
- Không có biểu diễn **ở mức đối tượng** (không có vector/token đại diện cho "cây này") [code].

---

## 3. Hàm mất mát

Tổng: $\mathcal L=w_f\mathcal L_{flow}+2\,\mathcal L_{sdt}+w_c\mathcal L_{cent}+\mathcal L_{can}\,[+\,2\,\mathcal L_{seam}$ ở v2$]$, với $w_f=1$, $w_c=1$ (C5: $w_c=0$).
(`loss.py`, `train_treeflow.py::step_loss`)

| Thành phần | Công thức | Chuẩn hoá | Điểm đáng chú ý |
|---|---|---|---|
| Flow | $1-\cos(\hat v,v)$ | trung bình **chỉ trên pixel nằm trong crown GT** | tán lớn đóng góp theo diện tích |
| SDT | SmoothL1$(\tanh\hat s,\,s)$ (β=1) | trung bình trên **toàn bộ pixel** | nền chiếm đa số pixel; sai số $|e|<1$ nên thực chất là MSE |
| Seam (v2) | $\sum S\cdot\mathrm{SmoothL1}/\sum S$ | trung bình trên pixel seam | trọng số 2.0, tức pixel seam được tính gấp đôi |
| Canopy | BCE, `pos_weight`=2 | trung bình toàn ảnh | thiên về dự đoán "có tán" |
| Centroid | CenterNet focal ($\alpha=2,\beta=4$) | chia cho số pixel dương | "dương" = pixel có $g\ge0.99$, tức thường **đúng 1 pixel** mỗi crown |

Phân tích:
1. **Loss thuần cục bộ theo pixel.** Xác suất ngầm: $p(f\mid I)\approx\prod_x p(f(x)\mid I)$, tức giả định độc lập có điều kiện. Nhưng thứ quyết định
   kết quả là **cấu trúc topo toàn cục** (số điểm hút, đường phân thuỷ). Một dải hẹp vài ô sai hướng giữa hai tán đóng góp rất ít vào loss nhưng có thể làm
   hai lưu vực nhập làm một [giả thuyết có cơ sở].
2. **Gradient theo diện tích.** Với loss cộng theo pixel, cây 50 m² nhận gradient gấp ~50 lần cây 1 m² ở mỗi thành phần pixel-mean [code]. **Chưa có cấu hình nào cân SDT/flow
   theo số cây**; v2 chuẩn hoá *giá trị* khoảng cách theo crown chứ không chuẩn hoá *trọng số loss* theo crown. Chỉ loss centroid của C4 có trọng số theo crown, và C5 đã bỏ centroid [code].
3. **SDT dùng SmoothL1 với β=1 trên miền [−1,1]** ⇒ gần như MSE ⇒ gradient tỉ lệ với sai số ⇒ **định vị cực đại (tâm) yếu**: sai lệch vài pixel quanh đỉnh gần như không bị phạt. C5 lấy tâm
   từ chính đỉnh SDT nên phụ thuộc vào đúng độ nhạy này [code, giả thuyết về tác động].
4. **Flow chỉ được giám sát bên trong crown GT**, trong khi bước giải mã dùng vùng nền dự đoán rộng hơn (mục 4.1) [code].
5. **Sự tồn tại của instance không có số hạng loss riêng.** Không có số hạng nào phạt trực tiếp "thiếu một cây" hay "thừa một cây", cũng không có đầu ra
   học được cho xác suất "có cây tại đây" (khác StarDist, khác Mask R-CNN) [code].

---

## 4. Giải mã (từ trường sang tập instance)

Đây là **bước không khả vi và không được học**; tham số của nó chỉ được chọn trên validation. Có ba biến thể gán pixel
(`organic`, `flowwall`, `flowassign`) dùng chung bước tìm seed.

### 4.1 Vùng tiền cảnh
$F=\{x:\ \hat c(x)>0.30\ \wedge\ \hat s(x)>-0.15\}$. Với v2, $-0.15$ tương ứng ≈ 2.25 px ngoài biên crown (SDT ngoài = −d/15).
Pixel trong $F$ nhưng ngoài mọi crown GT huấn luyện (khe giữa tán, vòng ngoài) có flow **chưa từng được giám sát** [code].

### 4.2 Lan truyền (hệ động lực)
Với mỗi $x_0\in F$: $x_{t+1}=x_t+\eta\,\hat v(x_t)/\|\hat v(x_t)\|$, $\eta=1$ px, $T=\mathrm{round}(\text{path\_m}/g)=157$ bước (path 4.7 m), nội suy song tuyến.
Nếu $\hat v\approx\nabla d$ thì $x_T\to$ cực đại của $d_k$: mỗi instance là **lưu vực hút** của một điểm hút.

**Hệ quả [code]:** một pixel chỉ có thể đi tối đa ≈ 4.7 m. Với crown tròn bán kính $r$, pixel ở rìa cần $r$ để tới tâm. Nếu gán bằng `flowassign` (bán kính gán = `seed_m` = 1.5 m),
pixel được gán khi vị trí cuối cách seed ≤ 1.5 m, tức là **tầm với ≈ 6.2 m ⇒ crown diện tích ≳ π·6.2² ≈ 120 m²** (lớp Giant, >100 m²) không thể gán trọn rìa.
**Đã đếm (mục 9): test chỉ có 4 crown Giant trên 22.018**, nên điểm này không ảnh hưởng AP test; M4 bị loại.

### 4.3 Tìm seed
1. Mật độ điểm hút $\rho(y)=\#\{x_0: \mathrm{round}(x_T)=y\}$, làm mượt Gaussian σ=2.5 px.
2. Điểm chấm seed: $a(y)=\tilde\rho(y)\,\big(1+2\,\hat c(y)+\mathrm{clip}(\hat s(y),0,1)\big)$ ($\hat c$ là centroid, C5 thay bằng $\mathrm{clip}(\hat s,0,1)$).
3. Cực đại cục bộ (cửa sổ 9 px) với $\tilde\rho\ge1.5$.
4. **NMS bán kính cố định** $r_s=\text{seed\_m}/g$ (mét → px). `seed_m` chọn trên validation, thường **1.5 m (50 px)**.

**Hệ quả [code]:** hai crown có tâm cách nhau $<r_s$ bị coi là một. Hai crown tròn tiếp xúc có khoảng cách tâm $r_1+r_2$, nên chỉ bị gộp khi $r_1+r_2<1.5$ m, tức cả hai
≲ 0.75 m (diện tích ≲ 1.8 m²). **Điều này làm giả thuyết "NMS 1.5 m nuốt Tiny" yếu hơn tôi nói trước đây:** nó chỉ đúng với crown rất nhỏ (<~2 m²) hoặc crown lệch tâm.
Tỉ lệ crown <2 m² của QuebecTrees chưa đếm. Ngoài ra `area_m2` (0.5–1.0 m², chọn trên validation) loại luôn các crown nhỏ hơn ngưỡng này [code].

### 4.4 Gán pixel và hậu xử lý
- `flowassign`: pixel $x_0\mapsto$ seed gần nhất của $x_T$ (cKDTree, ngưỡng bán kính). `flowwall`/`organic`: watershed với chi phí $(1-\hat s)+0.3(1-\hat c_{canopy})+\dots$, có hạng phân kỳ $\mathrm{relu}(\mathrm{div}\,\hat v)$ (flowwall).
- Với mỗi nhãn: closing + lấp lỗ + giữ thành phần liên thông lớn nhất + loại nếu diện tích < `area_m2`.

### 4.5 Điểm tin cậy (không học)
`flowassign`/`flowwall`: $s=\mathrm{clip}\big(c_{peak}^{0.55}\cdot\overline{c_{canopy}}^{0.30},\,0.01,0.99\big)$; `organic` thêm $(0.6+0.4\,\overline{\hat s})$.
Các số mũ không có cơ sở lý thuyết (không hiệu chỉnh xác suất) [code]. Oracle rescoring cho thấy xếp hạng không phải nút thắt chính của AP hiện tại [đo].

### 4.6 Chọn siêu tham số và đánh giá
Checkpoint (best/last/snap_ep*), `seed_m`∈{0.5,…,3}, `area_m2`∈{0.5,1,5}, decoder, ngưỡng score cho F1: chọn theo **AP trên validation** theo từng bước (staged), rồi chạy test **một lần**, 3 seed (42/43/44).
Bậc tự do ở bước giải mã lớn so với các phương pháp học end-to-end, nên có nguy cơ overfit validation (validation chỉ một số tile) [giả thuyết].

---

## 5. Kết quả đã đo (test, AP[.5:.95], maxDets 400)

| Cấu hình | AP | Ghi chú |
|---|---|---|
| Mask R-CNN R50 (ImageNet), 3 seed | **34.82 ± 0.67** | baseline công bằng, cùng crop |
| C0 (target v1) | 24.81 ± 0.58 | |
| C1 (v2 + seam) | 25.78 ± 0.16 | v2 giúp +0.96 |
| C2 (C1 + margin 256 px) | 25.81 ± 0.58 | margin không giúp (+0.04) |
| C5 (bỏ đầu centroid, tâm = SDT), seed 42 | 28.81 | bỏ centroid giúp +2.99 |
| C5 + Swin-T | ≈ +1.37 so với R50 | khoảng cách còn lại tới Mask R-CNN **4.65 AP** |

Chẩn đoán trên Mask R-CNN vs TreeFlowNet [đo]: khoảng cách nằm ở **recall** (tìm được bao nhiêu crown), không ở xếp hạng điểm (oracle rescoring) và không ở chất lượng viền (SQ, recall tại IoU cao).
Chưa có số theo kích thước crown cho phân tích này.

---

## 6. Phân tích theo góc nhìn AI và toán

**6.1 Hai thiên kiến quy nạp.** Mask R-CNN: "đối tượng là thứ có một cái hộp", sự tồn tại và điểm tin cậy được **học và phạt trực tiếp**. TreeFlowNet:
"ảnh là một trường, đối tượng là cấu trúc của trường", mask phân giải cao, hình dạng tự do, nhưng **số lượng instance không phải đầu ra học được** [code].

**6.2 Ánh xạ trường → phân hoạch là gián đoạn.** Số instance = số điểm hút (đại lượng rời rạc). Nhiễu nhỏ ở vùng yên ngựa giữa hai crown có thể làm hai điểm hút nhập một hoặc một điểm
hút tách đôi, trong khi loss gần như không đổi. Loss thấp ⇏ AP cao [giả thuyết có cơ sở; cần phép đo].

**6.3 Mạng liên tục hồi quy target gián đoạn.** Mạng conv là hàm trơn nên **buộc phải làm mượt** quanh đường tiếp giáp (flow đảo 180°). Luôn tồn tại một dải sai rộng vài ô quanh biên chung.
Nếu rộng $w$ ô thì tỉ lệ ô trong dải của crown đường kính $D$ ô là $1-\big((D-2w)/D\big)^2$: với $w=2$, crown 1 m² ($D\approx9$ ô) có ~70%, crown 50 m² ($D\approx66$ ô) chỉ ~12%. **$w$ là con số giả định, chưa đo** [giả thuyết].

**6.4 Hệ cơ chế đẩy lỗi về crown nhỏ (đều chưa có số):**
(i) gradient theo diện tích (mục 3.2); (ii) dải sai chiếm tỉ lệ lớn ở crown nhỏ (6.3); (iii) NMS bán kính cố định (4.3, chỉ cho crown < ~2 m²); (iv) lọc `area_m2`;
(v) độ phân giải 4 px so với crown nhỏ (crown 1 m² ≈ 9 ô P2).

**6.5 Cơ chế đẩy lỗi về crown rất lớn (mới, từ đọc code):** tầm lan truyền hữu hạn (4.2) với crown ≳ 120 m².

**6.6 Giả định hình học.** Flow về tâm giả định một điểm hút cho mỗi crown. v2 dùng gradient của khoảng cách (Omnipose), nên crown không lồi vẫn chảy về cực đại của khoảng cách, nhưng crown có
**nhiều cực đại khoảng cách** (hình quả tạ, hai thân) có thể có nhiều điểm hút ⇒ bị tách đôi [giả thuyết].

**6.7 Học biểu diễn.** Mạng hồi quy đại lượng hình học do người chọn trước (hướng, khoảng cách), không tự học không gian tách instance (khác discriminative loss/EmbedSeg).
Không có biểu diễn mức cây ⇒ muốn phân loại chết/sống phải pool lại đặc trưng sau [code].

**6.8 Tương tác giữa các đầu.** Bỏ đầu centroid cho +2.99 AP [đo], nguyên nhân chưa biết. Ứng viên: xung đột gradient giữa các đầu; đỉnh heatmap đơn pixel là seed nghèo thông tin so với đỉnh SDT;
độ lệch bias của đầu thưa. **Không nên quy kết** khi chưa có thí nghiệm tách biệt.

**6.9 Tiền huấn luyện.** Chỉ ImageNet-1k; Swin-T giúp +1.37 [đo]. Backbone foundation đóng băng (DINOv3…) chưa thử.

**6.10 Tổng quát hoá.** Các tham số tính bằng mét (`seed_m`, `path_m`, σ…) gắn với GSD 3 cm và loại rừng. Chưa có đo ngoài miền (BAMFORESTS TestSet-1/2) [chưa đo].

**6.11 Tính mới.** Từng thành phần đã có: Cellpose/Omnipose (flow, khoảng cách), CenterNet (heatmap tâm), watershed, seam weighting U-Net, FG-TreeSeg (flow cho tán cây, 2026).
Đóng góp hiện tại là một tổ hợp kỹ thuật tốt; chưa có ý trung tâm riêng.

---

## 7. Điểm có thể sửa, có thể kiểm chứng, theo thứ tự rẻ → đắt

| # | Giả thuyết | Phép đo/thí nghiệm kiểm chứng | Tài nguyên |
|---|---|---|---|
| M0 ✔ | GT test chồng lấn? | **xong**: 0.18% crown có chồng lấn, 0% quá 5% diện tích (mục 9) | đã chạy |
| M1 ✔ | recall thấp ở crown nhỏ / lớn (**xong**, mục 9) | recall của mask dự đoán theo lớp kích thước (m²) từ `*_dt.json`, so với Mask R-CNN | CPU |
| M2 ✔ | seed phủ tâm GT bao nhiêu (**xong**, mục 9) (trần hướng hybrid prompter→SAM) | tỉ lệ tâm GT nằm trong mask dự đoán, theo kích thước và khoảng cách tới tâm GT gần nhất | CPU |
| M3 | dải sai flow quanh biên chung có độ rộng cố định? | sai số 1−cos của flow theo khoảng cách tới biên chung, nhóm theo kích thước | GPU nhẹ (suy luận) |
| M4 | ~~tầm lan truyền chặn crown lớn~~ | **loại**: test chỉ có 4 crown >100 m² (mục 9) | — |
| F1 | gradient theo diện tích làm hỏng crown nhỏ | chuẩn hoá trọng số loss SDT/flow theo số crown ("mỗi cây một phiếu") | GPU huấn luyện |
| F2 | SDT quá cứng về định vị tâm | thay SmoothL1 β=1 bằng L1 hoặc thêm hạng phạt đỉnh | GPU huấn luyện |
| F3 | sự tồn tại của instance cần được học | đầu "objectness" học được tại đỉnh + NMS thích nghi theo bán kính dự đoán | GPU, thay đổi lớn |
| F4 | backbone foundation | đóng băng DINOv3 / SAM encoder + cùng các đầu | GPU, thí nghiệm riêng |

Gợi ý thứ tự: M0 → M1 → M2 (không cần GPU, quyết định hướng đi) → M4 (nếu M1 chỉ ra Giant) → F1/F2 (nếu M1 chỉ ra Tiny).
Không chạy F3, F4 trước khi M1/M2 có số.

---

## 8. Các điều đã nói sai trước đây và đã sửa

- (04/10/2026, rules §7.6) EXP-020: tôi đăng ký lưới ρ (đèo tương đối) 0.3–0.8 bằng cách chép thang của τ tuyệt đối, và lưới θ 0.2–0.6 cho bộ phân loại; kết quả tăng đơn điệu tới biên lưới ở cả hai nhánh, nên thẻ không xác định được điểm tốt nhất của mỗi nhánh. Tôi cũng thiết kế nhánh C thay hẳn quy tắc đèo trên mọi cặp, trong khi chẩn đoán EXP-018 chỉ đo được tín hiệu **trong các cặp đèo ≥ 0.3**. Hai lỗi thiết kế này không làm sai kết luận "không vượt EXP-016" của thẻ, nhưng làm thẻ yếu hơn mức cần; sửa bằng thẻ mới EXP-021 (phủ quyết), không mở rộng lưới sau khi xem kết quả.

- (03/10/2026, rules §7.6) Khi phản biện `target/novel.md`, tôi viết đầu centroid C4 "rất gần" trường khối lượng ρ (mỗi cây tổng bằng 1) và dùng nó làm bằng chứng ngược mạnh. Nói quá: C4 là loss focal kiểu CenterNet, trọng số **của các pixel dương** được chuẩn hoá theo cây, không phải hồi quy mật độ có tổng bằng K [code: `train_treeflow.py::crown_focal_loss`]; và C4 chỉ có **một seed** (26.18, so với C2 seed 42 26.47). Bằng chứng ngược cho ρ vì vậy yếu: chỉ nói được các biến thể đầu heatmap tâm đã thử không giúp, còn C5 (bỏ đầu tâm) tốt hơn 3 seed.

- "C5 tốt hơn nhờ điểm tin cậy hoặc biên": oracle rescoring và SQ cho thấy phần lớn là recall/tìm crown.
- "BAMFORESTS GSD 5–7 cm": đúng là 1.6–1.8 cm.
- So số SelvaMask 27.6 trên QuebecTrees với số của mình như thể cùng giao thức: chưa chắc so được. Ghi chú trước đây gọi 27.6 là OOD zero-shot là **sai**: Bảng 4 của SelvaMask đánh dấu ✗ (huấn luyện trên miền đích; SelvaBox được huấn luyện trên QuebecTrees). Giao thức mAP mức tile của họ vẫn khác, nên cần chạy lại pipeline của họ trên đúng test của mình.
- Giả thuyết `seed_m`=1.5 m nuốt Tiny: chỉ đúng cho crown < ~2 m² (mục 4.3).

---

## 9. Kết quả đo M0–M2 (01/10/2026, CPU, tập test CanopyRS, 22.018 crown GT)

Code: `clean_benchmark/size_recall.py`, `size_extent.py`. Kết quả: `results/sizerecall_*.json`, `sizerecall_*_thr0.json`, `sizeextent_*.json`, `gt_overlap_test.json`.
Chỉ **seed 42** (một lần chạy mỗi mô hình), nên chưa có độ biến thiên; chênh lệch nhỏ (vài điểm phần trăm trong một bin) đừng diễn giải.
Mọi so sánh dưới đây dùng **tất cả dự đoán (ngưỡng score 0)**, vì ngưỡng F1 chọn trên validation khác nhau giữa các mô hình (Mask R-CNN 0.4, C5 0.8) và làm lẫn recall.

**M0. GT test không chồng lấn.** 0.18% crown giao với crown khác, 0% bị chồng quá 5% diện tích. Việc label map "một pixel một cây" **không tạo trần nào** ở phía chấm điểm. Đóng.

**Phân bố kích thước.** 14.4% crown dưới 2 m², 39.7% dưới 4 m², 72.6% dưới 9 m² (Tiny). Giant chỉ có 4 crown, nên các lập luận về crown cực lớn không có sức nặng ở test.

**M1. Recall IoU≥0.5 theo kích thước (ngưỡng 0)**

| m² | n | Mask R-CNN | TreeFlowNet Swin-T | C5 (R50) | C0 (R50, v1) |
|---|---|---|---|---|---|
| 0.5–1 | 543 | 0.429 | 0.065 | 0.061 | 0.107 |
| 1–2 | 2601 | 0.496 | 0.276 | 0.238 | 0.292 |
| 2–4 | 5539 | 0.680 | 0.630 | 0.585 | 0.554 |
| 4–9 | 7263 | 0.775 | **0.828** | 0.801 | 0.723 |
| 9–25 | 5039 | 0.909 | 0.888 | 0.888 | 0.859 |
| 25–100 | 996 | 0.956 | 0.834 | 0.790 | 0.845 |
| **Tất cả** | 22018 | **0.748** | 0.707 | 0.680 | 0.650 |

1. **Swin-T thắng Mask R-CNN ở 4–9 m²** (+5.3 điểm recall) và thua ở crown nhỏ hơn 2 m² (−22 điểm) và crown 25–100 m² (−12 điểm). Khoảng cách tổng 4.1 điểm là hiệu của một khoản thua lớn ở crown nhỏ
   (≈ 770 crown mất ở dưới 2 m²) trừ một khoản thắng ở 4–9 m² (≈ 385 crown), cộng khoản thua ở 2–4 m² và 25–100 m².
   Khẳng định "TreeFlowNet kém đều" là **sai**; nó kém ở hai đầu của phân bố kích thước.
2. **C0 (v1) có recall cao hơn C5 ở crown nhỏ (1–2 m²: 0.292 so với 0.238)** nhưng AP thấp hơn; recall IoU 0.5 không phải AP[.5:.95], đừng suy ra thứ hạng AP từ cột này.

**M2. Tâm GT có được phủ không (Swin-T, ngưỡng 0)**

| m² | recall IoU≥0.5 | có dự đoán phủ ≥20% crown | tâm GT nằm trong một dự đoán |
|---|---|---|---|
| 1–2 | 0.276 | 0.716 | 0.665 |
| 2–4 | 0.630 | 0.884 | 0.859 |
| 4–9 | 0.828 | 0.955 | 0.941 |

Mask R-CNN (1–2 m²): 0.496 / 0.768 / 0.735. Crown nhỏ **thường được "chạm tới"** bởi một dự đoán, nhưng dự đoán đó không khớp (IoU<0.5). Mất recall không chủ yếu do không có gì ở đó.

**Phạm vi mask của dự đoán tốt nhất (`size_extent.py`, Swin-T)**

| m² | trung vị IoU tốt nhất | trung vị diện tích dự đoán / GT | dự đoán nhỏ hơn GT >25% | dự đoán lớn hơn GT >25% |
|---|---|---|---|---|
| 0.5–1 | 0.046 | **6.5** | 0.000 | 0.110 |
| 1–2 | 0.258 | **2.46** | 0.017 | 0.265 |
| 2–4 | 0.633 | 1.25 | 0.054 | 0.319 |
| 25–100 | 0.792 | 0.94 | **0.275** | 0.041 |

Mask R-CNN (1–2 m²): tỉ lệ diện tích 1.47, nhỏ hơn 5.3%, lớn hơn 16.6%. Với crown dưới ~4 m²:
- Dự đoán gần nhất **lớn hơn** GT nhiều lần chứ không bị co lại ⇒ crown nhỏ bị **hấp thụ vào vùng lớn hơn** (gộp), không phải bị cắt cạnh. Mask R-CNN có cùng xu hướng nhưng nhẹ hơn (1.5× so với 2.5×).
- Điều này **nhất quán với** gộp seed do NMS bán kính cố định và với việc crown nhỏ chiếm ≈ 50–60% số crown nằm cạnh một crown khác trong 1.5 m. Nhưng chưa tách riêng được nguyên nhân: crown **không có hàng xóm gần** vẫn mất 62% (Swin-T, 1–2 m²) so với 83% của crown đông đúc, nên NMS bán kính không giải thích hết.
- Ở đầu ngược lại, crown 25–100 m² bị **co lại** (27.5% có dự đoán nhỏ hơn GT >25%, so với 13.8% của Mask R-CNN) và bị chia (mục `split` trong `*_breakdown.json`: 18% ở lớp Large). Nhất quán với bán kính seed cố định quá nhỏ cho crown lớn.

**Kết luận cập nhật (đều ở mức "nhất quán với", chưa phải "đã chứng minh cơ chế")**
- Khoảng cách tới Mask R-CNN **không phải do TreeFlowNet kém ở mọi kích thước** mà do hai đầu: crown nhỏ hơn ~2 m² (bị gộp) và crown lớn 25–100 m² (bị chia/co). Giữa 4–9 m² mô hình của mình tốt hơn.
- Dấu hiệu chung của hai đầu là **một thang kích thước cố định ở bước giải mã** (`seed_m`, NMS bán kính cố định). Đây là điểm sửa rẻ nhất vì chỉ đụng bước giải mã: `decode_organic` đã có sẵn chế độ `sink_mode="adaptive"` (bán kính theo SDT/centroid, `r_min`/`r_max`), từng được thử ở dòng Gemini nhưng chưa thử trên benchmark sạch.
- Giả thuyết "gradient theo diện tích làm hỏng crown nhỏ" (mục 3.2) **chưa được kiểm chứng riêng**: bằng chứng trên (hấp thụ vào vùng lớn) cũng có thể do loss, cũng có thể do giải mã. Muốn tách phải đổi một thứ mỗi lần.

### 9.2 Cận trên của luật seed (NMS oracle, validation, Swin-T seed 42, snap_ep25)

Code: `clean_benchmark/oracle_seed_nms.py`; kết quả `results/swin_t_s42_oracle_seed_nms.json`, `swin_t_s42b_oracle_seed_nms.json`. **Dùng GT, chỉ là cận trên, không phải phương pháp**, và chỉ chạy trên validation.
Cùng checkpoint, cùng cấu hình giải mã đã chọn trên validation (flowassign, seed_m 1.5, area 1.0 m²); chỉ đổi luật chọn seed.
Kiểm tra pipeline: chế độ `fixed` cho AP **33.7991**, trùng khít số đã chọn trong `bb_swin_t_val_selection.json`.

| Chế độ | Ý nghĩa | AP | AP50 | recall IoU 0.5 |
|---|---|---|---|---|
| fixed | baseline | 33.80 | 63.97 | 0.736 |
| fixed_free (không GT) | baseline + gán pixel không giới hạn bán kính | 33.75 | 63.97 | 0.737 |
| nomerge | crown GT khác nhau không bao giờ loại nhau | 35.33 | 66.55 | 0.766 |
| nosplit | seed cùng một crown GT luôn bị loại | 34.74 | 65.93 | 0.737 |
| both | một seed cho mỗi crown GT | 36.21 | 68.61 | 0.764 |
| both_free | both + gán không giới hạn bán kính | **37.93** | 69.44 | 0.773 |

Mask R-CNN trên cùng validation: AP **38.21** (`mrcnn_s42_val_selection.json`).

1. **Luật seed cố định có thể đang chiếm gần hết khoảng cách.** Với seed oracle và gán pixel tự do, TreeFlowNet đạt 37.93 so với 38.21 của Mask R-CNN (validation). Đây là cận trên: nó dùng GT để biết hai peak thuộc hai cây hay một cây.
2. **Gộp quan trọng hơn chia** (nomerge +1.53, nosplit +0.94; không cộng thẳng vì hai chế độ chồng lấn: both +2.41).
3. **Bỏ giới hạn bán kính gán một mình không giúp** (fixed_free ≈ fixed). Khoản +1.72 giữa `both` và `both_free` chỉ có khi seed đã đúng: lúc đó pixel rìa của crown lớn bị rơi nếu bán kính gán bị chặn (recall 25–100 m²: 0.891 → 0.963).
4. **Peak trước NMS (luật fixed): crown nhỏ có peak riêng hay không.** Chỉ crown nhỏ bị mất ở cả hai tầng:

| m² | n_gt (validation) | có ≥1 peak trước NMS trong crown | còn seed sau NMS | mất ở trường (không peak) | mất ở NMS |
|---|---|---|---|---|---|
| 0.5–1 | 256 | 52.7% | 23.4% | 47.3% | 29.3 điểm |
| 1–2 | 1143 | 61.9% | 40.9% | 38.1% | 21.0 điểm |
| 2–4 | 2724 | 83.0% | 74.7% | 17.0% | 8.3 điểm |
| 4–9 | 4904 | 92.6% | 91.3% | 7.4% | 1.3 điểm |
| ≥ 9 | 6108 | ≥ 98% | ≥ 97.8% | ≤ 2% | ≤ 0.2 điểm |

   Với crown 1–2 m²: khoảng 38% không có peak nào trong crown (lỗi **trường**, chỉ huấn luyện lại mới sửa được) và khoảng 21 điểm có peak nhưng bị NMS loại (lỗi **giải mã**, sửa được không cần train). Tỉ lệ khoảng 2:1 nghiêng về lỗi trường. Ghi chú: "có peak trong crown" chưa bảo đảm peak đó là của riêng crown này.
5. **Phân loại crown nhỏ bị mất (test, `size_lost_cause.py`, `lostcause_*.json`).** 88–93% crown dưới 4 m² chạm biên crown khác (≤ 3 px). Trong crown chạm biên bị mất: 45–52% là "bị nuốt" (dự đoán phủ nó cũng phủ ≥ 50% một crown GT khác; Swin-T), 30–48% không có dự đoán nào phủ ≥ 20%. Crown đứng riêng hiếm khi bị mất. Mask R-CNN có tỉ lệ bị nuốt tương đương (47–61%), nên đây là khó khăn chung của dữ liệu này chứ không riêng TreeFlowNet.
6. **Bộ lọc `area_m2` = 1.0 m² (Swin-T) làm méo kết quả ở crown < 1 m²**: mọi dự đoán nhỏ hơn 1 m² bị loại trước khi chấm, nên tỉ lệ diện tích 6.5× ở bin 0.5–1 m² một phần là hệ quả của bộ lọc (ở 2–4 m² thì bộ lọc ít ảnh hưởng, nên tỉ lệ 1.25× và 32% "lớn hơn >25%" là hành vi thật). Selection trên validation đã thử 0.5 m² và chọn 1.0 m² vì AP cao hơn, nên hạ ngưỡng không tự động cải thiện.

**Cái gì còn chưa biết:** chỉ một seed và một checkpoint; "oracle" là cận trên, không hứa hẹn thực tế đạt được; chưa thử luật seed không dùng GT nào thực sự cải thiện (adaptive có sẵn không dùng được với v2, lý do ở đầu mục này).

### 9.3 Bộ chấm seed học được: kết quả TEST (Swin-T, 3 seed, chạy một lần)

Code: `clean_benchmark/seed_scorer.py` (kiểm tra trên validation, cross-validation theo tile) và `seed_scorer_test.py` (test); kết quả `results/swin_t_s4{2,3,4}_seed_scorer{,_test}.json`.
Giao thức chốt trước khi chạy test: bộ chấm (HistGradientBoosting, 15 đặc trưng của peak) fit trên **toàn bộ seed ứng viên của validation** của từng checkpoint, không dùng tập train và không dùng GT lúc suy luận; seed giữ khi p ≥ τ; pixel gán cho seed gần nhất; τ = 0.3 là số chính (tốt nhất trên cross-validation validation), τ = 0.5 là giá trị chốt từ đầu; điểm tin cậy giữ công thức cũ. Baseline qua cùng code tái hiện đúng AP test đã ghi (30.43 / 29.96 / 30.13).

| AP test | seed 42 | seed 43 | seed 44 | trung bình |
|---|---|---|---|---|
| fixed NMS (baseline) | 30.43 | 29.96 | 30.13 | 30.17 |
| bộ chấm τ = 0.3 (chính) | 32.05 | 31.74 | 31.78 | **31.85** |
| bộ chấm τ = 0.5 | 30.95 | 30.64 | 30.38 | 30.66 |

- **Mức tăng +1.62, +1.78, +1.65 AP (trung bình +1.68) với τ = 0.3**, cùng chiều ở cả ba seed; τ = 0.5 cho +0.5 trung bình. Tăng ở test (+1.7) gần bằng ở validation (+1.8), nên không thấy dấu hiệu overfit validation.
- **Khoảng cách tới Mask R-CNN (34.82 ± 0.67) còn khoảng 3.0 AP** (từ 4.65 xuống khoảng 3.0). Oracle trên validation gợi ý trần khoảng +4.3 AP, bộ chấm lấy được khoảng 40%.
- Hạn chế: một họ mô hình (Swin-T C5), bộ chấm huấn luyện trên chỉ 139 tile validation, τ = 0.3 chọn trên validation (không phải số chốt từ đầu; τ = 0.5, chốt từ đầu, chỉ +0.5). Chưa phân tích theo kích thước crown, chưa kiểm tra với mô hình khác. Không thể vá phần ~38% crown nhỏ không có peak nào.

---

## 10. Ba phép thử CPU về bản chất bài toán (R2, R5, R6), 03/10/2026

Code: `clean_benchmark/boundary_diagnostics.py`; kết quả `results/boundary_r2_{validation,test}.json`, `boundary_r5_*.json`, `boundary_r6_*.json`; log `logs/boundary_r*.log`.
Dự đoán đã có, seed 42, cấu hình đã chọn trên validation, mọi dự đoán (score ≥ 0), không train, không chọn lại siêu tham số. Crown chạm mép tile hoặc vùng không hợp lệ (≤ 2 px) bị loại ở R2/R5 vì diện tích bị cắt.
Quy tắc diễn giải viết trước khi chạy (docstring của script). Mô hình: TreeFlowNet Swin-T (NMS cố định), TreeFlowNet Swin-T + bộ chấm seed τ = 0.3 (chỉ có test), Mask R-CNN R50.

### R6. IoU có dung sai biên (bỏ qua pixel trong ±k px quanh biên GT) [đo]
Kiểm tra: k = 0 tái hiện đúng AP đã ghi (TreeFlowNet test 30.43, bộ chấm 32.05, val 33.80; Mask R-CNN val 38.21, test seed 42 34.29).

| AP | k = 0 | 1 px | 2 px | 3 px |
|---|---|---|---|---|
| test: Mask R-CNN | 34.29 | 37.57 | 40.41 | 42.70 |
| test: TreeFlowNet | 30.43 | 33.00 | 35.14 | 36.95 |
| test: TreeFlowNet + bộ chấm | 32.05 | 34.89 | 37.14 | 38.96 |
| **khoảng cách test (MRCNN − TFN)** | 3.86 | 4.57 | 5.27 | 5.75 |
| khoảng cách test (MRCNN − TFN + bộ chấm) | 2.24 | 2.68 | 3.27 | 3.74 |
| val: Mask R-CNN | 38.21 | 41.47 | 44.43 | 46.90 |
| val: TreeFlowNet | 33.80 | 36.46 | 38.56 | 40.48 |
| **khoảng cách val** | 4.41 | 5.01 | 5.87 | 6.42 |

Recall IoU ≥ 0.5 gần như không đổi khi nới biên (test TreeFlowNet 0.707 → 0.720; crown 1–2 m² 0.276 → 0.280; Mask R-CNN 0.748 → 0.756).
**Kết luận theo quy tắc:** khoảng cách **không thu hẹp mà nới rộng** khi nới biên ⇒ khoảng cách là lỗi thật (tìm/tách crown), không phải nhiễu vị trí biên. Dung sai chủ yếu nâng AP ở IoU cao cho mọi mô hình, và nâng Mask R-CNN nhiều hơn.

### R5. Thang kích thước có cấu trúc cục bộ không [đo]
| R² dự đoán log diện tích crown | validation (11.795 crown) | test (17.451 crown) |
|---|---|---|
| trung vị toàn cục | 0.00 | 0.00 |
| trung vị tile (bỏ chính nó) | 0.19 | 0.23 |
| trung vị các crown kề | −0.05 | 0.01 |
| hồi quy tile + kề (CV theo tile) | 0.23 (tile một mình 0.19) | 0.27 (tile một mình 0.24) |
| tương quan sau khi trừ trung bình tile | r = 0.19 | r = 0.20 |

**Kết luận theo quy tắc:** hàng xóm **không** vượt rõ baseline tile (chỉ thêm khoảng +0.035 R²) ⇒ thang chủ yếu là biến theo vùng; một ước lượng mỗi ảnh/tile đã nắm phần lớn, phần cục bộ có nhưng yếu. Hàng xóm một mình còn kém tile, có thể do crown lớn có nhiều hàng xóm nên bị lấy mẫu lệch [giả thuyết].

### R2. Bằng chứng biên có dự báo "bị nuốt" không [đo]
Cặp crown GT kề nhau (≤ 3 px, không chạm mép): 16.805 (val), 23.595 (test). "Bị nuốt": crown nhỏ không khớp và dự đoán khớp với crown kia phủ ≥ 50% crown nhỏ; "tách đúng": cả hai khớp. Đặc trưng là tỉ số so với nội tán. AUC > 0.5 nghĩa là bằng chứng biên nhiều hơn thì tách đúng.

Hệ số logistic của đặc trưng (đã chuẩn hoá, có kiểm soát log diện tích crown nhỏ; khoảng tin cậy 95% bootstrap theo tile; âm = biên rõ hơn thì ít bị nuốt hơn):

| | valley (rãnh tối giữa tán) | tex_diff | lab_diff | grad_ratio |
|---|---|---|---|---|
| val TreeFlowNet | −0.74 [−0.83, −0.66] | −0.70 | −0.49 | −0.16 |
| val Mask R-CNN | −0.78 [−0.86, −0.69] | −0.76 | −0.62 | −0.23 |
| test TreeFlowNet | −0.60 [−0.71, −0.51] | −0.49 | −0.39 | −0.28 |
| test TreeFlowNet + bộ chấm | −0.57 [−0.65, −0.49] | −0.55 | −0.48 | −0.30 |
| test Mask R-CNN | −0.71 [−0.80, −0.62] | −0.64 | −0.61 | −0.31 |

AUC theo bin kích thước: valley 0.61–0.79 ở mọi mô hình và bin; grad_ratio yếu nhất (0.42–0.62).

**Kết luận theo quy tắc:** biên yếu dự báo bị nuốt **ở cả hai mô hình**, với độ mạnh tương đương (Mask R-CNN thậm chí nhỉnh hơn) ⇒ đây là **độ khó nội tại của dữ liệu**, không phải điểm yếu riêng của cơ chế trường. Tín hiệu mạnh nhất là **rãnh tối giữa hai tán** (valley), không phải gradient biên; nhất quán với giả thuyết "mỗi tán là một vòm sáng, ranh giới là rãnh tối" nhưng chưa phải chứng minh (đặc trưng là tương quan, không phải can thiệp).
Điều khác giữa hai mô hình nằm ở **số cặp tách đúng**, không ở độ nhạy với biên: ở crown nhỏ dưới 1 m² trên test, Mask R-CNN tách đúng 219 cặp so với 22 của TreeFlowNet (bị nuốt 41 so với 81); ở 1–2 m²: 1.295 so với 565.

### Tổng hợp
1. Khoảng cách tới Mask R-CNN là lỗi thật (R6), không phải do nhãn biên lệch vài cm.
2. Thiếu bằng chứng biên làm khó **mọi** mô hình như nhau (R2). TreeFlowNet thua vì **tách được ít crown nhỏ hơn ngay cả khi biên rõ**, không phải vì nhạy hơn với biên yếu. Điều này chỉ về phía cơ chế đề xuất/seed (đã thấy ở oracle và bộ chấm) hơn là cơ chế "suy diễn biên".
3. Thang kích thước chủ yếu theo vùng (R5); một bộ ước lượng thang mức tile/ảnh có cơ sở hơn một thang theo từng hàng xóm.
4. Hạn chế: một seed mỗi mô hình; R2 bỏ qua nhóm "khác" (không bị nuốt cũng không tách đúng); các đặc trưng biên là tương quan, chưa can thiệp; bộ chấm chỉ có số test.
