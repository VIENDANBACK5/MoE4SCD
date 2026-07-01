# SECOND-OC — Literature Review (pre-work cho Vấn đề #2)

Tổng hợp ~24 paper liên quan trực tiếp hoặc gián tiếp đến SECOND-OC, chia theo mức độ liên quan. Mục tiêu: đủ để viết Related Work và xác định lại chính xác khoảng trống novelty.

**[CẬP NHẬT]** Đã đọc full-text AnyChange, SECOND-CC, và Referring Change Detection (RCD) — 3 paper áp sát nhất. Các entry Nhóm 1 bên dưới đã được sửa lại theo full-text (không còn dựa trên snippet). Thay đổi quan trọng nhất: SECOND-CC thực ra ĐÃ có nhãn class-transition dạng "low_veg → building" (30 category, không phải chỉ caption tự do như đánh giá lần trước) — nhưng chỉ 1 nhãn/cặp ảnh (cho thay đổi nổi bật nhất), không phải nhiều object/ảnh. Xem chi tiết bên dưới.

---

## ⚠️ Phát hiện ngoài phạm vi SECOND-OC — liên quan paper Token-MoE chính

**[Đã đọc full-text SCanNet/SCanFormer — đánh giá lại, bớt báo động hơn lần trước]** SCanNet (Ding et al., Information Engineering University + Univ. Trento, arXiv 2212.05245, "Joint Spatio-Temporal Modeling for Semantic Change Detection") gồm: 3 nhánh CNN (Triple Encoder-Decoder — semantic T1, semantic T2, change) → concat 3 đặc trưng → flatten thành "semantic token" (đơn giản là feature map được trải phẳng theo không gian, mỗi vị trí pixel/patch = 1 token, không liên quan SAM hay object) → SCanFormer áp dụng **CSWin self-attention** (1 biến thể windowed attention chuẩn, hoàn toàn không phải Mixture-of-Experts) lên các token đó để học tường minh quan hệ "from-to". Toàn bộ pipeline là dense/pixel-level, không có khái niệm object/instance, không dùng SAM/SAM2, không có expert routing nào. Tức là: **nếu Token-MoE thật sự là kiến trúc Mixture-of-Experts trên SAM2 tokens, 3 điểm khác biệt cốt lõi (MoE routing thay vì windowed self-attention / SAM2-derived tokens thay vì CNN-patch tokens / object-level thay vì luôn dense pixel-level) khiến SCanFormer không phải là "đã làm rồi" về mặt cơ chế** — chỉ trùng ở tầng ý tưởng chung ("model from-to transition tường minh trong không gian token"), vốn cũng là framing chung của cả mảng SCD chứ không phải IP riêng của SCanNet. Vẫn **bắt buộc cite** vì đây là baseline chuẩn mực nhất ngành dùng liên tục từ 2022 tới giờ (OA=87.86/mIoU=73.42/SeK=23.94/Fscd=63.66 trên SECOND) — reviewer chắc chắn sẽ hỏi so với SCanNet thì sao, dù không lo trùng kiến trúc.

**Phát hiện phụ quan trọng — không nhất quán về test split giữa các dòng paper:** SCanNet paper ghi rõ họ tách 1/5 dữ liệu, tức 593 cặp ảnh, làm test set và dùng phần còn lại (2.375 cặp) để train — tổng 2.968 (đúng số "có GT công khai"). Trong khi đó GSTM-SCD/TaCo lại mô tả split là 2.375 train / 593 val / **1.694 test** (tổng 4.662 = toàn bộ dataset). Tức là có khả năng **2 dòng baseline khác nhau đang dùng 2 test set khác nhau** (593 vs 1.694) mà không phải lúc nào cũng nói rõ. Việc này ảnh hưởng trực tiếp vấn đề #9 (chưa có baseline) — trước khi đưa bất kỳ con số nào (SCanNet, Bi-SRNet, ChangeStar2, TaCo...) vào bảng so sánh Phase 4, cần xác minh từng số liệu thực sự chạy trên đúng 1.694 test pairs hay trên 593, bằng cách kiểm tra trực tiếp code/data split của từng repo — không suy luận từ tên gọi "test set" trong bài.

---

## Nhóm 1 — Cạnh tranh trực tiếp, BẮT BUỘC cite + differentiate

**SECOND** (Yang et al., *TGRS* 2021, "Asymmetric Siamese Networks for Semantic Change Detection in Aerial Images") — Dataset gốc. 4.662 cặp ảnh từ Hàng Châu/Thành Đô/Thượng Hải, 2.968 cặp có nhãn công khai, split chuẩn 2.375/593/1.694 (train/val/test — đúng test set bạn đang dùng). 6 lớp: non-vegetated ground surface, tree, low vegetation, water, building, playground. Paper gốc cũng đề xuất **SeK (Separated Kappa)** làm metric chuẩn để xử lý label imbalance — xem Nhóm 6. Trang chính thức: captain-whu.github.io/SCD/

**SECOND-CC** (Karaca et al., Yildiz Technical University, arXiv 2501.10075, 1/2025, repo github.com/ChangeCapsInRS/SecondCC) — **[Đã đọc full-text, sửa lại đáng kể so với lần trước]**. Không chỉ caption tự do: mỗi cặp ảnh có thêm **1 nhãn class-transition** kiểu "low vegetation → building" (30 category — gần như khớp con số 30 trong plan gốc), cộng 5 caption mô tả. Nhưng nhãn này chỉ gắn cho **thay đổi nổi bật nhất** trong cặp ảnh ("the most significant change"), bỏ qua các thay đổi đồng thời khác — không phải multi-object. Về split: SECOND-CC chia ảnh 512×512 gốc thành 4 quadrant 256×256, rồi tự chia lại theo tỷ lệ 7:1:2 (4.219/595/1.227 = 6.041 tổng) — **khác hẳn** split chuẩn SECOND 512×512 mà gần như toàn bộ literature SCD dùng (2.375/593/1.694, đúng test set bạn đang dùng). Tức là không có overlap ảnh-đối-ảnh với 1.694 test pairs, dù cùng nguồn dữ liệu gốc. *Differentiate:* (1) định vị "official SECOND split, khớp với toàn bộ SCD baseline literature" thay vì "custom-cropped split" — điểm sạch và mạnh; (2) "exhaustive per-object enumeration" (mọi object thay đổi, không chỉ 1 cái nổi bật nhất) là khác biệt chính, không phải "có description hay không" nữa. Bắt buộc cite trực diện và nêu rõ 2 điểm này.

**AnyChange / "Segment Any Change"** (Zheng et al., Stanford + Wuhan University, NeurIPS 2024, arXiv 2402.01188) — **[Đã đọc full-text — phát hiện quan trọng nhất cho fix #3]**. Cơ chế thật sự (Bitemporal Latent Matching) **không** phải bipartite-match-rồi-so-sánh như Phase 2 của plan: với mỗi object mask m (lấy từ SAM ở T1 hoặc T2), họ áp dụng **đúng mask đó** lên embedding ảnh ở cả 2 thời điểm (z_t[m] và z_{t+1}[m]) rồi so cosine similarity — tận dụng trực tiếp việc T1/T2 đã co-registered, không cần "tìm" object tương ứng ở ảnh kia. Họ tự build baseline "SAM+Mask Match" (IoU giữa mask T1 và T2 riêng biệt, not-match → coi là changed) để so sánh — **đúng cách làm của Phase 2 hiện tại** — và nó thua rất xa: trên SECOND (ViT-H), SAM+Mask Match đạt F1=14.2/mask AR=3.7 trong khi AnyChange (latent matching) đạt F1=41.8/mask AR=29.0. Đây là bằng chứng số liệu trực tiếp, ngay trên SECOND, rằng hướng IoU-matching mà critique cũ nghi ngờ (#3) thực sự kém hẳn — và đồng thời gợi ý hướng fix tốt hơn cả "centroid distance" lẫn "pixel overlap many-to-many" tôi đề xuất trước: so sánh embedding của cùng 1 mask ở 2 thời điểm thay vì match 2 mask khác nhau. Method dùng SAM (ViT-B/L/H), chưa dùng SAM2 — cơ hội differentiate kỹ thuật thật. Object proposal evaluation vẫn binary (mask AR@1000, không gán class-transition) — không động đến phần semantic labeling của bạn. Baseline khác đáng ghi nhận từ related work của họ: **SAM-CD** (Ding et al., arXiv 2309.01429, PEFT fine-tuning SAM cho CD).

**SeFi-CD** (Zhao et al., *Remote Sensing* 2024, arXiv 2407.09874) — Đề xuất paradigm "semantic-first": dùng VLM để hiểu ngữ nghĩa vùng quan tâm trước, rồi mới tìm thay đổi thị giác tương ứng (thay vì ngược lại như các method cũ). Model AUWCD vượt SOTA supervised trung bình 5.01% F1 trên SECOND (tối đa 13.17%). *Differentiate:* đây là method, không phải benchmark/dataset — nhưng là baseline số liệu mạnh để so sánh, và route "VLM hiểu semantic trước" gần với Phase 3B của bạn.

**Referring Change Detection (RCD)** (Korkmaz et al., JHU + US Army Research Lab, arXiv 2512.11719, 12/2025 — rất mới) — **[Đã đọc full-text — sửa lại cơ chế]**. Lưu ý quan trọng: RCD **không sinh caption/mô tả tự nhiên** — model nhận 1 tên class làm prompt (qua CLIP text embedding) và trả về **bản đồ binary change cho riêng class đó** (giống "point query" của AnyChange nhưng dùng text thay vì click). Nên không cạnh tranh trực tiếp với Phase 3 (description generation) của bạn, chỉ cạnh tranh ở khía cạnh "linh hoạt theo class". Họ chỉ rõ đúng vấn đề #11 critique cũ nêu: định nghĩa lớp không nhất quán giữa dataset SCD (vd. "impervious surface" và "bare ground" tách riêng ở CNAM-CD nhưng gộp thành "non-vegetated ground surface" ở SECOND), và cho số liệu mất cân bằng lớp cụ thể trên SECOND: lớp "non-vegetated ground surface" chiếm 43% tổng diện tích thay đổi và xuất hiện ở 2.689 ảnh, trong khi "playground" chỉ 0.38% và xuất hiện ở 129 ảnh — đáng trích dẫn khi bàn về class imbalance/min_area (vấn đề #7, #11). Eval protocol dùng 4 metric chuẩn cho SECOND: **OA, mIoU, SeK, F_scd** (xem Nhóm 6). Trong Related Work của họ có nhắc tới **SCanNet/SCanFormer** — xem cảnh báo riêng ở đầu file. *Differentiate cho SECOND-OC:* vẫn giữ được điểm "kết hợp structured label + description tự nhiên + object-level", vì RCD chỉ làm phần "structured, theo class, pixel-level" — không đụng object-level lẫn ngôn ngữ tự do.

---

## Nhóm 2 — Methodology object/instance-level (liên quan Phase 1 fix)

**RSPrompter** (Chen et al., *TGRS* 2024, arXiv 2306.16269) — Học cách tạo prompt cho SAM để ra mask instance có gán semantic category trên ảnh viễn thám (SAM gốc chỉ category-agnostic). Test trên WHU building/NWPU VHR-10/SSDD, không phải SECOND. *Liên quan:* căn cứ kỹ thuật bổ sung cho việc chuyển Phase 1 sang dùng SAM2 — cho thấy "SAM cần học thêm prompt mới ra mask đúng semantic trên RS" là vấn đề đã biết, đáng cân nhắc khi thiết kế bước gán dominant-class cho SAM2 mask.

**SceneDiff** (arXiv 2512.16908, cuối 2025) — Benchmark change detection multiview *đầu tiên* với annotation object instance-level dày đặc (350 video pairs), kèm công cụ annotation dựa trên SAM2 và eval protocol per-view/per-scene. *Liên quan:* không phải remote sensing (indoor/outdoor scene thường), nhưng đúng ý tưởng "object-instance-level change benchmark bằng SAM2" — nên đọc kỹ eval protocol của họ để tham khảo thiết kế Object-F1, và cite như tín hiệu hướng đang nóng.

**SAGE-CC** (arXiv 2511.21420, "SAM Guided Semantic and Motion Changed Region Mining for RS Change Captioning") — Dùng SAM để xác định vùng thay đổi semantic/motion một cách tường minh trước khi caption, kết hợp knowledge graph từ LLM làm context. *Liên quan:* gần nhất với Phase 1→3 pipeline của bạn (SAM region → caption), đáng đọc full-text để tránh trùng lặp pipeline design.

---

## Nhóm 3 — Change captioning landscape (liên quan Phase 3 / định vị "captioning trên RS đã bão hòa cỡ nào")

**LEVIR-CC** (Liu et al. 2024) — Dataset captioning RS gốc, phổ biến nhất: 10.077 cặp ảnh, 5 caption/cặp (50.385 câu), xây từ LEVIR-CD (chỉ building, binary). Hầu hết các paper dưới đây train/eval trên đây.

**Semantic-CC** (arXiv 2407.14032) — Boost RSICC bằng foundational-knowledge + semantic guidance, train/eval trên LEVIR-CD/LEVIR-CC.

**SAT-Cap** (arXiv 2501.08114) — Single-stage transformer cho change captioning, so sánh trên LEVIR-CC + DUBAI-CCD (500 cặp, ảnh 50×50, đô thị Dubai).

**Diffusion-RSCC** (arXiv 2405.12875) — Diffusion probabilistic model cho captioning, train/eval trên LEVIR-CC.

**Mask Approximation Net** (arXiv 2412.19179) — Diffusion-based change captioning, thêm dataset WHU-CDC (ảnh độ phân giải rất cao, 0.075m, từ WHU-CD).

**Change3D** (arXiv 2503.18803) — Coi CD + captioning như bài toán video modeling. Liệt kê khá đầy đủ các dataset CD nhị phân (LEVIR-CD, WHU-CD, CLCD) và SCD (HRSCD — 291 cặp ảnh rất lớn 10.000×10.000, crop thành ~44.785/14.928/14.928). Đáng xem HRSCD như benchmark SCD thứ hai để mention bên cạnh SECOND.

→ **Nhận xét chung Nhóm 3:** gần như toàn bộ literature change-captioning hiện có build trên LEVIR (binary, chỉ building). Không có ai làm captioning multi-class land-cover ở cấp object — đây là chỗ trống thật, miễn phân biệt rõ với SECOND-CC (1 nhãn + caption cho thay đổi nổi bật nhất/cặp ảnh, không phải exhaustive per-object — xem Nhóm 1).

---

## Nhóm 4 — VLM/LLM interactive change analysis (liên quan Tier-1 / Phase 3B)

**ChangeChat** (Deng et al., arXiv 2409.08582, 9/2024) — VLM song thời điểm đầu tiên cho RS change analysis, multimodal instruction tuning, dataset ChangeChat-87k, xử lý được category-specific quantification và change localization. Build trên LEVIR.

**Change-Agent** (Liu et al., *TGRS* 2024) — Tích hợp MCI model (mắt) + LLM (não), hỗ trợ change detection, captioning, đếm object thay đổi, phân tích nguyên nhân. Build trên LEVIR.

**CDChat** (Noman et al., arXiv 2409.16261, 9/2024) — LMM cho RS change description, dùng GeoChat làm nền, chỉ ra GeoChat gốc yếu ở mô tả semantic change vì thiếu dữ liệu hội thoại bitemporal.

**CDQAG / "Show Me What and Where has Changed?"** (arXiv 2410.23828) — Đề xuất task Change Detection QA + Grounding (CDQAG): vừa trả lời "thay đổi gì" (text) vừa "ở đâu" (mask/box), dataset QAG-360K. Họ tự nhận xét RS land-cover hay trải rộng/fragment hơn ảnh tự nhiên nên grounding khó hơn object-level thường. *Liên quan:* gần với việc kết hợp structured + free-text + spatial localization mà SECOND-OC nhắm tới, nên đọc kỹ.

**Forest-Chat**, **TerraScope** — agent/VLM chuyên biệt cho phân tích thay đổi rừng và pixel-grounded reasoning cho EO nói chung; mức liên quan thấp hơn nhưng nên có trong câu "interactive RS change analysis đang phát triển nhanh" ở phần mở đầu Related Work.

---

## Nhóm 5 — Dễ nhầm tên / bối cảnh, liên quan thấp

**MMChange** (arXiv 2509.03961) — Trùng tên với "MMChange" critique cũ nhắc tới, nhưng là method khác hẳn: dùng VLM sinh mô tả semantic cho ảnh song thời điểm để hỗ trợ detection pixel-level, eval bằng P/R/IoU/F1 chuẩn trên LEVIR-CD/WHU-CD/SYSU-CD, không đụng SECOND, không phải benchmark/captioning.

**GSTM-SCD** (9/2025) và **CSD / Gaza-Change** (11/2025) — hai method SCD rất mới, đều report số trên SECOND. Đáng lấy số liệu của 2 paper này làm thêm baseline candidate cho Phase 4 (cùng với BIT-CD/ChangeFormer critique cũ đề xuất).

---

## Nhóm 6 — Tiền lệ metric (liên quan trực tiếp vấn đề #4/#9)

**[Đã đọc full-text RCD + SCanNet — bổ sung đầy đủ hơn]** Chính paper gốc SECOND đã đề xuất **SeK (Separated Kappa)** để xử lý label imbalance trong SCD. Bộ 4 metric chuẩn literature SCD trên SECOND dùng là **OA, mIoU, SeK, F_scd** — công thức chính xác (định nghĩa qua confusion matrix, dùng trong cả paper gốc lẫn SCanNet) đã có sẵn trong file SCanNet (arXiv 2212.05245, mục IV-B, eq. 11-20), không cần đoán lại từ code người khác. **Khuyến nghị:** Object-F1/Semantic-Object-F1 nên trình bày tương quan rõ với cả 4 metric này.

SCanNet paper tự cung cấp luôn 1 bảng baseline 10 method trên SECOND (OA/mIoU/SeK/Fscd đầy đủ): ResNet-GRU/LSTM, FC-Siam-conv/diff, HRSCD-str.2/3/4, SCDNet, SSCD-l, Bi-SRNet, TED, SCanNet — tiện lợi hơn nhiều so với gom từ nhiều nguồn, **nhưng nhớ kiểm tra split** (xem cảnh báo ⚠️ đầu file) trước khi đem so trực tiếp với baseline tự chạy trên 1.694 test pairs. Paper/baseline khác đáng lấy số liệu: **ChangeStar2** (arXiv 2406.15694), **TaCo** (arXiv 2511.20306, 11/2025, tự nhận đúng split 2.968 train/1.694 test), GSTM-SCD, CSD/Gaza-Change. Dataset SCD khác ngoài SECOND để mention: **CNAM-CD** (Zhou et al. 2023, "Signet" paper), **Landsat-SCD** (Yuan et al. 2022, dùng chính trong SCanNet làm dataset thứ 2), và **HRSCD** (Daudt et al., 291 cặp ảnh rất lớn).

---

## Tổng hợp — khoảng trống novelty (bản cập nhật sau full-text)

Không còn paper nào claim được "đầu tiên" cho từng mảnh riêng lẻ:
- "Object/instance-level trên SECOND" → AnyChange đã làm (NeurIPS 2024), kèm bằng chứng số liệu rằng cách matching IoU-based (như Phase 2 hiện tại) kém hẳn latent-matching.
- "Có nhãn class-transition trên SECOND" → SECOND-CC đã làm (1/2025), nhưng chỉ 1 nhãn/cặp ảnh cho thay đổi nổi bật nhất, trên split khác (256×256 quadrant, không phải 1.694 test pairs gốc).
- "Linh hoạt theo class thay vì cố định" → RCD đã làm (12/2025), nhưng ở dạng class-prompted binary segmentation, không sinh ngôn ngữ tự nhiên.

Cái chưa ai làm, sau khi đối chiếu kỹ cả 3: **exhaustive per-object enumeration** trên đúng split chuẩn SECOND (1.694 test pairs) — mọi object thay đổi trong ảnh, không chỉ 1 cái nổi bật nhất — kết hợp nhãn class-transition có cấu trúc + caption tự nhiên theo từng object + eval protocol Object-F1/Semantic-F1 (đối chiếu được với OA/mIoU/SeK/F_scd chuẩn) đóng gói thành benchmark release. Hẹp hơn nhiều so với "first benchmark" ban đầu nhưng giờ có ranh giới rõ ràng, chính xác, và 3 điểm khác biệt cụ thể (split chuẩn / exhaustive thay vì single-label / object-level thay vì pixel-level) thay vì chỉ nói chung chung "object-centric".

## Việc còn thiếu trước khi viết Related Work chính thức

1. ~~Đọc full-text AnyChange + SECOND-CC + RCD~~ — Xong, xem các entry Nhóm 1.
2. ~~Xác nhận SECOND-CC có overlap với 1.694 test pairs~~ — Không overlap ảnh-đối-ảnh (khác crop size + split ratio), nhưng cùng nguồn dữ liệu gốc — nên nói rõ trong Related Work để tránh reviewer hiểu nhầm là trùng lặp.
3. ~~Đọc SCanNet/SCanFormer~~ — Xong, rủi ro kiến trúc thấp hơn lo ban đầu (xem cảnh báo ⚠️ đầu file), nhưng phát sinh việc mới quan trọng hơn (#4 bên dưới).
4. **[MỚI, ưu tiên cao]** Xác minh trực tiếp trên data folder của bạn: 1.694 test pairs có đúng là cùng tập mà ChangeStar2/TaCo/GSTM-SCD dùng không, hay khác với tập 593-pair mà dòng SCanNet/Bi-SRNet dùng? Việc này quyết định baseline nào dùng được trực tiếp cho Phase 4, baseline nào cần tự chạy lại.
5. Venue fit: với delta hẹp hơn, NeurIPS D&B Track có thể quá tham vọng cho riêng phần dataset — cân nhắc TGRS (nơi cả SECOND gốc, Change-Agent, RSPrompter, SCanNet đều đăng) hoặc workshop trước, để paper chính (Token-MoE) đứng vững hơn benchmark phụ.