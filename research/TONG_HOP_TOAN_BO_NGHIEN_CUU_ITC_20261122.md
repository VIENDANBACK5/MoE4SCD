# Tổng hợp toàn bộ nghiên cứu ITC (RGB, tách tán cây cá thể), tính đến 22/11/2026

Mục đích: giúp chọn **đánh vào kết quả nào, giả thuyết nào, ở cấp nào**. File chỉ tổng hợp, không chạy gì mới.

## 0. Cách đọc

### 0.1 Nguồn và độ tin cậy
- Mọi dòng đều dẫn về thư mục trong `research/` hoặc thẻ `EXP-0xx` trong
  `crown_segmentation_research/target/NHAT_KY_THI_NGHIEM.md`.
- Mã quyết định (ví dụ `NOT_SUPPORTED`) được **chép nguyên** từ file quyết định của nhánh.
- Ba mức độ tin cậy:
  - **[đọc kỹ]**: agent đã làm hoặc đọc đầy đủ nhánh này (từ 19/11 trở đi: EXP-028 → 032, PICE, CCF, T4).
  - **[đọc dòng quyết định]**: chỉ đọc mã quyết định và vài dòng kết luận. Cột "cấp bác bỏ" khi đó là ước lượng của
    agent, cần đối chiếu file gốc trước khi dùng trong paper.
  - **[đang chạy]**: nhánh của luồng khác, chưa có kết luận.

### 0.2 Năm cấp bác bỏ
Phạm vi kết luận và yêu cầu bằng chứng tăng dần từ cấp 1 đến cấp 5.

| Cấp | Tên | Kết quả âm cho phép nói gì |
|---|---|---|
| 1 | Implementation | Code hoặc cách cài đặt cụ thể này không chạy được hoặc không hiệu quả |
| 2 | Mechanism | Cơ chế đã thiết kế không tạo hiệu ứng mong muốn, trong đúng điều kiện đã thử |
| 3 | Formulation | Cách đặt bài toán hoặc biểu diễn này không phù hợp, trong phạm vi đã thử |
| 4 | Principle | Nguyên lý nền sai trong phạm vi giả định đã nêu. Cần oracle hoặc can thiệp trực tiếp |
| 5 | Impossibility | Không thể giải với thông tin đã có. Cần chứng minh giới hạn thông tin |

### 0.3 Bốn trạng thái
- **CLOSED**: không chạy lại cùng recipe hay cùng formulation, kể cả khi chỉ đổi hyperparameter.
- **RESTRICTED**: chỉ thất bại dưới điều kiện cụ thể. Chỉ mở lại khi đổi giả định hoặc cơ chế có lý do khoa học.
- **OPEN**: nguyên lý chưa bị bác bỏ.
- **IMPOSSIBLE**: đã chứng minh giới hạn. **Hiện chưa có hướng nào ở trạng thái này.**

---

## 1. Một trang tóm tắt

**Kết quả chuẩn** (QuebecTrees test, 3 seed, chạy một lần): TreeFlowNet (TFN, Swin-T) **34.22 ± 0.30 AP**; Mask R-CNN
R50 **34.82 ± 0.67**. Chênh lệch +0.59 [0.15; 1.02]. Hai mô hình chưa công bằng về backbone.

**Đóng góp đã được ủng hộ trong pipeline TFN:**
- Gộp vùng hút theo đèo SDT (EXP-016): test +2.41.
- Bộ chấm điểm instance (EXP-023).
- Cổng SDT (EXP-025).

**Failure mode đã cô lập rõ nhất:** tán nhỏ (< 2 m²) nằm sát tán lớn bị nuốt. Năm bằng chứng:
- Thông tin sửa được lỗi là **trường thành viên** (flow), cộng với SDT để tách. Biết đúng tâm cây cứu được **0/210**
  tán (EXP-029).
- Lỗi xảy ra **cả trên ảnh train** (EXP-030).
- Mask R-CNN cũng hỏng ở các tán này (EXP-030).
- SAM, khi được cho đúng tâm, chỉ tách được khoảng 1/4 (EXP-032).
- Tán nhỏ bị nuốt cả khi so với tán vật lý TLS (T4).

**Chưa có phương pháp gốc nào được thiết lập.** Đây cũng là kết luận hợp nhất ngày 07/11:
`NO_NEW_METHOD_ESTABLISHED — INSTANCE_LEVEL_DECISION_ALIGNMENT_REMAINS_UNTESTED`. Gần như **mọi** kết quả âm đều ở
cấp 1–3, vài cái ở cấp 4. **Không cái nào ở cấp 5.**

**Hai câu hỏi quyết định loại paper:**
1. Có cơ chế nào **sửa được bước tách (khâu A)** cho tán nhỏ sát tán lớn không? → paper phương pháp.
2. Ở những chỗ đó, ảnh RGB **có thật sự cho thấy** tán riêng không? → nếu không, paper giới hạn (cấp 5).

---

## 2. Pipeline TFN và các thí nghiệm EXP-001 → 027 (dòng QuebecTrees) [đọc kỹ / đọc dòng quyết định]

| EXP | Câu hỏi | Kết quả | Cấp | Trạng thái |
|---|---|---|---|---|
| 001–004 | Chẩn đoán biên (R2/R5/R6), lặp seed | Mô tả | — | — |
| 003 | Bảng thác nước tán nhỏ bị mất | Tầng "không có ứng viên" lớn nhất ở < 2 m² | mô tả | — |
| 005/005b | H-001 (loss cân theo tán) trên thế giới tổng hợp | **Bộ thử không hợp lệ**: baseline không tái hiện lỗi | không bác bỏ gì | **OPEN** cho H-001; xem EXP-031b |
| 006/007 | Tán nhỏ bị cổng tiền cảnh chặn (H-002) | Ủng hộ một phần: nới cổng thì recall +0.03 | 2 | Đã dùng (EXP-025) |
| 008/008b | Phân tầng S1–S6 theo tiếp xúc và độ rõ biên | TFN không có điểm yếu riêng khi biên vắng | 2 | CLOSED |
| 010/011/012 | Tán không lồi bị chia (H-003) | **Ủng hộ**, cả tổng hợp lẫn thật (D = +0.017) | — | điểm yếu đã biết của biểu diễn |
| 013 | Phân rã khoảng cách MRCNN − TFN | < 2 m²: gộp (74%); 2–4: bỏ sót; ≥ 9: chia | mô tả | — |
| 014 | Tán bỏ sót 2–9 m² có điểm thấp không | Hỗn hợp: khoảng 2/5 là lỗi điểm | mô tả | — |
| 015 | Rò rỉ không gian giữa các split | Không đáng kể (viền 1 px) | — | — |
| **016** | Gộp theo đèo SDT thay NMS (H-004) | **Ủng hộ**: val +2.85, test +2.41 [2.06; 2.80] | — | **đóng góp đã có** |
| 017 | Oracle cỡ tán | +0.56 so với đèo, không kết luận | 4 (trần) | CLOSED cho "đầu cỡ tán" |
| 018 | Còn tín hiệu cặp ngoài đèo không | AUC 0.81 ở tán ≥ 4 m² | mô tả | — |
| 019 | Mắt agent có khớp số liệu không | κ = 0.72, chỉ dùng cho kết luận thô | — | — |
| 020/021 | Gộp bằng bộ phân loại cặp | Không hơn đèo +0.3 | 2 | CLOSED |
| 022 | Oracle chấm lại instance | +10.2 AP: xếp hạng là nút thắt | 4 (trần) | dẫn tới 023 |
| **023/024** | Bộ chấm điểm instance học được | **Ủng hộ**. Khoảng cách công bằng tới MRCNN cũng chấm lại: 1.06 | — | **đóng góp đã có** |
| **025** | Cổng SDT + chấm lại | **Ủng hộ**: val +0.51, recall 1–2 m² +0.06 | — | **đóng góp đã có** |
| 026 | Lỗi chia do target hay do dự đoán | Trường GT cho 91.8 AP, chia = 0: lỗi do **dự đoán** | 4 (oracle) | target Green không ưu tiên |
| 027 | Bộ dựng mask kiểu Omnipose so với đèo | (xem thẻ) | 2 | — |

---

## 3. Failure mode đã cô lập: chuỗi bằng chứng (19–22/11) [đọc kỹ]

Đây có thể là **phần đặt vấn đề của paper**.

| Bằng chứng | Kết quả | Nguồn |
|---|---|---|
| Phân tách nguồn lỗi (TFN s42, 4,301 tán duy nhất) | Tổng: mask 24%, gán 12%, định vị 8%, không giả thuyết 2.3%. **Tán < 2 m²: định vị 28%, không giả thuyết 12%, gán 14%, mask 26%** | ITC_INSTANCE_FORMATION_V1 |
| F1: lỗi chung giữa các mô hình | 86% lỗi < 1 m² là lỗi chung TFN–MRCNN | ITC_IDENTITY_ERROR_ORIGIN_V1 |
| F2: DSM | DSM quang trắc không thêm AUC (0.812 so với 0.811) | ITC_ERROR_CASE_TAXONOMY_V1 |
| F3: quy ước nhãn | Làm trơn kiểu QuebecTrees làm 17–78% tán < 9 m² tụt dưới IoU 0.75 | ITC_TLS_ANNOTATION_GEOMETRY_V1 |
| T4: tham chiếu vật lý TLS | Tán nhỏ bị nuốt cả với tán TLS (TFN 5.7×, MRCNN 8.8×) | ITC_THREE_TRACKS_V1/track1_tls |
| CCF / G0 | Gộp hay cắt trên cây phân cấp TFN có trần +1.72 AP; recall tán nhỏ **không** tăng | ITC_INSTANCE_FORMATION_V1 |
| EXP-028b (M3) | Loss canopy theo tán: ΔR < 2 m² +0.025 (dưới ngưỡng 0.03), ΔAP +0.19 | track3_formation |
| **EXP-029** (thang bơm thông tin oracle) | Riêng flow phục hồi 40% tán H. **S+F phục hồi 48/48 tán H** và 83% tán A; C+F 54% tán L. Không có flow thì không điều kiện nào vượt 25%. **Biết đúng tâm cây: 0/210** | ITC_INFORMATION_LADDER_V1 |
| **EXP-030** | Trên **train**, tỉ lệ H+A là 0.336, trên val 0.283 (không fit). MRCNN trúng 9% tán H+A, chỉ 1/48 tán H. Tương phản màu không khác (AUC 0.49) | ITC_INFORMATION_LADDER_V1 |
| **EXP-031b** (method 1: loss flow/SDT cân theo tán) | H 50 → 35 nhưng A 60 → 62. Trên train, H+A gần như không đổi (710 → 699). AP −1.26, recall < 2 m² +0.055 | track3_formation |
| **EXP-032** (SAM với đúng tâm tán) | Mask nhỏ nhất: trúng 26% tán H+A so với 39% tán đúng. SAM tự chọn mask thì hay lấy cả mảng tán | ITC_INFORMATION_LADDER_V1 |

**Diễn giải gộp:**
- Lỗi nằm ở **khâu tách** (A) và **khâu tạo giả thuyết** (H) của tán nhỏ sát tán lớn.
- Thông tin thiếu là **thành viên + tách**, không phải tồn tại, cỡ hay số đếm.
- Cân lại loss theo pixel (M3, method 1) không sửa được, vì quyết định gộp được đưa ra ở **cấp cặp vùng hút** (độ cao
  đèo), còn loss thì tính ở cấp pixel.
- Khoảng 70–75% tán lỗi chưa hệ nào tách được.

---

## 4. Sổ các hướng theo họ

### A. Gộp / phân hoạch / cấu trúc (graph, partition, uncertainty)
| Hướng | Nhánh | Đã thử gì | Mã quyết định | Cấp (ước lượng) | Trạng thái | Mở lại khi |
|---|---|---|---|---|---|---|
| Grouping baseline (multicut, GAEC) | grouping_baseline_study | Thiết kế; chưa được duyệt chạy | BLOCKED | — | **OPEN** | — |
| Evidence → partition | ITC_EVIDENCE_TO_PARTITION_V1 | Học từ cặp khó | `INSUFFICIENT_HARD_PAIR_SUPERVISION` (nhánh tự ghi: *không phải* bằng chứng multicut thất bại) | 1 (dữ liệu) | RESTRICTED | Có giám sát cặp đủ |
| Decision-aligned partition (2×2) | ITC_DECISION_ALIGNED_PARTITION_V1 | Structured hinge, local-edge cân theo instance | `STRUCTURED_PARTITION_FORMULATION_NOT_SUPPORTED` ("closes the tested recipe; does not falsify every form of decision-aligned instance learning") | 2–3 | CLOSED recipe; **nguyên lý OPEN** | Loss theo lỗi instance thật (IoU/FP) |
| Instance objective (finite bank) | ITC_INSTANCE_OBJECTIVE_PILOT_V1 | Chọn trong ≤ 13 phân hoạch dựng sẵn | `FINITE_BANK_INSTANCE_OBJECTIVE_NOT_SUPPORTED`; `UNRESTRICTED_INSTANCE_LOSS_GRAPH_LEARNING_NOT_RUN` | 2 | RESTRICTED | Bỏ giới hạn bank |
| Counterfactual individuation | ITC_COUNTERFACTUAL_INDIVIDUATION_V1 | Cây nhị phân, giám sát 1-vs-2 | `COUNTERFACTUAL_FORMATION_NOT_SUPPORTED_AS_IMPROVEMENT` (lệch giám sát / suy luận) | 2–3 | CLOSED recipe | — |
| CCBR | CCBR_VALIDATION_V1 | Solver số học | `SOLVER_CORRECTNESS_FAIL` (nhánh tự ghi: không phải phủ định formulation) | **1** | RESTRICTED | Solver đạt certificate |
| PICE | ITC_IDENTIFIABLE_STRUCTURE_V1 | Chứng nhận khoảng phân hoạch trên cây đèo | Lý thuyết CONDITIONAL GO; ITC NO-GO (−0.098 / −0.106 / −0.009) | 3 (ITC) | CLOSED cho ITC; **lý thuyết OPEN** | Tham chiếu TLS, hoặc đầu nút khác |
| CCF | ITC_INSTANCE_FORMATION_V1 | Cắt theo số đếm trên cây TFN | G0 +1.72 (< 2) | 3–4 (trần, **trên cây TFN**) | CLOSED trên nền hiện tại | Nếu sửa được khâu hình thành mảnh |
| MALIS (pairwise / structured identity) | ITC_TFN_STRUCTURED_IDENTITY_V1 | Thế giới tổng hợp | Chạm sàn, không mang thông tin | không bác bỏ gì | **OPEN** | — |
| Containment set | ITC_CONTAINMENT_SET_V1 | Luật hình học loại "phần trong toàn thể" | `NO_PRACTICAL_GAIN` (dư địa ≥ 2 có, nhưng luật hình học hại) | 2 | CLOSED | — |
| Part–whole rescoring | PWR_PART_WHOLE | Tương phản trường | M−B +0.25 (oracle +2.69) | 2 | CLOSED | — |
| Part-negative | ITC_PART_NEGATIVE_V1 | Lớp "phần" khi train | AP −1.82 | 2 | CLOSED | — |
| Part fusion | ITC_PART_FUSION_V1 | Ghép điểm 2 mô hình | +0.53 (cần +1) | 2 | CLOSED (đã biết) | — |
| Centre-consistent rescoring | ITC_CENTRE_CONSISTENCY_V1 | Bản đồ tâm để chấm lại | v1/v2 NEGATIVE (oracle +4.60) | 2 | CLOSED | Tâm thuộc khâu hình thành, không thuộc chấm lại |

### B. Loss / học trên TFN (dòng QuebecTrees)
| Hướng | Nhánh | Kết quả | Cấp | Trạng thái |
|---|---|---|---|---|
| H-001 trên thế giới tổng hợp | EXP-005/005b | Bộ thử không hợp lệ | — | — |
| M3: canopy theo tán + Lagrange | EXP-028b | Dưới ngưỡng; nhiễu snapshot 0.03–0.05 | 1–2 (thiếu độ nhạy) | RESTRICTED |
| Method 1: flow/SDT theo tán | EXP-031b | H giảm, A không; train không đổi; AP −1.26 | 2, và 4 cho "gradient domination" | CLOSED recipe; **H-001 thu hẹp** |
| **Căn chỉnh quyết định cấp instance trên đèo SDT** | chưa có | — | — | **OPEN, chưa từng thử** |

### C. Carrier dạng query (Mask DINO / EoMT / DETR), chủ yếu trên SelvaMask-type (TBS/BCI/ZF2) [đọc dòng quyết định]
| Hướng | Nhánh | Kết quả ghi nhận | Cấp | Trạng thái |
|---|---|---|---|---|
| Query birth | ITC_QUERY_BIRTH_LEARNABILITY_V1 | `ORACLE_BIRTH_LEARNABLE_RGB_WITNESS_OR_TRANSFER_NOT_ESTABLISHED`; học được có điều kiện (+1.1 recall tán nhỏ) | 2 | RESTRICTED |
| ECRQ (residual witness, query recycling) | ITC_ECRQ_V1 | NOT_SUPPORTED | 2 | CLOSED recipe |
| Set supervision / set reasoning | ITC_SET_SUPERVISION_DEEP_V1, ITC_SET_REASONING_DECISION_V1 | Không có đóng góp; tín hiệu hẹp phụ thuộc budget | 2 | CLOSED |
| Instance reconstruction / local formation / predicted-box local | ITC_INSTANCE_RECONSTRUCTION_STUDY_V1, ITC_LOCAL_FORMATION_V1, ITC_PREDICTED_BOX_LOCAL_CONDITIONING_V1 | PROTOTYPE_NOT_SUPPORTED; cơ hội < 2 pp | 2 | CLOSED |
| Carrier mechanisms (R5, RoI) | ITC_CARRIER_* , ITC_CROSS_CARRIER_FORMATION_V1, ITC_MASK_DINO_FORMATION_V1 | `LOCALIZED_MASK_COMPETENCE_SUPPORTED`; bổ trợ, không hệ nào trội | mô tả | — |
| **Redundancy removal (RR)** | ITC_ENCODER_SELECTION_CONTROLS_V1, ITC_RR_BASELINE_RELEASE_V1 | **Dương**: recall tán nhỏ +2.32 pp, +12 tán nhỏ/seed; FP tăng | — | **thành phần đã biết, dùng được** |
| History / role / residual / finite-set / mask-set / count | ITC_HISTORY_CONDITIONED_*, MASK_HISTORY_SUBSPACE, ROLE_CONDITIONED_QUERY, RESIDUAL_SET_CORRECTION, FINITE_SET_LIKELIHOOD, MASK_SET_KERNEL, MASK_PROCESS_FUNCTIONAL, QUANTITATIVE_MASK_SET | Đều `TARGET_NOT_MET`; count `REFUTED_WITHIN_SCOPE` | 2 | CLOSED các recipe |
| Coded masks | ITC_CODED_* | +0.42 AP (cần +1) | 2 | CLOSED |
| Query-mask algebra | ITC_QUERY_MASK_ALGEBRA_V1 (+ REPLICATION) | **Mục tiêu thăm dò đạt**: AP75 +0.55; cần lặp lại (chưa đọc được kết quả lặp) | ? | **cần kiểm** |
| Spatial operator / nonlinear mask / membership field / transport | ITC_QUERY_SPATIAL_OPERATOR, CONDITIONAL_NONLINEAR_MASK, QUERY_MEMBERSHIP_FIELD, RECURSIVE_MEMBERSHIP, SPATIAL_TRANSPORT | Toán tử đúng, không đạt mục tiêu thực dụng | 2 | CLOSED |
| Multi-view / cross-view | ITC_MULTIVIEW_ENTITY_FORMATION, ITC_CROSS_VIEW_INSTANCE | Thất bại; chuyển giao âm | 2 | CLOSED |
| **EoMT: đọc trường và hình thành tán nhỏ** | READOUT_EXPRESSIVITY, DENSE_MATCH, SEEDED_CLOSED_FORM, POINT_KERNEL, PARENT_FIELD_TRANSFER, FIELD_NORMALIZATION, SOURCE_TRANSFER, ZF2_INTERNAL | Trường **chứa 94% tán 1–4 m²** (oracle tuyến tính). Query không bắt được. Matching không phải nút thắt. Hạt oracle dạng đóng 0.394, point-kernel 0.290. Lệch riêng ở ZF2 | 4 (chẩn đoán) | **Khâu hình thành: OPEN** |
| Coverage completion / discriminant bisection | ITC_COVERAGE_COMPLETION, ITC_DISCRIMINANT_BISECTION | 70% tán nhỏ bỏ sót là bị **hấp thụ**; tách không giám sát +2.9 pp | 2 | CLOSED |
| Conditional / seed-conditioned set formation | ITC_CONDITIONAL_SET_FORMATION, ITC_SEED_CONDITIONED_FORMATION | **Năng lực tạo mask mới được hỗ trợ trong phạm vi**; mục tiêu thực dụng không đạt | 2 | RESTRICTED |
| SAM2.1 prompted / adaptation / intrinsic prompt | H, ITC_SAM_ADAPTATION_CONTROL, ITC_INTRINSIC_PROMPT_LEARNING | Có lợi hình học chung; AP −3.0 khi thích nghi | 2 | CLOSED |

### D. Ảnh / nhãn (photometric, annotation)
| Hướng | Kết quả | Trạng thái |
|---|---|---|
| PHOTO_CLEAN (augmentation ghép cặp) | **+2.67 AP**, recall tán nhỏ tăng; thành phần đã biết | **dùng làm baseline mạnh** |
| PHOTO_Q / NORM / SUPPORT_SPATIAL | Đúng chiều, không đạt mục tiêu | CLOSED |
| Annotation-coherent matching (biến dạng nhãn chung box–mask) | 4 hệ train đủ, mục tiêu chính không đạt; **"ứng viên phương pháp gốc"**, novelty chưa xác lập | RESTRICTED |

### E. Biểu diễn thay thế (generative, coloring, set-valued)
| Hướng | Kết quả | Trạng thái |
|---|---|---|
| DiffusionInst fidelity / generative prior / FlowSDF / probability flow / SDF-set flow | Sửa lỗi cài đặt; không có chuyển giao hữu ích | CLOSED |
| Latent colour field / local-set endpoints | Năng lực AP +11 nhưng cổng cơ chế không đạt. Nhánh tự ghi: *"weak short learning was not an adequate family-level negative conclusion"* | **RESTRICTED** |
| Set-valued refinement (CrowdDet-like) | Chỉ có trên dữ liệu tổng hợp; train thật không đạt | RESTRICTED |
| **Local-set native pilot (TFN)** | **[đang chạy, luồng khác, 22/11]** AP local khoảng 1.7 lúc đang tổng hợp | — |

### F. Bằng chứng RGB và đánh giá con người
| Hướng | Kết quả | Trạng thái |
|---|---|---|
| RGB_EVIDENCE_RECOVERABILITY | `NO_DEMONSTRATED_INCREMENTAL_RGB_EVIDENCE`. Nhánh tự ghi: *không thể kết luận RGB không chứa thông tin nếu mô hình nhỏ này không khai thác được* | cấp 2, **không phải cấp 5** |
| RGB_ITC_FIVE_RQ / REVIEW_PILOT | `EVIDENCE_INSUFFICIENT_FOR_METHOD_SELECTION` | — |
| Human review analysis | 2 người đều cho rằng 10 cặp là hai tán khác nhau nhìn thấy được; **Stage 2 sẵn sàng, 0 kết quả** | **chưa làm** |
| Strong-carrier blinded review | `PENDING_HUMAN_REVIEW`, 0 bài nộp | **chưa làm** |
| M1 observability scale | Có hiện tượng đủ lớn; chưa có điểm vào phương pháp | — |

### G. Dữ liệu và hạ tầng
- D0/SelvaMask, dataset contract, decisive campaign, factorial B: phần lớn BLOCKED hoặc dừng vì lý do kỹ thuật; không phải
  kết quả khoa học.
- a06 harness: 3 lượt train đã chạy xong qua harness (EXP-028b: M0' và M3; EXP-031b). Một lượt khác bị OOM vì GPU dùng
  chung.

---

## 5. Các nguyên lý còn OPEN: bằng chứng ủng hộ và chống

| Nguyên lý | Ủng hộ | Chống / giới hạn | Cấp cần đạt để thành paper |
|---|---|---|---|
| **P1. Căn chỉnh quyết định cấp instance** (loss trên chính quyết định gộp/tách) | Lỗi A là lỗi ở bước đèo (EXP-029/031b); loss pixel không chạm tới (M3, method 1); hồ sơ 07/11 ghi là chưa từng thử | Decision-aligned partition 2×2 (hinge local-edge) không được ủng hộ; MALIS chưa có bằng chứng thật | Cấp 2 dương: khâu A giảm, nhiều nhất ở tán sát tán lớn, kể cả trên train |
| **P2. Biểu diễn cấp đối tượng** (OCAAD-style) | SAM với đúng tâm tách được khoảng 1/4, TFN được 0; trường EoMT chứa 94% tán 1–4 m² | SAM yếu với tán nhỏ (39% ở tán dễ); readout cục bộ chuyển giao kém | Cấp 2 dương, kèm nguồn tốt hơn SAM |
| **P3. Hình thành giả thuyết cho tán nhỏ bị hấp thụ** (birth / seed-conditioned) | Năng lực tạo mask mới có (CONDITIONAL_SET, QUERY_BIRTH có điều kiện) | Mục tiêu thực dụng chưa đạt; FP tăng | Cấp 2 dương có kiểm soát FP |
| **P4. Bất định phân hoạch** (PICE theory) | Định lý đúng, kiểm bằng LP và DP-GMM | Trên ITC thua PW-CC | Paper lý thuyết riêng (đã CONDITIONAL GO) |
| **P5. Giới hạn bằng chứng RGB** | 3 kiến trúc hỏng cùng tán; lỗi xảy ra trên cả train; T4 | Tương phản màu không khác; người xem trong pilot cho rằng 10 cặp nhìn thấy được | **Cấp 5: cần bài kiểm của người** (hạ tầng đã có hai lần, chưa chạy) |

---

## 6. Các lựa chọn để đánh vào

| # | Đánh vào | Giả thuyết | Cấp nhắm | Thí nghiệm quyết định | Chi phí | Loại paper nếu thành | Rủi ro chính |
|---|---|---|---|---|---|---|---|
| 1 | **Loss căn chỉnh quyết định đèo** (method 2) | Lỗi gộp tán nhỏ là lệch đơn vị giữa loss pixel và quyết định cặp | P1, cấp 2 → 4 | Train 1 seed. Khâu A giảm ≥ 25%, tập trung ở tán sát tán lớn, và giảm **cả trên train** | khoảng 3 giờ GPU + 1 ngày code | Phương pháp, với motivation mục 3 | Tán lớn bị chia nhiều hơn (lặp lại cái giá của method 1) |
| 2 | **Bài kiểm mù bằng mắt** | Ở các tán H+A, người có thấy tán riêng không | P5, quyết định cấp 5 | Khoảng 150 crop, so tỉ lệ "thấy" giữa tán lỗi và tán đúng | 20–30 phút của bạn | Quyết định giữa paper phương pháp và paper giới hạn | Một người đánh giá; chỉ là bằng chứng RGB, không phải danh tính sinh học |
| 3 | **Hình thành tán nhỏ trên carrier query** | Trường có thông tin (94%) nhưng query không bắt được | P3, cấp 2 | Theo NEXT_ACTION của các nhánh EoMT | cao | Phương pháp, nhưng trên carrier khác TFN | Đã thử nhiều recipe gần |
| 4 | **Paper lý thuyết PICE** | Thông tin từng cặp không đủ cho danh tính nhóm | P4 | Đã có định lý và kiểm chứng | thấp (viết) | Lý thuyết / phương pháp tổng quát | Không phải ITC |
| 5 | **Paper thực nghiệm / cơ chế** (thang thông tin + lỗi chung + TLS) | Lỗi tán nhỏ sát tán lớn là failure mode có cấu trúc, chung cho các kiến trúc | mô tả + oracle | Đã có đủ số; cần 3 seed cho các phép so | thấp–trung | Phân tích / benchmark | Ít "method novelty" |

**Thứ tự đề xuất của agent:** chạy **2** song song với **1**.
- 2 không tốn GPU và quyết định 1 có đáng theo tới cùng không.
- **5** là lưới an toàn: phần lớn số liệu đã có.
- 3 tốn nhất và đã thử nhiều recipe gần, nên để sau.

---

## 7. Những điều file này chưa chắc
- Các nhánh trong mục 4C–4G chỉ được đọc qua dòng quyết định. Cột cấp là ước lượng. Đặc biệt **QUERY_MASK_ALGEBRA
  replication** chưa đọc được kết luận.
- Nhánh `ITC_LOCAL_SET_NATIVE_PILOT_V1_20261122` đang chạy ở luồng khác.
- Các số của dòng carrier (TBS/BCI/ZF2) và dòng QuebecTrees **không so trực tiếp được**: khác dữ liệu và khác cách
  đánh giá.
- Chưa có kết quả 3 seed cho EXP-028b/031b; chưa đánh giá trên test cho các thí nghiệm từ 19/11.
