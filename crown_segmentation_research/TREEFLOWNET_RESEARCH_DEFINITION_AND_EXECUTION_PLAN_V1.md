# TreeFlowNet — Research Definition, Evaluation Protocol & Method-Discovery Roadmap

> **Bản kế hoạch nghiên cứu có thể triển khai · v1.0-draft · 14/10/2026**  
> **Chủ đề:** Reliable RGB Individual Tree Crown Individuation and Segmentation under Structural Ambiguity and Distribution Shift  
> **Trạng thái:** *Development-stage research plan*, **không phải preregistration đã khóa**, **chưa có method mới đã chứng minh**, **chưa có F–G–Q benchmark hoàn chỉnh**.  
> **Phạm vi:** RGB orthomosaic, *individual visible/annotatable crown instances*; dữ liệu CHM/LiDAR nếu có chỉ được dùng theo giao thức reference/ablation công bố riêng.  
> **Quy tắc quan trọng:** Phân biệt tuyệt đối (1) sự kiện đã quan sát, (2) suy luận/giả thuyết, (3) quyết định thiết kế cần khóa, và (4) kết quả xác nhận độc lập.

---

## MỤC LỤC

1. [Mục tiêu, câu hỏi và ranh giới claim](#1-mục-tiêu-câu-hỏi-và-ranh-giới-claim)
2. [Sổ đăng ký bằng chứng hiện có](#2-sổ-đăng-ký-bằng-chứng-hiện-có)
3. [Định nghĩa bài toán và ontology target](#3-định-nghĩa-bài-toán-và-ontology-target)
4. [Reference, annotation, uncertainty, audit](#4-reference-annotation-uncertainty-audit)
5. [Data registry, spatial split và kiểm soát leakage](#5-data-registry-spatial-split-và-kiểm-soát-leakage)
6. [Định nghĩa các họ mô hình F/G/Q và đối chứng](#6-định-nghĩa-các-họ-mô-hình-fgq-và-đối-chứng)
7. [Metric: existence, individuation, geometry, scoring, robustness](#7-metric-existence-individuation-geometry-scoring-robustness)
8. [Sampling, C0/C1, tình huống khó và tính độc lập](#8-sampling-c0c1-tình-huống-khó-và-tính-độc-lập)
9. [Thiết kế train/validation/calibration/test chính thức](#9-thiết-kế-trainvalidationcalibrationtest-chính-thức)
10. [Thiết lập kỹ thuật, tài nguyên và reproducibility](#10-thiết-lập-kỹ-thuật-tài-nguyên-và-reproducibility)
11. [Ma trận thí nghiệm và thứ tự thực thi](#11-ma-trận-thí-nghiệm-và-thứ-tự-thực-thi)
12. [Giả thuyết method mới và các phép bác bỏ](#12-giả-thuyết-method-mới-và-các-phép-bác-bỏ)
13. [Thống kê, bất định và effect/equivalence margins](#13-thống-kê-bất-định-và-effectequivalence-margins)
14. [Cross-domain, transfer, calibration và triển khai](#14-cross-domain-transfer-calibration-và-triển-khai)
15. [Literature map và kiểm định tính mới](#15-literature-map-và-kiểm-định-tính-mới)
16. [Go/No-Go gates, rủi ro và lịch thực hiện](#16-gono-go-gates-rủi-ro-và-lịch-thực-hiện)
17. [Evidence ladder hướng tới Nature Portfolio](#17-evidence-ladder-hướng-tới-nature-portfolio)
18. [Checklist, manifest và execution config mẫu](#18-checklist-manifest-và-execution-config-mẫu)
19. [Nguồn và ghi chú xuất xứ](#19-nguồn-và-ghi-chú-xuất-xứ)

---

## 1. Mục tiêu, câu hỏi và ranh giới claim

### 1.1. Đặt tên công trình (working title)

**Reliable Object Individuation for RGB Tree Crown Instance Segmentation under Structural Ambiguity and Domain Shift.**

Một tiêu đề hướng phương pháp tổng quát hơn, **chỉ dùng nếu đã kiểm ngoài tác vụ cây**, là *Learning Reliable Object Individuation from Ambiguous Visual Structures*. Không tự động dùng tên này khi chỉ có bằng chứng trên một bộ dữ liệu tán cây.

### 1.2. Động lực khoa học

Từ một ảnh RGB UAV, việc phân biệt tán cây **không quy về** tô đúng vùng foreground. Tán phức tạp có thể nhiều thùy; hai cây có thể chạm hoặc che khuất; cây nhỏ nằm giữa các cây lớn có thể thiếu proposal; các quyết định tách/gộp và lựa chọn confidence có thể xóa những cá thể hợp lệ trước khi đánh giá chất lượng biên.

Do đó cần phân biệt:

1. **Observability:** thông tin có trong ảnh RGB có đủ để quyết định tán/cây nào tồn tại không?
2. **Representation:** encoder/heads giữ được tín hiệu liên quan tới từng cá thể không?
3. **Formation:** hệ thống có sinh được giả thuyết instance phù hợp không?
4. **Grouping/individuation:** các phần thuộc cùng tán có được gán chung, các tán khác nhau có được tách không?
5. **Geometry/delineation:** mask/biên của một instance đã được nhận diện có đúng không?
6. **Selection/ranking:** một instance hợp lệ có được giữ và xếp điểm hợp lý không?
7. **Transfer/reliability:** các tính chất trên có còn đúng khi thay đổi điều kiện và nguồn ảnh không?

Không khẳng định các bước này độc lập nhân quả hoặc thực thi tuần tự trong não người hay trong mạng. Đây là một **phân rã chẩn đoán** để thiết kế can thiệp và metric.

### 1.3. Research questions (RQs) — thứ tự ưu tiên

- **RQ0 — Reference/observability:** định nghĩa một instance tán quan sát được và quy ước annotation đang đo điều gì? Những trường hợp nào người gán nhãn bất đồng do RGB không đủ thông tin?
- **RQ1 — Error attribution:** cá thể biến mất ở bước candidate generation, basin extraction, merge/repartition, mask reconstruction hay confidence/selection? Mỗi nguồn lỗi ảnh hưởng thế nào theo kích thước/mật độ/shape?
- **RQ2 — Readout comparison:** khi cùng feature RGB và điều kiện huấn luyện được kiểm soát, field head + decoder (F), learned basin grouping (G) và query mask head (Q) khác nhau thế nào về *existence* và *individuation*, chứ không chỉ AP?
- **RQ3 — Irreversibility:** quyết định tạo seed hoặc gộp basin quá sớm có giới hạn không gian nghiệm đầu ra không? Hệ thống có cần cơ chế *birth / split / reassignment / merge* hay chỉ điều chỉnh score/merge?
- **RQ4 — Robustness:** kết quả có tồn tại trên spatially independent, cross-acquisition, cross-site, cross-GSD hoặc morphology shift không?
- **RQ5 — Method discovery:** chỉ sau khi RQ1–RQ4 có bằng chứng, cơ chế mới **tối thiểu** nào có khả năng sửa loại lỗi đã được xác nhận, khác thực chất với TreeCoG, InstanceCut, Deep Watershed, Mask2Former, Cellpose/Omnipose và tiền lệ trong repo?

### 1.4. Hypotheses cạnh tranh — không định hướng chứng minh Q

| Mã | Giả thuyết bác bỏ được | Kết quả phản chứng/giới hạn |
|---|---|---|
| H-F | Field hiện tại có tín hiệu đủ hữu ích nhưng decoder/selection gây mất instance | G/Q không giúp ở matched conditions; field raw không chứa proposal đủ tốt |
| H-G | Learned relational grouping trên atom/basin cố định có thể cải thiện so F | Không cải thiện khi khóa F atoms, hoặc chỉ tăng AP nhờ score |
| H-Q | Query mask prediction từ cùng feature tăng grouping và recall cây nhỏ | Không cải thiện trên unseen data, hoặc tăng AP mà grouping/recall không đổi |
| H-R | Những quyết định *irreversible* hạn chế việc cứu các basin đã trộn nhiều GT | G-merge đạt mức tương đương G-repartition/Q cả ở basin-mixed stratum |
| H-D | Cơ chế thành công in-domain duy trì ưu thế ở ngoại miền | Hiệu ứng mất/đảo dấu ở external site hoặc ở scale/crowding chưa thấy |

**Không** được suy ra: Q tốt hơn = não người hoạt động bằng query; G tốt hơn = encoder đã học bản sắc sinh học; F thất bại = RGB không đủ thông tin; AP tăng = individuation tăng.

### 1.5. Endpoint chính dự kiến

Trong benchmark chính, đề xuất đăng ký trước hai **co-primary families** (điều chỉnh multiplicity ở §13):

1. **All-mask, score-independent, cardinality-first one-to-one recall @ mask IoU ≥ 0.50** cho tán `<1 m²`, với cùng reference và mask output stage có thể so sánh.
2. **Anchor-based unconditional balanced same-tree/different-tree grouping**, với pair assignment coverage báo bắt buộc, và cùng cố định sampling/distance strata.

**Secondary:** COCO AP50:95, AP50/AP75, recall theo kích thước, split/merge/duplicate, geometry (IoU/BF1), confidence–coverage curve, false positive per hectare, count MAE/bias, latency/VRAM.

Một pilot tốt nhưng chưa đủ để thiết kế method: tăng 1–2 AP mà không có cải thiện ổn định ở ít nhất một primary endpoint, hoặc tăng recall nhưng FP nổ.

---

## 2. Sổ đăng ký bằng chứng hiện có

### 2.1. Tất cả số liệu lịch sử sau đây là **development/diagnostic**, không phải independent confirmation

| Nguồn/quan sát | Con số | Có thể kết luận | **Không thể kết luận** |
|---|---:|---|---|
| QuebecTrees Z1; F cũ và F cuối; 139 val tiles, seeds 42/43/44 | AP cũ `33.65 ± 0.33`; cuối `39.03 ± 0.20` | Decoder/pipeline cuối có AP val cao hơn trong cấu hình lịch sử | Tổng quát ngoài Z1; chênh lệch thuần do saddle; seed là bản lặp cấu hình đồng nhất |
| Oracle A, sửa pixel dải 4 px quanh biên | `52.50 ± 0.44` AP với F cuối (`+13.47`) | AP nhạy với GT-assisted geometry/topology intervention | Refinement thực tế đạt +13.47; chỉ thay biên không tác động topology |
| Oracle B, GT mask replacement với association IoU ≥0.25 | `77.86 ± 0.55` AP | Mask replacement theo liên kết GT có thể tạo điểm cao | Trần RGB; model đã nhận đúng identity; AP gap là các loại lỗi độc lập cộng được |
| Tán `<1 m²`, F cuối | `25.72%` recall nội vùng lịch sử | Small-crown under-detection nghiêm trọng theo protocol đó | Hết thông tin trong feature; thống kê mọi miền |
| 4–9 m² | recall `81.59% → 80.36%` | Có đánh đổi theo size giữa pipeline cũ/cuối | Chỉ merge hay saddle gây nên |
| Score-only OOF so với heuristic, giữ nguyên mask | `+1.89` AP trung bình | Ranking ảnh hưởng AP | Scorer generalizes trên spatially independent data |
| Core 4 px structural balanced, F old → final | `66.79% → 72.83%` | Mask đã lưu của final có grouping diagnostic cao hơn | Latent identity của encoder tốt hơn; metric raw-intrinsic |
| Core 4 px E2E threshold 0.3, old/final | `66.46% / 61.19%` | Score/threshold có thể đảo thứ hạng | Final grouping tất yếu kém hơn ở operating point công bằng |
| Same final masks, heuristic vs OOF @0.3 | `70.60% / 61.19%` balanced, coverage `82.22% / 71.00%` | Assignment phụ thuộc score ngay cả khi geometry cố định | Heuristic vượt OOF ở mọi threshold/criterion |
| Q train-only sanity fit | Có mask fit trên các ví dụ train 0/1/few/many | Q có thể fit một số cấu trúc train | Q generalizes, Q hơn F/G, Q có method mới |
| Pixel-overlapping tile pairs giữa các fold OOF | 378 cặp chia sẻ pixel hợp lệ, 370 cặp có GT-positive | Fold theo tile hiện tại không spatially held-out | Định lượng được bias AP/OOF chỉ từ số cặp |

> **Quan trọng:** `±` trên báo cáo lịch sử là *SD qua ba run* (đôi khi checkpoint/min-area khác nhau), **không phải 95% CI**. 139 tile cùng Z1, có overlap; không bootstrap từng tile như IID. Các số lịch sử không so trực tiếp với Nakada `23.1` hay TreeCoG `57.01` vì khác dữ liệu/protocol.

### 2.2. Các biến/nút nguồn đã biết

- F dự đoán **flow, SDT, centroid, canopy** bằng các head song song từ feature `P2`; vì vậy phát biểu “mạng học shape trước rồi mới nhận cây” là sai về cấu trúc.
- F historical: EXP003 selected; gate canopy `>0.30`, SDT `>−0.15`, fixed-radius NMS. F final: EXP025 gate SDT `>−0.9`, tất cả candidate, saddle `≥0.3`, HGB scorer (cần xác nhận code/manifests trước khi re-run).
- Historical seeds **không đồng nhất**: seed 42 min-area xấp xỉ 1 m², seed 43/44 xấp xỉ 0.5 m²; checkpoint seed 42/43 epoch 25, seed 44 epoch 20; radius 50 px, 157 advection steps theo báo cáo 11/10.
- G0 **đã từng thử**: EXP020 HGB pairwise/geometric merge và EXP021 veto merge; chênh lệch AP lịch sử rất nhỏ trong protocol cũ. **Không** tái nhãn chúng thành method mới.
- 12,383 annotations nội vùng từ original COCO; 4 annotation mất unique-support ở priority raster; 12,379 còn support. Tập `<1 m²` là 219 annotation IDs (không chắc là 219 cây sinh học độc lập do tile overlap).
- `core r4`: **không loại** annotation nhỏ trong pilot (219/219 còn ít nhất hai pixel), nhưng median retained core fraction nhóm `<1 m²` chỉ `51.87%`; `r1 = 87.21%`, `adaptive = 66.88%`. Metric core r4 kiểm phần lõi dễ hơn, không thay thế đánh giá toàn mask.
- `C_inside`: candidate của F-final **trước selection** nằm trong *original COCO mask*; fixed per paired seed khi so F/G/Q. `C0` không chứng minh thiếu tín hiệu ảnh; seed 44 có một recall match dù `C0`.

### 2.3. Những câu hỏi **chưa xác minh, phải điền bằng audit**

- [ ] Tổng diện tích, số orthomosaic và số **spatially distinct** surveys/sites của QuebecTrees.
- [ ] Train/val/test historical được chia ở cấp nào; có overlap **giữa train và test** hay không; nguồn tập test có bị xem/tune trước đây không.
- [ ] GSD/CRS cho **từng** mosaic, không suy từ một tile mẫu; sự nhất quán diện tích m² với độ phân giải ảnh.
- [ ] Số polygon theo split/site/size/density; tán truncated và nhãn overlap.
- [ ] Điều khoản sử dụng và chia sẻ QuebecTrees, ForestSeg, BAMFORESTS, dữ liệu Nhật/Borneo nếu tiếp cận được.
- [ ] Thông tin đầy đủ về architecture Q, số query, no-object loss, output stage, số bước fit và toàn bộ config; chưa có `Q_SANITY_REPORT.md` bản gốc trong bộ file đang làm kế hoạch.
- [ ] Kết quả các EXP020/021 theo từng seed và artifact hash, không dựa duy nhất vào mean tổng hợp.
- [ ] Nguồn dữ liệu **new external holdout** có được không và bên nào sẽ thực hiện annotation blind.

Không tự điền các giá trị trên bằng suy đoán. Mỗi mục phải có `evidence_path` và `verification_date` trong registry (§18).

---

## 3. Định nghĩa bài toán và ontology target

### 3.1. Hai levels cần tách: crown được gán nhãn và cây sinh học

**Primary estimand:** độ chính xác khi tìm và delineate **các tán cây riêng biệt có thể quan sát/gán nhãn theo quy ước dataset** từ một ảnh RGB top-down.

**Không phải primary estimand:** tổng số cây sinh học trong ô rừng, kể cả cây dưới tán hoặc cây đã mất tán; allometric biomass, species ID, mortality status. Muốn suy ra các đại lượng này phải có reference, phương pháp và validation riêng.

**Target classes/flags:**

| Mã nhãn | Khái niệm | Cách sử dụng |
|---|---|---|
| `visible_crown` | Vùng tán nhìn thấy, có thể phân biệt dưới quy ước annotation | Primary crown instance |
| `amodal_crown` | Vùng tán suy đoán cả phần không nhìn thấy | Chỉ làm benchmark riêng có reference phù hợp |
| `biological_tree` | Cá thể sinh học xác nhận bằng khảo sát/3D | Secondary validation nếu có |
| `partial_or_truncated` | Tán cắt ở mép ảnh/crop/orthomosaic | Flag để giữ hoặc ignore theo quy tắc khóa |
| `overlap_ambiguous` | Các polygon cùng phủ pixel hoặc annotators bất đồng identity | Flag; không xóa thầm |
| `uncertain_identity` | Không chắc một hay nhiều tán từ RGB | Đánh giá riêng, có thể multi-annotator |
| `non_crown_background` | Đất, đá, đường, bóng, gỗ chết không nhận diện là tán theo ontology | Negative, cần FP measurement |

**Quyết định ontology bắt buộc trước test:** `visible-only` hay theo đúng original annotated polygons? Nếu nhãn gốc có nội dung amodal không xác định, báo là **dataset-defined crown polygons**, không gắn nhãn visible truth quá tự tin.

### 3.2. Công thức nhiệm vụ

- Ảnh `I ∈ [0,1]^(H×W×3)`, pixel footprint với GSD/affine transform đã biết.
- Reference set `Y = {G_1,...,G_N}` với mỗi `G_i ⊂ Ω` là mask instance, ID permutation-invariant.
- Prediction `Ŷ = {(M_j,s_j,p_j)}_(j=1..K)`; `s_j` có **ngữ nghĩa cụ thể** (class score/objectness/mask quality), `p_j` là metadata optional (center, query/seed ID).
- Một đơn vị quan sát là **annotation crown instance** hoặc **biological tree** tùy level. Không lẫn chúng trong một metric.
- Một pixel thuộc nhiều polygon GT có thể được giữ bằng COCO/RLE overlap; *priority owner map* chỉ dành cho các phép đo yêu cầu độc nhất owner.

### 3.3. Bốn năng lực cần đo tách biệt

1. **Existence/coverage:** ít nhất một candidate/prediction phù hợp với một GT.
2. **Individuation:** các vùng cùng GT có predicted owner giống nhau và khác GT thì predicted owner khác nhau; không cho lợi thế khi abstain.
3. **Delineation:** mask, shape và boundary của những prediction được matching.
4. **Selection/score:** quyết định giữ/xóa/xếp hạng các instance ở mức operating policy cố định.

**Không cộng hiệu ứng AP của oracle, score và grouping**: các can thiệp thay đổi distribution của matching và dependency giữa FP/FN.

### 3.4. Taxonomy “cấu trúc tán phức tạp” — định nghĩa được từ dữ liệu

- **Small size:** diện tích `A_i = pixel_area × foreground_pixel_count`, bins `<1`, `1–2`, `2–4`, `4–9`, `≥9 m²` để duy trì comparability lịch sử. Chọn bins trước khi xem new test.
- **Physical crown radius:** `r_eq=sqrt(A/π)` dùng để chuẩn hóa khoảng cách; không giả định mask tròn.
- **Density:** số instance centers trên một diện tích mặt đất cố định, hoặc số GT neighbors trong 3, 5, 10 m (**chỉ báo cáo theo GSD/scene phù hợp**).
- **Neighbour distance:** center-to-center, boundary-to-boundary, distance normalized by `r_eq`; tuyệt đối không gọi nearest-5 = touching pair.
- **Touching/overlapping:** dựa trên khoảng cách biên của GT trong m và/hoặc polygon overlap; nếu thiếu 3D reference, gọi `projected contact/overlap`.
- **Multi-lobed/irregular shape:** `solidity`, `P²/(4πA)`, eccentricity, skeleton branches; kiểm tác động discretization và pixel size.
- **Texture ambiguity:** độ tương tự hai vùng trên RGB/embedding, định nghĩa cách lấy mẫu dải trong/ngoài trước; tránh dùng GT test để điều chỉnh feature/threshold.
- **Visibility/occlusion:** annotation từ chuyên gia/CHM/field, không infer chỉ từ RGB score.
- **Edge/truncation:** mask chạm tile/mosaic boundary, phần trăm bị cắt nếu có footprint đầy đủ.

### 3.5. Những giới hạn không được vượt trong claim

- RGB top-down không bảo đảm tìm được cây bị che kín; dù một người có thể cảm nhận nhanh 2–4 tán rõ ràng, không suy ra phép phân chia đúng với mọi tán chồng lấn.
- Hình mask hợp lý không xác nhận biological identity.
- Số lượng đúng không đồng nghĩa one-to-one matching đúng vì FP có thể bù FN.
- “No candidate inside GT” khác “no useful basin support” khác “no information in RGB/encoder”.

---

## 4. Reference, annotation, uncertainty, audit

### 4.1. Ba cấp reference, không tự tạo cấp mà dữ liệu chưa hỗ trợ

**R0 — Dataset-native reference:** original COCO polygons/RLE, không sửa, hash và lưu nguồn. Dùng để tái lập lịch sử và benchmark primary nếu chưa có annotation audit mới.

**R1 — Blinded multi-annotator subset:** chọn có chủ đích + ngẫu nhiên theo size/density/ambiguity trên dữ liệu **chưa được các nhà nghiên cứu nhìn prediction**, ít nhất 2 annotators độc lập; reviewer thứ ba xử lý mâu thuẫn. Lưu cả independent labels trước adjudication, không chỉ consensus. Các sampling fractions được công bố để tránh xem tập enrich ca khó như prevalence thực tế.

**R2 — Independent biological reference:** field tree register/LiDAR/UAV oblique imagery nếu có, với sai số registration và khả năng chỉ nhận cây trên tán được báo rõ. R2 dùng để đánh giá ý nghĩa sinh học, không tự động thay thế R0 khi ontology khác.

**Lưu ý từ Nakada et al. (2026):** 18,507 polygon do annotators có kinh nghiệm vẽ; một người khác double-check; CHM hỗ trợ khi có. Đây là annotation quality workflow tốt, **chưa phải định lượng inter-annotator agreement hay chứng nhận biological truth**. Xem PDF được cung cấp, pp. 4–5, §2.4–2.5.

### 4.2. Manual annotation protocol — mẫu bắt buộc

Mỗi polygon lưu: `image_id, polygon_id, annotator_id, annotator_version, geometry, visibility, truncation, overlap_flag, confidence_label, evidence_modalities, annotation_time, review_status, reviewer_id, QC_notes`.

**Quy định:**

1. Chuyên gia nhận ảnh RGB ở độ phân giải gốc cùng GSD; không hiện dự đoán model trong lượt độc lập.
2. Annotator chọn số tán **có đủ bằng chứng phân biệt**; cho phép đánh dấu *ambiguous 1-vs-2* thay vì cưỡng bức.
3. Nếu có CHM/3D trợ giúp, ghi `reference_assisted=true`, loại dữ liệu trợ giúp, chất lượng căn chỉnh và giai đoạn được xem.
4. Trường hợp tán bị che khuất: không suy ra amodal contour mà không có quy tắc riêng.
5. Annotation trùng pixel: giữ nguyên nhiều polygon và báo tỉ lệ overlap; không tự cho polygon nào “chủ pixel” là sinh học đúng.
6. Adjudication chỉ sau khi đã lưu nhãn độc lập; không dùng prediction để giải quyết mọi disagreement.
7. Tất cả trường hợp thuộc benchmark final phải khóa annotation **trước** khi chấm final method.

### 4.3. QC định lượng reference

- **Geometry:** valid polygons, self-intersection, holes, area>0, CRS/affine, duplicate IDs, outside-image fraction.
- **Ontology:** single vs multiple masks overlapping same pixels, class definitions, hidden crowns, incomplete tile edge.
- **Reproducibility:** polygon→RLE→polygon/mask IoU và area checks; rasterized overlap stats; preserve original ordering.
- **Inter-annotator:** instance count disagreement per region, center matching recall, matched polygon IoU/BF1, split/merge disagreements, ambiguous rate; báo thêm consensus-vs-individual results.
- **Biological reference (nếu có):** detection/registration error trên tree register, crownless/occluded exclusions, crown-to-tree association uncertainty.

### 4.4. Negative/unknown handling

- Không gán `unknown` thành background âm trong loss/evaluation mà không nói rõ.
- Một pixel ambiguous có thể được mask khỏi training loss trong thử nghiệm **ablation riêng**, không tự động được xóa khỏi denominator đánh giá chính.
- Cross-dataset label harmonization phải có bảng đối chiếu ontology từng nguồn; nếu không thể harmonize, báo cross-dataset **stress test có annotation mismatch**, không claim pure sensor/domain robustness.

---
## 5. Data registry, spatial split và kiểm soát leakage

### 5.1. Quy tắc bất biến

**Split ở mức source scene/survey/site trước khi crop/augmentation**, không random split các tile đã overlap. Nếu thiếu nguyên ảnh, dùng extent polygon/CRS/affine tạo **overlap graph** và xác định connected components theo footprint; sau đó kiểm thêm autocorrelation/buffer. Một connected component có thể rất lớn khiến không thể có val/test độc lập từ source hiện tại — khi đó phải báo thiếu test, không tách cưỡng bức.

### 5.2. Data hierarchy và identity

```text
biome / eco-region
  └── site / forest
      └── campaign / date / season
          └── flight / acquisition
              └── sensor + GSD + processing recipe
                  └── orthomosaic (spatial footprint, CRS, affine)
                      └── spatial block / scene partition
                          └── tile / crop
                              └── polygon instance / annotation ID
```

Không xem 3.029 crop là 3.029 vùng rừng độc lập; không xem 219 annotation IDs trong overlapping tiles là 219 cá thể sinh học độc lập. Tất cả thống kê uncertainty cần xác định **đơn vị độc lập cao nhất** thực tế.

### 5.3. Data registry tối thiểu

`data/registry/scenes.csv` cần có:

```csv
scene_id,dataset,site_id,survey_id,flight_id,date_utc,sensor_id,gsd_x_m,gsd_y_m,crs_epsg,orthomosaic_sha256,annotation_sha256,footprint_uri,license,annotation_protocol,reference_assistance,split_role,quality_flags
```

`data/registry/tiles.csv`:

```csv
tile_id,scene_id,source_image_id,x0_px,y0_px,width_px,height_px,world_bounds_wkt,valid_area_m2,crop_policy,overlap_parent_ids,split_role,annotation_ids,content_hash
```

`data/registry/annotations.csv`:

```csv
annotation_id,scene_id,tile_id,source_tree_id_if_known,area_m2,centroid_x_m,centroid_y_m,truncated,overlap_pixels,unique_support_pixels,ambiguity_flag,reference_level,annotator_batch
```

Nếu không có `source_tree_id`, phải phân tích trùng annotation ở các tile overlap bằng provenance và geometry; không tự động deduplicate theo IoU thấp mà không kiểm tra.

### 5.4. Overlap graph, cross-fold leakage audit

Tạo graph `G_overlap=(V,E)` với đỉnh là tile (hoặc primitive source fragment). Một cạnh tồn tại nếu hai tile **chia sẻ bất kỳ valid source pixel nào** sau khi quy về cùng mosaic-grid chính xác; có thể thêm metadata `GT-positive overlap`.

Quy trình bắt buộc:

1. Reconstruct geospatial footprint theo affine/CRS, ưu tiên exact pixel-grid alignment khi cùng source mosaic.
2. Kiểm số cặp pixel-overlap và GT-positive overlap; audit rounding/max registration error.
3. Tính connected components của tile overlap graph; không cho component cắt qua train/val/calibration/test.
4. Đánh giá spatial autocorrelation ở khoảng cách lớn hơn tile overlap; thiết kế block/buffer có căn cứ. Báo sensitivity với vài block scales **đã khóa trước confirmatory inference**.
5. Kiểm cả *near duplicates*, cùng UAV flight, scene texture và cùng biological tree qua tile (nếu xác định được).
6. Tạo `split_manifest.json` có source hashes, algorithm version, random seed và cross-split zero-overlap assertions.
7. Kiểm sau augmentations/cache feature cũng không còn cross-split leakage qua source IDs.

**Historical warning:** audit đã thấy 378 cặp khác fold chia sẻ valid pixels, 370 chia sẻ GT-positive; OOF scorer cũ **không** spatially held-out. Kết quả OOF vẫn là mô tả hệ thống đó, nhưng không thể dùng để xác nhận generalization.

### 5.5. Chia tập: dữ liệu cũ và dữ liệu mới phải có vai trò khác nhau

| Role | Được phép làm | Không được làm |
|---|---|---|
| `legacy_dev` | Phân tích A/B/C, chọn ý tưởng, debug metric, hiểu failure cases | Gọi là độc lập sau khi đã xem hàng loạt kết quả |
| `train` | Fit encoder/heads, augment, learned G edges | Sử dụng final test labels hoặc final test distribution để tối ưu |
| `dev_val` | Early stopping, HP tuning, choose candidate family, set threshold | Báo làm unbiased estimate of external transfer |
| `calibration` | Fit score calibration, threshold cho policy sau khi lựa chọn model | Cross-fit trên pixel-overlapping train/tune folds rồi gọi independent OOF |
| `lockbox_test` | Chấm *một lần* sau freeze config, confirm primary endpoints | Tuning, sửa metric, chọn visualization, chọn seed từ test |
| `external_test_site` | Đánh giá cross-site/site-morphology generalization trên annotation harmonized | Chọn config dựa vào external test rồi vẫn gọi zero-shot |

Nếu chỉ có Z1 và các vùng đã sử dụng, **không tự sản xuất `lockbox_test` bằng cách chia lại Z1**. Tối ưu nhất là thu thập hoặc truy cập scene/site mới và thực hiện blind annotation/harmonization.

### 5.6. External datasets: điều kiện admissibility

Nguồn xem xét: QuebecTrees (hiện có), ForestSeg và BAMFORESTS (nếu được cấp quyền và target hợp lệ), dữ liệu annotated broadleaf đa-site khác; Nakada dataset **không được giả định public** chỉ vì paper có 18.507 polygon.

Đánh giá trước khi tải/train:

| Câu hỏi | Điều kiện |
|---|---|
| Dữ liệu có legally usable license? | Lưu license URL/bản sao; phân biệt nghiên cứu, công bố mask, commercial use |
| Có georeferenced source / stable scene IDs? | Có cách chặn tile/site overlap |
| Nhãn là visible crown, biological tree hay amodal? | Harmonize hoặc giữ benchmark riêng |
| GSD và image footprint có đáng tin? | Bắt buộc cho m², physical context và size bins |
| Có polygon instance và empty images? | Cần cho recall và FP/negative calibration |
| Có test label được dùng trong phát triển của paper khác? | Ghi mức độc lập theo nghiên cứu của bạn |

**Không** biến điểm TreeCoG `57.01` AP ForestSeg và Nakada `23.1` AP test thành một bảng xếp hạng trực tiếp.

---

## 6. Định nghĩa các họ mô hình F/G/Q và đối chứng

### 6.1. Phân biệt feature/learning/inference/selection

Đặt `F = E_θ(I)` là feature map; cùng encoder **checkpoint**, không chỉ cùng tên backbone, mới là frozen-feature control. Mọi nhánh xuất `pre-selection hypotheses` và `post-selection predictions` dưới cùng common schema; lưu đầy đủ lineage.

| Branch | Learnable part | Allowed output space | Essential comparison |
|---|---|---|---|
| `F_legacy_old` | Historical field heads | Decoder fixed radius, historical selection | Diagnostic only |
| `F_legacy_final` | Historical field heads | Saddle/group selection lịch sử | Diagnostic only |
| `F_retrain` | Same frozen E; field heads fitted anew on train | Fields→basins→masks, locked decoder | Fair readout probe |
| `G0` | EXP020/021 historical reimplemented/adapted | Basin graph/merge restricted | Negative/previous learned-merge baseline |
| `G1_merge` | Feature-pooled basin relation predictor | **Union of intact existing atoms only** | Can better grouping suffice? |
| `G2_repartition` (optional) | Pixel/sub-basin reassignment | Split/reassign atom; new foreground only if separately allowed | Tests irreversible atom bottleneck |
| `Q_light` | Query masks from same frozen feature scale (e.g. P2) | Up to K query hypotheses, no F seed required | New instance formation readout |
| `Q_multiscale` | Query masks using P2–P5 | Extra feature info | Separate ablation, not pure head comparison |
| `Mask2Former_external` | Standard known method, trained from clean recipe | Query-based set prediction | Strong external baseline |
| `TreeCoG_external` | EDTER contours + GCN merge | Contour atoms and graph grouping | Direct related work for G, if executable |

### 6.2. Field branch F — reproducibility before retraining

`F_legacy` phải replay được *bitwise* hoặc trong ngưỡng đã định nghĩa trên cached outputs: field hashing, candidate count, basin count, mask count, same AP and size recall, score-only policy.

**F new probe:**

1. Chọn một encoder checkpoint from **train-only source** (nếu existing checkpoint từng tune bằng dev, ghi provenance/tuning exposure); freeze.
2. Dùng cùng `P2` và target flow/SDT/centroid/canopy như baseline, giữ loss/weights từ implementation đã audit, không chọn lại qua test.
3. Xuất cả `raw fields`, `candidate list`, `basin atoms`, `pre-merge masks`, `post-merge`, `pre-score`, `post-score`.
4. Khóa min-area policy và physical unit conversion. Có thể chấm cả policy không min-area như secondary, nhưng **không tune vì cây nhỏ** từ test.
5. Giữ tách selection stage để không đánh giá structural trên mask đã bị xóa bởi score.

### 6.3. G0 và G1: learned relation (không gọi đơn giản là “thêm graph”)

**G0:** reproduce EXP020 pair merge và EXP021 veto merge trong cùng artifact/metric nếu khả thi; lịch sử khoảng `36.48` so với reference `36.50` AP và veto tốt nhất `36.58` được nêu trong amendment, nhưng phải check raw experiment JSON trước reuse. Đây là các tiền lệ *nội bộ* bắt buộc vượt, không xóa khỏi related work vì kết quả không tốt.

**G1_merge design — một variant nhỏ và có thể kiểm soát:**

- Nodes: basins `B_i` từ F trước merge, có nonoverlap foreground đã xác nhận.
- Edges: adjacency theo boundary contact hoặc physical proximity đã khóa; ghi edge distribution, đảm bảo train/inference cùng proposal rule.
- Node features: mean/std/max pooled frozen P2 vector, log(area), centroid, compactness, peak/SDT summary, uncertainty.
- Pair features (symmetric): `|u_i-u_j|`, `u_i+u_j`, `u_i⊙u_j`, relative size, saddle/prominence, distance, contrast in boundary band; lựa chọn từng nhóm qua ablation.
- Targets: same-tree/different-tree lấy từ **train-only GT** trên vùng có unique GT support. Basin ownership xây từ histogram overlap, không đơn giản owner của seed candidate. Nếu purity thấp, có thể dùng soft target hoặc ignore trong *training* theo cutoff khóa; báo purity distribution/selection bias.
- Predictor: MLP nhỏ/graph network; báo hidden width, layers, parameters, class sampling, loss (BCE/focal), regularization.
- Global grouping: **khóa chính xác merge engine**, threshold hoặc objective. Một edge AUC cao không bảo đảm partition đúng; báo connected-component failure, transitivity violations, false bridge.
- Constraints: không split trong một basin, không thêm foreground nếu không có branch riêng. G1 không được nhận random GT counts at inference.

**Negative-control G:** giữ predictor features không RGB (geometry-only), shuffled RGB features trong đúng train protocol, và prediction with same merge engine. Nếu RGB feature không giúp, đừng claim relational visual reasoning.

**G2_repartition optional:** sử dụng F basins làm khởi tạo nhưng cho phép split/reassign pixel hoặc tạo hypotheses mới; phải công bố search-space expansion và thêm module riêng. Đây có thể là đường thiết kế method sau F–G–Q, nhưng **không** xem G2 là cùng một kiểm tra thuần merge với G1.

**Upper bound chẩn đoán atom:** với GT owner không overlap, một relaxation theo pixel overlap có thể viết `sum_i max_k |B_i∩G_k|` cho foreground pixel được gán đúng khi mỗi atom giữ nguyên; **không phải cận COCO AP** hoặc “oracle RGB”. Tính cùng background/ignore policy và công bố domain rõ.

### 6.4. Query branch Q — frozen readout, không giả định tốt hơn

- Same frozen encoder and **same feature scale P2** cho `Q_light` vs `F_retrain`; any P2–P5 multi-scale advantage = ablation riêng.
- Fixed `K_max` dựa trên **train distribution/memory**, không từ GT count trong validation/test; báo tỷ lệ ảnh có GT count > K_max và upper-bound truncation.
- Query outputs `{mask_logits, class/no-object logits, optional quality score}`; Hungarian matching train-only, report matching cost, unmatched slot handling.
- **Unmatched query trap:** set loss chuẩn có no-object classification supervision nhưng có thể không có mask loss cho unmatched. Raw low-score mask logits từ unmatched slots không có cùng semantics với F basins. So stage tương thích và đánh giá raw slots chỉ là diagnostic.
- Q sanity: train-only 0, 1, 2–4, >4 GT crops, nhưng **không cung cấp GT count tại inference**. Overfit sanity nhằm kiểm data/gradient/matching, không dùng làm generalization evidence.
- Train-target: original instance polygons/RLE. Cẩn thận overlap, truncation, visibility, no-object weight; khóa policy trước confirmatory runs.
- Q output: `slot logits → candidate masks → no-object/score selection → dedup/assignment`; xuất đủ stage để tách score from membership.

### 6.5. So sánh công bằng là **hai thiết kế thí nghiệm khác nhau**

**A. Mechanistic readout probe (frozen):** cùng `E` checkpoint và P2; match train split, crop, augmentation seed, inference image/annotation. F, G, Q thay readout; G phải trả giá cả việc tạo fields/basins. Kết quả trả lời “ai khai thác feature hiện có tốt hơn”, **không** kết luận kiến trúc tốt nhất nếu end-to-end được train riêng.

**B. End-to-end architecture comparison:** cùng training data/split, backbone family/pretraining, model selection budget, inference FOV. G/Q có thể loss khác; báo compute FLOPs/GPU-hours/parameters và convergence. Run stronger external Mask2Former baseline. Kết quả trả lời “phương pháp nào đạt hiệu năng tốt hơn trong cùng protocol”, chưa cô lập hoàn toàn mechanism.

**C. Post-selection control:** giữ nguyên masks/IDs, thay score heuristic vs OOF vs quality score; một policy mới phải được fit trên `train/calibration` spatially separated. Khi E2E grouping đổi trên cùng masks, tuyệt đối không gán phần lợi cho formation.

---

## 7. Metric: existence, individuation, geometry, scoring, robustness

### 7.1. Common evaluation contract

Với mỗi prediction, lưu `mask_rle`, `score`, `source_stage`, `query_id/seed_id`, `component_id`, `provenance`, `tile_id`, `mosaic_coords`, `transform`, `model_seed`, `config_hash`.

Một evaluation snapshot phải ghi rõ:

- Reference: original COCO instances hay canonical nonoverlap owner raster?
- Output stage: raw candidates / atom / before confidence / after score / after stitching?
- IoU threshold, maxDets, threshold policy, masking ignored regions, polygons truncated?
- Unit: tile, orthomosaic, physical crown, site?
- Macro weights per size/distance/site, sample count và empty-bin handling?

### 7.2. One-to-one instance detection (score-independent)

Với ground truths `G_i` và predictions `M_j`, xây bipartite edges `IoU(G_i,M_j) ≥ τ` với `τ` cố định. Tìm assignment **cardinality-first** (maximize số matched pairs rồi tối đa tổng IoU giữa các matching cùng cardinality). Không dùng score để chọn cặp trong endpoint này.

```
TP = number of matched GT–pred pairs
FN = N_GT - TP
FP = N_pred - TP
R_instance = TP / N_GT
P_instance = TP / N_pred  (nếu denominator > 0)
```

**Quan trọng:** `max_j IoU(M_j,G_i) ≥τ` trên từng GT chỉ là *proposal coverage* và có thể đếm một prediction cho nhiều GT. Không gọi đó là one-to-one recall.

Theo size strata dựa **GT physical area**, đếm TP một-một và denominator GT. Đối với zero-GT tile, recall không định nghĩa; phải báo FP/tile hoặc FP/hectare.

**All-mask** không có nghĩa “mọi raw unmatched Q mask logits đều thành prediction hợp lệ”; định nghĩa set prediction trước score phải có comparability contract (§6.4). Chấm cả upper-bound raw-stage diagnostics nhưng không gộp chúng vào primary endpoint nếu semantics khác nhau.

### 7.3. Anchor-based individuation (score-independent structural)

Reference chỉ lấy điểm thuộc **unique GT support**, các vùng GT overlap được tách riêng. Với mỗi GT `G_i`, tạo tối đa `8` anchors từ `r1`, `r4` hoặc `adaptive` core, điểm sampling và RNG khóa trong cache; không thay sampling theo model.

- `S`: tập cặp anchor same-tree.
- `D`: tập cặp anchor from different trees, gồm hard nearby pairs và các bins: `0–0.25`, `0.25–0.5`, `0.5–1`, `1–2`, `2–4`, `≥4 m` (chỉ chấm bin có đủ hai lớp).
- Cho `z(p)=0` nếu pixel không được gán hoặc bị ≥2 mask overlap trong *structural unique-membership policy*; ID >0 cho một mask hợp lệ.

\[
R_{same}=\frac{\sum_{(p,q)\in S}\mathbf1[z(p)=z(q)>0]}{|S|}
\]

\[
R_{diff}=\frac{\sum_{(p,q)\in D}\mathbf1[z(p)\ne z(q),\ z(p)>0,z(q)>0]}{|D|}
\]

\[
G_{balanced}=\tfrac12(R_{same}+R_{diff})
\]

Trong `unconditional R`, cặp bị abstain tính là sai (không thưởng im lặng); báo thêm:

- `pair coverage`: phần trăm pairs với cả hai endpoint assigned unambiguously.
- `conditional same/diff accuracy | covered` và denominators, `undefined` nếu zero covered.
- `duplicate/overlap rate`: nhiều raw/preselection masks cùng chứa anchor; phân tích riêng semantic khác nhau F/Q.
- Macro qua class/distance bins **đã khóa** và support; không xem hàng triệu pairs correlated là hàng triệu observations độc lập.

**Không dùng GT-based optimal assignment để chọn slot/prediction khi chấm grouping primary.** GT matching chỉ phục vụ diagnostics/oracle, nếu dùng phải gắn nhãn rõ.

### 7.4. E2E grouping and score control

Ở E2E, apply `score ≥ threshold`, `maxDets=400` (để tương thích historical COCO) và resolve overlap bằng highest-score rule đã khóa. Báo series threshold `0.1/0.3/0.5/0.7` theo lịch sử **và** accuracy–coverage curves; không so một threshold chung như calibration fair nếu score scales khác nhau.

Thiết kế matched-coverage comparison: nội suy/đo điểm trên vùng coverage chung có support, khóa trước cách interpolation và summary statistic; có thể báo area-under-risk–coverage curve.

**Score-only test:** assert mọi `mask_bytes, prediction_ids, ordering, geometry` giống hệt trước/sau rescore; chỉ score vector được thay. Structural metric phải giữ nguyên đến floating-point tolerance; nếu không, lỗi code/selection stage.

**AP vs grouping:** COCO AP50:95 là tích hợp PR theo score thresholds và IoU; E2E grouping theo một ngưỡng score là phép đo khác. AP tăng trong khi grouping@0.3 giảm **không mâu thuẫn**.

### 7.5. Geometry (không được nhầm với individuation)

- Matched mask IoU / Dice theo one-to-one matches đã định.
- BF1 tại khoảng cách `1,2,4,8 px` hoặc vật lý `0.03,0.06,0.12,0.24m` **chỉ khi GSD đó là đúng cho scene**; báo pixel units song song.
- Instance area error, centroid displacement (m), perimeter / shape error.
- Matched-pair metrics có **selection bias** vì unmatched GT/pred không có IoU được định nghĩa; luôn đi kèm recall and FP/FN.
- Boundary oracle A: thay pixel trong W_r có thể thay topology/lỗ/components; không gọi là “pure boundary fix”.
- Original mask overlap và ambiguous GT assignment phải được đo, không forced owner âm thầm.

### 7.6. Explicit split/merge/duplicate taxonomy

Dựa trên overlap graph giữa GT và predicted masks (với ngưỡng area-relative+absolute đã khóa):

- **Miss:** GT không có matched prediction theo primary matching.
- **Split candidate:** ≥2 predictions cùng phủ đáng kể một GT.
- **Merge candidate:** 1 prediction phủ đáng kể ≥2 GT.
- **Duplicate:** nhiều predictions gần trùng nhau về geometry và cùng liên kết GT.
- **Hallucinated foreground FP:** prediction không có GT match trên background/negative areas.
- **Abstention/overlap:** anchor unassigned hoặc ambiguous.

Các định nghĩa trên dựa trên **annotation geometry**, không tự động là bằng chứng nhiều cây sinh học bị gộp. Phải phân biệt `error incidence` với one-to-one matching transitions và candidate coverage.

### 7.7. Robustness & deployment metrics

- `AP, R_small, G_balanced, G_coverage` theo site/season/camera/GSD/density/shape.
- Worst-group recall (chỉ khi group đủ support), group mean, site-specific confidence intervals.
- Performance drops `Δ_metric = metric_target − metric_source` nhưng chú thích composition shift; thêm matched strata.
- FP per hectare, count bias (signed), MAE per non-overlapping survey unit, missed-tree rate.
- Score reliability: calibration cho **biến cố được xác định** (`one-to-one TP@IoU≥τ` dưới specified assignment policy) và risk–coverage, không coi raw score là xác suất biological identity.
- Runtime: mean/median/p95 latency per **physical area** và tile, peak GPU RAM, CPU stitching, energy optional; benchmark trên cùng phần cứng/precision.

### 7.8. Toy/unit tests — buộc chạy trước metric pilot và trước benchmark

| Case | Kỳ vọng |
|---|---|
| Empty GT + empty pred | FP=0; recall undefined, zero positive units |
| Empty GT + spurious pred | FP>0; không bị “good grouping” che |
| 1 perfect mask | Same pairs correct; different pairs absent/not fabricated; coverage 1 |
| 2 GT merged into 1 mask | Different-tree pairs sai, one-to-one recall ≤1/2 |
| 1 GT split into 2 masks | Same-tree pairs có thể sai; duplicate/split diagnostic |
| 2 GT + extra duplicate prediction | FP tăng theo one-to-one matching; raw overlap diagnostic |
| Score permutation, exact same masks | Structural invariant; E2E/AP có thể đổi |
| Permuted prediction IDs | Metrics invariant nếu semantic scores/masks preserved |
| Overlap masks both cover anchor | Structural abstain; E2E resolve by score |
| Truncated/overlapping GT | Ignore/eligibility theo policy cố định |
| Sparse tiny polygon, erosion r4 | Eligibility & fraction được báo; không crash/auto-drop |
| RLE/native raster | pycocotools equivalence trong tolerance |

**Existing pilot** đã có RLE membership test so với pycocotools trên synthetic/real masks. Chạy lại khi thay implementation/schema, không chỉ tin vào báo cáo trước.

---
## 8. Sampling, C0/C1, tình huống khó và tính độc lập

### 8.1. Primary analysis population

- Annotation instances trong `lockbox_test` theo ontology đã khóa, có flag truncated/ambiguous.
- Size `<1m²` xác định bằng *original GT polygon* và scene pixel-area; không dùng prediction area.
- Dữ liệu empty crops được giữ cho FP evaluation; không chỉ chọn ảnh có GT rồi đánh giá.
- Không chỉ train/eval crop có 2–4 cây: đưa 0,1,2–4,>4 instance crops, và mosaic-level dense forest. Không suy ra GT count tại test.

### 8.2. Các strata khóa từ **F baseline**

- `C_inside = 1` nếu tồn tại F-final preselection candidate **nằm trong** original GT polygon. `C_inside = 0` nếu không; tạo **một lần** trong paired run rồi dùng chung cho F/G/Q. C là diagnostic stratum, không phải nguyên nhân và không phải nhãn cây thật.
- `B_support(t)` = phần diện tích GT được phủ bởi union of F basins, đánh giá curve theo `t` hoặc bins đã khóa. C0 vẫn có thể có basin hữu ích, C1 vẫn có thể có basin sai.
- `B_mixed` = atom/basin có overlap đáng kể với nhiều GT theo train/evaluation reference. Chỉ là geometry-defined ambiguity, không nhầm biological merge.
- `crowded/difficult`: nearest-neighbour physical distance, pair-distance bins, morphology, truncation/annotation uncertainty, tile-edge incidence.

**Chú ý:** C0/C1 từ F của một seed có thể khác seed khác. Ghép paired comparisons **trong seed**; không gộp một cách hồn nhiên ba seed như ba sampling độc lập của GT.

### 8.3. Anchor sampling and per-bin caps

- 8 anchors/GT tối đa, theo RNG fix và GT-unique core; sample from components in a way đã khóa; không chọn anchor dựa vào prediction.
- Same-tree pairs gồm within-lobe, cross-lobe và khoảng cách xa trong cùng crown; nếu không có manual lobes, chọn cặp đa khoảng cách định nghĩa theo mask skeleton/geodesic nhưng tránh thiết kế từ result.
- Different-tree pairs gồm nearest-5 representative centers từ core anchor mean như pilot (không gọi centroid toàn mask) và bổ sung hard touch/contact strata từ annotated polygons.
- Tối đa 8 cặp mỗi focal annotation/class/distance-bin như pilot; macro six bins có support cả hai classes. Báo `N_GT,N_pairs,N_tiles,N_unique_source_areas`.
- Nếu tạo protocol mới để cải thiện validity, **version** nó khác metric pilot cũ; không âm thầm đổi cách lấy cặp rồi so percentages.

### 8.4. Sampling bias và các cận diễn giải

- Core r4 không loại tán nhỏ trong Z1 pilot, nhưng chỉ giữ median 51.87% pixels của tán `<1m²`, nên các điểm chấm có thể tập trung vùng giữa dễ hơn.
- Different pairs gần nhau chưa chắc là biologically contacting trees; đặc biệt cross-crown proximity trên 2D có thể đánh lừa.
- Same/different anchors chỉ đo nhất quán assignment đối với **nhãn GT**, không đo cơ chế thần kinh hay “objectness” nội tại.
- Pilot grouping có thể có dependency do cùng cây, cùng tile, cùng mosaic, cùng annotator, cùng seed. Mọi inferential interval phải xác định clustering hierarchy.

---

## 9. Thiết kế train/validation/calibration/test chính thức

### 9.1. Sơ đồ protocol phiên bản mới

```text
ALL AVAILABLE GEOSPATIAL SCENES
  -> target/reference audit + license check
  -> scene/site/acquisition registry (freeze source IDs)
  -> overlap graph + spatial blocking + split_manifest hash
     |-- TRAIN          : supervised fitting + augment only
     |-- DEV_VAL        : checkpoint / HP / family selection
     |-- CALIBRATION    : score/threshold, AFTER model selection
     |-- LOCKBOX_TEST   : precommitted one-time evaluation
     `-- EXTERNAL_TEST  : independent sites/cameras/forests, harmonized target

LEGACY Z1 VAL & OLD TEST (PREVIOUSLY ANALYZED)
  -> diagnostics / hypothesis discovery only, unless qualified as "reused"
```

**Khả thi tối thiểu khi không có new independent site:** chạy `train + development CV with spatial grouping`, báo study là *exploratory* và **không** ghi “verified cross-domain generalization”. Trước khi nộp paper cần kế hoạch external validation thực tế.

### 9.2. Số lượng và kích thước split

Không ấn định 70/15/15 cho mọi dataset: nếu có 7 site như Nakada, có thể tách theo site; nếu chỉ có 1 site, random fractions không tạo site-independent test. Quy tắc:

1. Ưu tiên **leave-site(s)-out** hoặc train/val/cal/test ở các campaign/scene độc lập nếu số site đủ.
2. Nếu chỉ có vài site, **study design power** phải kiểm theo số site; báo range không chắc chắn rộng thay vì xuất CI giả chính xác.
3. Với dữ liệu nhiều site nhưng mất cân bằng, có thể stratify split theo climate/biome, crown size histograms mà **không chọn bằng kết quả mô hình**.
4. Nếu chỉ Z1 available cho pilot, dùng grouped folds để phát triển và reserve nguồn khác cho confirmatory; **không dùng những Z1 tiles đã xem làm final lockbox**.
5. Mỗi split phải được ký bằng `split_manifest_sha256` và archive trước train.

### 9.3. Cross-validation, threshold và score calibration

- Development CV sử dụng spatial/source groups, không random tile-fold; không cho source-overlap component cắt fold.
- Out-of-fold (OOF) score từ train/development chỉ được gọi **spatial OOF** sau pixel overlap and autocorrelation safeguards.
- Calibration fit trên independent calibration scenes với fixed model; class score/mask quality score calibration có thể học logistic/isotonic/temperature tùy variable, nhưng phải chọn loại và model trước final test.
- Threshold cho operational result chọn theo calibration, ví dụ maximize F1 subject to minimum precision hoặc hit budget FP/hectare. Không chọn từ lockbox.
- Structural readout không dùng threshold nhưng cần output-stage definition. E2E confidence sweeps secondary phải diễn giải đúng score scales.

### 9.4. Crop training và inference không được gây confound

**Physical FOV:** Nakada dùng `1024²` crop, GSD 2.5 cm (`25.6m × 25.6m`); TreeFlowNet từng dùng crop nhỏ hơn. So sánh phải có kiểm soát physical context: cùng meter-per-crop hoặc ablation riêng crop size. Không cho Q multi-scale/bigger context rồi gán chiến thắng thuần cho query mechanism.

**Train crop policy:** sampling theo spatial groups, giữ empty/low-density/high-density. Có thể balanced sample cây nhỏ **chỉ trên TRAIN**. Nếu oversample narrow strata, báo distribution và evaluation unbiased by sampling weights.

**Truncated crown:** một crown cắt bởi crop không có nghĩa thành instance mới. Có thể ignore partials trên train hoặc dùng full-mosaic GT clips theo recipe cụ thể; xác nhận crop shift tests trên eval.

**Inference tiling:** khóa tile overlap, border touching rejection, cross-tile dedup và mosaic reconstruction. Chạy sensitivity crop origin/overlap như ablation riêng, đánh giá trên cùng ground-coordinate mosaic frame.

### 9.5. Chính sách test release

Trước lần chấm đầu tiên phải có `LOCKED_EVAL_PROTOCOL.md` chứa:

- Hash commit/config/data/split/weights.
- Primary endpoints (2 families), matching, macro weighting, thresholds, confidence control.
- Small crown bins và analysis strata, allowed exclusions, negative crop handling.
- Number of final methods, training seeds, comparison margins and inference time budget.
- Statistical plan and multiple testing correction.
- External test and annotation harmonization.
- Pre-defined **failure conditions** (model must not collapse 4–9m² recall, FP budgets).

**Test xem một lần không đồng nghĩa bảo đảm độc lập**, nếu cùng location/scene từng dùng để phát triển hoặc chọn phương pháp. Ghi lịch sử model-selection exposure.

---

## 10. Thiết lập kỹ thuật, tài nguyên và reproducibility

### 10.1. Kiến trúc repo — gợi ý, không ghi đè dự án đang chạy

```text
crown_segmentation_research/
├── existing_methods/             # symlink/read-only references to historical code
├── research_protocol/
│   ├── README.md
│   ├── RQ_HYPOTHESES.md
│   ├── TARGET_ONTOLOGY.md
│   ├── ANNOTATION_QA.md
│   ├── LOCKED_EVAL_PROTOCOL.md  # chỉ ký khi đủ điều kiện
│   ├── execution_config.yaml
│   └── literature_matrix.csv
├── data_registry/
│   ├── scenes.csv
│   ├── tiles.csv
│   ├── annotations.csv
│   ├── spatial_overlap_graph.parquet
│   ├── split_manifest.json
│   └── source_hashes.json
├── src/
│   ├── common/                  # prediction schema + geometry + RLE
│   ├── data/                    # crop / augmentation / grouped sampler
│   ├── F/                       # frozen + historical replay
│   ├── G0/                      # EXP020/021 replay
│   ├── G1_merge/                # learned atom association
│   ├── Q_light/                 # query mask baseline
│   ├── evaluation/              # common, score-blind and E2E
│   └── diagnostics/             # candidate lineage, basin/GT support
├── tests/                        # invariants, spatial, matching, score-only
├── experiments/
│   ├── manifest.csv
│   └── EXPNNN_NAME/{config.yaml,train.log,metrics.json,hashes.json}
├── outputs/
│   ├── dev/
│   └── confirmatory_test/       # read-only after export
└── paper/
    ├── figures/
    ├── tables/
    └── claims_evidence_matrix.md
```

**Không di chuyển/ghi đè** `diagnosis_2026-10-11/`, cached prediction, A/B/C, EXP020/021, hay historical test/val. Tạo repo branch/copy cho protocol mới và commit SHA chính xác.

### 10.2. Môi trường và dependencies

- Trước hết dùng **môi trường hiện hành `cs2_venv/bin/python -B` nếu nó tồn tại trong repo** (theo báo cáo), khóa `pip freeze`, torch/torchvision/CUDA versions, GPU driver và hash của scripts.
- Lưu full `sys.version`, `torch.__version__`, `torchvision.__version__`, `nvidia-smi`, OS, GPU, memory and package hashes.
- Tránh tự động nâng `torchvision` khi environment hiện có có compatibility issue. Nếu cần env mới, tạo **separate reproducible env**, chạy full metric parity tests.
- Nếu dùng external Mask2Former official implementation, nên pin exact commit và container digest; **không** phỏng đoán config/default details từ paper Nakada (paper không báo đầy đủ query/loss/augmentation).

### 10.3. GPU ngân sách hạn chế: setup thực tế

Không bắt chước trực tiếp Nakada dùng RTX 3090 24GB cho `1024²` Mask2Former với GPU laptop giới hạn VRAM. Kế hoạch:

1. **Stage 0 no GPU:** geometry/RLE tests, split audit, metrics, seed/score controls.
2. **Stage 1 frozen probe:** cache P2 features train-only; F, G1 MLP và Q_light nhỏ; crop `256²` ban đầu, sau đó `512²` hoặc physical-context tiling nếu VRAM cho phép.
3. AMP `bf16` nếu phần cứng/hỗ trợ ổn, ngược lại `fp16` + GradScaler; gradient accumulation và activation checkpointing cần lưu config.
4. Batch size thực tế chọn theo VRAM profiling trước train, không âm thầm so model có effective batch khác vì OOM.
5. Cache embeddings trên disk để giảm GPU-hours cho G và grouping evaluation; phải giữ source split, no cross-fold contamination.
6. Full Mask2Former `1024²` có thể cần GPU server/cloud 16–24GB+; nếu không có compute thì đánh giá bản nhỏ trước và **không gọi nó equivalent to Nakada architecture/training**.
7. Đồng hồ benchmark: wall time, effective batch, number of updates, total GT-instance exposures, TFLOPs/FLOPs nếu tính được, peak VRAM và convergence.

### 10.4. Training recipe common control

- Same RGB normalization/color space; augmentation train-only; paired augmentation RNG where meaningful.
- Same source train data and label QA; no test crop stats used for normalization if normalization fitted rather than fixed constants.
- Train logging per step: loss components, effective LR, batch instance-count histogram, empty-crop rate, gradient norms, NaN/overflow, pixel targets, query match counts, candidate counts.
- Seed plan: at least 3 newly controlled paired seeds initially, **khác** với historic 42/43/44 config; final seed count và model-selection exposure dựa trên power/resource plan.
- Hyperparameter search budget: number of trials, GPU-hours, tuning metric from dev only. G1 extra head params and Q queries count charged to total budget.
- Early stopping/patience/validation interval/checkpoint choice **predeclared**. Thử nghiệm architecture khác loss được chấp nhận nhưng ghi loss/convergence confound.

### 10.5. Visual debugging không được thay metric

Chọn một **train-only canonical panel** gồm 0/1/2–4/>4 tán, high/low density, shadow, truncated, ambiguous. Sau fit Q, xuất raw logits, matched train queries, no-object scores, postthreshold masks, query duplicate overlay. Panel để debug, **không dùng để claim val/test**.

Ở development, tạo fixed **error gallery** từ frozen randomly selected examples và full quantitative counts; không cherry-pick 5 ảnh đẹp của external site.

### 10.6. Reproducibility invariants

- Kết quả `config_hash`, `dataset_hash`, `source_code_sha`, `weights_sha`, `split_hash`, `prediction_sha`, `metric_version` lưu một lần ở report.
- Mọi CSV row mang `scene_id`, `tile_id`, `annotation_id`, `model_seed`, `stage`, `policy_id`.
- Determinism: ghi bật/tắt CuDNN benchmark, deterministic algos, dataloader worker seed, system RNG; không tự khẳng định bitwise reproducibility trên mọi GPU.
- Calibration và model weights được export một cách riêng biệt; `score_only_control` assert byte-identical geometry.
- `evaluation.py` không được đọc `train_masks` hay GT matching để sửa mask prediction tại inference.

---
## 11. Ma trận thí nghiệm và thứ tự thực thi

> **Phân biệt:** `DONE` = đã báo chạy từ tài liệu dự án (phải kiểm file/manifest để tái lập); `PLANNED` = chưa được chạy; `BLOCKED` = cần dữ liệu/compute/protocol. Không mặc định kết quả bất kỳ thử nghiệm planned nào.

### 11.1. Phase 0 — Evidence freeze và audit (bắt buộc trước training chính)

| ID | Tác vụ | Input | Output + acceptance |
|---|---|---|---|
| `A00` | Hash/read-only lịch sử | A/B/C, EXP020/021, F–G–Q pilot, Q sanity report, three pasted reports | `source_ledger.csv`; no overwritten bytes |
| `A01` | Xác minh ontology + RLE/COCO | Original polygons, raster ownership | `reference_audit.csv`; 4 priority-empty IDs xử lý đúng |
| `A02` | Spatial overlap graph | Scene metadata + valid pixel bounds | `spatial_audit.json`; cross-split exact pixel overlaps = 0 cho split mới |
| `A03` | Split policy | Group graph + survey IDs + consent/license | `split_manifest.json` signed; `lockbox` chỉ nếu thật độc lập |
| `A04` | Metric parity tests | Synthetic & real tests | `unit_tests.json` all green, pycocotools parity |
| `A05` | Pre-register measurement spec v3 | Sampling, endpoints, C, ablations | File hash before train; no retrospective relabeling |
| `A06` | Data feasibility audit | ForestSeg/BAM/japan/private sources | `external_data_feasibility.csv` showing permitted benchmark options |

**Stop Gate G0:** Nếu thiếu `split_manifest`, target không thống nhất, hoặc không có raw stage outputs mà đề tài muốn đo, không được chạy confirmatory benchmark; chỉ được sửa hạ tầng hoặc chạy diagnostic pilot có gắn nhãn.

### 11.2. Phase 1 — Controlled training sanity and readout probes

| ID | Planned experiment | Cố định | Biến đổi | Primary readout |
|---|---|---|---|---|
| `B00` | Re-run historical F metric only | Historical hashes/models | Metric checker | Reproduce AP/TP/FP/groups trong tolerance |
| `B01` | F frozen probe | Frozen train-only E/P2, train split | Fit heads F, locked decoder | Baseline grouping/recall |
| `B02` | G0 replay | Cached fields/basins of F | Historical EXP020/021 control | Changed merge counts, pair metric |
| `B03` | G1_geometry | Same atoms/merge engine | Geometry-only learned edge score | Test basic learned-grouping gain |
| `B04` | G1_RGB | Same atoms/merge engine/budget | Pooled RGB feature relations | Incremental effect vs G1_geometry |
| `B05` | Q_light train-only sanity | Frozen E/P2 & train | Query masks | Overfit, no-object, query matching |
| `B06` | Q_light unseen dev | Frozen E/P2 train split | Q readout | Recall C0/C1, grouping, FP |
| `B07` | Score-only control | Byte-identical masks | Heuristic/OOF/calibrated scores | Structural invariant; E2E/rank change |
| `B08` | Output-stage parity | Pre- vs post-selection | Common schema | No fake Q disadvantage from unmatched slots |

**Sanity acceptance:** no NaNs, loss gradients, per-crop match count visible, 0-instance crop does not produce large persistent FP at chosen train diagnostic policy, basic train overfit possible. Không đặt threshold AP thấp/cao tùy cảm tính; lưu curves theo iterations và visual QC có provenance.

**Stop Gate G1:** Nếu Q sanity thất bại do target/matching pipeline, sửa implementation; không diễn giải đó là “query cannot individuate”. Nếu G1 edge targets chủ yếu ambiguous/mixed, định nghĩa lại target trên TRAIN hoặc tách branch G2; không tùy ý lọc khó trên test.

### 11.3. Phase 2 — Diagnostic intervention, architecture comparison

| ID | Experiment | Mục tiêu giải thích |
|---|---|---|
| `C00` | F/G0/G1/Q frozen readout, 3 paired seeds new | Compare *same representation* readouts |
| `C01` | `C0/C1 × size × density` under one-to-one recall | Detect whether Q rescues F proposal failures |
| `C02` | Basin-support curve + pure/mixed atoms | Determine G-merge reachable solution constraints |
| `C03` | `same/diff × core r1/r4/adaptive × distance` | Check grouping robust to anchor selection |
| `C04` | Threshold/score-policy × fixed geometry | Separate ranking/selection |
| `C05` | Candidate threshold + NMS + merge trace | Stage at which each GT/pred changes matching |
| `C06` | `G1 RGB ablation / geometry-only / shuffled features` | Test contribution of learned RGB relations |
| `C07` | Query K sweep on dev only, count-capacity audit | Separate query scarcity from individuation |
| `C08` | Morphology/scale/crop-sampling sensitivity | Test FOV/crop/stride confounds |

C0/C1 comparisons cần dùng fixed C from F within each paired seed. Tránh chọn stratum khi xem Q result. Đặc biệt nếu Q tốt ở C0, ghi thêm background FP, count error và query capacity; không gọi đó là pure representation advantage.

**Stop Gate G2:** Đánh giá chiến lược method chỉ sau khi biết loại lỗi nào được cải thiện và cái giá về FP/large crowns. Nếu G1 đủ tốt như Q nhưng confidence khác, ưu tiên sửa readout/score policy; không vội đặt tên method mới.

### 11.4. Phase 3 — End-to-end và stronger external baselines

| ID | Experiment | Guardrail |
|---|---|---|
| `D00` | Matched F end-to-end | Same pretraining, crop, selection budget, new seeds |
| `D01` | Matched G1 end-to-end / joint training (if justified) | G field-training cost included; ablate frozen vs joint |
| `D02` | Matched Q end-to-end | Same physical FOV or factorized comparison |
| `D03` | Mask2Former standard baseline | Pinned official recipe; scale compute or report limits |
| `D04` | TreeCoG baseline if code/data/license available | Compare fair dataset/split; otherwise literature-only |
| `D05` | Historical method vs reimplementation check | Don't claim superior if baseline undertrained |

Nếu compute có hạn, **không** phát biểu outperform Mask2Former nếu chỉ dùng Q_light 256² đối đầu kết quả public trên dữ liệu khác. Báo compute matching transparent; có thể mô tả constrained-resource comparison mà không thành SOTA claim.

### 11.5. Phase 4 — Robustness and independent confirmation

| ID | Test | Thiết kế |
|---|---|---|
| `E00` | Spatial new scene holdout | New spatial blocks/scene not used for method selection |
| `E01` | Cross-site unseen forest | Train-domain -> unseen site, same target definition |
| `E02` | Cross-acquisition/camera | Separate UAV flight/camera; GSD metadata + registration |
| `E03` | Cross-GSD | True optical res variation when available; synthetic downsample separate stress test |
| `E04` | Morphology/density OOD | Stratify by shapes/contacts seen in train vs holdout |
| `E05` | Crop-origin / tiling consistency | Shift tile origins on same world scene; match predictions across tilings |
| `E06` | Multi-annotator / field reference subset | Inter-annotator and semantic target uncertainty |
| `E07` | Calibration / abstention / operational policy | Threshold from calibration, evaluate per scene FP/hectare |

**Robustness claim ladder:** (1) synthetic perturbations, (2) independent spatial areas, (3) new campaign/sensor, (4) new forest/biome. Không gọi synthetic blur “cross-sensor validation”.

### 11.6. Minimum evidence table mỗi experiment phải xuất

```text
experiment_id | config_hash | dataset_sha | split_sha | seed | model_sha
site_id | scene_id | n_tiles | n_gt_unique | n_small_gt
ap50_95 | ap50 | ap75 | allmask_r_small | pred_fp_per_hectare
same_uncond | diff_uncond | balanced | pair_coverage
same_cond | diff_cond | boundary_f1_r4 | count_bias
C0_r_small | C1_r_small | mixed_basin_r | crop_edge_miss
runtime_p50/p95 | peak_vram | gpu_hours | note_on_selection_exposure
```

Những hàng trống phải là `NA(reason)`, không điền `0` hoặc xóa khỏi bảng.

---

## 12. Giả thuyết method mới và các phép bác bỏ

### 12.1. **Không phát minh method trước khi biết failure mechanism**

Chọn một và chỉ một **scientific mechanism claim** cho phiên bản method đầu:

- **M1 — Proposal recovery:** hình thành instance mới khi F không có candidate/seed đủ tốt, giữ FP được kiểm soát.
- **M2 — Repartition mixed atoms:** cho phép phân một basin đã chứa nhiều GT mà merge-only không thể giải quyết.
- **M3 — Revisable part–whole grouping:** kiểm soát quyết định gộp/tách giữa các vùng/cụm dựa trên bằng chứng identity và global consistency, thay vì quyết định vĩnh viễn theo saddle cục bộ.
- **M4 — Cross-crop identity consistency:** dự đoán cùng một tán ổn định khi dịch tile/scale, không dựa thuần threshold IoU 0.1 stitching.
- **M5 — Morphology-robust representation:** giữ chất lượng individuation khi hình thái tán chưa thấy thay đổi, không chỉ tăng chất lượng mask in-domain.

Không thể claim cả M1–M5 nếu chỉ có vài ablation.

### 12.2. M1/M2/M3 design sketches — **chưa phải method**

**M1:** candidate generation có `birth proposal` từ RGB object features/queries nếu F peak không đủ; train existence với positive/negative **train GT**; evaluate rescue in fixed F-C0, FP negatives và duplicate. Compare directly Q_light và generic proposal augmentation.

**M2:** `G_repartition`: start from basins, predict pixel-to-hypothesis affinities hoặc sub-basin cut; min-cost assignment under exclusivity/overlap rules; compare to applying learned merge only and generic Mask2Former. Cần **tạo extra supervision** chứ không dùng GT at inference.

**M3:** iterative hypothesis repair, nhận graph of tentative regions, có actions `{keep, merge, split, abstain}` và learned cost. Đặt tiêu chuẩn stopping/cycle control, permutation invariance, cost/accuracy ablations. Phải đối chiếu InstanceCut, TreeCoG, recurrent grouping, object queries và correlation clustering để tránh đặt tên mới cho ý tưởng cũ.

### 12.3. Objective minh họa — không mặc định tối ưu được

Đối với partition `Z` và hypotheses `H`:

\[
(Z^*,H^*)=\arg\min_{Z,H}
\left( E_{exist}(H\mid I)
+\lambda_g E_{group}(Z,H\mid I)
+\lambda_s E_{shape}(Z,H\mid I)
+\lambda_c E_{consistency}(Z,H)\right).
\]

Một công trình method chỉ được trình bày như vậy khi:

1. Định nghĩa chính xác mỗi term, variables, constraints (incl. background, no-object, overlap).
2. Có thuật toán training/inference tính được, complexity scaling theo pixels/instances.
3. Có proof/structural property **nếu** tuyên bố; không nhất thiết phải có theorem mới để publish.
4. Có ablation cho thấy term/action mới khắc phục failure dự kiến.
5. Có external comparisons và reported negative cases.

### 12.4. Falsification tests bắt buộc cho mọi method proposal

| Claim | Bằng chứng bắt buộc | Kết quả phủ định |
|---|---|---|
| “Improves individuation” | Unconditional same/diff, coverage, one-to-one recall | Chỉ AP/ranking tăng |
| “Recovers missing objects” | C0 rescue + FP control + one-to-one matching | Chỉ nhiều raw proposals, FP tăng |
| “Solves mixed basin issue” | Performance on `B_mixed` vs G-merge atom bound | Không cứu được atom mixed, chỉ merge khác |
| “More generalizable” | New site/flight plus harmonized target, CI | Chỉ synthetic shift hoặc selected examples |
| “Independent of boundary” | Grouping core sensitivity + geometry controls | Chênh lệch hết sau mask/score alignment |
| “Better than query/graph prior work” | Matched trained strong baselines | Baseline weak/undertrained hoặc không cùng protocol |
| “Novel AI method” | Related-work differentiation at component level | Chỉ rebranding InstanceCut/TreeCoG/Mask2Former |

### 12.5. Stop rule để tránh endless architecture search

Nếu sau **hai chu kỳ** hypothesis-driven prototype (mỗi chu kỳ 1 mechanism + locked dev ablation) không có cải thiện reliable trên primary endpoint và FP guardrail, dừng thêm module mới; quay lại target/reference/representation diagnosis. Số chu kỳ có thể thay bằng ngân sách đã thống nhất; không sửa stop rule sau khi thấy test.

---

## 13. Thống kê, bất định và effect/equivalence margins

### 13.1. Đơn vị inferential, không phải số anchor pairs

- Với site-level generalization, **site** hoặc **independent acquisition** mới là đơn vị mạnh. Nhiều tile từ một mosaic không thành nhiều site samples.
- Within-site paired analysis: dùng overlapping spatial groups và scene-aware clustered bootstrap/block randomization; **không** bootstrap từng anchor pair, crown–seed hay pixel IID.
- Seeds là yếu tố optimization repeat, không giải quyết spatial sampling uncertainty. Báo variation across seeds **và** across independent scenes/sites.
- Nếu số site độc lập rất nhỏ, sử dụng descriptive per-site/paired differences, nêu rõ CI site-level không ổn định. Không giả tạo 95% CI rất hẹp từ hàng triệu pair.

### 13.2. Paired comparisons

So `F vs G1`, `F vs Q`, `G1 vs Q` trên **cùng holdout scenes** và annotation IDs. Báo paired difference per site/block, not only pooled AP. Mỗi pair có config provenance.

Mean difference theo source unit `u`:

\[
\Delta_{A,B}=\frac{1}{U}\sum_{u=1}^U (m_A(u)-m_B(u))
\]

Biểu thức trên là minh họa macro. Nếu metric như AP không thể cộng tuyến tính theo scene, tạo pooled AP và scene-level metrics riêng; không gọi mean per-scene AP là pooled COCO AP.

### 13.3. Effect/multiplicity/equivalence — **giá trị cần khóa, chưa chốt**

Đề xuất quy trình:

1. Chọn một primary contrast `F vs Q` **hoặc** `F vs G1` làm confirmatory; các contrast còn lại secondary hoặc điều chỉnh multiple testing.
2. Xác định **minimum practically meaningful improvement** trên grouping và small-crown recall với chuyên gia lâm nghiệp và pilot variability từ **development**, không dựa trên final test.
3. Chọn test/CI và α adjustment trước run: family-wise Holm/closed testing hoặc cách tương đương được biện minh; báo p-values và interval khi dữ liệu hỗ trợ.
4. `G ≈ Q` là claim equivalence chỉ khi CI nằm hoàn toàn trong `[-δ,+δ]` đã khóa cho endpoint tương ứng, không phải p>0.05.
5. Báo tradeoff noninferiority cho medium/large crowns và FP budget. Không gọi “better method” nếu cứu small nhưng phá phần lớn 4–9m².

**Minh họa, KHÔNG phải cutoff chính thức:** giả sử nhóm nghiên cứu chọn trước `δ_group = 0.03` absolute probability và `ΔR_small_min = +0.05` absolute, phải giải thích vì sao các giá trị có ý nghĩa; nếu không có căn cứ/power, đánh dấu TBD và không hứa superiority. Thay bằng ngưỡng đã ký trong `LOCKED_EVAL_PROTOCOL.md`.

### 13.4. Missing/negative results

- Không chọn best seed post hoc để so với mean baseline.
- Report outliers, all seeds/site subgroup including failed fits; `training_failure` có mã nguyên nhân (OOM/NaN/bug) được định nghĩa.
- Metric có denominator 0 => `NA`, giữ count. Không bỏ những fold có zero positive và chỉ báo cáo positive crops.
- Giữ test untouched sau chấm, ngay cả khi phát hiện model thua; nếu discovery mới từ test cần external new confirmation độc lập.

---

## 14. Cross-domain, transfer, calibration và triển khai

### 14.1. Taxonomy domain shift

| Shift | Biến quan sát | Test phù hợp | Khẳng định tối đa |
|---|---|---|---|
| Radiometric | brightness/contrast/gamma/shadow | Controlled perturbations train/val plus test | Sensitivity to synthetic nuisance |
| Blur/noise | imaging degradation | Controlled corruption curves | Corruption robustness |
| Resolution / GSD | m/pixel, resampling kernel | Real cameras/altitudes, simulated separately | Real-resolution robustness nếu có real acquisitions |
| Phenology | leaf-on/off, seasonal texture | New acquisition season same site | Cross-season transfer trong sites được kiểm |
| Morphology/species | crown topology/shape, forest type | Held-out forest composition | Morphological/site generalization |
| Density/structure | objects per hectare, crown contact | Stratified held-out scenes | Robustness by density/contact |
| Sensor | camera, radiometry, lens, SfM processing | Cross-camera/flight | Cross-sensor performance |
| Annotation ontology | polygon tracing, ambiguity, amodal policy | Harmonized reference audit | Label convention robustness, **not** pure covariate shift |

### 14.2. Preprocessing invariance không được chọn theo external test

- Khi dùng augmentation radiometric, noise, rotation, crop shift, lock ranges từ train/development protocol. Không tune để thắng cụ thể Borneo/ForestSeg final site.
- Geometric transform cần áp dụng đồng bộ RGB/GT/flows; flow vector phải rotate/scale đúng, SDT physical unit cần recompute hoặc scale đúng, centroid offsets tương ứng. Đây là một nguồn code bugs rất dễ tạo false method gain.
- Tách **equivariance** (dự đoán biến đổi theo phép transform) khỏi **invariance** (prediction identity/score giữ nguyên) theo đại lượng phù hợp.
- Crown instance identity **không** theo query index ổn định; qua transform phải match masks/GT theo world coordinates.

### 14.3. Cross-crop/tiling identity consistency

Cùng orthomosaic `I`, lấy hai tilings khác nhau `a,b` với crop origins/overlap cố định trước. Reproject masks về cùng world-coordinate domain. Match predictions one-to-one **không dùng GT để chọn correspondence trong primary consistency metric**; GT chỉ dùng đánh giá accuracy sau đó.

Báo:

- Pairwise matched prediction IoU cho các instance có match, **cùng omission/appearance rate** để không chỉ chấm những instance sống sót cả hai tiling.
- Count agreement and over/under-seg transitions per physical region.
- Jaccard of covered foreground, unassigned/overlap rate.
- Error at tile border vs interior, performance under shift of tiling grid.
- Effect of cross-tile IoU merge thresholds; so sánh với threshold 0.1 trong Nakada **chỉ như tiền lệ**, không giả định threshold ấy là optimal hoặc source of errors ở paper.

### 14.4. Confidence/rejection for operational reliability

Nếu `s` dự đoán quality, định nghĩa event `TP under one-to-one matching @τ` và compute calibration plots/ECE/Brier chỉ với sample unit/policy phù hợp. Đánh giá `selective risk vs coverage` và `FP per hectare` tại threshold calibration khóa. Không dùng output score để suy ra xác suất cây sinh học tồn tại khi reference chỉ là polygon RGB.

### 14.5. Science-outcome ladder

1. **Algorithm result:** cải thiện individual-crown AP/grouping trên unseen site.
2. **Measurement result:** giảm **bias** trong count/density/crown-size distribution so với audited reference; không nhất thiết cần phát minh ecological theory.
3. **Ecological result (optional):** nếu có field-verified tree inventories, làm rõ sai số phân đoạn ảnh hưởng ước lượng cấu trúc rừng. Không tự đồng nhất diện tích tán nhỏ với cây non hoặc biomass.
4. **Wide-scope claim:** nhiều hệ sinh thái/cảm biến/sites và có uncertainty rõ; chỉ mở khi đủ evidence.

---
## 15. Literature map và kiểm định tính mới

> Nguồn dưới đây là **tiền lệ cần đọc/đối chiếu**, không phải bằng chứng đã có algorithmic novelty. Với source HTML/PDF thay đổi phiên bản, lưu DOI/arXiv version, date accessed và code commit trong `literature_matrix.csv`. Không copy AP từ dataset khác để trực tiếp so với QuebecTrees.

### 15.1. Literature bắt buộc nhóm 1 — tree crown instance segmentation

| Công trình | Phương pháp/dữ liệu | Liên quan trực tiếp | Chưa được dùng để chứng minh |
|---|---|---|---|
| [Nakada et al. (2026), *Highly Detailed and Generalizable Broadleaf Tree Crown Instance Segmentation from UAV Imagery*](https://arxiv.org/abs/2605.15673) | 18,507 manual polygons từ 7 Japanese forest development sites; Mask2Former R50/R101/Swin-T/S; RGB-only **model input**; CHM có thể trợ giúp annotate; best Swin-S `mAP=23.1±0.8`, test site; 3 external sites assessed visually | Strong Q baseline; explicit train/test site split; tile inference with 80% overlap and IoU≥0.1 merge | Quantitative AP on Borneo; proof CHM unnecessary universally; proof high-quality annotation causes transfer; query solves identity |
| [Do et al. (2026), *A UAV RGB Dataset and Method for Instance Tree Crown Segmentation for Biodiversity Monitoring*](https://www.nature.com/articles/s41598-026-36541-y) | TreeCoG: contour extraction → feature representation → GCN merging; ForestSeg multi-acquisition 2,944 images | **Closest G competitor**; part-to-whole merging; cross-dataset stress testing | New GGCN alone is novel; papers' AP can be compared to Z1 AP |
| [Ball et al. (2023), *Accurate delineation of individual tree crowns in tropical forests from aerial RGB imagery using Mask R-CNN*](https://doi.org/10.1002/rse2.332) | Mask R-CNN for tropical crowns | Direct RGB crown instance baseline (Detectree2); manual reference có thể được hỗ trợ bằng LiDAR/hyperspectral, vì vậy không đồng nhất RGB-input với RGB-only annotation | Object queries are first method to tackle RGB crowns |
| [Weinstein et al. (2020), *Cross-site learning in deep learning RGB tree crown detection*](https://doi.org/10.1016/j.ecoinf.2020.101061) | RGB crown detection across sites | Tree localization generalization, detection vs masks | Detection boxes = full crown delineation |
| [Zheng et al. (2025), *A Review of Individual Tree Crown Detection and Delineation from Optical Remote Sensing Images*](https://doi.org/10.1109/MGRS.2024.3479871) | Survey | Taxonomy/baselines/datasets | Comprehensive novelty clearance by survey alone |

**Source-specific warning — Nakada p. 6–10:** quantitative test is one held-out development site; external Sanpoku/Sadayama/Borneo results visual only; selected example panels are not statistical proof of external generalization. Train tiles 50% overlap, inference 80%; external stage removes tile-edge touching predictions and merges duplicates at mask IoU≥0.1. Need report this as a **hypothesis-sensitive design choice**, not accuse source of a measured mistake without evidence.

### 15.2. Nhóm 2 — fields, center, distance, flow

| Paper | Basic mechanism | Novelty danger for TreeFlowNet |
|---|---|---|
| [Bai & Urtasun (2017), Deep Watershed Transform](https://openaccess.thecvf.com/content_cvpr_2017/html/Bai_Deep_Watershed_Transform_CVPR_2017_paper.html) | Learned energy/watershed-like field | Basin-to-instance is established |
| [Cheng et al. (2020), Panoptic-DeepLab](https://openaccess.thecvf.com/content_CVPR_2020/html/Cheng_Panoptic-DeepLab_A_Simple_Strong_and_Fast_Baseline_for_Bottom-Up_Panoptic_CVPR_2020_paper.html) | Center heatmap + per-pixel offsets | Center/flow assignment is established |
| [Neven et al. (2019), Spatial embeddings and clustering bandwidth](https://openaccess.thecvf.com/content_CVPR_2019/html/Neven_Instance_Segmentation_by_Jointly_Optimizing_Spatial_Embeddings_and_Clustering_Bandwidth_CVPR_2019_paper.html) | Embedding-based instance clustering | Learned identity embeddings/bandwidth not new |
| [Stringer et al. (2021), Cellpose](https://www.nature.com/articles/s41592-020-01018-x) | Learn flows, obtain cell instances | Flow-based object extraction has strong predecessors |
| [Cutler et al. (2022), Omnipose](https://www.nature.com/articles/s41592-022-01639-4) | Gradient/distance-based segmentation robust to complex cell shapes | Changing distance/flow representations is not automatically novel |

Không ngầm giả định cây và tế bào cùng topology/signal/noise. Cellpose/Omnipose có thể cung cấp **principle** để kiểm, không phải trực tiếp chứng minh tán cây sẽ hưởng lợi.

### 15.3. Nhóm 3 — global grouping, part–whole, queries

| Paper | Basic mechanism | Novelty danger |
|---|---|---|
| [Kirillov et al. (2017), InstanceCut](https://openaccess.thecvf.com/content_cvpr_2017/html/Kirillov_InstanceCut_From_Edges_CVPR_2017_paper.html) | Semantic + instance boundary, global multicut | Graph partition + contour evidence is old |
| [De Brabandere et al. (2017), Discriminative Loss](https://arxiv.org/abs/1708.02551) | Pull same-instance pixels, push different-instance embeddings | Per-instance embedding supervision old |
| [Kong & Fowlkes (2018), Recurrent Pixel Embedding](https://openaccess.thecvf.com/content_cvpr_2018/html/Kong_Recurrent_Pixel_Embedding_CVPR_2018_paper.html) | Iterative pixel grouping | “Repeated grouping correction” may have prior art |
| [Carion et al. (2020), DETR](https://arxiv.org/abs/2005.12872) | End-to-end set prediction, bipartite matching | Query-based object output old |
| [Tian et al. (2020), CondInst](https://arxiv.org/abs/2003.05664) | Dynamic per-instance mask prediction | Dynamic mask head old |
| [Cheng et al. (2022), Mask2Former](https://arxiv.org/abs/2112.01527) | Masked attention and mask classification | Stronger Q baseline; query masks already established |
| [Locatello et al. (2020), Slot Attention](https://proceedings.neurips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html) | Iterative competitive attention over slots | Object-centric ≠ guaranteed complete crown |

### 15.4. Nhóm 4 — independent validation và distribution shift

| Paper | Dùng làm gì trong protocol |
|---|---|
| [Roberts et al. (2017), Cross-validation strategies for structured data](https://doi.org/10.1111/ecog.02881) | Spatial/temporal/hierarchical dependencies and blocked validation |
| [Valavi et al. (2019), blockCV](https://doi.org/10.1111/2041-210X.13107) | Spatial/environmental folds, block size considerations |
| [Koh et al. (2021), WILDS](https://proceedings.mlr.press/v139/koh21a.html) | Realistic in-the-wild distribution shift protocols |
| [Gulrajani & Lopez-Paz (2021), DomainBed](https://arxiv.org/abs/2007.01434) | Fair model selection and strong ERM baseline in domain generalization |
| [Steier et al. (2024), *Is Your Training Data Really Ground Truth?*](https://doi.org/10.3390/rs16152786) | Manual annotation reliability vs tree reference and UAV structural data |
| [Chowdhury et al. (online 2026 / volume 2027), standing dead-tree review](https://doi.org/10.1016/j.isprsjprs.2026.09.050) | Target definition, independent field references, transferability; note dead-tree != ordinary crown task |

### 15.5. Mechanism-vs-prior-art matrix — phải hoàn thành trước method proposal

Mỗi prospective method module ghi theo định dạng:

```text
Proposed operation:          _______
Nearest prior method:       _______
Prior method's assumption:  _______
Observed failure evidence:  _______
New mathematical operation: _______
New supervision/constraint: _______
What prior cannot express?  _______  (proof, counterexample, or bounded evidence)
Minimal ablation:           _______
Added parameters/compute:   _______
Failure case where ours is no better: _______
Datasets with independent confirmation: _______
```

**Không** coi tổng hợp các module quen thuộc là novelty tự động. Novelty phải được đối chiếu ở mức *capability*, constraint, objective hoặc inference operator; không chỉ kiến trúc block names.

---

## 16. Go/No-Go gates, rủi ro và lịch thực hiện

### 16.1. Lịch 12 tuần mẫu — research sprint, không hứa publish trong 12 tuần

| Tuần | Deliverable | Gate |
|---|---|---|
| W1 | Evidence registry/hash, source/annotation audit, leakage graph | Không chạy confirmatory nếu source unresolved |
| W2 | Split/metric v3, toy+RLE tests, sampling strata | `G0` pass |
| W3 | Q sanity + F/G0 replay, train-only masks lineage | No-object/assignment sanity pass |
| W4 | Frozen F/G1_geometry/G1_RGB/Q probe pilot | Compute/feature parity verified |
| W5 | Grouping metrics, C0/C1, FP count, geometry controls | `G1`/`G2` family decision |
| W6 | Mechanism counterexamples on dev, TreeCoG/Mask2Former audit | Novelty map v1 |
| W7 | Stronger baseline/matched end-to-end (as compute permits) | Quality vs compute documented |
| W8 | Decide one new method mechanism OR stop/redefine | Hypothesis and effect size logged |
| W9 | One-module method prototype + negative ablations | No post hoc tests |
| W10 | Spatially separated dev evaluation + reliability | External holdout ready? |
| W11 | Freeze method/config, independent confirmation if available | Pre-test lock complete |
| W12 | Failure analysis, claims–evidence matrix, draft | No Nature claim without broad confirmation |

**Nếu independent external data chưa có ở W11:** không chạy fake confirmatory test. Chuyển tuần W11–W12 thành acquire/annotate/QA hoặc viết exploratory methods report; lùi publication timeline.

### 16.2. Decision tree

```text
Do data/reference and independent split pass audit?
  NO -> stop method claims; repair dataset/annotation/split.
  YES -> run controlled F/G0/G1/Q on development.

Does Q recover F-C0 small GT while controlling FP?
  YES -> object formation/proposal-recovery candidate method.
  NO  -> analyze missing feature, mask supervision and geometry.

Does G1 improve with intact F basins?
  YES -> relational grouping candidate, test whether query still adds value.
  NO  -> check basin impurity/limited search space; consider repartition.

Does any winner only improve AP, not grouping/recall?
  YES -> score/ranking work, not object-individuation claim.

Does an advantage survive new sites/acquisitions and annotation audit?
  YES -> stronger reliable method claim.
  NO  -> domain/reference-specific phenomenon, revise scope.
```

### 16.3. Risk register

| Risk | Observable indicator | Mitigation / honest limitation |
|---|---|---|
| Data leakage | Shared source pixels across folds, same crowns/campaign | Split at scene graph + source IDs, new holdout |
| Reference mismatch | Inter-annotator splits, 4 priority-empty masks, CHM-assisted inconsistency | R0/R1/R2 levels, original RLE, annotation policy |
| Q unmatched slot bias | Low-score raw logits overlap GT | Track no-object semantics, avoid raw-slot unfair metric |
| G graph false bridges | Edge AUC high but partition low | Evaluate global connected components and FP merge cost |
| G cannot split basin | Atom purity low, mixed GT inside basin | G2 separate search space, atom bound diagnostic |
| Small crown erosion bias | Core r4 retained area drops to ~52% median | r1/r4/adaptive sensitivity; no silent exclusions |
| Confidence scale confound | Rank flips under threshold selection | Structural + matched coverage + score-only controls |
| Cross-site ontology | Different tree definitions/labels | Harmonization, label uncertainty/abstain |
| Compute constraints | OOM, crop FOV mismatch | Frozen readouts, AMP, matching physical FOV, compute report |
| Repeated method search | Many trials on 139 Z1 val | Lockbox untouched; hypothesis-driven iterations |
| Overclaim for Nature | One dataset AP gain, no independent proof | Evidence ladder, external confirmation, ecology impact optional |

### 16.4. Minimal paper artifacts

- `target_ontology.md` + dataset/label QA report.
- `experimental_design.md` with split definition and exposure history.
- `model_comparison.csv` and per-site independent metrics.
- `candidate_lineage.parquet` and interpretability diagnostics.
- `pairwise_grouping_report.csv` and all policy/core stratification.
- `score_control_report.csv`, standardized calibration curves.
- `morphology_ood_report.csv`, `tiling_consistency_report.csv`.
- `ablation_matrix.csv`, `complexity_runtime.csv`, `reproducibility_manifest.json`.
- `failure_cases_blind_gallery` chosen by predeclared/random rules plus representative success.
- `claims_evidence_matrix.md` with precise counterexamples and limitations.

---

## 17. Evidence ladder hướng tới Nature Portfolio

**Nature Portfolio không có cơ chế bảo đảm chấp nhận bằng một mức AP.** Xác định đúng đóng góp và audience trước khi chọn nơi nộp. `Nature` (flagship) khác `Nature Communications`, `Nature Machine Intelligence`, `Nature Computational Science`, `Communications Earth & Environment`, `Scientific Reports` và `Scientific Data`.

| Evidence level | Loại kết quả | Mức claim hợp lệ | Publication positioning, không bảo đảm |
|---|---|---|---|
| **L0** | Pilot, oracle, val Z1 đã dùng phát triển | Diagnosis/hypothesis generation | Internal report/workshop nếu phù hợp |
| **L1** | Method mới vs strong baselines, sound independent test trong một domain | Application-specific algorithmic improvement | Computer Vision/Remote Sensing journal; tùy mức mới có thể Scientific Reports |
| **L2** | Mechanism explanation + ablations + multiple independent forests/sensors + reference QC | Reliable RGB crown individuation under tested shifts | Strong applied RS/forest journals; Nature Portfolio scope tùy significance |
| **L3** | Tác động đo đạc quần thể cây/định lượng uncertainty có field-verified consequences | New *scientifically meaningful* tree inventory capability | Communications Earth & Environment hoặc Nature Communications nếu insight đủ quan trọng |
| **L4** | General individuation principle tested across tree crowns **and unrelated object domains** | Broad AI/ML mechanism, with mathematical/empirical novelty | Nature Machine Intelligence/Nature Computational Science nếu thật sự vượt field |

### 17.1. Mức bằng chứng cần cho AI-method claim mạnh

1. **Named mechanism** khác với đã biết (TreeCoG/InstanceCut/Mask2Former/...); có theoretical argument hoặc counterexample đủ rõ.
2. **Failure provenance:** biết method sửa đúng loại lỗi nào, không chỉ AP.
3. **Independent replication:** scenes/sites/sensors độc lập, locked protocol.
4. **Strong baselines** đủ budget, reproducible open implementation.
5. **Uncertainty/statistics:** site/scene-level sampling error, seed variance, negative results.
6. **Reference validity:** multi-annotator hoặc field/CHM-assisted verification phù hợp target, không overclaim biological tree.
7. **Efficiency/deployment:** chi phí tính toán và mosaic inference/stitching được mô tả.

### 17.2. Mức bằng chứng cần cho ecological-science claim

- Independent inventories hoặc forest plot data có nguồn xác minh.
- Khả năng giảm bias của crown-size/density distribution trên cùng target.
- Phân tích propagating measurement uncertainty tới chỉ số sinh thái được định nghĩa trước.
- Không suy ra tree mortality, biomass, biodiversity hay carbon chỉ từ AP/contour tốt hơn.
- Sự hợp tác với ecologist/forest inventor và field validation sẽ làm tăng chất lượng research design, nhưng không tự động tạo nature-level discovery.

### 17.3. Vùng claim phải từ chối cho tới khi có bằng chứng

- “First model to learn object identity from RGB.” (Có nhiều tiền lệ.)
- “Human-like perception of crowns.” (Không có nghiên cứu nhận thức học/brain evidence.)
- “Generalizable worldwide.” (Một test site + 3 inference illustrations không đủ.)
- “Biologically accurate tree count.” (RGB dataset polygon không phải field-verified all stems.)
- “This oracle guarantees +13.47 AP.” (Oracle uses GT and may change topology.)
- “Q learned intrinsic identity because its AP rose.” (Score/geometry can affect AP.)
- “Spatially independent OOF.” (Historical 378 cross-fold overlapping pixel pairs.)

### 17.4. Khung bài báo tương lai (không viết kết quả trước)

1. **Introduction:** crown individuation under ambiguous multi-part geometry; operational cross-site need.
2. **Problem definition/reference:** target, observability, label uncertainty.
3. **Methods:** representation, formation, grouping and confidence; mechanism claim precisely isolated.
4. **Study design:** geospatial split, labels, external domains, controls and strong baselines.
5. **Results I:** where legacy field/query/graph methods fail (mechanistic diagnosis).
6. **Results II:** primary grouping/existence endpoints + AP/geometry; ablations.
7. **Results III:** independent site/camera/morphology and tiling consistency.
8. **Discussion:** mechanism limits, annotation ambiguity, operational costs, biological scope.
9. **Reproducibility:** code and configs/hash, data terms, full failure cases.

---

## 18. Checklist, manifest và execution config mẫu

### 18.1. One-page experiment readiness checklist

#### Target & data

- [ ] Primary target `visible/annotated crown` được viết không mơ hồ.
- [ ] Original COCO RLE và priority raster được phân biệt; flags for overlapping annotations.
- [ ] Source GSD/CRS/projection valid; area `m²` audited.
- [ ] Split at scene/site/campaign/source component; cross-split pixel overlaps `0` cho confirmatory.
- [ ] New independent holdout xác định trước (hoặc study chỉ exploratory).
- [ ] Rights/permissions cho external dataset rõ.

#### Model & train

- [ ] F old/final replay; G0 EXP020/021 raw files checked.
- [ ] F, G1, Q input feature scales và checkpoint được công bố.
- [ ] Q no-object policy/K_max; G1 atom purity/edge target và split-support rõ.
- [ ] Train-only augmentations, sampler, budget, paired seeds locked.
- [ ] Strong external Mask2Former/TreeCoG baseline feasibility decided.

#### Evaluation

- [ ] One-to-one cardinality-first matching tested; COCO AP parity.
- [ ] Structural vs E2E and output-stage compatibility.
- [ ] R1/R4/adaptive anchors and all distance/class bins locked.
- [ ] Score-only geometry hash assertion.
- [ ] FP/ha, count error, large-crown guardrail, no-object crops.
- [ ] Site-level uncertainty plan, effect/equivalence margins and multiple-testing locked.
- [ ] Lockbox test once after model freeze; all negative results kept.

### 18.2. Execution config mẫu — **schema đề xuất, KHÔNG phải YAML mà repo hiện tại chắc chắn đọc được**

```yaml
protocol:
  id: TFN_FGQ_V3_DRAFT
  mode: development             # development | confirmatory
  frozen_at_utc: null           # set once before confirmatory test
  study_target: dataset_defined_crown_instances
  no_biological_tree_claim: true

sources:
  source_registry: data_registry/scenes.csv
  tile_registry: data_registry/tiles.csv
  annotation_registry: data_registry/annotations.csv
  original_reference: original_coco_rle
  original_reference_is_read_only: true
  source_hash_manifest: data_registry/source_hashes.json

split:
  manifest: data_registry/split_manifest.json
  source_group: scene_or_survey
  split_before_tiling: true
  forbid_cross_split_valid_pixel_overlap: true
  buffer_distance_m: null      # choose based on correlation study; do not invent
  independent_final_test_available: false   # UPDATE only after audit
  legacy_Z1_role: hypothesis_development

input:
  modality: RGB
  gsd_m: per_scene
  crop_px: 256                 # starting low-VRAM probe, not final claim
  inference_crop_px: null      # match physics FOV or separate ablation
  color_norm: train_recipe_pin
  augmentation_recipe: train_only_pin

feature:
  encoder_checkpoint: null    # exact hashed train-only checkpoint
  backbone: existing_TreeFlowNet_encoder
  frozen_probe: true
  scales_primary: [P2]
  multiscale_is_separate_ablation: true

branches:
  F:
    enabled: true
    heads: [flow, sdt, centroid, canopy]
    decoder: locked_field_saddle
    fields_cached: true
    score_policy: locked
  G0:
    enabled: true
    reference_experiments: [EXP020, EXP021]
  G1_merge:
    enabled: true
    atom_source: F_premerge_basins
    node_features: [pooled_P2, area, geometry, sdt]
    edge_features: [symmetric_pooled, saddle, distance]
    target_policy: TBD_before_fit
    global_merge_engine: same_as_F_initial_ablation
    can_split_existing_atom: false
  Q_light:
    enabled: true
    source_features: [P2]
    query_count: null          # define by training distribution / budget
    matcher: hungarian
    no_object_weight: null
    unmatched_mask_loss_policy: standard_explicit
    output_stage_contract: TBD

training:
  gpu_memory_budget_gb: null
  precision: auto_bf16_or_fp16
  crop_sampler: include_zero_one_few_many
  paired_seeds: [101, 202, 303]
  effective_batch: null
  max_optimizer_updates: null
  learning_rate: null
  weight_decay: null
  checkpoint_selection: development_only
  hparam_trials_per_branch: null
  gpu_hours_budget_per_branch: null
  reproducibility_manifest: required

measurement:
  primary:
    - one_to_one_allmask_recall_iou50_lt1m2
    - anchor_unconditional_group_balanced
  prediction_stage: common_preselection_semantics
  reference: original_RLE
  size_bins_m2: [1, 2, 4, 9]
  core_modes: [r1, r4, adaptive]
  primary_core_mode: null      # lock before benchmark
  max_anchors_per_gt: 8
  max_pairs_per_focal_class_distance_bin: 8
  distance_bins_m: [0, 0.25, 0.5, 1, 2, 4, null]
  pair_assignment_unassigned_is_incorrect: true
  overlap_structural_policy: abstain
  cardinality_first_matching: true
  iou_threshold_recall: 0.50
  e2e_score_threshold_grid: [0.1, 0.3, 0.5, 0.7]
  e2e_max_dets: 400
  e2e_overlap_policy: highest_score
  score_only_geometry_hash_assertion: true
  output_counts_and_na_reasons: true

candidate_strata:
  baseline_source: F_final_preselection
  C_inside_reference: original_COCO_mask
  C_is_fixed_within_paired_seed: true
  basin_support_curve: true
  mixed_atom_diagnostic: true
  no_causal_claim_from_C0_C1: true

statistics:
  unit_of_independence: site_or_spatial_scene
  clustered_by: [site_id, survey_id]
  sample_count_primary: unique_geospatial_units
  primary_contrasts: TBD_before_confirmatory_test
  min_effect_group_absolute: null
  min_effect_small_recall_absolute: null
  equivalence_margin: null
  multiple_testing_correction: TBD
  ci_method: TBD_by_independent_unit_count

external:
  datasets_under_consideration: [ForestSeg, BAMFORESTS]
  license_verified: false
  ontology_harmonized: false
  test_never_used_for_tuning: true
  cross_gsd_and_cross_site_results_required_for_broad_claims: true

report:
  include_negative_runs: true
  save_predictions_all_stages: true
  save_per_annotation_stats: true
  save_per_scene_stats: true
  save_hashes: true
  forbid_claim_novelty_before_prior_art_map: true
```

**Lưu ý:** các trường `TBD` và `null` là **blockers cần giải quyết**, không được auto-fill bằng giá trị convenient ngay trước test. Các giá trị seeds `101/202/303`, crop `256` là ví dụ để đối chứng mới, không phải lựa chọn đã được kiểm nghiệm.

### 18.3. Templates bàn giao agent

#### `experiment_manifest.csv`

```csv
experiment_id,branch,goal,source_commit,source_hash,config_hash,data_hash,split_hash,encoder_hash,seed,run_status,train_budget_gpu_h,dev_metric_primary_a,dev_metric_primary_b,selection_used_data,output_dir,comment
```

#### `claims_evidence_matrix.md`

```markdown
| Claim | Source experiment | Validation level | Primary metric | Effect size + CI | Known confound | Counterexample | Permitted paper wording |
|-------|-------------------|------------------|----------------|------------------|---------------|----------------|-------------------------|
| ...   | ...               | ...              | ...            | ...              | ...           | ...            | ...                     |
```

#### One run acceptance

```text
[ ] env/version/hash printed
[ ] loaded exact split manifest and no cross split overlap
[ ] source GT/feature hashes match
[ ] no train exposure to val/test source fragment
[ ] train loss/matcher/sampler/no-object logs valid
[ ] prediction lineage exported by stage
[ ] score-only test passes identity geometry assertion
[ ] all metric unit tests pass
[ ] macro stats include denominators & NA reasons
[ ] per-scene and per-seed metrics exported
[ ] no unintended oracle or GT usage at inference
[ ] report clearly marks exploratory vs confirmatory
```

### 18.4. “Next action” — nhiệm vụ đầu tiên cho agent

> **Task A00–A05 (không train):** kiểm toán repo hiện tại theo các nguồn `Pasted markdown(5/6/7)` và source paths nêu trong đó; dựng `source_ledger.csv`, data registry skeleton, exact spatial leakage report, validate split provenance, produce `measurement_spec_v3_draft.md`, tests common RLE/matching/anchor semantics, rồi liệt kê blockers trước confirmatory split. **Không sửa artifact lịch sử, không train F/G/Q, không chạy test.** Bàn giao diff của docs/config và command line dùng được thực tế (không dùng các placeholder command của tài liệu này như code chạy sẵn).

> **Task B00–B08 (chỉ sau A pass):** chạy controlled F/G/Q frozen probe trên train và dev với cùng P2, paired seed, candidate/score/geometry controls. Chưa chọn method mới hoặc claim generalization.

---

## 19. Nguồn và ghi chú xuất xứ

### 19.1. Tài liệu dự án đã được đối chiếu khi soạn kế hoạch

1. `Pasted markdown(5).md` — *Diagnosing Representation and Reconstruction Bottlenecks in RGB Tree Crown Instance Segmentation*, báo cáo A/B/C ngày 11/10/2026. Nguồn của số AP, oracle, C, seed config, priority raster overlap và cảnh báo liên kết GT.
2. `Pasted markdown(6).md` — *Từ trực giác tri giác đến giả thuyết kiểm chứng được về từng tán cây*, 12/10/2026; protocol Q ban đầu; phân biệt RGB object individuation và mô phỏng tri giác.
3. `Pasted markdown(7).md` — *F–G–Q: đối chứng formation, grouping, geometry và confidence*, 13/10/2026; định nghĩa H-Q/H-G, G0 prior work, G-merge vs G-repartition, frozen probe, structural/E2E, C0/C1 và holdout limitations.
4. User-provided 13/10/2026 *Báo cáo sửa protocol F–G–Q và pilot metric grouping*: structural r1/r4/adaptive, score-control, 219 small anchors, candidate recall C0/C1, 378 pixel/370 GT-positive cross-fold overlap, Q sanity state. Cần lấy các bản `FGQ_PROTOCOL_V2.md`, `MEASUREMENT_SPEC.md`, `SCORE_CONTROL_AMENDMENT.md`, `grouping.py`, `Q_SANITY_REPORT.md`, `manifest.json` từ repo để xác minh ở A00.
5. **Nakada et al.**, *Highly Detailed and Generalizable Broadleaf Tree Crown Instance Segmentation from UAV Imagery*, arXiv:2605.15673v1 (15 May 2026), 12 pp.; supplied `/mnt/data/HIGHLY DETAILED AND GENERALIZABLE BROADLEAF TREE.pdf`: §2.4 manual+CHM reference; §2.5 50%-overlap tiles and split by forest; §2.7 COCO mAP and Swin-S; §2.8 80%-overlap inference, border removal, IoU≥0.1 stitching; §2.9 visual-only external inference; Table 3 best Swin-S `23.1 ±0.8` AP. [Preprint](https://arxiv.org/abs/2605.15673).
6. **Chowdhury et al.**, *Remote sensing and deep learning for standing dead-tree detection and mapping: a review of advances, challenges, and future directions*, ISPRS JPRS, online 7 Oct 2026; Volume 243 (2027), pp. 1–22, supplied review PDF. **Scope khác:** standing-dead-tree mortality; sử dụng để tham chiếu target/reference/transfer principles, không coi là benchmark direct regular crown segmentation. [DOI](https://doi.org/10.1016/j.isprsjprs.2026.09.050).

### 19.2. Khác biệt giữa nguồn, đề xuất và phần cần xác minh

- **Số đo lịch sử:** lấy từ user reports, chưa chạy lại trong lần soạn tài liệu này.
- **Paper Nakada:** lấy từ full 12-page PDF, các claim external quantitative được hiệu chỉnh đúng (không có AP external trong PDF).
- **F/G/Q recipe mới:** là **đề xuất protocol v3**, chưa train, chưa có positive results.
- **Literature map:** điểm khởi đầu, **chưa phải systematic novelty search**; phải mở rộng theo citation graph/keywords và kiểm code/assumptions trước paper submission.
- **Nature Portfolio:** chỉ là *research aspiration*, không đưa ra “khả năng đỗ %” hoặc hứa tạp chí chấp nhận.
- **Mọi config TBD:** chưa được ký và chưa thể gọi preregistered. Phải khóa ở protocol execution thật trước benchmark chính.

---

## FINAL RESEARCH DECISION (14/10/2026)

**Đích trước mắt không phải Nature hay method mới, mà là một phép kiểm có thể loại bỏ giả thuyết:** cùng dữ liệu/reference/feature, việc hình thành instance bằng F/G/Q khác nhau thế nào về existence, grouping, geometry và score; điều gì xảy ra ở những trường hợp field candidate vắng, basin pha trộn và cross-site mới.

**Bằng chứng mạnh nhất cần giành được:** cải thiện **one-to-one small-crown recall + same/different-tree grouping** ở coverage phù hợp, không phá FP/mid-large crowns, được xác nhận trên spatially independent và cross-site imagery có target harmonized; nguồn lợi được định vị nhờ matching, score-only và geometry controls.

**Chỉ sau đó** chọn một mechanism mới cho `birth`, `split/repartition` hoặc `part–whole grouping`, đối chiếu công khai TreeCoG/InstanceCut/Mask2Former/Deep Watershed/Cellpose–Omnipose và các EXP020/021. Nếu không có bằng chứng vượt prior art, công trình vẫn có thể trở thành một failure-analysis/benchmark/reliability study có giá trị, nhưng không được trình bày như phát minh cơ chế AI mới.
