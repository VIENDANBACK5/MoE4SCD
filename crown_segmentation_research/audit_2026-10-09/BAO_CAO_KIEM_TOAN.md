# Kiểm toán tính đúng đắn và liêm chính nghiên cứu

**Dự án:** `crown_segmentation_research` · **Ngày:** 09/10/2026 · **Loại:** kiểm toán mã, hồ sơ và kiểm chứng thực thi độc lập trên dữ liệu hiện có.

## 1. Kết luận

**Dự án có kết quả thực nghiệm hiện tại kiểm chứng được, nhưng chưa đủ điều kiện để công bố các kết luận mạnh về tính mới, ưu thế phương pháp hoặc hiệu quả tổng quát.** Cần phân biệt ba vấn đề:

1. **Các kết quả lịch sử không hợp lệ để làm bằng chứng nghiên cứu:** checkpoint từng thấy vùng test, thước đo F1 bị gọi là AP, và các bảng/biểu đồ dùng số cố định không được script tính từ dự đoán tương ứng. Các tài liệu mới đã rút lại phần lớn kết quả này; các tệp và công cụ cũ vẫn còn và vẫn có thể tạo báo cáo sai.
2. **Nhánh `clean_benchmark` đã cải thiện đáng kể:** dùng split công khai, khởi tạo ImageNet, COCO AP thực sự, nhiều seed, lưu dự đoán mask, có baseline mạnh hơn phương pháp đề xuất, và công khai các thí nghiệm âm. Việc chấm lại độc lập được ghi đầy đủ ở mục 3.
3. **Tính đúng của AP không bảo đảm tính hợp lệ của kết luận:** test đã tham gia vòng phát triển qua phân tích lỗi; lấy mẫu crop có lỗi RNG; ngân sách lựa chọn và backbone chưa cân bằng; khoảng tin cậy chưa phản ánh đầy đủ phụ thuộc không gian và bất định giữa lần train; provenance chưa gắn với commit bất biến.

**Không có đủ bằng chứng để kết luận có chủ ý gian lận.** Có bằng chứng trực tiếp về lỗi metric, trình bày số thiếu nguồn thực nghiệm đồng bộ, mất một kết quả/log đã được nhật ký thừa nhận và diễn giải quá mức. Đó là những vấn đề liêm chính cần sửa ở cấp hồ sơ và công bố, bất kể nguyên nhân là lỗi người làm hay agent.

Không cấp chứng nhận ARA: skill `ara-rigor-reviewer` được đọc nhưng điều kiện ARA/Level 1 không tồn tại, và workflow đó không phù hợp với kiểm toán có thực thi này.

## 2. Phạm vi và cách kiểm tra

Kiểm kê toàn bộ cây dự án: **17.902 tệp, khoảng 24,45 GB**, không tính thư mục audit và `__pycache__`. Phân tích cú pháp **286 tệp Python**, kiểm tra **1.376 JSON**, nhận dạng trang đầu **79 PDF** trong hai kho `paper/` và `literature/`. Tổng số PDF toàn cây là 99; 20 PDF khác chưa được kiểm nhận dạng tự động như nhóm 79 nói trên.

Đọc sâu các đường dẫn quyết định kết quả: chuẩn bị dữ liệu; dataset; target và loss; huấn luyện hai họ mô hình; checkpoint/resume; chọn cấu hình trên validation; đánh giá; giải mã và hậu xử lý EXP-016/023/024/025/026/027; bootstrap; mã benchmark cũ; lineage; CLI suy luận; nhật ký, bảng kết quả, tài liệu bàn giao và bản thảo. Kiểm kê và kiểm tra cú pháp toàn cây **không có nghĩa đã kiểm chứng từng dòng mã, từng ảnh, mọi thí nghiệm phụ hoặc mọi luận điểm trong 99 PDF**.

Kiểm chứng thực thi gồm:

- Chấm độc lập 30 bộ dự đoán của bảng chính bằng `pycocotools.COCOeval`, không dùng hàm AP nội bộ của dự án: AP, AP50, AP75, AP@300 và năm lớp diện tích.
- Đối chiếu toàn bộ **90.462 annotation** đã chuẩn bị với annotation trong Parquet nguồn; kiểm tra label/RLE và vùng không hợp lệ ở chín tile mẫu.
- Chạy lại kiểm tra giao nhau không gian EXP-015, chuyển đường dẫn đầu ra sang thư mục audit.
- Kiểm tra 23 hồ sơ lựa chọn validation, checkpoint được trỏ tới và cấu hình test tương ứng khi tồn tại.
- Chạy 39 kiểm thử có sẵn; dựng phản ví dụ metric, RNG và các mệnh đề toán học; tính lại accuracy/kappa của EXP-019.

Không train lại, không chạy lại toàn bộ suy luận từ checkpoint, không tune và không sửa mã/dữ liệu/kết quả gốc. Vì vậy audit xác nhận **số từ mask đã lưu**, không chứng minh độc lập rằng mọi mask được sinh từ đúng checkpoint, đúng phiên bản mã và đúng quá trình train lịch sử. Hash được lập ở thời điểm audit giúp kiểm tra thay đổi về sau, không khôi phục provenance quá khứ.

## 3. Kết quả đã kiểm chứng

### 3.1. Bảng chính

**Hoàn tất 30/30 bộ dự đoán: 270 giá trị metric khớp bảng tới hai chữ số thập phân; không có detection sai ID ảnh/lớp, sai kích thước RLE hoặc score không hữu hạn theo các kiểm tra đã thực hiện.**

| Cấu hình | AP độc lập, mean ± SD | AP từng seed 42 / 43 / 44 |
|---|---:|---|
| C0 TreeFlowNet v1 targets | 24.81 ± 0.58 | 24.922071 / 24.188008 / 25.334754 |
| C1 v2 targets | 25.78 ± 0.16 | 25.864855 / 25.592283 / 25.869980 |
| C2 v2 + target margin 256 | 25.81 ± 0.58 | 26.474709 / 25.561669 / 25.396030 |
| C5 C2 without centroid head | 28.80 ± 0.11 | 28.811752 / 28.903069 / 28.683018 |
| TFN = C5 + Swin-T (baseline decoder) | 30.17 ± 0.23 | 30.428671 / 29.966016 / 30.127158 |
| TFN + saddle merge (EXP-016) | 32.58 ± 0.22 | 32.838705 / 32.449592 / 32.462991 |
| TFN + saddle + rescoring (EXP-023) | 33.75 ± 0.32 | 34.115908 / 33.555250 / 33.564091 |
| TFN + saddle + rescoring + SDT gate (EXP-025) | 34.22 ± 0.30 | 34.556618 / 34.095085 / 33.994525 |
| Mask R-CNN | 34.82 ± 0.67 | 34.293772 / 34.598031 / 35.570490 |
| Mask R-CNN + rescoring (EXP-024) | 34.80 ± 0.59 | 34.168997 / 34.920518 / 35.324404 |

SD dùng ddof=1; mean và SD ở bảng audit tính từ AP chưa làm tròn. EXP-023 có mean 33,745083… AP: làm tròn từ số đầy đủ là **33,75**, giống bảng hiện tại; sai khác làm tròn cần đọc theo từng cấu hình, không mặc định mọi dòng sai.

Kiểm tra cuối đối với 17902 tệp gốc: 0 thay đổi size/mtime/hash phát hiện; 447 tệp văn bản có hash được đối chiếu. 

Chi tiết từng seed, SHA-256 của tệp mask, metric chưa làm tròn và sai khác so với bảng nằm trong [recomputed_metrics.json](recomputed_metrics.json). Sai số do lấy trung bình sau khi làm tròn từng seed được phân biệt với sai metric.

Kết luận được phép: trên tập test và các mask hiện có, ba bước hậu xử lý cải thiện TFN Swin-T; cấu hình cuối vẫn có AP trung bình thấp hơn Mask R-CNN. Không được suy ra rằng TFN đã vượt baseline hoặc đạt SOTA. Tên TFN trong bảng này là họ mô hình của dự án, không tự động đồng nhất với một công trình có cùng tên ngoài dự án.

Khoảng tin cậy được dự án lưu cho chênh lệch Mask R-CNN chấm lại − TFN cuối là **0,589 AP [0,155; 1,025]**. Audit kiểm tra công thức và tính AP của bootstrap bằng phản ví dụ/toy, nhưng **không tái sinh toàn bộ 2.000 bootstrap**. Cần đọc giới hạn suy luận ở F08, thay vì coi CI này là chứng minh xác nhận sau lựa chọn phương pháp.

### 3.2. Dữ liệu, split và nhãn

| Split | Tile | Annotation | Raster |
|---|---:|---:|---|
| Train | 148 | 53.289 | Z1: 66; Z2: 82 |
| Validation | 139 | 15.155 | Z1 |
| Test | 168 | 22.018 | Z1: 30; Z3: 138 |

Tất cả trường `segmentation`, `area`, `bbox`, `iscrowd` của toàn bộ annotation trong COCO chuẩn bị khớp nguồn Parquet, không thiếu tile, không trùng tên tile nguồn trong split. Nguồn: [source_annotation_checks.json](source_annotation_checks.json), script [verify_source_annotations.py](verify_source_annotations.py). Các ID ảnh/annotation duy nhất trong từng split; toàn bộ `iscrowd=0`.

Chín tile mẫu tái dựng label đúng thứ tự ưu tiên annotation: label trùng hoàn toàn, RGB ngoài `valid` bằng 0, không thấy label ngoài `valid`. Nhãn huấn luyện giải quyết vùng chồng bằng ưu tiên mask ghi trước; nhãn đánh giá giữ RLE gốc. Đây là khác biệt biểu diễn cần mô tả, không phải bằng chứng sửa GT đánh giá.

Diện tích `area` trong nguồn là số thực, còn đếm pixel RLE là số nguyên. Có rất nhiều chênh lệch vượt 0,1 pixel, nhưng sai khác tương đối trung vị chỉ khoảng **0,051% / 0,059% / 0,071%** ở train/validation/test. Giá trị được sao chép đúng nguồn, không có căn cứ quy lỗi sửa diện tích cho pipeline. Có ngoại lệ lớn hơn; lớp kích thước cần giữ cùng quy ước với benchmark nguồn. Không thay `area` bằng đếm pixel rồi so trực tiếp với số đã công bố mà không khai báo thay đổi giao thức. Xem [area_source_checks.json](area_source_checks.json).

Chạy lại EXP-015 khớp kết quả đã lưu: ở Z1, train–validation **14.094 pixel**, train–test **12.966 pixel = 11,669 m²**, validation–test **11.895 pixel**. Có **221 tán test chạm vùng giao nhau của các split**, nhưng chỉ số này không đồng nghĩa 221 tán bị nhìn thấy toàn bộ trong train. Z2/Z3 không có giao nhau giữa split. Khoảng cách train–test gần nhất ở Z1 bằng 0. Phần giao rất nhỏ so với diện tích test, nên không được đánh đồng với rò rỉ toàn vùng Z3 của checkpoint cũ. Cũng không nên viết “không một pixel nào trùng”.

Chạy lại này dùng logic hiện có của EXP-015, là tái lập kết quả chứ không phải một thuật toán kiểm không gian độc lập thứ hai. Các chi tiết bổ sung trong lời diễn giải như “bào mòn 1 px hết giao nhau”, số mảnh, số tán chạm train và phân bố phần trăm giao **chưa được audit tái tính** bằng script đó. Xem [exp015_split_leak_check.json](exp015_split_leak_check.json).

### 3.3. Mã và kiểm thử

286/286 Python parse được. 1.376 JSON đọc được bằng parser Python; một tệp chứa `NaN` không thuộc JSON chuẩn. **39 kiểm thử qua, một cảnh báo**, khoảng 48 giây; [pytest_cs2.log](pytest_cs2.log). Đây là bằng chứng các chức năng được test hoạt động, không phải bảo đảm toàn bộ nghiên cứu đúng.

Trong 23 hồ sơ `*_val_selection.json`, checkpoint được chọn đều tồn tại và cấu hình chọn có AP cao nhất **trong tập thử được lưu**. 21 hồ sơ có tệp test tương ứng; hai backbone thăm dò không có test, phù hợp với việc dừng ở validation. Với test tồn tại, các trường cấu hình đã đối chiếu khớp lựa chọn. Không suy ra rằng không có thử nghiệm khác ngoài hồ sơ, vì không có registry bất biến.

## 4. Phát hiện và ảnh hưởng

Mức **nghiêm trọng**: không được dùng kết quả/tuyên bố đó khi công bố. **Cao**: thay đổi tính hợp lệ hoặc khả năng tái lập của kết luận chính. **Vừa**: cần thu hẹp diễn giải hoặc hoàn thiện công cụ. **Thấp**: sửa biểu diễn và metadata. Mỗi phát hiện nêu rõ lịch sử hay hiện tại.

### F01 — Nghiêm trọng, lịch sử: checkpoint đã thấy test

`lineage.py:5–10` ghi V1 train với Quebec Z1+Z2+Z3 và BAMFORESTS; V2/V3/HR thừa kế V1; SimMIM dùng ảnh Z3. Các điểm Quebec Z3 và tuyên bố zero-shot BAM từ lineage đó không phải đánh giá trên miền chưa thấy. Mã legacy và tài liệu bàn giao xác nhận đường dữ liệu cũ. Không cần train lại để nhận ra giao thức này sai.

Các tài liệu hiện tại đã rút lại tuần 1, `large_scale_v2`, few-shot và số liên quan. Đây là hành động sửa hồ sơ tích cực. Nhánh clean khởi tạo ImageNet trong mã và không mặc định warm-start V1; **audit không phát hiện bằng chứng trực tiếp rằng các run clean đã dùng V1**. Guard theo chuỗi tên đường dẫn trong `lineage.py` vẫn không thay được hash và lineage manifest; đổi tên tệp có thể vượt guard.

**Xử lý:** đánh dấu archive và invalid ngay trong từng JSON/figure/bản thảo cũ; liên kết thông báo rút lại; không xóa chúng. Ghi hash trọng số khởi tạo và provenance vào manifest của mỗi run mới.

### F02 — Nghiêm trọng, lịch sử: F1 bị gọi là AP

`methods/tree_flow/hamiltonian_watershed.py:222` đặt `ap50 = 2*precision*recall/(precision+recall)`. `benchmark_all_paradigms.py` dùng các số này dưới nhãn mAP. F1 tại một ngưỡng không phải diện tích dưới đường precision–recall.

Phản ví dụ độc lập: một GT, một TP có điểm 0,9 và một FP có điểm 0,1. Hàm cũ trả **“AP50” 66,67**, đúng bằng F1; COCO AP50 thực sự **100**. [probe_results.json](probe_results.json) lưu phép kiểm. Lỗi không hiện diện trong hàm COCO AP của nhánh clean đã chấm độc lập.

**Xử lý:** đổi tên metric cũ thành F1, rút lại mọi so sánh mAP dựa vào nó. Nếu cần AP lịch sử, tính lại từ mask và score gốc; không quy đổi từ một giá trị F1.

### F03 — Nghiêm trọng, lịch sử: số baseline cố định và bản thảo không khớp đầu ra

`evaluation/benchmark_all_methods.py:191–250` ghi cố định các bảng StarDist, CrownTransformerSAM và TreeFlowNet, gồm AP50 **14,20 / 29,38 / 19,85**, cùng latency/VRAM/PQ khác. Chỉ nhánh CanopyWatershedNet ở script được tính tại runtime. Do đó đầu ra không phải một benchmark đồng bộ vừa chạy cho tất cả baseline.

`unified_benchmark_results.json` có CanopyWatershedNet AP50 khoảng **2,56**, trong khi `docs/paper_draft.md:139` ghi **18,45**, AP75 3,90 và latency 129,2 ms. Không thấy phép tính trong script này giải thích bảng bản thảo. `evaluation/plot_comprehensive_sota_and_nextgen.py:42` dùng mảng AP50 cố định; script còn dựng một đường SimMIM giả định nếu không có log.

**Bằng chứng chứng minh được:** số hằng và sự không khớp nguồn. **Điều không chứng minh được:** mọi số hằng đều bịa ở mọi nơi, hoặc có ý đồ gian lận. Nếu số lấy từ một run khác, cần chỉ rõ run/metric/split, không trình bày như benchmark cùng giao thức.

`docs/EXECUTIVE_SUMMARY.md:51` còn khẳng định ExG ≥ 15 chứng minh 94,1% phát hiện ngoài nhãn là cây thật. Chỉ số màu xanh không phân biệt cây, bụi và các thực vật khác, nên không xác nhận nhãn cây độc lập và không đủ để “hiệu chỉnh precision”.

**Xử lý:** ngừng dùng bản thảo/slide lịch sử; gắn mỗi ô bảng với prediction hash và evaluator; figure không có log phải báo thiếu dữ liệu thay vì dựng đường thay thế.

### F04 — Cao, hiện tại: test tham gia phát triển phương pháp

Nhật ký sử dụng phân tích test trong EXP-003/006/007/008/012/013 để hình thành các giả thuyết và thiết kế sửa decoder, scorer, gate ở EXP-016/023/025. H-004 viện dẫn lỗi nhỏ/lớn trên test để đề xuất tiêu chí đèo. EXP-005 còn mô phỏng gần phân bố kích thước của CanopyRS **test**. `HOAN_THANH_DU_AN/13` thừa nhận chọn tuần tự sau khi thấy số test.

Chọn tham số cuối trên validation là điểm tốt, nhưng **“mỗi cấu hình test một lần” không làm toàn bộ test trở lại độc lập**: tập test đã cung cấp thông tin để chọn họ sửa lỗi. Đây là thích nghi với test qua quá trình nghiên cứu, khác với đưa nhãn test trực tiếp vào loss.

Không thấy nhãn test dùng để fit HGB trong mã EXP-023/025: `y` test được để `NaN`, fit dùng `Xv,yv`. Vì vậy cần diễn đạt đúng loại hạn chế, không gán sai thành “train trực tiếp trên test”.

**Xử lý:** gọi kết quả hiện tại là phát triển/thăm dò trên benchmark đã sử dụng nhiều lần; khai báo đầy đủ lịch sử. Chốt phương pháp và giao thức trước, xác nhận trên holdout chưa xem hoặc dataset ngoài miền chưa dùng để sửa phương pháp. Nếu tiếp tục dùng cùng test, không gắn ý nghĩa xác nhận cho CI/p-value sau lựa chọn.

### F05 — Cao, hiện tại: RNG làm giảm đa dạng crop và augmentation

`clean_benchmark/dataset.py:46` và `train_maskrcnn.py:45` tạo lại RNG bằng `(torch.initial_seed()+i) mod 2**32` mỗi lần lấy item. Cả trainer dùng `persistent_workers=True`. Seed nền của worker không tự đổi qua epoch; với cùng item và cùng worker, crop và augmentation lặp lại.

Thử độc lập hai epoch, hai worker, không shuffle: **4/4 sample lặp lại y hệt ở cả hai dataset**. Training thực dùng shuffle, nên item có thể chuyển worker: không được khẳng định mọi crop của training thực luôn giống ở mọi epoch. Tuy nhiên, với tám worker và cùng chỉ số item, có tối đa tám chuỗi crop/augmentation theo worker trong một lần chạy liên tục, thay vì lấy mới mỗi epoch như mô tả. Cả hai họ cùng chịu lỗi, nên không tự động là lợi thế gian lận của một họ; lỗi vẫn ảnh hưởng độ bao phủ dữ liệu và độ tin cậy khi so công thức train.

**Xử lý:** dùng RNG tiến triển trong worker hoặc seed có epoch/global sample counter với thiết kế truyền epoch vào persistent worker đúng cách. Test cần kiểm tra khác crop qua epoch, tái lập cùng seed và hành vi resume. Sau sửa, chạy lại các run quyết định kết luận; không ghi đè checkpoint cũ.

### F06 — Cao, hiện tại: provenance chưa đủ để tái lập lịch sử

HEAD repo lúc audit: `0ad872f006da576b17443ff2d832b91e4c33335b`. Có 343 tệp dự án được Git theo dõi, **không có tệp `clean_benchmark` nào được track**. Nhiều thẻ nói “chưa commit”. Có args, history, seed và checkpoint, nhưng thiếu liên kết bất biến giữa mã thực sự chạy, dữ liệu, khởi tạo, predictions và kết quả.

Ngày ghi trong Markdown và mtime không chứng minh độc lập rằng preregistration đã có trước kết quả. EXP-002 ghi bù và EXP-006 thăm dò được ghi nhận minh bạch; không nên chuyển chúng thành xác nhận đã đăng ký trước. Audit không kết luận các ngày bị làm giả.

Tài liệu dọn đĩa 05/10 cho biết xóa **162 checkpoint không được chọn**, giữ 22 checkpoint được chọn. Việc dọn này đã được ghi nhận; không đồng nghĩa xóa kết quả âm. Nó làm mất khả năng chọn lại/resume các ứng viên cũ; chỉ còn AP của chúng trong hồ sơ. `checkpointing.py:4–8` cũng nói worker có seed mới khi resume: phục hồi optimizer/scheduler không đồng nghĩa phục hồi bitwise toàn bộ trajectory.

**Xử lý:** đóng băng snapshot, run registry append-only với hash mã/inputs/checkpoints/outputs, dependency lock và trạng thái resume. Preregistration tương lai cần mốc có thể kiểm chứng. Không tự bổ sung timestamp “trước chạy” cho thí nghiệm đã kết thúc.

### F07 — Cao, hiện tại: chưa cô lập được hiệu ứng phương pháp khi so baseline

TFN cuối dùng Swin-T; Mask R-CNN dùng R50. TFN R50 C5 với decoder gốc chỉ **28,80 AP**. Không thể lấy kết quả Swin-T + hậu xử lý để quy toàn bộ chênh lệch cho biểu diễn trường so với Mask R-CNN.

Hồ sơ selection thực lưu **18–19 lượt** cho phần lớn TFN, **8 lượt** cho mỗi Mask R-CNN, chưa kể lưới decoder/gate/scorer ở thí nghiệm sau. Các optimizer khác nhau là lựa chọn recipe có thể hợp lý; vấn đề là chưa đo và kiểm soát tổng ngân sách lựa chọn, backbone và chất lượng recipe. Số epoch/crop giống nhau không đủ chứng minh công bằng.

HGB hậu xử lý được học bằng nhãn validation. Như vậy cấu hình cuối sử dụng giám sát train **và** validation, dù không train lại mạng chính. Năm fold theo tile chống leakage trực tiếp trong việc fit scorer, nhưng checkpoint/decoder đã được chọn trên toàn validation trước đó: không phải nested CV cho toàn pipeline. EXP-024 áp scorer cho MRCNN giúp cân bằng một phần quyền sử dụng nhãn, không loại hết các khác biệt trên.

**Xử lý:** so cùng backbone hoặc ít nhất đo riêng tác động backbone; dành calibration split hoặc nested selection; báo train/calibration/test rõ ràng; ghi số trial và cho baseline ngân sách tương đương. Với EXP-025, gate và scorer được refit cùng lúc: +0,47 là tác động **gói can thiệp**, chưa phải hiệu ứng thuần của gate nếu thiếu đối chứng giữ scorer cố định.

### F08 — Cao đối với suy luận, hiện tại: CI bootstrap có phạm vi hẹp hơn lời kết luận

`bootstrap_gaps.py` dùng cùng sample tile cho các phương pháp và seed; tái tích lũy precision đúng cấu trúc COCO; phản ví dụ/toy và duplicate-tile check khớp COCO. Không phát hiện lỗi công thức AP trong đường bootstrap đã kiểm.

Tuy nhiên, lấy tile IID có thể đánh giá thấp bất định nếu các tile cùng orthomosaic tương quan không gian. Cùng ngày bay và test Z1 sát train là yếu tố thực tế. Mean của ba seed được bootstrap trên tile nhưng **không resample seed/lần train**: CI có điều kiện trên ba model đã train, không bao trùm biến thiên của thuật toán training. Seed 42 cho MRCNN chấm lại − TFN cuối có hiệu số **−0,388**, trong khi seed 43/44 lần lượt **+0,825 / +1,330**.

Thêm lựa chọn thích nghi F04 khiến nominal CI không phải đảm bảo hậu lựa chọn. Câu “thua có ý nghĩa” trong tài liệu nên kèm phạm vi này; không suy ra kiểm định xác nhận trên quần thể site/rừng.

**Xử lý:** báo CI theo tile có điều kiện, SD giữa seed; thêm bootstrap block không gian/site và phân tích độ nhạy; nếu mục tiêu suy luận về training cần thiết kế nhiều run/seed phù hợp. Xác nhận phương pháp trên holdout độc lập trước khi đưa kết luận thống kê mạnh.

### F09 — Cao với tổng quan và tính mới: ít nhất tám PDF bị đặt tên sai bài

Nhận dạng từ trang đầu và đối chiếu arXiv xác nhận:

| Tệp trong `paper/` | Nội dung thực tế | Nguồn chính thức |
|---|---|---|
| `Cellpose_NatureMethods_2006.10636.pdf` | Space-borne quantum memories for global quantum communication | [arXiv 2006.10636](https://arxiv.org/abs/2006.10636) |
| `2111.02534_Omnipose_NatureMethods.pdf` | Tensor Product Scheme for Quantum Mechanical Three-Body Systems | [arXiv 2111.02534](https://arxiv.org/abs/2111.02534) |
| `Topological_Loss_Persistence_1906.00684.pdf` | DANE: Domain Adaptive Network Embedding | [arXiv 1906.00684](https://arxiv.org/abs/1906.00684) |
| `Cellpose3_2408.00874.pdf` | Medical SAM 2 | [arXiv 2408.00874](https://arxiv.org/abs/2408.00874) |
| `2404.05696_LoftUp_CVPR2024.pdf` | BOLD v4, hệ thống dữ liệu mã vạch DNA | [arXiv 2404.05696](https://arxiv.org/abs/2404.05696) |
| `2307.05051_BoundaryDoU_Loss.pdf` | Microstructure of a spark-plasma-sintered Fe2VAl-type Heusler alloy | [arXiv 2307.05051](https://arxiv.org/abs/2307.05051) |
| `2503.04561_SAMRefiner_ICLR2025.pdf` | Mordell-Weil rank và 2-Selmer group của elliptic curves | [arXiv 2503.04561](https://arxiv.org/abs/2503.04561) |
| `1812.06453_HoVerNet_MedIA.pdf` | Gravitational resonances in mimetic thick branes | [arXiv 1812.06453](https://arxiv.org/abs/1812.06453) |

Tài liệu mới đã biết ba trường hợp đầu; audit phát hiện thêm năm trường hợp. `graham2019_hovernet.pdf` khác trong kho chứa HoVer-Net đúng: lỗi một bản không đồng nghĩa kho thiếu mọi nguồn đúng. Các tài liệu mới còn viện dẫn full text/mã chính thức Omnipose riêng, nên không được tự suy rằng toàn bộ diễn giải Omnipose lấy từ PDF sai.

Những PDF này **không thể làm nguồn cho công trình ghi trong tên**. Xem [pdf_identity.json](pdf_identity.json) với trang đầu của 79 tệp đã kiểm.

Tổng quan hiện tại đã thừa nhận flow/SDT gần Omnipose và Deep Watershed, chưa tìm kiếm hệ thống để khẳng định tính mới. EXP-027 mới có kết quả phase 1 của seed **42 và 44**, chưa có seed 43, selection tổng hợp hoặc kết quả rescore tương ứng trong snapshot audit. Không thể kết luận đã hoàn tất đối chứng ba seed. Dù hoàn tất, chạy decoder kiểu Omnipose trên field stride-4 của TFN chỉ so **decoder trên field này**, không so cả mô hình Omnipose đã train đúng target/resolution của nó.

**Xử lý:** tạo thư mục bibliographic manifest DOI/arXiv/hash/title; cách ly các bản sai nhưng giữ lịch sử; sửa trích dẫn trước công bố. Hoàn tất đối chứng decoder và đối chiếu công trình đúng; giới hạn đóng góp hiện tại ở phân tích và module có bằng chứng, không gọi gradient-distance flow là phát minh mới.

### F10 — Vừa/Cao, hiện tại: mệnh đề toán học sai hoặc thiếu giả thiết

`HOAN_THANH_DU_AN/16_MO_HINH_TOAN_HINH_THUC.md` có nhiều định nghĩa hữu ích, nhưng cần bỏ nhãn định lý cho các khẳng định chưa đúng:

- **Mục 5:** xếp theo `max IoU` không bảo đảm AP bằng recall. Hai mask trùng cùng GT có IoU 1 và đứng trước mask của GT khác tạo FP duplicate. Phản ví dụ độc lập đạt recall **1**, mọi `maxIoU=1`, nhưng COCO AP **83,498%**. [math_environment_checks.json](math_environment_checks.json). Oracle này đo một heuristic chấm điểm có dùng GT, không tự động là nghiệm tối ưu AP hoặc trần đạt được duy nhất.
- **Mục 8, dòng 152:** `distance(center_a,center_b) ≥ r_a+r_b` với `r=sqrt(area/pi)` chỉ phù hợp khi giả sử các đối tượng là đĩa không giao nhau, không đúng cho tán tùy ý. Hai hình chữ nhật rời nhau rộng 1, cao 100, tâm cách 1,1 có tổng bán kính tương đương **11,284**. Suy luận “gần như chắc cùng tán” không được coi là định lý cho crown không lồi.
- **Mục 7:** nguyên lý cực đại cho nghiệm Green Dirichlet trên miền liên thông không tự bảo đảm tính chất tương tự sau làm trơn gradient, pooling stride 4, nội suy và tích phân Euler hữu hạn. “Có đường đi không giảm” cũng cần quy tắc chống chu trình khi dùng các bước bằng giá trị. Phải tách tính chất nghiệm rời rạc gốc khỏi decoder triển khai.
- Các bảo đảm hội tụ từ gradient ascent liên tục hoặc thuật toán tối ưu lý tưởng không tự chuyển sang AdamW stochastic 30 epoch và Euler hữu hạn. Huber cần nêu rõ quy ước chuẩn hóa mới khẳng định hằng số trơn; projected dual với chặn `lambda_max` không tự bảo đảm ràng buộc khả thi hoặc diễn giải KKT chính xác tại giới hạn chặn.

**Xử lý:** giữ kết quả union-find/thành phần liên thông đúng trong điều kiện đã nêu, viết các phần còn lại thành giả thuyết hoặc trực giác có điều kiện; dùng phản ví dụ và kiểm chứng số phù hợp. Không dùng một thí nghiệm oracle để thay chứng minh.

### F11 — Vừa, hiện tại: oracle target không cô lập nguyên nhân mọi lỗi

EXP-026 đạt AP khoảng 92–96 khi decoder nhận trường GT và không thấy chia ở nhóm tán lớn đang xét. Điều này là bằng chứng tích cực về **khả năng giải mã trong điều kiện lý tưởng đã dựng**.

Nhưng `GT-v2-s4` được tạo bằng average pooling dense GT rồi nội suy, không chứng minh network thực có thể học, biểu diễn và đạt đúng field đó ở mọi biên/tán. Oracle dùng label giải quyết overlap, còn đánh giá dùng RLE gốc; min-area oracle cũng khác một số run dự đoán. Kết luận tài liệu “lỗi chia không nằm trong target; phần mất còn lại nằm ở chất lượng trường” quá tuyệt đối nếu diễn giải như loại trừ mọi tương tác target–loss–stride–network.

**Xử lý:** viết “trên oracle và nhóm tán đã xét, decoder không tạo lỗi chia; ưu tiên khảo sát chất lượng field dự đoán”. Muốn kết luận nhân quả cần giữ decoder và điều kiện chung, can thiệp target/loss/resolution và đo nhiều seed, cùng phân tích học được hay không.

### F12 — Vừa, hiện tại: EXP-019 không xác minh chất lượng annotation bằng người độc lập

Tính lại CSV cho **accuracy 0,775, kappa 0,71875**, khớp kết quả lưu. Nhưng 40 mẫu được cân bằng theo năm loại lỗi, ảnh hiển thị outline GT và đáp án được định nghĩa bằng chính thuật toán phân loại lỗi; người trả lời là agent và `_KEY/key.json` tồn tại trong cùng workspace.

Không chứng minh agent đã đọc key, song thiết kế này không tạo sự che giấu đáp án độc lập có thể kiểm chứng. Nó kiểm mức nhất quán đọc gallery với quy tắc lỗi trên một mẫu chọn theo tầng, không đo accuracy annotation trên quần thể, không chứng minh các tán nhỏ “thực sự là cây”. EXP-009 với ba người gán độc lập vẫn được tài liệu ghi là chờ.

**Xử lý:** đổi tên/mô tả EXP-019 đúng chức năng; hoàn tất đánh giá người độc lập có che GT/dự đoán và key tách khỏi người gán; báo inter-rater agreement và thiết kế lấy mẫu. Không đem tỷ lệ cân bằng này áp lên toàn dataset.

### F13 — Cao về bảo toàn hồ sơ: đã mất kết quả; guard hiện tại còn yếu

Nhật ký EXP-004, dòng **600–601**, ghi rõ file kết quả Boundary AP=0 với dải 3 px bị xóa và log bị ghi đè trước khi đổi sang định nghĩa d=2% đường chéo ảnh. Việc sửa một metric không phù hợp có thể hợp lý; xóa kết quả và thay log làm mất khả năng kiểm toán bản ban đầu. Tài liệu đã thừa nhận vi phạm, đây là điểm minh bạch nhưng không phục hồi được file.

Rủi ro còn trong mã hiện tại: `exp016_saddle_merge.py:176–179` ghi `*_test_*_dt.json` **trước** `if a.limit: return`, nên chạy thử có `--limit` trên test có thể ghi đè mask toàn bộ bằng một phần. EXP-023/024/025 ghi đầu ra test bằng mode `w`; guard “test một lần” ở selector không bảo vệ toàn bộ pipeline. Cache ở selector/feature dùng tên tệp, chưa xác minh config/checkpoint/code hash.

**Audit không chạy các đường ghi này và không khẳng định chúng đã ghi đè kết quả hiện có**, ngoài trường hợp nhật ký EXP-004 thừa nhận.

**Xử lý:** run ID và đường output riêng, không dùng tên full-run cho smoke test, chặn overwrite và ghi atomically; cache key gồm hash; lưu failure/negative cùng metadata; mọi correction là phiên bản mới kèm liên kết bản cũ.

### F14 — Vừa, hiện tại: CLI công khai không tái hiện phương pháp cuối

`predict.py:31–42` mặc định V3 và checkpoint legacy, chỉ cho v2/v3; gọi `decode_organic`, không triển khai đầy đủ clean Swin-T + saddle + scorer + gate cuối. README gốc và executive summary vẫn tập trung vào nhánh cũ. Người chỉ đọc điểm vào dự án dễ chạy một pipeline khác với pipeline đạt AP 34,22.

**Xử lý:** chỉ rõ điểm vào clean benchmark và phiên bản phương pháp; cung cấp CLI nhận bundle checkpoint/scorer/config/gsd đã đóng băng; kiểm thử tái hiện một tile và hash đầu ra. Giữ CLI legacy nhưng ghi trạng thái không dùng cho số chính.

### F15 — Vừa, setup: chưa có môi trường tái lập tự đủ

Python mặc định thất bại khi import torchvision với lỗi `operator torchvision::nms does not exist`. `cs2_venv` chạy được kiểm tra; phiên bản lúc audit: Python **3.11.5**, torch **2.4.0+cu121**, torchvision **0.19.0+cu121**, NumPy **1.26.4**, SciPy **1.17.1**, scikit-learn **1.9.0**, pycocotools **2.0.11**, pyarrow **24.0.0**. Đây là môi trường audit hiện tại, không chứng minh môi trường các run lịch sử giống hệt.

`cs2_venv` chưa có pytest; audit tận dụng pytest có sẵn bằng cách append đường site-packages sau thư viện môi trường, tắt tự nạp plugin và cache. Không cài hay nâng cấp dependency. Một số mã legacy còn đường conda hardcode ngoài máy hiện tại. Data nằm qua đường dẫn/symlink ngoài thư mục dự án, không đủ chỉ copy riêng `crown_segmentation_research` là tái lập.

**Xử lý:** requirements/lock tương thích và container/environment recipe; kiểm tra import trước train; manifest dữ liệu với revision/hash, đường dẫn cấu hình; một lệnh chạy từ checkout sạch. Không gán sự thất bại của môi trường mặc định thành bằng chứng kết quả clean sai.

### F16 — Thấp/Vừa, báo cáo: làm tròn, JSON và maxDets

`master_table.py` tổng hợp từ metric từng seed đã làm tròn hai chữ số: có thể làm mean cuối khác 0,01 AP so với tính từ số đầy đủ. Đây là sai số báo cáo nhỏ, không phải bằng chứng giả mạo. Dùng số đầy đủ tới bước trình bày cuối.

Ví dụ thực tế: mean C1 từ số đầy đủ là **25,78**, bảng cũ **25,77**; TFN Swin-T gốc **30,17** thay vì **30,18**; SD Mask R-CNN chấm lại **0,59** thay vì **0,58**. AP từng seed đều khớp tới hai chữ số. TFN cuối **34,22 ± 0,30** và Mask R-CNN **34,82 ± 0,67** không thay đổi khi sửa cách tổng hợp này.

`results/boundary_r2_validation.json` chứa NaN: Python đọc được, parser JSON chuẩn có thể từ chối. Dùng `null` kèm lý do thiếu kết quả.

Lập luận trong tài liệu “maxDets 400 không cắt vì nhiều nhất 387 GT” không đúng: cutoff áp cho **dự đoán**, không cho GT. TFN cuối seed 43 có một tile trên 400 dự đoán (max 406), seed 44 có bốn tile (max 441). Chấm maxDets 400 vẫn là một giao thức hợp lệ nếu công bố rõ, nhưng không được nói không cắt dự đoán. AP@300 chỉ điều chỉnh một yếu tố so sánh ngoài dự án.

## 5. Đối chiếu công trình ngoài dự án

[SelvaMask, arXiv 2602.02426v1](https://arxiv.org/html/2602.02426v1) trình bày giới hạn đầu ra tối đa 300 instance/tile và Table 4 có kết quả QuebecTrees **27,6 ± 0,3** cho cấu hình tương ứng. Cấu hình đó có fine-tuning trên miền mục tiêu, không phải một baseline zero-shot chưa thấy QuebecTrees. “Đầu ra tối đa 300” cũng cần đối chiếu chính xác evaluator; không dùng câu đó để suy mọi chi tiết COCO maxDets của họ.

Dataset [CanopyRS/QuebecTrees](https://huggingface.co/datasets/CanopyRS/QuebecTrees) được phát hành như bản preprocessed cho SelvaBox. Audit đối chiếu local Parquet với prepared COCO, chưa kiểm hash local shards với revision tải từ Hugging Face lúc các run diễn ra. GSD bản release local là 3 cm, khác con số độ phân giải gốc trong bảng mô tả dữ liệu của bài; cần kiểm toàn pipeline resampling chứ không kết luận sai chỉ từ hai số này.

Vì vậy **34,22 > 27,6 không đủ chứng minh vượt SelvaMask** khi chưa đồng nhất revision/split, input GSD/crop, preprocessing, pretrained data, prediction filtering, scoring và metric. Không có baseline SelvaMask tái chạy trong audit này. Nguồn COCO chính thức cho cách tích lũy AP: [cocoeval.py](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py). AP tùy chỉnh maxDets 400 cần đọc đúng precision tensor; clean evaluator làm điều này, kiểm chứng độc lập xác nhận số đã lưu.

## 6. Điều có thể bảo vệ và điều chưa thể bảo vệ

| Tuyên bố | Đánh giá |
|---|---|
| Mask dự đoán đã lưu cho bảng chính cho ra AP như báo cáo | Kiểm chứng trực tiếp bằng evaluator độc lập; xem mục 3 |
| Annotation chuẩn bị giữ nguyên nguồn Parquet | Kiểm chứng toàn bộ 90.462 annotation |
| TFN Swin-T cải thiện sau chuỗi hậu xử lý trên benchmark hiện có | Có bằng chứng số; phải ghi quá trình phát triển đã xem test |
| TFN cuối vượt Mask R-CNN hoặc đã đạt SOTA | Không được số hiện tại ủng hộ |
| Checkpoint lịch sử cho kết quả holdout/zero-shot hợp lệ | Không; lineage đã thấy test/miền đích |
| Mọi train crop được lấy ngẫu nhiên mới mỗi epoch | Không đúng với thiết kế RNG hiện tại |
| Test được giữ kín trong toàn quá trình phát triển | Không; có phân tích lỗi test trước thay đổi phương pháp |
| CI hiện tại chứng minh khác biệt tổng quát giữa phương pháp | Chưa; có điều kiện trên model/tile và sau lựa chọn |
| 92–96 AP oracle loại trừ mọi giới hạn target | Quá mức; chỉ chứng minh năng lực trong điều kiện oracle đã dựng |
| EXP-019 xác nhận GT độc lập bằng người | Không; kiểm gallery/quy tắc bằng agent |
| Flow/SDT hoặc xếp hạng mask bằng IoU là ý tưởng hoàn toàn mới | Không có căn cứ; tài liệu mới cũng đã thu hẹp |
| Những vấn đề này chứng minh tác giả cố ý gian lận | Không có đủ bằng chứng để kết luận ý đồ |

## 7. Thứ tự khắc phục trước khi công bố

1. **Chỉnh hồ sơ ngay:** đánh dấu rõ mọi kết quả bị rút lại và bản thảo/plot dùng số hằng; bảo toàn bản cũ; sửa PDF sai và các mệnh đề toán học; sửa câu “test một lần/sạch tuyệt đối” và các suy luận nhân quả tuyệt đối.
2. **Đóng băng trạng thái tái lập:** snapshot mã, config, môi trường, dữ liệu, init, checkpoint, scorer, predictions, evaluator bằng hash và run ID. Lưu đầy đủ negative/failure, chặn overwrite và cache cũ.
3. **Sửa RNG crop** rồi train lại bộ run tối thiểu quyết định câu chuyện khoa học. Giữ checkpoint/result cũ với nhãn phiên bản, không ghi đè. Kiểm resume theo phạm vi thống kê hay bitwise được công bố.
4. **Cân bằng thiết kế:** cùng backbone hoặc factorial đo hiệu ứng; cùng ngân sách tune; validation/calibration được tách hoặc nested selection; baseline đủ mạnh; ablation bỏ riêng từng module và phân biệt gate với refit scorer.
5. **Xác nhận độc lập:** giữ kín holdout mới hoặc miền ngoài chưa dùng để thiết kế, chốt protocol trước khi chạy. Báo phân tích không gian/block và bất định giữa seed cùng kết quả thăm dò cũ.
6. **Hoàn thiện bằng chứng đóng góp:** hoàn tất EXP-027, đối chiếu literature đúng, hoàn tất người gán EXP-009, benchmark thời gian/FLOPs nếu muốn tuyên bố hiệu quả, và CLI tái hiện đúng pipeline cuối.

Đây là khuyến nghị kỹ thuật và nghiên cứu; audit chỉ tạo báo cáo, chưa thực hiện các thay đổi ảnh hưởng kết quả hay train mới.

## 8. Hồ sơ kiểm chứng và giới hạn còn lại

| Tệp audit | Nội dung |
|---|---|
| [inventory.json](inventory.json) | Toàn bộ tệp, size/mtime, hash nguồn văn bản |
| [git_provenance.json](git_provenance.json) | HEAD và phạm vi Git track |
| [static_checks.json](static_checks.json), [json_validation.json](json_validation.json) | Parse Python/JSON toàn cây |
| [verify_saved_results.py](verify_saved_results.py), [recomputed_metrics.json](recomputed_metrics.json), [rescore.log](rescore.log) | Chấm AP độc lập 30 run |
| [verify_source_annotations.py](verify_source_annotations.py), [source_annotation_checks.json](source_annotation_checks.json) | So toàn bộ annotation với Parquet |
| [area_source_checks.json](area_source_checks.json) | Phân tích diện tích raster/metadata và mẫu nguồn |
| [probes.py](probes.py), [probe_results.json](probe_results.json) | Phản ví dụ F1/AP, crop lặp, dữ liệu mẫu, selection, EXP-019 |
| [exp015_split_leak_check.json](exp015_split_leak_check.json) | Chạy lại kiểm giao nhau không gian |
| [math_environment_checks.json](math_environment_checks.json) | Phản ví dụ toán và phiên bản môi trường |
| [TAI_LAP_KIEM_TOAN.md](TAI_LAP_KIEM_TOAN.md), [input_checkpoint_manifest.json](input_checkpoint_manifest.json) | Lệnh tái kiểm và hash 22 checkpoint được chọn + GT COCO |
| [pytest_cs2.log](pytest_cs2.log), [environment_freeze.txt](environment_freeze.txt) | Kiểm thử và package snapshot |
| [pdf_identity.json](pdf_identity.json) | Trang đầu 79 PDF |
| [audit_summary.json](audit_summary.json) | Tóm tắt machine-readable, bảng AP và kiểm tra bảo toàn originals |

Chưa tái train, chưa suy luận lại toàn test từ checkpoint, chưa tái chạy mọi experiment auxiliary hoặc 2.000 bootstrap, chưa xác nhận GT bằng người độc lập, chưa kiểm toàn văn/novelty toàn bộ tài liệu. Nhiều kiểm tra phụ xác minh tính nhất quán của hồ sơ, không thể thay thế provenance bất biến. Kết luận luôn giới hạn ở snapshot và các đường kiểm chứng nêu trên.
