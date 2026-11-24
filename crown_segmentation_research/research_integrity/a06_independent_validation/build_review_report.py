"""Assemble one self-contained independent review from executed local evidence."""
from pathlib import Path
import csv
import datetime
import hashlib
import json

HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/name).read_text())
def dump(name,obj):(HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

findings=[
('V01','CRITICAL','DEFECT_CONFIRMED','Sai repo root; bỏ qua MISSING và chấp nhận manifest rỗng','Không xác minh hash thực tế nhưng báo 18/18 PASS','preflight.py:23,87–122; generate_baseline_hashes.py:6','missing_baseline / empty_baseline; source_analysis.json','Sửa root, bắt buộc tập file và digest hợp lệ; thiếu file/hash phải FAIL'),
('V02','CRITICAL','DEFECT_CONFIRMED','--config chỉ được lưu, không bao giờ đọc','Config hỏng, dữ liệu cấm, capacity thiếu và đích ghi nguy hiểm đều qua gate','preflight.py:39–41,260–269','missing_config / malformed_config / forbidden_roles_capacity_paths','Parse schema, kiểm tra ngữ nghĩa/provenance và từ chối lỗi trước runtime'),
('V03','CRITICAL','MISSING_CONTROL','Calibration/crop/output checks là thông báo PASS cố định; không tích hợp trainer','Không có kiểm soát vận hành cho các invariant quan trọng','preflight.py:157–185; train_treeflow.py:86–125','CLI sandbox không có dữ liệu vẫn PASS; runtime_integration_references=[]','Gọi validator thật, gắn gate vào entrypoint; UNKNOWN/thiếu bằng chứng không được tính PASS'),
('V04','HIGH','MISSING_CONTROL','Hai test rỗng, nhiều fixture không kiểm tra production code','24/24 không chứng minh đã chống hồi quy pipeline','test_suite.py:509–556','AST hai hàm pass; capacity fixture chỉ 7>4','Thay bằng test âm/dương của API thật; thêm output/capacity/AP/boundary/stitching regression'),
('V05','HIGH','DEFECT_CONFIRMED','11/12 tên test trong ROOT_CAUSE_MATRIX không tồn tại trong suite','Traceability cho VERIFIED_FIXED không đáng tin','ROOT_CAUSE_MATRIX.csv','evidence_crosschecks.json','Dẫn test thực sự; kiểm tra liên kết tự động; hạ trạng thái chưa sửa'),
('V06','HIGH','DEFECT_CONFIRMED','5 SHA256 in trong báo cáo khác SHA256 tính từ file hiện tại','Bằng chứng xác minh hash không được artifact hỗ trợ','HARDENING_REPORT.md:125–130','5/5 report_hash_comparisons=false','Sinh hash từ dữ liệu thực và ghi thời điểm/phạm vi; không sửa báo cáo cũ'),
('V07','HIGH','DEFECT_CONFIRMED','Helper epoch RNG nằm trong test, không được loader thật sử dụng','Crop lặp qua epoch trong cấu hình persistent workers đang dùng','dataset.py:46; train_treeflow.py:120–121; train_maskrcnn.py:112–113','4/4 crop lặp với workers=0 hoặc persistent_workers=True','API epoch-aware cho loader, truyền epoch tới worker; regression qua DataLoader thật'),
('V08','HIGH','DEFECT_CONFIRMED','Mô tả legacy last-wins trái code first-wins; nhầm 63 unique-empty với 10 priority-empty','Đổi reference/tái dựng lịch sử sai','REFERENCE_SEMANTICS_V3.md:40–47; FIX_LOG.md:31; prepare_canopyrs.py:54–56','legacy_priority_replay đổi 9/25 pixel; data_summary.json','Tách legacy first-wins và phương án ownership mới; giữ cả 6 support; không đổi nhãn gốc'),
('V09','MEDIUM','DEFECT_CONFIRMED','Mẫu số 0 được trả 0, thiếu class vẫn tính balanced','Macro/fold statistics bị biến NA thành thất bại số học','test_suite.py:180–181,220–223','zero_denominator; no_different_pairs','Trả null/NA(reason), counts và common-support policy đã khóa'),
('V10','HIGH','REQUIRES_SCIENTIFIC_DECISION','A06 đổi E2E highest-score/top400 thành unique-cover; mọi metric bị ràng vào Stage2','Khác câu hỏi đo và khác score dependence so với protocol trước','METRIC_DEFINITIONS_V3.md:46–68; OUTPUT_CONTRACT_V3.md; execution plan:466','highest_score_overlap_E2E; hai quy tắc cho kết quả khác','Giữ metric lịch sử; nếu chọn metric mới phải đặt tên/version và trình bày khác biệt'),
('V11','HIGH','REQUIRES_SCIENTIFIC_DECISION','Buffer 30m, min-area .25m² và threshold được đưa vào như mặc định đã chốt','Thay quần thể đánh giá hoặc chính sách độc lập chưa có căn cứ','SPATIAL_SPLIT_POLICY_V3.md:38; OUTPUT_CONTRACT_V3.md:43–56','Đối chiếu tài liệu; không có phê chuẩn/ước lượng range được dẫn','Đánh dấu PROPOSED; cấu hình/version/waiver/ảnh hưởng claim rõ ràng'),
('V12','HIGH','SEMANTIC_AMBIGUITY','G1_merge đổi thành cut-and-merge; multicut trên atom bị coi có thể tách nội basin','Thay đối chứng và suy luận nguyên nhân thất bại G0 quá mức','FGQ_FAIR_COMPARISON_V3.md:43; execution plan:343,366','Đối chiếu search space atomic basin và protocol G1_merge','Giữ G0/G1_merge/G_repartition riêng; không quy toàn bộ EXP020/021 cho một nguyên nhân'),
('V13','HIGH','DEFECT_CONFIRMED','Preflight luôn ghi đè READINESS_REPORT.json cùng đường dẫn','Mất lịch sử lần kiểm tra; không có output destination safety','preflight.py:250–252','CLI chạy ở sandbox làm đổi bytes báo cáo trong 7/8 case','Dùng run directory mới, exclusive create, lineage và hash của config/input'),
('V14','HIGH','DEFECT_CONFIRMED','Lệnh next-step không tương ứng implementation hoặc config thật','Không có recipe Q/G executable như readiness tuyên bố','NEXT_EXPERIMENT_DECISION.md:36–55; train_treeflow.py:86–111; exp021_veto_merge.py:47–53','--help + AST options; Q YAML không tồn tại','Xóa trạng thái executable khỏi báo cáo mới; chỉ công bố lệnh đã kiểm parser/dry-run an toàn'),
('V15','HIGH','MISSING_CONTROL','Schema/docs chưa thành registry runtime, output adapters, access policy, CI gate','Không chống leakage hoặc thay output semantics khi code đổi','PROVENANCE_REGISTRY_SCHEMA.json; OUTPUT_CONTRACT_V3.md; production entrypoints','Không có integration reference trong clean_benchmark/methods/tests','Tái dùng registry A00; thêm API/gate/CI và Stage0→3 lineage tests; không tạo framework trùng'),
('V16','MEDIUM','DEFECT_CONFIRMED','Không từ chối NaN area/affine suy biến; overlap và buffer chưa tách invariant','Invalid input có thể nhận size-bin/area hoặc chứng nhận split sai','test_suite.py:32–65,235–249','NaN→ge9; det=0→0; overlapping boxes buffer0→valid','Kiểm finite/det/CRS/shape và overlap policy riêng trước tính metric'),
('V17','HIGH','NOT_VERIFIABLE','23 file A06 biến mất ngoài thao tác review sau khi kiểm tra lần đầu','Bản bàn giao hiện tại không còn executable; nguyên nhân/chủ thể chưa xác định','external_source_drift.json; reviewed_a06_snapshot/','Snapshot khôi phục đủ25file khớp hash ban đầu; không khôi phục đè thư mục nguồn','Điều phối writer; xác định phiên bản đang được sửa; chỉ review revision có hash cố định'),
]

requirements=[
('R01','Historical preservation','generate_baseline_hashes.generate / PreflightChecker.check_baseline_integrity','V01,V06,V13','DEFECT_CONFIRMED'),
('R02','All-annotation integrity','Existing A00 audit.py; A06 does not call it','source_annotation_metadata_scan.json; V03','MISSING_CONTROL'),
('R03','Six support definitions','test_suite.derive_support_sets; prepare_canopyrs.prepare_split','support helper tests; V08','SEMANTIC_AMBIGUITY'),
('R04','Canonical area','test_suite.compute_canonical_area / assign_size_bin','test_canonical_area_and_determinant; V16','MISSING_CONTROL'),
('R05','Provenance registry','PROVENANCE_REGISTRY_SCHEMA.json; existing A00 registry','V15','MISSING_CONTROL'),
('R06','Spatial overlap/buffer','PreflightChecker.check_spatial_splits / validate_spatial_buffer','V03,V11,V16','MISSING_CONTROL'),
('R07','Dataset roles/test access','PreflightChecker.check_calibration_standards','forbidden_roles_capacity_paths; V02,V03','MISSING_CONTROL'),
('R08','Full-crown sampler','CanopyCropDataset / get_crop_rng helper','actual_loader_rng_probe.json; V07','DEFECT_CONFIRMED'),
('R09','Query capacity','test_fixture_18_crops_with_more_gt_than_queries','V02,V04','MISSING_CONTROL'),
('R10','Stage0–3 output/lineage','PreflightChecker.check_output_contract','V03,V15','MISSING_CONTROL'),
('R11','One-to-one matching','test_suite.cardinality_first_match','perfect/duplicate/merged/split helper fixtures','VERIFIED_CORRECT'),
('R12','Grouping NA/assignment policy','evaluate_structural_and_e2e_grouping','metric_probes.json; V09,V10','DEFECT_CONFIRMED'),
('R13','AP/boundary/count/C0C1 metrics','Definitions only in A06; historical code exists','V04,V15','MISSING_CONTROL'),
('R14','Calibration/score provenance','PreflightChecker.check_calibration_standards','forbidden_roles_capacity_paths; V03','MISSING_CONTROL'),
('R15','FGQ fairness executable recipe','FGQ_FAIR_COMPARISON_V3 / NEXT_EXPERIMENT_DECISION','advertised_commands_check.json; V12,V14','DEFECT_CONFIRMED'),
('R16','Fail-closed config/manifest/output','PreflightChecker.run_all / main','preflight_probes.json; V01–V03,V13','DEFECT_CONFIRMED'),
('R17','Regression coverage/CI','test_suite.run_tests; ROOT_CAUSE_MATRIX.csv','V04,V05,V15','MISSING_CONTROL'),
('R18','Independent confirmatory reference','Historical role manifest; hardcoded confirmatory block','confirmatory_no_inputs exit1; no independent holdout','NOT_VERIFIABLE'),
]

def main():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    p=read('preflight_probes.json');s=read('source_analysis.json');e=read('evidence_crosschecks.json')
    for case in p:
        case['expected_exit_code']=1
        case['specification_check_passed']=case['exit_code']!=0
    dump('adversarial_expectations.json',p)
    with (HERE/'INDEPENDENT_FINDINGS.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','severity','classification','root_cause','scientific_impact','source','verification','required_fix'])
        w.writerows(findings)
    with (HERE/'REQUIREMENT_IMPLEMENTATION_TEST_MAP.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['requirement_id','scientific_requirement','implementation','test_or_evidence','status']);w.writerows(requirements)
    verdict={'review_version':'A06-independent-validation-v1','generated_UTC':now,'target':'Original A06 delivery frozen before external file disappearance; byte-exact snapshot retained',
             'overall_status':'FAIL','DATA_INTEGRITY_READY':False,'EXPLORATORY_FGQ_READY':False,'CONFIRMATORY_BENCHMARK_READY':False,
             'CONFIRMATORY_NOT_READY':True,'EXTERNAL_HOLDOUT_REQUIRED':True,
             'readiness_interpretation':'False means delivered preventive controls have not met acceptance; it does not establish that the original imagery or all annotations are corrupt.',
             'original_tests':read('original_tests_reexecuted.json')['summary'],
             'original_test_empty_bodies':2,'unresolved_findings':len(findings),'adversarial_CLI_cases':8,'unexpected_acceptances':6,
             'critical_blockers':['V01','V02','V03'],'external_worktree_drift':'V17',
             'review_scope_exclusions':['No full training','No oracle rerun','No historical test model-performance evaluation','No source annotation changes','No A06 repair or scientific policy adoption'],
             'historical_hash_check':read('historical_source_check.json'),'original_source_drift_after_first_review':read('external_source_drift.json')}
    dump('INDEPENDENT_READINESS_REPORT.json',verdict)
    header='''# Kiểm định độc lập bản bàn giao A06 — TreeFlowNet

**Ngày:** 15/10/2026. **Phiên bản:** A06-independent-validation-v1. **Phán quyết:** KHÔNG ĐẠT tiêu chuẩn hardening; chưa dùng PASS của A06 để bắt đầu benchmark F–G–Q.

Tài liệu này là một file tổng hợp dành cho người nghiên cứu và agent sửa tiếp. Phần I–VII chứa kết luận, nguyên nhân, bằng chứng và tiêu chí sửa. Phụ lục nhúng đầy đủ kết quả JSON, log test và source của chương trình kiểm định; không cần mở CSV/JSON bên ngoài để hiểu các phát hiện chính.

## I. Executive verdict

Đã đọc trực tiếp implementation, test, hợp đồng và báo cáo A06; đối chiếu loader/trainer/prepare/evaluator liên quan và artifact A00–A05. Đã chạy lại suite A06, chạy 8 case preflight trong sandbox không có dữ liệu nghiên cứu, 9 probe metric/semantics, kiểm tra loader thật bằng dữ liệu tổng hợp trong 3 cấu hình worker, chạy hai CLI chỉ với `--help`, quét metadata/RLE nén của cả 90.462 annotation và đối chiếu hash 1.026 file đã đăng ký trước đây.

**Kết luận:** A06 có phần tài liệu và helper toán học hữu ích, nhưng chưa phải hệ thống preventive controls đã tích hợp. Config không được đọc; hash thiếu được bỏ qua; nhiều check tự trả PASS; RNG thật chưa sửa; recipe thí nghiệm không chạy được như mô tả. `24/24 PASS` là kết quả có thật, nhưng hai test rỗng và các fixture yếu khiến kết quả này không hỗ trợ claim mọi lỗi `VERIFIED_FIXED`.

**Đã sửa trong phiên review này:** chỉ chương trình kiểm định và cách lưu bằng chứng của review. **Chưa sửa A06 hoặc production pipeline.** Đây là kiểm định bản agent đã bàn giao; không sửa âm thầm implementation đang được đánh giá, không phê chuẩn buffer/ownership/threshold mới.

### Sự kiện thay đổi ngoài phiên kiểm tra

Sau khi hoàn tất lượt kiểm định đầu, 23/25 file A06 biến mất ở thư mục gốc. Không có lệnh review nào xóa/di chuyển chúng. Đã hỏi người dùng về writer khác; nguyên nhân vẫn UNKNOWN nếu chưa có trả lời. Không quy sự kiện này cho agent A06 hoặc kết luận cố ý. Bản sao `reviewed_a06_snapshot/` chứa đủ25 file, mỗi file khớp SHA256 chụp ban đầu. Các kiểm tra đã được chạy lại trên snapshot; không phục hồi đè thư mục nguồn.

`review_preservation.json` ghi trạng thái **trước** sự kiện ngoài phiên: 29 file A06/production theo dõi không đổi. `external_source_drift.json` ghi trạng thái **sau** sự kiện: 23 file A06 thiếu. `snapshot_replay_preservation.json` xác nhận snapshot và bốn source production không đổi qua lần replay. Không dùng kết quả trước sự kiện để tuyên bố worktree hiện tại vẫn nguyên vẹn.

### Các phần được xác nhận đúng trong phạm vi kiểm tra

- Suite A06 có24 test, chạy đạt24/24, 0failure, 0error, 0skip. Python3.11.5, torch2.4.0+cu121, NumPy1.26.4 khớp các phiên bản lõi ghi trong báo cáo.
- Công thức diện tích theo định thức affine đúng trên fixture hợp lệ có pixel vuông/không vuông; các helper matching xử lý được perfect/empty/duplicate/split/merge trong những fixture đã chạy. Không suy rộng thành đã kiểm toàn bộ evaluator production.
- Helper structural grouping bỏ qua score tại lúc đo và abstain khi overlap; invariant score-only trong fixture tương ứng chạy đạt.
- Confirmatory CLI trả exit1 trên trường hợp đã thử; lịch sử Z1/test đã dùng trong phát triển không được đổi thành unseen holdout.
- 1.026 file có hash từ đợt trước khớp tại lượt kiểm tra này. Đây là kiểm tra đúng phạm vi manifest, không chứng nhận toàn bộ file chưa từng đăng ký.

### Các phần chưa xác minh lại từ nguồn nguyên thủy

Không đọc lại mọi byte ảnh/valid-mask trên455 tile; không tái sinh prepared data từ raw parquet; không tính lại mọi quan hệ ground support hoặc polygon–valid intersection; không chạy oracle, training hay COCO model-performance evaluation. Các kết quả spatial/unique/priority/OOF tiếp tục được ghi là artifact audit cũ đã đọc và đối chiếu, không phải lượt tái lập pixel-level mới. Survey/acquisition độc lập, reference sinh học và external holdout vẫn chưa được xác lập.

## II. Findings table

Các trạng thái dưới đây mô tả bản bàn giao được chụp. Không có hàng nào được nâng thành FIXED chỉ vì có tài liệu đề xuất.

| ID | Severity | Phân loại | Nguyên nhân / tác động | Source / test | Việc cần sửa |
|---|---|---|---|---|---|
'''
    lines=[header]
    for fid,severity,status,cause,impact,source,test,fix in findings:
        lines.append(f'| {fid} | {severity} | {status} | {cause}. {impact}. | `{source}`; {test} | {fix} |\n')
    lines.append('''
## III. Bằng chứng hành vi, nguyên nhân và tiêu chí sửa

### 1. Hash integrity đang báo thành công khi chưa xác minh file nào

`preflight.py:23` và `generate_baseline_hashes.py:6` dùng `Path(__file__).resolve().parents[4]`. Với vị trí file ban đầu, nó là `/home/chung/RS`; repo thật là `/home/chung/RS/Image Segmentation` (parents[3]). Cả18 đường dẫn trong `BASELINE_HASHES.json` ghi `MISSING`, mặc dù18 file tương ứng tồn tại ở repo thật.

Trong `check_baseline_integrity`, giá trị `MISSING` bị `continue`. Manifest rỗng hoặc18 giá trị MISSING đều không tạo mismatch, nên hàm báo `All 18 baseline artifacts match SHA256 exactly` hoặc `All 0 ...` với PASS. Khi xóa manifest, hàm trả WARN rồi readiness vẫn true. Không có baseline được sinh động như lời thông báo.

**Minimal counterexample:** baseline `{}` → exploratory exit0, DATA_INTEGRITY_READY=true. **Positive control:** thay một file có digest hợp lệ bằng nội dung sai → exit1. Vì thế vòng so digest có hoạt động khi được cấp entry thật; lỗi nằm ở root, yêu cầu tính đầy đủ và cách xử lý missing/empty.

**Sửa tối thiểu:** xác định root đúng và xác minh sentinel; schema/hash-list/version rõ; bắt buộc số file/tập file yêu cầu; thiếu file, MISSING, digest sai hoặc manifest rỗng phải FAIL. Baseline mới tạo trong version mới, không sửa baseline lịch sử để ngụy tạo kiểm chứng trước đó. Regression phải bao gồm đúng file, thay nội dung, missing, empty, sai root và traversal ngoài root.

**Bằng chứng báo cáo:** cả5 digest in trong `HARDENING_REPORT.md` đều khác digest tính hiện tại. Giá trị đầy đủ nằm trong phụ lục `evidence_crosschecks.json`. Không có provenance giải thích những digest báo cáo đó; không được coi chúng là đã xác minh. Kết quả này không tự chứng minh ý định gian dối.

### 2. Config và fail-closed

`self.config_path` chỉ có một AST usage kiểu Store tại dòng41. Không có read, parser/schema validator hoặc truy cập nội dung. Do đó đổi mọi trường config không thể làm thay đổi checks. Các case dưới đây chạy bằng source A06 trong sandbox riêng; không có imagery, split registry hay audit summary ở sandbox.

| Case | Điều kiện cố ý thiếu/sai | Kết quả thực tế | Mong đợi theo task |
|---|---|---|---|
''')
    details={
        'no_config':'Không có config và không có dữ liệu/audit source',
        'missing_config':'--config trỏ file không tồn tại',
        'malformed_config':'File config chứa JSON hỏng',
        'forbidden_roles_capacity_paths':'Train/calibration mang CONFIRMATORY_TEST, Q=1/GT=7, nguồn thiếu, output đè vùng cũ, buffer âm, mode mâu thuẫn',
        'missing_baseline':'Xóa BASELINE_HASHES.json; config sai vẫn đi kèm',
        'empty_baseline':'BASELINE_HASHES.json={}; config sai vẫn đi kèm',
        'tampered_real_digest':'Một digest hợp lệ nhưng nội dung file khác',
        'confirmatory_no_inputs':'Confirmatory không có nguồn/holdout'}
    for r in p:
        lines.append(f"| `{r['case']}` | {details[r['case']]} | exit{r['exit_code']}, {r['overall']} | exit≠0 |\n")
    lines.append('''
**6/8 case bị gate chấp nhận trái yêu cầu; 2/8 case bị từ chối đúng.** Các case gộp không cô lập từng field riêng, nhưng config không được đọc là bằng chứng trực tiếp tất cả field đều không được kiểm. Không mô tả đây là6 unit test độc lập đã sửa.

`check_calibration_standards`, `check_crop_sampling_invariants`, `check_output_contract` chỉ tạo string rồi `add_result(PASS)`. Không đọc sampler, stage output hoặc calibration registry. `run_all` không yêu cầu mọi critical check hiện diện/PASS; WARN không chặn readiness. Confirmatory bị block bằng thông báo cố định và `has_registered_external_holdout=False`, không phải validator của một proposed independent split.

**Sửa tối thiểu:** parse/validate config thật; xây tập requirement bắt buộc theo mode/study; UNKNOWN/thiếu check không thành PASS; kiểm data roles, labels accessed, checkpoint/hash, capacity, destination, calibration và split policy trước runtime. Cho phép exploratory exception bằng waiver có đủ trường, không bypass critical confirmatory blocker. Trainer/evaluator phải gọi cùng gate trước đọc nhãn/ghi artifact; chạy CLI thủ công không bảo vệ được lệnh gọi trực tiếp.

### 3. RNG helper không phải bản sửa loader đang hoạt động

Loader thật giữ `default_rng(torch.initial_seed()+i)`; không có `set_epoch`. Cả trainer F và Mask R-CNN đang bật persistent_workers. Probe dùng bốn NPZ tổng hợp, crop12px, augment=False, DataLoader thật, không checkpoint/GPU/train:

| Workers | Persistent | Số crop giống hệt qua hai epoch | Diễn giải |
|---:|---|---:|---|
| 0 | false | 4/4 | Không có reseed theo epoch trong loader |
| 2 | false | 0/4 | Worker mới được seed lại; không quy lỗi cho mọi DataLoader |
| 2 | true | 4/4 | Tái hiện đúng rủi ro của recipe đang dùng |

`get_crop_rng` trong test tạo seed có epoch đúng như ý tưởng, nhưng không được production loader/trainer gọi. Test validation chỉ gọi cùng epoch0 hai lần; helper tự nó không khóa validation epoch khi caller truyền epoch khác.

**Sửa tối thiểu:** tích hợp epoch-aware seed vào sampler/loader và đường truyền epoch tới persistent workers; khóa validation sampling ở API/policy; kiểm reproducibility giữa run, diversity giữa epoch và behavior theo worker count. Khi sửa phải ghi migration: training stream mới khác lịch sử, không gọi đó là tái lập bitwise các checkpoint cũ.

### 4. Test coverage và liên kết regression

Hai hàm `test_fixture_12_confidence_only_changes` (509) và `test_fixture_17_nonsquare_gsd_determinant` (546) chỉ `pass`. Các invariant tương tự có test khác, nên không nói hoàn toàn thiếu kiểm tra score/non-square; vấn đề là hai hàng PASS này không thêm bằng chứng.

Capacity fixture chỉ tính `N_gt > Q_capacity` với7 và4. Score-tie fixture không dùng score, không kiểm tie policy. Border fixture chỉ kiểm mask vừa tạo có pixel ở hàng0. Partial fixture chỉ matching một mask bị cắt với bản sao hoàn hảo; không kiểm full-crown sampling hoặc fragment policy. Không có Stage0→3 lineage/uniqueID/inactiveQ/stitching implementation fixture như báo cáo claim. Không có test AP/Boundary F1 thực thi trong suite này.

Trong `ROOT_CAUSE_MATRIX.csv`,11/12 tên regression được dẫn không tồn tại trong suite bàn giao. Một test khác có thể gần ý nghĩa, nhưng không làm cho link sai trở thành bằng chứng đúng. Trạng thái tất cả `VERIFIED_FIXED` phải được thay bằng đánh giá từng finding trong báo cáo sửa tiếp, giữ nguyên báo cáo cũ.

### 5. Reference và diện tích

Code `prepare_canopyrs.py` chỉ gán nơi `sub==0`: annotation xuất hiện trước thắng. A06 gọi `index_descending`/last-wins là legacy. Fixture hai mask lồng nhau làm9/25 pixel đổi chủ sở hữu; đó là policy khác, không phải replay lịch sử.

Audit đã lưu phân biệt train63 unique-empty và10 priority-empty; validation4 và4. A06 executive có nơi ghi đúng, nhưng FIX_LOG và findings table lại gọi63 training annotations bị priority erase. Unique-empty cũng không chứng minh cây hoàn toàn bị che hoặc nằm dưới tán; chỉ chứng minh mask không có pixel độc quyền trong bộ reference hiện tại.

Area-ascending priority có thể nghiên cứu như một phương án supervision, nhưng không được tự thay first-wins rồi gọi là sửa reference đúng hơn. Tie-break phải dựa ID ổn định và được khai báo; giữ original RLE cho primary matching. Mọi support-derived area cần tên và mẫu số riêng.

Probe numeric: NaN area được xếp `ge9`; affine determinant0 trả area0; bbox giao nhau với buffer0 được helper cho valid. Shape mask sai được NumPy từ chối bằng IndexError trong fixture đã thử; **không kết luận hàm âm thầm chấp nhận sai shape**. Cần validation sớm có lỗi rõ, nhưng case này đã bị từ chối.

### 6. Grouping và F–G–Q policy

Khi không có pair, helper trả bốn số0 và bỏ counts; khi chỉ có một same-tree pair dự đoán đúng, same=1, diff=0, balanced=.5. Theo execution plan, denominator0 phải NA(reason), không tạo điểm sai nhân tạo cho class không được lấy mẫu. Phải báo class counts, unconditional/conditional accuracy, coverage, abstention và overlap riêng.

Execution plan trước (dòng466) và MEASUREMENT_SPEC đã khóa E2E top400 + highest-score overlap resolution. A06 áp unique-cover cả ở E2E. Hai lựa chọn đều có thể định nghĩa thành metric, nhưng đây là **thay đổi policy**, không phải sửa lỗi tất yếu. Fixture p=(0,0),q=(0,1), masks A={p,q},B={p,r}: prior highest-score rule cho same accuracy1 khi scoreA>scoreB và0 khi đảo score; A06 abstain ở p trong cả hai trường hợp. Structural vẫn đúng là invariant score khi giữ set mask; set Stage2 tạo từ confidence không tự trở thành hoàn toàn score-independent.

Không đồng nhất query unmatched khi huấn luyện bằng GT với inactive/no-object khi inference. Inference không được xem GT association để quyết định giữ query. Stage adapters và mask/score/provenance cần API thật, không chỉ thông báo PASS.

Execution plan gọi `G1_merge` là learned association trên các atom nguyên vẹn. A06 đổi G1 thành cut-and-merge, kèm multicut có thể tách basin. Multicut chỉ phân hoạch **tập node hiện có**; nếu node là atomic basin, nó không chia pixel bên trong node. Repartition cần representation/phép split bổ sung. Không dùng hạn chế merge-only để giải thích toàn bộ kết quả âm EXP020/021 khi chưa có can thiệp phân biệt nguyên nhân.

Buffer30m chưa có estimate autocorrelation hay căn cứ crown diameter được dẫn. Default min-area .25m²/threshold.5 cũng chưa được protocol phê chuẩn. Các con số này phải là đề xuất cấu hình mới, không gọi là scientific facts đã verified. Không đòi mọi head cùng tham số để tuyên bố fairness; cần báo capacity/loss/budget, hai study type và supervision thực tế.

### 7. Artifact safety và lệnh thí nghiệm

Preflight luôn `write_text` vào cùng `READINESS_REPORT.json`; generator cũng ghi lại baseline cùng đường dẫn. Ở7/8 sandbox, bytes readiness khác trước chạy; confirmatory case cho bytes giống nhưng vẫn thực hiện lệnh ghi. Phải dùng destination mới/exclusive create và manifest mỗi run, không dùng output lịch sử làm file trạng thái có thể ghi đè.

`NEXT_EXPERIMENT_DECISION.md` đưa lệnh Q gọi `train_treeflow.py`, nhưng trainer này là F, bắt buộc `--out`, không nhận `--config`, `--crop-size`, `--query-capacity`, `--device`; YAML được dẫn không tồn tại. EXP021 nhận seed/split/theta/device, không có `--mode exploratory_repartition`; đó vẫn là veto merge lịch sử, không implementation G-repartition. Chỉ `--help` được thực thi để xác minh; không chạy những lệnh nghiên cứu này.

**Tiêu chí sửa:** mỗi recipe có parser/config thật, smoke hoặc dry-run không training, đúng head và output version; không sửa tên cờ rồi mặc định field trainer đã thành Q. Không thực hiện kiến trúc mới trong hardening task; recipe chưa có phải ghi BLOCKED.

## IV. Verification evidence

### Quét COCO trực tiếp trong lượt review này

Quét toàn bộ prepared COCO bằng `pycocotools.area/toBbox` trên RLE nén; kiểm ID trong **từng split**, image association, RLE dimensions, zero/invalid area và bbox nằm trong ảnh. Không kiểm global-ID uniqueness giữa các COCO riêng vì chúng có scope khác nhau.

| Split | Images | Annotations | Duplicate IDs / orphan / size mismatch / zero RLE area được phát hiện | Native area khác RLE | Native bbox khác RLE bbox |
|---|---:|---:|---:|---:|---:|
''')
    for r in read('source_annotation_metadata_scan.json')['splits']:
        flags=r['flags'];bad=sum(v for k,v in flags.items() if k not in ('metadata_RLE_area_difference','metadata_RLE_bbox_difference'))
        lines.append(f"| {r['split']} | {r['images']} | {r['annotations']} | {bad} | {flags.get('metadata_RLE_area_difference',0)} | {flags.get('metadata_RLE_bbox_difference',0)} |\n")
    lines.append('''
Khác biệt metadata được đếm với atol1e-6 và rtol0. Đây là flags giữa vector/native metadata và raster geometry, **không phải90.462 nhãn hỏng**. Không thay đổi annotation, bbox, area hoặc denominator. Những kiểm tra này không thay thế decode/image alignment, invalid-support intersection hoặc xác nhận sinh học từ nguồn độc lập.

### Commands và cách tái kiểm tra an toàn

Từ repo root `/home/chung/RS/Image Segmentation`:

```bash
# Chương trình review chạy trên snapshot đã xác minh hash; tất cả CLI mutations ở sandbox.
./cs2_venv/bin/python -B crown_segmentation_research/research_integrity/a06_independent_validation/validate_a06.py

# Chỉ tổng hợp các bằng chứng đã lưu, không chạy thí nghiệm.
./cs2_venv/bin/python -B crown_segmentation_research/research_integrity/a06_independent_validation/build_review_report.py
```

Exit0 của review runner nghĩa là **đã thu thập probe thành công và snapshot không bị thay đổi**, không phải dự án READY. Gate độc lập nằm trong `INDEPENDENT_READINESS_REPORT.json` và hiện FAIL. Lệnh chính xác của mỗi subprocess (gồm8 sandbox CLI) được lưu trong phụ lục JSON.

Lượt đầu đã thực thi `python -B -m crown_segmentation_research.research_integrity.a06_hardening.test_suite` trước khi file nguồn biến mất. Lượt replay thực thi `python -B <reviewed_a06_snapshot/test_suite.py>` trên nội dung cùng hash. Cả hai đều24/24; đây vẫn là24 test khác nhau, không cộng thành48 invariant.

Các lỗi của chương trình review cũng được giữ lại: lần đầu tính nhầm ROOT làm import fail; lần kế tiếp probe shape sai nhận IndexError nhưng runner chưa catch. Đã sửa **runner của review**, lưu log hai lần dừng, và chạy toàn bộ pipeline review lại thành công. Không đưa các lần dừng đó vào kết quả pass của A06; không bỏ chúng khỏi provenance.

### Những gì không được thực thi

Không full training F/G/Q; không model sanity fit mới; không oracle A/B/C; không model-performance evaluation trên historical test; không AP search/threshold tuning; không download/install dependency; không tái tạo dữ liệu nguồn. Hai `--help` không train/evaluate. Quét annotation test để kiểm metadata không phải đánh giá hiệu năng model trên test.

## V. Scientific implications

Các AP39.03, A4px52.50, B0.25 77.86, small recall25.72% và score-only+1.89 vẫn là **historical findings** theo protocol tương ứng; không được phiên review này tái lập thành performance mới. Lỗi A06 không tự chứng minh các con số đó sai, cũng không làm chúng thành confirmatory evidence.

Không diễn giải oracle như phân rã AP cộng được hay chứng minh biological identity. Không dùng Q fragment sanity để kết luận query kém/giỏi về generalization. Dữ liệu có sparse boundary overlap không đồng nghĩa extensive test contamination; tổng pairwise same-grid overlap38.955px phù hợp14.094+12.966+11.895. Còn2 cross-grid ground-overlap cases sau erosion là artifact audit cũ đã kiểm tra báo cáo, không bị xóa khỏi graph để tuyên bố clean split. Loại bỏ shared pixels không chứng minh spatial independence.

Không coi provenance mosaic là ba independent sites/surveys; không suy acquisition date từ filename. OOF tile-fold khác nhau không bảo đảm độc lập không gian. Mức độ inflation do OOF chưa được review này ước lượng. Ưu tiên next step là hoàn thiện control và measurement contract đã được chốt; chưa có căn cứ đổi toàn bộ method hoặc thêm LiDAR từ các lỗi hardening này.

## VI. Readiness decisions

```json
{
  "DATA_INTEGRITY_READY": false,
  "EXPLORATORY_FGQ_READY": false,
  "CONFIRMATORY_BENCHMARK_READY": false,
  "CONFIRMATORY_NOT_READY": true,
  "EXTERNAL_HOLDOUT_REQUIRED": true
}
```

**Level1 không đạt:** nguồn annotation có nhiều kiểm tra cơ bản đạt, nhưng preventive controls về hash/config/reference/runtime/metric chưa đạt tiêu chuẩn nghiệm thu (V01–V09,V13,V15,V16). `false` nghĩa chưa chứng minh hạ tầng đủ tin cậy, không khẳng định data gốc đã hỏng.

**Level2 không đạt:** RNG chưa tích hợp, full-crown sampler/capacity/output adapter/access restrictions chưa có kiểm chứng vận hành, grouping policy thay đổi và recipe không executable (V02–V04,V07,V10–V15). Thư mục nguồn hiện còn thiếu file (V17). Không nên dùng A06 PASS để mở benchmark F–G–Q.

**Level3 không đạt:** ngoài blockers kỹ thuật, chưa có genuinely appropriate independent holdout, provenance/selection/calibration/primary margins chưa hoàn tất. Confirmatory block hiện tại đúng về lựa chọn không chạy Z1 lịch sử, nhưng không phải gate data-driven đã sẵn sàng cho holdout mới. External data cần thiết; không đổi tên/random-split dữ liệu đã xem thành lockbox.

## VII. Minimal next-step execution plan — giao lại agent sửa

1. **P0 — Chốt phiên bản/writer và bảo toàn:** dùng snapshot/hash đã chụp để sửa trong version/branch mới; xác định23 file mất vì thao tác nào. Không khôi phục đè khi agent khác có thể đang viết. Giữ nguyên A06 report và mọi original data/output.
2. **P0 — Gate thật:** sửa root, manifest completeness, config parsing/schema, status aggregation; thêm6 negative cases đã chứng minh lọt gate và positive controls tương ứng. Expected nonzero phải được assert ở integration test, không chỉ in log.
3. **P0 — Runtime enforcement:** gắn preflight vào entrypoints trước đọc forbidden labels/ghi output. Tái dùng A00 registry/validators, kiểm split roles và calibration provenance, fail-closed khi thiếu critical source. Output là run directory mới.
4. **P1 — Sampler/target:** tích hợp epoch RNG và full/partial/ignored accounting; test DataLoader worker modes, crop/resize alignment và capacity API. Không đổi fragment ontology hoặc bỏ target; scientific choice cần ghi rõ.
5. **P1 — Metrics/output:** tách structural/E2E, giữ lịch sử highest-score/top400 hoặc đăng ký metric khác; NA/counts/common-support; thực thi Stage adapters/provenance/inactive Q/ID invariance và scene stitching nếu có claim scene-level.
6. **P1 — Policy/fairness:** sửa first-wins/counts và G1 naming; đánh dấu30m/.25m²/threshold là proposed; ngân sách/recipes thực tế. Không tạo method mới để lấp placeholder.
7. **P1 — Regression/CI và bằng chứng:** thay no-op/constant fixtures; kiểm mọi link test; thêm RLE/alignment/AP/Boundary/valid denominator synthetic tests cho code thật; ghi commands/environment/hash từ máy, không chép con số thủ công. Chạy synthetic unit/integration, rồi gate âm/dương. Chưa train/benchmark trong hardening task.

Nghiệm thu lại chỉ sau khi các counterexample trong phụ lục bị từ chối đúng, config hợp lệ có đầy đủ source thật đi được tới dry-run an toàn, sampler/output/metric integration tests đạt, snapshot/historical registered artifacts không đổi, và toàn bộ scientific decision còn mở được báo đúng trạng thái. Không đưa lệnh training vào next steps ở mức readiness hiện tại.

## Phụ lục A. Requirement → implementation → test → status

`VERIFIED_CORRECT` cho matching helper chỉ có phạm vi các fixture cụ thể đã thực thi; không phải chứng nhận mọi corner case/cardinality ở arbitrary capacity hoặc COCO AP.

| ID | Scientific requirement | Implementation | Test/evidence | Status |
|---|---|---|---|---|
''')
    for r in requirements:lines.append('| '+' | '.join(r)+' |\n')
    lines.append('\n## Phụ lục B. Bằng chứng máy đọc đầy đủ\n\nCác block sau được nhúng từ artifact đã thực thi; không chỉ dẫn link. SHA256 của source/report có trong manifest riêng và block liên quan.\n')
    evidence=['environment.json','original_tests_reexecuted.json','first_execution_original_tests_reexecuted.json',
              'historical_source_check.json','source_annotation_metadata_scan.json','evidence_crosschecks.json',
              'source_analysis.json','adversarial_expectations.json','metric_probes.json','actual_loader_rng_probe.json',
              'advertised_commands_check.json','review_preservation.json','external_source_drift.json',
              'snapshot_replay_preservation.json','INDEPENDENT_READINESS_REPORT.json']
    for name in evidence:
        lines.append(f'\n### {name}\n\n```json\n{(HERE/name).read_text().rstrip()}\n```\n')
    lines.append('\n## Phụ lục C. Log suite replay\n\n```text\n'+(HERE/'original_tests.log').read_text().rstrip()+'\n```\n')
    lines.append('\n## Phụ lục D. Source chương trình kiểm định để agent đọc/tái sử dụng\n\nCode này điều tra bản A06 được snapshot, không thay thế production preflight.\n\n```python\n'+(HERE/'validate_a06.py').read_text().rstrip()+'\n```\n')
    (HERE/'A06_INDEPENDENT_VALIDATION_REPORT.md').write_text(''.join(lines))
    manifest={}
    for path in HERE.iterdir():
        if path.is_file() and path.name!='REVIEW_ARTIFACT_HASHES.json':manifest[path.name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    dump('REVIEW_ARTIFACT_HASHES.json',{'generated_UTC':now,'scope':'Top-level independent review files; snapshot originals covered by review_source_freeze and snapshot replay manifest','files':manifest})
    print({'report':str(HERE/'A06_INDEPENDENT_VALIDATION_REPORT.md'),'findings':len(findings),'readiness':'FAIL'})

if __name__=='__main__':main()
