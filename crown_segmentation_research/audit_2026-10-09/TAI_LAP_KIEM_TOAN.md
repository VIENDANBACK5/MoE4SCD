# Tái lập các kiểm tra trong audit

Chạy từ `/home/chung/RS/Image Segmentation`. Các script bên dưới chỉ ghi thư mục audit, không ghi mã/dữ liệu/result gốc. Script AP tự bỏ qua các run đã có trong `recomputed_metrics.json`; để chấm lại từ đầu cần giữ bản JSON/log cũ dưới tên khác trước khi chạy. Không coi thao tác resume đó là một lượt chấm mới nếu không xóa danh sách completed bằng một bản audit mới.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 cs2_venv/bin/python -B crown_segmentation_research/audit_2026-10-09/verify_saved_results.py
PYTHONDONTWRITEBYTECODE=1 cs2_venv/bin/python -B crown_segmentation_research/audit_2026-10-09/verify_source_annotations.py
PYTHONDONTWRITEBYTECODE=1 cs2_venv/bin/python -B crown_segmentation_research/audit_2026-10-09/probes.py
```

`probes.py` dùng DataLoader multiprocessing; trong sandbox hiện tại cần quyền IPC ngoài sandbox. Không cần GPU. Thư mục `/tmp` dùng cho bộ dữ liệu toy rồi được dọn bởi TemporaryDirectory. Kết quả `area_mismatches` ở probe chỉ là danh sách diện tích metadata khác đếm pixel >0,1; xem `area_source_checks.json` và báo cáo để diễn giải đúng, không tự coi toàn bộ danh sách là annotation sai.

Lệnh tương đương đã dùng cho 39 test, tận dụng pytest có sẵn nhưng giữ ưu tiên thư viện cs2:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 cs2_venv/bin/python -B - <<'PY'
import sys
sys.path.append('/home/ubuntu/anaconda3/lib/python3.11/site-packages')
import pytest
raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider',
    'tests/test_clean_benchmark.py',
    'tests/test_legacy_training_fixes.py',
    'crown_segmentation_research/legacy/tests']))
PY
```

Đường pytest trên là phụ thuộc máy audit; khi dựng môi trường riêng nên cài pytest vào môi trường với phiên bản cố định, không thêm đường máy khác.

EXP-015 được chạy qua module với `RES` đổi sang thư mục audit trước khi gọi `main()`; không chạy CLI mặc định ghi results gốc. Snapshot thư viện ở `environment_freeze.txt` và `math_environment_checks.json`; không suy đây là môi trường run lịch sử.

`finalize_audit.py` xác nhận 30 run/270 metric, đối chiếu inventory gốc và ghi hash các checkpoint/GT. Không nên chạy nó sau khi cố tình sửa mã gốc rồi diễn giải khác biệt là do audit. Bảng báo cáo đã được điền; rerun finalize không thay bảng nếu placeholder không còn.

Tài liệu này không cung cấp lệnh train mới: việc xác nhận toàn quá trình train và xác nhận trên holdout mới nằm ngoài các lượt thực thi của audit đã hoàn tất.
