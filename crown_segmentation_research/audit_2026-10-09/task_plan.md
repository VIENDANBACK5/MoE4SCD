# Kế hoạch kiểm toán nghiên cứu

Mục tiêu: xác minh mã, setup, dữ liệu, thực nghiệm và độ trung thực của các tuyên bố; không thay đổi kết quả gốc.

- [x] 1. Kiểm kê toàn bộ cây dự án, hướng dẫn và hồ sơ nghiên cứu.
- [x] 2. Truy vết dữ liệu, split, annotation, metric, training và selection.
- [x] 3. Đối chiếu kết quả, logs, provenance, baseline và tuyên bố.
- [x] 4. Chạy kiểm tra độc lập phù hợp với tài nguyên và dữ liệu hiện có.
- [x] 5. Lập báo cáo tiếng Việt, xếp mức độ, bằng chứng và khuyến nghị.

## Quyết định
- Kiểm toán chỉ đọc đối với mã/dữ liệu/kết quả gốc. Tệp mới chỉ trong thư mục audit.
- ara-rigor-reviewer đã đọc nhưng không áp dụng chính thức: chưa có ARA/Level 1, cần kiểm toán động.
- Phân biệt lỗi đã chứng minh với thiếu bằng chứng và không suy đoán ý đồ.

## Lỗi công cụ
- rg tìm AGENTS.md ở lần đầu không có kết quả (exit 1), không phải lỗi dự án.
- Python mặc định không tương thích torch/torchvision; cs2_venv nạp được nhưng không có pytest. Dùng pytest có sẵn qua sys.path.append sau site-packages của cs2, tắt plugin tự động; không cài thư viện.
- Script kiểm toán ban đầu đặt ROOT sai một cấp; đã sửa trước khi chấm lại dữ liệu, không ảnh hưởng kết quả gốc.
- Kiểm DataLoader trong sandbox thất bại vì IPC socket; đã chạy lại cùng script với quyền ngoài sandbox được công cụ phê duyệt. Không thay đường đọc/ghi hay cài dependency.
- Kiểm diện tích >0,1 pixel ban đầu gắn tên area_mismatches; kiểm bổ sung xác nhận đây là diện tích thực từ Parquet, khác rasterization, không chứng minh lỗi chuẩn bị nhãn.

## Trạng thái
Hoàn tất báo cáo và các kiểm chứng đã chọn. 30/30 run, 270 metric khớp tới 2 chữ số; 90.462 annotation khớp nguồn; 39 test qua. Không thay đổi size/mtime của 17.902 tệp gốc, 447 hash văn bản gốc cũng khớp. Lập hash 22 checkpoint được chọn. Không train hoặc suy luận lại toàn test; các giới hạn ghi trong báo cáo.
