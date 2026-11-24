# Bằng chứng kiểm toán

Ngày: 2026-10-09. Phạm vi: crown_segmentation_research và phụ thuộc liên quan trong workspace.

Kết quả đầy đủ: BAO_CAO_KIEM_TOAN.md. Kết quả machine-readable: audit_summary.json.

- Nhánh clean: 30 bộ mask × 9 metric COCO khớp từng seed; số tổng hợp cần làm tròn sau mean/SD. TFN cuối 34.215409 ± 0.299743; MRCNN 34.820764 ± 0.666865.
- Toàn bộ 90.462 annotation giữ đúng Parquet. 9 tile mẫu label/RLE/valid đúng. EXP-015 chạy lại khớp; giao train-test Z1 12.966 pixel; không đồng nghĩa rò rỉ toàn vùng như lineage V1.
- Legacy: khởi tạo từng thấy test; evaluate_metrics tính F1 dưới nhãn AP; benchmark cũ hardcode baseline; bản thảo không khớp output. Những số này đã được docs mới rút lại nhưng công cụ cũ còn tái sinh được.
- Test được dùng nhiều lần trong vòng phân tích/thiết kế; chọn cuối trên val không khôi phục test giữ kín. HGB không fit nhãn test trong code đã kiểm.
- Lỗi RNG persistent worker: 4/4 sample toy lặp ở hai epoch cho cả hai họ. Shuffle thực có thể đổi worker, nên giới hạn tối đa 8 chuỗi/item/run liên tục, không khẳng định mọi epoch thực y hệt.
- 0 clean files tracked trong Git; provenance chưa bất biến. 23 selections, 21 test tương ứng, 22 unique selected checkpoints còn và đã hash.
- 8 PDF sai bài; EXP-027 mới có phase 1 s42/s44, chưa đủ 3 seed/rescore để kết luận.
- Phản ví dụ oracle maxIoU: recall1 nhưng AP83.498%; bán kính tương đương không có bảo đảm center-distance cho crown tùy ý.
- Accuracy/kappa EXP-019 khớp .775/.71875 nhưng không phải human GT validation độc lập.
- Xóa Boundary AP=0/ghi đè log đã được EXP-004 thừa nhận. --limit test EXP-016 có đường overwrite trước return; audit không chạy đường đó.
- Kiểm cuối: 17.902 originals không thay size/mtime; 447 hash văn bản gốc khớp. Không suy đoán ý đồ gian lận.
