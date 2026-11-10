# Kết quả kiểm tra M1 bằng đối chứng huấn luyện thực

Đã chạy xong hai Mask DINO từ cùng checkpoint gốc, cùng 796 TRAIN windows, 398 updates và 300 queries. S0 huấn luyện tiếp ở 1024 pixels; S1 thêm phép nhân đôi pixel không mất thông tin cho một nửa mẫu. Đánh giá 24 lượt trên ba TRAIN windows đã được sử dụng trước đó, gồm 298 reference occurrences và 133 trường hợp nhỏ. Đây không phải SAME6 hay kiểm định địa lý độc lập.

Chênh lệch raw coverage50 giữa ảnh gốc và ảnh nhân đôi giảm 147→36 (75,5%). Trên ảnh nhân đôi, final TP tăng 114→236; 107 trong 123 matches mới chưa có qualifying raw mask ở S0. Giữ cùng 443 predictions vẫn tăng 114→233 TP. Hiệu ứng có thay đổi actual mask formation, nhưng đến từ một augmentation thông thường.

Trên ảnh gốc: TP 276→276, nhưng hai matches bị mất đều là tán nhỏ; small TP 118→116, AP 62,601→61,641, AP75 77,169→74,496 và FP 163→222. Như vậy chưa có lợi ích thực dụng về small recall. Tính nhất quán scale tổng quát cũng chưa được thiết lập: cặp C/D vẫn chênh lệch lớn.

Kết luận: phép thử cho thấy ảnh hưởng rõ của điều kiện huấn luyện và tính toán; chưa cung cấp căn cứ cho nguyên lý observability mới, nguyên nhân lỗi trên BAM hay một hệ thống tốt hơn. S1 xử lý tổng số pixels gấp 2,5 lần S0, nên không tách riêng tác động của scale exposure khỏi compute. Đóng recipe một pass; giữ các baselines đã xác minh. Không thêm epoch, đổi threshold, mở V2 hoặc tuyên bố novelty. Mục tiêu tìm original method vẫn chưa đạt.

[Báo cáo đầy đủ](../../ITC_M1_OBSERVABILITY_SCALE_RESEARCH_V1_20261110/experiments/scale_exposure_control_v1/REPORT.md).
