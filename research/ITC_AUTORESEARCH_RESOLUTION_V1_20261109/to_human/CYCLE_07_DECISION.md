# Kết quả kiểm tra M1 bằng đối chứng huấn luyện thực

Đã chạy xong hai Mask DINO cùng checkpoint gốc, cùng796TRAIN windows/398updates/300queries. S0huấn luyện tiếp ở1024; S1thêm phép nhân đôi pixel không mất thông tin cho một nửa mẫu. Đánh giá24lượt trên baTRAIN windows đãexposed,298references/133nhỏ; không phải SAME6hay kiểm định địa lý độc lập.

Chênh lệch rawcoverage50 giữa ảnh gốc và ảnh nhân đôi giảm147→36(75,5%). Trên ảnh nhân đôi, finalTPtăng114→236;107trong123matches mới chưa có qualifyingrawmask ởS0. Giữ cùng443predictions vẫn tăng114→233TP. Đây là thay đổiformation thực, nhưng từ một augmentation đã có.

Trên ảnh gốc: TP276→276, nhưng2matches mất đều là tán nhỏ; smallTP118→116, AP62,601→61,641, AP7577,169→74,496 vàFP163→222. Như vậy không có practicalsmall-recall improvement. Tính nhất quán scale tổng quát cũng chưa được thiết lập: cặpC/Dvẫn chênh lớn.

Kết luận: phép thử đã cho thấy ảnh hưởng rõ của điều kiện huấn luyện/compute; chưa cung cấp căn cứ cho nguyên lýobservabilitymới, nguyên nhânBAMhay một hệ thống tốt hơn. Đóng recipe mộtpass; giữ cácbaselineđã xác minh. Không thêmepoch, đổi threshold, mởV2hoặc tuyên bốnovelty. Mục tiêu tìmoriginalmethod vẫn chưa đạt.

[Báo cáo đầy đủ](../../ITC_M1_OBSERVABILITY_SCALE_RESEARCH_V1_20261110/experiments/scale_exposure_control_v1/REPORT.md).
