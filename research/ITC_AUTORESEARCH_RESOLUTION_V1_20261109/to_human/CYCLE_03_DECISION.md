# Vòng 3 — đã chuẩn bị bước kiểm chứng độc lập, chưa có nhãn mới

**Quyết định: `BLIND_REFERENCE_HANDOFF_PREPARED_INDEPENDENT_EVIDENCE_NOT_COLLECTED`. Mục tiêu nghiên cứu tổng thể vẫn chưa đạt.**

Phần đã làm thật: xuất sáu ảnh RGB 2048×2048 từ cohort BAM TRAIN đã khóa, ba ảnh mỗi nhóm địa lý, kèm validity và hướng dẫn chú giải mù. Chọn bằng vị trí metadata [0,4,8,12,16,20], không chọn theo performance. Quy tắc đã commit trước khi xuất gói, nhưng sau nghiên cứu BAM; đây không phải cohort confirmatory mới.

[Gói dành cho người chú giải](../experiments/reference_evidence_handoff/RGB_BLIND_PACKET.zip) đã được kiểm tra pixel/hash, không chứa predictions, polygons cũ, số cây tham chiếu hay tên ảnh nguồn. Bản đồ nguồn được giữ riêng. Chú giải cho phép tán chồng lấp, nhiều phần rời, bất định presence/identity/boundary và không đủ bằng chứng; không ép mọi vùng thành một cây hoặc background. Bộ nhập nhãn đã kiểm tra các ngữ nghĩa này bằng fixtures kỹ thuật tạm, không tạo nhãn người giả.

Hiện có **0 người chú giải được xác nhận, 0 lượt trả lời thực tế**. Gói chưa được gửi cho ai. Câu hỏi cụ thể đã đặt với người dùng: có thể bố trí ba người có kinh nghiệm thực hiện độc lập cùng sáu ảnh hay không. Không có phản hồi thì không được coi là đã có người/đã được đánh giá.

Nguồn Steier 2024 đã xác minh trên publisher: dữ liệu phải xin tác giả; chưa có tệp nhãn và chưa liên hệ. SelvaMask repeated annotations thuộc test zones theo kiểm tra trước, không mở để né ranh giới TEST. Nguồn hiện có chưa thay thế được independent RGB annotation evidence.

Nếu có người tham gia, sáu ảnh chỉ là pilot tính khả thi/reference interpretation. Chưa đủ để tuyên bố lỗi có hệ thống, biological identity hay generalization. Không có nhãn độc lập thì chưa phân biệt được 447 common final omissions của vòng BAM là model limitation đến đâu và reference interpretation đến đâu.

Hướng giải quyết thực dụng hiện giữ nguyên RR_MASK_INIT và các carrier mạnh. Vòng BAM đã cho lợi ích RR nhỏ và lẫn chiều, không xác lập small-source gain. Không mở kiến trúc, loss, selector hoặc decoder mới để thay cho bằng chứng còn thiếu.

Trạng thái: gói và bộ nhập **EXECUTED/VERIFIED**; chú giải con người và independent-reference analysis **NOT_RUN**; original-method foundation **NOT_ESTABLISHED**. Không có tiến trình train/inference đang chạy, không có scheduler riêng được tạo.

Chi tiết: [hướng dẫn điều phối và lệnh tái lập](../experiments/reference_evidence_handoff/README.md), [verification](../experiments/reference_evidence_handoff/VERIFICATION.json).
