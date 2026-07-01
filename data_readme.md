Dưới đây là bản "bản đồ" chi tiết về 2 bộ dữ liệu khổng lồ mà chúng ta vừa đưa về máy. Đây là những bộ dữ liệu thuộc hàng "Huyền thoại" trong lĩnh vực Viễn thám (Remote Sensing):

---

### 1. WHU Building Dataset (Bộ dữ liệu Tòa nhà WHU)
Đây là bộ dữ liệu chuyên biệt để nhận diện **Mái nhà/Tòa nhà**. Nó cực kỳ sạch và có độ phân giải rất cao.

*   **Mục tiêu:** Giúp AI biết đâu là nhà, đâu là đường/cây.
*   **Đặc điểm:** Ảnh được cắt sẵn (Tiling) về cỡ 512x512 pixel.
*   **Cấu trúc thư mục:**
    *   `image/`: Chứa ảnh vệ tinh màu (RGB) 3 kênh.
    *   `label/`: Chứa ảnh đen trắng (Binary Mask).
*   **Ví dụ:**
    *   **Ảnh gốc (`.tif`):** Một bức ảnh chụp từ máy bay thấy khu dân cư với các mái nhà ngói đỏ.
    *   **Ảnh nhãn (Mask):** Những ô vuông màu **Trắng** (giá trị 255) đúng hình dáng cái nhà, còn lại là màu **Đen** (giá trị 0).
    *   *Tính năng:* Độ chính xác về hình học rất cao, giúp mô hình SAM2 của bạn học được cách phân tách rìa tòa nhà cực sắc nét.

---

### 2. iSAID Dataset (DOTA for Instance Segmentation)
Đây là "quái vật" trong làng nhận diện đối tượng từ trên cao. Nó không chỉ phân loại mà còn tách biệt từng cá thể.

*   **Mục tiêu:** Nhận diện 15 loại đối tượng khác nhau (Máy bay, Tàu thủy, Xe cộ, Sân tennis, Cầu cảng...).
*   **Đặc điểm:** Ảnh gốc có kích thước siêu lớn (có tấm lên tới 4000x4000 pixel), chứa hàng nghìn đối tượng nhỏ xíu.
*   **Cấu trúc thư mục:**
    *   `images/`: Ảnh màu gốc (RGB).
    *   `Semantic_masks/`: Nhãn phân vùng (Ví dụ: Tất cả tàu thủy là màu Xanh, tất cả máy bay là màu Đỏ).
    *   `Instance_masks/`: Nhãn từng cá thể (Ví dụ: Tàu số 1 màu Xanh nhạt, tàu số 2 màu Xanh đậm để AI phân biệt được có 2 con tàu sát nhau).
*   **Ví dụ:**
    *   **Ảnh gốc:** Một bãi đỗ xe hoặc sân bay nhìn từ vệ tinh.
    *   **Ảnh nhãn:** Thấy rõ từng chiếc xe được "tô màu" riêng biệt.
    *   *Tính năng:* Thử thách cực đại cho AI vì có những đối tượng rất nhỏ (Small Objects) và mật độ dày đặc.

---

### 📂 Cấu trúc thư mục hiện tại trên máy của bạn:

```bash
data/raw/
├── whu/
│   ├── aerial/             # Ảnh chụp từ máy bay (512x512)
│   └── change/             # Ảnh phục vụ nhận diện biến đổi (Building Change)
└── isaid/
    ├── test_images/        # Thư mục chứa 6.3GB ảnh gốc (Vừa tải & giải nén)
    ├── training/           # Dữ liệu huấn luyện
    │   ├── Semantic_masks/ # Nhãn loại đối tượng
    │   └── Instance_masks/ # Nhãn từng cá thể
    └── validation/         # Dữ liệu kiểm thử
```

### 💡 Tại sao kết hợp 2 bộ này lại "Mạnh"?
*   **WHU** giúp mô hình giỏi về **Hình học & Cấu trúc** (Tòa nhà thường có khung cứng).
*   **iSAID** giúp mô hình giỏi về **Đa dạng đối tượng & Tỉ lệ** (Từ máy bay to đến chiếc xe nhỏ xíu).

Khi đưa vào pipeline **SAM2 + MoE** của bạn:
*   Chúng ta sẽ dùng **WHU** để gác cổng về độ chính xác ranh giới.
*   Dùng **iSAID** để mở rộng khả năng nhận diện "vật thể lạ" trong các bài toán thay đổi địa hình.

Bạn có muốn tôi hiển thị thử 1 cặp ảnh (Ảnh gốc + Nhãn) của bộ nào để bạn "tai nghe mắt thấy" không?