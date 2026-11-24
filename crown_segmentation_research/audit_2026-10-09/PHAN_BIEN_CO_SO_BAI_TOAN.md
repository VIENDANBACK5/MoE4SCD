# Phản biện cơ sở bài toán, toán học, vật lý và nghiên cứu AI

Ngày 09/10/2026. Bổ sung cho BAO_CAO_KIEM_TOAN.md sau câu hỏi về cách chuyển bài toán thực tế sang phương pháp. Phần này tập trung vào tính hợp lệ của lập luận; không sửa code, không train hoặc chạy một thí nghiệm benchmark mới.

## 1. Bài toán được hiểu như thế nào

Theo `target/Lộ trình nghiên cứu tách tán cây từ ảnh RGB.md`, mục tiêu là tách từng tán cây trong orthomosaic RGB khi tán kề nhau, biên thiếu/mơ hồ, kích thước khác nhau, số tán K không biết trước. Đầu ra phục vụ nhận diện từng cá thể và các phép đếm/đo trên tán. Câu hỏi nghiên cứu dự kiến là suy diễn ranh giới dựa vào bằng chứng ảnh và prior hình dạng/kích thước/ngữ cảnh, trong một phân hoạch nhất quán.

Pipeline hiện tại: ảnh → backbone/FPN → canopy, SDT chuẩn hóa, flow → ứng viên và vùng hút → gộp theo đèo SDT → làm sạch mask → học score dự đoán IoU. Centroid head vẫn tồn tại nhưng loss weight=0 trong C5.

Đây là một cách mô hình hóa instance segmentation có cơ sở. Nhưng **định nghĩa bài toán hợp lý không làm mọi giả thuyết giải thích thất bại hoặc mọi bảo đảm của decoder trở thành đúng**.

## 2. Mục tiêu quan sát và giới hạn vật lý

### 2.1. Tán trên ảnh và cây ngoài thực địa

Cần chốt rõ đối tượng là tán nhìn thấy, ranh giới tán đầy đủ bao gồm phần bị che, hay mỗi cây sinh học/thân cây. Các đại lượng đó không tự đồng nhất. Với chỉ một ảnh RGB nhìn từ trên, nếu hai cấu hình rừng khác số cây nhưng cho cùng ảnh do che khuất hoàn toàn, không có hàm chỉ phụ thuộc ảnh có thể luôn xác định đúng cả hai số lượng.

Đây là giới hạn nhận dạng của bài toán ngược: học prior có thể cho dự đoán xác suất hợp lý, không thể đảm bảo khôi phục thông tin không quan sát được. Bởi vậy cần khai báo annotation policy và chỉ tuyên bố trong phạm vi đối tượng quan sát/được gán nhãn. Không đánh đồng mọi thiếu ứng viên với lỗi network; một phần có thể là nhãn mơ hồ hoặc thiếu bằng chứng ảnh, cần người độc lập kiểm.

### 2.2. Flow và “vật lý”

Flow trong dự án được dựng từ hình học mask GT; SDT không phải độ cao tán, năng lượng sinh học hay một đại lượng vật lý được camera đo. Dùng gradient, khuếch tán và Poisson là các lựa chọn mathematical prior có thể tốt. Muốn gọi là mô hình vật lý cần chỉ ra forward model tạo ảnh, đơn vị, điều kiện biên và những quan hệ vật lý được kiểm chứng.

Gradient ascent cũng không tự trở thành Hamiltonian dynamics. Với dot(x)=grad(U), dU/dt=||grad(U)||², U tăng; còn hệ Hamiltonian tự trị chuẩn có H được bảo toàn vì grad(H)^T J grad(H)=0 với J phản đối xứng. Gọi thuật toán đi lên trường hình học là “Hamiltonian” không bổ sung bảo đảm và có thể sai bản chất. Các diễn giải “Morse” cũng cần điều kiện trơn/điểm tới hạn không suy biến; EDT có ridge và plateau, smoothing không tự chứng minh mọi điều kiện.

## 3. Phản biện chuyển sang toán học

### 3.1. SDT chuẩn hóa không xóa mọi thông tin cỡ tán

Target v2 là d=e/R, R=max(e). Chuẩn hóa làm max(d)=1, bỏ thang trực tiếp ở biên độ. Nhưng khi tọa độ pixel và GSD được giữ, hình dạng/phạm vi không gian vẫn đổi theo cỡ. Tại nơi EDT liên tục khả vi và có ||grad(e)||=1, ta có ||grad(d)||=1/R. Vì vậy gradient magnitude của trường SDT lý tưởng còn mang thông tin R.

Ví dụ giải tích trên đĩa bán kính R: d(x)=1-||x-c||/R. Hai đĩa đều có đỉnh 1, nhưng độ dốc theo pixel khác nhau. Không thể suy từ “đỉnh đều bằng 1” sang “mọi thông tin kích thước biến mất”. Chuẩn hóa vector flow riêng loại magnitude của vector đó; mạng vẫn nhìn ảnh, tọa độ không gian và trường SDT/canopy.

Kiểm trực tiếp bằng `targets_v2`: hình vuông cạnh 21 và 41 px đều có max SDT=1, nhưng độ dốc tại vị trí được xét là 1/11 và 1/21. Nghịch đảo độ dốc phục hồi max EDT 11 và 21 px trong ví dụ. Xem `conceptual_probes.json`.

**Cách viết đúng:** chuẩn hóa khiến cách dùng trực tiếp giá trị SDT như khoảng cách tuyệt đối không còn hợp lệ; decoder hiện tại chưa khai thác tốt thông tin thang còn trong trường. Thêm size head là giả thuyết về biểu diễn/giám sát phụ giúp khai thác thang, không phải khôi phục một đại lượng đã chứng minh mất hoàn toàn.

### 3.2. Nhiều cực đại không suy ra nhiều tán đầu ra

Một mask không lồi có thể có nhiều cực đại EDT. Điều này đúng như một khả năng hình học, không đúng nếu viết mỗi thùy của mọi mask đều bắt buộc có một cực đại.

Số ứng viên/điểm hút và số instance cuối là hai biến khác nhau. EXP-026 ghi 65% tán lớn có ít nhất hai ứng viên trên target v2, nhưng decoder đèo gộp chúng và tỷ lệ chia của nhóm đó bằng 0. Do đó chuỗi suy luận “nhiều cực đại → mạng học đúng vẫn chia → bắt buộc target một điểm hút” không còn được kết quả hiện tại ủng hộ cho decoder cuối.

Một điểm hút mỗi tán là một prior thiết kế, không phải yêu cầu cần thiết của instance segmentation. Yêu cầu đúng là mọi vùng của cùng tán được gom đúng và các tán khác không bị gộp.

### 3.3. Tính chất target không tự chuyển sang field được mạng dự đoán

`model.py` có flow và SDT head độc lập. Dù target flow được suy từ EDT, kiến trúc/loss hiện tại không ràng buộc flow dự đoán phải cùng hướng gradient SDT dự đoán. Vì vậy định lý quỹ đạo của một trường gradient lý tưởng không tự áp dụng cho hai output mạng. Sau nội suy, noise và Euler hữu hạn, một trajectory có thể gặp hành vi khác target lý tưởng.

Chuẩn hóa gradient cũng cần được phân biệt với gradient nguyên gốc: vector grad(U)/||grad(U)|| không nhất thiết curl-free. Không kiểm curl của unit flow rồi kết luận nó không thể tương đương gradient dynamics; điều cần kiểm là sự tồn tại quan hệ phù hợp với một potential và chiều tăng potential, cùng quy tắc tại điểm gradient nhỏ.

### 3.4. Oracle tốt không loại trừ target khó học hoặc decoder nhạy

EXP-026 cho trường GT đạt khoảng 91,8–95,7 AP, có giá trị như kiểm năng lực decoder trong điều kiện lý tưởng. Nó không chứng minh field đó dễ học từ RGB, loss khuyến khích đúng sai số quan trọng, hoặc decoder ổn định với sai số mạng thực.

Một decoder xử lý tốt target chính xác nhưng sụp khi field lệch rất ít vẫn là một điểm yếu của decoder. Target khó học và decoder nhạy có thể tương tác. Không thể chia toàn bộ lỗi còn lại thành “lỗi network” chỉ bằng thay toàn bộ field bằng GT.

### 3.5. Một nền toán học hữu ích hơn: độ ổn định quanh ngưỡng đèo

Với **vùng hút và biên chung cố định**, định nghĩa p_ab(d)=max trên biên của min(d_p,d_q). Nếu ||d_hat-d||_infty ≤ epsilon, các phép min/max đều 1-Lipschitz, nên |p_ab(d_hat)-p_ab(d)| ≤ epsilon. Nếu |p_ab(d)-tau|>epsilon, quyết định cạnh đó không đổi.

Đây là một bảo đảm có điều kiện kiểm được: margin đèo càng nhỏ càng nhạy với lỗi SDT. Union-find còn có thể khuếch đại lỗi một cạnh thành gộp cả chuỗi. **Bảo đảm trên không áp dụng nguyên xi khi flow thay đổi làm vùng hút/biên chung thay đổi.** Khi đó cần đo thêm độ ổn định của proposal và basin assignment.

Hướng thí nghiệm: trên validation, giữ mask/basin để kiểm độ nhạy SDT riêng, sau đó nhiễu flow riêng, gate riêng và blur/stride riêng. Đo noise → đổi cạnh → split/merge → AP. Đó là thiết kế đề xuất, chưa chạy trong lượt này.

### 3.6. Loss theo pixel: đúng về số hạng, chưa đủ về nguyên nhân

Tán lớn đóng góp nhiều pixel vào loss là đúng. Nhưng gradient theo tham số là tổng grad_theta ell_x, phụ thuộc residual và Jacobian, có thể cùng chiều hoặc triệt tiêu. Không thể khẳng định một tán diện tích gấp 50 có gradient norm gấp đúng 50, hoặc đó là nguyên nhân chủ yếu làm mất tán nhỏ.

Thế giới tổng hợp hiện chưa tái hiện được lỗi thiếu ứng viên nhỏ. Do đó cân theo tán là can thiệp đáng thử, chưa phải nguyên nhân đã xác nhận. Loss canopy cho foreground nhỏ còn phải giữ kiểm soát false positive ở background.

### 3.7. Những sai lệch toán khác cần sửa

- Xếp theo max IoU không đảm bảo AP=recall khi có mask duplicate; audit trước đã dựng recall1 nhưng AP83,498%.
- Khoảng cách hai tâm không bị chặn dưới bởi tổng bán kính tương đương cho crown tùy ý; bảo đảm đó chỉ có trong mô hình đĩa phù hợp.
- Green Dirichlet rời rạc là một bài toán hợp lý với tính chất cực đại có điều kiện; smoothing/pooling/flow normalization/Euler và network approximation cần phân tích riêng.
- Global consistency của multicut đảm bảo một phân hoạch nhất quán về quan hệ nối/cắt, không đảm bảo đó là phân hoạch đúng cây. Tối ưu đúng một surrogate còn có thể kém AP nếu chi phí cạnh sai.
- Phương án M3 **chưa train** dùng L_small-stop_gradient(L_medium). Gradient của biểu thức đó khác gradient L_small-L_medium của ràng buộc được viết. Toy theta=.75 cho gradient 1,5 thay vì 2. Có thể dùng heuristic này, nhưng không gọi là primal update chính xác của Lagrangian đã nêu. Cần chọn formulation đúng trước khi cài và diễn giải lambda/KKT.

## 4. Phản biện nghiên cứu AI và kết quả

### 4.1. Khoảng trống “biên vắng” chưa khớp nguyên nhân khoảng cách

R2, EXP-008/008b ghi biên yếu làm khó cả hai model. Sau điều chỉnh thành phần kích thước, khoảng cách recall MRCNN-TFN ở tầng biên rõ lớn hơn tầng gần như không biên khoảng 0,019 [0,007;0,031]. Thế giới tổng hợp cho TFN nhạy hình không lồi hơn và không giảm nhiều hơn MRCNN vì mất biên.

Các số này là dữ liệu chẩn đoán được hồ sơ lưu, audit trước chưa tái thực thi toàn bộ phép ước lượng phụ/CI này. Chúng vẫn buộc diễn giải nhất quán với chính kết quả dự án: **chưa có bằng chứng rằng “không xử lý biên vắng” là nguyên nhân chính TFN thua**. Không được chuyển điều đó thành “biên yếu không bao giờ quan trọng”; proxy biên đo được cũng không chứng minh toàn ảnh không có thông tin phân tách.

### 4.2. AP tăng gồm nhiều loại cải thiện

Chấm lại mask bằng IoU dự đoán có thể cải thiện thứ hạng mà không đổi mask. Với tập mask giữ nguyên và không có top-K/score filtering khác, geometry, count thô và recall trên toàn proposal không tự tốt hơn. Vì vậy +1,16 AP từ scorer không phải bằng chứng rằng đã tìm thêm cây hoặc giải ranh giới mơ hồ tốt hơn.

Muốn bảo vệ mục tiêu đếm/đo tán, cần đếm theo raster/AOI không trùng tile, bias/MAE của K, split/merge, recall ở mức precision đã chọn trên calibration, sai số diện tích và quality boundary. Cần phân biệt metric trên tất cả proposal với metric sau threshold/top-K.

### 4.3. Giá trị hiện có và giới hạn của nó

AP độc lập từ mask đã lưu: TFN Swin-T gốc 30,17 → cuối 34,22; Mask R-CNN 34,82; MRCNN chấm lại 34,80. Có bằng chứng phương pháp hậu xử lý cải thiện baseline TFN trên benchmark hiện có. TFN chưa có ưu thế AP tổng so baseline mạnh.

Cùng model/cached fields giúp ablation decoder có giá trị. Nhưng test được xem khi phát triển, backbone/ngân sách tune khác, validation dùng nhiều vai trò và crop RNG có lỗi, nên chưa xác nhận superiority hoặc cơ chế tổng quát.

Thua AP tổng không tự làm nghiên cứu vô giá trị. Một kết quả cơ chế được cô lập tốt, kết quả âm có giải thích, ưu thế đếm/tán nhỏ có kiểm chứng hoặc chi phí thấp hơn có thể có giá trị. Chưa có số để mặc định nhận các ưu thế đó. Không thể kết luận “đủ Q1/A*” bằng checklist hay một ngưỡng AP tự đặt.

### 4.4. Novelty

Distance/energy representation và cắt energy thành component đã có trong [Deep Watershed Transform](https://arxiv.org/abs/1611.08303). Flow/cell grouping có nền liên quan trong [Omnipose, tài liệu chính thức](https://omnipose.readthedocs.io/); cần đối chiếu full method đúng phiên bản. Học mask IoU để chấm điểm đã có trong [Mask Scoring R-CNN](https://arxiv.org/abs/1903.00241).

Phần nghiên cứu có thể bảo vệ ở dự án là cách phân tích error theo quy mô/hình dạng và kiểm can thiệp có đối chứng, hoặc một nguyên lý ổn định/lựa chọn nhóm được chứng minh và kiểm nghiệm. Phải hoàn tất đối chiếu phương pháp cùng họ và dataset độc lập. Đổi thuật ngữ sang vật lý/toán cao cấp không tạo novelty.

### 4.5. “Không thua” cần kiểm định đúng mục tiêu

`02_MUC_TIEU_CONG_BO.md` dùng CI chứa 0 làm điều kiện “không thua”. Điều đó sai về logic kiểm định: không phát hiện khác biệt không chứng minh tương đương hoặc non-inferiority.

Nếu tuyên bố không thua quá delta, cần khai báo margin delta và thiết kế CI/kiểm định phù hợp; cận dưới của hiệu số TFN-baseline phải vượt -delta theo giao thức đã chọn. CI rộng chứa cả lợi ích và thiệt hại lớn chỉ là thiếu bằng chứng kết luận. CI sau thích nghi test còn không có phạm vi xác nhận như một phép thử đã chốt trên holdout mới.

## 5. Câu hỏi nghiên cứu đề xuất

“Trong phân đoạn tán cây từ RGB, tương tác giữa độ chính xác của trường dự đoán và quy tắc gom nhóm làm phát sinh lỗi gộp tán nhỏ, chia tán không lồi như thế nào; và có thể giảm các lỗi đó bằng một decoder ổn định với sai số trường, dưới cùng điều kiện huấn luyện và ngân sách lựa chọn không?”

Đây là đề xuất dựa trên số hiện có, không phải khẳng định toàn bộ lĩnh vực chưa ai trả lời. Nó cần literature search để chốt khoảng trống và phép xác nhận trên dữ liệu chưa dùng phát triển.

Các việc quyết định giá trị câu chuyện:

1. Chốt định nghĩa visible crown/annotation policy, phạm vi count và kích thước tối thiểu.
2. Sửa lập luận scale, multiple-attractor, ideal-field và các formulation tối ưu chưa khớp.
3. Sửa RNG và thiết kế baseline/selection công bằng; không ghi đè run cũ.
4. Tách ranking gain khỏi geometry/count gain; can thiệp từng field và kiểm robustness quanh saddle margin trên validation.
5. Đóng băng phương pháp, xác nhận trên holdout/domain chưa dùng phát triển, báo uncertainty/khả năng mơ hồ của nhãn.

Đánh giá cuối: **cách mô hình hóa có cơ sở; chuỗi chứng minh nguyên nhân hiện chưa chặt; một số mệnh đề cụ thể sai; kết quả cải thiện số là thật trong phạm vi đã kiểm, nhưng novelty và superiority chưa được xác nhận.**
