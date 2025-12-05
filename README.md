
# 📖 HƯỚNG DẪN SỬ DỤNG: AI FINANCIAL CONTROLLER
*(Hệ thống Kiểm soát Tài chính Thông minh)*

## 🔐 PHẦN 1: ĐĂNG NHẬP & PHÂN QUYỀN
Hệ thống này bảo mật 3 lớp. Tùy vào tài khoản đăng nhập mà Chị sẽ nhìn thấy các tính năng khác nhau.

**Danh sách tài khoản (Demo):**
1.  **`admin_cfo`** / mật khẩu: `mai_hanh_vip`
    *   *Quyền:* **CFO (Giám đốc Tài chính)**. Xem được TẤT CẢ (Bao gồm dự báo chiến lược).
2.  **`chief_acc`** / mật khẩu: `ketoantruong`
    *   *Quyền:* **Kế toán trưởng**. Xem được Rủi ro, Quản lý Luật. (Không xem dự báo chiến lược).
3.  **`staff_01`** / mật khẩu: `nv123`
    *   *Quyền:* **Nhân viên**. Chỉ xem Báo cáo chung và Chat.

---

## ⚙️ PHẦN 2: THIẾT LẬP DỮ LIỆU (SIDEBAR BÊN TRÁI)

Sau khi đăng nhập thành công, nhìn sang cột bên trái (Sidebar):

**1. Chọn Ngôn ngữ (Language):**
*   Bấm vào ô chọn 🌐. Chọn **Tiếng Việt**, **English**, hoặc **中文** (Tiếng Trung).
*   *Tác dụng:* Toàn bộ giao diện sẽ đổi ngôn ngữ ngay lập tức. (Dùng cái này để lòe sếp Trung Quốc rất hiệu quả).

**2. Chọn Nguồn Dữ liệu:**
*   **Cách 1: Dữ liệu Demo (Khuyên dùng khi Phỏng vấn):**
    *   Chọn vào nút tròn **"🎲 Dữ liệu Giả lập"**.
    *   Bấm nút **"Tạo dữ liệu mẫu"**.
    *   *Kết quả:* Hệ thống tự sinh ra một bảng doanh thu/chi phí giả để Chị thao tác ngay mà không cần file thật.
*   **Cách 2: Upload Excel Thực tế:**
    *   Chọn nút tròn **"📂 Upload Excel Thực tế"**.
    *   Kéo thả file Excel của công ty vào.
    *   *Lưu ý:* File Excel cần có các cột cơ bản như: *Ngày tháng, Doanh Thu, Chi Phí.*

---

## 🚀 PHẦN 3: HƯỚNG DẪN CHI TIẾT TỪNG TAB (THẺ)

### 📊 TAB 1: DASHBOARD (BÁO CÁO TỔNG QUAN)
*Dành cho: Tất cả mọi người.*

*   **Chức năng:** Xem nhanh Doanh thu, Chi phí, Lợi nhuận và Biểu đồ.
*   **TÍNH NĂNG "SÁT THỦ" (Chỉ Admin/Chief thấy):** Nút **"🇨🇳 Báo Cáo Sếp"**.
    *   **Cách dùng:** Bấm vào nút này.
    *   **Kết quả:** AI sẽ đóng vai Kế toán trưởng, viết một đoạn báo cáo ngắn gọn, chuyên nghiệp bằng **Tiếng Trung Thương mại** để Chị copy gửi Wechat cho sếp Tổng.

### 🕵️ TAB 2: SOI RỦI RO (RISK AUDIT)
*Dành cho: Kế toán trưởng & CFO. (Nhân viên không thấy).*

*   **Chức năng:** Tìm gian lận hoặc sai sót nhập liệu.
*   **Cách dùng:**
    1.  Bấm nút **"🔍 Quét Rủi Ro Ngay"**.
    2.  Hệ thống dùng thuật toán Học máy (Machine Learning) quét qua hàng nghìn dòng.
    3.  Nếu thấy khoản chi nào bất thường (ví dụ: quá lớn so với trung bình, hoặc số tiền lạ), nó sẽ bôi đỏ và hiện ra bảng cảnh báo.
    4.  AI sẽ tự động đưa ra nhận định: *"Có thể do nhập thừa số 0 hoặc đây là khoản chi mùa vụ..."*

### 🔮 TAB 3: DỰ BÁO (FORECAST)
*Dành cho: Chỉ CFO (Admin).*

*   **Chức năng:** Nhìn về tương lai.
*   **Cách dùng:** Vào Tab này, Chị sẽ thấy biểu đồ có các chấm xanh và đường kẻ đỏ.
    *   Đó là đường xu hướng (Trendline) do máy tính vẽ.
    *   Nó sẽ dự báo chính xác dòng tiền của **tháng sau, 2 tháng sau, 3 tháng sau**.
*   **Tác dụng:** Giúp Chị trả lời câu hỏi của Sếp: *"Tháng sau công ty có thiếu tiền mặt không?"*

### 💬 TAB 4: CHAT TÀI CHÍNH
*Dành cho: Tất cả mọi người.*

*   **Chức năng:** Hỏi đáp nhanh.
*   **Cách dùng:**
    1.  Gõ câu hỏi vào ô chat: *"Lợi nhuận tháng này so với tháng trước thế nào?"*
    2.  AI sẽ đọc số liệu và trả lời ngay lập tức.

---

### 📚 TAB 5: THƯ VIỆN LUẬT & CẢNH BÁO (QUAN TRỌNG NHẤT)
*Dành cho: Kế toán trưởng (Chief) & CFO.*

Đây là nơi Chị quản lý tính Pháp lý. Tab này có 2 khu vực:

#### KHU VỰC 1: BẢNG QUẢN LÝ HIỆU LỰC (Ở TRÊN CÙNG)
Đây là cái bảng Chị nhìn thấy ngay khi vào Tab. **Chị có thể sửa trực tiếp vào bảng này.**

*   **Nhiệm vụ của Chị:** Nhập tên các văn bản luật quan trọng vào đây.
*   **Cột quan trọng:**
    *   `Ten_Van_Ban`: Tên nghị định/thông tư (VD: *Nghị định 51*).
    *   `Trang_Thai`: Chị gõ vào là **"Hiệu lực"** hoặc **"Hết hiệu lực"**.
    *   `Thay_The_Boi`: Nếu hết hiệu lực, thì văn bản mới thay thế là gì? (VD: *Nghị định 123*).

#### KHU VỰC 2: TRA CỨU & HỎI ĐÁP (Ở DƯỚI)
*   **Bước 1 (Upload):** Chị có file PDF luật mới (VD: Nghị định 123.pdf)? Kéo thả vào ô Upload. AI sẽ đọc nội dung file đó.
*   **Bước 2 (Hỏi):** Chị gõ câu hỏi: *"Quy định về hóa đơn theo Nghị định 51 là gì?"*
*   **Bước 3 (Bấm nút):** Bấm **"Kiểm tra hiệu lực & Hỏi AI"**.

👉 **KẾT QUẢ "THẦN KỲ":**
1.  AI sẽ trả lời nội dung về Nghị định 51 (dựa trên kiến thức của nó).
2.  **ĐỒNG THỜI**, Hệ thống sẽ quét cái Bảng ở Khu vực 1.
3.  Nó thấy *Nghị định 51* có trạng thái là *"Hết hiệu lực"*.
4.  Nó hiện ngay một khung **MÀU ĐỎ**: *"🚨 CẢNH BÁO: Nghị định 51 đã hết hiệu lực! Hãy dùng Nghị định 123."*

**Tóm lại:** Upload file để AI có kiến thức trả lời. Nhập bảng để Hệ thống có kiến thức cảnh báo.

