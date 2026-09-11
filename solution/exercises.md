# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Temperature thấp (0.0–0.5) khiến phản hồi ít biến thiên hơn, còn temperature cao (1.0–1.5) có thể làm cách diễn đạt và lựa chọn nội dung đa dạng hơn nhưng không đảm bảo chính xác hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ bắt đầu với temperature = 0.2 để chatbot trả lời ít biến thiên, phù hợp với yêu cầu nhất quán khi hỗ trợ khách hàng. Tuy nhiên, temperature thấp không đảm bảo thông tin chính xác; chatbot vẫn cần dựa trên tài liệu và chính sách đã được xác minh.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Theo bảng giá trong bài lab, tổng token đầu ra mỗi ngày là 10.000 × 3 × 350 = **10.500.000 token**. GPT-4o tốn khoảng **105 USD/ngày**, còn GPT-4o-mini tốn **6,30 USD/ngày**, nên GPT-4o đắt hơn khoảng **16,7 lần**, chỉ tính chi phí đầu ra.
>
> GPT-4o có thể xứng đáng với chi phí khi xử lý khiếu nại phức tạp, cần tổng hợp nhiều thông tin và vận dụng chính sách, nếu thử nghiệm cho thấy chất lượng tốt hơn rõ rệt. Với câu hỏi thường gặp như giờ mở cửa hoặc hướng dẫn đổi mật khẩu, nên dùng mini để tiết kiệm chi phí.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với persona giáo viên tiểu học, phản hồi thường ngắn gọn, dùng từ dễ hiểu và ví dụ gần gũi như một cuốn sổ được nhiều người cùng giữ. Với persona chuyên gia tài chính, phản hồi thường chi tiết hơn, dùng thuật ngữ như sổ cái phân tán, cơ chế đồng thuận và hàm băm, kèm ví dụ về giao dịch tài chính. System prompt định hướng vai trò, đối tượng người đọc và mức độ chuyên sâu, nên cùng một câu hỏi có thể nhận được cách giải thích khác nhau.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Đoạn văn có 95 đơn vị cách nhau bằng khoảng trắng; công thức chia cho 0,75 ước lượng được 126,67 token, còn count_tokens trả về 123 token, thấp hơn khoảng 2,89% so với ước lượng. Tokenizer chia văn bản theo các mẫu ký tự đã học, không đơn thuần theo từ hay khoảng trắng; tiếng Việt có dấu và các tiếng có thể bị tách thành nhiều token, nên có thể tốn nhiều token hơn tiếng Anh có độ dài tương đương, tùy tokenizer. Tuy nhiên, phép đo này chỉ so sánh hai cách đếm trên tiếng Việt, chưa chứng minh được chênh lệch với tiếng Anh.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming hữu ích nhất trong chatbot tương tác hoặc khi câu trả lời dài, vì người dùng có thể đọc phần đầu ngay thay vì chờ toàn bộ phản hồi hoàn tất. Nó giúp giảm cảm giác chờ đợi, dù không nhất thiết giảm tổng thời gian xử lý. Non-streaming phù hợp hơn với tác vụ chạy nền hoặc khi ứng dụng cần nhận đủ kết quả để kiểm tra, phân tích JSON hay xử lý tiếp trước khi hiển thị.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff tăng dần thời gian chờ sau mỗi lần thất bại, giúp giảm tần suất request và cho server thêm thời gian phục hồi khi quá tải. Nếu hàng nghìn client cùng retry theo khoảng chờ cố định và cùng thời điểm, chúng có thể tạo những đợt request dồn dập, khiến tình trạng quá tải kéo dài. Backoff theo cấp số nhân vẫn có thể bị đồng bộ giữa các client, nên thực tế thường thêm jitter — một khoảng ngẫu nhiên — để phân tán thời điểm retry.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Tôi chọn persona trợ giảng AI với system prompt: “Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt.” Cụm “trợ giảng thân thiện” định hướng cách giải thích dễ tiếp cận cho người đang học. Yêu cầu “ngắn gọn bằng tiếng Việt” giúp câu trả lời tập trung vào ý chính và phù hợp với ngôn ngữ học tập của tôi.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là trợ lý chỉ giữ ba lượt hội thoại gần nhất, nên có thể mất thông tin quan trọng người dùng cung cấp từ đầu phiên. Tôi đề xuất tóm tắt những lượt sắp bị loại bỏ và lưu vào một biến riêng. Mỗi request sẽ gửi kèm bản tóm tắt cùng ba lượt gần nhất; giới hạn độ dài bản tóm tắt giúp kiểm soát token, nhưng cần lưu ý việc tóm tắt có thể làm mất hoặc sai lệch thông tin.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
