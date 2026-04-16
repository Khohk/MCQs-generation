## Rubric chi tiết để nhúng vào system prompt

Relevance:
- 5: Câu hỏi hỏi trực tiếp về khái niệm/thông tin chính trong source
- 4: Bám sát nội dung, hỏi về chi tiết quan trọng
- 3: Liên quan nhưng hỏi về nội dung phụ hoặc ví dụ nhỏ
- 2: Liên quan mờ nhạt, phần lớn không có trong source
- 1: Không liên quan hoặc hallucination — hỏi thứ không có trong source

Answerability:
- 5: Đọc source là trả lời được chắc chắn, không cần kiến thức ngoài
- 4: Trả lời được với một chút suy luận từ source
- 3: Cần suy luận đáng kể hoặc kiến thức nền nhẹ
- 2: Source có gợi ý nhưng không đủ để trả lời chắc chắn
- 1: Source không có đủ thông tin để trả lời

Bloom Alignment:
- 5: bloom_level khớp hoàn toàn với cognitive demand thực tế của câu hỏi
- 4: Lệch nhẹ (ví dụ: gán understand nhưng thực tế là remember)
- 3: Lệch 1 bậc rõ ràng
- 2: Lệch 2 bậc
- 1: Lệch hoàn toàn (ví dụ: gán evaluate nhưng chỉ là remember)

Answer Correctness:
- 5: Đáp án được gán hoàn toàn chính xác theo source
- 4: Đúng nhưng explanation chưa đầy đủ
- 3: Đáp án đúng nhưng có thể tranh luận
- 2: Đáp án có thể sai hoặc misleading
- 1: Đáp án sai so với source