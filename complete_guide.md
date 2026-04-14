# MCQ Generator — Complete Build Guide
> Thesis Project: PDF Slides → MCQ tự động với Gemini API + Streamlit  
> Stack: Python · Streamlit · Gemini API · PyMuPDF · pdfplumber

---

## Mục lục

1. [Setup môi trường](#bước-0-setup-môi-trường)
2. [pdf_parser.py](#bước-1-pdf_parserpy)
3. [chunker.py](#bước-2-chunkerpy)
4. [prompts/ + generator.py](#bước-3-prompts--generatorpy)
5. [validator.py](#bước-4-validatorpy)
6. [app.py — Streamlit UI](#bước-5-apppy--streamlit-ui)
7. [exporter.py](#bước-6-exporterpy)
8. [scorer.py — Evaluation](#bước-7-scorerpy--evaluation)

---

## Tổng quan Pipeline

```
PDF Slides
   │
   ▼
[1] pdf_parser.py   →  List[{page_num, title, text, char_count}]
   │
   ▼
[2] chunker.py      →  List[{chunk_id, topic, pages, text}]
   │
   ▼
[3] generator.py    →  List[MCQ raw JSON]          ← cần Gemini API key
   │
   ▼
[4] validator.py    →  List[MCQ valid JSON]
   │
   ├──▶ [5] app.py       (Streamlit UI 4 tab)
   │
   ├──▶ [6] exporter.py  (Kahoot .xlsx / Quizizz .csv)
   │
   └──▶ [7] scorer.py    (BERTScore, BLEU, cosine sim)
```

| Bước | File | Cần API key? |
|------|------|-------------|
| 0 | Setup | Không |
| 1 | pdf_parser.py | Không |
| 2 | chunker.py | Không |
| 3 | generator.py | **Có (Gemini)** |
| 4 | validator.py | Không |
| 5 | app.py | **Có (Gemini)** |
| 6 | exporter.py | Không |
| 7 | scorer.py | Không |

---

## Bước 0: Setup môi trường

### Tạo cấu trúc thư mục

```bash
mkdir -p mcq_generator/{pipeline,prompts,evaluation}
cd mcq_generator

touch app.py
touch pipeline/__init__.py pipeline/pdf_parser.py pipeline/chunker.py
touch pipeline/generator.py pipeline/validator.py pipeline/exporter.py
touch prompts/__init__.py prompts/mcq_prompt.py
touch evaluation/__init__.py evaluation/scorer.py
echo "GEMINI_API_KEY=your_key_here" > .env
```

### Cài dependencies (phase 1 — chưa cần torch)

```bash
pip install -r requirements.txt
```

### ✅ Kiểm tra setup

| Kiểm tra | Lệnh | Kết quả mong đợi |
|----------|------|-----------------|
| Thư mục đúng | `ls` | Có `pipeline/`, `prompts/`, `evaluation/`, `.env` |
| Import packages | `python -c "import fitz, pdfplumber, streamlit"` | Không có exception |
| .env tồn tại | `cat .env` | Có dòng `GEMINI_API_KEY=...` |

---

## Bước 1: `pdf_parser.py`

**Input:** đường dẫn file PDF  
**Output:** `List[dict]` — mỗi dict là 1 trang

### Output schema

```json
{
  "page_num": 3,
  "title": "Gradient Descent",
  "text": "Gradient Descent là thuật toán tối ưu...\nw = w - lr * grad",
  "char_count": 312
}
```

### Logic cần implement

- Dùng **PyMuPDF** (`fitz`) làm primary extractor
- Fallback sang **pdfplumber** nếu page trả về text rỗng
- **Detect title:** text block đầu tiên có font size lớn nhất
- **Filter trang rác:** `char_count < 50` → bỏ qua (trang trống, chỉ có ảnh)
- Trả về list đã sort theo `page_num`

### Test script

Thêm vào cuối `pdf_parser.py`:

```python
if __name__ == "__main__":
    import sys, json
    pages = parse_pdf(sys.argv[1])
    print(f"Total pages parsed: {len(pages)}")
    for p in pages[:5]:
        print(f"  [{p['page_num']:02d}] {p['title'][:50]:<50} | {p['char_count']} chars")
    print("\n--- Sample text (page 1) ---")
    if pages:
        print(pages[0]['text'][:300])
```

```bash
python pipeline/pdf_parser.py path/to/slides.pdf
```

### ✅ Kiểm tra output

| Kiểm tra | Kết quả mong đợi | Cách check |
|----------|-----------------|------------|
| Import ok | Không exception | `python -c "from pipeline.pdf_parser import parse_pdf"` |
| Số trang hợp lý | Gần bằng số slide thực tế | In `len(pages)` |
| Title detect | Mỗi page có title khác nhau, không rỗng | In `p['title']` cho 5 page đầu |
| Không có trang rác | Không page nào `char_count < 50` | `min(p['char_count'] for p in pages)` |
| Encoding đúng | Tiếng Việt / ký tự đặc biệt hiển thị đúng | In thử text page có dấu |
| Fallback hoạt động | Page dùng pdfplumber vẫn có text | Log library nào được dùng per page |

### Dấu hiệu lỗi thường gặp

- **text rỗng nhiều trang** → PDF là scan (ảnh), cần OCR — thêm `pytesseract` vào pipeline
- **title toàn None** → logic detect font size chưa đúng, debug `page.get_text("dict")` để xem blocks
- **UnicodeDecodeError** → thêm `errors="ignore"` khi join text blocks

---

## Bước 2: `chunker.py`

**Input:** `List[page_dict]` từ bước 1  
**Output:** `List[chunk_dict]`

### Output schema

```json
{
  "chunk_id": "chunk_003",
  "topic": "Gradient Descent",
  "pages": [3, 4, 5],
  "text": "Gradient Descent là...\nLearning rate...\nMomentum..."
}
```

### Chiến lược chunking

| Chiến lược | Khi nào dùng | Ưu điểm |
|------------|-------------|---------|
| **Fixed-size** (N trang/chunk) | PDF không có cấu trúc rõ | Đơn giản, dễ debug |
| **Title-based** (group theo title giống nhau) | Slide có section heading | Chunk ngữ nghĩa tốt |
| **Semantic** (cosine sim giữa các page) | Cần chất lượng cao | Chính xác nhất, phức tạp hơn |

**Khuyến nghị cho thesis:** bắt đầu với **title-based**, fallback sang fixed-size (3–5 trang/chunk) nếu title không detect được.

### Tham số cần expose

```python
def chunk_pages(
    pages: list[dict],
    strategy: str = "title",   # "title" | "fixed" | "semantic"
    max_pages: int = 5,        # max số trang trong 1 chunk
    min_chars: int = 200,      # chunk phải có ít nhất N chars
) -> list[dict]:
```

### Test script

Thêm vào cuối `chunker.py`:

```python
if __name__ == "__main__":
    import sys
    from pipeline.pdf_parser import parse_pdf
    pages = parse_pdf(sys.argv[1])
    chunks = chunk_pages(pages)
    print(f"Pages: {len(pages)}  →  Chunks: {len(chunks)}")
    for c in chunks[:5]:
        print(f"  [{c['chunk_id']}] topic='{c['topic'][:40]}' | pages={c['pages']} | {len(c['text'])} chars")
```

```bash
python pipeline/chunker.py path/to/slides.pdf
```

### ✅ Kiểm tra output

| Kiểm tra | Kết quả mong đợi | Cách check |
|----------|-----------------|------------|
| Số chunks hợp lý | 15–50% số trang gốc (vì gom lại) | In tỉ lệ `len(chunks)/len(pages)` |
| Không chunk rỗng | Mọi chunk có `len(text) >= min_chars` | `min(len(c['text']) for c in chunks)` |
| chunk_id unique | Không có 2 chunk trùng id | `len(set(c['chunk_id'] for c in chunks)) == len(chunks)` |
| Pages không chồng | Mỗi trang chỉ thuộc 1 chunk | Collect tất cả pages, check không duplicate |
| Topic có nghĩa | Topic phản ánh nội dung chunk | Đọc 5 chunk đầu, verify manually |

### Dấu hiệu lỗi thường gặp

- **Chunk quá nhỏ** (< 200 chars) → merge với chunk liền kề
- **Chunk quá lớn** (> 2000 chars) → Gemini có thể bị context quá dài → thêm `max_pages` limit
- **Tất cả pages vào 1 chunk** → title detect không hoạt động, switch sang fixed-size

---

## Bước 3: `prompts/` + `generator.py`

**Input:** `List[chunk_dict]`  
**Output:** `List[MCQ raw JSON]`

> ⚠️ **Bước này cần Gemini API key** trong `.env`

### MCQ JSON schema (target)

```json
{
  "question": "...",
  "A": "...",
  "B": "...",
  "C": "...",
  "D": "...",
  "answer": "B",
  "explanation": "...",
  "bloom_level": "understand",
  "difficulty": "medium",
  "source_chunk": "chunk_003"
}
```

### `prompts/mcq_prompt.py` — kỹ thuật prompt

Prompt cần include:
- **Bloom's Taxonomy instruction:** map `difficulty` → bloom levels
  - `easy` → remember, understand
  - `medium` → apply, analyze
  - `hard` → evaluate, create
- **Distractor quality control:** distractors phải cùng loại khái niệm với answer, plausible nhưng sai
- **Format instruction:** trả về JSON array, không có text thừa
- **Few-shot example:** 1 MCQ mẫu đúng format

### `generator.py` — flow gọi API

```
chunk → build_prompt(chunk) → call Gemini API → parse JSON → tag source_chunk
```

Gọi với `response_mime_type="application/json"` để Gemini trả về JSON clean.

### Test script

```python
if __name__ == "__main__":
    # Test với 1 chunk giả
    test_chunk = {
        "chunk_id": "chunk_001",
        "topic": "Gradient Descent",
        "pages": [3, 4],
        "text": "Gradient Descent cập nhật tham số theo hướng ngược gradient..."
    }
    mcqs = generate_mcqs([test_chunk], n_questions=2)
    import json
    print(json.dumps(mcqs, ensure_ascii=False, indent=2))
```

```bash
python pipeline/generator.py  # test với 1 chunk
```

### ✅ Kiểm tra output

| Kiểm tra | Kết quả mong đợi | Cách check |
|----------|-----------------|------------|
| API call thành công | Không raise exception | Chạy test script |
| JSON parse được | `json.loads()` không lỗi | Wrap trong try/except, log raw response nếu fail |
| Có đủ 9 fields | question, A, B, C, D, answer, explanation, bloom_level, difficulty, source_chunk | Check keys |
| answer hợp lệ | Giá trị là A, B, C, hoặc D | `mcq['answer'] in ['A','B','C','D']` |
| bloom_level hợp lệ | 1 trong 6 level của Bloom | Check against whitelist |
| source_chunk đúng | Trùng với chunk_id đầu vào | Compare |
| Distractor plausible | Đọc thủ công 5 câu | Manual review |

### Dấu hiệu lỗi thường gặp

- **JSONDecodeError** → Gemini trả về markdown (```json ...```), cần strip trước khi parse
- **answer = "A" tất cả** → prompt bị bias, thêm instruction "distribute answers evenly"
- **Rate limit** → thêm `time.sleep(1)` giữa các chunk, hoặc batch processing

---

## Bước 4: `validator.py`

**Input:** `List[MCQ raw JSON]` (có thể thiếu field, sai format)  
**Output:** `List[MCQ valid JSON]` (đã lọc sạch)

### Validation rules

```
1. Có đủ 10 fields bắt buộc
2. answer ∈ {A, B, C, D}
3. bloom_level ∈ {remember, understand, apply, analyze, evaluate, create}
4. difficulty ∈ {easy, medium, hard}
5. question.strip() != "" và len > 10
6. A, B, C, D đều không rỗng
7. explanation.strip() != ""
8. Không duplicate (so sánh question text)
```

### Test script

```python
if __name__ == "__main__":
    # Test với data giả — mix valid + invalid
    raw = [
        {"question": "...", "A": "...", "B": "...", "C": "...", "D": "...",
         "answer": "B", "explanation": "...", "bloom_level": "understand",
         "difficulty": "medium", "source_chunk": "chunk_001"},
        {"question": "", "A": "...", "answer": "X"},  # invalid
    ]
    valid, invalid = validate_mcqs(raw)
    print(f"Valid: {len(valid)} | Invalid: {len(invalid)}")
    for inv in invalid:
        print(f"  REJECTED: {inv['reason']} | data: {inv['mcq']}")
```

### ✅ Kiểm tra output

| Kiểm tra | Kết quả mong đợi | Cách check |
|----------|-----------------|------------|
| Valid rate hợp lý | > 80% MCQ pass validation | `len(valid)/len(raw)` |
| Rejected có reason | Mỗi rejected MCQ có field `reason` | In ra invalid list |
| Không duplicate | Không có 2 MCQ trùng question | Check length sau dedup |
| Schema clean | Tất cả valid MCQ có đủ 10 fields | `all(len(m)==10 for m in valid)` |

---

## Bước 5: `app.py` — Streamlit UI

**4 tab:** Upload → Processing → Review → Export

### Tab structure

```
Tab 1 — Upload
  ├── File uploader (PDF)
  ├── Cấu hình: strategy (title/fixed), difficulty mix, n_questions per chunk
  └── Nút "Start Processing"

Tab 2 — Processing
  ├── Progress bar: Parse → Chunk → Generate → Validate
  ├── Log từng bước (st.expander)
  └── Thống kê: X pages → Y chunks → Z MCQs generated → W valid

Tab 3 — Review
  ├── Filter theo: bloom_level, difficulty, chunk/topic
  ├── Hiển thị từng MCQ (question + 4 options, highlight answer)
  ├── Nút Edit inline (optional)
  └── Nút Delete MCQ

Tab 4 — Export
  ├── Nút "Download Kahoot (.xlsx)"
  ├── Nút "Download Quizizz (.csv)"
  └── Preview bảng MCQ
```

### Session state cần manage

```python
st.session_state.pages    # output bước 1
st.session_state.chunks   # output bước 2
st.session_state.mcqs     # output bước 4 (validated)
```

### ✅ Kiểm tra UI

| Kiểm tra | Kết quả mong đợi |
|----------|-----------------|
| Upload PDF | File được nhận, hiển thị tên + số trang |
| Processing flow | Progress bar chạy đúng thứ tự 4 bước |
| Review tab | MCQ hiển thị đẹp, filter hoạt động |
| Export tab | File download được, mở được trong Excel |
| Error handling | Upload file không phải PDF → show warning |
| Session persist | Chuyển tab không mất data |

```bash
streamlit run app.py
# Mở http://localhost:8501
```

---

## Bước 6: `exporter.py`

**Input:** `List[MCQ valid JSON]`  
**Output:** Kahoot `.xlsx` hoặc Quizizz `.csv`

### Kahoot format (Excel)

| Question | Answer 1 | Answer 2 | Answer 3 | Answer 4 | Time | Correct Answer |
|----------|----------|----------|----------|----------|------|----------------|
| ... | A text | B text | C text | D text | 30 | 1 hoặc 2 hoặc 3 hoặc 4 |

> `Correct Answer` là số thứ tự (1-indexed) của đáp án đúng.

### Quizizz format (CSV)

```
Question Text, Option 1, Option 2, Option 3, Option 4, Correct Answer, Time in seconds
```

### Test script

```python
if __name__ == "__main__":
    import json
    with open("sample_mcqs.json") as f:
        mcqs = json.load(f)
    export_kahoot(mcqs, "output_kahoot.xlsx")
    export_quizizz(mcqs, "output_quizizz.csv")
    print("Files created!")
```

### ✅ Kiểm tra output

| Kiểm tra | Kết quả mong đợi | Cách check |
|----------|-----------------|------------|
| File tạo được | Không exception | Chạy test script |
| Kahoot mở được | Excel không báo lỗi | Mở file trong Excel / Google Sheets |
| Quizizz import được | Upload lên quizizz.com thành công | Test thật trên platform |
| Correct answer đúng | Đáp án map đúng A/B/C/D → 1/2/3/4 | Verify thủ công 5 câu |
| Không mất MCQ | Số row = số MCQ | `len(df) == len(mcqs)` |

---

## Bước 7: `scorer.py` — Evaluation

> ⚠️ **Cài thêm dependencies nặng** trước bước này:
> ```bash
> pip install torch bert-score sentence-transformers rouge-score nltk
> ```

**Input:** `List[MCQ valid JSON]` + optional: ground truth  
**Output:** Dict scores per MCQ + aggregate statistics

### 4 metrics cần implement

#### 1. BERTScore — question relevance
Đo mức độ câu hỏi liên quan đến source chunk text.

```python
from bert_score import score
P, R, F1 = score(
    cands=[mcq['question'] for mcq in mcqs],
    refs=[get_chunk_text(mcq['source_chunk']) for mcq in mcqs],
    lang="en"
)
```

#### 2. BLEU / ROUGE — text overlap
Dùng làm baseline so sánh question vs source text.

```python
from rouge_score import rouge_scorer
scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer_obj.score(reference_text, generated_question)
```

#### 3. NLI-based answerability
Kiểm tra answer có thể được infer từ source chunk không (entailment).

```python
from sentence_transformers import CrossEncoder
nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-small')
# premise = source_chunk text, hypothesis = question + " " + correct_answer_text
score = nli_model.predict([(premise, hypothesis)])
```

#### 4. Cosine similarity — distractor plausibility
Distractor nên gần answer trong embedding space (cùng domain) nhưng không quá giống.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([answer_text, distractor_A, distractor_B, distractor_C])
# Cosine sim(answer, distractor) nên nằm trong [0.3, 0.8]
# < 0.3 → distractor quá khác (dễ loại)
# > 0.8 → distractor quá giống (ambiguous)
```

### Output schema của scorer

```json
{
  "mcq_id": "chunk_001_q0",
  "bert_score_f1": 0.82,
  "rouge_l": 0.41,
  "nli_entailment": 0.91,
  "distractor_cosine_mean": 0.54,
  "distractor_cosine_min": 0.38,
  "distractor_cosine_max": 0.71,
  "overall_quality": "good"
}
```

### ✅ Kiểm tra output

| Kiểm tra | Kết quả mong đợi | Threshold tham khảo |
|----------|-----------------|---------------------|
| BERTScore F1 | MCQ liên quan đến chunk | > 0.60 |
| ROUGE-L | Có overlap với source | > 0.15 |
| NLI entailment | Answer derive được từ source | > 0.70 |
| Distractor cosine | Plausible nhưng không ambiguous | 0.30 – 0.80 |

### Test script

```python
if __name__ == "__main__":
    import json
    with open("sample_mcqs.json") as f:
        mcqs = json.load(f)
    with open("sample_chunks.json") as f:
        chunks = {c['chunk_id']: c['text'] for c in json.load(f)}

    results = score_mcqs(mcqs, chunks)
    
    # Aggregate stats
    avg_bert = sum(r['bert_score_f1'] for r in results) / len(results)
    avg_cosine = sum(r['distractor_cosine_mean'] for r in results) / len(results)
    print(f"Avg BERTScore F1: {avg_bert:.3f}")
    print(f"Avg Distractor Cosine: {avg_cosine:.3f}")
    print(f"Quality distribution:")
    from collections import Counter
    print(Counter(r['overall_quality'] for r in results))
```

---

## Checklist tổng kết

| # | File | Done | Test pass |
|---|------|------|-----------|
| 0 | Setup môi trường | ☐ | ☐ |
| 1 | pdf_parser.py | ☐ | ☐ |
| 2 | chunker.py | ☐ | ☐ |
| 3 | prompts/ + generator.py | ☐ | ☐ |
| 4 | validator.py | ☐ | ☐ |
| 5 | app.py (Streamlit) | ☐ | ☐ |
| 6 | exporter.py | ☐ | ☐ |
| 7 | scorer.py | ☐ | ☐ |

---

## Quick Commands Reference

```bash
# Run từng module standalone
python pipeline/pdf_parser.py slides.pdf
python pipeline/chunker.py slides.pdf
python pipeline/generator.py          # test 1 chunk
python pipeline/validator.py          # test với sample data
python pipeline/exporter.py           # test export

# Run full app
streamlit run app.py

# Install phase 1 (không torch)
pip install -r requirements.txt

# Install phase 2 (scorer — nặng ~2GB)
pip install torch bert-score sentence-transformers rouge-score nltk
```

---

*Generated for thesis: MCQ Generator từ PDF Slides — Data Science Graduation Project*