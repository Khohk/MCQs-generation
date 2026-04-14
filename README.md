# MCQ Generator — Tạo câu hỏi trắc nghiệm tự động từ PDF Slides

> Thesis Project: PDF Slides → MCQ tự động với Gemini API + Gradio  
> Stack: Python · Gradio · Gemini API · PyMuPDF · pdfplumber

---

## Pipeline

```
PDF / PPTX / DOCX Slides
         │
         ▼
[1] file_router.py   →  Chọn parser phù hợp theo loại file
         │
         ▼
[2] pdf_parser.py    →  List[{page_num, title, text, char_count}]
    pptx_parser.py
    docx_parser.py
         │
         ▼
[3] chunker.py       →  List[{chunk_id, topic, pages, text}]
         │
         ▼
[4] generator.py     →  List[MCQ raw JSON]          ← cần Gemini API key
         │
         ▼
[5] validator.py     →  List[MCQ valid JSON]
         │
         ├──▶ [6] app.py / gradio_app.py   (Streamlit / Gradio UI)
         │
         ├──▶ [7] exporter.py              (Kahoot .xlsx / Quizizz .csv)
         │
         └──▶ [8] scorer.py               (BERTScore, BLEU, cosine sim)
```

| Bước | File | Cần API key? |
|------|------|-------------|
| 1 | file_router.py | Không |
| 2 | pdf/pptx/docx_parser.py | Không |
| 3 | chunker.py | Không |
| 4 | generator.py | **Có (Gemini)** |
| 5 | validator.py | Không |
| 6 | app.py / gradio_app.py | **Có (Gemini)** |
| 7 | exporter.py | Không |
| 8 | scorer.py | Không |

---

## Cài đặt

```bash
# Clone / mở thư mục project
cd Graduation_project

# Tạo file .env
echo "GEMINI_API_KEY=your_key_here" > .env

# Cài dependencies cơ bản
pip install -r requirements.txt


```

---

## Chạy nhanh

```bash
# Streamlit UI
streamlit run app.py

# Gradio UI
python gradio_app.py

# Chạy từng module standalone
python pipeline/pdf_parser.py    slides.pdf
python pipeline/chunker.py       slides.pdf
python pipeline/generator.py                 # test 1 chunk
python pipeline/validator.py                 # test với sample data
python pipeline/exporter.py                  # test export
```

---

## Cấu trúc thư mục

```
Graduation_project/
├── app.py                  # Streamlit UI (4 tab)
├── gradio_app.py           # Gradio UI
├── requirements.txt
├── .env                    # GEMINI_API_KEY (không commit)
│
├── pipeline/
│   ├── file_router.py      # Điều hướng parser theo loại file
│   ├── pdf_parser.py       # Parse PDF → List[page_dict]
│   ├── pptx_parser.py      # Parse PowerPoint
│   ├── docx_parser.py      # Parse Word
│   ├── chunker.py          # Chia chunk theo topic/title
│   ├── generator.py        # Gọi Gemini API sinh MCQ
│   ├── validator.py        # Lọc MCQ hợp lệ
│   └── exporter.py         # Xuất Kahoot .xlsx / Quizizz .csv
│
├── prompts/
│   └── mcq_prompt.py       # Prompt template (Bloom's Taxonomy)
│
├── evaluation/
│   └── scorer.py           # BERTScore, BLEU, NLI, cosine sim
│
└── data/                   # File mẫu, kết quả test
```

---

## MCQ JSON Schema

```json
{
  "question":    "...",
  "A":           "...",
  "B":           "...",
  "C":           "...",
  "D":           "...",
  "answer":      "B",
  "explanation": "...",
  "bloom_level": "understand",
  "difficulty":  "medium",
  "source_chunk":"chunk_003"
}
```

---



## Export Format

**Kahoot (.xlsx)**

| Question | Answer 1 | Answer 2 | Answer 3 | Answer 4 | Time | Correct Answer |
|----------|----------|----------|----------|----------|------|----------------|
| ...      | A text   | B text   | C text   | D text   | 30   | 1/2/3/4        |

**Quizizz (.csv)**
```
Question Text, Option 1, Option 2, Option 3, Option 4, Correct Answer, Time in seconds
```

---

*Graduation Project: MCQ Generator từ PDF Slides — Data Science*
