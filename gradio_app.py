"""
gradio_app.py
-------------
MCQ Generator — Gradio web app
Deploy: Hugging Face Spaces (SDK: gradio)

Chạy local:
  pip install gradio
  python gradio_app.py

Deploy HF Spaces:
  Tạo Space → SDK: Gradio → push code lên
  Thêm GEMINI_API_KEY vào Settings → Secrets
"""

import os
import sys
import json
import tempfile
import time
from pathlib import Path

import gradio as gr

# ── i18n ──────────────────────────────────────────────────────────
I18N_DIR = Path(__file__).parent / "i18n"

def load_lang(lang: str) -> dict:
    path = I18N_DIR / f"{lang}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)

LANGS = {"vi": load_lang("vi"), "en": load_lang("en")}

def t(lang: str, *keys) -> str:
    """Lấy string theo key path, ví dụ t('vi', 'upload', 'tab')"""
    obj = LANGS[lang]
    for k in keys:
        obj = obj.get(k, k)
    return obj


# ── Density → n_questions mapping ────────────────────────────────
DENSITY_MAP = {"low": 1, "Ít": 1, "Low": 1,
               "medium": 2, "Vừa": 2, "Medium": 2,
               "high": 3, "Nhiều": 3, "High": 3}

DIFFICULTY_MAP = {"Dễ": "easy", "Easy": "easy",
                  "Trung bình": "medium", "Medium": "medium",
                  "Khó": "hard", "Hard": "hard"}

LANG_MAP = {"Tiếng Việt": "vi", "Vietnamese": "vi",
            "English": "en", "EN": "en"}


# ── Backend helpers ───────────────────────────────────────────────

def estimate_questions(files, density_label: str) -> str:
    """Ước tính số câu dựa trên file đã upload."""
    if not files:
        return ""
    n_per_chunk = DENSITY_MAP.get(density_label, 2)
    total_pages = 0
    for f in files:
        try:
            from pipeline.file_router import get_metadata
            meta = get_metadata(f.name)
            total_pages += meta.get("total_pages", 5)
        except Exception:
            total_pages += 5
    avg_pages_per_chunk = 4
    chunks = max(1, total_pages // avg_pages_per_chunk)
    estimated = chunks * n_per_chunk
    return f"~{estimated}"


def run_pipeline(files, difficulty_label, density_label, q_lang_label, lang,
                 on_step=None):
    """
    Chay toan bo pipeline, tra ve (mcqs_json, error_msg).
    on_step(pct, msg): callback update UI theo tung buoc.
    """
    def step(pct, msg):
        if on_step:
            on_step(pct, msg)

    if not files:
        return None, t(lang, "upload", "no_file")

    difficulty  = DIFFICULTY_MAP.get(difficulty_label, "medium")
    n_per_chunk = DENSITY_MAP.get(density_label, 2)
    all_pages   = []

    # Step 1: Parse
    step(0.1, "Dang doc tai lieu...")
    try:
        from pipeline.file_router import parse_file
        for f in files:
            pages = parse_file(f.name)
            all_pages.extend(pages)
    except Exception as e:
        return None, f"Loi parse file: {e}"

    # Step 2: Chunk
    step(0.3, "Dang phan tich cau truc...")
    try:
        from pipeline.chunker import chunk_pages
        chunks = chunk_pages(all_pages)
    except Exception as e:
        return None, f"Loi chunking: {e}"

    # Step 3: Generate
    step(0.5, f"Dang tao cau hoi tu {len(chunks)} chu de...")
    try:
        from pipeline.generator import generate_mcqs
        pdf_name = Path(files[0].name).stem if files else "upload"
        raw_mcqs = generate_mcqs(
            chunks,
            n_per_chunk=n_per_chunk,
            difficulty=difficulty,
            pdf_name=pdf_name,
        )
    except Exception as e:
        return None, f"Loi generate: {e}"

    # Step 4: Validate
    step(0.9, "Dang kiem tra chat luong...")
    try:
        from pipeline.validator import validate_mcqs
        valid, _ = validate_mcqs(raw_mcqs)
    except Exception as e:
        return None, f"Loi validate: {e}"

    step(1.0, f"Hoan tat! {len(valid)} cau hoi da san sang.")
    return json.dumps(valid, ensure_ascii=False), None


# ── MCQ card HTML ─────────────────────────────────────────────────

def render_mcq_card(mcq: dict, idx: int, lang: str) -> str:
    """Render 1 MCQ thành HTML card."""
    diff = mcq.get("difficulty", "medium")
    bloom = mcq.get("bloom_level", "")
    q = mcq.get("question", "")
    answer = mcq.get("answer", "A")
    explanation = mcq.get("explanation", "")

    diff_vi = {"easy": "Dễ", "medium": "Trung bình", "hard": "Khó"}.get(diff, diff)
    bloom_vi = {"remember":"Ghi nhớ","understand":"Hiểu","apply":"Vận dụng",
                "analyze":"Phân tích","evaluate":"Đánh giá","create":"Sáng tạo"}.get(bloom, bloom)
    tag_label = bloom_vi if lang == "vi" else bloom.capitalize()
    diff_color = {"easy": "#27AE60", "medium": "#F39C12", "hard": "#E74C3C"}.get(diff, "#666")

    options_html = ""
    for opt in ["A", "B", "C", "D"]:
        text = mcq.get(opt, "")
        is_correct = opt == answer
        if is_correct:
            bg, border, text_color, weight = "#1a4731", "#27AE60", "#6ee7b7", "600"
            prefix = "[Dung] "
        else:
            bg, border, text_color, weight = "#2d2d2d", "#444", "#e5e7eb", "400"
            prefix = ""
        options_html += f"""
        <div style="padding:8px 12px;margin:4px 0;border-radius:8px;
                    background:{bg};border:1.5px solid {border};
                    font-weight:{weight};font-size:13px;color:{text_color};">
            {prefix}<b>{opt}.</b> {text}
        </div>"""

    return f"""
    <div style="border:1px solid #444;border-radius:12px;padding:16px;
                margin-bottom:12px;background:#1e1e2e;">
        <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
            <span style="font-weight:600;color:#e5e7eb;">Cau {idx+1}</span>
            <span style="background:{diff_color};color:white;padding:2px 10px;
                         border-radius:10px;font-size:12px;">{tag_label}</span>
        </div>
        <p style="font-size:14px;font-weight:500;margin:0 0 10px;color:#f3f4f6;">{q}</p>
        {options_html}
        <details style="margin-top:10px;">
            <summary style="font-size:12px;color:#9ca3af;cursor:pointer;">
                Giai thich
            </summary>
            <p style="font-size:12px;color:#d1d5db;margin:6px 0 0;
                      padding:8px;background:#2d2d2d;border-radius:6px;">
                {explanation}
            </p>
        </details>
    </div>"""


def render_review(mcqs_json: str, lang: str,
                  filter_diff: str, search: str) -> str:
    """Render toàn bộ review tab từ MCQ JSON."""
    if not mcqs_json:
        return f"<p style='color:#888;text-align:center;padding:40px'>{t(lang,'review','no_mcq')}</p>"

    try:
        mcqs = json.loads(mcqs_json)
    except Exception:
        return "<p>Lỗi parse MCQ.</p>"

    # Filter
    filtered = mcqs
    if filter_diff and filter_diff != t(lang, "review", "all"):
        diff_key = DIFFICULTY_MAP.get(filter_diff, filter_diff)
        filtered = [m for m in filtered if m.get("difficulty") == diff_key]
    if search:
        search_lower = search.lower()
        filtered = [m for m in filtered if search_lower in m.get("question","").lower()]

    if not filtered:
        return "<p style='color:#888;text-align:center;padding:20px'>Không tìm thấy câu hỏi phù hợp.</p>"

    cards = "".join(render_mcq_card(m, i, lang) for i, m in enumerate(filtered))
    return f'<div style="max-height:70vh;overflow-y:auto;padding:4px">{cards}</div>'


def get_stats_html(mcqs_json: str, lang: str) -> str:
    """Header stats: X câu, phân bố easy/medium/hard."""
    if not mcqs_json:
        return ""
    try:
        mcqs = json.loads(mcqs_json)
    except Exception:
        return ""

    total = len(mcqs)
    easy   = sum(1 for m in mcqs if m.get("difficulty") == "easy")
    medium = sum(1 for m in mcqs if m.get("difficulty") == "medium")
    hard   = sum(1 for m in mcqs if m.get("difficulty") == "hard")

    header = t(lang, "review", "header").format(n=total)
    return f"""
    <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;
                padding:12px 16px;background:#F0F4FF;border-radius:10px;margin-bottom:12px;">
        <span style="font-size:18px;font-weight:600;color:#1E3A5F;">{header}</span>
        <span style="background:#27AE60;color:white;padding:3px 10px;
                     border-radius:8px;font-size:12px;">Dễ: {easy}</span>
        <span style="background:#F39C12;color:white;padding:3px 10px;
                     border-radius:8px;font-size:12px;">TB: {medium}</span>
        <span style="background:#E74C3C;color:white;padding:3px 10px;
                     border-radius:8px;font-size:12px;">Khó: {hard}</span>
    </div>"""


# ── Export helpers ─────────────────────────────────────────────────

def export_kahoot(mcqs_json: str):
    if not mcqs_json:
        return None
    mcqs = json.loads(mcqs_json)
    from pipeline.exporter import export_kahoot as _kahoot
    data = _kahoot(mcqs)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    tmp.write(data)
    tmp.close()
    return tmp.name

def export_quizizz(mcqs_json: str):
    if not mcqs_json:
        return None
    mcqs = json.loads(mcqs_json)
    from pipeline.exporter import export_quizizz as _quizizz
    data = _quizizz(mcqs)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(data)
    tmp.close()
    return tmp.name

def export_json_file(mcqs_json: str):
    if not mcqs_json:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json",
                                      mode="w", encoding="utf-8")
    tmp.write(mcqs_json)
    tmp.close()
    return tmp.name


# ── Quiz Player helpers ───────────────────────────────────────────

def render_quiz_question(mcqs_json: str, idx: int) -> tuple:
    """Render 1 câu hỏi cho quiz player. Returns (html, idx, total)."""
    if not mcqs_json:
        return "<p>Không có câu hỏi.</p>", 0, 0, gr.update(visible=False), gr.update(visible=False)
    try:
        mcqs = json.loads(mcqs_json)
    except Exception:
        return "<p>Lỗi.</p>", 0, 0, gr.update(visible=False), gr.update(visible=False)

    total = len(mcqs)
    if total == 0:
        return "<p>Không có câu hỏi.</p>", 0, 0, gr.update(visible=False), gr.update(visible=False)

    idx = max(0, min(idx, total - 1))
    mcq = mcqs[idx]
    q   = mcq.get("question", "")
    diff = mcq.get("difficulty", "medium")
    diff_label = {"easy": "Dễ", "medium": "Trung bình", "hard": "Khó"}.get(diff, diff)
    diff_color = {"easy": "#27AE60", "medium": "#F39C12", "hard": "#E74C3C"}.get(diff, "#666")

    html = f"""
    <div style="padding:20px;background:#F8F9FA;border-radius:12px;min-height:120px">
        <div style="display:flex;justify-content:space-between;margin-bottom:16px">
            <span style="font-size:14px;color:#888">Câu {idx+1} / {total}</span>
            <span style="background:{diff_color};color:white;padding:2px 10px;
                         border-radius:8px;font-size:12px">{diff_label}</span>
        </div>
        <p style="font-size:16px;font-weight:600;color:#1A1A1A;line-height:1.5;margin:0">{q}</p>
    </div>"""

    show_prev = gr.update(visible=(idx > 0))
    show_next = gr.update(visible=(idx < total - 1))
    return html, idx, total, show_prev, show_next


def check_answer(mcqs_json: str, idx: int, chosen: str) -> str:
    """Kiểm tra đáp án, trả về HTML kết quả."""
    if not mcqs_json or chosen is None:
        return ""
    try:
        mcqs = json.loads(mcqs_json)
        mcq  = mcqs[idx]
    except Exception:
        return ""

    correct = mcq.get("answer", "A")
    explanation = mcq.get("explanation", "")
    is_correct  = chosen == correct
    correct_text = mcq.get(correct, "")

    icon  = "[Dung]" if is_correct else "[Sai]"
    bg    = "#1a4731" if is_correct else "#4a1c1c"
    border = "#27AE60" if is_correct else "#E74C3C"
    text_color = "#6ee7b7" if is_correct else "#fca5a5"
    msg   = "Chinh xac!" if is_correct else f"Chua dung. Dap an dung la: {correct}. {correct_text}"

    return f"""
    <div style="padding:14px;border-radius:10px;background:{bg};
                border:1.5px solid {border};margin-top:8px">
        <p style="margin:0 0 8px;font-size:15px;font-weight:600;color:{text_color}">
            {icon} {msg}
        </p>
        <p style="margin:0;font-size:13px;color:#d1d5db">
            Giai thich: {explanation}
        </p>
    </div>"""


def get_option_labels(mcqs_json: str, idx: int) -> list:
    """Lấy 4 options cho radio/button."""
    if not mcqs_json:
        return ["A", "B", "C", "D"]
    try:
        mcq = json.loads(mcqs_json)[idx]
        return [
            f"A. {mcq.get('A','')}",
            f"B. {mcq.get('B','')}",
            f"C. {mcq.get('C','')}",
            f"D. {mcq.get('D','')}",
        ]
    except Exception:
        return ["A", "B", "C", "D"]


def get_final_score(mcqs_json: str, answers: dict) -> str:
    """Tính điểm cuối — answers = {idx: chosen_letter}."""
    if not mcqs_json:
        return ""
    try:
        mcqs = json.loads(mcqs_json)
    except Exception:
        return ""

    total   = len(mcqs)
    correct = sum(
        1 for i, mcq in enumerate(mcqs)
        if answers.get(i) == mcq.get("answer")
    )
    pct = round(correct / total * 100) if total else 0
    color = "#27AE60" if pct >= 70 else "#F39C12" if pct >= 50 else "#E74C3C"

    wrong_html = ""
    for i, mcq in enumerate(mcqs):
        if answers.get(i) != mcq.get("answer"):
            wrong_html += f"""
            <div style="padding:8px 12px;margin:4px 0;background:#FEF0F0;
                        border-radius:8px;font-size:13px">
                <b>Câu {i+1}:</b> {mcq.get('question','')[:80]}...<br>
                <span style="color:#E74C3C">Bạn chọn: {answers.get(i,'—')}</span> |
                <span style="color:#27AE60">Đúng: {mcq.get('answer','')}. {mcq.get(mcq.get('answer','A'),'')}</span>
            </div>"""

    return f"""
    <div style="text-align:center;padding:24px">
        <div style="font-size:48px;font-weight:700;color:{color}">{pct}%</div>
        <div style="font-size:18px;color:#555;margin:8px 0">
            {correct} / {total} câu đúng
        </div>
        {"<p style='color:#27AE60;font-size:16px'>Xuất sắc! 🎉</p>" if pct >= 80 else
         "<p style='color:#F39C12;font-size:16px'>Khá tốt! Ôn thêm một chút nhé.</p>" if pct >= 50 else
         "<p style='color:#E74C3C;font-size:16px'>Cần ôn tập thêm nhé! 📚</p>"}
        {"<div style='margin-top:16px;text-align:left'><b style='font-size:14px'>Câu sai:</b>" + wrong_html + "</div>" if wrong_html else ""}
    </div>"""


# ── Build Gradio UI ────────────────────────────────────────────────

def build_app():
    with gr.Blocks(title="AI MCQ Generator") as demo:

        # State — khai báo bên trong Blocks context
        mcqs_state   = gr.State(value=None)
        lang_state   = gr.State(value="vi")
        quiz_idx     = gr.State(value=0)
        user_answers = gr.State(value={})

        # ── Header ────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=8):
                gr.HTML("""
                <div class="main-header">
                    <h1>🎯 AI MCQ Generator</h1>
                    <p id="subtitle">Tạo câu hỏi trắc nghiệm từ tài liệu của bạn</p>
                </div>""")
            with gr.Column(scale=1, min_width=120):
                lang_btn = gr.Button("🌐 English", size="sm", variant="secondary")

        # ── Tabs ──────────────────────────────────────────────────
        with gr.Tabs() as tabs:

            # ════ TAB 1: UPLOAD & GENERATE ════════════════════════
            with gr.TabItem("📤 Tạo Quiz", id="tab_upload") as tab_upload:
                with gr.Row(equal_height=True):

                    # Left: file upload
                    with gr.Column(scale=3):
                        file_input = gr.File(
                            label="Kéo file vào đây · PDF, PPTX, DOCX, XLSX, JPG, PNG",
                            file_count="multiple",
                            file_types=[".pdf", ".pptx", ".docx",
                                        ".xlsx", ".xls",
                                        ".jpg", ".jpeg", ".png", ".webp"],
                            height=220,
                        )
                        file_info = gr.HTML("<p class='estimate-text'></p>")

                    # Right: config
                    with gr.Column(scale=2):
                        gr.Markdown("**Cấu hình**")
                        difficulty_radio = gr.Radio(
                            choices=["Dễ", "Trung bình", "Khó"],
                            value="Trung bình",
                            label="Độ khó",
                        )
                        density_radio = gr.Radio(
                            choices=["Ít", "Vừa", "Nhiều"],
                            value="Vừa",
                            label="Mật độ câu hỏi",
                        )
                        qlang_radio = gr.Radio(
                            choices=["Tiếng Việt", "English"],
                            value="English",
                            label="Ngôn ngữ câu hỏi",
                        )
                        estimate_box = gr.Markdown(
                            value="", elem_classes=["estimate-text"]
                        )

                # Generate button + progress
                generate_btn = gr.Button(
                    "🚀 Tạo Quiz →",
                    variant="primary",
                    size="lg",
                    elem_classes=["generate-btn"],
                )
                progress_text = gr.Textbox(
                    label="Tien trinh",
                    interactive=False,
                    visible=True,
                    value="San sang.",
                    max_lines=1,
                )

            # ════ TAB 2: QUIZ PLAYER ══════════════════════════════════
            with gr.TabItem("▶ Chơi thử", id="tab_quiz") as tab_quiz:
                with gr.Row():
                    with gr.Column():
                        quiz_start_btn = gr.Button(
                            "▶ Bắt đầu Quiz", variant="primary", size="lg"
                        )

                # Màn quiz (ẩn ban đầu)
                with gr.Column(visible=False) as quiz_screen:
                    question_html = gr.HTML("")
                    answer_radio  = gr.Radio(
                        choices=["A","B","C","D"],
                        label="Chọn đáp án:",
                        interactive=True,
                    )
                    check_btn   = gr.Button("Kiểm tra", variant="primary")
                    result_html = gr.HTML("")
                    with gr.Row():
                        prev_btn = gr.Button("← Câu trước", visible=False, size="sm")
                        next_btn = gr.Button("Câu tiếp →", variant="primary",
                                             visible=False, size="sm")
                        finish_btn = gr.Button("Xem kết quả 🏁", variant="primary",
                                               visible=False, size="sm")

                # Màn kết quả (ẩn ban đầu)
                with gr.Column(visible=False) as quiz_result_screen:
                    score_html  = gr.HTML("")
                    replay_btn  = gr.Button("Chơi lại", variant="secondary")

            # ════ TAB 4: REVIEW & EDIT ═════════════════════════════
            with gr.TabItem("✏️ Review & Chỉnh sửa", id="tab_review") as tab_review:
                stats_html = gr.HTML("")

                with gr.Row():
                    # Sidebar filters
                    with gr.Column(scale=1, min_width=180):
                        gr.Markdown("**Lọc**")
                        diff_filter = gr.Radio(
                            choices=["Tất cả", "Dễ", "Trung bình", "Khó"],
                            value="Tất cả",
                            label="Độ khó",
                        )
                        search_box = gr.Textbox(
                            placeholder="Tìm câu hỏi...",
                            label="Tìm kiếm",
                            max_lines=1,
                        )
                        with gr.Row():
                            play_btn   = gr.Button("▶ Chơi thử", variant="secondary", size="sm")
                            export_btn = gr.Button("⬇ Tải về",  variant="secondary", size="sm")

                    # MCQ cards
                    with gr.Column(scale=4):
                        mcq_display = gr.HTML("")

            # ════ TAB 4: EXPORT ════════════════════════════════════
            with gr.TabItem("💾 Tải về", id="tab_export") as tab_export:
                gr.Markdown("### Chọn định dạng xuất")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 🎮 Kahoot\nFile Excel import vào Kahoot")
                        kahoot_btn  = gr.Button("Tải Kahoot (.xlsx)", variant="primary")
                        kahoot_file = gr.File(label="", visible=False)
                        gr.Markdown("""
> **Hướng dẫn import:**
> 1. Vào [create.kahoot.it](https://create.kahoot.it)
> 2. Chọn **Import → Spreadsheet**
> 3. Upload file vừa tải
> 4. Kiểm tra và **Publish**
                        """)

                    with gr.Column():
                        gr.Markdown("#### 📊 Quizizz\nFile CSV import vào Quizizz")
                        quizizz_btn  = gr.Button("Tải Quizizz (.csv)", variant="primary")
                        quizizz_file = gr.File(label="", visible=False)
                        gr.Markdown("""
> **Hướng dẫn import:**
> 1. Vào [quizizz.com](https://quizizz.com) → Create
> 2. Chọn **Import từ file**
> 3. Upload file CSV
> 4. Review và **Save**
                        """)

                    with gr.Column():
                        gr.Markdown("#### 📄 JSON\nDữ liệu đầy đủ cho nghiên cứu")
                        json_btn  = gr.Button("Tải JSON", variant="secondary")
                        json_file = gr.File(label="", visible=False)

        # ── Event handlers ────────────────────────────────────────

        # Estimate khi đổi file hoặc density
        def update_estimate(files, density):
            if not files:
                return ""
            est = estimate_questions(files, density)
            return f"Ước tính: **{est} câu**"

        file_input.change(update_estimate,
                          inputs=[file_input, density_radio],
                          outputs=[estimate_box])
        density_radio.change(update_estimate,
                             inputs=[file_input, density_radio],
                             outputs=[estimate_box])

        # Generate pipeline
        def on_generate(files, difficulty, density, qlang, lang,
                        progress=gr.Progress(track_tqdm=False)):
            if not files:
                return (
                    gr.update(value="Vui long upload it nhat 1 file.", visible=True),
                    None, "", ""
                )

            progress(0, desc="Dang khoi dong...")

            def on_step(pct, msg):
                progress(pct, desc=msg)

            mcqs_json, err = run_pipeline(
                files, difficulty, density, qlang, lang, on_step=on_step
            )
            progress(1, desc="Hoan tat!")

            if err:
                return (gr.update(value=f"Loi: {err}", visible=True), None, "", "")
            if not mcqs_json:
                return (gr.update(value="Khong sinh duoc MCQ nao. Thu lai.", visible=True), None, "", "")

            try:
                n = len(json.loads(mcqs_json))
            except Exception:
                n = 0

            final_stats = get_stats_html(mcqs_json, lang)
            final_cards = render_review(mcqs_json, lang, "Tat ca", "")

            return (
                gr.update(value=f"Hoan tat! Da tao {n} cau hoi.", visible=True),
                mcqs_json,
                final_stats,
                final_cards,
            )

        generate_btn.click(
            fn=on_generate,
            inputs=[file_input, difficulty_radio, density_radio,
                    qlang_radio, lang_state],
            outputs=[progress_text, mcqs_state, stats_html, mcq_display],
        )

        # Filter / search in review
        def on_filter(mcqs_json, diff_f, search, lang):
            stats = get_stats_html(mcqs_json, lang)
            cards = render_review(mcqs_json, lang, diff_f, search)
            return stats, cards

        diff_filter.change(on_filter,
                           inputs=[mcqs_state, diff_filter, search_box, lang_state],
                           outputs=[stats_html, mcq_display])
        search_box.change(on_filter,
                          inputs=[mcqs_state, diff_filter, search_box, lang_state],
                          outputs=[stats_html, mcq_display])

        # Export buttons


        def on_export_kahoot(mcqs_json):
            path = export_kahoot(mcqs_json)
            return gr.update(value=path, visible=True) if path else gr.update(visible=False)

        def on_export_quizizz(mcqs_json):
            path = export_quizizz(mcqs_json)
            return gr.update(value=path, visible=True) if path else gr.update(visible=False)

        def on_export_json(mcqs_json):
            path = export_json_file(mcqs_json)
            return gr.update(value=path, visible=True) if path else gr.update(visible=False)

        kahoot_btn.click(on_export_kahoot,
                         inputs=[mcqs_state], outputs=[kahoot_file])
        quizizz_btn.click(on_export_quizizz,
                          inputs=[mcqs_state], outputs=[quizizz_file])
        json_btn.click(on_export_json,
                       inputs=[mcqs_state], outputs=[json_file])

        # ── Quiz Player events ────────────────────────────────────
        def start_quiz(mcqs_json):
            if not mcqs_json:
                return (gr.update(), gr.update(visible=False),
                        gr.update(visible=False), 0, {})
            labels = get_option_labels(mcqs_json, 0)
            q_html, idx, total, show_prev, show_next = render_quiz_question(mcqs_json, 0)
            show_finish = gr.update(visible=(total == 1))
            show_next_  = gr.update(visible=(total > 1))
            return (
                gr.update(visible=True),   # quiz_screen
                gr.update(visible=False),  # quiz_result_screen
                q_html,
                gr.update(choices=labels, value=None),
                "",         # result_html
                show_prev,
                show_next_,
                show_finish,
                0,          # quiz_idx
                {},         # user_answers
            )

        quiz_start_btn.click(
            start_quiz,
            inputs=[mcqs_state],
            outputs=[quiz_screen, quiz_result_screen, question_html,
                     answer_radio, result_html,
                     prev_btn, next_btn, finish_btn,
                     quiz_idx, user_answers],
        )

        def on_check(mcqs_json, idx, chosen, answers):
            if chosen is None:
                return "", answers, gr.update(visible=False)
            letter = chosen[0]  # "A. text" → "A"
            result = check_answer(mcqs_json, idx, letter)
            answers = dict(answers)
            answers[idx] = letter
            try:
                total = len(json.loads(mcqs_json))
                is_last = (idx >= total - 1)
            except Exception:
                is_last = False
            return (
                result,
                answers,
                gr.update(visible=True),   # next_btn
                gr.update(visible=is_last), # finish_btn
            )

        check_btn.click(
            on_check,
            inputs=[mcqs_state, quiz_idx, answer_radio, user_answers],
            outputs=[result_html, user_answers, next_btn, finish_btn],
        )

        def go_next(mcqs_json, idx):
            new_idx = idx + 1
            labels  = get_option_labels(mcqs_json, new_idx)
            q_html, new_idx, total, show_prev, show_next = render_quiz_question(mcqs_json, new_idx)
            is_last = (new_idx >= total - 1)
            return (
                q_html,
                gr.update(choices=labels, value=None),
                "",
                show_prev,
                gr.update(visible=not is_last),
                gr.update(visible=is_last),
                new_idx,
            )

        next_btn.click(
            go_next,
            inputs=[mcqs_state, quiz_idx],
            outputs=[question_html, answer_radio, result_html,
                     prev_btn, next_btn, finish_btn, quiz_idx],
        )

        def go_prev(mcqs_json, idx):
            new_idx = max(0, idx - 1)
            labels  = get_option_labels(mcqs_json, new_idx)
            q_html, new_idx, total, show_prev, show_next = render_quiz_question(mcqs_json, new_idx)
            return (
                q_html,
                gr.update(choices=labels, value=None),
                "",
                show_prev,
                show_next,
                gr.update(visible=False),
                new_idx,
            )

        prev_btn.click(
            go_prev,
            inputs=[mcqs_state, quiz_idx],
            outputs=[question_html, answer_radio, result_html,
                     prev_btn, next_btn, finish_btn, quiz_idx],
        )

        def show_score(mcqs_json, answers):
            score = get_final_score(mcqs_json, answers)
            return score

        finish_btn.click(
            show_score,
            inputs=[mcqs_state, user_answers],
            outputs=[score_html],
        ).then(
            lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[quiz_screen, quiz_result_screen],
        )

        def replay(mcqs_json):
            labels = get_option_labels(mcqs_json, 0)
            q_html, _, total, show_prev, show_next = render_quiz_question(mcqs_json, 0)
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                q_html,
                gr.update(choices=labels, value=None),
                "",
                show_prev,
                gr.update(visible=(total > 1)),
                gr.update(visible=(total == 1)),
                0, {},
            )

        replay_btn.click(
            replay,
            inputs=[mcqs_state],
            outputs=[quiz_screen, quiz_result_screen, question_html,
                     answer_radio, result_html,
                     prev_btn, next_btn, finish_btn,
                     quiz_idx, user_answers],
        )

        # Language switch
        def switch_lang(current_lang):
            new_lang = "en" if current_lang == "vi" else "vi"
            btn_label = "🌐 Tiếng Việt" if new_lang == "en" else "🌐 English"
            return new_lang, btn_label

        lang_btn.click(switch_lang,
                       inputs=[lang_state],
                       outputs=[lang_state, lang_btn])

    return demo


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )