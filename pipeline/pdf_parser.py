"""
pipeline/pdf_parser.py
----------------------
PDF → List[{page_num, title, text, char_count}]

Primary:  PyMuPDF (fitz)  — fast, font-size-aware
Fallback: pdfplumber      — layout-aware, tốt hơn cho slide dạng column
"""

import fitz          # PyMuPDF
import pdfplumber
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────
MIN_CHARS = 50          # trang có ít hơn N chars → bỏ qua (trang ảnh / trống)
TITLE_MAX_CHARS = 200   # title không dài hơn N chars


# ── Main function ──────────────────────────────────────────────────

def parse_pdf(pdf_path: str) -> list[dict]:
    """
    Parse PDF slides → list of page dicts.

    Args:
        pdf_path: đường dẫn tới file PDF

    Returns:
        List[{page_num, title, text, char_count}]
        Đã filter trang rác (char_count < MIN_CHARS)
        Sort theo page_num tăng dần
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File không tồn tại: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File phải là .pdf, nhận được: {path.suffix}")

    pages = []

    with fitz.open(pdf_path) as doc:
        total = len(doc)
        for page_idx in range(total):
            page_num = page_idx + 1
            fitz_page = doc[page_idx]

            # --- Thử extract bằng PyMuPDF trước ---
            text, title = _extract_fitz(fitz_page)

            # --- Fallback: pdfplumber nếu text quá ngắn ---
            if len(text.strip()) < MIN_CHARS:
                text_pb, title_pb = _extract_pdfplumber(pdf_path, page_idx)
                if len(text_pb.strip()) > len(text.strip()):
                    text = text_pb
                    title = title_pb or title
                    _log(f"  page {page_num}: fallback → pdfplumber")

            text = text.strip()
            char_count = len(text)

            # --- Filter trang rác ---
            if char_count < MIN_CHARS:
                _log(f"  page {page_num}: SKIP (only {char_count} chars)")
                continue

            # Convert sang markdown để đồng nhất với các parser khác
            title_str = title or f"Page {page_num}"
            md_text   = _to_markdown(title_str, text)

            pages.append({
                "page_num":   page_num,
                "title":      title_str,
                "text":       md_text,
                "char_count": len(md_text),
                "has_image":  False,   # PDF plain text, không detect ảnh
            })

    pages.sort(key=lambda p: p["page_num"])
    return pages


# ── Extractors ─────────────────────────────────────────────────────

def _extract_fitz(page) -> tuple[str, str | None]:
    """
    Extract text + detect title từ 1 fitz page.
    Title = text block có font size lớn nhất, nằm trong top 30% trang.
    """
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

    lines_text = []
    title = None
    max_font_size = 0
    page_height = page.rect.height
    top_zone = page_height * 0.35   # chỉ tìm title trong 35% trên của trang

    for block in blocks:
        if block.get("type") != 0:  # 0 = text block
            continue
        for line in block.get("lines", []):
            line_text = " ".join(
                span["text"] for span in line.get("spans", [])
            ).strip()
            if not line_text:
                continue
            lines_text.append(line_text)

            # --- Detect title ---
            for span in line.get("spans", []):
                span_text = span["text"].strip()
                font_size = span.get("size", 0)
                y_pos = span["origin"][1]

                if (
                    font_size > max_font_size
                    and y_pos < top_zone
                    and len(span_text) > 3
                    and len(span_text) < TITLE_MAX_CHARS
                ):
                    max_font_size = font_size
                    title = span_text

    text = "\n".join(lines_text)
    return text, title


def _extract_pdfplumber(pdf_path: str, page_idx: int) -> tuple[str, str | None]:
    """
    Fallback extractor dùng pdfplumber.
    Title = dòng đầu tiên không rỗng.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_idx]
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            title = lines[0] if lines else None
            return text, title
    except Exception as e:
        _log(f"  pdfplumber error on page {page_idx+1}: {e}")
        return "", None


# ── Markdown converter ─────────────────────────────────────────────

def _to_markdown(title: str, text: str) -> str:
    """
    Convert plain text page → markdown string.
    Format: ## title + bullet lines.
    """
    lines = [line.rstrip() for line in text.splitlines()]
    body  = "\n".join(
        f"- {line}" if line and not line.startswith("-") else line
        for line in lines
    )
    return f"## {title}\n\n{body}"


# ── Utility ────────────────────────────────────────────────────────

def _log(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


def get_pdf_metadata(pdf_path: str) -> dict:
    """Trả về metadata cơ bản của PDF (dùng để hiển thị trong Streamlit UI)."""
    with fitz.open(pdf_path) as doc:
        meta = doc.metadata
        return {
            "total_pages": len(doc),
            "title":       meta.get("title", ""),
            "author":      meta.get("author", ""),
            "file_size_kb": round(Path(pdf_path).stat().st_size / 1024, 1),
        }


# ── CLI test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    # Fix encoding trên Windows terminal
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("Usage: python pipeline/pdf_parser.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # Metadata
    meta = get_pdf_metadata(pdf_path)
    print(f"\n{'='*55}")
    print(f"  File     : {pdf_path}")
    print(f"  Pages    : {meta['total_pages']}")
    print(f"  Size     : {meta['file_size_kb']} KB")
    print(f"{'='*55}\n")

    # Parse
    pages = parse_pdf(pdf_path)

    print(f"[OK] Parsed: {len(pages)} / {meta['total_pages']} pages kept\n")

    # Preview 5 trang đầu
    print(f"{'PAGE':<6} {'TITLE':<45} {'CHARS'}")
    print("-" * 60)
    for p in pages[:5]:
        print(f"  {p['page_num']:<4} {p['title'][:43]:<45} {p['char_count']}")

    if len(pages) > 5:
        print(f"  ... ({len(pages)-5} more pages)")

    # Sample text
    print(f"\n--- Sample text (page {pages[0]['page_num']}) ---")
    print(pages[0]["text"][:400])
    print()

    # Validation tự động
    print("--- Auto checks ---")
    assert len(pages) > 0, "FAIL: Không parse được trang nào"
    assert all(p["char_count"] >= MIN_CHARS for p in pages), "FAIL: Có trang dưới MIN_CHARS"
    assert all(p["title"] for p in pages), "FAIL: Có trang thiếu title"
    titles = [p["title"] for p in pages]
    unique_ratio = len(set(titles)) / len(titles)
    print(f"  Unique title ratio : {unique_ratio:.0%} {'[OK]' if unique_ratio > 0.5 else '[WARN] nhieu title bi trung'}")
    avg_chars = sum(p["char_count"] for p in pages) / len(pages)
    print(f"  Avg chars/page     : {avg_chars:.0f} {'[OK]' if avg_chars > 100 else '[WARN] text ngan, kiem tra lai PDF'}")
    print(f"  All pages have text: [OK]")
    print(f"\nDone.\n")