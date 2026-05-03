"""
pipeline/pdf_parser.py
----------------------
PDF → List[{page_num, title, text, char_count, has_image}]

Uses pymupdf4llm for LLM-optimized Markdown output.

Fixes:
  #3  detect garbled Vietnamese font → pdfplumber fallback
  #5  split trang quá dài theo paragraph boundary
  #8  _extract_title bỏ qua dòng alt-text ảnh (![...)
  #9  scanned PDF page → pytesseract OCR fallback
"""

import re
import statistics
import fitz
import pymupdf4llm
from pathlib import Path

from pipeline.layout_analyzer import (
    analyze_text_layout,
    analyze_pdf_page_layout,
    merge_layout_flags,
)


MIN_CHARS       = 50
TITLE_MAX_CHARS = 200
MAX_PAGE_CHARS  = 8000   # #5: trang vượt ngưỡng này sẽ bị split
_GARBLED_RATIO  = 0.25   # #3: tỉ lệ ký tự đơn / tổng từ để coi là lỗi font


def parse_pdf(pdf_path: str) -> list[dict]:
    """
    Parse PDF → list of page dicts.
    Returns: List[{page_num, title, text, char_count, has_image,
                   has_table, has_formula, has_caption, has_columns,
                   layout_sensitive, vision_used, parser, ocr_used,
                   text_quality, warnings}]
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File không tồn tại: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File phải là .pdf, nhận được: {path.suffix}")

    chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)

    total_chars = sum(len(c["text"].strip()) for c in chunks)
    n_pages     = len(chunks)
    if n_pages > 0 and total_chars / n_pages < MIN_CHARS:
        _log(f"  [scan] PDF có vẻ là bản scan (avg {total_chars // n_pages} chars/trang) → thử OCR từng trang")

    pages = []
    for i, chunk in enumerate(chunks):
        meta      = chunk.get("metadata", {})
        raw_page  = meta.get("page", meta.get("page_number", i))
        page_num  = raw_page + 1
        text      = chunk["text"].strip()
        has_image = len(chunk.get("images", [])) > 0
        parser    = "pymupdf4llm"
        ocr_used  = False
        warnings  = []

        # #9: trang ít text + có ảnh → có thể là scan → thử pytesseract
        if len(text) < MIN_CHARS:
            if has_image:
                warnings.append("low_text_with_image")
                _log(f"  page {page_num}: ít text ({len(text)} chars) + có ảnh → thử OCR")
                ocr_text = _pytesseract_extract_page(str(path), i)
                if len(ocr_text.strip()) >= MIN_CHARS:
                    _log(f"  page {page_num}: OCR OK ({len(ocr_text.strip())} chars)")
                    text = ocr_text.strip()
                    parser = "pytesseract"
                    ocr_used = True
                else:
                    _log(f"  page {page_num}: SKIP (OCR cũng không đủ text)")
                    continue
            else:
                _log(f"  page {page_num}: SKIP (only {len(text)} chars)")
                continue

        # #3: detect garbled Vietnamese font → thử pdfplumber
        if _is_garbled(text):
            warnings.append("garbled_text_detected")
            _log(f"  page {page_num}: WARN font lỗi → thử pdfplumber")
            fallback = _pdfplumber_extract(str(path), page_num)
            if fallback and not _is_garbled(fallback) and len(fallback) >= MIN_CHARS:
                _log(f"  page {page_num}: pdfplumber OK ({len(fallback)} chars)")
                text = fallback
                parser = "pdfplumber"
            else:
                _log(f"  page {page_num}: pdfplumber không cải thiện → giữ nguyên")

        title = _extract_title(text) or f"Page {page_num}"

        text_layout = analyze_text_layout(text, has_image=has_image)
        pdf_layout  = analyze_pdf_page_layout(str(path), i)
        layout      = merge_layout_flags(text_layout, pdf_layout)
        warnings   += [w for w in pdf_layout.get("warnings", [])
                       if w not in ("no_text_blocks",)]

        pages.append({
            "page_num":        page_num,
            "title":           title,
            "text":            text,
            "char_count":      len(text),
            "has_image":       has_image,
            "parser":          parser,
            "ocr_used":        ocr_used,
            "text_quality":    _text_quality(text),
            "warnings":        warnings,
            "has_table":       layout["has_table"],
            "has_formula":     layout["has_formula"],
            "has_caption":     layout["has_caption"],
            "has_columns":     layout["has_columns"],
            "layout_sensitive": layout["layout_sensitive"],
            "vision_used":     layout["vision_used"],
        })

    pages.sort(key=lambda p: p["page_num"])

    # #5: split trang quá dài → renumber sau khi split
    pages = _split_oversized_pages(pages, MAX_PAGE_CHARS)

    return pages


# ── Fix #8: title extraction ─────────────────────────────────────────

def _extract_title(md_text: str) -> str | None:
    """
    Extract title từ markdown heading đầu tiên; fallback về dòng text đầu tiên.
    Bỏ qua dòng là image alt-text (bắt đầu bằng '![').
    """
    # Ưu tiên: dòng heading #
    for line in md_text.splitlines():
        line = line.strip()
        if not line.startswith("#"):
            continue
        title = line.lstrip("#").strip()
        if title.startswith("!["):
            continue
        if 3 < len(title) < TITLE_MAX_CHARS:
            return title

    # Fallback: dòng text đầu tiên có nghĩa, không phải list/table/alt-text
    for line in md_text.splitlines():
        line = line.strip()
        if (line
                and not line.startswith(("-", "*", "|", "![", "#"))
                and len(line) > 3):
            return line[:TITLE_MAX_CHARS]

    return None


# ── Fix #3: garbled font ──────────────────────────────────────────────

def _is_garbled(text: str) -> bool:
    """
    Detect Vietnamese font garbling.
    Dấu hiệu: nhiều ký tự đơn alpha xen kẽ khoảng trắng liên tiếp.
    Ví dụ: "M c tiêu h c t" thay vì "Mục tiêu học tập".
    """
    words = text.split()
    if len(words) < 15:
        return False
    single_alpha = sum(1 for w in words if len(w) == 1 and w.isalpha())
    return single_alpha / len(words) > _GARBLED_RATIO


def _pdfplumber_extract(pdf_path: str, page_num: int) -> str:
    """Fallback extraction dùng pdfplumber cho trang bị lỗi font."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            if page_num - 1 >= len(pdf.pages):
                return ""
            return pdf.pages[page_num - 1].extract_text() or ""
    except Exception:
        return ""


def _pytesseract_extract_page(pdf_path: str, page_index: int) -> str:
    """
    #9: Render 1 trang PDF thành ảnh và OCR bằng pytesseract.
    Dùng fitz (đã có sẵn) để render, không cần poppler/pdf2image.
    lang="vie+eng" để nhận diện cả tiếng Việt lẫn tiếng Anh.
    """
    try:
        import pytesseract
        from PIL import Image

        with fitz.open(pdf_path) as doc:
            if page_index >= len(doc):
                return ""
            page = doc[page_index]
            mat  = fitz.Matrix(2, 2)   # 2x zoom → OCR chính xác hơn
            pix  = page.get_pixmap(matrix=mat)
            img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        return pytesseract.image_to_string(img, lang="vie+eng")

    except ImportError:
        _log("  [ocr] pytesseract chưa cài → pip install pytesseract pillow")
        return ""
    except Exception as e:
        _log(f"  [ocr] pytesseract lỗi trang {page_index + 1}: {e}")
        return ""


# ── Fix #5: split oversized pages ────────────────────────────────────

def _split_oversized_pages(pages: list[dict], max_chars: int) -> list[dict]:
    """
    Split các page có text > max_chars thành nhiều sub-pages theo paragraph.
    Renumber page_num tuần tự sau khi split để đảm bảo unique.
    """
    result = []
    for page in pages:
        if len(page["text"]) <= max_chars:
            result.append(page)
        else:
            subs = _split_page_by_paragraphs(page, max_chars)
            _log(f"  page {page['page_num']}: SPLIT → {len(subs)} sub-pages ({len(page['text'])} chars)")
            result.extend(subs)

    for i, p in enumerate(result):
        p["page_num"] = i + 1

    return result


def _split_page_by_paragraphs(page: dict, max_chars: int) -> list[dict]:
    """Tách text của 1 page theo paragraph boundary (\\n\\n)."""
    paragraphs  = re.split(r"\n{2,}", page["text"])
    sub_pages   = []
    current     = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > max_chars and current:
            text = "\n\n".join(current).strip()
            sub_pages.append(_make_subpage(page, text))
            current     = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para)

    if current:
        text = "\n\n".join(current).strip()
        if text:
            sub_pages.append(_make_subpage(page, text))

    return sub_pages if sub_pages else [page]


def _make_subpage(original: dict, text: str) -> dict:
    has_image   = original.get("has_image", False)
    text_layout = analyze_text_layout(text, has_image=has_image)
    return {
        "page_num":        original["page_num"],  # tạm thời, renumber ở _split_oversized_pages
        "title":           _extract_title(text) or original["title"],
        "text":            text,
        "char_count":      len(text),
        "has_image":       has_image,
        "parser":          original.get("parser", "pymupdf4llm"),
        "ocr_used":        original.get("ocr_used", False),
        "text_quality":    _text_quality(text),
        "warnings":        list(original.get("warnings", [])) + ["split_oversized_page"],
        "has_table":       text_layout["has_table"],
        "has_formula":     text_layout["has_formula"],
        "has_caption":     text_layout["has_caption"],
        "has_columns":     text_layout["has_columns"],
        "layout_sensitive": text_layout["layout_sensitive"],
        "vision_used":     original.get("vision_used", False),
    }


def _text_quality(text: str) -> str:
    """Classify extracted text quality for debugging and experiments."""
    n_chars = len(text.strip())
    if n_chars < MIN_CHARS:
        return "low_text"
    if _is_garbled(text):
        return "garbled"
    words = text.split()
    if words:
        avg_word_len = statistics.mean(len(w) for w in words)
        if avg_word_len < 2:
            return "noisy"
    return "good"


# ── Metadata & helpers ────────────────────────────────────────────────

def get_pdf_metadata(pdf_path: str) -> dict:
    with fitz.open(pdf_path) as doc:
        meta = doc.metadata
        return {
            "total_pages":  len(doc),
            "title":        meta.get("title", ""),
            "author":       meta.get("author", ""),
            "file_size_kb": round(Path(pdf_path).stat().st_size / 1024, 1),
        }


def _log(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


# ── CLI test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("Usage: python pipeline/pdf_parser.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    meta = get_pdf_metadata(pdf_path)
    print(f"\n{'='*55}")
    print(f"  File     : {pdf_path}")
    print(f"  Pages    : {meta['total_pages']}")
    print(f"  Size     : {meta['file_size_kb']} KB")
    print(f"{'='*55}\n")

    pages = parse_pdf(pdf_path)

    print(f"[OK] Parsed: {len(pages)} pages\n")
    print(f"{'PAGE':<6} {'GARBLED':<9} {'CHARS':<7} TITLE")
    print("-" * 65)
    for p in pages:
        garbled_flag = "WARN" if _is_garbled(p["text"]) else "-"
        print(f"  {p['page_num']:<4} {garbled_flag:<9} {p['char_count']:<7} {p['title'][:43]}")

    print(f"\n--- Sample text (page {pages[0]['page_num']}) ---")
    print(pages[0]["text"][:400])
    print()

    print("--- Auto checks ---")
    assert len(pages) > 0, "FAIL: Không parse được trang nào"
    assert all(p["char_count"] >= MIN_CHARS for p in pages), "FAIL: Có trang dưới MIN_CHARS"
    assert all(p["title"] for p in pages), "FAIL: Có trang thiếu title"
    assert len([p["page_num"] for p in pages]) == len(set(p["page_num"] for p in pages)), "FAIL: page_num trùng"

    garbled_pages = [p for p in pages if _is_garbled(p["text"])]
    alt_text_titles = [p for p in pages if p["title"].startswith("![")]

    print(f"  Trang lỗi font      : {len(garbled_pages)}/{len(pages)}")
    print(f"  Title là alt-text   : {len(alt_text_titles)}/{len(pages)}")
    avg_chars = sum(p["char_count"] for p in pages) / len(pages)
    print(f"  Avg chars/page      : {avg_chars:.0f}")
    max_chars = max(p["char_count"] for p in pages)
    print(f"  Max chars/page      : {max_chars}  {'[OK]' if max_chars <= MAX_PAGE_CHARS else '[WARN] đã split'}")
    print(f"\nDone.\n")
