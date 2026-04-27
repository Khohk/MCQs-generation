"""
pipeline/file_router.py
-----------------------
Router trung tâm: nhận bất kỳ file nào → trả về List[page_dict]
cùng schema với pdf_parser để chunker/generator không cần sửa gì.

Supported formats:
  .pdf   → pdf_parser.py   (pymupdf4llm — LLM-optimized Markdown)
  .docx  → docx_parser.py  (python-docx)
  .pptx  → pptx_parser.py  (python-pptx)

Usage:
  from pipeline.file_router import parse_file, get_metadata, SUPPORTED_EXTENSIONS
  pages = parse_file("lecture.pptx")
  pages = parse_file("notes.docx")
  pages = parse_file("slides.pdf")
"""

from pathlib import Path

# .pdf → pdf_parser (pymupdf4llm, LLM-optimized Markdown)
# các format còn lại → markitdown_parser
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".pptx", ".docx",
    ".xlsx", ".xls",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
}

_MARKITDOWN_EXTENSIONS = {
    ".pptx", ".docx",
    ".xlsx", ".xls",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
}


def parse_file(file_path: str, **kwargs) -> list[dict]:
    """
    Auto-detect file type và gọi đúng parser.

    Routing:
      .pdf              → pdf_parser.py  (pymupdf4llm)
      .pptx/.docx/.xlsx → markitdown_parser.py
      images            → markitdown_parser.py (vision)

    Returns:
        List[{page_num, title, text, char_count, has_image?}]
    """
    path = Path(file_path)
    ext  = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"File khong ton tai: {file_path}")

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Format '{ext}' chua duoc ho tro. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".pdf":
        from pipeline.pdf_parser import parse_pdf
        return parse_pdf(file_path, **kwargs)

    elif ext in _MARKITDOWN_EXTENSIONS:
        from pipeline.markitdown_parser import parse_to_pages
        return parse_to_pages(file_path)


def get_metadata(file_path: str) -> dict:
    """
    Lấy metadata cơ bản của file (total_pages, title, file_size_kb).
    Trả về dict chuẩn bất kể format.
    """
    path = Path(file_path)
    ext  = path.suffix.lower()

    if ext == ".pdf":
        from pipeline.pdf_parser import get_pdf_metadata
        return get_pdf_metadata(file_path)

    # Với các format MarkItDown: đếm sections làm "pages"
    if ext in _MARKITDOWN_EXTENSIONS:
        from pipeline.markitdown_parser import parse_to_pages
        pages = parse_to_pages(file_path)
        return {
            "total_pages"  : len(pages),
            "title"        : pages[0]["title"] if pages else "",
            "file_size_kb" : round(path.stat().st_size / 1024, 1),
        }

    return {"total_pages": 0, "title": "", "file_size_kb": 0}


def is_supported(file_path: str) -> bool:
    """Kiểm tra file có được support không."""
    return Path(file_path).suffix.lower() in SUPPORTED_EXTENSIONS


def get_file_type(file_path: str) -> str:
    """Trả về tên loại file dễ đọc."""
    ext = Path(file_path).suffix.lower()
    return {
        ".pdf" : "PDF",
        ".pptx": "PowerPoint",
        ".docx": "Word",
        ".xlsx": "Excel", ".xls": "Excel",
        ".jpg" : "Image",  ".jpeg": "Image",
        ".png" : "Image",  ".gif" : "Image",
        ".bmp" : "Image",  ".webp": "Image",
    }.get(ext, "Unknown")


# ── CLI test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.file_router <file>")
        print(f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)

    file_path = sys.argv[1]
    file_type = get_file_type(file_path)
    meta      = get_metadata(file_path)

    print(f"\n{'='*55}")
    print(f"  File : {file_path}")
    print(f"  Type : {file_type}")
    print(f"  Pages: {meta.get('total_pages', '?')}")
    print(f"  Size : {meta.get('file_size_kb', '?')} KB")
    print(f"{'='*55}\n")

    pages = parse_file(file_path)

    print(f"[OK] Parsed: {len(pages)} sections/pages/slides\n")
    print(f"{'NUM':<5} {'TITLE':<45} {'CHARS'}")
    print("-" * 60)
    for p in pages[:8]:
        print(f"  {p['page_num']:<3} {p['title'][:43]:<45} {p['char_count']}")
    if len(pages) > 8:
        print(f"  ... ({len(pages)-8} more)")

    if pages:
        print(f"\n--- Sample text (section 1) ---")
        print(pages[0]["text"][:400])

    # Verify schema
    print("\n--- Schema check ---")
    required = {"page_num", "title", "text", "char_count"}
    for p in pages:
        missing = required - set(p.keys())
        if missing:
            print(f"  [FAIL] page {p.get('page_num','?')} missing: {missing}")
            break
    else:
        print(f"  [OK] Tat ca {len(pages)} pages co du 4 fields")
        avg = sum(p["char_count"] for p in pages) / len(pages)
        print(f"  [OK] Avg chars: {avg:.0f}")