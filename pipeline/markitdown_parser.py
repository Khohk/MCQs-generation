"""
pipeline/markitdown_parser.py
-----------------------------
Universal file parser: PDF / PPTX / DOCX / XLSX / Images → Markdown

Uses : microsoft/markitdown
Vision: Groq meta-llama/llama-4-scout-17b-16e-instruct (free, OpenAI-compat)
        → mô tả ảnh trong file khi GROQ_API_KEY có trong .env
        → nếu không có key thì bỏ qua ảnh, vẫn parse text bình thường

Supported formats:
  .pdf  .pptx  .docx  .xlsx  .xls
  .jpg  .jpeg  .png  .gif  .bmp  .webp

Install:
  pip install markitdown[all] openai python-dotenv

.env:
  GROQ_API_KEY=your_groq_key_here
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from markitdown import MarkItDown

load_dotenv()

# ── Supported extensions ────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {
    ".pdf", ".pptx", ".docx",
    ".xlsx", ".xls",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
}

# ── Build MarkItDown instance ───────────────────────────────────────

def _build_converter() -> MarkItDown:
    """
    Tạo MarkItDown instance.
    Nếu có GROQ_API_KEY → bật vision (mô tả ảnh bằng Llama 3.2 Vision).
    Nếu không            → parse text-only, ảnh bị bỏ qua.
    """
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()

    if groq_key:
        try:
            from openai import OpenAI
            llm_client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_key,
            )
            _log("[vision] GROQ_API_KEY found → dùng meta-llama/llama-4-scout-17b-16e-instruct")
            return MarkItDown(
                llm_client=llm_client,
                llm_model="meta-llama/llama-4-scout-17b-16e-instruct",
            )
        except Exception as e:
            _log(f"[vision] Không khởi tạo được Groq client: {e} → text-only mode")

    _log("[vision] Không có GROQ_API_KEY → text-only mode (ảnh bị bỏ qua)")
    return MarkItDown()


# Singleton converter — khởi tạo 1 lần
_converter: MarkItDown | None = None

def _get_converter() -> MarkItDown:
    global _converter
    if _converter is None:
        _converter = _build_converter()
    return _converter

def reset_converter():
    """Force re-initialization of the converter (e.g. after env change)."""
    global _converter
    _converter = None


# ── Main public API ─────────────────────────────────────────────────

def parse_to_markdown(file_path: str) -> dict:
    """
    Parse file bất kỳ → dict chứa markdown + metadata.

    Returns:
        {
            "markdown"   : str,   # toàn bộ nội dung dạng markdown
            "file_name"  : str,
            "file_type"  : str,   # "PDF" | "PowerPoint" | ...
            "char_count" : int,
            "has_images" : bool,  # có ảnh được mô tả bởi vision model
            "image_count": int,
        }
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File không tồn tại: {file_path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Format '{path.suffix}' chưa được hỗ trợ. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    converter = _get_converter()
    result    = converter.convert(str(path))
    markdown  = result.text_content or ""

    # Đếm ảnh đã được mô tả (MarkItDown bọc mô tả ảnh trong ![...](...))
    image_matches = re.findall(r"!\[.*?\]\(.*?\)", markdown)
    image_count   = len(image_matches)

    return {
        "markdown"   : markdown,
        "file_name"  : path.name,
        "file_type"  : _file_type_label(path.suffix),
        "char_count" : len(markdown),
        "has_images" : image_count > 0,
        "image_count": image_count,
    }


def parse_to_pages(file_path: str) -> list[dict]:
    """
    Parse file → List[page_dict] tương thích với pipeline cũ.
    Split markdown theo heading ## (title-based).

    Returns:
        List[{page_num, title, text, char_count, has_image}]
    """
    result   = parse_to_markdown(file_path)
    markdown = result["markdown"]
    sections = _split_by_heading(markdown)

    pages = []
    for i, (title, body) in enumerate(sections, start=1):
        text       = body.strip()
        char_count = len(text)
        if char_count < 30:          # bỏ section rỗng / quá ngắn
            continue
        has_image = bool(re.search(r"!\[.*?\]\(.*?\)", text))
        pages.append({
            "page_num"  : i,
            "title"     : title,
            "text"      : text,
            "char_count": char_count,
            "has_image" : has_image,
        })

    return pages


# ── Helpers ─────────────────────────────────────────────────────────

def _split_by_heading(markdown: str) -> list[tuple[str, str]]:
    """
    Split markdown theo thứ tự ưu tiên:
      1. <!-- Slide number: N --> (PPTX / PDF có slide structure)
      2. # hoặc ## heading
      3. Fallback: toàn bộ text là 1 section "Document"
    Trả về List[(title, body_text)].
    """
    # ── Ưu tiên 1: slide comment từ PPTX ───────────────────────────
    slide_pattern = re.compile(
        r"<!--\s*Slide number:\s*\d+\s*-->", re.IGNORECASE
    )
    slide_markers = list(slide_pattern.finditer(markdown))

    if slide_markers:
        sections = []
        for idx, marker in enumerate(slide_markers):
            chunk_start = marker.end()
            chunk_end   = slide_markers[idx + 1].start() if idx + 1 < len(slide_markers) else len(markdown)
            body        = markdown[chunk_start:chunk_end].strip()

            # Lấy dòng # heading đầu tiên trong slide làm title
            heading_match = re.search(r"^#{1,3}\s+(.+)$", body, re.MULTILINE)
            if heading_match:
                title = _clean_title(heading_match.group(1))
                # Bỏ dòng heading ra khỏi body để không bị lặp
                body  = body[heading_match.end():].strip()
            else:
                # Lấy dòng đầu không rỗng làm title
                first_line = next((l.strip() for l in body.splitlines() if l.strip()), "")
                title = _clean_title(first_line) if first_line else f"Slide {idx + 1}"

            sections.append((title, body))
        return sections

    # ── Ưu tiên 2: markdown heading # / ## ─────────────────────────
    pattern = re.compile(r"^(#{1,2})\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(markdown))

    if matches:
        sections = []
        for idx, match in enumerate(matches):
            title      = _clean_title(match.group(2))
            body_start = match.end()
            body_end   = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
            body       = markdown[body_start:body_end].strip()
            sections.append((title, body))
        return sections

    # ── Fallback: 1 section duy nhất ───────────────────────────────
    return [("Document", markdown)]


def _clean_title(raw: str) -> str:
    """Normalize title: bỏ newline, khoảng trắng thừa, giới hạn 80 ký tự."""
    cleaned = " ".join(raw.split())   # collapse mọi whitespace kể cả \n
    return cleaned[:80]


def _file_type_label(suffix: str) -> str:
    return {
        ".pdf" : "PDF",
        ".pptx": "PowerPoint",
        ".docx": "Word",
        ".xlsx": "Excel",
        ".xls" : "Excel",
        ".jpg" : "Image", ".jpeg": "Image",
        ".png" : "Image", ".gif" : "Image",
        ".bmp" : "Image", ".webp": "Image",
    }.get(suffix.lower(), "Unknown")


def _log(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


# ── CLI test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("Usage: python pipeline/markitdown_parser.py <file>")
        print(f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)

    file_path = sys.argv[1]

    print(f"\n{'='*60}")
    print(f"  File      : {file_path}")
    print(f"{'='*60}")

    # ── Test parse_to_markdown ──────────────────────────────────────
    print("\n[1] parse_to_markdown()")
    print("-" * 40)
    md_result = parse_to_markdown(file_path)

    print(f"  file_type  : {md_result['file_type']}")
    print(f"  char_count : {md_result['char_count']:,}")
    print(f"  has_images : {md_result['has_images']}")
    print(f"  image_count: {md_result['image_count']}")
    print(f"\n--- Markdown preview (first 800 chars) ---")
    print(md_result["markdown"][:800])
    if len(md_result["markdown"]) > 800:
        print(f"\n... ({len(md_result['markdown']) - 800:,} more chars)")

    # ── Test parse_to_pages ─────────────────────────────────────────
    print(f"\n\n[2] parse_to_pages()")
    print("-" * 40)
    pages = parse_to_pages(file_path)
    print(f"  Total sections: {len(pages)}\n")

    print(f"  {'NUM':<4} {'HAS_IMG':<8} {'CHARS':<7} TITLE")
    print(f"  {'-'*55}")
    for p in pages:
        img_flag = "YES" if p["has_image"] else "-"
        title    = p["title"][:42]
        print(f"  {p['page_num']:<4} {img_flag:<8} {p['char_count']:<7} {title}")

    # ── Sample text từ section đầu tiên ────────────────────────────
    if pages:
        print(f"\n--- Sample text (section 1: '{pages[0]['title']}') ---")
        print(pages[0]["text"][:500])
        if len(pages[0]["text"]) > 500:
            print(f"... ({len(pages[0]['text']) - 500} more chars)")

    # ── Auto checks ─────────────────────────────────────────────────
    print(f"\n--- Auto checks ---")
    checks_passed = True

    if md_result["char_count"] > 0:
        print(f"  [OK] Markdown không rỗng ({md_result['char_count']:,} chars)")
    else:
        print(f"  [FAIL] Markdown rỗng — kiểm tra lại file hoặc MarkItDown install")
        checks_passed = False

    if len(pages) > 0:
        print(f"  [OK] Tách được {len(pages)} section(s)")
    else:
        print(f"  [WARN] Không tách được section — không có heading ## trong markdown")

    short_pages = [p for p in pages if p["char_count"] < 50]
    if short_pages:
        print(f"  [WARN] {len(short_pages)} section quá ngắn (< 50 chars) — đã bị bỏ qua")
    else:
        print(f"  [OK] Tất cả section đủ nội dung")

    img_sections = [p for p in pages if p["has_image"]]
    if img_sections:
        print(f"  [OK] {len(img_sections)} section có ảnh được mô tả")
    elif md_result["has_images"]:
        print(f"  [INFO] Có ảnh trong file nhưng chưa được mô tả (cần GROQ_API_KEY)")
    else:
        print(f"  [INFO] Không phát hiện ảnh trong file")

    print(f"\n{'OK' if checks_passed else 'FAILED'} — Done.\n")
