"""
pipeline/markitdown_parser.py
-----------------------------
Universal file parser: PPTX / DOCX / XLSX / Images → Markdown → pages

Vision provider chain (fallback tự động khi rate limit):
  1. Groq          (GROQ_API_KEY)
  2. Cerebras      (CEREBRAS_API_KEY)
  3. OpenRouter    (OPENROUTER_API_KEY)
  4. text-only     (luôn có, fallback cuối)

Fixes:
  #5  split section quá dài theo paragraph boundary
  #6  provider fallback chain khi rate limit (429)
  #8  _split_by_heading bỏ qua dòng alt-text ảnh khi lấy title
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from markitdown import MarkItDown

load_dotenv()

SUPPORTED_EXTENSIONS = {
    ".pdf", ".pptx", ".docx",
    ".xlsx", ".xls",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
}

MAX_SECTION_CHARS = 6000   # #5: section > ngưỡng này sẽ bị split theo paragraph


# ── Fix #6: vision provider chain ────────────────────────────────────

_PROVIDERS: list[tuple[str, MarkItDown]] | None = None


def _build_providers() -> list[tuple[str, MarkItDown]]:
    """
    Xây danh sách (provider_name, MarkItDown) theo thứ tự ưu tiên.
    Chỉ thêm provider nào có API key. Text-only luôn ở cuối làm fallback.
    """
    providers: list[tuple[str, MarkItDown]] = []

    # 1. Groq
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_key,
            )
            providers.append(("groq", MarkItDown(
                llm_client=client,
                llm_model="meta-llama/llama-4-scout-17b-16e-instruct",
            )))
            _log("[vision] Groq provider sẵn sàng")
        except Exception as e:
            _log(f"[vision] Groq init thất bại: {e}")

    # 2. OpenRouter (Cerebras không có vision model → bỏ qua)
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
            )
            providers.append(("openrouter", MarkItDown(
                llm_client=client,
                llm_model="meta-llama/llama-3.2-11b-vision-instruct",
            )))
            _log("[vision] OpenRouter provider sẵn sàng")
        except Exception as e:
            _log(f"[vision] OpenRouter init thất bại: {e}")

    # Fallback cuối: text-only, không cần key
    providers.append(("text-only", MarkItDown()))

    if len(providers) == 1:
        _log("[vision] Không có API key nào → text-only mode")

    return providers


def _get_providers() -> list[tuple[str, MarkItDown]]:
    global _PROVIDERS
    if _PROVIDERS is None:
        _PROVIDERS = _build_providers()
    return _PROVIDERS


def reset_providers():
    """Force re-init providers (ví dụ sau khi thay đổi env)."""
    global _PROVIDERS
    _PROVIDERS = None


def _convert_with_fallback(path: Path) -> str:
    """
    Thử convert file qua từng provider theo thứ tự.
    Nếu gặp rate limit (429) → chuyển sang provider tiếp theo tự động.
    """
    for name, converter in _get_providers():
        try:
            result = converter.convert(str(path))
            if name != "text-only":
                _log(f"[vision] dùng {name}")
            return result.text_content or ""
        except Exception as e:
            err_str = str(e).lower()
            is_skippable = (
                "429" in err_str
                or "rate_limit" in err_str
                or "ratelimit" in err_str
                or "rate limit" in err_str
                or "404" in err_str
                or "not_found" in err_str
                or "model_not_found" in err_str
                or "does not exist" in err_str
                or "do not have access" in err_str
            )
            if is_skippable and name != "text-only":
                _log(f"[vision] {name}: {type(e).__name__} → thử provider tiếp theo")
                continue
            raise

    # Tất cả vision provider đều rate limit → text-only hard fallback
    _log(f"[vision] Tất cả provider đều rate limit → text-only fallback")
    result = MarkItDown().convert(str(path))
    return result.text_content or ""


# ── Main public API ───────────────────────────────────────────────────

def parse_to_markdown(file_path: str) -> dict:
    """
    Parse file → dict chứa markdown + metadata.
    Returns: {markdown, file_name, file_type, char_count, has_images, image_count}
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File không tồn tại: {file_path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Format '{path.suffix}' chưa được hỗ trợ. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    markdown     = _convert_with_fallback(path)
    image_count  = len(re.findall(r"!\[.*?\]\(.*?\)", markdown))

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
    Parse file → List[page_dict] tương thích với pipeline.
    Returns: List[{page_num, title, text, char_count, has_image}]
    """
    result   = parse_to_markdown(file_path)
    markdown = result["markdown"]
    sections = _split_by_heading(markdown)

    pages = []
    for i, (title, body) in enumerate(sections, start=1):
        text = body.strip()
        if len(text) < 30:
            continue
        pages.append({
            "page_num"  : i,
            "title"     : title,
            "text"      : text,
            "char_count": len(text),
            "has_image" : bool(re.search(r"!\[.*?\]\(.*?\)", text)),
        })

    return pages


# ── Fix #5 + #8: split by heading ────────────────────────────────────

def _split_by_heading(markdown: str) -> list[tuple[str, str]]:
    """
    Split markdown theo thứ tự ưu tiên:
      1. <!-- Slide number: N --> (PPTX)
      2. # / ## heading
      3. Paragraph split nếu nội dung quá dài (#5)
      4. Fallback: toàn bộ là 1 section "Document"

    #8: khi lấy title từ first_line, bỏ qua dòng bắt đầu bằng '!['.
    """
    # ── Ưu tiên 1: slide comment từ PPTX ─────────────────────────────
    slide_pattern = re.compile(r"<!--\s*Slide number:\s*\d+\s*-->", re.IGNORECASE)
    slide_markers = list(slide_pattern.finditer(markdown))

    if slide_markers:
        sections = []
        for idx, marker in enumerate(slide_markers):
            chunk_start = marker.end()
            chunk_end   = slide_markers[idx + 1].start() if idx + 1 < len(slide_markers) else len(markdown)
            body        = markdown[chunk_start:chunk_end].strip()

            heading_match = re.search(r"^#{1,3}\s+(.+)$", body, re.MULTILINE)
            if heading_match:
                title = _clean_title(heading_match.group(1))
                body  = body[heading_match.end():].strip()
            else:
                # #8: bỏ qua dòng alt-text khi lấy title
                first_line = next(
                    (l.strip() for l in body.splitlines()
                     if l.strip() and not l.strip().startswith("![")),
                    ""
                )
                title = _clean_title(first_line) if first_line else f"Slide {idx + 1}"

            sections.append((title, body))
        return sections

    # ── Ưu tiên 2: markdown heading # / ## ───────────────────────────
    pattern = re.compile(r"^(#{1,2})\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(markdown))

    if matches:
        sections = []
        for idx, match in enumerate(matches):
            # #8: bỏ title là alt-text
            raw_title = match.group(2).strip()
            if raw_title.startswith("!["):
                # tìm dòng text tiếp theo không phải alt-text làm title
                body_start = match.end()
                body_end   = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
                body       = markdown[body_start:body_end].strip()
                first_line = next(
                    (l.strip() for l in body.splitlines()
                     if l.strip() and not l.strip().startswith("![")),
                    ""
                )
                title = _clean_title(first_line) if first_line else f"Section {idx + 1}"
            else:
                title = _clean_title(raw_title)
                body_start = match.end()
                body_end   = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
                body       = markdown[body_start:body_end].strip()

            sections.append((title, body))
        return sections

    # ── Fallback: split theo paragraph nếu dài, không thì 1 section ──
    if len(markdown) > MAX_SECTION_CHARS:
        _log(f"  [markitdown] không có heading, nội dung dài ({len(markdown)} chars) → split theo paragraph")
        return _split_by_paragraphs(markdown)

    return [("Document", markdown)]


def _split_by_paragraphs(markdown: str) -> list[tuple[str, str]]:
    """
    #5: Split nội dung không có heading thành nhiều sections theo paragraph.
    Title của mỗi section = dòng text đầu tiên của đoạn đó.
    """
    paragraphs = re.split(r"\n{2,}", markdown.strip())
    sections: list[tuple[str, str]] = []
    current_parts: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > MAX_SECTION_CHARS and current_parts:
            body  = "\n\n".join(current_parts)
            title = _first_meaningful_line(body)
            sections.append((title, body))
            current_parts = [para]
            current_len   = len(para)
        else:
            current_parts.append(para)
            current_len += len(para)

    if current_parts:
        body  = "\n\n".join(current_parts)
        title = _first_meaningful_line(body)
        sections.append((title, body))

    return sections if sections else [("Document", markdown)]


def _first_meaningful_line(text: str) -> str:
    """Lấy dòng text đầu tiên có nghĩa, bỏ qua alt-text / list / table."""
    for line in text.splitlines():
        line = line.strip()
        # #8: bỏ qua alt-text
        if not line or line.startswith(("![", "-", "*", "|", "#")):
            continue
        clean = re.sub(r"[*_`#]", "", line).strip()
        if len(clean) > 3:
            return clean[:80]
    return "Section"


def _clean_title(raw: str) -> str:
    cleaned = " ".join(raw.split())
    # #8: nếu sau clean vẫn là alt-text thì strip markdown
    if cleaned.startswith("!["):
        cleaned = re.sub(r"!\[.*?\]\(.*?\)", "", cleaned).strip()
    return cleaned[:80]


# ── Helpers ───────────────────────────────────────────────────────────

def _file_type_label(suffix: str) -> str:
    return {
        ".pdf" : "PDF",
        ".pptx": "PowerPoint",
        ".docx": "Word",
        ".xlsx": "Excel",  ".xls": "Excel",
        ".jpg" : "Image",  ".jpeg": "Image",
        ".png" : "Image",  ".gif" : "Image",
        ".bmp" : "Image",  ".webp": "Image",
    }.get(suffix.lower(), "Unknown")


def _log(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


# ── CLI test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("Usage: python pipeline/markitdown_parser.py <file>")
        print(f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)

    file_path = sys.argv[1]

    print(f"\n{'='*60}")
    print(f"  File : {file_path}")
    print(f"{'='*60}")

    print("\n[1] parse_to_markdown()")
    print("-" * 40)
    md_result = parse_to_markdown(file_path)
    print(f"  file_type  : {md_result['file_type']}")
    print(f"  char_count : {md_result['char_count']:,}")
    print(f"  has_images : {md_result['has_images']}")
    print(f"  image_count: {md_result['image_count']}")
    print(f"\n--- Markdown preview (800 chars) ---")
    print(md_result["markdown"][:800])

    print(f"\n[2] parse_to_pages()")
    print("-" * 40)
    pages = parse_to_pages(file_path)
    print(f"  Total sections: {len(pages)}\n")
    print(f"  {'NUM':<4} {'HAS_IMG':<8} {'CHARS':<7} TITLE")
    print(f"  {'-'*55}")
    for p in pages:
        img_flag = "YES" if p["has_image"] else "-"
        print(f"  {p['page_num']:<4} {img_flag:<8} {p['char_count']:<7} {p['title'][:42]}")

    if pages:
        print(f"\n--- Sample text (section 1) ---")
        print(pages[0]["text"][:500])

    print(f"\n--- Auto checks ---")
    assert len(pages) > 0 or md_result["char_count"] == 0, "FAIL: 0 sections với file có nội dung"
    alt_titles   = [p for p in pages if p["title"].startswith("![")]
    long_sections = [p for p in pages if p["char_count"] > MAX_SECTION_CHARS]
    print(f"  Title là alt-text  : {len(alt_titles)}/{len(pages)}")
    print(f"  Section quá dài    : {len(long_sections)}/{len(pages)}  (ngưỡng {MAX_SECTION_CHARS})")
    print(f"\nDone.\n")
