"""
pipeline/chunker.py
-------------------
List[page_dict] → List[{chunk_id, topic, pages, text}]

Strategies:
  - "title"  : gom các page liên tiếp có cùng/liên quan title (default)
  - "fixed"  : cứng N trang/chunk
  - "auto"   : thử title trước, fallback sang fixed nếu title quá đơn điệu
"""

from __future__ import annotations


# ── Constants ──────────────────────────────────────────────────────
MIN_CHUNK_CHARS = 150    # chunk quá ngắn → merge với chunk trước
MAX_PAGES_PER_CHUNK = 6  # giới hạn trên để tránh chunk quá dài cho Gemini


# ── Main function ──────────────────────────────────────────────────

def chunk_pages(
    pages: list[dict],
    strategy: str = "auto",
    max_pages: int = MAX_PAGES_PER_CHUNK,
    min_chars: int = MIN_CHUNK_CHARS,
    fixed_size: int = 4,
) -> list[dict]:
    """
    Gom pages thành chunks ngữ nghĩa.

    Args:
        pages      : output từ pdf_parser.parse_pdf()
        strategy   : "title" | "fixed" | "auto"
        max_pages  : số trang tối đa trong 1 chunk
        min_chars  : chunk ngắn hơn N chars sẽ bị merge vào chunk trước
        fixed_size : số trang/chunk khi dùng strategy="fixed"

    Returns:
        List[{chunk_id, topic, pages, text}]
    """
    if not pages:
        return []

    # Dedup: bỏ các page có nội dung trùng hoàn toàn với page trước
    pages = _dedup_pages(pages)

    if strategy == "auto":
        chunks = _chunk_by_title(pages, max_pages)
        unique_topics = len(set(c["topic"] for c in chunks))
        total_chunks = len(chunks)
        # Nếu title quá đơn điệu (< 40% unique) → dùng fixed
        if total_chunks > 0 and unique_topics / total_chunks < 0.4:
            _log(f"  title strategy: chỉ {unique_topics}/{total_chunks} unique topics → switch to fixed")
            chunks = _chunk_fixed(pages, fixed_size, max_pages)
        else:
            _log(f"  title strategy: {unique_topics}/{total_chunks} unique topics [OK]")
    elif strategy == "title":
        chunks = _chunk_by_title(pages, max_pages)
    elif strategy == "fixed":
        chunks = _chunk_fixed(pages, fixed_size, max_pages)
    else:
        raise ValueError(f"strategy phải là 'title', 'fixed', hoặc 'auto'. Nhận: {strategy}")

    # Merge chunk quá ngắn
    chunks = _merge_short_chunks(chunks, min_chars)

    # Gán chunk_id chuẩn
    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"chunk_{i+1:03d}"

    return chunks


# ── Strategies ─────────────────────────────────────────────────────

def _dedup_pages(pages: list[dict]) -> list[dict]:
    """
    Bỏ các page có text trùng hoàn toàn hoặc gần giống (> 95%) với page trước.
    Giữ lại page đầu tiên, log những page bị bỏ.
    """
    if not pages:
        return pages

    seen_texts = []
    unique = [pages[0]]
    seen_texts.append(pages[0]["text"].strip())

    for page in pages[1:]:
        current = page["text"].strip()
        is_dup = any(
            _similarity(current, seen) > 0.95
            for seen in seen_texts[-3:]  # chỉ so với 3 page gần nhất
        )
        if is_dup:
            _log(f"  page {page['page_num']}: DEDUP (trùng nội dung với page trước)")
        else:
            unique.append(page)
            seen_texts.append(current)

    return unique


def _similarity(a: str, b: str) -> float:
    """Tính độ giống nhau đơn giản dựa trên ký tự chung (Jaccard trên trigrams)."""
    if not a or not b:
        return 0.0
    # Dùng set của words thay vì trigrams — đủ nhanh và chính xác cho slide
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _chunk_by_title(pages: list[dict], max_pages: int) -> list[dict]:
    """
    Gom các trang liên tiếp có title giống nhau vào 1 chunk.
    Nếu 1 chunk vượt max_pages → split.
    """
    if not pages:
        return []

    chunks = []
    current_pages = [pages[0]]
    current_title = _normalize_title(pages[0]["title"])

    for page in pages[1:]:
        norm_title = _normalize_title(page["title"])
        same_topic = (norm_title == current_title)
        too_long = (len(current_pages) >= max_pages)

        if same_topic and not too_long:
            current_pages.append(page)
        else:
            chunks.append(_build_chunk(current_pages))
            current_pages = [page]
            current_title = norm_title

    # Flush chunk cuối
    if current_pages:
        chunks.append(_build_chunk(current_pages))

    return chunks


def _chunk_fixed(pages: list[dict], size: int, max_pages: int) -> list[dict]:
    """Cứng N trang/chunk."""
    step = min(size, max_pages)
    chunks = []
    for i in range(0, len(pages), step):
        group = pages[i:i + step]
        chunks.append(_build_chunk(group))
    return chunks


# ── Helpers ────────────────────────────────────────────────────────

def _build_chunk(pages: list[dict]) -> dict:
    """Tạo chunk dict từ list pages."""
    # Topic = title của trang đầu tiên trong chunk
    topic = pages[0]["title"] if pages else "Unknown"
    text = "\n\n".join(p["text"] for p in pages)
    page_nums = [p["page_num"] for p in pages]
    return {
        "chunk_id": "",          # sẽ gán lại ở cuối
        "topic":    topic,
        "pages":    page_nums,
        "text":     text,
    }


def _merge_short_chunks(chunks: list[dict], min_chars: int) -> list[dict]:
    """
    Merge chunk quá ngắn (< min_chars) vào chunk liền trước.
    Nếu là chunk đầu tiên → merge vào chunk sau.
    """
    if len(chunks) <= 1:
        return chunks

    merged = [chunks[0]]
    for chunk in chunks[1:]:
        if len(chunk["text"]) < min_chars:
            # Merge vào chunk trước
            prev = merged[-1]
            prev["text"] += "\n\n" + chunk["text"]
            prev["pages"] += chunk["pages"]
            _log(f"  merge chunk ngắn ({len(chunk['text'])} chars) vào chunk trước")
        else:
            merged.append(chunk)

    # Edge case: chunk đầu tiên quá ngắn nhưng đã không được merge
    if len(merged) > 1 and len(merged[0]["text"]) < min_chars:
        second = merged[1]
        second["text"] = merged[0]["text"] + "\n\n" + second["text"]
        second["pages"] = merged[0]["pages"] + second["pages"]
        merged = merged[1:]

    return merged


def _normalize_title(title: str) -> str:
    """
    Chuẩn hóa title để so sánh:
    - lowercase
    - bỏ số ở đầu (vd "1. Introduction" → "introduction")
    - bỏ khoảng trắng thừa
    """
    t = title.lower().strip()
    # Bỏ số thứ tự đầu dòng: "1.", "1)", "Chapter 1", v.v.
    import re
    t = re.sub(r"^(\d+[\.\)]\s*|chapter\s+\d+\s*[\:\-]?\s*)", "", t)
    return t.strip()


def _log(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


# ── CLI test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pipeline.pdf_parser import parse_pdf

    # Fix encoding trên Windows terminal
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("Usage: python pipeline/chunker.py <path_to_pdf> [strategy] [max_pages]")
        print("       strategy: auto | title | fixed  (default: auto)")
        print("       max_pages: int                  (default: 6)")
        sys.exit(1)

    pdf_path   = sys.argv[1]
    strategy   = sys.argv[2] if len(sys.argv) > 2 else "auto"
    max_pages  = int(sys.argv[3]) if len(sys.argv) > 3 else MAX_PAGES_PER_CHUNK

    print(f"\nParsing PDF...")
    pages = parse_pdf(pdf_path)
    print(f"Pages parsed: {len(pages)}\n")

    print(f"Chunking (strategy='{strategy}', max_pages={max_pages})...")
    chunks = chunk_pages(pages, strategy=strategy, max_pages=max_pages)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Pages  : {len(pages)}")
    print(f"  Chunks : {len(chunks)}  (ratio: {len(pages)/len(chunks):.1f} pages/chunk)")
    print(f"{'='*60}\n")

    print(f"{'CHUNK':<12} {'TOPIC':<40} {'PAGES':<15} {'CHARS'}")
    print("-" * 75)
    for c in chunks:
        pages_str = str(c["pages"]) if len(c["pages"]) <= 4 else f"[{c['pages'][0]}..{c['pages'][-1]}]"
        print(f"  {c['chunk_id']:<10} {c['topic'][:38]:<40} {pages_str:<15} {len(c['text'])}")

    # ── Auto checks ──
    print("\n--- Auto checks ---")

    all_page_nums = [p for c in chunks for p in c["pages"]]
    no_dup = len(all_page_nums) == len(set(all_page_nums))
    print(f"  No duplicate pages : {'[OK]' if no_dup else '[FAIL] CO TRANG BI TRUNG'}")

    min_chars = min(len(c["text"]) for c in chunks)
    print(f"  Min chunk chars    : {min_chars}  {'[OK]' if min_chars >= MIN_CHUNK_CHARS else '[WARN] qua ngan'}")

    avg_chars = sum(len(c["text"]) for c in chunks) / len(chunks)
    print(f"  Avg chunk chars    : {avg_chars:.0f}")

    unique_topics = len(set(c["topic"] for c in chunks))
    print(f"  Unique topics      : {unique_topics}/{len(chunks)}  {'[OK]' if unique_topics/len(chunks) > 0.5 else '[WARN] nhieu topic bi trung'}")

    ids = [c["chunk_id"] for c in chunks]
    print(f"  chunk_id unique    : {'[OK]' if len(ids) == len(set(ids)) else '[FAIL]'}")

    print(f"\nDone.\n")