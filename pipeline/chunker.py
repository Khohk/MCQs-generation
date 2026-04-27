"""
pipeline/chunker.py
-------------------
List[page_dict] → List[{chunk_id, chunk_type, topic, pages, text, has_image}]

Strategies:
  - "title"   : gom pages liên tiếp có cùng title (tốt với document có heading rõ)
  - "overlap" : gom theo keyword overlap giữa trang liền kề (tốt với slide deck)
  - "fixed"   : cứng N trang/chunk
  - "auto"    : thử title → nếu đơn điệu hoặc slide deck → overlap

chunk_type (do _classify_chunk_type gán):
  - "conceptual"   : nội dung kiến thức → dùng để gen MCQ
  - "structural"   : tiêu đề, tổng quan, Q&A → skip khi gen MCQ
  - "instructional": bài tập, hands-on, demo → skip hoặc dùng cho loại câu hỏi khác
  - "stub"         : quá ngắn hoặc chủ yếu là ảnh → skip

Fixes:
  #1  tag chunk_type cho mọi chunk
  #2  merge image chunk yếu vào chunk liền kề
  #4  tag stub thay vì merge mù
  #7  _chunk_by_overlap thay thế fixed(3) khi slide deck
"""

from __future__ import annotations

import re

# ── Constants ─────────────────────────────────────────────────────────
MIN_CHUNK_CHARS    = 150
MAX_PAGES_PER_CHUNK = 6
OVERLAP_THRESHOLD  = 0.12   # #7: ít nhất 12% từ chung → giữ trong cùng chunk
MIN_IMAGE_CHARS    = 200    # #2: image chunk có text < ngưỡng này → merge

# ── Keywords cho chunk_type ───────────────────────────────────────────
_STRUCTURAL_KW = [
    "q&a", "conclusion", "learning objectives", "learning outcomes",
    "agenda", "table of contents", "references", "bibliography",
    "acknowledgement", "thank you", "introduction",
    "mục tiêu học", "kết luận", "tài liệu tham khảo", "câu hỏi", "ôn tập",
]
_INSTRUCTIONAL_KW = [
    "hands-on", "hands on", "exercise", "practice", "lab",
    "assignment", "workshop", "tutorial", "demo", "demonstration",
    "tools to explore", "best practice", "recommended resources", "further reading",
    "learning resources", "further resources",
    "thực hành", "bài tập",
]


# ── Main function ─────────────────────────────────────────────────────

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
        pages      : output từ pdf_parser / markitdown_parser
        strategy   : "title" | "overlap" | "fixed" | "auto"
        max_pages  : số trang tối đa trong 1 chunk
        min_chars  : ngưỡng để tag chunk là "stub"
        fixed_size : số trang/chunk khi dùng strategy="fixed"

    Returns:
        List[{chunk_id, chunk_type, topic, pages, text, has_image}]
    """
    if not pages:
        return []

    pages = _dedup_pages(pages)

    if strategy == "auto":
        chunks = _chunk_by_title(pages, max_pages)
        unique_topics = len(set(c["topic"] for c in chunks))
        total_chunks  = len(chunks)
        ratio = unique_topics / total_chunks if total_chunks > 0 else 0

        if ratio < 0.4:
            _log(f"  title: {unique_topics}/{total_chunks} unique → switch to overlap")
            chunks = _chunk_by_overlap(pages, max_pages)
        elif ratio > 0.85 and total_chunks > 15:
            _log(f"  title: {unique_topics}/{total_chunks} unique (slide deck) → switch to overlap")
            chunks = _chunk_by_overlap(pages, max_pages)
        else:
            _log(f"  title: {unique_topics}/{total_chunks} unique [OK]")

    elif strategy == "title":
        chunks = _chunk_by_title(pages, max_pages)
    elif strategy == "overlap":
        chunks = _chunk_by_overlap(pages, max_pages)
    elif strategy == "fixed":
        chunks = _chunk_fixed(pages, fixed_size, max_pages)
    else:
        raise ValueError(f"strategy phải là 'title'|'overlap'|'fixed'|'auto'. Nhận: {strategy}")

    # #2: merge image chunk yếu vào chunk liền kề
    chunks = _handle_weak_image_chunks(chunks)

    # #4 + #1: tag chunk_type (thay vì merge mù)
    for chunk in chunks:
        chunk["chunk_type"] = _classify_chunk_type(chunk, min_chars)

    # Gán chunk_id
    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"chunk_{i+1:03d}"

    return chunks


# ── Strategies ────────────────────────────────────────────────────────

def _dedup_pages(pages: list[dict]) -> list[dict]:
    """Bỏ page có text trùng > 95% với page trước (window 3)."""
    if not pages:
        return pages
    seen  = [pages[0]["text"].strip()]
    unique = [pages[0]]
    for page in pages[1:]:
        current = page["text"].strip()
        if any(_similarity(current, s) > 0.95 for s in seen[-3:]):
            _log(f"  page {page['page_num']}: DEDUP")
        else:
            unique.append(page)
            seen.append(current)
    return unique


def _chunk_by_title(pages: list[dict], max_pages: int) -> list[dict]:
    """Gom pages liên tiếp có cùng normalized title."""
    if not pages:
        return []
    chunks        = []
    current_pages = [pages[0]]
    current_title = _normalize_title(pages[0]["title"])

    for page in pages[1:]:
        norm = _normalize_title(page["title"])
        if norm == current_title and len(current_pages) < max_pages:
            current_pages.append(page)
        else:
            chunks.append(_build_chunk(current_pages))
            current_pages = [page]
            current_title = norm

    if current_pages:
        chunks.append(_build_chunk(current_pages))
    return chunks


def _chunk_by_overlap(pages: list[dict], max_pages: int) -> list[dict]:
    """
    #7: Gom pages theo keyword overlap thay vì fixed(N).
    So sánh page mới với 2 trang gần nhất của chunk hiện tại.
    Nếu similarity >= OVERLAP_THRESHOLD → tiếp tục chunk, ngược lại → chunk mới.
    """
    if not pages:
        return []
    chunks  = []
    current = [pages[0]]

    for page in pages[1:]:
        context = " ".join(p["text"] for p in current[-2:])
        sim     = _similarity(context, page["text"])
        if sim >= OVERLAP_THRESHOLD and len(current) < max_pages:
            current.append(page)
        else:
            chunks.append(_build_chunk(current))
            current = [page]

    if current:
        chunks.append(_build_chunk(current))
    return chunks


def _chunk_fixed(pages: list[dict], size: int, max_pages: int) -> list[dict]:
    """Cứng N trang/chunk."""
    step   = min(size, max_pages)
    chunks = []
    for i in range(0, len(pages), step):
        chunks.append(_build_chunk(pages[i:i + step]))
    return chunks


# ── Fix #2: weak image chunks ─────────────────────────────────────────

def _handle_weak_image_chunks(chunks: list[dict]) -> list[dict]:
    """
    Merge chunk có ảnh nhưng text thực quá ít vào chunk liền kề.
    Điều kiện: has_image=True AND text < MIN_IMAGE_CHARS AND chủ yếu là placeholder ảnh.
    """
    if len(chunks) <= 1:
        return chunks

    result: list[dict] = []
    for i, chunk in enumerate(chunks):
        if _is_weak_image_chunk(chunk):
            if result:
                _merge_into(result[-1], chunk)
                _log(f"  merge image chunk yếu ({len(chunk['text'])} chars) → chunk trước")
            elif i + 1 < len(chunks):
                _merge_into_front(chunks[i + 1], chunk)
                _log(f"  merge image chunk yếu ({len(chunk['text'])} chars) → chunk sau")
            else:
                result.append(chunk)
        else:
            result.append(chunk)
    return result


def _is_weak_image_chunk(chunk: dict) -> bool:
    if not chunk.get("has_image"):
        return False
    if len(chunk["text"]) >= MIN_IMAGE_CHARS:
        return False
    lines = [l.strip() for l in chunk["text"].splitlines() if l.strip()]
    if not lines:
        return True
    image_lines = sum(1 for l in lines if l.startswith("![") or "intentionally omitted" in l)
    return image_lines / len(lines) > 0.6


def _merge_into(target: dict, source: dict):
    target["text"]     += "\n\n" + source["text"]
    target["pages"]    += source["pages"]
    target["has_image"] = True


def _merge_into_front(target: dict, source: dict):
    target["text"]     = source["text"] + "\n\n" + target["text"]
    target["pages"]    = source["pages"] + target["pages"]
    target["has_image"] = True


# ── Fix #1 + #4: classify chunk_type ─────────────────────────────────

def _classify_chunk_type(chunk: dict, min_chars: int) -> str:
    """
    Gán chunk_type:
      stub         — quá ngắn hoặc image chunk yếu
      structural   — tiêu đề, tổng quan, Q&A (không có kiến thức để gen MCQ)
      instructional— bài tập, hands-on, demo
      conceptual   — nội dung kiến thức, dùng để gen MCQ
    """
    if len(chunk["text"]) < min_chars:
        return "stub"

    topic_lower = chunk["topic"].lower()
    # Bỏ markdown formatting trước khi match
    topic_clean = re.sub(r"[*_`#~]", "", topic_lower).strip()

    # Chunk dài (>3000 chars) là nội dung thực — bỏ qua structural KW
    # để tránh nhầm heading chương ("1. Tổng quan về X") với slide overview
    if len(chunk["text"]) <= 3000:
        for kw in _STRUCTURAL_KW:
            if kw in topic_clean:
                return "structural"

    for kw in _INSTRUCTIONAL_KW:
        if kw in topic_clean:
            return "instructional"

    return "conceptual"


# ── Helpers ───────────────────────────────────────────────────────────

def _build_chunk(pages: list[dict]) -> dict:
    topic     = pages[0]["title"] if pages else "Unknown"
    text      = "\n\n".join(p["text"] for p in pages)
    page_nums = [p["page_num"] for p in pages]
    has_image = any(p.get("has_image", False) for p in pages)
    return {
        "chunk_id"  : "",
        "chunk_type": "",        # gán sau bởi _classify_chunk_type
        "topic"     : topic,
        "pages"     : page_nums,
        "text"      : text,
        "has_image" : has_image,
    }


def _similarity(a: str, b: str) -> float:
    """Jaccard similarity trên word set."""
    if not a or not b:
        return 0.0
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _normalize_title(title: str) -> str:
    """Lowercase + bỏ số thứ tự đầu dòng + bỏ markdown."""
    t = re.sub(r"[*_`#~]", "", title).lower().strip()
    t = re.sub(r"^(\d+[\.\)]\s*|chapter\s+\d+\s*[\:\-]?\s*)", "", t)
    return t.strip()


def _log(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


# ── CLI test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.file_router import parse_file, get_file_type

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("Usage: python pipeline/chunker.py <file> [strategy] [max_pages]")
        sys.exit(1)

    file_path = sys.argv[1]
    strategy  = sys.argv[2] if len(sys.argv) > 2 else "auto"
    max_pages = int(sys.argv[3]) if len(sys.argv) > 3 else MAX_PAGES_PER_CHUNK

    print(f"\n{'='*65}")
    print(f"  File     : {file_path}")
    print(f"  Type     : {get_file_type(file_path)}")
    print(f"  Strategy : {strategy}")
    print(f"{'='*65}\n")

    pages  = parse_file(file_path)
    chunks = chunk_pages(pages, strategy=strategy, max_pages=max_pages)

    print(f"\n{'='*65}")
    print(f"  Pages  : {len(pages)}")
    print(f"  Chunks : {len(chunks)}  ({len(pages)/len(chunks):.1f} pages/chunk)")
    print(f"{'='*65}\n")

    type_counts = {}
    for c in chunks:
        type_counts[c["chunk_type"]] = type_counts.get(c["chunk_type"], 0) + 1

    print(f"  chunk_type breakdown: {type_counts}")
    print(f"  → dùng được cho MCQ : {type_counts.get('conceptual', 0)} conceptual chunks\n")

    print(f"{'CHUNK':<10} {'TYPE':<14} {'IMG':<5} {'CHARS':<7} TOPIC")
    print("-" * 70)
    for c in chunks:
        pages_str = str(c["pages"]) if len(c["pages"]) <= 3 else f"[{c['pages'][0]}..{c['pages'][-1]}]"
        img_flag  = "Y" if c.get("has_image") else "-"
        print(f"  {c['chunk_id']:<8} {c['chunk_type']:<14} {img_flag:<5} {len(c['text']):<7} {c['topic'][:35]}")

    print("\n--- Auto checks ---")
    all_pages = [p for c in chunks for p in c["pages"]]
    print(f"  No dup pages   : {'[OK]' if len(all_pages) == len(set(all_pages)) else '[FAIL]'}")
    print(f"  chunk_id unique: {'[OK]' if len(set(c['chunk_id'] for c in chunks)) == len(chunks) else '[FAIL]'}")
    conceptual = [c for c in chunks if c["chunk_type"] == "conceptual"]
    print(f"  Conceptual chunks  : {len(conceptual)}/{len(chunks)}")
    print(f"  Avg chars (conceptual): {sum(len(c['text']) for c in conceptual)/len(conceptual):.0f}" if conceptual else "  No conceptual chunks")
    print(f"\nDone.\n")
