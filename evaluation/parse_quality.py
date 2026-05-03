"""
evaluation/parse_quality.py
---------------------------
Axis 0: Parse Quality Benchmark

Đo chất lượng text extraction per page:
  1. Text extraction quality  — char_count, word_count, single_char_ratio, text_quality, parser_used
  2. Table detection          — has_table flag vs | separator presence (readable vs flattened)
  3. Image/figure handling    — image pages, image-only pages (blind pages)
  4. Layout coherence         — proxy metric: ratio of complete sentences (>= 5 words)

Usage (standalone):
  python evaluation/parse_quality.py data/slides.pdf data/lecture.pptx
  python evaluation/parse_quality.py data/*.pdf --json evaluation/results/parse_quality.json
  python evaluation/parse_quality.py data/slides.pdf --pages   # show per-page detail

Usage (from benchmark.py):
  python evaluation/benchmark.py data/*.pdf --parse
  python evaluation/benchmark.py data/*.pdf --parse --judge
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── Thresholds ─────────────────────────────────────────────────────
IMAGE_ONLY_CHAR_THRESHOLD = 80    # page có ảnh + chars < ngưỡng → blind page
TABLE_MIN_PIPE_LINES      = 2     # cần ít nhất 2 dòng có | để coi table còn cấu trúc
COHERENCE_MIN_WORDS       = 10    # page ít hơn ngưỡng này → bỏ qua coherence score
COMPLETE_SENTENCE_WORDS   = 5     # câu >= N từ → "complete"


# ── Per-page analysis ──────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(text.split())


def _single_char_ratio(text: str) -> float:
    """
    Tỉ lệ từ đơn ký tự alpha / tổng từ.
    Ngưỡng > 0.25 → font tiếng Việt bị corrupt (ví dụ: "M c tiêu" thay vì "Mục tiêu").
    """
    words = text.split()
    if not words:
        return 0.0
    singles = sum(
        1 for w in words
        if len(w.strip(".,;:!?\"'()[]{}")) == 1 and w.isalpha()
    )
    return round(singles / len(words), 3)


def _table_readable(text: str) -> bool:
    """
    Table còn cấu trúc nếu có >= TABLE_MIN_PIPE_LINES dòng với >= 2 ký tự |.
    Nếu chỉ có text thuần (flatten) → trả về False.
    """
    pipe_lines = [l for l in text.splitlines() if l.count("|") >= 2]
    return len(pipe_lines) >= TABLE_MIN_PIPE_LINES


def _is_image_only(page: dict) -> bool:
    """
    Trang bị "blind": có ảnh nhưng không đủ text để sinh MCQ.
    Không bao gồm trang đã qua OCR thành công.
    """
    if page.get("ocr_used", False):
        return False  # OCR đã recover text rồi
    char_count = page.get("char_count", len(page.get("text", "")))
    return page.get("has_image", False) and char_count < IMAGE_ONLY_CHAR_THRESHOLD


def _coherence_score(text: str) -> float:
    """
    Proxy metric cho layout preservation.
    = tỉ lệ "câu đầy đủ" (>= COMPLETE_SENTENCE_WORDS từ) / tổng câu.

    Lý do dùng proxy này:
      - Multi-column bị merge → câu bị cắt đôi → nhiều fragment ngắn → score thấp
      - Bullet hierarchy bị flatten → câu ngắn → score thấp
      - Extract tốt → câu đầy đủ subject+verb → score cao
    """
    raw = re.split(r"(?<=[.!?。])\s+|\n+", text)
    sentences = [s.strip() for s in raw if s.strip()]
    if not sentences:
        return 0.0
    complete = sum(1 for s in sentences if len(s.split()) >= COMPLETE_SENTENCE_WORDS)
    return round(complete / len(sentences), 3)


def analyze_page_extraction(page: dict) -> dict:
    """
    Tính đầy đủ các metrics cho 1 page dict từ pdf_parser / markitdown_parser.

    Input fields dùng:
        page_num, text, char_count, has_image, has_table, has_formula,
        has_columns, text_quality, parser, ocr_used, layout_sensitive
    """
    text       = page.get("text", "")
    char_count = page.get("char_count", len(text))
    wc         = _word_count(text)
    sc_ratio   = _single_char_ratio(text)

    has_table   = page.get("has_table", False)
    table_ok    = _table_readable(text) if has_table else False

    has_image   = page.get("has_image", False)
    img_only    = _is_image_only(page)

    coherence   = _coherence_score(text) if wc >= COHERENCE_MIN_WORDS else None

    return {
        "page_num"         : page.get("page_num", 0),
        "char_count"       : char_count,
        "word_count"       : wc,
        "single_char_ratio": sc_ratio,
        "text_quality"     : page.get("text_quality", "unknown"),
        "parser_used"      : page.get("parser", "unknown"),
        "has_table"        : has_table,
        "table_readable"   : table_ok,
        "has_image"        : has_image,
        "is_image_only"    : img_only,
        "has_formula"      : page.get("has_formula", False),
        "has_columns"      : page.get("has_columns", False),
        "layout_sensitive" : page.get("layout_sensitive", False),
        "ocr_used"         : page.get("ocr_used", False),
        "coherence_score"  : coherence,
    }


# ── Per-file aggregate ─────────────────────────────────────────────

def run_parse_quality(file_path: str, pages: list[dict] | None = None) -> dict:
    """
    Chạy parse + đo quality cho 1 file.

    Args:
        file_path : đường dẫn file gốc (dùng để lấy metadata)
        pages     : nếu đã parse sẵn (ví dụ từ run_axis1), truyền vào để tránh parse 2 lần.
                    Nếu None → tự parse.

    Returns dict gồm aggregate metrics + list per-page metrics.
    """
    from pipeline.file_router import parse_file, get_metadata

    meta = get_metadata(file_path)

    if pages is None:
        pages = parse_file(file_path)

    if not pages:
        return {
            "file"             : Path(file_path).name,
            "total_pages"      : meta.get("total_pages", 0),
            "extracted_pages"  : 0,
            "avg_chars_per_page": 0,
            "table_pages"      : 0, "table_readable": 0, "table_flat": 0,
            "image_pages"      : 0, "image_only_pages": 0,
            "garbled_pages"    : 0, "garbled_ratio": 0.0,
            "avg_layout_coherence": 0.0,
            "parser_dist"      : {}, "quality_dist": {},
            "page_metrics"     : [],
            "error"            : "no pages extracted",
        }

    page_metrics = [analyze_page_extraction(p) for p in pages]
    total        = len(page_metrics)

    # ── Text extraction ────────────────────────────────────────────
    char_counts = [m["char_count"] for m in page_metrics]
    avg_chars   = round(sum(char_counts) / total, 1)

    # ── Table ──────────────────────────────────────────────────────
    table_pages   = [m for m in page_metrics if m["has_table"]]
    readable_tbls = [m for m in table_pages  if m["table_readable"]]
    flat_tbls     = len(table_pages) - len(readable_tbls)

    # ── Image ──────────────────────────────────────────────────────
    image_pages    = [m for m in page_metrics if m["has_image"]]
    image_only_pgs = [m for m in page_metrics if m["is_image_only"]]

    # ── Garbled ────────────────────────────────────────────────────
    garbled_pages = [m for m in page_metrics if m["text_quality"] == "garbled"]

    # ── Layout coherence (chỉ tính trang có đủ text) ───────────────
    coherence_vals = [m["coherence_score"] for m in page_metrics
                      if m["coherence_score"] is not None]
    avg_coherence  = round(sum(coherence_vals) / len(coherence_vals), 3) \
                     if coherence_vals else 0.0

    # ── Distributions ──────────────────────────────────────────────
    parser_dist  = dict(Counter(m["parser_used"] for m in page_metrics))
    quality_dist = dict(Counter(m["text_quality"] for m in page_metrics))

    return {
        "file"               : Path(file_path).name,
        "total_pages"        : meta.get("total_pages", total),
        "extracted_pages"    : total,
        "avg_chars_per_page" : avg_chars,
        # table
        "table_pages"        : len(table_pages),
        "table_readable"     : len(readable_tbls),
        "table_flat"         : flat_tbls,
        # image
        "image_pages"        : len(image_pages),
        "image_only_pages"   : len(image_only_pgs),
        # garbled
        "garbled_pages"      : len(garbled_pages),
        "garbled_ratio"      : round(len(garbled_pages) / total, 3),
        # layout
        "avg_layout_coherence": avg_coherence,
        # distributions
        "parser_dist"        : parser_dist,
        "quality_dist"       : quality_dist,
        # per-page detail
        "page_metrics"       : page_metrics,
    }


# ── Print helpers ──────────────────────────────────────────────────

def print_parse_quality_table(results: list[dict]):
    """In bảng tổng hợp Axis 0 ra stdout."""
    print(f"\n{'═'*112}")
    print("  AXIS 0 — PARSE QUALITY")
    print(f"{'═'*112}")

    h = (
        f"  {'Dataset':<36} {'Pages':>7} {'Avg chr':>8} "
        f"{'Tbl pgs':>8} {'Tbl ok':>7} {'Img-only':>9} "
        f"{'Garbled':>8} {'Coherence':>10}  Parsers"
    )
    print(h)
    print(f"  {'─'*108}")

    for r in results:
        if r.get("error"):
            print(f"  {r['file']:<36}  ERROR: {r['error']}")
            continue

        pages_str  = f"{r['extracted_pages']}/{r['total_pages']}"
        table_str  = f"{r['table_readable']}/{r['table_pages']}"
        parser_str = ", ".join(
            f"{k}:{v}" for k, v in sorted(r["parser_dist"].items())
            if k != "unknown"
        ) or "unknown"

        print(
            f"  {r['file']:<36} "
            f"{pages_str:>7} "
            f"{r['avg_chars_per_page']:>8.0f} "
            f"{r['table_pages']:>8} "
            f"{table_str:>7} "
            f"{r['image_only_pages']:>9} "
            f"{r['garbled_pages']:>8} "
            f"{r['avg_layout_coherence']:>10.2f}  "
            f"{parser_str}"
        )

    # Footer: quality distribution nếu có dữ liệu
    all_quality: Counter = Counter()
    for r in results:
        all_quality.update(r.get("quality_dist", {}))
    if all_quality:
        print(f"\n  Text quality distribution (all files): {dict(all_quality)}")


def print_page_detail(result: dict, max_rows: int = 30):
    """In bảng per-page chi tiết cho 1 file."""
    print(f"\n  {'─'*90}")
    print(f"  Per-page detail: {result['file']}")
    print(f"  {'─'*90}")
    print(f"  {'Pg':>4} {'Chars':>6} {'Words':>6} {'SC%':>5} {'Quality':<10} "
          f"{'Parser':<12} {'Tbl':>4} {'TblOK':>6} {'Img':>4} {'Blind':>6} "
          f"{'Coh':>5}")
    print(f"  {'─'*90}")

    for m in result["page_metrics"][:max_rows]:
        sc  = f"{m['single_char_ratio']*100:.0f}%"
        coh = f"{m['coherence_score']:.2f}" if m["coherence_score"] is not None else "  —"
        print(
            f"  {m['page_num']:>4} {m['char_count']:>6} {m['word_count']:>6} "
            f"{sc:>5} {m['text_quality']:<10} {m['parser_used']:<12} "
            f"{'Y' if m['has_table'] else '-':>4} "
            f"{'Y' if m['table_readable'] else ('-' if not m['has_table'] else 'N'):>6} "
            f"{'Y' if m['has_image'] else '-':>4} "
            f"{'Y' if m['is_image_only'] else '-':>6} "
            f"{coh:>5}"
        )

    total = len(result["page_metrics"])
    if total > max_rows:
        print(f"  ... ({total - max_rows} more pages, use --pages-all to show all)")


# ── CLI ────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Axis 0: Parse Quality — đo chất lượng text extraction"
    )
    parser.add_argument("files",      nargs="+",
                        help="File cần đánh giá (PDF/PPTX/DOCX/...)")
    parser.add_argument("--json",     default=None,
                        help="Lưu kết quả ra file JSON")
    parser.add_argument("--pages",    action="store_true",
                        help="In per-page detail cho mỗi file (30 trang đầu)")
    parser.add_argument("--pages-all", action="store_true",
                        help="In toàn bộ per-page detail (không giới hạn)")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  PARSE QUALITY BENCHMARK — {datetime.now():%Y-%m-%d %H:%M}")
    print(f"{'═'*60}")

    all_results = []
    for fp in args.files:
        if not Path(fp).exists():
            print(f"  [SKIP] {fp} — not found")
            continue
        print(f"  Parsing {Path(fp).name}...", end=" ", flush=True)
        try:
            r = run_parse_quality(fp)
            all_results.append(r)
            print(f"OK ({r['extracted_pages']} pages)")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()

    if not all_results:
        print("  No results.")
        return

    print_parse_quality_table(all_results)

    if args.pages or args.pages_all:
        max_rows = 9999 if args.pages_all else 30
        for r in all_results:
            if not r.get("error"):
                print_page_detail(r, max_rows=max_rows)

    if args.json:
        out = {
            "timestamp": datetime.now().isoformat(),
            "results": [
                {k: v for k, v in r.items() if k != "page_metrics"}
                for r in all_results
            ],
            "per_page": {
                r["file"]: r["page_metrics"] for r in all_results
            },
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n  Saved → {args.json}\n")


if __name__ == "__main__":
    main()
