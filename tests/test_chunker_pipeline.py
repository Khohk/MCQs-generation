"""
tests/test_chunker_pipeline.py
------------------------------
In report trực quan: titles từng page → chunks kết quả → đánh dấu chunk đáng ngờ.

Chạy:
    python tests/test_chunker_pipeline.py
    python tests/test_chunker_pipeline.py > data/test_results/chunker_report.txt
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"

from pipeline.file_router import parse_file, get_file_type
from pipeline.chunker import chunk_pages, MIN_CHUNK_CHARS

# ── File matrix ───────────────────────────────────────────────────────

PDF_FILES = [
    "Day_10_Data_Pipeline_and_Data_Observability.pdf",
    "NLP_Week_3.pdf",
    "NLP_Week_5.pdf",
    "W02-VersionControl.pdf",
]
PPTX_FILES = [
    "Ch01-Database Systems copy.pptx",
    "chapter_2_DataModels.pptx",
    "W02-VersionControl.pptx",
    "W03 - Data Engineering in MLOps.pptx",
]
DOCX_FILES = [
    "Linear_regression.docx",
    "Lý_thuyết_Scaling.docx",
    "Multimodal_AI_history.docx",
    "PCA.docx",
]
IMAGE_FILES = [
    "ml.jpg",
]

W = 72  # console width


# ── Helpers ───────────────────────────────────────────────────────────

def sep(char="="):       print(char * W)
def sep2():              print("-" * W)

def flag_title(title: str) -> str:
    """Trả về cảnh báo nếu title trông có vấn đề."""
    t = title.strip()
    if len(t) <= 3:
        return "  ⚠ QUÁ NGẮN"
    if re.fullmatch(r"[\d\s\.\,\:\-\|\/\\]+", t):
        return "  ⚠ CHỈ SỐ/KÝ TỰ"
    if t.lower().startswith(("bảng", "hình", "figure", "table", "page ")):
        return "  ⚠ LÀ CAPTION"
    return ""

def flag_chunk(chunk: dict) -> list[str]:
    """Trả về danh sách cảnh báo cho 1 chunk."""
    warns = []
    if len(chunk["text"]) < MIN_CHUNK_CHARS * 2:
        warns.append(f"⚠ TEXT QUÁ NGẮN ({len(chunk['text'])} chars)")
    if len(chunk["text"]) > 12000:
        warns.append(f"⚠ TEXT QUÁ DÀI ({len(chunk['text'])} chars) — LLM có thể bỏ sót")
    topic = chunk["topic"].strip()
    if len(topic) <= 3 or re.fullmatch(r"[\d\s\.\,\:\-\|\/\\]+", topic):
        warns.append("⚠ TOPIC VÔ NGHĨA")
    return warns


def print_page_titles(pages: list[dict]):
    print(f"  {'PAGE':<6} {'CHARS':<7} TITLE")
    print(f"  {'----':<6} {'-----':<7} -----")
    for p in pages:
        warn = flag_title(p["title"])
        print(f"  {p['page_num']:<6} {p['char_count']:<7} {p['title'][:60]}{warn}")


def print_chunks(chunks: list[dict]):
    topics_seen: dict[str, int] = {}

    # Thống kê chunk_type
    type_counts: dict[str, int] = {}
    for c in chunks:
        ct = c.get("chunk_type", "?")
        type_counts[ct] = type_counts.get(ct, 0) + 1
    print(f"  chunk_type: {type_counts}")
    conceptual = type_counts.get("conceptual", 0)
    print(f"  → dùng được cho MCQ: {conceptual}/{len(chunks)} conceptual\n")

    print(f"  {'CHUNK':<10} {'TYPE':<14} {'PAGES':<14} {'CHARS':<7} TOPIC")
    print(f"  {'-----':<10} {'----':<14} {'-----':<14} {'-----':<7} -----")
    for c in chunks:
        topic     = c["topic"]
        ctype     = c.get("chunk_type", "?")
        pages_str = str(c["pages"]) if len(c["pages"]) <= 4 else f"[{c['pages'][0]}..{c['pages'][-1]}]"
        dup_warn  = ""
        if topic in topics_seen:
            dup_warn = f"  ⚠ TOPIC TRÙNG VỚI chunk_{topics_seen[topic]:03d}"
        else:
            topics_seen[topic] = int(c["chunk_id"].split("_")[1])

        print(f"  {c['chunk_id']:<10} {ctype:<14} {pages_str:<14} {len(c['text']):<7} {topic[:45]}{dup_warn}")

        warns = flag_chunk(c)
        for w in warns:
            print(f"  {'':10} {'':14} {'':14} {'':7} {w}")

        # Preview 2 dòng đầu
        preview_lines = [l.strip() for l in c["text"].splitlines() if l.strip()][:2]
        for line in preview_lines:
            print(f"  {'':10} {'':14} {'':14} {'':7} │ {line[:55]}")
        print()


def run_file(filename: str):
    path = str(DATA / filename)
    ftype = get_file_type(path)

    sep()
    print(f"  FILE : {filename}")
    print(f"  TYPE : {ftype}")
    sep()

    # Parse
    print("\n[1] PAGE TITLES SAU KHI PARSE\n")
    t0 = time.perf_counter()
    pages = parse_file(path)
    parse_time = time.perf_counter() - t0
    print(f"  → {len(pages)} pages  ({parse_time:.2f}s)\n")
    print_page_titles(pages)

    # Chunk
    print(f"\n[2] CHUNKS (strategy=auto)\n")
    t1 = time.perf_counter()
    chunks = chunk_pages(pages, strategy="auto")
    chunk_time = time.perf_counter() - t1
    ratio = len(pages) / len(chunks) if chunks else 0
    print(f"  → {len(chunks)} chunks  ({len(pages)} pages, ratio {ratio:.1f} pages/chunk, {chunk_time:.3f}s)\n")
    print_chunks(chunks)

    # Tổng kết cảnh báo
    bad_chunks = [c for c in chunks if flag_chunk(c)]
    bad_titles = [p for p in pages if flag_title(p["title"])]
    print(f"  Tổng kết: {len(bad_chunks)} chunk đáng ngờ / {len(bad_titles)} title đáng ngờ")
    print()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    groups = [
        ("PDF",   PDF_FILES),
        ("PPTX",  PPTX_FILES),
        ("DOCX",  DOCX_FILES),
        ("IMAGE", IMAGE_FILES),
    ]

    for group_name, files in groups:
        sep("=")
        print(f"  [{group_name}]")
        sep("=")
        print()
        for filename in files:
            run_file(filename)


if __name__ == "__main__":
    main()
