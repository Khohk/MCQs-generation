"""
pipeline/layout_analyzer.py
---------------------------
Lightweight layout-aware extraction helpers.

This module intentionally stays heuristic-based. It enriches parsed pages with
layout metadata without changing the downstream chunker/generator contract.
"""

from __future__ import annotations

import re
from statistics import median


CAPTION_RE = re.compile(
    r"^\s*(figure|fig\.|table|hinh|hinh\s+\d+|bang|bang\s+\d+|bảng|hình)\b",
    re.IGNORECASE,
)
FORMULA_RE = re.compile(
    r"(\$\$.*?\$\$|\$[^$\n]{2,}\$|\\\(.+?\\\)|\\\[.+?\\\]|[A-Za-z]\s*=\s*[^,\n]{2,})",
    re.DOTALL,
)
MARKDOWN_TABLE_RE = re.compile(r"^\s*\|.+\|\s*$\n^\s*\|[\s:\-\|]+\|\s*$", re.MULTILINE)


def analyze_text_layout(text: str, has_image: bool = False) -> dict:
    """
    Detect layout-sensitive structures from plain text or Markdown.

    Returns flags that can be merged directly into a page/section dict.
    """
    text = text or ""
    lines = [line.rstrip() for line in text.splitlines()]
    non_empty = [line for line in lines if line.strip()]

    has_table = _has_markdown_table(text) or _looks_like_table(non_empty)
    has_formula = bool(FORMULA_RE.search(text))
    has_caption = any(CAPTION_RE.search(_strip_markdown(line)) for line in non_empty)
    has_columns = _looks_like_columns(non_empty)

    layout_sensitive = any((
        has_table,
        has_formula,
        has_caption and has_image,
        has_columns,
    ))

    return {
        "has_table": has_table,
        "has_formula": has_formula,
        "has_caption": has_caption,
        "has_columns": has_columns,
        "layout_sensitive": layout_sensitive,
        "vision_used": False,
        "layout_flags": {
            "has_table": has_table,
            "has_formula": has_formula,
            "has_caption": has_caption,
            "has_columns": has_columns,
            "layout_sensitive": layout_sensitive,
        },
    }


def analyze_pdf_page_layout(pdf_path: str, page_index: int, max_blocks: int = 80) -> dict:
    """
    Extract simple layout blocks from a PDF page via PyMuPDF.

    Output is intentionally compact: text, bbox, font metadata, block_type,
    column_index and reading_order. It is primarily for diagnostics and for
    deciding whether the page is layout-sensitive.
    """
    try:
        import fitz
    except ImportError:
        return {"layout_blocks": [], "layout_flags": {}, "warnings": ["fitz_unavailable"]}

    with fitz.open(pdf_path) as doc:
        if page_index >= len(doc):
            return {"layout_blocks": [], "layout_flags": {}, "warnings": ["page_index_out_of_range"]}
        page = doc[page_index]
        page_rect = page.rect
        raw = page.get_text("dict")

    spans = []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        bbox = block.get("bbox", [0, 0, 0, 0])
        block_text_parts = []
        font_sizes = []
        font_names = []
        is_bold = False
        for line in block.get("lines", []):
            line_parts = []
            for span in line.get("spans", []):
                span_text = span.get("text", "")
                if span_text.strip():
                    line_parts.append(span_text)
                    font_sizes.append(float(span.get("size", 0) or 0))
                    font = str(span.get("font", ""))
                    font_names.append(font)
                    if "bold" in font.lower():
                        is_bold = True
            if line_parts:
                block_text_parts.append(" ".join(line_parts))
        text = "\n".join(block_text_parts).strip()
        if not text:
            continue
        spans.append({
            "text": text,
            "bbox": [round(float(x), 2) for x in bbox],
            "font_size": round(max(font_sizes) if font_sizes else 0, 2),
            "font_name": font_names[0] if font_names else "",
            "is_bold": is_bold,
        })

    if not spans:
        return {"layout_blocks": [], "layout_flags": {}, "warnings": ["no_text_blocks"]}

    body_sizes = [b["font_size"] for b in spans if b["font_size"] > 0]
    base_size = median(body_sizes) if body_sizes else 0
    columns = _assign_columns(spans, page_rect.width)
    has_columns = len(set(columns)) > 1

    layout_blocks = []
    caption_count = 0
    table_like_count = 0
    formula_count = 0
    header_footer_count = 0

    for idx, block in enumerate(spans):
        block_type = _classify_pdf_block(block, base_size, page_rect.height)
        if block_type == "caption":
            caption_count += 1
        elif block_type == "table":
            table_like_count += 1
        elif block_type == "formula":
            formula_count += 1
        elif block_type in {"header", "footer"}:
            header_footer_count += 1

        layout_blocks.append({
            "text": block["text"],
            "block_type": block_type,
            "bbox": block["bbox"],
            "font_size": block["font_size"],
            "font_name": block["font_name"],
            "is_bold": block["is_bold"],
            "column_index": columns[idx],
        })

    layout_blocks.sort(key=lambda b: (b["column_index"], b["bbox"][1], b["bbox"][0]))
    for order, block in enumerate(layout_blocks):
        block["reading_order"] = order

    flags = {
        "has_columns": has_columns,
        "has_table": table_like_count > 0,
        "has_formula": formula_count > 0,
        "has_caption": caption_count > 0,
        "has_header_footer": header_footer_count > 0,
    }
    flags["layout_sensitive"] = any((
        flags["has_columns"],
        flags["has_table"],
        flags["has_formula"],
        flags["has_caption"],
    ))

    return {
        "layout_blocks": layout_blocks[:max_blocks],
        "layout_flags": flags,
        "warnings": ["layout_blocks_truncated"] if len(layout_blocks) > max_blocks else [],
    }


def merge_layout_flags(*flag_dicts: dict) -> dict:
    """Merge bool layout flags from text and PDF layout analyzers."""
    merged = {
        "has_table": False,
        "has_formula": False,
        "has_caption": False,
        "has_columns": False,
        "layout_sensitive": False,
        "vision_used": False,
    }
    nested = {}
    for flags in flag_dicts:
        if not flags:
            continue
        source = flags.get("layout_flags", flags)
        for key in list(merged):
            merged[key] = bool(merged[key] or source.get(key, flags.get(key, False)))
        for key, value in source.items():
            if isinstance(value, bool):
                nested[key] = bool(nested.get(key, False) or value)
    merged["layout_flags"] = nested or {k: v for k, v in merged.items() if k != "layout_flags"}
    return merged


def _has_markdown_table(text: str) -> bool:
    return bool(MARKDOWN_TABLE_RE.search(text or ""))


def _looks_like_table(lines: list[str]) -> bool:
    candidates = 0
    for line in lines:
        stripped = line.strip()
        if "|" in stripped and stripped.count("|") >= 2:
            candidates += 1
        elif re.search(r"\S+\s{2,}\S+\s{2,}\S+", stripped):
            tokens = re.split(r"\s{2,}", stripped)
            if len(tokens) >= 3 and sum(1 for t in tokens if len(t) <= 18) >= 2:
                candidates += 1
    return candidates >= 2


def _looks_like_columns(lines: list[str]) -> bool:
    columnish = 0
    for line in lines:
        if len(line) < 40:
            continue
        if re.search(r"[a-zA-Z0-9\)]\s{6,}[A-Z0-9]", line):
            columnish += 1
    return columnish >= 3


def _classify_pdf_block(block: dict, base_size: float, page_height: float) -> str:
    text = block["text"].strip()
    clean = _strip_markdown(text)
    y0, y1 = block["bbox"][1], block["bbox"][3]
    font_size = block["font_size"]

    if y1 < page_height * 0.08:
        return "header"
    if y0 > page_height * 0.92:
        return "footer"
    if CAPTION_RE.search(clean):
        return "caption"
    if FORMULA_RE.search(clean):
        return "formula"
    if _looks_like_table(clean.splitlines()):
        return "table"
    if re.match(r"^(\d+[\.\)]|[-*•])\s+", clean):
        return "list"
    if base_size and font_size >= base_size * 1.45:
        return "title"
    if block.get("is_bold") or (base_size and font_size >= base_size * 1.18):
        return "heading"
    return "text"


def _assign_columns(blocks: list[dict], page_width: float) -> list[int]:
    centers = [((b["bbox"][0] + b["bbox"][2]) / 2) for b in blocks]
    if not centers:
        return []
    left = [c for c in centers if c < page_width * 0.45]
    right = [c for c in centers if c > page_width * 0.55]
    if len(left) >= 2 and len(right) >= 2:
        return [0 if c < page_width * 0.5 else 1 for c in centers]
    return [0 for _ in centers]


def _strip_markdown(text: str) -> str:
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"[*_`#~]", "", text)
    return text.strip()
