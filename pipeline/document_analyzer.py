"""
pipeline/document_analyzer.py
-----------------------------
Lightweight document diagnostics for parsed pages/sections.

Produces metadata for logs, UI, and thesis experiments, and exposes
select_chunk_strategy() so callers don't need to re-implement the logic.
"""

from __future__ import annotations

from collections import Counter

from pipeline.chunker import WHOLE_DOC_THRESHOLD


LOW_TEXT_THRESHOLD  = 120
SCAN_LOW_TEXT_RATIO = 0.6
SCAN_IMAGE_RATIO    = 0.5


def analyze_document(pages: list[dict]) -> dict:
    """
    Summarize parsed document units including layout flags.

    Args:
        pages: List of page/section dicts from file_router.parse_file().

    Returns:
        Dict with counts, ratios, layout flags, parser usage, and chunk strategy.
    """
    total = len(pages)
    if total == 0:
        return {
            "total_units": 0,
            "avg_chars": 0,
            "low_text_units": 0,
            "image_units": 0,
            "ocr_units": 0,
            "fallback_units": 0,
            "scan_suspected": False,
            "table_units": 0,       "table_ratio": 0.0,
            "formula_units": 0,     "formula_ratio": 0.0,
            "column_units": 0,      "column_ratio": 0.0,
            "layout_sensitive_units": 0, "layout_sensitive_ratio": 0.0,
            "garbled_ratio": 0.0,
            "parser_counts": {},
            "text_quality_counts": {},
            "recommended_strategy": "no_content",
            "chunk_strategy": "auto",
        }

    char_counts    = [int(p.get("char_count", len(p.get("text", "")))) for p in pages]
    low_text_units = sum(1 for n in char_counts if n < LOW_TEXT_THRESHOLD)
    image_units    = sum(1 for p in pages if p.get("has_image", False))
    ocr_units      = sum(1 for p in pages if p.get("ocr_used", False))
    parser_counts  = Counter(str(p.get("parser", "unknown")) for p in pages)
    quality_counts = Counter(str(p.get("text_quality", "unknown")) for p in pages)
    fallback_units = sum(
        1 for p in pages
        if str(p.get("parser", "")) in {"pdfplumber", "pytesseract"}
    )

    # Layout flags — populated by pdf_parser via layout_analyzer
    table_units            = sum(1 for p in pages if p.get("has_table", False))
    formula_units          = sum(1 for p in pages if p.get("has_formula", False))
    column_units           = sum(1 for p in pages if p.get("has_columns", False))
    layout_sensitive_units = sum(1 for p in pages if p.get("layout_sensitive", False))

    low_text_ratio         = low_text_units / total
    image_ratio            = image_units / total
    table_ratio            = table_units / total
    formula_ratio          = formula_units / total
    column_ratio           = column_units / total
    layout_sensitive_ratio = layout_sensitive_units / total
    garbled_ratio          = quality_counts.get("garbled", 0) / total

    scan_suspected = (
        ocr_units > 0
        or (low_text_ratio >= SCAN_LOW_TEXT_RATIO and image_ratio >= SCAN_IMAGE_RATIO)
    )

    total_chars = sum(char_counts)

    if scan_suspected:
        recommended = "ocr_review"
    elif low_text_ratio > 0.3:
        recommended = "low_text_review"
    elif image_ratio > 0.4:
        recommended = "vision_or_image_review"
    elif total_chars <= WHOLE_DOC_THRESHOLD:
        recommended = "whole_doc"
    else:
        recommended = "native_text"

    analysis = {
        "total_units":   total,
        "total_chars":   total_chars,
        "avg_chars":     round(total_chars / total, 1),
        "min_chars":     min(char_counts),
        "max_chars":     max(char_counts),
        "low_text_units": low_text_units,
        "low_text_ratio": round(low_text_ratio, 3),
        "image_units":   image_units,
        "image_ratio":   round(image_ratio, 3),
        "ocr_units":     ocr_units,
        "fallback_units": fallback_units,
        "scan_suspected": scan_suspected,
        "table_units":            table_units,
        "table_ratio":            round(table_ratio, 3),
        "formula_units":          formula_units,
        "formula_ratio":          round(formula_ratio, 3),
        "column_units":           column_units,
        "column_ratio":           round(column_ratio, 3),
        "layout_sensitive_units": layout_sensitive_units,
        "layout_sensitive_ratio": round(layout_sensitive_ratio, 3),
        "garbled_ratio":          round(garbled_ratio, 3),
        "parser_counts":          dict(parser_counts),
        "text_quality_counts":    dict(quality_counts),
        "recommended_strategy":   recommended,
        "chunk_strategy":         select_chunk_strategy({
            "total_chars":            total_chars,
            "scan_suspected":         scan_suspected,
            "low_text_ratio":         low_text_ratio,
            "avg_chars":              total_chars / total,
            "garbled_ratio":          garbled_ratio,
            "table_ratio":            table_ratio,
            "layout_sensitive_ratio": layout_sensitive_ratio,
            "column_ratio":           column_ratio,
        }),
    }
    return analysis


def select_chunk_strategy(doc_analysis: dict) -> str:
    """
    Chọn chunking strategy tối ưu dựa trên các signal từ analyze_document().

    Priority:
      whole_doc  — document đủ nhỏ để fit 1 context window
      overlap    — heading không đáng tin (scan / nhiều ảnh / font lỗi / bảng)
      title      — document có cấu trúc cột rõ (textbook / paper)
      auto       — mặc định, tự điều chỉnh theo title-ratio tại runtime

    Args:
        doc_analysis: dict từ analyze_document() hoặc subset các key cần thiết.
    """
    total_chars            = doc_analysis.get("total_chars", 0)
    scan_suspected         = doc_analysis.get("scan_suspected", False)
    low_text_ratio         = doc_analysis.get("low_text_ratio", 0.0)
    avg_chars              = doc_analysis.get("avg_chars", 0.0)
    garbled_ratio          = doc_analysis.get("garbled_ratio", 0.0)
    table_ratio            = doc_analysis.get("table_ratio", 0.0)
    layout_sensitive_ratio = doc_analysis.get("layout_sensitive_ratio", 0.0)
    column_ratio           = doc_analysis.get("column_ratio", 0.0)

    # 1. Đủ nhỏ → gửi toàn bộ document 1 lần
    if total_chars <= WHOLE_DOC_THRESHOLD:
        return "whole_doc"

    # 2. Heading không đáng tin → dùng similarity thay vì heading
    if (scan_suspected
            or low_text_ratio > 0.5
            or garbled_ratio > 0.3
            or avg_chars < 300):
        return "overlap"

    # 3. Nhiều bảng / layout nhạy cảm → nội dung trải nhiều trang → giữ liền nhau
    if table_ratio > 0.3 or layout_sensitive_ratio > 0.4:
        return "overlap"

    # 4. Document nhiều cột (textbook / paper) → heading thường rõ
    if column_ratio > 0.4:
        return "title"

    # 5. Để chunker tự quyết tại runtime
    return "auto"
