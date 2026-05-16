"""
pipeline/pass2_extractor.py
----------------------------
Pass 2: per-segment KU extraction using Pass 1 output.

Usage:
    from pipeline.pass2_extractor import run_pass2
    result2 = run_pass2(pass1_result, llm_fn)

    result2.all_kus : list[dict]              — all KUs (deduped)
    result2.graph   : dict[str, list[str]]    — KU adjacency graph
    result2.pool    : dict[str, list[dict]]   — distractor pool by type
"""

from __future__ import annotations

import time
import sys
import re
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.pass1_extractor import Pass1Result, Pass1Segment
from pipeline.knowledge_extractor import (
    extract_kus, dedup_kus,
    build_ku_graph_with_cross, build_distractor_pool, get_distractors,
)

MAX_PASS2_CHARS = 5_000

_RECAP_KW = {"conclusion", "summary", "recap", "review", "tong ket", "ket luan", "q&a"}

_SKIP_KW  = {
    "hands-on", "hands on", "lab ", " lab", "exercise", "thuc hanh", "thực hành",
    "practice", "card sort", "group activity", "bai tap", "bài tập",
    "references", "bibliography", "appendix", "citation",
    "thi nghiem", "thí nghiệm",
    "thông tin môn", "course info", "syllabus",
    "ký hiệu toán", "mathematical notation", "notation",
    "tài liệu tham khảo",
}


def _is_recap(label: str) -> bool:
    low = label.lower()
    return any(kw in low for kw in _RECAP_KW)


def _is_skip(label: str) -> bool:
    """Return True for segments that should be entirely skipped (lab, references, etc.)."""
    low = label.lower()
    for kw in _SKIP_KW:
        kw = kw.strip().lower()
        if not kw:
            continue
        # Avoid false positives such as "lab" matching "labeling".
        if re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", low):
            return True
    return False


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class Pass2Result:
    all_kus   : list[dict]                    = field(default_factory=list)
    graph     : dict[str, list[str]]          = field(default_factory=dict)
    pool      : dict[str, list[dict]]         = field(default_factory=dict)
    edge_types: dict[tuple[str, str], str]    = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.all_kus)

    def get_distractors(self, anchor_ku_id: str, n: int = 3) -> list[dict]:
        return get_distractors(anchor_ku_id, self.graph, self.edge_types, self.all_kus, n)


# ── Cross-rel filter ───────────────────────────────────────────────────────

def _relevant_cross_rels(seg_concepts: list[str],
                         relationships: list[dict]) -> list[dict]:
    """Return relationships that mention any concept from this segment."""
    if not seg_concepts or not relationships:
        return []
    concept_set = {c.lower() for c in seg_concepts}
    return [
        r for r in relationships
        if r.get("from_concept", "").lower() in concept_set
        or r.get("to_concept",   "").lower() in concept_set
    ]


# ── Segment splitter ───────────────────────────────────────────────────────

def _split_segment(seg: Pass1Segment, max_chars: int) -> list[Pass1Segment]:
    """
    Split a large segment at page boundaries into sub-segments.
    Sub-segments get ids like seg_004a / seg_004b and inherit all metadata.
    Pages are computed proportionally from block indices so each sub-seg
    gets its own slice of source_pages (not the full parent range).
    """
    if len(seg.text) <= max_chars:
        return [seg]

    blocks = seg.text.split("\n\n---\n\n")
    if len(blocks) <= 1:
        return [seg]

    # Track block index ranges per group for proportional page slicing
    groups: list[tuple[list[str], int, int]] = []  # (block_texts, first_idx, last_idx)
    current: list[str] = []
    current_start: int = 0
    current_len: int = 0

    for bidx, block in enumerate(blocks):
        blen = len(block) + 9  # 9 = len("\n\n---\n\n")
        if current and current_len + blen > max_chars:
            groups.append((current, current_start, bidx - 1))
            current = [block]
            current_start = bidx
            current_len = blen
        else:
            current.append(block)
            current_len += blen
    if current:
        groups.append((current, current_start, len(blocks) - 1))

    if len(groups) <= 1:
        return [seg]

    n_blocks = len(blocks)
    n_pages  = len(seg.source_pages)
    result: list[Pass1Segment] = []
    for i, (grp_blocks, start_idx, end_idx) in enumerate(groups):
        text = "\n\n---\n\n".join(grp_blocks)
        # Proportional page slice: block position → page position
        if n_blocks > 1 and n_pages > 0:
            p_start = int(start_idx * n_pages / n_blocks)
            p_end   = min(int(end_idx * n_pages / n_blocks) + 1, n_pages)
            pages   = seg.source_pages[p_start:p_end] or seg.source_pages
        else:
            pages = seg.source_pages
        result.append(Pass1Segment(
            segment_id          = f"{seg.segment_id}{chr(97 + i)}",
            label               = seg.label,
            concepts            = list(seg.concepts),
            source_pages        = pages,
            text                = text,
            parent_concept_id   = seg.parent_concept_id,
            concept_id          = seg.concept_id,
            parent_concept_name = seg.parent_concept_name,
            parent_segment_id   = seg.segment_id,
        ))

    _log(f"[Pass2] split '{seg.label[:40]}' ({len(seg.text):,} chars) → "
         f"{len(result)} sub-segs pages={[s.source_pages for s in result]}")
    return result


# ── Main entry point ───────────────────────────────────────────────────────

def run_pass2(
    pass1: Pass1Result,
    llm_fn,
    delay_between: float = 3.0,
) -> Pass2Result:
    """
    Extract KUs from every segment in pass1.segments.

    Args:
        pass1         : Pass1Result from run_pass1()
        llm_fn        : callable(prompt: str, json_mode: bool) -> str
        delay_between : seconds between API calls (avoid rate limit)

    Returns:
        Pass2Result — all_kus deduped, graph includes cross-segment edges.
    """
    if not pass1.ok:
        _log("[Pass2] no segments in Pass1Result — skip")
        return Pass2Result()

    all_doc_concepts: list[str] = [sc["name"] for sc in pass1.sub_concepts]
    for l1 in getattr(pass1, "raw_hierarchy", []) or []:
        all_doc_concepts.append(l1.get("name", ""))
        all_doc_concepts.extend(c.get("name", "") for c in l1.get("children", []) or [])
    all_doc_concepts = [c for c in dict.fromkeys(all_doc_concepts) if c]
    seen_concepts:    list[str] = []
    all_kus:          list[dict] = []

    # Split large segments before extraction
    expanded_segs: list[Pass1Segment] = []
    for seg in pass1.segments:
        expanded_segs.extend(_split_segment(seg, MAX_PASS2_CHARS))

    for i, seg in enumerate(expanded_segs, 1):
        _log(f"\n[Pass2] [{i}/{len(expanded_segs)}] {seg.segment_id} "
             f"— {seg.label[:50]}")
        _log(f"  pages={seg.source_pages}  chars={len(seg.text)}")

        if _is_skip(seg.label):
            _log(f"  [skip] non-content segment (lab/references/activity) — skip")
            continue

        l1_parent = seg.parent_concept_name
        l1_seen   = bool(l1_parent and l1_parent in seen_concepts)

        chunk        = seg.to_chunk_dict()
        chunk["l1_already_seen"] = l1_seen
        cross_rels   = _relevant_cross_rels(seg.concepts, pass1.relationships)
        seg_concepts = seg.concepts
        # Include L1 parent in valid concept names when L1 content hasn't been covered yet;
        # this lets the LLM assign concept = "{parent_name}" to L1 KUs it finds in the text.
        concepts_for_doc = seg_concepts + [l1_parent] if l1_parent else seg_concepts
        chunk["owner_concepts"] = concepts_for_doc
        doc_concepts = {"seg": concepts_for_doc, "all": all_doc_concepts}

        if cross_rels:
            _log(f"  cross_rels: {[r['relation']+' '+r['from_concept'][:20] for r in cross_rels]}")

        is_recap = _is_recap(seg.label)
        if is_recap:
            _log(f"  [recap] passing seen_concepts ({len(seen_concepts)}) as filter")
        elif l1_parent:
            _log(f"  [l1] parent='{l1_parent}'  l1_seen={l1_seen}")

        kus = extract_kus(
            chunk,
            llm_fn,
            cross_rels=cross_rels,
            doc_concepts=doc_concepts,
            seen_concepts=seen_concepts if is_recap else None,
        )
        _log(f"  -> {len(kus)} KUs")
        all_kus.extend(kus)

        seen_concepts = list(dict.fromkeys(seen_concepts + seg_concepts))
        # Mark L1 parent as seen so subsequent L2 siblings skip re-extracting L1 KUs
        if l1_parent and l1_parent not in seen_concepts:
            seen_concepts.append(l1_parent)

        if i < len(expanded_segs):
            time.sleep(delay_between)

    # Dedup across segments, drop peripheral, build graph + pool
    all_kus              = dedup_kus(all_kus)
    n_before = len(all_kus)
    all_kus  = [ku for ku in all_kus if ku.get("prominence") != "peripheral"]
    _log(f"[Pass2] peripheral filter: {n_before} → {len(all_kus)} KUs")
    graph, edge_types    = build_ku_graph_with_cross(
                               all_kus, pass1.relationships, pass1.concept_hierarchy,
                               expanded_segs)
    pool                 = build_distractor_pool(all_kus)

    n_edges   = sum(len(v) for v in graph.values()) // 2
    n_sibling = sum(1 for r in edge_types.values() if r == "SIBLING_OF")
    n_cross   = sum(1 for r in edge_types.values() if r not in {"SIBLING_OF"})
    _log(f"\n[Pass2] done: {len(all_kus)} KUs (deduped), {n_edges} graph edges "
         f"({n_sibling} sibling, {n_cross} other)")

    return Pass2Result(all_kus=all_kus, graph=graph, pool=pool, edge_types=edge_types)


# ── Utility ────────────────────────────────────────────────────────────────

def _log(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))
