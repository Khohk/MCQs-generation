"""
pipeline/pass1_extractor.py
----------------------------
Pass 1: whole-document semantic segmentation + cross-segment relationship discovery.
Replaces the chunker — takes raw pages from parse_file(), returns Pass1Result.

Usage:
    from pipeline.pass1_extractor import run_pass1
    result = run_pass1(pages, llm_fn)

    # result.main_concept      : str
    # result.sub_concepts      : list[dict]  — derived from L1 concept_hierarchy
    # result.relationships     : list[dict]  — cross-segment only
    # result.segments          : list[Pass1Segment]
    # result.concept_hierarchy : list[dict]  — {parent, children} for SIBLING_OF

LLM output schema (new):
    main_concept      : str
    concept_hierarchy : [{concept_id, name, level, evidence, children: [L2...]}]
    segments          : [{segment_id, concept_id, label, pages}]
    relationships     : [{from_concept, to_concept, from_segment, to_segment,
                          relation, evidence, evidence_type}]

Each Pass1Segment:
    segment_id   : "seg_001", ...
    label        : L1 concept name
    concepts     : [L1 name] + [L2 child names]
    source_pages : actual PDF page numbers (code maps from LLM's sequential indices)
    text         : reconstructed text (ready for Pass 2)
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class Pass1Segment:
    segment_id          : str
    label               : str
    concepts            : list[str]
    source_pages        : list[int]
    text                : str  = ""
    parent_concept_id   : str  = ""  # L1 concept_id if this is an L2 segment; empty for L1
    concept_id          : str  = ""  # own concept_id from hierarchy
    parent_concept_name : str  = ""  # L1 name if this is an L2 segment; empty for L1
    parent_segment_id   : str  = ""  # set when this segment was split from a larger one

    def to_chunk_dict(self) -> dict:
        """Convert to chunk-dict format compatible with extract_kus()."""
        return {
            "chunk_id"           : self.segment_id,
            "chunk_type"         : "conceptual",
            "topic"              : self.label,
            "pages"              : self.source_pages,
            "text"               : self.text,
            "has_image"          : False,
            "concept_id"         : self.concept_id,
            "parent_concept_name": self.parent_concept_name,
        }


@dataclass
class Pass1Result:
    main_concept      : str
    sub_concepts      : list[dict]         = field(default_factory=list)
    relationships     : list[dict]         = field(default_factory=list)
    segments          : list[Pass1Segment] = field(default_factory=list)
    concept_hierarchy : list[dict]         = field(default_factory=list)
    raw_hierarchy     : list[dict]         = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.segments)


# ── Whole-doc text builder ─────────────────────────────────────────────────

def build_whole_doc_text(pages: list[dict]) -> str:
    """
    Format all pages as a single string with page markers.
    Format: [Page Title]\npage text\n\n---\n\n[Next Title]\n...
    """
    content_pages = [p for p in pages if p.get("text", "").strip()]
    parts = []
    for p in content_pages:
        header = f"[{p['title']}]" if p.get("title") else f"[Page {p['page_num']}]"
        parts.append(f"{header}\n{p['text'].strip()}")
    return "\n\n---\n\n".join(parts)


def _build_doc_text_seq(sorted_pages: list[dict]) -> str:
    """
    Like build_whole_doc_text but uses sequential 1-based indices in headers.
    This prevents LLM from confusing internal slide numbering (e.g. "7/25" footer)
    with actual PDF page numbers. LLM sees Page 1..N; code maps back to PDF pages.
    """
    parts = []
    for idx, p in enumerate(sorted_pages, 1):
        parts.append(f"[Page {idx}]\n{p['text'].strip()}")
    return "\n\n---\n\n".join(parts)


# ── Segment builder ────────────────────────────────────────────────────────


def _build_segments_from_breaks(
    raw_segments: list[dict],
    pages: list[dict],
    absorb_trailing: bool = True,
) -> list[Pass1Segment]:
    """
    Build Pass1Segments from LLM output (each entry has last_page).
    Code owns all page math — LLM only specifies the cut point.

    Algorithm:
      - Sort pages by page_num
      - For each segment i: pages from prev_last+1 to last_page (inclusive)
      - If absorb_trailing=True (default): final segment takes all remaining pages.
        If False: final segment uses its last_page exactly (drops trailing pages).
    """
    if not raw_segments or not pages:
        return []

    all_page_nums = sorted(p["page_num"] for p in pages)
    page_map      = {p["page_num"]: p for p in pages}

    segments: list[Pass1Segment] = []
    used_up_to = -1  # index into all_page_nums

    for i, seg in enumerate(raw_segments):
        last_page = seg.get("last_page")
        is_last   = (i == len(raw_segments) - 1)

        if last_page is None:
            _log(f"[Pass1] segment {i+1} missing last_page — skip")
            continue

        # Find index of last_page in all_page_nums
        try:
            end_idx = all_page_nums.index(last_page)
        except ValueError:
            candidates = [j for j, p in enumerate(all_page_nums)
                          if p <= last_page and j > used_up_to]
            end_idx = candidates[-1] if candidates else used_up_to

        if is_last and absorb_trailing:
            # absorb_trailing: final segment takes all remaining pages
            seg_page_nums = all_page_nums[used_up_to + 1:]
        else:
            seg_page_nums = all_page_nums[used_up_to + 1: end_idx + 1]

        used_up_to = max(used_up_to, end_idx)

        if not seg_page_nums:
            _log(f"[Pass1] segment {i+1} ({seg.get('label','')!r}) got no pages — skip")
            continue

        text = _reconstruct_text(seg_page_nums, page_map)
        if not text.strip():
            _log(f"[Pass1] segment {i+1} empty text — skip")
            continue

        segments.append(Pass1Segment(
            segment_id          = f"seg_{len(segments)+1:03d}",
            label               = seg.get("label", ""),
            concepts            = seg.get("concepts", []),
            source_pages        = seg_page_nums,
            text                = text,
            parent_concept_id   = seg.get("parent_concept_id", ""),
            concept_id          = seg.get("concept_id", ""),
            parent_concept_name = seg.get("parent_concept_name", ""),
        ))

    return segments


def _derive_segments_from_hierarchy(
    concept_hierarchy: list[dict],
    pages: list[dict],
    seq_to_actual: dict,
) -> list[Pass1Segment]:
    """
    Flatten L1 (no children) and L2 concepts, map their last_page (seq index)
    to actual PDF page numbers, then feed into _build_segments_from_breaks.
    absorb_trailing=False drops trailing lab/practice pages automatically.
    Code derives first_page as prev_last+1 — LLM only specifies last_page.
    """
    concepts_by_id: dict[str, list[str]] = {}
    l2_to_l1: dict[str, str] = {}
    raw_segments: list[dict] = []

    for l1 in concept_hierarchy:
        l1_cid      = l1.get("concept_id", "")
        l1_children = l1.get("children", [])
        children_names = [c["name"] for c in l1_children]
        # L1 concepts_by_id includes children names for concept_to_seg coverage
        concepts_by_id[l1_cid] = [l1["name"]] + children_names

        if l1_children:
            # Consecutive L2s with the same last_page do not have separable page
            # ranges. Treat them as one shared segment while preserving each
            # child name in `concepts` so Pass 2 can still assign KUs to the
            # right L2 owner.
            grouped_children: list[tuple[int, list[dict]]] = []
            current_last = None
            current_group: list[dict] = []
            for child in l1_children:
                child_cid = child.get("concept_id", "")
                concepts_by_id[child_cid] = [child["name"]]
                l2_to_l1[child_cid] = l1_cid

                child_last = child.get("last_page")
                if child_last is None:
                    continue
                actual_last = seq_to_actual.get(child_last, child_last)
                if current_group and actual_last == current_last:
                    current_group.append(child)
                    continue
                if current_group:
                    grouped_children.append((current_last, current_group))
                current_last = actual_last
                current_group = [child]
            if current_group:
                grouped_children.append((current_last, current_group))

            for actual_last, group in grouped_children:
                names = [child["name"] for child in group]
                label = " + ".join(names)
                concept_id = group[0].get("concept_id", "")
                if len(group) > 1:
                    _log(
                        f"[Pass1] shared L2 page range under '{l1['name']}': "
                        f"{names} -> one segment"
                    )
                raw_segments.append({
                    "label"              : label,
                    "concepts"           : names,
                    "last_page"          : actual_last,
                    "parent_concept_id"  : l1_cid,
                    "concept_id"         : concept_id,
                    "parent_concept_name": l1["name"],
                })
        else:
            l1_last = l1.get("last_page")
            if l1_last is not None:
                raw_segments.append({
                    "label"              : l1["name"],
                    "concepts"           : concepts_by_id[l1_cid],
                    "last_page"          : seq_to_actual.get(l1_last, l1_last),
                    "parent_concept_id"  : "",
                    "concept_id"         : l1_cid,
                    "parent_concept_name": "",
                })

    if not raw_segments:
        return []

    raw_segments.sort(key=lambda s: s["last_page"])
    _log("[Pass1] L2 last_page dump (sorted):")
    for s in raw_segments:
        _log(f"  last_page={s['last_page']:3d}  label={s['label']!r}")
    # absorb_trailing=False: trailing lab/Q&A pages at end of file are dropped
    return _build_segments_from_breaks(raw_segments, pages, absorb_trailing=False)


def _reconstruct_text(page_nums: list[int], page_map: dict) -> str:
    """Build segment text by joining page texts in order."""
    parts = []
    for pn in page_nums:
        page = page_map.get(pn)
        if page and page.get("text", "").strip():
            header = f"[{page['title']}]" if page.get("title") else f"[Page {pn}]"
            parts.append(f"{header}\n{page['text'].strip()}")
    return "\n\n---\n\n".join(parts)



# ── Small-segment merge ───────────────────────────────────────────────────

MIN_SEGMENT_CHARS = 500


def _merge_two_segs(a: Pass1Segment, b: Pass1Segment) -> Pass1Segment:
    return Pass1Segment(
        segment_id        = a.segment_id,
        label             = f"{a.label} + {b.label}",
        concepts          = a.concepts + [c for c in b.concepts if c not in a.concepts],
        source_pages      = sorted(set(a.source_pages + b.source_pages)),
        text              = "\n\n---\n\n".join(t for t in [a.text, b.text] if t.strip()),
        parent_concept_id = a.parent_concept_id,
        concept_id        = a.concept_id,
        parent_concept_name = a.parent_concept_name,
        parent_segment_id = a.parent_segment_id,
    )


def _merge_adjacent_small(siblings: list[Pass1Segment], min_chars: int) -> list[Pass1Segment]:
    """Greedily merge each too-small segment into its right neighbor."""
    segs = list(siblings)
    changed = True
    while changed:
        changed = False
        merged: list[Pass1Segment] = []
        i = 0
        while i < len(segs):
            if len(segs[i].text) < min_chars and i + 1 < len(segs):
                merged.append(_merge_two_segs(segs[i], segs[i + 1]))
                i += 2
                changed = True
            else:
                merged.append(segs[i])
                i += 1
        segs = merged
    return segs


def _merge_small_segments(
    segments: list[Pass1Segment],
    concept_hierarchy: list[dict],
    min_chars: int = MIN_SEGMENT_CHARS,
) -> list[Pass1Segment]:
    """
    Post-process segments: merge L2 siblings that are too small.
    If ALL siblings of an L1 are small → L1 fallback (merge all into one L1 segment).
    L1 segments (no parent) pass through unchanged.
    """
    l1_info: dict[str, dict] = {
        l1["concept_id"]: {
            "name"    : l1["name"],
            "children": [c["name"] for c in l1.get("children", [])],
        }
        for l1 in concept_hierarchy
        if l1.get("concept_id")
    }

    l2_groups: dict[str, list[Pass1Segment]] = {}
    result: list[Pass1Segment] = []

    for seg in segments:
        if seg.parent_concept_id:
            l2_groups.setdefault(seg.parent_concept_id, []).append(seg)
        else:
            result.append(seg)

    for parent_cid, siblings in l2_groups.items():
        siblings.sort(key=lambda s: s.source_pages[0])

        if all(len(s.text) < min_chars for s in siblings):
            # L1 fallback: all siblings too small → collapse into one L1 segment
            l1 = l1_info.get(parent_cid, {})
            merged = Pass1Segment(
                segment_id        = siblings[0].segment_id,
                label             = l1.get("name", parent_cid),
                concepts          = [l1.get("name", parent_cid)] + l1.get("children", []),
                source_pages      = sorted(set(p for s in siblings for p in s.source_pages)),
                text              = "\n\n---\n\n".join(s.text for s in siblings if s.text.strip()),
                parent_concept_id = "",
                concept_id        = parent_cid,
                parent_concept_name = "",
            )
            result.append(merged)
            _log(f"[Pass1] L1 fallback: {len(siblings)} L2s → '{merged.label}'")
        else:
            result.extend(_merge_adjacent_small(siblings, min_chars))

    result.sort(key=lambda s: s.source_pages[0])
    for i, seg in enumerate(result):
        seg.segment_id = f"seg_{i + 1:03d}"
    return result


# ── Relationship remapping ─────────────────────────────────────────────────

def _norm_c(s: str) -> str:
    """Lowercase + collapse whitespace + strip hyphens."""
    return re.sub(r"\s+", " ", s.lower().strip().replace("-", " "))


def _fuzzy_match_concept(name_norm: str, concept_to_seg: dict[str, str]) -> str | None:
    """
    Token-subset match on a normalized-key dict.
    Handles: long descriptive names ("The LSTM: Long short-term memory network" → "lstm"),
             plurals ("rnns" → "rnn"), acronym expansions.
    """
    def _depl(t: str) -> str:
        return t.rstrip("s") if len(t) > 3 else t

    query_tokens = {_depl(t) for t in re.findall(r"\w+", name_norm) if t}
    if not query_tokens:
        return None

    for key, sid in concept_to_seg.items():
        key_tokens = {_depl(t) for t in re.findall(r"\w+", key) if t}
        if not key_tokens:
            continue
        if key_tokens <= query_tokens or query_tokens <= key_tokens:
            return sid
    return None


def _remap_relationships(
    relationships: list[dict],
    segments: list[Pass1Segment],
    concept_hierarchy: list[dict],
) -> list[dict]:
    """
    Map from_concept/to_concept → segment_ids.
    Uses normalized keys + token-subset fuzzy fallback to handle LLM concept name drift
    (e.g. "The LSTM: Long short-term memory network" → "lstm" segment).
    Covers all L1 and L2 names from concept_hierarchy:
      - Concepts with a dedicated segment: map directly.
      - Orphaned L2 (no segment): map to nearest sibling with a segment.
      - L1 with children but no direct segment: map to last child segment.
    Drops: intra-segment pairs, duplicates.
    """
    # Build lookup with normalized keys
    seg_by_id: dict[str, Pass1Segment] = {seg.segment_id: seg for seg in segments}
    concept_to_seg: dict[str, str] = {}
    for seg in segments:
        for name in seg.concepts:
            concept_to_seg[_norm_c(name)] = seg.segment_id

    # Extend map to cover orphaned L1/L2 names
    for l1 in concept_hierarchy:
        l1_norm  = _norm_c(l1["name"])
        children = l1.get("children", [])
        if not children:
            continue

        # L1 name → last known child segment
        if l1_norm not in concept_to_seg:
            for child in reversed(children):
                sid = concept_to_seg.get(_norm_c(child["name"]))
                if sid:
                    concept_to_seg[l1_norm] = sid
                    break

        # Orphaned L2 → nearest sibling (right first, then left)
        for i, child in enumerate(children):
            cnorm = _norm_c(child["name"])
            if cnorm in concept_to_seg:
                continue
            found = None
            for j in range(i + 1, len(children)):
                found = concept_to_seg.get(_norm_c(children[j]["name"]))
                if found:
                    break
            if not found:
                for j in range(i - 1, -1, -1):
                    found = concept_to_seg.get(_norm_c(children[j]["name"]))
                    if found:
                        break
            if found:
                concept_to_seg[cnorm] = found
                _log(f"[Pass1] '{child['name']}' orphaned → mapped to sibling segment '{found}'")
                sibling_seg = seg_by_id.get(found)
                if sibling_seg and child["name"] not in sibling_seg.concepts:
                    sibling_seg.concepts.append(child["name"])

    valid: list[dict] = []
    seen: set[tuple] = set()

    for r in relationships:
        fc = r.get("from_concept", "")
        tc = r.get("to_concept",   "")

        fc_norm  = _norm_c(fc)
        tc_norm  = _norm_c(tc)
        from_seg = concept_to_seg.get(fc_norm)
        to_seg   = concept_to_seg.get(tc_norm)

        if not from_seg:
            from_seg = _fuzzy_match_concept(fc_norm, concept_to_seg)
            if from_seg:
                _log(f"[Pass1] fuzzy '{fc}' → segment '{from_seg}'")
        if not to_seg:
            to_seg = _fuzzy_match_concept(tc_norm, concept_to_seg)
            if to_seg:
                _log(f"[Pass1] fuzzy '{tc}' → segment '{to_seg}'")

        if not from_seg:
            _log(f"[Pass1] rel drop: '{fc}' has no segment")
            continue
        if not to_seg:
            _log(f"[Pass1] rel drop: '{tc}' has no segment")
            continue
        if from_seg == to_seg:
            _log(f"[Pass1] rel intra-seg (→ Pass 2): {fc} → {tc}")
            continue

        key = (from_seg, to_seg, r.get("relation", ""))
        if key in seen:
            continue
        seen.add(key)

        rel = {k: v for k, v in r.items() if k not in ("from_segment", "to_segment")}
        rel["from_segment"] = from_seg
        rel["to_segment"]   = to_seg
        valid.append(rel)

    return valid


# ── JSON parsing ───────────────────────────────────────────────────────────

def _parse_response(raw: str) -> dict:
    text = raw.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    return json.loads(text)


# ── Main entry point ───────────────────────────────────────────────────────

def run_pass1(pages: list[dict], llm_fn) -> Pass1Result:
    """
    Run Pass 1 on the whole document.

    Args:
        pages  : list[dict] from parse_file() — {page_num, title, text, ...}
        llm_fn : callable(prompt: str, json_mode: bool) -> str

    Returns:
        Pass1Result with segments populated and text ready for Pass 2.
        Returns Pass1Result(main_concept="Unknown") on failure.
    """
    from prompts.pass1_prompt import build_pass1_prompt

    if not pages:
        _log("[Pass1] no pages — skip")
        return Pass1Result(main_concept="Unknown")

    # Use sequential indices so LLM isn't confused by internal slide numbering
    # (e.g. a slide footer "7/25" on PDF page 11 would cause last_page mismatch)
    content_pages    = sorted([p for p in pages if p.get("text", "").strip()],
                               key=lambda p: p["page_num"])
    actual_page_nums = [p["page_num"] for p in content_pages]
    seq_to_actual    = {i + 1: pn for i, pn in enumerate(actual_page_nums)}

    whole_doc_text = _build_doc_text_seq(content_pages)
    doc_title      = content_pages[0].get("title", "Document") if content_pages else "Document"
    seq_page_list  = list(range(1, len(content_pages) + 1))
    prompt         = build_pass1_prompt(whole_doc_text, doc_title, seq_page_list)

    raw = ""
    try:
        raw  = llm_fn(prompt, False)
        if not raw or not raw.strip():
            _log("[Pass1] empty response from model")
            return Pass1Result(main_concept="Unknown")

        data = _parse_response(raw)

    except json.JSONDecodeError as e:
        _log(f"[Pass1] JSON parse error: {e}")
        _log(f"[Pass1] raw preview: {repr(raw[:300])}")
        return Pass1Result(main_concept="Unknown")
    except Exception as e:
        _log(f"[Pass1] failed: {str(e)[:200]}")
        return Pass1Result(main_concept="Unknown")

    main_concept      = data.get("main_concept",      "Unknown")
    concept_hierarchy = data.get("concept_hierarchy", [])
    relationships     = data.get("relationships",     [])

    if not concept_hierarchy:
        _log("[Pass1] no concept_hierarchy returned — check prompt / model")
        return Pass1Result(main_concept=main_concept, relationships=relationships)

    # Derive segments from concept last_page values (seq → actual mapping inside)
    segments = _derive_segments_from_hierarchy(concept_hierarchy, pages, seq_to_actual)

    # Merge L2 siblings that are too small; L1 fallback if all children small
    segments = _merge_small_segments(segments, concept_hierarchy)

    # Remap concept-name relationships → actual segment_ids after merge.
    # Orphaned L2 concepts (no dedicated segment) map to nearest sibling.
    relationships = _remap_relationships(relationships, segments, concept_hierarchy)

    # Derive sub_concepts from L1 entries — pages from actual segments
    seg_by_label = {s.label: s.source_pages for s in segments}
    sub_concepts = []
    for l1 in concept_hierarchy:
        sub_concepts.append({
            "name"    : l1["name"],
            "evidence": l1.get("evidence", ""),
            "pages"   : seg_by_label.get(l1["name"], []),
        })

    # Convert to {parent, children} format expected by knowledge_extractor (SIBLING_OF)
    hierarchy_compat = [
        {"parent": l1["name"], "children": [c["name"] for c in l1.get("children", [])]}
        for l1 in concept_hierarchy
        if l1.get("children")
    ]

    _log(f"[Pass1] done: {len(segments)} segments, "
         f"{len(relationships)} cross-relationships, "
         f"main={main_concept}")
    _log(f"[Pass1] hierarchy: {len(hierarchy_compat)} groups with children")

    return Pass1Result(
        main_concept      = main_concept,
        sub_concepts      = sub_concepts,
        relationships     = relationships,
        segments          = segments,
        concept_hierarchy = hierarchy_compat,
        raw_hierarchy     = concept_hierarchy,
    )


# ── Utility ────────────────────────────────────────────────────────────────

def _log(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


# ── CLI unit tests ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    pages = [
        {"page_num": p, "title": f"Slide {p}", "text": f"content of page {p}"}
        for p in range(1, 21)
    ]

    # Test 1: normal split
    raw_segs = [
        {"label": "Intro",     "concepts": [], "last_page": 3},
        {"label": "Main",      "concepts": [], "last_page": 10},
        {"label": "Conclusion","concepts": [], "last_page": 20},
    ]
    segs = _build_segments_from_breaks(raw_segs, pages)
    assert len(segs) == 3
    assert segs[0].source_pages == list(range(1, 4))
    assert segs[1].source_pages == list(range(4, 11))
    assert segs[2].source_pages == list(range(11, 21))
    print("[PASS] normal 3-segment split")

    # Test 2: last segment gets all remaining even if last_page is wrong
    raw_segs2 = [
        {"label": "A", "concepts": [], "last_page": 5},
        {"label": "B", "concepts": [], "last_page": 99},  # 99 > max page
    ]
    segs2 = _build_segments_from_breaks(raw_segs2, pages)
    assert segs2[0].source_pages == list(range(1, 6))
    assert segs2[1].source_pages == list(range(6, 21))
    print("[PASS] last segment gets remaining pages")

    # Test 3: single segment covers all pages
    raw_segs3 = [{"label": "All", "concepts": [], "last_page": 20}]
    segs3 = _build_segments_from_breaks(raw_segs3, pages)
    assert len(segs3) == 1
    assert segs3[0].source_pages == list(range(1, 21))
    print("[PASS] single segment covers all pages")

    # Test 4: build_whole_doc_text
    text = build_whole_doc_text(pages[:3])
    assert "[Slide 1]" in text
    assert "content of page 2" in text
    print("[PASS] build_whole_doc_text format correct")

    print("\n[OK] pass1_extractor unit tests passed")
