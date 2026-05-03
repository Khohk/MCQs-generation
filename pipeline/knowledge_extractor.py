"""
pipeline/knowledge_extractor.py
--------------------------------
Stage 1 — KU Extraction  : extract_kus(chunk, llm_fn)
Stage 2 — KU Graph Build : build_ku_graph, build_distractor_pool, compute_priority

llm_fn signature: (prompt: str, json_mode: bool) -> str
Pass generator's fallback wrapper or a mock for tests.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.schemas import KUItem
from pydantic import ValidationError


# ── Normalize ──────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip().replace("-", " "))


# ── verbatim grounding check ───────────────────────────────────────────────

def verify_ku(ku: dict, chunk_text: str) -> bool:
    """
    Check each verbatim fragment is grounded in chunk_text.
    Metric: recall = |evidence_words ∩ chunk_words| / |evidence_words| >= 0.8
    Uses re.findall(r'\\w+') to strip punctuation before comparison.
    Handles Combine case: "fragment A [...] fragment B".
    """
    evidence_raw = ku.get("verbatim_evidence", "").strip()
    if not evidence_raw:
        return False

    chunk_words = set(re.findall(r'\w+', chunk_text.lower()))
    parts = [p.strip() for p in evidence_raw.split("[...]") if p.strip()]

    for part in parts:
        part_words = set(re.findall(r'\w+', part.lower()))
        if not part_words:
            continue
        recall = len(part_words & chunk_words) / len(part_words)
        if recall < 0.8:
            return False

    return True


# ── Filter ─────────────────────────────────────────────────────────────────

def filter_kus(kus: list[dict], chunk_text: str) -> list[dict]:
    """Keep complete KUs whose verbatim_evidence is grounded in chunk_text."""
    return [
        ku for ku in kus
        if ku.get("completeness") == "complete" and verify_ku(ku, chunk_text)
    ]


# ── Dedup ──────────────────────────────────────────────────────────────────

def dedup_kus(all_kus: list[dict]) -> list[dict]:
    """Remove duplicate KUs by (type, normalized concept). Keep first occurrence."""
    seen: set[tuple] = set()
    result = []
    for ku in all_kus:
        key = (ku["type"], _normalize(ku["concept"]))
        if key not in seen:
            seen.add(key)
            result.append(ku)
    return result


# ── Parse LLM response ─────────────────────────────────────────────────────

def _parse_ku_response(raw: str, chunk_id: str) -> list[dict]:
    """Parse and Pydantic-validate LLM JSON response into list of KU dicts."""
    text = raw.strip()

    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()

    data = json.loads(text)

    if isinstance(data, dict):
        for key in ("knowledge_units", "kus", "units", "items"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            data = [data]

    validated = []
    for i, item in enumerate(data, start=1):
        # Auto-fix ku_id if model didn't follow format
        if not isinstance(item, dict):
            _log(f"    [KU pydantic] skip item {i} — not a dict: {str(item)[:80]}")
            continue
        ku_id = item.get("ku_id", "")
        if not ku_id or not str(ku_id).startswith(chunk_id):
            item["ku_id"] = f"{chunk_id}_ku_{i:02d}"

        try:
            validated.append(KUItem.model_validate(item).model_dump())
        except ValidationError as e:
            fields = [err["loc"] for err in e.errors()]
            _log(f"    [KU pydantic] skip item {i} — {fields}")
            _log(f"    [KU pydantic] item keys: {list(item.keys())[:8]}")

    return validated


# ── Main extraction entry point ────────────────────────────────────────────

def extract_kus(chunk: dict, llm_fn) -> list[dict]:
    """
    Extract Knowledge Units from 1 chunk.

    Args:
        chunk  : chunk dict from chunker {chunk_id, topic, pages, text, ...}
        llm_fn : callable(prompt: str, json_mode: bool) -> str

    Returns:
        List of validated, grounded KU dicts (completeness=complete, verify passed).
        Empty list if extraction fails — caller decides whether to fallback.
    """
    from prompts.ku_extraction_prompt import build_extraction_prompt

    chunk_id   = chunk["chunk_id"]
    chunk_text = chunk["text"]
    prompt     = build_extraction_prompt(chunk)

    try:
        # json_mode=False: extraction prompt returns a JSON array [], but
        # OpenAI-compat json_object mode requires a {} wrapper → conflict.
        # Gemini also hardcodes MCQItem schema in json_mode=True, wrong for KU.
        # Prompt is explicit enough; let the model follow instructions directly.
        raw = llm_fn(prompt, False)

        if not raw or not raw.strip():
            _log(f"    [KU] {chunk_id}: empty response from model — skip")
            return []

        kus  = _parse_ku_response(raw, chunk_id)
        kept = filter_kus(kus, chunk_text)
        _log(f"    [KU] {chunk_id}: {len(kus)} extracted → {len(kept)} passed filter")
        return kept

    except json.JSONDecodeError as e:
        _log(f"    [KU] {chunk_id}: JSON parse error — {e}")
        _log(f"    [KU] raw response preview: {repr(raw[:200]) if 'raw' in dir() else 'N/A'}")
        return []
    except Exception as e:
        _log(f"    [KU] {chunk_id}: extraction failed — {str(e)[:120]}")
        return []


# ── Graph build ────────────────────────────────────────────────────────────

def build_ku_graph(all_kus: list[dict]) -> dict[str, list[str]]:
    """
    Build bidirectional adjacency dict: ku_id → [connected_ku_id, ...].
    Edge created when KU_A.related_concepts overlaps with KU_B.concept.
    Multiple KUs can share a concept name across types (→ list in index).
    """
    concept_index: dict[str, list[str]] = {}
    for ku in all_kus:
        key = _normalize(ku["concept"])
        concept_index.setdefault(key, []).append(ku["ku_id"])

    graph: dict[str, list[str]] = {ku["ku_id"]: [] for ku in all_kus}

    for ku in all_kus:
        for related in ku.get("related_concepts", []):
            for target_id in concept_index.get(_normalize(related), []):
                if target_id == ku["ku_id"]:
                    continue
                if target_id not in graph[ku["ku_id"]]:
                    graph[ku["ku_id"]].append(target_id)
                if ku["ku_id"] not in graph.setdefault(target_id, []):
                    graph[target_id].append(ku["ku_id"])

    return graph


def build_distractor_pool(all_kus: list[dict]) -> dict[str, list[dict]]:
    """
    Group KUs by type → {type: [{concept, content}, ...]}.
    Generation prompt injects same-type entries as distractor raw material.
    """
    pool: dict[str, list[dict]] = {}
    for ku in all_kus:
        pool.setdefault(ku["type"], []).append({
            "concept": ku["concept"],
            "content": ku["content"],
        })
    return pool


# ── Priority score ─────────────────────────────────────────────────────────

def compute_priority(ku: dict, graph: dict, all_kus: list[dict]) -> float:
    """
    Score 0.1 (isolated peripheral) → ~2.6 (primary hub, frequently referenced).
    Used to gate cross-concept MCQ Tier 2 / Tier 3.

    Components:
      prominence    : primary=1.0, supporting=0.5, peripheral=0.1
      centrality    : edges * 0.3, capped at 1.0
      frequency     : how often this concept appears in others' related_concepts
    """
    prominence_map = {"primary": 1.0, "supporting": 0.5, "peripheral": 0.1}
    prominence     = prominence_map.get(ku.get("prominence", "supporting"), 0.5)

    n_edges    = len(graph.get(ku["ku_id"], []))
    centrality = min(n_edges * 0.3, 1.0)

    concept_norm = _normalize(ku["concept"])
    frequency = sum(
        1 for k in all_kus
        if concept_norm in [_normalize(r) for r in k.get("related_concepts", [])]
    )
    frequency_score = min(frequency * 0.2, 0.6)

    return round(prominence + centrality + frequency_score, 3)


# ── Utility ────────────────────────────────────────────────────────────────

def _log(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))
