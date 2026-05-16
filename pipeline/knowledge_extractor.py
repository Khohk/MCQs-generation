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

from pipeline.schemas import KUItem, HORIZONTAL_RELATIONS, STRUCTURAL_RELATIONS
from pydantic import ValidationError


# ── Normalize ──────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip().replace("-", " "))


def _fuzzy_lookup(name: str, concept_index: dict[str, list[str]]) -> list[str]:
    """
    Exact normalized match first, then token-subset fallback.
    Handles long descriptive names ("The LSTM: Long short-term memory network" → "lstm")
    and plurals ("rnns" → "rnn") via simple de-pluralizer.
    """
    norm = _normalize(name)
    exact = concept_index.get(norm)
    if exact:
        return exact

    def _depl(t: str) -> str:
        return t.rstrip("s") if len(t) > 3 else t

    query_tokens = {_depl(t) for t in re.findall(r"\w+", norm) if t}
    if not query_tokens:
        return []

    results: list[str] = []
    seen_ids: set[str] = set()
    for key, ids in concept_index.items():
        key_tokens = {_depl(t) for t in re.findall(r"\w+", key) if t}
        if not key_tokens:
            continue
        if key_tokens <= query_tokens or query_tokens <= key_tokens:
            for kid in ids:
                if kid not in seen_ids:
                    results.append(kid)
                    seen_ids.add(kid)
    return results


def _seg_of_ku(ku_id: str) -> str:
    """Extract segment prefix: 'seg_003_ku_01' → 'seg_003'."""
    parts = ku_id.rsplit("_ku_", 1)
    return parts[0] if len(parts) == 2 else ""


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

def _normalize_ku_ownership(ku: dict, chunk: dict) -> dict:
    """Backfill KU ownership metadata so UI mapping does not rely on fuzzy names."""
    ku = dict(ku)
    topic = str(chunk.get("topic", "") or "").strip()
    parent_l1 = str(chunk.get("parent_concept_name", "") or "").strip()
    parent_l2 = topic if parent_l1 else ""
    concept = str(ku.get("concept", "") or "").strip()
    owner_concepts = [
        str(c or "").strip()
        for c in (chunk.get("owner_concepts") or [])
        if str(c or "").strip()
    ]

    ku["local_concept"] = str(ku.get("local_concept") or concept).strip()
    ku["parent_l1"] = str(ku.get("parent_l1") or parent_l1 or topic).strip()
    ku["parent_l2"] = str(ku.get("parent_l2") or parent_l2).strip()

    owner_concept = str(ku.get("owner_concept") or "").strip()
    owner_level = str(ku.get("owner_level") or "").strip()
    valid_owners = owner_concepts or [c for c in (parent_l1, topic) if c]

    def _owner_match(name: str) -> str:
        norm = _normalize(name)
        for owner in valid_owners:
            if _normalize(owner) == norm:
                return owner
        return ""

    if owner_concept not in valid_owners:
        matched = _owner_match(owner_concept) or _owner_match(concept) or _owner_match(ku["local_concept"])
        if matched:
            owner_concept = matched
        elif parent_l1 and _normalize(concept) == _normalize(parent_l1):
            owner_concept = parent_l1
        elif topic in valid_owners:
            owner_concept = topic
        elif parent_l1:
            owner_concept = parent_l1
        elif valid_owners:
            owner_concept = valid_owners[0]

    if owner_level not in {"L1", "L2", "local"}:
        if owner_concept and parent_l1 and _normalize(owner_concept) == _normalize(parent_l1):
            owner_level = "L1"
        elif owner_concept and (
            _normalize(owner_concept) == _normalize(topic)
            or owner_concept in valid_owners
        ):
            owner_level = "L2" if parent_l1 else "L1"
        else:
            owner_level = "local"

    if parent_l1 and owner_level == "L2" and owner_concept:
        ku["parent_l2"] = owner_concept

    ku["owner_concept"] = owner_concept
    ku["owner_level"] = owner_level
    return ku


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


# ── Edge selection constants ───────────────────────────────────────────────

_HORIZONTAL = {"CONTRASTS_WITH", "ALTERNATIVE_TO", "SIMILAR_TO", "SIBLING_OF"}

# KU types that are meaningful endpoints for each relation
RELATION_TO_KU_TYPES: dict[str, list[str]] = {
    "CONTRASTS_WITH" : ["trade_off", "failure_mode", "definition"],
    "ALTERNATIVE_TO" : ["definition", "procedure", "mechanism"],
    "SIMILAR_TO"     : ["definition", "mechanism"],
    "SIBLING_OF"     : ["definition", "mechanism"],
    "ENABLES"        : ["definition", "mechanism"],
    "APPLIES_TO"     : ["application", "procedure"],
    "EXTENDS"        : ["definition", "mechanism"],
}

# Per-relation intra-segment edge caps
MAX_INTRA_PER_RELATION: dict[str, int] = {
    "CONTRASTS_WITH" : 2,
    "ALTERNATIVE_TO" : 2,
    "SIMILAR_TO"     : 1,
}
_DEFAULT_INTRA_CAP = 1


# ── Main extraction entry point ────────────────────────────────────────────

def _verify_and_cap_edges(ku_dict: dict, chunk_text: str) -> dict:
    edges = ku_dict.get("related_kus", [])
    if not edges:
        return ku_dict

    verified = [
        e for e in edges
        if e.get("evidence") and verify_ku({"verbatim_evidence": e["evidence"]}, chunk_text)
    ]

    # Group by relation, apply per-relation caps (horizontal first within each group)
    from collections import defaultdict
    by_rel: dict[str, list] = defaultdict(list)
    for e in verified:
        by_rel[e.get("relation", "")].append(e)

    capped = []
    # Emit horizontal groups first, then structural
    for rel in sorted(by_rel, key=lambda r: 0 if r in _HORIZONTAL else 1):
        cap = MAX_INTRA_PER_RELATION.get(rel, _DEFAULT_INTRA_CAP)
        capped.extend(by_rel[rel][:cap])

    _log(f"    [edge-cap] {ku_dict.get('ku_id','')} "
         f"raw={len(edges)} verified={len(verified)} capped={len(capped)}")

    ku_dict = dict(ku_dict)
    ku_dict["related_kus"]      = capped
    ku_dict["related_concepts"] = [e["target_concept"] for e in capped]
    return ku_dict


def extract_kus(
    chunk: dict,
    llm_fn,
    cross_rels: list[dict] | None = None,
    doc_concepts: dict | None = None,
    seen_concepts: list[str] | None = None,
) -> list[dict]:
    """
    Extract Knowledge Units from 1 chunk.

    Args:
        chunk         : chunk dict from chunker {chunk_id, topic, pages, text, ...}
        llm_fn        : callable(prompt: str, json_mode: bool) -> str
        cross_rels    : cross-segment relationship hints from Pass 1
        doc_concepts  : {"seg": [...], "all": [...]} concept lists for naming constraint
        seen_concepts : concepts already covered — triggers recap filter when provided

    Returns:
        List of validated, grounded KU dicts (completeness=complete, verify passed).
        Empty list if extraction fails — caller decides whether to fallback.
    """
    from prompts.ku_extraction_prompt import build_extraction_prompt

    chunk_id   = chunk["chunk_id"]
    chunk_text = chunk["text"]
    prompt     = build_extraction_prompt(chunk, cross_rels, doc_concepts, seen_concepts)

    try:
        # json_mode=False: extraction prompt returns a JSON object {}, but
        # OpenAI-compat json_object mode hardcodes MCQItem schema in json_mode=True.
        # Prompt is explicit enough; let the model follow instructions directly.
        raw = llm_fn(prompt, False)

        if not raw or not raw.strip():
            _log(f"    [KU] {chunk_id}: empty response from model — skip")
            return []

        kus  = _parse_ku_response(raw, chunk_id)
        kus  = [_normalize_ku_ownership(ku, chunk) for ku in kus]
        kus  = [_verify_and_cap_edges(ku, chunk_text) for ku in kus]
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
        if ku.get("related_kus"):
            related_names = [e["target_concept"] for e in ku["related_kus"]]
        else:
            related_names = ku.get("related_concepts", [])
        for related in related_names:
            for target_id in concept_index.get(_normalize(related), []):
                if target_id == ku["ku_id"]:
                    continue
                if target_id not in graph[ku["ku_id"]]:
                    graph[ku["ku_id"]].append(target_id)
                if ku["ku_id"] not in graph.setdefault(target_id, []):
                    graph[target_id].append(ku["ku_id"])

    return graph


def build_ku_graph_with_cross(
    all_kus: list[dict],
    cross_relationships: list[dict],
    concept_hierarchy:   list[dict] | None = None,
    segments:            list | None = None,
) -> tuple[dict[str, list[str]], dict[tuple[str, str], str]]:
    """
    Build graph + edge_types from:
      - KU related_kus (intra-segment, typed)
      - Pass 1 cross-segment relationships
      - SIBLING_OF auto-generated from Pass 1 concept_hierarchy

    segments: Pass1Segment list — extends concept_index with segment labels so
              cross-relationships using label names ("Tokenizing") resolve correctly.

    Returns:
        graph      : ku_id → [connected ku_id, ...]  (bidirectional)
        edge_types : (ku_id_a, ku_id_b) → relation   (canonical: a < b)
    """
    graph:      dict[str, list[str]]       = {ku["ku_id"]: [] for ku in all_kus}
    edge_types: dict[tuple[str, str], str] = {}
    ku_by_id:   dict[str, dict]            = {ku["ku_id"]: ku for ku in all_kus}
    _PROMINENCE_ORDER = {"primary": 0, "supporting": 1, "peripheral": 2}
    _PROMINENCE_BONUS = {"primary": 15, "supporting": 6, "peripheral": 0}
    _RELATION_TYPE_BONUS: dict[str, dict[str, int]] = {
        # For broad concept-level links, definition/mechanism usually represent
        # the concept better than a narrow failure mode.
        "CONTRASTS_WITH": {"definition": 24, "mechanism": 20, "trade_off": 16, "failure_mode": 12},
        "ALTERNATIVE_TO": {"definition": 24, "mechanism": 20, "procedure": 16, "application": 12},
        "SIMILAR_TO": {"definition": 22, "mechanism": 18, "procedure": 10, "application": 8},
        "SIBLING_OF": {"definition": 22, "mechanism": 18, "procedure": 10, "application": 8},
        "EXTENDS": {"definition": 22, "mechanism": 18, "procedure": 10, "application": 8},
        "ENABLES": {"mechanism": 20, "procedure": 18, "definition": 14, "application": 12},
        "APPLIES_TO": {"application": 22, "procedure": 20, "mechanism": 14, "definition": 12},
    }

    def _tokens(text: str) -> set[str]:
        def _depl(t: str) -> str:
            return t.rstrip("s") if len(t) > 3 else t
        return {_depl(t) for t in re.findall(r"\w+", _normalize(text)) if t}

    def _same_concept(a: str, b: str) -> bool:
        ka = ku_by_id.get(a, {})
        kb = ku_by_id.get(b, {})
        oa = _normalize(ka.get("owner_concept") or ka.get("parent_l2") or "")
        ob = _normalize(kb.get("owner_concept") or kb.get("parent_l2") or "")
        if oa and ob and oa != ob:
            return False
        ca = _normalize(ka.get("concept", ""))
        cb = _normalize(kb.get("concept", ""))
        return bool(ca and cb and ca == cb)

    def _text_match_score(value: str, target_name: str, exact: int, subset: int, overlap_base: int) -> int:
        value_norm = _normalize(str(value or ""))
        target_norm = _normalize(str(target_name or ""))
        if not value_norm or not target_norm:
            return 0
        if value_norm == target_norm:
            return exact

        vt = _tokens(value_norm)
        tt = _tokens(target_norm)
        if not vt or not tt:
            return 0
        overlap = len(vt & tt)
        if not overlap:
            return 0
        if vt <= tt or tt <= vt:
            return subset + min(12, overlap * 3)
        return overlap_base + min(18, overlap * 4)

    def _local_match_score(kid: str, target_name: str) -> int:
        ku = ku_by_id.get(kid, {})
        return max(
            _text_match_score(ku.get("concept", ""), target_name, 120, 94, 44),
            _text_match_score(ku.get("local_concept", ""), target_name, 116, 90, 42),
        )

    def _owner_match_score(kid: str, target_name: str) -> int:
        ku = ku_by_id.get(kid, {})
        return max(
            _text_match_score(ku.get("owner_concept", ""), target_name, 35, 26, 8),
            _text_match_score(ku.get("parent_l2", ""), target_name, 30, 22, 7),
            # parent_l1 is intentionally weak. It helps true L1 lookup but
            # should not make every child KU look like the L1 representative.
            _text_match_score(ku.get("parent_l1", ""), target_name, 14, 10, 4),
        )

    def _evidence_score(kid: str) -> int:
        words = re.findall(r"\w+", str(ku_by_id.get(kid, {}).get("verbatim_evidence", "")))
        if len(words) < 6:
            return -5
        return min(8, len(words) // 10)

    def _type_bonus(kid: str, relation: str) -> int:
        ku_type = ku_by_id.get(kid, {}).get("type", "")
        return (_RELATION_TYPE_BONUS.get(relation, {}) or {}).get(ku_type, 0)

    def _representative_score(kid: str, target_name: str = "", relation: str = "") -> tuple[int, int, int]:
        """Higher is better. Concept/local match dominates all tie-breakers."""
        local_score = _local_match_score(kid, target_name)
        owner_score = _owner_match_score(kid, target_name)
        prominence = _PROMINENCE_BONUS.get(ku_by_id.get(kid, {}).get("prominence", "peripheral"), 0)
        centrality = min(len(graph.get(kid, [])), 3) * 2
        fallback_penalty = 0
        if target_name and local_score == 0 and owner_score > 0:
            # KU belongs to the requested owner, but its precise tested concept
            # does not resemble the target. Keep it as fallback, not as first pick.
            fallback_penalty = -18
        elif target_name and local_score == 0 and owner_score == 0:
            fallback_penalty = -40

        total = (
            local_score
            + owner_score
            + _type_bonus(kid, relation)
            + prominence
            + _evidence_score(kid)
            + centrality
            + fallback_penalty
        )
        return total, local_score, owner_score

    def _concept_match_score(kid: str, target_name: str) -> int:
        """Compatibility helper: higher is better."""
        if not target_name:
            return 0
        return _representative_score(kid, target_name)[0]

    def _rank_ids(ku_ids: list[str], target_name: str = "", relation: str = "") -> list[str]:
        seen: set[str] = set()
        unique = [kid for kid in ku_ids if kid in ku_by_id and not (kid in seen or seen.add(kid))]
        return sorted(
            unique,
            key=lambda kid: (
                -_representative_score(kid, target_name, relation)[0],
                -_representative_score(kid, target_name, relation)[1],
                -_representative_score(kid, target_name, relation)[2],
                _PROMINENCE_ORDER.get(ku_by_id.get(kid, {}).get("prominence", "peripheral"), 2),
                kid,
            ),
        )

    def _add_edge(a: str, b: str, relation: str) -> bool:
        if a == b:
            return False
        if a not in ku_by_id or b not in ku_by_id:
            return False
        if _same_concept(a, b):
            return False
        if b not in graph.setdefault(a, []):
            graph[a].append(b)
        if a not in graph.setdefault(b, []):
            graph[b].append(a)
        key = (min(a, b), max(a, b))
        if key not in edge_types:
            edge_types[key] = relation
        return True

    def _pick_primary(ku_ids: list[str], target_name: str = "", relation: str = "") -> str | None:
        ranked = _rank_ids(ku_ids, target_name, relation)
        return ranked[0] if ranked else None

    def _ku_type(kid: str) -> str:
        return str(ku_by_id.get(kid, {}).get("type", "") or "")

    def _mode_bonus(fid: str, tid: str, relation: str, mode: str) -> int | None:
        ft, tt = _ku_type(fid), _ku_type(tid)
        pair = {ft, tt}
        broad = {"definition", "mechanism", "trade_off"}
        process = {"procedure", "application", "mechanism"}

        if mode == "compare":
            return 90 if ft in broad and tt in broad else None
        if mode == "limitation":
            return 78 if "failure_mode" in pair and (pair - {"failure_mode"}) & {"definition", "mechanism", "procedure"} else None
        if mode == "same_type":
            return 84 if ft == tt else None
        if mode == "approach":
            return 76 if ft in {"definition", "procedure", "mechanism", "application"} and tt in {"definition", "procedure", "mechanism", "application"} else None
        if mode == "application":
            return 76 if ft in process and tt in process else None
        if mode == "builds_on":
            return 72 if ft in {"definition", "mechanism"} and tt in {"definition", "mechanism"} else None
        return None

    def _relation_modes(relation: str) -> tuple[list[str], int]:
        if relation == "CONTRASTS_WITH":
            return ["compare", "limitation"], 2
        if relation == "ALTERNATIVE_TO":
            return ["same_type", "approach", "limitation"], 2
        if relation in {"SIMILAR_TO", "SIBLING_OF"}:
            return ["same_type", "compare"], 1
        if relation == "APPLIES_TO":
            return ["application", "approach"], 1
        if relation in {"EXTENDS", "ENABLES"}:
            return ["builds_on", "application"], 1
        return ["compare", "same_type"], 1

    def _select_endpoints(
        from_ids: list[str],
        to_ids: list[str],
        relation: str,
        from_name: str = "",
        to_name: str = "",
    ) -> list[tuple[str, str]]:
        """Pick representative endpoint pairs by relation/type mode."""
        f_ranked = _rank_ids(from_ids, from_name, relation)
        t_ranked = _rank_ids(to_ids, to_name, relation)
        modes, max_pairs = _relation_modes(relation)
        selected: list[tuple[str, str]] = []
        selected_keys: set[tuple[str, str]] = set()

        def _best_for_mode(mode: str) -> tuple[str, str] | None:
            best_pair: tuple[str, str] | None = None
            best_score = -10**9
            for f in f_ranked[:6]:
                for t in t_ranked[:6]:
                    if f == t or _same_concept(f, t):
                        continue
                    key = (min(f, t), max(f, t))
                    if key in selected_keys:
                        continue
                    bonus = _mode_bonus(f, t, relation, mode)
                    if bonus is None:
                        continue
                    score = (
                        _representative_score(f, from_name, relation)[0]
                        + _representative_score(t, to_name, relation)[0]
                        + bonus
                    )
                    if score > best_score:
                        best_score = score
                        best_pair = (f, t)
            return best_pair

        for mode in modes:
            pair = _best_for_mode(mode)
            if not pair:
                continue
            key = (min(pair[0], pair[1]), max(pair[0], pair[1]))
            selected.append(pair)
            selected_keys.add(key)
            if len(selected) >= max_pairs:
                return selected

        # Fallback keeps one reasonable pair if no typed mode matched.
        if not selected:
            best_pair: tuple[str, str] | None = None
            best_score = -10**9
            for f in f_ranked[:4]:
                for t in t_ranked[:4]:
                    if f == t or _same_concept(f, t):
                        continue
                    score = (
                        _representative_score(f, from_name, relation)[0]
                        + _representative_score(t, to_name, relation)[0]
                    )
                    if score > best_score:
                        best_score = score
                        best_pair = (f, t)
            if best_pair:
                selected.append(best_pair)
        return selected

    concept_index: dict[str, list[str]] = {}
    for ku in all_kus:
        for name in (
            ku.get("concept", ""),
            ku.get("local_concept", ""),
            ku.get("owner_concept", ""),
            ku.get("parent_l1", ""),
            ku.get("parent_l2", ""),
        ):
            norm_name = _normalize(str(name or ""))
            if not norm_name:
                continue
            existing = concept_index.setdefault(norm_name, [])
            if ku["ku_id"] not in existing:
                existing.append(ku["ku_id"])
    _log(f"  [graph] concept_index from KU concepts: {len(concept_index)} keys")

    # Extend concept_index with segment labels + seg.concepts so cross-relationships
    # that reference segment labels ("Tokenizing") resolve even when no KU has that name.
    if segments:
        seg_ku_map: dict[str, list[str]] = {}
        for ku in all_kus:
            seg_ku_map.setdefault(_seg_of_ku(ku["ku_id"]), []).append(ku["ku_id"])
        for seg in segments:
            ku_ids = seg_ku_map.get(seg.segment_id, [])
            if not ku_ids:
                _log(f"  [graph] segment {seg.segment_id} ('{seg.label[:30]}') has no KUs — skip")
                continue
            for raw_name in [seg.label] + list(getattr(seg, "concepts", []) or []):
                norm_key = _normalize(raw_name)
                if not norm_key or concept_index.get(norm_key):
                    continue
                # Segment aliases are broad. Index only representative KUs so an
                # orphan L2 mapped to a sibling segment does not connect to every
                # KU in that segment.
                alias_ids = _rank_ids(ku_ids, raw_name)[:2]
                existing = concept_index.setdefault(norm_key, [])
                for kid in alias_ids:
                    if kid not in existing:
                        existing.append(kid)
        _log(f"  [graph] concept_index after segment labels: {len(concept_index)} keys")
        _log(f"  [graph] segment KU map: { {sid: len(ids) for sid, ids in seg_ku_map.items()} }")

    def _resolve_ids(name: str, relation: str = "", max_ids: int = 3) -> list[str]:
        return _rank_ids(_fuzzy_lookup(name, concept_index), name, relation)[:max_ids]

    def _has_seq_rel(name_a: str, name_b: str) -> bool:
        """True if cross_relationships has ENABLES/EXTENDS between name_a and name_b."""
        def _depl(t: str) -> str:
            return t.rstrip("s") if len(t) > 3 else t
        def _toks(s: str) -> set:
            return {_depl(t) for t in re.findall(r"\w+", _normalize(s)) if t}
        ta, tb = _toks(name_a), _toks(name_b)
        for r in cross_relationships:
            if r.get("relation") not in {"ENABLES", "EXTENDS"}:
                continue
            tf = _toks(r.get("from_concept", ""))
            tt = _toks(r.get("to_concept", ""))
            if not tf or not tt:
                continue
            fma = tf <= ta or ta <= tf
            tmb = tt <= tb or tb <= tt
            fmb = tf <= tb or tb <= tf
            tma = tt <= ta or ta <= tt
            if (fma and tmb) or (fmb and tma):
                return True
        return False

    # Intra-segment edges from related_kus
    n_intra = 0
    for ku in all_kus:
        edges = ku.get("related_kus", [])
        for edge in edges:
            rel    = edge.get("relation", "") if isinstance(edge, dict) else edge.relation
            t_name = edge.get("target_concept", "") if isinstance(edge, dict) else edge.target_concept
            matches = _resolve_ids(t_name, rel, max_ids=1)
            if not matches:
                _log(f"  [graph] intra miss: '{t_name}' not in concept_index")
            for tid in matches:
                if _add_edge(ku["ku_id"], tid, rel):
                    n_intra += 1
    _log(f"  [graph] intra edges added: {n_intra}")

    # Cross-segment edges from Pass 1 relationships
    n_cross = 0
    _log(f"  [graph] processing {len(cross_relationships)} cross-relationships")
    for rel in cross_relationships:
        relation = rel.get("relation", "ENABLES")
        fc       = rel.get("from_concept", "")
        tc       = rel.get("to_concept",   "")
        fc_norm  = _normalize(fc)
        tc_norm  = _normalize(tc)

        from_ids = _resolve_ids(fc, relation, max_ids=3)
        to_ids   = _resolve_ids(tc, relation, max_ids=3)

        from_how = "exact" if concept_index.get(fc_norm) else ("fuzzy" if from_ids else "MISS")
        to_how   = "exact" if concept_index.get(tc_norm) else ("fuzzy" if to_ids else "MISS")

        _log(f"  [graph] cross [{relation}]  '{fc}' ({from_how},{len(from_ids)}) "
             f"→ '{tc}' ({to_how},{len(to_ids)})")

        if from_how == "MISS":
            # Show normalized key + nearest concept_index keys to help diagnose mismatch
            _log(f"    norm='{fc_norm}'")
            close = [k for k in concept_index if any(
                t in k for t in fc_norm.split() if len(t) > 3)][:5]
            _log(f"    closest index keys: {close}")
        if to_how == "MISS":
            _log(f"    norm='{tc_norm}'")
            close = [k for k in concept_index if any(
                t in k for t in tc_norm.split() if len(t) > 3)][:5]
            _log(f"    closest index keys: {close}")

        if from_ids and to_ids:
            pairs = _select_endpoints(from_ids, to_ids, relation, fc, tc)
            if not pairs:
                from_types = [ku_by_id.get(i, {}).get("type", "?") for i in from_ids[:3]]
                to_types   = [ku_by_id.get(i, {}).get("type", "?") for i in to_ids[:3]]
                _log(f"    _select_endpoints: 0 pairs  "
                     f"from_types={from_types}  to_types={to_types}  "
                     f"target_types={RELATION_TO_KU_TYPES.get(relation, ['?'])}")
            for fid, tid in pairs:
                added = _add_edge(fid, tid, relation)
                if added:
                    _log(f"    edge added: {fid} ↔ {tid}  [{relation}]")
                    n_cross += 1
    _log(f"  [graph] cross edges added: {n_cross}")

    def _matches_child_owner(kid: str, child_name: str) -> bool:
        target = _normalize(child_name)
        if not target:
            return False
        ku = ku_by_id.get(kid, {})
        for key in ("owner_concept", "parent_l2", "concept", "local_concept"):
            if _normalize(str(ku.get(key, "") or "")) == target:
                return True
        return False

    def _owner_key(kid: str) -> str:
        ku = ku_by_id.get(kid, {})
        return _normalize(str(
            ku.get("owner_concept")
            or ku.get("parent_l2")
            or ku.get("concept")
            or ""
        ))

    # Auto SIBLING_OF from Pass 1 hierarchy.
    # Connect primary KU per sibling pair — skip pairs connected by ENABLES/EXTENDS
    # (sequential pipeline steps are not good distractors for each other).
    n_sibling = 0
    for group in (concept_hierarchy or []):
        children = group.get("children", [])
        child_ku_groups = []
        for name in children:
            resolved = _resolve_ids(name, "SIBLING_OF", max_ids=4)
            child_ku_groups.append([
                kid for kid in resolved
                if _matches_child_owner(kid, name)
            ])
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                if _has_seq_rel(children[i], children[j]):
                    _log(f"  [graph] sibling skip (sequential): {children[i]} ↔ {children[j]}")
                    continue
                pa = _pick_primary(child_ku_groups[i], children[i], "SIBLING_OF")
                pb = _pick_primary(child_ku_groups[j], children[j], "SIBLING_OF")
                if not (pa and pb) or pa == pb:
                    continue
                if not _owner_key(pa) or _owner_key(pa) == _owner_key(pb):
                    continue
                added = _add_edge(pa, pb, "SIBLING_OF")
                if added:
                    n_sibling += 1
                    _log(f"  [graph] sibling: {pa} â†” {pb}")
                continue
                if pa and pb:
                    if _seg_of_ku(pa) == _seg_of_ku(pb):
                        _log(f"  [graph] sibling skip (same seg): {pa} ↔ {pb}")
                        continue
                    added = _add_edge(pa, pb, "SIBLING_OF")
                    if added:
                        n_sibling += 1
                        _log(f"  [graph] sibling: {pa} ↔ {pb}")
    _log(f"  [graph] sibling edges added: {n_sibling}")

    cap_graph_edges(graph, edge_types)
    return graph, edge_types


def build_distractor_pool(all_kus: list[dict]) -> dict[str, list[dict]]:
    """
    Tier 3 fallback: group KUs by type → {type: [{concept, content}, ...]}.
    Used only when graph-neighbor distractors (Tier 1/2) are insufficient.
    """
    pool: dict[str, list[dict]] = {}
    for ku in all_kus:
        pool.setdefault(ku["type"], []).append({
            "concept": ku["concept"],
            "content": ku["content"],
        })
    return pool


def get_distractors(
    anchor_ku_id: str,
    graph:        dict[str, list[str]],
    edge_types:   dict[tuple[str, str], str],
    all_kus:      list[dict],
    n:            int = 3,
) -> list[dict]:
    """
    Select up to n distractor KUs for an anchor KU.

    Candidates are scored instead of appended bucket-by-bucket so we can keep
    provenance and prefer plausible-but-distinct distractors:
      - explicit horizontal edges first
      - same owner_concept + same type
      - same parent_l1 + same type
      - same parent_l1 + related type
      - global same type
      - LLM-generated filler if fewer than three distractors remain
    """
    ku_by_id    = {ku["ku_id"]: ku for ku in all_kus}
    anchor      = ku_by_id.get(anchor_ku_id)
    if not anchor:
        return []

    anchor_type    = anchor.get("type")
    anchor_concept = _normalize(anchor.get("concept", ""))
    anchor_owner   = _normalize(anchor.get("owner_concept") or anchor.get("parent_l2") or "")
    anchor_l1      = _normalize(anchor.get("parent_l1", ""))
    hub_ids        = get_hub_ids(graph)

    def _rel(a: str, b: str) -> str:
        return edge_types.get((min(a, b), max(a, b)), "")

    neighbors  = graph.get(anchor_ku_id, [])
    candidates = [nid for nid in neighbors if nid in ku_by_id and nid not in hub_ids]

    def _strong_distractor_edge(nid: str) -> bool:
        rel = _rel(anchor_ku_id, nid)
        other_type = ku_by_id[nid].get("type", "")
        pair = {anchor_type, other_type}
        if rel in {"CONTRASTS_WITH", "ALTERNATIVE_TO"}:
            if "failure_mode" in pair and not (pair <= {"failure_mode", "trade_off"}):
                # Useful for cross-MCQ limitation questions, but usually too
                # asymmetric to be a strong single-KU distractor.
                return False
            return True
        if rel == "SIMILAR_TO":
            same_type = other_type == anchor_type
            same_owner = _same_owner(ku_by_id[nid])
            same_l1 = _same_l1(ku_by_id[nid])
            related_type = (
                pair <= {"definition", "mechanism"}
                or pair <= {"mechanism", "procedure", "application"}
                or pair <= {"failure_mode", "trade_off"}
            )
            return same_type or ((same_owner or same_l1) and related_type)
        return False

    def _tokens(text: str) -> set[str]:
        return {t for t in re.findall(r"\w+", _normalize(text or "")) if len(t) > 2}

    anchor_concept_tokens = _tokens(anchor.get("concept", ""))
    anchor_content_tokens = _tokens(anchor.get("content", ""))

    def _owner_of(ku: dict) -> str:
        return _normalize(ku.get("owner_concept") or ku.get("parent_l2") or "")

    def _same_owner(ku: dict) -> bool:
        return bool(anchor_owner and _owner_of(ku) == anchor_owner)

    def _same_l1(ku: dict) -> bool:
        return bool(anchor_l1 and _normalize(ku.get("parent_l1", "")) == anchor_l1)

    def _related_type_bonus(other_type: str) -> int:
        if other_type == anchor_type:
            return 14
        type_groups = [
            {"definition", "mechanism"},
            {"mechanism", "procedure", "application"},
            {"failure_mode", "trade_off"},
        ]
        if any(anchor_type in group and other_type in group for group in type_groups):
            return 5
        return -10

    def _overlap_ratio(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / max(1, min(len(a), len(b)))

    def _duplicate_or_too_similar_penalty(ku: dict) -> int:
        """Penalize near-duplicates that differ mostly by wording."""
        concept_overlap = _overlap_ratio(anchor_concept_tokens, _tokens(ku.get("concept", "")))
        content_overlap = _overlap_ratio(anchor_content_tokens, _tokens(ku.get("content", "")))
        penalty = 0
        if concept_overlap >= 0.90:
            penalty += 28
        elif concept_overlap >= 0.70:
            penalty += 14
        if content_overlap >= 0.82:
            penalty += 24
        elif content_overlap >= 0.68:
            penalty += 12
        if _same_owner(ku) and concept_overlap >= 0.65:
            penalty += 8
        return penalty

    def _topic_distance_penalty(ku: dict, source: str, rel: str) -> int:
        """Penalize random-but-same-type fallbacks and structural-only links."""
        if source == "explicit_horizontal":
            return 0

        penalty = 0
        ku_l1 = _normalize(ku.get("parent_l1", ""))
        if anchor_l1 and ku_l1 and ku_l1 != anchor_l1:
            penalty += 18
        if not _same_owner(ku) and not _same_l1(ku):
            penalty += 12
        if source == "global_same_type" and anchor_l1 and not _same_l1(ku):
            penalty += 10
        if rel in STRUCTURAL_RELATIONS:
            penalty += 18
        return penalty

    def _evidence_bonus(ku: dict) -> int:
        n_words = len(re.findall(r"\w+", str(ku.get("verbatim_evidence", ""))))
        if n_words < 6:
            return -4
        return min(6, n_words // 12)

    scored: dict[str, dict] = {}

    def _add_candidate(kid: str, source: str, base: int, rel: str = "") -> None:
        if kid == anchor_ku_id or kid in hub_ids or kid not in ku_by_id:
            return
        ku = ku_by_id[kid]
        if _normalize(ku.get("concept", "")) == anchor_concept:
            return

        score = base
        score += _related_type_bonus(str(ku.get("type", "") or ""))
        score += 12 if _same_owner(ku) else 0
        score += 7 if _same_l1(ku) else 0
        score += _evidence_bonus(ku)
        score -= _duplicate_or_too_similar_penalty(ku)
        score -= _topic_distance_penalty(ku, source, rel)

        if rel in {"CONTRASTS_WITH", "ALTERNATIVE_TO"}:
            score += 8
        elif rel == "SIMILAR_TO":
            score += 3

        prev = scored.get(kid)
        if not prev or score > prev["score"]:
            scored[kid] = {
                "kid": kid,
                "source": source,
                "relation": rel,
                "score": score,
            }

    for nid in candidates:
        rel = _rel(anchor_ku_id, nid)
        other = ku_by_id[nid]
        if rel == "SIBLING_OF":
            # SIBLING_OF is auto-generated, so treat it as same-L1 evidence
            # instead of a strong explicit relation.
            if other.get("type") == anchor_type and _same_l1(other):
                _add_candidate(nid, "same_l1_same_type", 72, rel)
            elif _same_l1(other) and _related_type_bonus(str(other.get("type", "") or "")) > 0:
                _add_candidate(nid, "same_l1_related_type", 54, rel)
        elif rel in HORIZONTAL_RELATIONS:
            if _strong_distractor_edge(nid):
                _add_candidate(nid, "explicit_horizontal", 108, rel)
            else:
                _add_candidate(nid, "same_l1_related_type", 54, rel)
        elif rel in STRUCTURAL_RELATIONS:
            _add_candidate(nid, "structural_only_edge", 32, rel)

    for ku in all_kus:
        kid = ku.get("ku_id", "")
        if ku.get("type") == anchor_type and _same_owner(ku):
            _add_candidate(kid, "same_owner_same_type", 82)
        if ku.get("type") == anchor_type and _same_l1(ku):
            _add_candidate(kid, "same_l1_same_type", 72)
        elif _same_l1(ku) and _related_type_bonus(str(ku.get("type", "") or "")) > 0:
            _add_candidate(kid, "same_l1_related_type", 54)
        if ku.get("type") == anchor_type:
            _add_candidate(kid, "global_same_type", 44)

    source_rank = {
        "explicit_horizontal": 0,
        "same_owner_same_type": 1,
        "same_l1_same_type": 2,
        "same_l1_related_type": 3,
        "global_same_type": 4,
        "structural_only_edge": 5,
    }
    ranked = sorted(
        scored.values(),
        key=lambda item: (-item["score"], source_rank.get(item["source"], 9), item["kid"]),
    )

    result: list[dict] = []
    for item in ranked[:n]:
        ku = dict(ku_by_id[item["kid"]])
        ku["_distractor_source"] = item["source"]
        ku["_distractor_relation"] = item["relation"]
        ku["_distractor_score"] = item["score"]
        result.append(ku)

    return result


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


# ── Hub detection ──────────────────────────────────────────────────────────

HUB_THRESHOLD = 4


def get_hub_ids(
    graph: dict[str, list[str]],
    threshold: int = HUB_THRESHOLD,
) -> set[str]:
    """Return ku_ids with >= threshold edges. Hubs are excluded from cross-concept MCQ anchors."""
    return {ku_id for ku_id, neighbors in graph.items() if len(neighbors) >= threshold}


# ── Post-build edge cap ────────────────────────────────────────────────────

MAX_EDGES_PER_KU = 4

_EDGE_PRIO = {
    "CONTRASTS_WITH" : 1,
    "ALTERNATIVE_TO" : 1,
    # Pass1 cross/structural edges carry document-level signal, so keep them
    # before auto-generated SIBLING_OF edges when the per-KU cap is reached.
    "EXTENDS"        : 2,
    "APPLIES_TO"     : 2,
    "ENABLES"        : 2,
    "SIMILAR_TO"     : 3,
    "SIBLING_OF"     : 5,
}


def cap_graph_edges(
    graph:      dict[str, list[str]],
    edge_types: dict[tuple[str, str], str],
) -> None:
    """
    Post-build: cap each KU to MAX_EDGES_PER_KU edges, keeping highest-priority
    (horizontal first) edges. Mutates graph and edge_types together so downstream
    MCQ generation cannot use edges that are no longer visible in the graph.
    """
    n_removed = 0
    for ku_id in list(graph.keys()):
        neighbors = list(graph[ku_id])
        if len(neighbors) <= MAX_EDGES_PER_KU:
            continue

        def _prio(nid: str, _kid: str = ku_id) -> int:
            key = (min(_kid, nid), max(_kid, nid))
            return _EDGE_PRIO.get(edge_types.get(key, ""), 9)

        to_keep   = set(sorted(neighbors, key=_prio)[:MAX_EDGES_PER_KU])
        to_remove = [nid for nid in neighbors if nid not in to_keep]

        for nid in to_remove:
            key = (min(ku_id, nid), max(ku_id, nid))
            rel = edge_types.get(key, "?")
            if nid in graph[ku_id]:
                graph[ku_id].remove(nid)
            if ku_id in graph.get(nid, []):
                graph[nid].remove(ku_id)
            edge_types.pop(key, None)
            n_removed += 1
            _log(f"  [graph] cap: remove {rel} {ku_id} ↔ {nid}")
    for a, b in list(edge_types.keys()):
        if b not in graph.get(a, []) or a not in graph.get(b, []):
            edge_types.pop((a, b), None)
            n_removed += 1
            _log(f"  [graph] cap: drop stale edge {a} â†” {b}")

    if n_removed:
        _log(f"  [graph] cap: removed {n_removed} edges (max {MAX_EDGES_PER_KU}/KU)")


# ── Utility ────────────────────────────────────────────────────────────────

def _log(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))
