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

from pipeline.schemas import KUItem, HORIZONTAL_RELATIONS
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

    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"\w+", _normalize(text)))

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

    def _concept_match_score(kid: str, target_name: str) -> int:
        """Lower is better; prefer the exact UI owner before broad parents."""
        if not target_name:
            return 20
        ku = ku_by_id.get(kid, {})
        target = _normalize(target_name)
        if not target:
            return 20

        fields = [
            ("owner_concept", 0),
            ("concept", 1),
            ("local_concept", 2),
            ("parent_l2", 3),
            # parent_l1 is intentionally weaker: it is useful for true L1
            # lookups, but should not beat a specific L2/local match.
            ("parent_l1", 8),
        ]
        exact_scores = [
            base for key, base in fields
            if _normalize(str(ku.get(key, "") or "")) == target
        ]
        if exact_scores:
            return min(exact_scores)

        tt = _tokens(target)
        best = 20
        for key, base in fields:
            value = _normalize(str(ku.get(key, "") or ""))
            vt = _tokens(value)
            if not vt or not tt:
                continue
            if vt <= tt or tt <= vt:
                best = min(best, base + 4)
                continue
            overlap = len(vt & tt)
            if overlap:
                best = min(best, base + 10 - overlap)
        return best

    def _rank_ids(ku_ids: list[str], target_name: str = "", relation: str = "") -> list[str]:
        target_types = RELATION_TO_KU_TYPES.get(relation, [])

        def _type_score(kid: str) -> int:
            ku_type = ku_by_id.get(kid, {}).get("type", "")
            return target_types.index(ku_type) if ku_type in target_types else len(target_types) + 1

        seen: set[str] = set()
        unique = [kid for kid in ku_ids if kid in ku_by_id and not (kid in seen or seen.add(kid))]
        return sorted(
            unique,
            key=lambda kid: (
                _concept_match_score(kid, target_name),
                _type_score(kid),
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

    def _pick_primary(ku_ids: list[str]) -> str | None:
        return min(
            ku_ids,
            key=lambda kid: _PROMINENCE_ORDER.get(
                ku_by_id.get(kid, {}).get("prominence", "peripheral"), 2),
            default=None,
        )

    def _select_endpoints(
        from_ids: list[str],
        to_ids: list[str],
        relation: str,
        from_name: str = "",
        to_name: str = "",
    ) -> list[tuple[str, str]]:
        """Pick representative endpoints instead of a cartesian product."""
        target_types = RELATION_TO_KU_TYPES.get(relation, ["definition"])
        f_cands = [k for k in from_ids if ku_by_id.get(k, {}).get("type") in target_types]
        t_cands = [k for k in to_ids   if ku_by_id.get(k, {}).get("type") in target_types]
        if not f_cands: f_cands = from_ids
        if not t_cands: t_cands = to_ids
        f_ranked = _rank_ids(f_cands, from_name, relation)
        t_ranked = _rank_ids(t_cands, to_name, relation)
        for f in f_ranked[:3]:
            for t in t_ranked[:3]:
                if f != t and not _same_concept(f, t):
                    return [(f, t)]
        return []

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
                pa = _pick_primary(child_ku_groups[i])
                pb = _pick_primary(child_ku_groups[j])
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

    Tier 1a: SIBLING_OF, same type as anchor
    Tier 1b: SIBLING_OF, different type (fallback within siblings)
    Tier 2 : other horizontal (CONTRASTS_WITH, ALTERNATIVE_TO, SIMILAR_TO)
    Tier 3 : same type, non-hub, global fallback
    """
    ku_by_id    = {ku["ku_id"]: ku for ku in all_kus}
    anchor      = ku_by_id.get(anchor_ku_id)
    if not anchor:
        return []

    anchor_type    = anchor.get("type")
    anchor_concept = _normalize(anchor.get("concept", ""))
    hub_ids        = get_hub_ids(graph)

    def _rel(a: str, b: str) -> str:
        return edge_types.get((min(a, b), max(a, b)), "")

    neighbors  = graph.get(anchor_ku_id, [])
    candidates = [nid for nid in neighbors if nid in ku_by_id and nid not in hub_ids]

    tier1a = [nid for nid in candidates
              if _rel(anchor_ku_id, nid) == "SIBLING_OF"
              and ku_by_id[nid].get("type") == anchor_type]
    tier1b = [nid for nid in candidates
              if _rel(anchor_ku_id, nid) == "SIBLING_OF"
              and ku_by_id[nid].get("type") != anchor_type]
    tier2  = [nid for nid in candidates
              if _rel(anchor_ku_id, nid) in HORIZONTAL_RELATIONS
              and _rel(anchor_ku_id, nid) != "SIBLING_OF"]

    pool_ids = tier1a + tier1b + tier2
    seen     = {anchor_ku_id} | set(pool_ids)

    if len(pool_ids) < n:
        for ku in all_kus:
            if len(pool_ids) >= n:
                break
            kid = ku["ku_id"]
            if (kid not in seen and kid not in hub_ids
                    and ku.get("type") == anchor_type
                    and _normalize(ku.get("concept", "")) != anchor_concept):
                pool_ids.append(kid)
                seen.add(kid)

    return [ku_by_id[nid] for nid in pool_ids[:n] if nid in ku_by_id]


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
    "SIMILAR_TO"     : 2,
    "SIBLING_OF"     : 3,
    "EXTENDS"        : 4,
    "APPLIES_TO"     : 4,
    "ENABLES"        : 5,
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
