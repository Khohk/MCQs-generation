"""
gradio_app.py
-------------
MCQ Generator — Gradio web app
Deploy: Hugging Face Spaces (SDK: gradio)

Chạy local:
  pip install gradio
  python gradio_app.py

Deploy HF Spaces:
  Tạo Space → SDK: Gradio → push code lên
  Thêm GEMINI_API_KEY vào Settings → Secrets
"""

import os
import sys
import json
import html
import re
import tempfile
import time
from pathlib import Path

import gradio as gr

# ── i18n ──────────────────────────────────────────────────────────
I18N_DIR = Path(__file__).parent / "i18n"

def load_lang(lang: str) -> dict:
    path = I18N_DIR / f"{lang}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)

LANGS = {"vi": load_lang("vi"), "en": load_lang("en")}

def t(lang: str, *keys) -> str:
    """Lấy string theo key path, ví dụ t('vi', 'upload', 'tab')"""
    obj = LANGS[lang]
    for k in keys:
        obj = obj.get(k, k)
    return obj


DIFFICULTY_MAP = {"Dễ": "easy", "Easy": "easy",
                  "Trung bình": "medium", "Medium": "medium",
                  "Khó": "hard", "Hard": "hard"}

LANG_MAP = {"Tiếng Việt": "vi", "Vietnamese": "vi",
            "English": "en", "EN": "en"}

_BLOOM_TO_DIFF: dict[str, str] = {
    "remember"  : "easy",
    "understand": "easy",
    "apply"     : "medium",
    "analyze"   : "medium",
    "evaluate"  : "hard",
    "create"    : "hard",
}


def _make_llm_fn(temperature: float = 0.2, max_tokens: int = 4096):
    """Return a llm_fn(prompt, json_mode[, temperature_override]) backed by providers."""
    import pipeline.generator as gen

    def _llm_fn(prompt: str, json_mode: bool, temperature_override: float | None = None) -> str:
        call_temperature = temperature if temperature_override is None else temperature_override
        attempts = len(gen.PROVIDERS) * gen.RETRY_LIMIT
        for _ in range(attempts):
            idx = gen._next_available_provider()
            if idx is None:
                time.sleep(5)
                continue
            gen._current_provider_idx = idx
            try:
                return gen._call_provider(
                    prompt,
                    idx,
                    json_mode,
                    max_tokens=max_tokens,
                    temperature=call_temperature,
                )
            except Exception as e:
                err = str(e)
                if any(k in err for k in ("404", "NOT_FOUND", "model_not_found")) \
                        or "not found in .env" in err:
                    gen._mark_disabled(idx, "404/nokey")
                elif any(k in err for k in ("429", "RESOURCE_EXHAUSTED", "rate_limit")):
                    gen._mark_cooldown(idx)
                else:
                    time.sleep(3)
                time.sleep(gen.SWITCH_DELAY)
        raise RuntimeError("All providers exhausted")

    return _llm_fn


def _mcq_to_dict(m) -> dict:
    """Convert MCQItem to the flat dict format expected by the UI."""
    d = {
        "question"    : m.question,
        "A"           : m.options.get("A", ""),
        "B"           : m.options.get("B", ""),
        "C"           : m.options.get("C", ""),
        "D"           : m.options.get("D", ""),
        "answer"      : m.answer,
        "bloom_level" : m.bloom_level,
        "explanation" : _clean_explanation_text(m.explanation),
        "difficulty"  : _BLOOM_TO_DIFF.get(m.bloom_level, "medium"),
        "priority"    : m.priority,
        "source_pages": m.source_pages,
    }
    if m.is_cross:
        d["is_cross"]      = True
        d["edge_relation"] = m.edge_relation
    return d


# ── Backend helpers ───────────────────────────────────────────────

def run_pipeline(files, difficulty_label, q_lang_label, lang,
                 on_step=None):
    """
    KU-based pipeline: Parse → Pass 1 → Pass 2 → MCQ generation.
    Returns (mcqs_json, error_msg).
    on_step(pct, msg): progress callback.
    """
    def step(pct, msg):
        if on_step:
            on_step(pct, msg)

    _no_data = (None, None, None, None)

    if not files:
        return (None, t(lang, "upload", "no_file"), None, None)

    difficulty = DIFFICULTY_MAP.get(difficulty_label, "medium")
    question_lang = LANG_MAP.get(q_lang_label)
    if question_lang not in {"vi", "en"}:
        q_label = str(q_lang_label or "")
        question_lang = "vi" if ("Vi" in q_label or "Ti" in q_label) else "en"

    # ── Step 1: Parse ──────────────────────────────────────────────
    step(0.05, "Dang doc tai lieu...")
    try:
        from pipeline.file_router import parse_file
        all_pages = []
        for f in files:
            all_pages.extend(parse_file(f.name))
    except Exception as e:
        return (None, f"Loi parse file: {e}", None, None)

    _log_parse(all_pages, files)

    # ── Step 2: Pass 1 — segmentation + concept hierarchy ─────────
    step(0.15, f"Dang phan tich cau truc ({len(all_pages)} trang)...")
    try:
        import pipeline.generator as _gen
        from pipeline.generator import reset_provider_stats
        from pipeline.pass1_extractor import run_pass1
        reset_provider_stats()
        pass1_llm = _make_llm_fn(temperature=0.1, max_tokens=8192)
        pass1  = run_pass1(all_pages, pass1_llm)
    except Exception as e:
        return (None, f"Loi Pass 1: {e}", None, None)

    if not pass1.ok:
        return (None, "Khong phan tich duoc cau truc tai lieu.", None, None)

    # ── Step 3: Pass 2 — KU extraction ────────────────────────────
    step(0.35, f"Dang trich xuat kien thuc ({len(pass1.segments)} doan)...")
    try:
        from pipeline.pass2_extractor import run_pass2
        pass2_llm = _make_llm_fn(temperature=0.15, max_tokens=4096)
        pass2 = run_pass2(pass1, pass2_llm, delay_between=2.0)
    except Exception as e:
        return (None, f"Loi Pass 2: {e}", None, None)

    if not pass2.ok:
        return (None, "Khong trich xuat duoc kien thuc.", None, None)

    # ── Step 4: Single-KU MCQs ─────────────────────────────────────
    step(0.55, f"Dang tao cau hoi ({len(pass2.all_kus)} KUs)...")
    try:
        from pipeline.mcq_generator import run_mcq_generation, run_cross_mcq_generation
        mcq_llm = _make_llm_fn(temperature=0.35, max_tokens=4096)
        mcqs = run_mcq_generation(pass2, mcq_llm, delay_between=2.0, language=question_lang)
    except Exception as e:
        return (None, f"Loi MCQ generation: {e}", None, None)

    # ── Step 5: Cross-concept MCQs (medium / hard only) ───────────
    cross_mcqs = []
    if difficulty in {"medium", "hard"}:
        step(0.8, "Dang tao cau hoi cross-concept...")
        try:
            cross_mcqs = run_cross_mcq_generation(
                pass2,
                mcq_llm,
                delay_between=2.0,
                language=question_lang,
                horizontal_only=(difficulty != "hard"),
                max_questions=6 if difficulty == "medium" else 8,
                answer_offset=len(mcqs),
            )
        except Exception as e:
            print(f"[Gradio] cross MCQ warning: {e}")

    # ── Filter by difficulty ───────────────────────────────────────
    all_mcqs_objs = mcqs + cross_mcqs
    if difficulty == "easy":
        all_mcqs_objs = [m for m in all_mcqs_objs if m.priority <= 2]
    elif difficulty == "medium":
        all_mcqs_objs = [m for m in all_mcqs_objs if m.priority <= 5]

    result = [_mcq_to_dict(m) for m in all_mcqs_objs if m.ok]

    # ── Graph data ─────────────────────────────────────────────────
    graph_data = None
    try:
        graph_data = build_graph_data(pass2, all_mcqs_objs, pass1)
    except Exception as e:
        print(f"[Gradio] graph build warning: {e}")

    pdf_name = Path(files[0].name).name if files else "upload"
    try:
        _log_provider_stats(_gen.get_provider_stats(), pdf_name)
    except Exception:
        pass

    step(1.0, f"Hoan tat! {len(result)} cau hoi san sang.")
    return (json.dumps(result, ensure_ascii=False), None, graph_data, all_mcqs_objs)


def _log_provider_stats(stats: dict, pdf_name: str):
    import json, datetime
    log_dir = Path("data/provider_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "provider_usage.jsonl"
    entry = {
        "ts"            : datetime.datetime.now().isoformat(),
        "file"          : pdf_name,
        "providers"     : stats.get("providers", []),
        "chunks_skipped": stats.get("chunks_skipped", []),
        "chunk_logs"    : stats.get("chunk_logs", []),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _log_document_analysis(analysis: dict, files) -> None:
    import json, datetime
    log_dir = Path("data/dev_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "document_analysis.jsonl"
    entry = {
        "ts": datetime.datetime.now().isoformat(),
        "files": [Path(x.name).name for x in files],
        "analysis": analysis,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _log_parse(pages: list, files) -> None:
    import datetime
    log_dir = Path("data/dev_logs"); log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "parse_chunks.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] PARSE\n")
        f.write(f"Files: {[Path(x.name).name for x in files]}\n")
        f.write(f"Total pages kept: {len(pages)}\n")
        f.write(f"{'─'*60}\n")
        f.write(f"{'PAGE':<6} {'TITLE':<42} {'CHARS'}\n")
        for p in pages:
            f.write(f"{p['page_num']:<6} {p['title'][:40]:<42} {p['char_count']}\n")


def _log_chunks(chunks: list, strategy: str = "auto") -> None:
    import datetime
    log_dir = Path("data/dev_logs"); log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "parse_chunks.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] CHUNKS → {len(chunks)} total | strategy={strategy}\n")
        f.write(f"{'─'*60}\n")
        f.write(f"{'ID':<12} {'TOPIC':<35} {'PAGES':<16} {'CHARS'}\n")
        for c in chunks:
            pages_str = str(c['pages']) if len(c['pages']) <= 4 else f"[{c['pages'][0]}..{c['pages'][-1]}]({len(c['pages'])}p)"
            f.write(f"{c['chunk_id']:<12} {c['topic'][:33]:<35} {pages_str:<16} {len(c['text'])}\n")
        avg = sum(len(c['text']) for c in chunks) / len(chunks) if chunks else 0
        f.write(f"{'─'*60}\n")
        f.write(f"Avg chars/chunk: {avg:.0f}\n")
        f.write(f"{'='*60}\n")


def _log_rejected(rejected: list, pdf_name: str):
    import json, datetime
    log_dir = Path("data/validation_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"rejected_{pdf_name}.jsonl"
    timestamp = datetime.datetime.now().isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps({"ts": timestamp, "reason": r["reason"],
                                "question": r["mcq"].get("question", "")[:80]},
                               ensure_ascii=False) + "\n")


# ── KU Graph helpers ──────────────────────────────────────────────

_TYPE_COLORS: dict[str, str] = {
    "definition"  : "#3b82f6",
    "mechanism"   : "#22c55e",
    "procedure"   : "#f97316",
    "trade_off"   : "#a855f7",
    "failure_mode": "#ef4444",
    "application" : "#14b8a6",
}
_TYPE_ICONS: dict[str, str] = {
    "definition"  : "📖",
    "mechanism"   : "⚙️",
    "procedure"   : "📋",
    "trade_off"   : "⚖️",
    "failure_mode": "⚠️",
    "application" : "🎯",
}
_TYPE_ORDER = ["definition", "mechanism", "procedure", "trade_off", "failure_mode", "application"]
_EDGE_STYLE: dict[str, dict] = {
    "CONTRASTS_WITH" : {"dash": "6,3",  "color": "#f97316", "label": "vs"},
    "ALTERNATIVE_TO" : {"dash": "6,3",  "color": "#a855f7", "label": "or"},
    "SIMILAR_TO"     : {"dash": "4,4",  "color": "#9ca3af", "label": "≈"},
    "SIBLING_OF"     : {"dash": "none", "color": "#4b5563", "label": ""},
    "ENABLES"        : {"dash": "none", "color": "#4b5563", "label": "→"},
    "APPLIES_TO"     : {"dash": "none", "color": "#4b5563", "label": "⊂"},
    "EXTENDS"        : {"dash": "none", "color": "#4b5563", "label": "↑"},
}


def build_graph_data(pass2_result, mcqs_all=None, pass1_result=None) -> dict:
    """Concept-level mind-map data with KU payload for the detail panel."""
    hierarchy = getattr(pass1_result, "raw_hierarchy", None) or getattr(pass1_result, "concept_hierarchy", []) or []
    relationships = getattr(pass1_result, "relationships", []) or []
    segments = getattr(pass1_result, "segments", []) or []
    ku_graph = {k: list(v) for k, v in (getattr(pass2_result, "graph", {}) or {}).items()}
    ku_edge_types = [
        {"source": a, "target": b, "relation": rel}
        for (a, b), rel in (getattr(pass2_result, "edge_types", {}) or {}).items()
        if a < b
    ]

    concept_nodes: dict[str, dict] = {}
    concept_order: list[str] = []
    hierarchy_edges: list[dict] = []
    seen_hierarchy_edges: set[tuple[str, str]] = set()
    child_to_parent: dict[str, str] = {}
    seg_lookup: dict[str, dict] = {}

    def _ensure_concept(name: str, level: str = "L2", parent: str | None = None) -> dict:
        name = str(name or "").strip()
        if not name:
            return {}
        node = concept_nodes.get(name)
        if not node:
            node = {
                "id": name,
                "label": name,
                "level": level,
                "parent": parent,
                "ku_ids": [],
                "kus": [],
                "ku_count": 0,
            }
            concept_nodes[name] = node
            concept_order.append(name)
        else:
            if level == "L1":
                node["level"] = "L1"
            if parent and not node.get("parent"):
                node["parent"] = parent
        return node

    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").lower().strip().replace("-", " "))

    def _token_set(text: str) -> set[str]:
        return set(re.findall(r"\w+", _norm(text)))

    def _best_match(name: str, candidates: list[str]) -> str | None:
        norm = _norm(name)
        if not norm or not candidates:
            return None
        query_tokens = _token_set(norm)
        best_node = None
        best_score = 0
        for candidate in candidates:
            node_tokens = _token_set(candidate)
            if not node_tokens:
                continue
            if norm == _norm(candidate):
                return candidate
            if node_tokens <= query_tokens or query_tokens <= node_tokens:
                score = len(node_tokens & query_tokens)
                if score > best_score:
                    best_score = score
                    best_node = candidate
        return best_node

    def _base_seg_id(ku_id: str) -> str:
        head = str(ku_id or "").split("_ku_", 1)[0]
        return re.sub(r"[a-z]+$", "", head)

    # Build concept hierarchy from Pass 1.
    for l1 in hierarchy:
        if not isinstance(l1, dict):
            continue
        l1_name = l1.get("name", "") or l1.get("parent", "")
        if not l1_name:
            continue
        _ensure_concept(l1_name, "L1", None)
        for child in l1.get("children", []) or []:
            child_name = str(child.get("name", "") or child.get("concept_id", "") or "").strip() if isinstance(child, dict) else str(child or "").strip()
            if not child_name:
                continue
            _ensure_concept(child_name, "L2", l1_name)
            child_to_parent[child_name] = l1_name
            key = (l1_name, child_name)
            if key not in seen_hierarchy_edges:
                seen_hierarchy_edges.add(key)
                hierarchy_edges.append({
                    "source": l1_name,
                    "target": child_name,
                    "relation": "CONTAINS",
                    "edge_type": "hierarchy",
                    "has_cross_mcq": False,
                })

    # Build a segment -> candidate concept mapping from Pass 1's actual segmentation.
    for seg in segments:
        seg_id = str(getattr(seg, "segment_id", "") or "").strip()
        if not seg_id:
            continue
        label = str(getattr(seg, "label", "") or "").strip()
        parent_name = str(getattr(seg, "parent_concept_name", "") or "").strip()
        seg_concepts = [str(c or "").strip() for c in (getattr(seg, "concepts", []) or []) if str(c or "").strip()]

        candidates: list[str] = []
        if parent_name:
            candidates = [c for c in seg_concepts if child_to_parent.get(c) == parent_name]
            if label and label in child_to_parent and label not in candidates:
                candidates.insert(0, label)
        else:
            if label in concept_nodes:
                candidates = [label]
            else:
                candidates = [c for c in seg_concepts if c in concept_nodes]

        seg_lookup[seg_id] = {
            "label": label,
            "parent": parent_name,
            "candidates": candidates,
        }

    def _resolve_concept_node(name: str, segment_id: str = "") -> str | None:
        """Resolve Pass1 relationship names to graph node ids with graceful fallback."""
        raw_name = str(name or "").strip()
        if raw_name in concept_nodes:
            return raw_name

        # If Pass1 remapped the relationship to a segment, trust that segment first.
        seg_info = seg_lookup.get(str(segment_id or "").strip(), {}) if segment_id else {}
        seg_candidates = list(seg_info.get("candidates", []) or [])
        node_id = _best_match(raw_name, seg_candidates) if raw_name else None
        if node_id:
            return node_id
        if len(seg_candidates) == 1:
            return seg_candidates[0]

        label = str(seg_info.get("label", "") or "").strip()
        if label in concept_nodes:
            return label

        # Fuzzy fallback covers small naming drift like "Simple Recurrent Net"
        # vs "Simple Recurrent Nets (Elman nets)".
        node_id = _best_match(raw_name, list(concept_nodes.keys()))
        if node_id:
            return node_id

        parent = str(seg_info.get("parent", "") or "").strip()
        if parent in concept_nodes:
            return parent
        return None

    # Group KUs by concept, but keep the full KU payload for the detail pane.
    ku_index: dict[str, dict] = {}
    node_ku_map: dict[str, list[dict]] = {}
    for ku in pass2_result.all_kus:
        ku_copy = dict(ku)
        ku_id = str(ku_copy.get("ku_id", "")).strip()
        concept = str(ku_copy.get("concept", "unknown")).strip() or "unknown"
        owner_concept = str(ku_copy.get("owner_concept", "") or "").strip()
        ku_copy.setdefault("ku_id", ku_id)
        ku_copy.setdefault("concept", concept)
        ku_index[ku_id] = ku_copy

        seg_id = str(ku_id).split("_ku_", 1)[0]
        seg_info = seg_lookup.get(seg_id) or seg_lookup.get(_base_seg_id(seg_id), {})
        seg_candidates = list(seg_info.get("candidates", []) or [])
        node_id = _best_match(owner_concept, list(concept_nodes.keys())) if owner_concept else None
        if not node_id and owner_concept:
            node_id = _best_match(owner_concept, seg_candidates)
        if not node_id:
            node_id = _best_match(concept, seg_candidates)

        if not node_id and seg_info.get("label") in concept_nodes:
            node_id = seg_info.get("label")
        if not node_id and seg_info.get("parent"):
            node_id = seg_info.get("parent")
        if not node_id:
            node_id = _best_match(concept, list(concept_nodes.keys()))

        if node_id and node_id in concept_nodes:
            concept_nodes[node_id]["ku_ids"].append(ku_id)
            ku_copy["ui_node_id"] = node_id
            node_ku_map.setdefault(node_id, []).append(ku_copy)

    for concept, node in concept_nodes.items():
        kus = node_ku_map.get(concept, [])
        node["kus"] = [
            {
                "ku_id": ku.get("ku_id", ""),
                "type": ku.get("type", ""),
                "concept": ku.get("concept", concept),
                "local_concept": ku.get("local_concept", ku.get("concept", concept)),
                "owner_level": ku.get("owner_level", ""),
                "owner_concept": ku.get("owner_concept", concept),
                "parent_l1": ku.get("parent_l1", ""),
                "parent_l2": ku.get("parent_l2", ""),
                "content": ku.get("content", ""),
                "verbatim_evidence": ku.get("verbatim_evidence", ""),
                "prominence": ku.get("prominence", ""),
                "source_pages": ku.get("source_pages", []),
                "ui_node_id": ku.get("ui_node_id", concept),
            }
            for ku in kus
        ]
        node["ku_count"] = len(kus)

    nodes = [concept_nodes[name] for name in concept_order]

    edges: list[dict] = []
    seen_edges: set[tuple[str, str]] = set()
    edges.extend(hierarchy_edges)
    seen_edges.update((e["source"], e["target"]) for e in hierarchy_edges)

    # Cross edges are kept at concept level only, to reduce clutter.
    # Use the same forgiving name resolution as the backend graph builder so
    # relationship edges do not disappear due to minor Pass1 naming drift.
    for rel in relationships:
        src = _resolve_concept_node(rel.get("from_concept", ""), rel.get("from_segment", ""))
        tgt = _resolve_concept_node(rel.get("to_concept", ""), rel.get("to_segment", ""))
        relation = rel.get("relation", "RELATED_TO")
        if not src or not tgt or src == tgt or src not in concept_nodes or tgt not in concept_nodes:
            continue
        key = (min(src, tgt), max(src, tgt))
        if key in seen_edges:
            # Keep the hierarchy edge but make it visibly meaningful when the
            # same pair also has a relationship, e.g. RNN -> LSTM EXTENDS.
            for e in edges:
                if (min(e["source"], e["target"]), max(e["source"], e["target"])) == key:
                    e["relation"] = relation
                    e["edge_type"] = "cross"
                    e["has_cross_mcq"] = False
                    break
            continue
        seen_edges.add(key)
        edges.append({
            "source": src,
            "target": tgt,
            "relation": relation,
            "edge_type": "cross",
            "has_cross_mcq": False,
        })

    if mcqs_all:
        cross_pairs: set[tuple[str, str]] = set()
        for m in mcqs_all:
            if getattr(m, "anchor_ku_id_b", ""):
                ku_a = ku_index.get(m.anchor_ku_id, {})
                ku_b = ku_index.get(m.anchor_ku_id_b, {})
                ca = ku_a.get("concept", "")
                cb = ku_b.get("concept", "")
                if ca and cb:
                    cross_pairs.add((min(ca, cb), max(ca, cb)))
        for e in edges:
            e["has_cross_mcq"] = (min(e["source"], e["target"]), max(e["source"], e["target"])) in cross_pairs

    ku_edges: list[dict] = []
    seen_ku_edges: set[tuple[str, str]] = set()
    for (src, tgt), relation in pass2_result.edge_types.items():
        key = (min(src, tgt), max(src, tgt))
        if key in seen_ku_edges:
            continue
        seen_ku_edges.add(key)
        ku_edges.append({"source": src, "target": tgt, "relation": relation})

    return {
        "nodes": nodes,
        "edges": edges,
        "kus": list(ku_index.values()),
        "ku_graph": ku_graph,
        "ku_edge_types": ku_edge_types,
        "ku_edges": ku_edge_types,
        "concepts": concept_nodes,
    }


_GRAPH_BLOCKS_JS = """
() => {
  // Concept-level mind-map: L1 (large) + L2 (small), hierarchy + cross edges.
  // window.initKUGraph defined synchronously; D3 loads lazily when first called.

  const _PALETTE = ["#60a5fa","#34d399","#fb923c","#a78bfa","#f472b6",
                    "#facc15","#38bdf8","#4ade80","#f87171","#818cf8"];
  const _ES = {
    "CONTRASTS_WITH": {"color":"#f97316","label":"vs"},
    "ALTERNATIVE_TO": {"color":"#a78bfa","label":"or"},
    "SIMILAR_TO":     {"color":"#9ca3af","label":"≈"},
    "ENABLES":        {"color":"#34d399","label":"→"},
    "APPLIES_TO":     {"color":"#60a5fa","label":"⊂"},
    "EXTENDS":        {"color":"#38bdf8","label":"↑"},
    "RELATED_TO":     {"color":"#6b7280","label":"~"}
  };

  function _tip(ev, html) {
    const tip = document.getElementById('ku-tip');
    if (!tip) return;
    const root = document.getElementById('ku-graph-root');
    const rect = root ? root.getBoundingClientRect() : {left:0,top:0};
    tip.innerHTML = html; tip.style.display = 'block';
    tip.style.left = (ev.clientX - rect.left + 14) + 'px';
    tip.style.top  = (ev.clientY - rect.top  -  8) + 'px';
  }
  function _hideTip() {
    const t = document.getElementById('ku-tip');
    if (t) t.style.display = 'none';
  }
  function _showError(msg) {
    const root = document.getElementById('ku-graph-root');
    if (!root) return;
    const fb = root.querySelector('#ku-fb');
    if (fb) {
      fb.innerHTML = '<div style="padding:12px;color:#fca5a5;font-size:13px">' + msg + '</div>';
    }
  }
  function _notify(data) {
    const el = document.getElementById('graph_click_data');
    if (!el) return;
    const field = el.querySelector('textarea, input');
    if (!field) return;
    const proto = field.tagName === 'TEXTAREA'
      ? HTMLTextAreaElement.prototype
      : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
    setter.call(field, JSON.stringify(data));
    field.dispatchEvent(new Event('input',  {bubbles:true}));
    field.dispatchEvent(new Event('change', {bubbles:true}));
  }

  function _render(G) {
    const d3   = window.d3;
    const root = document.getElementById('ku-graph-root');
    if (!root) return;
    if (root._sim) { root._sim.stop(); root._sim = null; }
    root.querySelector('#ku-nodes').innerHTML = '';
    root.querySelector('#ku-links').innerHTML = '';

    // ── Color palette: one color per L1, L2 inherits parent ──────────────
    const colorMap = {};
    let pIdx = 0;
    G.nodes.filter(n => n.level === 'L1').forEach(n => {
      colorMap[n.id] = _PALETTE[pIdx++ % _PALETTE.length];
    });
    G.nodes.filter(n => n.level === 'L2').forEach(n => {
      colorMap[n.id] = colorMap[n.parent] || '#94a3b8';
    });

    // ── Pre-seed positions: L1 on circle, L2 near parent ─────────────────
    const W = root.offsetWidth || 800, H = 540;
    const cx = W/2, cy = H/2;
    const nodeMap = {};
    G.nodes.forEach(n => nodeMap[n.id] = n);
    const l1s = G.nodes.filter(n => n.level === 'L1');
    const R1  = Math.min(W, H) * 0.28;
    l1s.forEach((n, i) => {
      const a = (2*Math.PI*i / Math.max(l1s.length,1)) - Math.PI/2;
      n.x = cx + R1*Math.cos(a); n.y = cy + R1*Math.sin(a);
    });
    G.nodes.filter(n => n.level === 'L2').forEach(n => {
      const par = nodeMap[n.parent];
      const a   = Math.random()*2*Math.PI;
      const r   = 65 + Math.random()*25;
      n.x = (par ? par.x : cx) + r*Math.cos(a);
      n.y = (par ? par.y : cy) + r*Math.sin(a);
    });

    // ── SVG + zoom ────────────────────────────────────────────────────────
    const svg = d3.select(root.querySelector('#ku-svg'));
    const g   = svg.select('#ku-zoom');
    svg.call(d3.zoom().scaleExtent([.1,4]).on('zoom', e => g.attr('transform', e.transform)));

    // ── Legend ────────────────────────────────────────────────────────────
    const fb = root.querySelector('#ku-fb');
    fb.innerHTML = '';
    l1s.forEach(n => {
      const b = document.createElement('span');
      b.style.cssText = 'padding:2px 10px;border-radius:10px;font-size:11px;cursor:pointer;border:1.5px solid '
                        +colorMap[n.id]+';color:'+colorMap[n.id];
      b.textContent = n.label;
      b.title = n.ku_count + ' KUs';
      fb.appendChild(b);
    });

    // ── Simulation ────────────────────────────────────────────────────────
    const sim = d3.forceSimulation(G.nodes)
      .force('link',    d3.forceLink(G.edges).id(d => d.id)
               .distance(e => e.edge_type==='hierarchy' ? 80 : 160)
               .strength(e => e.edge_type==='hierarchy' ? 0.9 : 0.25))
      .force('charge',  d3.forceManyBody().strength(d => d.level==='L1' ? -400 : -160))
      .force('center',  d3.forceCenter(cx, cy).strength(0.04))
      .force('collide', d3.forceCollide(d => d.level==='L1' ? 42 : 28));
    root._sim = sim;

    // ── Edges ─────────────────────────────────────────────────────────────
    const lSel = g.select('#ku-links').selectAll('line').data(G.edges).join('line')
      .attr('stroke', d => d.edge_type==='hierarchy' ? '#374151' : (_ES[d.relation]||{}).color||'#4b5563')
      .attr('stroke-dasharray', d => d.edge_type==='hierarchy' ? '4,3' : null)
      .attr('stroke-width', d => d.has_cross_mcq ? 2.5 : (d.edge_type==='hierarchy' ? 1 : 1.8))
      .attr('opacity',   d => d.edge_type==='hierarchy' ? 0.35 : 0.75)
      .style('cursor',   d => d.edge_type!=='hierarchy' ? 'pointer' : 'default')
      .on('click', (ev, d) => {
        if (d.edge_type==='hierarchy') return;
        ev.stopPropagation();
        _notify({type:'edge', source:d.source.id||d.source, target:d.target.id||d.target, relation:d.relation});
      })
      .on('mouseover', (ev, d) => {
        if (d.edge_type==='hierarchy') return;
        _tip(ev, '<b>'+d.relation+'</b>'+(d.has_cross_mcq?' <span style="color:#34d399">✓ MCQ</span>':''));
      })
      .on('mouseout', _hideTip);

    // ── Nodes ─────────────────────────────────────────────────────────────
    const drag = d3.drag()
      .on('start', (ev,d) => { if(!ev.active) sim.alphaTarget(.3).restart(); d.fx=d.x; d.fy=d.y; })
      .on('drag',  (ev,d) => { d.fx=ev.x; d.fy=ev.y; })
      .on('end',   (ev,d) => { if(!ev.active) sim.alphaTarget(0); d.fx=null; d.fy=null; });

    const nSel = g.select('#ku-nodes').selectAll('g').data(G.nodes).join('g')
      .style('cursor','pointer').call(drag)
      .on('click', (ev, d) => {
        ev.stopPropagation();
        g.selectAll('#ku-nodes circle').attr('stroke-width', 0);
        d3.select(ev.currentTarget).select('circle').attr('stroke','#fff').attr('stroke-width', 2.5);
        _notify({type:'concept', id: d.id});
      })
      .on('mouseover', (ev, d) => _tip(ev,
          '<b style="color:'+colorMap[d.id]+'">'+d.label+'</b>'
          +' <span style="color:#9ca3af;font-size:10px">'+d.level+'</span><br>'
          +'<span style="color:#6b7280">'+d.ku_count+' KUs</span>'))
      .on('mouseout', _hideTip);

    nSel.append('circle')
      .attr('r',            d => d.level==='L1' ? 20 : 12)
      .attr('fill',         d => colorMap[d.id] || '#94a3b8')
      .attr('fill-opacity', d => d.level==='L1' ? 0.9 : 0.65)
      .attr('stroke-width', 0);

    // KU count badge inside L1 nodes
    nSel.filter(d => d.level==='L1')
      .append('text')
      .attr('text-anchor','middle').attr('dy','0.35em')
      .attr('font-size',10).attr('font-weight','700').attr('fill','#fff')
      .attr('pointer-events','none')
      .text(d => d.ku_count || '');

    nSel.append('text').attr('class','node-lbl')
      .attr('dy', d => (d.level==='L1' ? 20 : 12) + 14)
      .attr('text-anchor','middle')
      .attr('font-size', d => d.level==='L1' ? 12 : 10)
      .attr('font-weight', d => d.level==='L1' ? '600' : '400')
      .text(d => d.label.length > 22 ? d.label.slice(0,21)+'…' : d.label);

    sim.on('tick', () => {
      lSel.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)
          .attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
      nSel.attr('transform', d => `translate(${d.x},${d.y})`);
    });
  }

  window.initKUGraph = async function(G) {
    if (!G || !G.nodes || !G.nodes.length) return;
    if (!window.d3) {
      await new Promise(resolve => {
        const s = document.createElement('script');
        s.src = 'https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js';
        s.onload = resolve; s.onerror = resolve;
        document.head.appendChild(s);
      });
    }
    if (!window.d3) {
      console.error('[KUGraph] D3 failed to load');
      _showError('KU Graph could not load D3.js. Check network access to cdnjs.cloudflare.com.');
      return;
    }
    requestAnimationFrame(() => _render(G));
  };

  document.addEventListener('click', (ev) => {
    const kuBtn = ev.target.closest && ev.target.closest('[data-ku-id]');
    if (!kuBtn) return;
    ev.preventDefault();
    _notify({
      type: 'ku',
      id: kuBtn.getAttribute('data-ku-id'),
      concept: kuBtn.getAttribute('data-concept') || ''
    });
  });

  new MutationObserver(() => {
    const root = document.getElementById('ku-graph-root');
    if (!root) return;
    const raw = root.getAttribute('data-graph');
    if (!raw || raw === root._lastGraph) return;
    root._lastGraph = raw;
    try {
      const G = JSON.parse(raw);
      if (G && G.nodes && G.nodes.length) window.initKUGraph(G);
    } catch(e) { console.error('[KUGraph] parse error', e); }
  }).observe(document.body, {childList:true, subtree:true, attributes:true, attributeFilter:['data-graph']});
}
"""


_GRAPH_BLOCKS_JS_V2 = """
(() => {
  const _PALETTE = ["#60a5fa", "#34d399", "#fb923c", "#a78bfa", "#f472b6",
                    "#facc15", "#38bdf8", "#4ade80", "#f87171", "#818cf8"];
  const _REL = {
    "CONTRASTS_WITH": "#f97316",
    "ALTERNATIVE_TO": "#a855f7",
    "SIMILAR_TO": "#9ca3af",
    "ENABLES": "#34d399",
    "APPLIES_TO": "#60a5fa",
    "EXTENDS": "#38bdf8",
    "RELATED_TO": "#6b7280",
  };

  function _escapeHtml(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function _notify(data) {
    const el = document.getElementById("graph_click_data");
    if (!el) return;
    const field = el.querySelector("textarea, input");
    if (!field) return;
    const proto = field.tagName === "TEXTAREA"
      ? HTMLTextAreaElement.prototype
      : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, "value").set;
    setter.call(field, JSON.stringify(data));
    field.dispatchEvent(new Event("input", { bubbles: true }));
    field.dispatchEvent(new Event("change", { bubbles: true }));

    const triggerWrap = document.getElementById("graph_click_trigger");
    const triggerBtn = triggerWrap && triggerWrap.querySelector("button");
    if (triggerBtn) {
      setTimeout(() => triggerBtn.click(), 25);
    }
  }

  function _tip(ev, html) {
    const tip = document.getElementById("ku-tip");
    const root = document.getElementById("ku-graph-root");
    if (!tip || !root) return;
    const rect = root.getBoundingClientRect();
    tip.innerHTML = html;
    tip.style.display = "block";
    tip.style.left = (ev.clientX - rect.left + 14) + "px";
    tip.style.top = (ev.clientY - rect.top - 8) + "px";
  }

  function _hideTip() {
    const tip = document.getElementById("ku-tip");
    if (tip) tip.style.display = "none";
  }

  function _showError(msg) {
    const root = document.getElementById("ku-graph-root");
    if (!root) return;
    const fb = root.querySelector("#ku-fb");
    if (fb) {
      fb.innerHTML = '<div style="padding:12px;color:#fca5a5;font-size:13px">' + msg + "</div>";
    }
  }

  function _setSvgSize(svg, w, h) {
    svg.setAttribute("viewBox", "0 0 " + w + " " + h);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  }

  function _setView(root, x, y, w, h) {
    const svg = root && root.querySelector("#ku-svg");
    if (!svg) return;
    root._kuView = { x, y, w, h };
    svg.setAttribute("viewBox", [x, y, w, h].join(" "));
  }

  function _fitView(root, G) {
    const svg = root && root.querySelector("#ku-svg");
    if (!svg || !G || !G.nodes || !G.nodes.length) return;
    const xs = G.nodes.map(n => Number(n.x || 0));
    const ys = G.nodes.map(n => Number(n.y || 0));
    const pad = 110;
    const minX = Math.min(...xs) - pad;
    const maxX = Math.max(...xs) + pad;
    const minY = Math.min(...ys) - pad;
    const maxY = Math.max(...ys) + pad;
    const baseW = Number(root._kuBaseW || svg.clientWidth || 900);
    const baseH = Number(root._kuBaseH || svg.clientHeight || 620);
    let vx = minX;
    let vy = minY;
    let vw = Math.max(260, maxX - minX);
    let vh = Math.max(220, maxY - minY);
    const graphRatio = vw / vh;
    const screenRatio = baseW / baseH;
    if (graphRatio > screenRatio) {
      const nextH = vw / screenRatio;
      vy -= (nextH - vh) / 2;
      vh = nextH;
    } else {
      const nextW = vh * screenRatio;
      vx -= (nextW - vw) / 2;
      vw = nextW;
    }
    _setView(root, vx, vy, vw, vh);
  }

  function _zoomView(root, factor) {
    const v = root && root._kuView;
    if (!v) return;
    const nw = Math.max(180, v.w * factor);
    const nh = Math.max(150, v.h * factor);
    _setView(root, v.x + (v.w - nw) / 2, v.y + (v.h - nh) / 2, nw, nh);
  }

  function _wireGraphControls(root) {
    if (!root || root._controlsWired) return;
    root._controlsWired = true;
    root.addEventListener("click", ev => {
      const btn = ev.target.closest && ev.target.closest("[data-graph-action]");
      if (!btn || !root.contains(btn)) return;
      ev.preventDefault();
      ev.stopPropagation();
      const action = btn.getAttribute("data-graph-action");
      if (action === "fit") _fitView(root, root._lastGraphObj);
      if (action === "in") _zoomView(root, 0.78);
      if (action === "out") _zoomView(root, 1.28);
      if (action === "expand") {
        root.classList.toggle("ku-expanded");
        btn.textContent = root.classList.contains("ku-expanded") ? "Close" : "Fullscreen";
        setTimeout(() => root._lastGraphObj && _render(root._lastGraphObj), 80);
      }
    });
  }

  function _clearNode(node) {
    while (node && node.firstChild) node.removeChild(node.firstChild);
  }

  function _render(G) {
    const root = document.getElementById("ku-graph-root");
    if (!root || !G || !G.nodes || !G.nodes.length) return;

    const svg = root.querySelector("#ku-svg");
    const links = root.querySelector("#ku-links");
    const nodes = root.querySelector("#ku-nodes");
    const fb = root.querySelector("#ku-fb");
    if (!svg || !links || !nodes || !fb) return;
    _wireGraphControls(root);

    _clearNode(links);
    _clearNode(nodes);
    _clearNode(fb);

    const W = Math.max(root.clientWidth || 800, 640);
    const H = root.classList.contains("ku-expanded")
      ? Math.max(window.innerHeight - 112, 620)
      : 620;
    root._kuBaseW = W;
    root._kuBaseH = H;
    root._lastGraphObj = G;
    _setSvgSize(svg, W, H);

    const nodeMap = {};
    G.nodes.forEach(n => { nodeMap[n.id] = n; });

    const l1s = G.nodes.filter(n => n.level === "L1");
    const l2s = G.nodes.filter(n => n.level === "L2");
    const L1_PALETTE = ["#ef4444", "#f97316", "#dc2626", "#fb7185", "#b91c1c"];
    const L2_PALETTE = ["#60a5fa", "#3b82f6", "#2563eb", "#93c5fd", "#1d4ed8"];
    const colorMap = {};
    l1s.forEach((n, idx) => { colorMap[n.id] = L1_PALETTE[idx % L1_PALETTE.length]; });
    l2s.forEach((n, idx) => {
      colorMap[n.id] = L2_PALETTE[idx % L2_PALETTE.length];
    });

    const cx = W / 2;
    const cy = H / 2 + 10;
    const ring = Math.min(W, H) * 0.28;

    l1s.forEach((n, idx) => {
      const a = (2 * Math.PI * idx / Math.max(l1s.length, 1)) - Math.PI / 2;
      n.x = cx + ring * Math.cos(a);
      n.y = cy + ring * Math.sin(a);
    });

    const parentSlots = {};
    l2s.forEach((n, idx) => {
      const parent = nodeMap[n.parent];
      const key = n.parent || "__root__";
      const slot = parentSlots[key] || 0;
      parentSlots[key] = slot + 1;
      const angle = (slot * 2.2) + (idx % 2 ? 0.6 : 0.2);
      const dist = parent ? 84 + (slot % 3) * 18 : 112 + (slot % 3) * 12;
      n.x = (parent ? parent.x : cx) + dist * Math.cos(angle);
      n.y = (parent ? parent.y : cy) + dist * Math.sin(angle);
    });

    fb.innerHTML = "";
    l1s.forEach(n => {
      const pill = document.createElement("span");
      pill.className = "fb-btn";
      pill.style.borderColor = colorMap[n.id] || "#94a3b8";
      pill.style.color = colorMap[n.id] || "#94a3b8";
      pill.textContent = n.label;
      pill.title = (n.ku_count || 0) + " KUs";
      fb.appendChild(pill);
    });

    G.edges.forEach(e => {
      const s = typeof e.source === "string" ? nodeMap[e.source] : e.source;
      const t = typeof e.target === "string" ? nodeMap[e.target] : e.target;
      if (!s || !t) return;
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", s.x);
      line.setAttribute("y1", s.y);
      line.setAttribute("x2", t.x);
      line.setAttribute("y2", t.y);
      line.setAttribute("stroke", e.edge_type === "hierarchy" ? "#556070" : (_REL[e.relation] || "#6b7280"));
      line.setAttribute("stroke-width", e.has_cross_mcq ? "2.4" : (e.edge_type === "hierarchy" ? "1.5" : "1.8"));
      line.setAttribute("opacity", e.edge_type === "hierarchy" ? "0.5" : "0.78");
      if (e.edge_type === "hierarchy") {
        line.setAttribute("stroke-dasharray", "4 3");
      } else {
        line.style.cursor = "pointer";
        line.addEventListener("click", ev => {
          ev.stopPropagation();
          _notify({
            type: "edge",
            source: s.id,
            target: t.id,
            relation: e.relation
          });
        });
        line.addEventListener("mouseover", ev => {
          _tip(ev, "<b>" + _escapeHtml(e.relation) + "</b>" + (e.has_cross_mcq ? ' <span style="color:#34d399">MCQ</span>' : ""));
        });
        line.addEventListener("mouseout", _hideTip);
      }
      links.appendChild(line);
    });

    let selectedId = null;
    function highlight(id) {
      selectedId = id;
      nodes.querySelectorAll("[data-node-id]").forEach(g => {
        const isOn = g.getAttribute("data-node-id") === id;
        const circle = g.querySelector("circle");
        if (circle) {
          circle.setAttribute("stroke", isOn ? "#fff" : "rgba(255,255,255,0)");
          circle.setAttribute("stroke-width", isOn ? "2.5" : "0");
        }
      });
    }

    G.nodes.forEach(n => {
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("data-node-id", n.id);
      g.style.cursor = "pointer";
      g.style.pointerEvents = "all";
      g.addEventListener("click", ev => {
        ev.stopPropagation();
        highlight(n.id);
        _notify({ type: "concept", id: n.id });
      });
      g.addEventListener("mouseover", ev => {
        _tip(ev,
          "<b style='color:" + (colorMap[n.id] || "#94a3b8") + "'>" + _escapeHtml(n.label) + "</b>"
          + " <span style='color:#9ca3af;font-size:10px'>" + _escapeHtml(n.level) + "</span><br>"
          + "<span style='color:#6b7280'>" + (n.ku_count || 0) + " KUs</span>"
        );
      });
      g.addEventListener("mouseout", _hideTip);

      const hit = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      hit.setAttribute("cx", n.x);
      hit.setAttribute("cy", n.y);
      hit.setAttribute("r", n.level === "L1" ? "38" : "30");
      hit.setAttribute("fill", "transparent");
      hit.setAttribute("stroke", "transparent");
      hit.setAttribute("pointer-events", "all");
      g.appendChild(hit);

      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", n.x);
      circle.setAttribute("cy", n.y);
      circle.setAttribute("r", n.level === "L1" ? "24" : "15");
      circle.setAttribute("fill", colorMap[n.id] || "#94a3b8");
      circle.setAttribute("fill-opacity", n.level === "L1" ? "0.95" : "0.9");
      circle.setAttribute("stroke", n.level === "L1" ? "rgba(255,255,255,.28)" : "rgba(255,255,255,.18)");
      circle.setAttribute("stroke-width", n.level === "L1" ? "2" : "1.4");
      g.appendChild(circle);

      if (n.level === "L1" && n.ku_count) {
        const count = document.createElementNS("http://www.w3.org/2000/svg", "text");
        count.setAttribute("x", n.x);
        count.setAttribute("y", n.y + 0.5);
        count.setAttribute("text-anchor", "middle");
        count.setAttribute("font-size", "10");
        count.setAttribute("font-weight", "700");
        count.setAttribute("fill", "#fff");
        count.setAttribute("pointer-events", "none");
        count.textContent = String(n.ku_count);
        g.appendChild(count);
      }

      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", n.x);
      label.setAttribute("y", n.y + (n.level === "L1" ? 40 : 31));
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("font-size", n.level === "L1" ? "15" : "12");
      label.setAttribute("font-weight", n.level === "L1" ? "800" : "600");
      label.setAttribute("fill", n.level === "L1" ? "#fecaca" : "#bfdbfe");
      label.setAttribute("pointer-events", "none");
      label.setAttribute("paint-order", "stroke");
      label.setAttribute("stroke", "#111827");
      label.setAttribute("stroke-width", n.level === "L1" ? "4" : "3");
      label.setAttribute("stroke-linejoin", "round");
      label.textContent = n.label.length > (n.level === "L1" ? 24 : 18)
        ? n.label.slice(0, n.level === "L1" ? 23 : 17) + "…"
        : n.label;
      g.appendChild(label);

      nodes.appendChild(g);
    });

    highlight(selectedId);
    _fitView(root, G);
  }

  window.initKUGraph = function(G) {
    if (!G || !G.nodes || !G.nodes.length) return;
    requestAnimationFrame(() => _render(G));
  };

  document.addEventListener("click", ev => {
    const kuBtn = ev.target.closest && ev.target.closest("[data-ku-id]");
    if (!kuBtn) return;
    ev.preventDefault();
    _notify({
      type: "ku",
      id: kuBtn.getAttribute("data-ku-id"),
      concept: kuBtn.getAttribute("data-concept") || ""
    });
  });

  new MutationObserver(() => {
    const root = document.getElementById("ku-graph-root");
    if (!root) return;
    const raw = root.getAttribute("data-graph");
    if (!raw || raw === root._lastGraph) return;
    root._lastGraph = raw;
    try {
      const G = JSON.parse(raw);
      if (G && G.nodes && G.nodes.length) window.initKUGraph(G);
    } catch (e) {
      console.error("[KUGraph] parse error", e);
      _showError("KU Graph JSON parse error.");
    }
  }).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["data-graph"] });

  const initialRoot = document.getElementById("ku-graph-root");
  if (initialRoot && initialRoot.getAttribute("data-graph")) {
    try {
      const initialGraph = JSON.parse(initialRoot.getAttribute("data-graph"));
      if (initialGraph && initialGraph.nodes && initialGraph.nodes.length) {
        window.initKUGraph(initialGraph);
      }
    } catch (e) {
      console.error("[KUGraph] initial parse error", e);
    }
  }
})()
"""


def _build_graph_html_container(graph_data=None) -> str:
    """SVG container; if graph_data provided, embeds as data-graph attr for MutationObserver."""
    import html as _html
    data_attr = ""
    if graph_data:
        escaped = _html.escape(json.dumps(graph_data, ensure_ascii=False), quote=True)
        data_attr = f' data-graph="{escaped}"'
    return f"""<div id="ku-graph-root"{data_attr}
  style="width:100%;background:#1e1e2e;border-radius:10px;overflow:hidden;position:relative;user-select:none">
<style>
#ku-graph-root svg{{width:100%;height:620px;display:block}}
#ku-graph-root .node-lbl{{font-size:10px;fill:#94a3b8;pointer-events:none}}
.ku-tip{{position:absolute;background:#2d2d3d;border:1px solid #555;border-radius:8px;
  padding:8px 12px;font-size:12px;color:#e5e7eb;pointer-events:none;max-width:240px;
  z-index:50;display:none;line-height:1.5}}
.ku-graph-toolbar{{display:flex;gap:8px;align-items:center;justify-content:flex-end;
  padding:8px 12px;background:#111827;border-bottom:1px solid #2d2d3d}}
.ku-graph-tool{{padding:5px 10px;border-radius:9px;border:1px solid #374151;
  background:#1f2937;color:#dbeafe;font-size:12px;cursor:pointer;font-family:inherit}}
.ku-graph-tool:hover{{border-color:#60a5fa;color:#fff;background:#263244}}
#ku-graph-root.ku-expanded{{position:fixed!important;inset:18px!important;z-index:9999!important;
  border:1px solid #475569;box-shadow:0 24px 80px rgba(0,0,0,.55)}}
#ku-graph-root.ku-expanded svg{{height:calc(100vh - 112px)}}
#ku-fb{{display:flex;flex-wrap:wrap;gap:6px;padding:8px 12px;
  background:#161622;border-bottom:1px solid #2d2d3d}}
.fb-btn{{padding:3px 10px;border-radius:10px;border:1.5px solid;background:transparent;
  cursor:pointer;font-size:11px;transition:opacity .2s;font-family:inherit}}
.fb-btn.off{{opacity:.28}}
</style>
<div class="ku-graph-toolbar">
  <button type="button" class="ku-graph-tool" data-graph-action="fit">Fit</button>
  <button type="button" class="ku-graph-tool" data-graph-action="in">+</button>
  <button type="button" class="ku-graph-tool" data-graph-action="out">-</button>
  <button type="button" class="ku-graph-tool" data-graph-action="expand">Fullscreen</button>
</div>
<div id="ku-fb"></div>
    <svg id="ku-svg">
  <defs>
    <marker id="ku-arr" viewBox="0 -3 8 6" refX="16" refY="0"
      markerWidth="5" markerHeight="5" orient="auto">
      <path d="M0,-3L8,0L0,3" fill="#4b5563"/>
    </marker>
  </defs>
  <g id="ku-zoom"><g id="ku-links"></g><g id="ku-nodes"></g></g>
</svg>
<div class="ku-tip" id="ku-tip"></div>
<style>
#graph_click_data, #graph_click_trigger {{ display:none !important; }}
</style>
</div>"""


class _Pass2Stub:
    """Lightweight adapter so render helpers work from serialized graph_data dict."""
    def __init__(self, graph_data: dict):
        self.concepts: dict[str, dict] = graph_data.get("concepts", {}) or {}
        if not self.concepts:
            for n in graph_data.get("nodes", []):
                nid = n.get("id", "")
                if nid:
                    self.concepts[nid] = dict(n)

        raw_kus = list(graph_data.get("kus", []) or [])
        if not raw_kus:
            for node in self.concepts.values():
                raw_kus.extend(node.get("kus", []) or [])

        self.all_kus: list[dict] = []
        self.ku_map: dict[str, dict] = {}
        self.concept_kus: dict[str, list[dict]] = {}

        def _add_concept_ku(concept_id: str, ku_node: dict) -> None:
            concept_id = str(concept_id or "").strip()
            ku_id = str(ku_node.get("ku_id", "") or "").strip()
            if not concept_id or not ku_id:
                return
            bucket = self.concept_kus.setdefault(concept_id, [])
            if all(existing.get("ku_id") != ku_id for existing in bucket):
                bucket.append(ku_node)

        for n in raw_kus:
            node = dict(n)
            node.setdefault("ku_id", node.get("id", ""))
            node.setdefault("id", node.get("ku_id", ""))
            node.setdefault("concept", node.get("concept", ""))
            self.all_kus.append(node)
            if node["ku_id"]:
                self.ku_map[node["ku_id"]] = node
            _add_concept_ku(node.get("ui_node_id", ""), node)
            _add_concept_ku(node.get("owner_concept", ""), node)
            _add_concept_ku(node.get("concept", ""), node)

        self.graph: dict[str, list[str]] = {k: list(v) for k, v in (graph_data.get("ku_graph", {}) or {}).items()}
        self.edge_types: dict = {}
        edge_items = graph_data.get("ku_edge_types", []) or graph_data.get("ku_edges", []) or graph_data.get("edges", [])
        for e in edge_items:
            s, t = e["source"], e["target"]
            self.graph.setdefault(s, [])
            self.graph.setdefault(t, [])
            if t not in self.graph[s]:
                self.graph[s].append(t)
            if s not in self.graph[t]:
                self.graph[t].append(s)
            self.edge_types[(s, t)] = e["relation"]
            self.edge_types[(t, s)] = e["relation"]

    @property
    def ok(self) -> bool:
        return bool(self.all_kus)

    def get_distractors(self, anchor_ku_id, n=3):
        return []


def _fmt_pages(pages: list) -> str:
    """Format page numbers as a human-readable label."""
    if not pages:
        return ""
    nums = sorted(set(int(p) for p in pages))
    groups, cur = [], [nums[0]]
    for n in nums[1:]:
        if n == cur[-1] + 1:
            cur.append(n)
        else:
            groups.append(cur); cur = [n]
    groups.append(cur)
    parts = [f"{g[0]}-{g[-1]}" if len(g) > 1 else str(g[0]) for g in groups]
    prefix = "Page" if len(nums) == 1 and len(groups) == 1 and len(groups[0]) == 1 else "Pages"
    return prefix + " " + ", ".join(parts)


def _clean_explanation_text(text: str) -> str:
    """Keep feedback learner-facing even if LLM leaks authoring terms."""
    cleaned = str(text or "")
    replacements = [
        ("phương án gây nhiễu", "lựa chọn chưa đúng"),
        ("đáp án gây nhiễu", "lựa chọn chưa đúng"),
        ("các phương án gây nhiễu", "các lựa chọn chưa đúng"),
        ("gây nhiễu", "chưa đúng"),
        ("distractor", "lựa chọn chưa đúng"),
        ("distractors", "các lựa chọn chưa đúng"),
        ("wrong answer candidate", "lựa chọn chưa đúng"),
        ("wrong answer candidates", "các lựa chọn chưa đúng"),
        ("nội dung nguồn", "kiến thức cần nắm"),
        ("bằng chứng nguồn", "thông tin đã cho"),
        ("source evidence", "thông tin đã cho"),
        ("source content", "nội dung bài học"),
        ("the source", "bài học"),
        ("nguồn", "bài học"),
        ("KU", "khái niệm"),
        ("Knowledge Unit", "khái niệm"),
    ]
    for old, new in replacements:
        cleaned = re.sub(re.escape(old), new, cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def _render_mcq_block(title: str, mcq, edge_color: str = "#4b5563") -> str:
    if not mcq:
        return ""
    answer = str(getattr(mcq, "answer", "") or "").strip().upper()
    options = getattr(mcq, "options", {}) or {}
    correct_text = html.escape(str(options.get(answer, "")), quote=True)
    explanation = html.escape(_clean_explanation_text(getattr(mcq, "explanation", "") or ""), quote=True)
    is_cross = bool(getattr(mcq, "anchor_ku_id_b", ""))
    card_label = "CONNECTION QUESTION" if is_cross else title
    opts = ""
    for k, v in options.items():
        opt = html.escape(str(k), quote=True)
        text = html.escape(str(v), quote=True)
        opts += f"""
  <button type="button" data-option="{opt}" onclick="(function(opt){{
    var card=opt.closest('[data-mcq-card]');
    if(!card || card.getAttribute('data-answered')==='1') return;
    card.setAttribute('data-answered','1');
    var correct=card.getAttribute('data-answer');
    card.querySelectorAll('[data-option]').forEach(function(x){{
      var letter=x.getAttribute('data-option');
      x.style.pointerEvents='none';
      x.style.opacity='1';
      if(letter===correct){{
        x.style.background='#153d2b';
        x.style.borderColor='#34d399';
        x.style.color='#bbf7d0';
        var st=x.querySelector('[data-status]');
        if(st) st.textContent='Đáp án đúng';
      }}
    }});
    if(opt.getAttribute('data-option')!==correct){{
      opt.style.background='#3b1821';
      opt.style.borderColor='#fb7185';
      opt.style.color='#fecdd3';
      var bad=opt.querySelector('[data-status]');
      if(bad) bad.textContent='Your choice';
    }}
    var fb=card.querySelector('[data-feedback]');
    if(fb) fb.style.display='block';
  }})(this)"
    style="width:100%;text-align:left;padding:9px 12px;margin:4px 0;border-radius:8px;
    background:#2d2d3d;border:1px solid #44465a;font-size:13px;color:#e5e7eb;
    cursor:pointer;line-height:1.55;transition:all .14s ease">
    <span style="display:flex;align-items:flex-start;justify-content:space-between;gap:10px">
      <span><b>{opt}.</b> {text}</span>
      <span data-status style="flex:0 0 auto;font-size:10px;letter-spacing:.06em;color:inherit;opacity:.85"></span>
    </span>
  </button>"""
    bloom_badge = (
        f'<span style="padding:2px 8px;border-radius:8px;border:1px solid #4b5563;'
        f'color:#9ca3af;font-size:11px">{mcq.bloom_level}</span>'
    )
    relation_badge = ""
    if is_cross:
        relation_badge = (
            f'<span style="padding:2px 8px;border-radius:8px;border:1px solid {edge_color};'
            f'color:{edge_color};font-size:11px">concept link</span>'
        )
    page_label = _fmt_pages(getattr(mcq, "source_pages", []) or [])
    source_line = ""
    if page_label:
        source_text = page_label
        if is_cross:
            source_text += " (kết hợp từ 2 khái niệm)"
        source_line = f'<div style="font-size:11px;color:#9ca3af;margin:-2px 0 10px">Trang: {source_text}</div>'
    cross_note = ""
    return f"""
<div data-mcq-card data-answer="{answer}" style="margin-top:16px;border:1px solid {edge_color};border-radius:8px;padding:12px;background:#252535">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
    <span style="font-size:11px;color:#6b7280;letter-spacing:.06em">{card_label}</span>
    {bloom_badge}
    {relation_badge}
  </div>
  {source_line}
  {cross_note}
  <p style="margin:0 0 10px;font-size:14px;color:#f3f4f6;line-height:1.5">{mcq.question}</p>
  {opts}
  <div data-feedback style="display:none;margin-top:10px;padding:10px 12px;border-radius:8px;background:#171724;border:1px solid #343447;color:#d1d5db;font-size:12px;line-height:1.55">
    <div><b style="color:#34d399">Đáp án đúng: {answer}.</b> {correct_text}</div>
    {f'<div style="margin-top:6px;color:#9ca3af">{explanation}</div>' if explanation else ''}
  </div>
</div>"""


def _render_concept_panel(d: dict, pass2_result, mcqs_all=None) -> str:
    concept = d.get("id", "")
    info = getattr(pass2_result, "concepts", {}).get(concept, {}) or {}
    # Prefer the serialized node payload because it already uses ui_node_id.
    # Falling back to concept_kus keeps older graph_data payloads working.
    kus = list(info.get("kus", []) or [])
    if not kus:
        kus = list(getattr(pass2_result, "concept_kus", {}).get(concept, []))

    color = info.get("color", "#60a5fa")
    level = info.get("level", "L2")
    parent = info.get("parent", "")

    ku_cards = ""
    for ku in kus:
        pages_s = _fmt_pages(ku.get("source_pages", []))
        ku_cards += f"""
<button type="button" data-ku-id="{ku.get('ku_id','')}" data-concept="{concept}"
  style="width:100%;text-align:left;padding:10px 12px;margin:4px 0;border-radius:8px;
  background:#252535;border:1px solid #374151;color:#e5e7eb;cursor:pointer">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:8px">
    <div style="min-width:0">
      <div style="font-size:13px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{ku.get('concept','')}</div>
      <div style="font-size:11px;color:#9ca3af;margin-top:2px">{ku.get('type','')} · {pages_s or 'p.-'}</div>
    </div>
    <span style="padding:2px 8px;border-radius:8px;border:1px solid #4b5563;color:#9ca3af;font-size:11px">Open</span>
  </div>
</button>"""

    if not ku_cards:
        ku_cards = '<p style="color:#9ca3af;font-size:13px">No KU found under this concept.</p>'

    return f"""<div style="padding:16px;background:#1e1e2e;min-height:620px;font-family:system-ui,sans-serif;overflow-y:auto">
  <h2 style="margin:0 0 10px;font-size:19px;font-weight:700;color:#f3f4f6">{concept}</h2>
  <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px">
    <span style="padding:3px 10px;border-radius:12px;border:1.5px solid {color};color:{color};font-size:12px">{level}</span>
    <span style="padding:3px 10px;border-radius:12px;border:1.5px solid #4b5563;color:#9ca3af;font-size:12px">{len(kus)} KUs</span>
    {f'<span style="padding:3px 10px;border-radius:12px;border:1.5px solid #4b5563;color:#6b7280;font-size:12px">{parent}</span>' if parent else ""}
  </div>
  <div style="font-size:11px;letter-spacing:.07em;color:#6b7280;margin-bottom:6px">KUS IN THIS CONCEPT</div>
  <div style="display:flex;flex-direction:column;gap:0;margin-bottom:14px">{ku_cards}</div>
</div>"""


def _visible_ku_relations(ku_id: str, pass2_result) -> list[tuple[str, str]]:
    """Relationships shown in the UI. Hide auto-generated sibling edges."""
    visible: list[tuple[str, str]] = []
    for nid in getattr(pass2_result, "graph", {}).get(ku_id, []):
        rel = (
            pass2_result.edge_types.get((ku_id, nid))
            or pass2_result.edge_types.get((nid, ku_id))
            or "RELATED_TO"
        )
        if rel == "SIBLING_OF":
            continue
        visible.append((nid, rel))
    return visible


def _render_ku_panel(ku: dict, pass2_result, mcqs_all=None) -> str:
    ku_map = getattr(pass2_result, "ku_map", {}) or {k["ku_id"]: k for k in pass2_result.all_kus}
    color = _TYPE_COLORS.get(ku["type"], "#888")
    icon = _TYPE_ICONS.get(ku["type"], "📘")
    pages_s = _fmt_pages(ku.get("source_pages", []))
    bloom = ku.get("bloom_level") or ""
    difficulty = ku.get("difficulty") or ""

    neighbors = _visible_ku_relations(ku["ku_id"], pass2_result)
    related_cards = ""
    for nid, rel in neighbors:
        other = ku_map.get(nid)
        if not other:
            continue
        rel_color = (_EDGE_STYLE.get(rel) or {}).get("color", "#4b5563")
        rel_label = (_EDGE_STYLE.get(rel) or {}).get("label", rel)
        other_pages = _fmt_pages(other.get("source_pages", []))
        related_cards += f"""
<div style="padding:10px;background:#252535;border-radius:8px;border-left:3px solid {rel_color};margin-bottom:8px">
  <div style="font-size:11px;color:{rel_color};letter-spacing:.06em;margin-bottom:4px">{rel_label} · {rel}</div>
  <div style="font-weight:600;color:#e5e7eb;font-size:13px">{other.get('concept','')}</div>
  <div style="font-size:11px;color:#9ca3af;margin-top:3px">{other.get('type','')} · {other_pages or 'p.-'}</div>
</div>"""
    if not related_cards:
        related_cards = '<p style="color:#9ca3af;font-size:13px">No KU relationships found.</p>'

    practice_mcq = None
    cross_mcq = None
    if mcqs_all:
        ku_mcqs = [m for m in mcqs_all if not getattr(m, "anchor_ku_id_b", "") and m.anchor_ku_id == ku["ku_id"]]
        if ku_mcqs:
            practice_mcq = ku_mcqs[0]
        cross_mcqs = [
            m for m in mcqs_all
            if getattr(m, "anchor_ku_id_b", "") and (m.anchor_ku_id == ku["ku_id"] or m.anchor_ku_id_b == ku["ku_id"])
        ]
        if cross_mcqs:
            cross_mcq = cross_mcqs[0]

    practice_bloom = practice_mcq.bloom_level if practice_mcq else "understand"
    difficulty_badge = practice_mcq.difficulty if practice_mcq else "medium"
    mcq_html = _render_mcq_block("PRACTICE QUESTION", practice_mcq, color)
    cross_mcq_html = _render_mcq_block("CROSS-CONCEPT QUESTIONS", cross_mcq, color)
    return f"""<div style="padding:16px;background:#1e1e2e;min-height:420px;font-family:system-ui,sans-serif;overflow-y:auto">
  <h2 style="margin:0 0 10px;font-size:19px;font-weight:700;color:#f3f4f6">{ku["concept"]}</h2>
  <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px">
    <span style="padding:3px 10px;border-radius:12px;border:1.5px solid {color};color:{color};font-size:12px">{icon} {ku["type"].replace('_',' ')}</span>
    <span style="padding:3px 10px;border-radius:12px;border:1.5px solid #4b5563;color:#9ca3af;font-size:12px">{practice_bloom}</span>
    <span style="padding:3px 10px;border-radius:12px;border:1.5px solid #facc15;color:#facc15;font-size:12px">{difficulty_badge}</span>
    {f'<span style="padding:3px 10px;border-radius:12px;border:1.5px solid #4b5563;color:#6b7280;font-size:12px">{pages_s}</span>' if pages_s else ""}
  </div>
  <div style="font-size:11px;letter-spacing:.07em;color:#6b7280;margin-bottom:5px">VERBATIM EVIDENCE</div>
  <div style="padding:10px 14px;background:#252535;border-left:3px solid {color};border-radius:0 8px 8px 0;margin-bottom:14px">
    <em style="font-size:13px;color:#d1d5db;line-height:1.6">"{ku.get('verbatim_evidence','')}"</em>
  </div>
  <div style="font-size:11px;letter-spacing:.07em;color:#6b7280;margin-bottom:5px">KNOWLEDGE UNIT CONTENT</div>
  <div style="padding:12px;background:#252535;border-radius:8px;margin-bottom:14px">
    <p style="margin:0;font-size:14px;color:#e5e7eb;line-height:1.6">{ku.get('content','')}</p>
  </div>
  <div style="font-size:11px;letter-spacing:.07em;color:#6b7280;margin-bottom:6px">RELATED KUS</div>
  <div style="margin-bottom:14px">{related_cards}</div>
  {mcq_html}
  {cross_mcq_html}
</div>"""


def _render_ku_panel_v2(ku: dict, pass2_result, mcqs_all=None) -> str:
    ku_map = getattr(pass2_result, "ku_map", {}) or {k["ku_id"]: k for k in pass2_result.all_kus}
    color = _TYPE_COLORS.get(ku["type"], "#888")
    icon = _TYPE_ICONS.get(ku["type"], "[KU]")
    pages_s = _fmt_pages(ku.get("source_pages", []))
    practice_bloom = ku.get("bloom_level") or "understand"
    difficulty_badge = ku.get("difficulty") or "medium"

    neighbors = _visible_ku_relations(ku["ku_id"], pass2_result)
    related_cards = ""
    for nid, rel in neighbors:
        other = ku_map.get(nid)
        if not other:
            continue
        rel_color = (_EDGE_STYLE.get(rel) or {}).get("color", "#4b5563")
        other_pages = _fmt_pages(other.get("source_pages", []))
        related_cards += f"""
<button type="button" data-ku-id="{other.get('ku_id','')}" data-concept="{other.get('concept','')}"
  style="border:0;background:transparent;padding:0;cursor:pointer;text-align:left;min-width:0;flex:1 1 240px">
  <div style="height:100%;padding:10px 12px;background:#252535;border:1px solid #343447;border-radius:12px">
    <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:6px">
      <span style="padding:2px 8px;border-radius:999px;border:1px solid {rel_color};color:{rel_color};font-size:10px;letter-spacing:.05em">{rel}</span>
      <span style="padding:2px 8px;border-radius:999px;border:1px solid #4b5563;color:#9ca3af;font-size:10px">{other.get('type','')}</span>
    </div>
    <div style="font-weight:700;color:#f3f4f6;font-size:13px;line-height:1.35;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{other.get('concept','')}</div>
    <div style="font-size:11px;color:#9ca3af;margin-top:3px">{other_pages or 'p.-'}</div>
  </div>
</button>"""
    if not related_cards:
        related_cards = '<p style="color:#9ca3af;font-size:13px">No KU relationships found.</p>'

    practice_mcq = None
    cross_mcq = None
    if mcqs_all:
        ku_mcqs = [m for m in mcqs_all if not getattr(m, "anchor_ku_id_b", "") and m.anchor_ku_id == ku["ku_id"]]
        if ku_mcqs:
            practice_mcq = ku_mcqs[0]
        cross_mcqs = [
            m for m in mcqs_all
            if getattr(m, "anchor_ku_id_b", "") and (m.anchor_ku_id == ku["ku_id"] or m.anchor_ku_id_b == ku["ku_id"])
        ]
        if cross_mcqs:
            cross_mcq = cross_mcqs[0]

    mcq_html = _render_mcq_block("PRACTICE QUESTION", practice_mcq, color)
    cross_mcq_html = _render_mcq_block("CROSS-CONCEPT QUESTIONS", cross_mcq, color)
    panel_id = re.sub(r"[^a-zA-Z0-9_-]", "-", ku.get("ku_id", "ku-panel"))
    mcq_cards = mcq_html + cross_mcq_html
    if not mcq_cards.strip():
        mcq_cards = '<div style="padding:14px 16px;background:#252535;border:1px solid #313145;border-radius:14px;color:#9ca3af;font-size:13px">No MCQ available for this KU yet.</div>'

    stats_html = f"""
    <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px">
      <div style="padding:12px 14px;background:#252535;border:1px solid #313145;border-radius:14px">
        <div style="font-size:11px;letter-spacing:.08em;color:#6b7280;margin-bottom:6px">TYPE</div>
        <div style="font-size:15px;font-weight:700;color:#f3f4f6">{ku.get('type','').replace('_',' ')}</div>
      </div>
      <div style="padding:12px 14px;background:#252535;border:1px solid #313145;border-radius:14px">
        <div style="font-size:11px;letter-spacing:.08em;color:#6b7280;margin-bottom:6px">PAGES</div>
        <div style="font-size:15px;font-weight:700;color:#f3f4f6">{pages_s or 'Unknown'}</div>
      </div>
      <div style="padding:12px 14px;background:#252535;border:1px solid #313145;border-radius:14px">
        <div style="font-size:11px;letter-spacing:.08em;color:#6b7280;margin-bottom:6px">RELATED KUS</div>
        <div style="font-size:15px;font-weight:700;color:#f3f4f6">{len(neighbors)}</div>
      </div>
      <div style="padding:12px 14px;background:#252535;border:1px solid #313145;border-radius:14px">
        <div style="font-size:11px;letter-spacing:.08em;color:#6b7280;margin-bottom:6px">MCQ</div>
        <div style="font-size:15px;font-weight:700;color:#f3f4f6">{int(practice_mcq is not None) + int(cross_mcq is not None)} available</div>
      </div>
    </div>"""

    return f"""<div id="{panel_id}" style="background:#1e1e2e;min-height:620px;font-family:system-ui,sans-serif;overflow:hidden;border-radius:14px">
  <style>
    #{panel_id} .ku-tab-btn {{
      flex:1 1 0;
      padding:16px 12px;
      border:0;
      border-bottom:2px solid transparent;
      background:transparent;
      color:#8f96b2;
      font-size:13px;
      letter-spacing:.08em;
      cursor:pointer;
      transition:all .18s ease;
    }}
    #{panel_id} .ku-tab-btn.is-active {{
      color:#f3f4f6;
      border-bottom-color:#60a5fa;
      background:linear-gradient(180deg, rgba(96,165,250,.10), rgba(96,165,250,0));
    }}
    #{panel_id} .ku-tab-pane {{
      display:none;
      padding:18px;
      min-height:560px;
      overflow-y:auto;
    }}
    #{panel_id} .ku-tab-pane.is-active {{
      display:block;
    }}
  </style>
  <div style="display:flex;align-items:center;border-bottom:1px solid #2a2a3c;background:#171724">
    <button type="button" class="ku-tab-btn is-active" data-tab-target="detail"
      onclick="(function(btn){{var root=btn.closest('div[id]');root.querySelectorAll('.ku-tab-btn').forEach(function(x){{x.classList.remove('is-active')}});root.querySelectorAll('.ku-tab-pane').forEach(function(x){{x.classList.remove('is-active')}});btn.classList.add('is-active');root.querySelector('[data-tab-pane=detail]').classList.add('is-active');}})(this)">KU Detail</button>
    <button type="button" class="ku-tab-btn" data-tab-target="mcq"
      onclick="(function(btn){{var root=btn.closest('div[id]');root.querySelectorAll('.ku-tab-btn').forEach(function(x){{x.classList.remove('is-active')}});root.querySelectorAll('.ku-tab-pane').forEach(function(x){{x.classList.remove('is-active')}});btn.classList.add('is-active');root.querySelector('[data-tab-pane=mcq]').classList.add('is-active');}})(this)">MCQ</button>
    <button type="button" class="ku-tab-btn" data-tab-target="stats"
      onclick="(function(btn){{var root=btn.closest('div[id]');root.querySelectorAll('.ku-tab-btn').forEach(function(x){{x.classList.remove('is-active')}});root.querySelectorAll('.ku-tab-pane').forEach(function(x){{x.classList.remove('is-active')}});btn.classList.add('is-active');root.querySelector('[data-tab-pane=stats]').classList.add('is-active');}})(this)">Stats</button>
  </div>

  <div class="ku-tab-pane is-active" data-tab-pane="detail">
    <div style="padding:4px 0 14px;border-bottom:1px solid #2a2a3c;margin-bottom:16px">
      <h2 style="margin:0 0 10px;font-size:22px;font-weight:800;letter-spacing:-.02em;color:#f3f4f6">{ku["concept"]}</h2>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <span style="padding:4px 11px;border-radius:12px;border:1.5px solid {color};color:{color};font-size:12px;font-weight:600">{icon} {ku["type"].replace('_',' ')}</span>
        <span style="padding:4px 11px;border-radius:12px;border:1.5px solid #4b5563;color:#9ca3af;font-size:12px">{practice_bloom}</span>
        <span style="padding:4px 11px;border-radius:12px;border:1.5px solid #facc15;color:#facc15;font-size:12px">{difficulty_badge}</span>
        {f'<span style="padding:4px 11px;border-radius:12px;border:1.5px solid #4b5563;color:#6b7280;font-size:12px">{pages_s}</span>' if pages_s else ""}
      </div>
    </div>

    <div style="margin-bottom:16px">
      <div style="font-size:11px;letter-spacing:.12em;color:#6b7280;margin-bottom:8px">VERBATIM EVIDENCE</div>
      <div style="padding:14px 16px;background:#252535;border-left:3px solid {color};border-radius:0 12px 12px 0">
        <em style="font-size:13px;color:#d1d5db;line-height:1.7">"{ku.get('verbatim_evidence','')}"</em>
      </div>
    </div>

    <div style="margin-bottom:16px">
      <div style="font-size:11px;letter-spacing:.12em;color:#6b7280;margin-bottom:8px">KNOWLEDGE UNIT CONTENT</div>
      <div style="padding:15px 16px;background:#252535;border:1px solid #313145;border-radius:14px;box-shadow:0 6px 20px rgba(0,0,0,.12)">
        <p style="margin:0;font-size:15px;color:#f3f4f6;line-height:1.75">{ku.get('content','')}</p>
      </div>
    </div>

    <div style="margin-bottom:16px">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:10px">
        <div style="font-size:11px;letter-spacing:.12em;color:#6b7280">RELATED KUS</div>
        <div style="font-size:11px;color:#9ca3af">tap to jump</div>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:8px">{related_cards}</div>
    </div>

    <div style="display:grid;grid-template-columns:1fr;gap:10px;margin-top:18px">
      <button type="button" style="width:100%;padding:14px 16px;border-radius:14px;border:1.5px solid #60a5fa;background:#1b2639;color:#60a5fa;font-size:15px;font-weight:600;cursor:pointer"
        onclick="(function(btn){{var root=btn.closest('div[id]');root.querySelectorAll('.ku-tab-btn').forEach(function(x){{x.classList.remove('is-active')}});root.querySelectorAll('.ku-tab-pane').forEach(function(x){{x.classList.remove('is-active')}});root.querySelector('[data-tab-target=mcq]').classList.add('is-active');root.querySelector('[data-tab-pane=mcq]').classList.add('is-active');}})(this)">Practice this concept →</button>
      <button type="button" style="width:100%;padding:14px 16px;border-radius:14px;border:1.5px solid #3a3e52;background:#20212d;color:#f3f4f6;font-size:15px;font-weight:600;cursor:pointer"
        onclick="(function(btn){{var root=btn.closest('div[id]');root.querySelectorAll('.ku-tab-btn').forEach(function(x){{x.classList.remove('is-active')}});root.querySelectorAll('.ku-tab-pane').forEach(function(x){{x.classList.remove('is-active')}});root.querySelector('[data-tab-target=mcq]').classList.add('is-active');root.querySelector('[data-tab-pane=mcq]').classList.add('is-active');}})(this)">Cross-concept questions →</button>
    </div>
  </div>

  <div class="ku-tab-pane" data-tab-pane="mcq">
    <div style="display:grid;grid-template-columns:1fr;gap:12px">
      {mcq_cards}
    </div>
  </div>

  <div class="ku-tab-pane" data-tab-pane="stats">
    {stats_html}
  </div>
</div>"""


def render_default_detail(n_kus: int = 0, n_mcqs: int = 0) -> str:
    return f"""<div style="padding:32px 20px;text-align:center;font-family:system-ui,sans-serif;
                           background:#1e1e2e;min-height:620px;border-radius:10px">
  <p style="font-size:20px;font-weight:700;color:#e5e7eb;margin:0 0 6px">{n_kus} Knowledge Units · {n_mcqs} MCQs</p>
  <p style="font-size:13px;color:#6b7280;margin:0">Click a node to explore KUs · Click an edge for cross-concept questions</p>
</div>"""


def render_ku_detail(click_json: str, pass2_result, mcqs_all=None) -> str:
    """Render right-panel detail for a clicked node."""
    import json as _j
    try:
        d = _j.loads(click_json)
    except Exception:
        return render_default_detail()

    if d.get("type") == "concept":
        return _render_concept_panel(d, pass2_result, mcqs_all)
    if d.get("type") == "ku":
        ku_map = getattr(pass2_result, "ku_map", {}) or {ku["ku_id"]: ku for ku in pass2_result.all_kus}
        ku = ku_map.get(d.get("id", ""))
        return _render_ku_panel_v2(ku, pass2_result, mcqs_all) if ku else render_default_detail()
    if d.get("type") == "edge":
        return _render_edge_panel(d, pass2_result, mcqs_all)

    # --- node panel ---
    ku_map  = {ku["ku_id"]: ku for ku in pass2_result.all_kus}
    node_id = d.get("id", "")
    ku      = ku_map.get(node_id)
    if not ku:
        return render_default_detail()

    color    = _TYPE_COLORS.get(ku["type"], "#888")
    icon     = _TYPE_ICONS.get(ku["type"], "📖")
    pages_s  = _fmt_pages(ku.get("source_pages", []))
    neighbors= _visible_ku_relations(node_id, pass2_result)
    nb_chips = "".join(
        f'<span style="padding:3px 9px;border-radius:8px;background:#252535;border:1px solid #374151;'
        f'color:#9ca3af;font-size:12px">{ku_map[n]["concept"]}</span>'
        for n, _rel in neighbors if n in ku_map
    )

    # find single MCQ for this KU
    mcq_html = ""
    if mcqs_all:
        ku_mcqs = [m for m in mcqs_all
                   if not getattr(m, "anchor_ku_id_b", "") and m.anchor_ku_id == node_id]
        if ku_mcqs:
            m = ku_mcqs[0]
            opts = "".join(
                f'<div style="padding:6px 10px;margin:3px 0;border-radius:6px;'
                f'background:#2d2d3d;border:1px solid #444;font-size:13px;color:#e5e7eb">'
                f'<b>{k}.</b> {v}</div>'
                for k, v in m.options.items()
            )
            bloom_badge = f'<span style="padding:2px 8px;border-radius:8px;border:1px solid #4b5563;color:#9ca3af;font-size:11px">{m.bloom_level}</span>'
            mcq_html = f"""
<div style="margin-top:16px;border:1px solid #333;border-radius:8px;padding:12px;background:#252535">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
    <span style="font-size:11px;color:#6b7280;letter-spacing:.06em">PRACTICE QUESTION</span>
    {bloom_badge}
  </div>
  <p style="margin:0 0 10px;font-size:14px;color:#f3f4f6;line-height:1.5">{m.question}</p>
  {opts}
</div>"""

    nb_section = (
        f'<div style="font-size:11px;letter-spacing:.07em;color:#6b7280;margin-bottom:6px">RELATED CONCEPTS</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px">{nb_chips}</div>'
    ) if nb_chips else ""

    return f"""<div style="padding:16px;background:#1e1e2e;min-height:420px;
                           font-family:system-ui,sans-serif;overflow-y:auto">
  <h2 style="margin:0 0 10px;font-size:19px;font-weight:700;color:#f3f4f6">{ku["concept"]}</h2>
  <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px">
    <span style="padding:3px 10px;border-radius:12px;border:1.5px solid {color};color:{color};font-size:12px">{icon} {ku["type"].replace("_"," ")}</span>
    <span style="padding:3px 10px;border-radius:12px;border:1.5px solid #4b5563;color:#9ca3af;font-size:12px">{"primary" if ku.get("prominence")=="primary" else "supporting"}</span>
    {f'<span style="padding:3px 10px;border-radius:12px;border:1.5px solid #4b5563;color:#6b7280;font-size:12px">p.{pages_s}</span>' if pages_s else ""}
  </div>
  <div style="font-size:11px;letter-spacing:.07em;color:#6b7280;margin-bottom:5px">VERBATIM EVIDENCE</div>
  <div style="padding:10px 14px;background:#252535;border-left:3px solid {color};border-radius:0 8px 8px 0;margin-bottom:14px">
    <em style="font-size:13px;color:#d1d5db;line-height:1.6">"{ku.get("verbatim_evidence","")}"</em>
  </div>
  <div style="font-size:11px;letter-spacing:.07em;color:#6b7280;margin-bottom:5px">KNOWLEDGE UNIT CONTENT</div>
  <div style="padding:12px;background:#252535;border-radius:8px;margin-bottom:14px">
    <p style="margin:0;font-size:14px;color:#e5e7eb;line-height:1.6">{ku.get("content","")}</p>
  </div>
  {nb_section}
  {mcq_html}
</div>"""


def _render_edge_panel(d: dict, pass2_result, mcqs_all=None) -> str:
    """Render right-panel for a clicked concept-level edge."""
    src_id   = d.get("source", "")
    tgt_id   = d.get("target", "")
    relation = d.get("relation", "")

    concept_map = getattr(pass2_result, "concepts", {}) or {}
    info_a = concept_map.get(src_id, {}) or {}
    info_b = concept_map.get(tgt_id, {}) or {}

    if not src_id or not tgt_id:
        return render_default_detail()

    edge_color = (_EDGE_STYLE.get(relation) or {}).get("color", "#6b7280")

    cross_html = ""
    if mcqs_all and relation != "CONTAINS":
        ku_map = getattr(pass2_result, "ku_map", {}) or {}
        cross = []
        for m in mcqs_all:
            if not getattr(m, "anchor_ku_id_b", ""):
                continue
            ka = ku_map.get(m.anchor_ku_id, {})
            kb = ku_map.get(m.anchor_ku_id_b, {})
            ca = ka.get("ui_node_id", "") or ka.get("concept", "")
            cb = kb.get("ui_node_id", "") or kb.get("concept", "")
            if (ca == src_id and cb == tgt_id) or (ca == tgt_id and cb == src_id):
                cross.append(m)
        if cross:
            cross_html = _render_mcq_block("CROSS-CONCEPT QUESTION", cross[0], edge_color)

    def _concept_card(cid, info):
        lvl    = info.get("level", "L2")
        kuc    = info.get("ku_count", 0) or len(info.get("kus", []))
        parent = info.get("parent", "")
        sub    = f" · {parent}" if parent else ""
        return (
            f'<div style="padding:12px;background:#252535;border-radius:10px;border:1px solid #374151">'
            f'<div style="font-size:10px;color:#6b7280;margin-bottom:3px">{lvl}{sub}</div>'
            f'<div style="font-weight:700;color:#f3f4f6;font-size:14px;margin-bottom:5px">{cid}</div>'
            f'<div style="font-size:11px;color:#9ca3af">{kuc} KU{"s" if kuc != 1 else ""}</div>'
            f'</div>'
        )

    return f"""<div style="padding:16px;background:#1e1e2e;min-height:420px;
                           font-family:system-ui,sans-serif;overflow-y:auto">
  <div style="margin-bottom:14px">
    <span style="font-size:14px;font-weight:600;color:#f3f4f6">{src_id}</span>
    <span style="margin:0 8px;color:{edge_color};font-size:16px">↔</span>
    <span style="font-size:14px;font-weight:600;color:#f3f4f6">{tgt_id}</span>
    <div style="margin-top:6px">
      <span style="padding:2px 10px;border-radius:12px;border:1.5px solid {edge_color};
                   color:{edge_color};font-size:12px">{relation}</span>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px">
    {_concept_card(src_id, info_a)}{_concept_card(tgt_id, info_b)}
  </div>
  {cross_html}
</div>"""


# ── MCQ card HTML ─────────────────────────────────────────────────

def render_mcq_card(mcq: dict, idx: int, lang: str) -> str:
    """Render 1 MCQ thành HTML card."""
    diff = mcq.get("difficulty", "medium")
    bloom = mcq.get("bloom_level", "")
    q = mcq.get("question", "")
    answer = mcq.get("answer", "A")
    explanation = _clean_explanation_text(mcq.get("explanation", ""))

    diff_vi = {"easy": "Dễ", "medium": "Trung bình", "hard": "Khó"}.get(diff, diff)
    bloom_vi = {"remember":"Ghi nhớ","understand":"Hiểu","apply":"Vận dụng",
                "analyze":"Phân tích","evaluate":"Đánh giá","create":"Sáng tạo"}.get(bloom, bloom)
    tag_label = bloom_vi if lang == "vi" else bloom.capitalize()
    diff_color = {"easy": "#27AE60", "medium": "#F39C12", "hard": "#E74C3C"}.get(diff, "#666")

    options_html = ""
    for opt in ["A", "B", "C", "D"]:
        text = mcq.get(opt, "")
        is_correct = opt == answer
        if is_correct:
            bg, border, text_color, weight = "#1a4731", "#27AE60", "#6ee7b7", "600"
            prefix = "[Dung] "
        else:
            bg, border, text_color, weight = "#2d2d2d", "#444", "#e5e7eb", "400"
            prefix = ""
        options_html += f"""
        <div style="padding:8px 12px;margin:4px 0;border-radius:8px;
                    background:{bg};border:1.5px solid {border};
                    font-weight:{weight};font-size:13px;color:{text_color};">
            {prefix}<b>{opt}.</b> {text}
        </div>"""

    return f"""
    <div style="border:1px solid #444;border-radius:12px;padding:16px;
                margin-bottom:12px;background:#1e1e2e;">
        <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
            <span style="font-weight:600;color:#e5e7eb;">Cau {idx+1}</span>
            <span style="background:{diff_color};color:white;padding:2px 10px;
                         border-radius:10px;font-size:12px;">{tag_label}</span>
        </div>
        <p style="font-size:14px;font-weight:500;margin:0 0 10px;color:#f3f4f6;">{q}</p>
        {options_html}
        <details style="margin-top:10px;">
            <summary style="font-size:12px;color:#9ca3af;cursor:pointer;">
                Giai thich
            </summary>
            <p style="font-size:12px;color:#d1d5db;margin:6px 0 0;
                      padding:8px;background:#2d2d2d;border-radius:6px;">
                {explanation}
            </p>
        </details>
    </div>"""


def render_review(mcqs_json: str, lang: str,
                  filter_diff: str, search: str) -> str:
    """Render toàn bộ review tab từ MCQ JSON."""
    if not mcqs_json:
        return f"<p style='color:#888;text-align:center;padding:40px'>{t(lang,'review','no_mcq')}</p>"

    try:
        mcqs = json.loads(mcqs_json)
    except Exception:
        return "<p>Lỗi parse MCQ.</p>"

    # Filter
    filtered = mcqs
    if filter_diff and filter_diff != t(lang, "review", "all"):
        diff_key = DIFFICULTY_MAP.get(filter_diff, filter_diff)
        filtered = [m for m in filtered if m.get("difficulty") == diff_key]
    if search:
        search_lower = search.lower()
        filtered = [m for m in filtered if search_lower in m.get("question","").lower()]

    if not filtered:
        return "<p style='color:#888;text-align:center;padding:20px'>Không tìm thấy câu hỏi phù hợp.</p>"

    cards = "".join(render_mcq_card(m, i, lang) for i, m in enumerate(filtered))
    return f'<div style="max-height:70vh;overflow-y:auto;padding:4px">{cards}</div>'


def get_stats_html(mcqs_json: str, lang: str) -> str:
    """Header stats: X câu, phân bố easy/medium/hard."""
    if not mcqs_json:
        return ""
    try:
        mcqs = json.loads(mcqs_json)
    except Exception:
        return ""

    total = len(mcqs)
    easy   = sum(1 for m in mcqs if m.get("difficulty") == "easy")
    medium = sum(1 for m in mcqs if m.get("difficulty") == "medium")
    hard   = sum(1 for m in mcqs if m.get("difficulty") == "hard")

    header = t(lang, "review", "header").format(n=total)
    return f"""
    <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;
                padding:12px 16px;background:#F0F4FF;border-radius:10px;margin-bottom:12px;">
        <span style="font-size:18px;font-weight:600;color:#1E3A5F;">{header}</span>
        <span style="background:#27AE60;color:white;padding:3px 10px;
                     border-radius:8px;font-size:12px;">Dễ: {easy}</span>
        <span style="background:#F39C12;color:white;padding:3px 10px;
                     border-radius:8px;font-size:12px;">TB: {medium}</span>
        <span style="background:#E74C3C;color:white;padding:3px 10px;
                     border-radius:8px;font-size:12px;">Khó: {hard}</span>
    </div>"""


# ── Google Forms status helper ────────────────────────────────────

def _gforms_status_html() -> str:
    try:
        from pipeline.google_forms_exporter import auth_mode, has_credentials_file
        mode = auth_mode()
        if mode == "service_account":
            return "<span style='color:#27AE60'>🟢 Sẵn sàng (Service Account)</span>"
        if mode == "oauth":
            return "<span style='color:#27AE60'>🟢 Đã kết nối Google</span>"
        if has_credentials_file():
            return "<span style='color:#888'>🔴 Chưa kết nối — nhấn Kết nối Google</span>"
        return "<span style='color:#F39C12'>⚠ Thiếu credentials.json</span>"
    except ImportError:
        return "<span style='color:#E74C3C'>🔴 Chưa cài thư viện Google</span>"


# ── Export helpers ─────────────────────────────────────────────────

def export_kahoot(mcqs_json: str):
    if not mcqs_json:
        return None
    mcqs = json.loads(mcqs_json)
    from pipeline.exporter import export_kahoot as _kahoot
    data = _kahoot(mcqs)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    tmp.write(data)
    tmp.close()
    return tmp.name

def export_quizizz(mcqs_json: str):
    if not mcqs_json:
        return None
    mcqs = json.loads(mcqs_json)
    from pipeline.exporter import export_quizizz as _quizizz
    data = _quizizz(mcqs)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(data)
    tmp.close()
    return tmp.name

def export_json_file(mcqs_json: str):
    if not mcqs_json:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json",
                                      mode="w", encoding="utf-8")
    tmp.write(mcqs_json)
    tmp.close()
    return tmp.name


# ── Quiz Player helpers ───────────────────────────────────────────

def render_quiz_question(mcqs_json: str, idx: int) -> tuple:
    """Render 1 câu hỏi cho quiz player. Returns (html, idx, total)."""
    if not mcqs_json:
        return "<p>Không có câu hỏi.</p>", 0, 0, gr.update(visible=False), gr.update(visible=False)
    try:
        mcqs = json.loads(mcqs_json)
    except Exception:
        return "<p>Lỗi.</p>", 0, 0, gr.update(visible=False), gr.update(visible=False)

    total = len(mcqs)
    if total == 0:
        return "<p>Không có câu hỏi.</p>", 0, 0, gr.update(visible=False), gr.update(visible=False)

    idx = max(0, min(idx, total - 1))
    mcq = mcqs[idx]
    q   = mcq.get("question", "")
    diff = mcq.get("difficulty", "medium")
    diff_label = {"easy": "Dễ", "medium": "Trung bình", "hard": "Khó"}.get(diff, diff)
    diff_color = {"easy": "#27AE60", "medium": "#F39C12", "hard": "#E74C3C"}.get(diff, "#666")

    html = f"""
    <div style="padding:20px;background:#F8F9FA;border-radius:12px;min-height:120px">
        <div style="display:flex;justify-content:space-between;margin-bottom:16px">
            <span style="font-size:14px;color:#888">Câu {idx+1} / {total}</span>
            <span style="background:{diff_color};color:white;padding:2px 10px;
                         border-radius:8px;font-size:12px">{diff_label}</span>
        </div>
        <p style="font-size:16px;font-weight:600;color:#1A1A1A;line-height:1.5;margin:0">{q}</p>
    </div>"""

    show_prev = gr.update(visible=(idx > 0))
    show_next = gr.update(visible=(idx < total - 1))
    return html, idx, total, show_prev, show_next


def check_answer(mcqs_json: str, idx: int, chosen: str) -> str:
    """Kiểm tra đáp án, trả về HTML kết quả."""
    if not mcqs_json or chosen is None:
        return ""
    try:
        mcqs = json.loads(mcqs_json)
        mcq  = mcqs[idx]
    except Exception:
        return ""

    correct = mcq.get("answer", "A")
    explanation = _clean_explanation_text(mcq.get("explanation", ""))
    is_correct  = chosen == correct
    correct_text = mcq.get(correct, "")

    icon  = "[Dung]" if is_correct else "[Sai]"
    bg    = "#1a4731" if is_correct else "#4a1c1c"
    border = "#27AE60" if is_correct else "#E74C3C"
    text_color = "#6ee7b7" if is_correct else "#fca5a5"
    msg   = "Chinh xac!" if is_correct else f"Chua dung. Dap an dung la: {correct}. {correct_text}"

    return f"""
    <div style="padding:14px;border-radius:10px;background:{bg};
                border:1.5px solid {border};margin-top:8px">
        <p style="margin:0 0 8px;font-size:15px;font-weight:600;color:{text_color}">
            {icon} {msg}
        </p>
        <p style="margin:0;font-size:13px;color:#d1d5db">
            Giai thich: {explanation}
        </p>
    </div>"""


def get_option_labels(mcqs_json: str, idx: int) -> list:
    """Lấy 4 options cho radio/button."""
    if not mcqs_json:
        return ["A", "B", "C", "D"]
    try:
        mcq = json.loads(mcqs_json)[idx]
        return [
            f"A. {mcq.get('A','')}",
            f"B. {mcq.get('B','')}",
            f"C. {mcq.get('C','')}",
            f"D. {mcq.get('D','')}",
        ]
    except Exception:
        return ["A", "B", "C", "D"]


def _extract_option_letter(chosen) -> str | None:
    """Normalize Gradio radio output back to A/B/C/D."""
    if chosen is None:
        return None
    if isinstance(chosen, str):
        text = chosen.strip()
        if not text:
            return None
        first = text[0].upper()
        return first if first in {"A", "B", "C", "D"} else None
    return None


def get_final_score(mcqs_json: str, answers: dict) -> str:
    """Tính điểm cuối — answers = {idx: chosen_letter}."""
    if not mcqs_json:
        return ""
    try:
        mcqs = json.loads(mcqs_json)
    except Exception:
        return ""

    total   = len(mcqs)
    correct = sum(
        1 for i, mcq in enumerate(mcqs)
        if answers.get(i) == mcq.get("answer")
    )
    pct = round(correct / total * 100) if total else 0
    color = "#27AE60" if pct >= 70 else "#F39C12" if pct >= 50 else "#E74C3C"

    wrong_html = ""
    for i, mcq in enumerate(mcqs):
        if answers.get(i) != mcq.get("answer"):
            wrong_html += f"""
            <div style="padding:8px 12px;margin:4px 0;background:#FEF0F0;
                        border-radius:8px;font-size:13px">
                <b>Câu {i+1}:</b> {mcq.get('question','')[:80]}...<br>
                <span style="color:#E74C3C">Bạn chọn: {answers.get(i,'—')}</span> |
                <span style="color:#27AE60">Đúng: {mcq.get('answer','')}. {mcq.get(mcq.get('answer','A'),'')}</span>
            </div>"""

    return f"""
    <div style="text-align:center;padding:24px">
        <div style="font-size:48px;font-weight:700;color:{color}">{pct}%</div>
        <div style="font-size:18px;color:#555;margin:8px 0">
            {correct} / {total} câu đúng
        </div>
        {"<p style='color:#27AE60;font-size:16px'>Xuất sắc! 🎉</p>" if pct >= 80 else
         "<p style='color:#F39C12;font-size:16px'>Khá tốt! Ôn thêm một chút nhé.</p>" if pct >= 50 else
         "<p style='color:#E74C3C;font-size:16px'>Cần ôn tập thêm nhé! 📚</p>"}
        {"<div style='margin-top:16px;text-align:left'><b style='font-size:14px'>Câu sai:</b>" + wrong_html + "</div>" if wrong_html else ""}
    </div>"""


# ── Build Gradio UI ────────────────────────────────────────────────

def build_app():
    with gr.Blocks(title="AI MCQ Generator", theme=gr.themes.Soft(), js=_GRAPH_BLOCKS_JS_V2) as demo:

        # State — khai báo bên trong Blocks context
        mcqs_state       = gr.State(value=None)
        lang_state       = gr.State(value="vi")
        quiz_idx         = gr.State(value=0)
        user_answers     = gr.State(value={})
        graph_data_state = gr.State(value=None)   # dict for D3
        all_mcqs_state   = gr.State(value=None)   # list[MCQItem]

        # ── Header ────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=8):
                gr.HTML("""
                <div class="main-header">
                    <h1>🎯 AI MCQ Generator</h1>
                    <p id="subtitle">Tạo câu hỏi trắc nghiệm từ tài liệu của bạn</p>
                </div>""")
            with gr.Column(scale=1, min_width=120):
                lang_btn = gr.Button("🌐 English", size="sm", variant="secondary")

        # ── Tabs ──────────────────────────────────────────────────
        with gr.Tabs() as tabs:

            # ════ TAB 1: UPLOAD & GENERATE ════════════════════════
            with gr.TabItem("📤 Tạo Quiz", id="tab_upload") as tab_upload:
                with gr.Row(equal_height=True):

                    # Left: file upload
                    with gr.Column(scale=3):
                        file_input = gr.File(
                            label="Kéo file vào đây · PDF, PPTX, DOCX, XLSX, JPG, PNG",
                            file_count="multiple",
                            file_types=[".pdf", ".pptx", ".docx",
                                        ".xlsx", ".xls",
                                        ".jpg", ".jpeg", ".png", ".webp"],
                            height=220,
                        )
                        file_info = gr.HTML("<p class='estimate-text'></p>")

                    # Right: config
                    with gr.Column(scale=2):
                        gr.Markdown("**Cấu hình**")
                        difficulty_radio = gr.Radio(
                            choices=["Dễ", "Trung bình", "Khó"],
                            value="Trung bình",
                            label="Độ khó",
                        )
                        qlang_radio = gr.Radio(
                            choices=["Tiếng Việt", "English"],
                            value="Tiếng Việt",
                            label="Ngôn ngữ câu hỏi",
                        )

                # Generate button + progress
                generate_btn = gr.Button(
                    "🚀 Tạo Quiz →",
                    variant="primary",
                    size="lg",
                    elem_classes=["generate-btn"],
                )
                progress_text = gr.Textbox(
                    label="Tien trinh",
                    interactive=False,
                    visible=True,
                    value="San sang.",
                    max_lines=1,
                )

            # ════ TAB 2: KU GRAPH ════════════════════════════════════
            with gr.TabItem("🧠 KU Graph", id="tab_graph"):
                with gr.Row():
                    # Left: graph, Right: detail panel
                    with gr.Column(scale=1):
                        graph_html = gr.HTML(_build_graph_html_container())
                        # Hidden textbox receives click events from D3 JS
                        graph_click_input = gr.Textbox(
                            visible=True, interactive=False, value="", elem_id="graph_click_data"
                        )
                        graph_click_trigger = gr.Button(
                            visible=True, elem_id="graph_click_trigger"
                        )
                    with gr.Column(scale=4):
                        ku_detail_html = gr.HTML(render_default_detail())

            # ════ TAB 3: QUIZ PLAYER ══════════════════════════════════
            with gr.TabItem("▶ Chơi thử", id="tab_quiz") as tab_quiz:
                with gr.Row():
                    with gr.Column():
                        quiz_start_btn = gr.Button(
                            "▶ Bắt đầu Quiz", variant="primary", size="lg"
                        )

                # Màn quiz (ẩn ban đầu)
                with gr.Column(visible=False) as quiz_screen:
                    question_html = gr.HTML("")
                    answer_radio  = gr.Radio(
                        choices=["A","B","C","D"],
                        label="Chọn đáp án:",
                        interactive=True,
                    )
                    check_btn   = gr.Button("Kiểm tra", variant="primary")
                    result_html = gr.HTML("")
                    with gr.Row():
                        prev_btn = gr.Button("← Câu trước", visible=False, size="sm")
                        next_btn = gr.Button("Câu tiếp →", variant="primary",
                                             visible=False, size="sm")
                        finish_btn = gr.Button("Xem kết quả 🏁", variant="primary",
                                               visible=False, size="sm")

                # Màn kết quả (ẩn ban đầu)
                with gr.Column(visible=False) as quiz_result_screen:
                    score_html  = gr.HTML("")
                    replay_btn  = gr.Button("Chơi lại", variant="secondary")

            # ════ TAB 4: REVIEW & EDIT ═════════════════════════════
            with gr.TabItem("✏️ Review & Chỉnh sửa", id="tab_review") as tab_review:
                stats_html = gr.HTML("")

                with gr.Row():
                    # Sidebar filters
                    with gr.Column(scale=1, min_width=180):
                        gr.Markdown("**Lọc**")
                        diff_filter = gr.Radio(
                            choices=["Tất cả", "Dễ", "Trung bình", "Khó"],
                            value="Tất cả",
                            label="Độ khó",
                        )
                        search_box = gr.Textbox(
                            placeholder="Tìm câu hỏi...",
                            label="Tìm kiếm",
                            max_lines=1,
                        )
                        with gr.Row():
                            play_btn   = gr.Button("▶ Chơi thử", variant="secondary", size="sm")
                            export_btn = gr.Button("⬇ Tải về",  variant="secondary", size="sm")

                    # MCQ cards
                    with gr.Column(scale=4):
                        mcq_display = gr.HTML("")

            # ════ TAB 4: EXPORT ════════════════════════════════════
            with gr.TabItem("💾 Tải về", id="tab_export") as tab_export:
                gr.Markdown("### Chọn định dạng xuất")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 🎮 Kahoot\nFile Excel import vào Kahoot")
                        kahoot_btn  = gr.Button("Tải Kahoot (.xlsx)", variant="primary")
                        kahoot_file = gr.File(label="", visible=False)
                        gr.Markdown("""
> **Hướng dẫn import:**
> 1. Vào [create.kahoot.it](https://create.kahoot.it)
> 2. Chọn **Import → Spreadsheet**
> 3. Upload file vừa tải
> 4. Kiểm tra và **Publish**
                        """)

                    with gr.Column():
                        gr.Markdown("#### 📊 Quizizz\nFile CSV import vào Quizizz")
                        quizizz_btn  = gr.Button("Tải Quizizz (.csv)", variant="primary")
                        quizizz_file = gr.File(label="", visible=False)
                        gr.Markdown("""
> **Hướng dẫn import:**
> 1. Vào [quizizz.com](https://quizizz.com) → Create
> 2. Chọn **Import từ file**
> 3. Upload file CSV
> 4. Review và **Save**
                        """)

                    with gr.Column():
                        gr.Markdown("#### 📄 JSON\nDữ liệu đầy đủ cho nghiên cứu")
                        json_btn  = gr.Button("Tải JSON", variant="secondary")
                        json_file = gr.File(label="", visible=False)

                    with gr.Column():
                        gr.Markdown("#### 📝 Google Forms\nTạo form với auto-grading")
                        gforms_status = gr.HTML(_gforms_status_html())
                        from pipeline.google_forms_exporter import has_service_account
                        gforms_connect_btn = gr.Button(
                            "Kết nối Google", variant="secondary", size="sm",
                            visible=not has_service_account(),
                        )
                        gforms_title = gr.Textbox(
                            label="Tên form",
                            value="MCQ Quiz",
                            max_lines=1,
                        )
                        gforms_btn = gr.Button("Tạo Google Form", variant="primary")
                        gr.Markdown(
                            "> **HF Spaces**: Thêm `GOOGLE_SERVICE_ACCOUNT_JSON` vào Secrets.\n\n"
                            "> **Local**: Cần `credentials.json` (OAuth 2.0 → Desktop app)."
                        )

        # ── Event handlers ────────────────────────────────────────

        # Generate pipeline
        def on_generate(files, difficulty, qlang, lang,
                        progress=gr.Progress(track_tqdm=False)):
            if not files:
                return (gr.update(value="Vui long upload it nhat 1 file.", visible=True),
                        None, "", "", None, None,
                        _build_graph_html_container(), render_default_detail())

            progress(0, desc="Dang khoi dong...")

            def on_step(pct, msg):
                progress(pct, desc=msg)

            mcqs_json, err, graph_data, all_mcqs_objs = run_pipeline(
                files, difficulty, qlang, lang, on_step=on_step
            )
            progress(1, desc="Hoan tat!")

            if err:
                return (gr.update(value=f"Loi: {err}", visible=True),
                        None, "", "", None, None,
                        _build_graph_html_container(), render_default_detail())
            if not mcqs_json:
                return (gr.update(value="Khong sinh duoc MCQ nao. Thu lai.", visible=True),
                        None, "", "", None, None,
                        _build_graph_html_container(), render_default_detail())

            try:
                n = len(json.loads(mcqs_json))
            except Exception:
                n = 0

            final_stats = get_stats_html(mcqs_json, lang)
            final_cards = render_review(mcqs_json, lang, "Tat ca", "")

            n_kus  = len(graph_data["kus"]) if graph_data and graph_data.get("kus") else (len(graph_data["nodes"]) if graph_data else 0)
            n_mcqs = len(all_mcqs_objs) if all_mcqs_objs else 0

            return (
                gr.update(value=f"Hoan tat! Da tao {n} cau hoi.", visible=True),
                mcqs_json,
                final_stats,
                final_cards,
                graph_data,
                all_mcqs_objs,
                _build_graph_html_container(graph_data),
                render_default_detail(n_kus, n_mcqs),
            )

        generate_btn.click(
            fn=on_generate,
            inputs=[file_input, difficulty_radio, qlang_radio, lang_state],
            outputs=[progress_text, mcqs_state, stats_html, mcq_display,
                     graph_data_state, all_mcqs_state,
                     graph_html, ku_detail_html],
        )

        # Graph node / edge click
        def on_graph_click(click_json, graph_data, all_mcqs_objs):
            if not click_json or not graph_data:
                return render_default_detail()
            try:
                pass2_stub = _Pass2Stub(graph_data)
                return render_ku_detail(click_json, pass2_stub, all_mcqs_objs)
            except Exception as e:
                return f"<p style='color:#ef4444'>Error: {e}</p>"

        graph_click_input.change(
            fn=on_graph_click,
            inputs=[graph_click_input, graph_data_state, all_mcqs_state],
            outputs=[ku_detail_html],
        )
        graph_click_trigger.click(
            fn=on_graph_click,
            inputs=[graph_click_input, graph_data_state, all_mcqs_state],
            outputs=[ku_detail_html],
        )

        # Filter / search in review
        def on_filter(mcqs_json, diff_f, search, lang):
            stats = get_stats_html(mcqs_json, lang)
            cards = render_review(mcqs_json, lang, diff_f, search)
            return stats, cards

        diff_filter.change(on_filter,
                           inputs=[mcqs_state, diff_filter, search_box, lang_state],
                           outputs=[stats_html, mcq_display])
        search_box.change(on_filter,
                          inputs=[mcqs_state, diff_filter, search_box, lang_state],
                          outputs=[stats_html, mcq_display])

        # Export buttons


        def on_export_kahoot(mcqs_json):
            path = export_kahoot(mcqs_json)
            return gr.update(value=path, visible=True) if path else gr.update(visible=False)

        def on_export_quizizz(mcqs_json):
            path = export_quizizz(mcqs_json)
            return gr.update(value=path, visible=True) if path else gr.update(visible=False)

        def on_export_json(mcqs_json):
            path = export_json_file(mcqs_json)
            return gr.update(value=path, visible=True) if path else gr.update(visible=False)

        kahoot_btn.click(on_export_kahoot,
                         inputs=[mcqs_state], outputs=[kahoot_file])
        quizizz_btn.click(on_export_quizizz,
                          inputs=[mcqs_state], outputs=[quizizz_file])
        json_btn.click(on_export_json,
                       inputs=[mcqs_state], outputs=[json_file])

        # Google Forms
        def on_gforms_connect():
            try:
                from pipeline.google_forms_exporter import authenticate
                success, msg = authenticate()
                status = (
                    "<span style='color:#27AE60'>🟢 Đã kết nối Google</span>"
                    if success else
                    f"<span style='color:#E74C3C'>🔴 {msg}</span>"
                )
                return status
            except ImportError:
                return "<span style='color:#E74C3C'>🔴 Chưa cài: pip install google-api-python-client google-auth-oauthlib</span>"

        def on_gforms_export(mcqs_json, title):
            if not mcqs_json:
                return "<span style='color:#F39C12'>⚠ Chưa có MCQ. Tạo quiz trước.</span>"
            try:
                from pipeline.google_forms_exporter import export_to_google_forms
                mcqs = json.loads(mcqs_json)
                success, result = export_to_google_forms(mcqs, title=title or "MCQ Quiz")
                if success:
                    return (
                        f"<span style='color:#27AE60'>🟢 Tạo form thành công!</span>"
                        f"<a href='{result}' target='_blank' "
                        f"style='display:block;padding:10px 14px;margin-top:8px;"
                        f"background:#1a4731;border:1.5px solid #27AE60;border-radius:8px;"
                        f"color:#6ee7b7;font-size:13px;text-decoration:none;'>"
                        f"🔗 Mở Google Form →</a>"
                        f"<p style='font-size:11px;color:#888;margin:4px 0 0;word-break:break-all'>{result}</p>"
                    )
                return f"<span style='color:#E74C3C'>🔴 {result}</span>"
            except Exception as e:
                return (
                    f"<span style='color:#E74C3C'>🔴 Lỗi: {e}</span>",
                    gr.update(visible=False),
                )

        gforms_connect_btn.click(
            on_gforms_connect,
            outputs=[gforms_status],
        )
        gforms_btn.click(
            on_gforms_export,
            inputs=[mcqs_state, gforms_title],
            outputs=[gforms_status],
        )

        # ── Quiz Player events ────────────────────────────────────
        def start_quiz(mcqs_json):
            if not mcqs_json:
                return (
                    gr.update(visible=False), gr.update(visible=False),
                    "<p style='color:#888;padding:20px'>Chưa có câu hỏi. Tạo quiz trước.</p>",
                    gr.update(choices=["A","B","C","D"], value=None),
                    "", gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False), 0, {},
                )
            labels = get_option_labels(mcqs_json, 0)
            q_html, idx, total, show_prev, show_next = render_quiz_question(mcqs_json, 0)
            show_finish = gr.update(visible=(total == 1))
            show_next_  = gr.update(visible=(total > 1))
            return (
                gr.update(visible=True),   # quiz_screen
                gr.update(visible=False),  # quiz_result_screen
                q_html,
                gr.update(choices=labels, value=None),
                "",         # result_html
                show_prev,
                show_next_,
                show_finish,
                0,          # quiz_idx
                {},         # user_answers
            )

        quiz_start_btn.click(
            start_quiz,
            inputs=[mcqs_state],
            outputs=[quiz_screen, quiz_result_screen, question_html,
                     answer_radio, result_html,
                     prev_btn, next_btn, finish_btn,
                     quiz_idx, user_answers],
        )

        def on_check(mcqs_json, idx, chosen, answers):
            if chosen is None:
                return "", answers, gr.update(visible=False)
            letter = chosen[0]  # "A. text" → "A"
            result = check_answer(mcqs_json, idx, letter)
            answers = dict(answers)
            answers[idx] = letter
            try:
                total = len(json.loads(mcqs_json))
                is_last = (idx >= total - 1)
            except Exception:
                is_last = False
            return (
                result,
                answers,
                gr.update(visible=True),   # next_btn
                gr.update(visible=is_last), # finish_btn
            )

        def on_check_v2(mcqs_json, idx, chosen, answers):
            letter = _extract_option_letter(chosen)
            if letter is None:
                return (
                    "<p style='color:#fca5a5;font-size:13px;margin:8px 0 0'>Vui long chon dap an truoc khi kiem tra.</p>",
                    answers,
                    gr.update(visible=False),
                    gr.update(visible=False),
                )
            result = check_answer(mcqs_json, idx, letter)
            answers = dict(answers)
            answers[idx] = letter
            try:
                total = len(json.loads(mcqs_json))
                is_last = (idx >= total - 1)
            except Exception:
                is_last = False
            return (
                result,
                answers,
                gr.update(visible=not is_last),
                gr.update(visible=is_last),
            )

        check_btn.click(
            on_check_v2,
            inputs=[mcqs_state, quiz_idx, answer_radio, user_answers],
            outputs=[result_html, user_answers, next_btn, finish_btn],
        )

        def go_next(mcqs_json, idx):
            new_idx = idx + 1
            labels  = get_option_labels(mcqs_json, new_idx)
            q_html, new_idx, total, show_prev, show_next = render_quiz_question(mcqs_json, new_idx)
            is_last = (new_idx >= total - 1)
            return (
                q_html,
                gr.update(choices=labels, value=None),
                "",
                show_prev,
                gr.update(visible=not is_last),
                gr.update(visible=is_last),
                new_idx,
            )

        next_btn.click(
            go_next,
            inputs=[mcqs_state, quiz_idx],
            outputs=[question_html, answer_radio, result_html,
                     prev_btn, next_btn, finish_btn, quiz_idx],
        )

        def go_prev(mcqs_json, idx):
            new_idx = max(0, idx - 1)
            labels  = get_option_labels(mcqs_json, new_idx)
            q_html, new_idx, total, show_prev, show_next = render_quiz_question(mcqs_json, new_idx)
            return (
                q_html,
                gr.update(choices=labels, value=None),
                "",
                show_prev,
                show_next,
                gr.update(visible=False),
                new_idx,
            )

        prev_btn.click(
            go_prev,
            inputs=[mcqs_state, quiz_idx],
            outputs=[question_html, answer_radio, result_html,
                     prev_btn, next_btn, finish_btn, quiz_idx],
        )

        def show_score(mcqs_json, answers):
            score = get_final_score(mcqs_json, answers)
            return score

        finish_btn.click(
            show_score,
            inputs=[mcqs_state, user_answers],
            outputs=[score_html],
        ).then(
            lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[quiz_screen, quiz_result_screen],
        )

        def replay(mcqs_json):
            labels = get_option_labels(mcqs_json, 0)
            q_html, _, total, show_prev, show_next = render_quiz_question(mcqs_json, 0)
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                q_html,
                gr.update(choices=labels, value=None),
                "",
                show_prev,
                gr.update(visible=(total > 1)),
                gr.update(visible=(total == 1)),
                0, {},
            )

        replay_btn.click(
            replay,
            inputs=[mcqs_state],
            outputs=[quiz_screen, quiz_result_screen, question_html,
                     answer_radio, result_html,
                     prev_btn, next_btn, finish_btn,
                     quiz_idx, user_answers],
        )

        # Language switch
        def switch_lang(current_lang):
            new_lang = "en" if current_lang == "vi" else "vi"
            btn_label = "🌐 Tiếng Việt" if new_lang == "en" else "🌐 English"
            return new_lang, btn_label

        lang_btn.click(switch_lang,
                       inputs=[lang_state],
                       outputs=[lang_state, lang_btn])

    return demo


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
