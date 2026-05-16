"""
pipeline/mcq_generator.py
--------------------------
Steps 3–7 of the MCQ pipeline: distractor selection, MCQ generation,
validation, and priority ranking.

Single-KU MCQ  : 1 anchor KU + graph-based distractors
Cross-concept  : 2 KUs connected by an edge → tests the relationship

Usage:
    from pipeline.mcq_generator import run_mcq_generation, run_cross_mcq_generation
    mcqs       = run_mcq_generation(pass2_result, llm_fn)
    cross_mcqs = run_cross_mcq_generation(pass2_result, llm_fn)
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.pass2_extractor import Pass2Result


# ── Constants ──────────────────────────────────────────────────────────────

BLOOM_MAP: dict[str, str] = {
    "definition"   : "understand",
    "mechanism"    : "understand",
    "procedure"    : "apply",
    "trade_off"    : "evaluate",
    "failure_mode" : "analyze",
    "application"  : "apply",
}

_BLOOM_META: dict[str, dict] = {
    "remember"  : {"description": "retrieve factual knowledge",
                   "verbs": "recall, identify, list, name"},
    "understand": {"description": "construct meaning from content",
                   "verbs": "explain, describe, summarize, classify"},
    "apply"     : {"description": "carry out a procedure in a given situation",
                   "verbs": "use, apply, demonstrate, implement"},
    "analyze"   : {"description": "break material into parts, detect relationships",
                   "verbs": "distinguish, compare, identify, examine"},
    "evaluate"  : {"description": "make judgments based on criteria and standards",
                   "verbs": "judge, justify, assess, decide"},
    "create"    : {"description": "put elements together to form a coherent whole",
                   "verbs": "design, develop, formulate, construct"},
}

# Cross-concept: edge type → bloom level
_CROSS_BLOOM_MAP: dict[str, str] = {
    "CONTRASTS_WITH" : "evaluate",
    "ALTERNATIVE_TO" : "evaluate",
    "SIMILAR_TO"     : "analyze",
    "SIBLING_OF"     : "analyze",
    "ENABLES"        : "understand",
    "EXTENDS"        : "analyze",
    "APPLIES_TO"     : "apply",
}

_CROSS_RELATION_DESC: dict[str, str] = {
    "CONTRASTS_WITH" : "A and B are opposites or trade-offs",
    "ALTERNATIVE_TO" : "A and B serve the same goal via different approaches",
    "SIMILAR_TO"     : "A and B are similar but subtly different",
    "SIBLING_OF"     : "A and B are parallel members of the same category",
    "ENABLES"        : "A is a prerequisite that makes B possible",
    "EXTENDS"        : "B builds on A as a special case",
    "APPLIES_TO"     : "A is used in the context of B",
}

_HORIZONTAL_EDGES = {"CONTRASTS_WITH", "ALTERNATIVE_TO", "SIMILAR_TO", "SIBLING_OF"}

MIN_EVIDENCE_WORDS = 10
TIER_A_THRESHOLD   = 3
TIER_B_THRESHOLD   = TIER_A_THRESHOLD  # Backward-compatible import name.
MAX_CROSS_MCQS     = 8
MAX_CROSS_PER_KU   = 2


def _norm_concept_name(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", " ", str(text).lower())).strip()


def _lang_name(language: str) -> str:
    return "Vietnamese (Tiếng Việt)" if language == "vi" else "English"


def _language_rules(language: str) -> str:
    lang = _lang_name(language)
    if language == "vi":
        return f"""## LANGUAGE
Write ALL question text, answer options, and explanation in {lang}.
Keep technical terms such as TF-IDF, LSTM, encoder-decoder, and attention unchanged when needed.
Do NOT answer in English."""
    return f"""## LANGUAGE
Write ALL question text, answer options, and explanation in {lang}.
Do NOT mix languages."""


def _explanation_rules(language: str) -> str:
    if language == "vi":
        return """## LEARNER-FACING EXPLANATION
Write the explanation for a learner, not for the test author.
- Use 1-2 concise sentences.
- Explain why the correct option is correct and why the other options are not suitable.
- Do NOT mention: "nguồn", "source", "evidence", "distractor", "phương án gây nhiễu", "gây nhiễu", "KU", "edge", "prompt".
- Do NOT say options were generated from other concepts.
- Prefer wording like: "A đúng vì...", "B/C/D chưa đúng vì..."."""
    return """## LEARNER-FACING EXPLANATION
Write the explanation for a learner, not for the test author.
- Use 1-2 concise sentences.
- Explain why the correct option is correct and why the other options are not suitable.
- Do NOT mention: "source", "evidence", "distractor", "wrong answer candidate", "KU", "edge", "prompt".
- Do NOT say options were generated from other concepts.
- Prefer wording like: "A is correct because...", "B/C/D are not suitable because..."."""


# ── MCQ dataclass ──────────────────────────────────────────────────────────

@dataclass
class MCQItem:
    mcq_id            : str
    anchor_ku_id      : str
    anchor_concept    : str
    anchor_type       : str
    distractor_ku_ids : list[str]      = field(default_factory=list)
    distractor_tier   : str            = "A"
    question          : str            = ""
    options           : dict[str, str] = field(default_factory=dict)
    answer            : str            = ""
    bloom_level       : str            = ""
    explanation       : str            = ""
    priority          : int            = 4
    source_pages      : list[int]      = field(default_factory=list)
    # Cross-concept fields (empty for single-KU MCQs)
    anchor_ku_id_b    : str            = ""
    edge_relation     : str            = ""

    @property
    def ok(self) -> bool:
        return bool(
            self.question
            and self.options
            and self.answer in self.options
            and len(self.options) == 4
        )

    @property
    def is_cross(self) -> bool:
        return bool(self.anchor_ku_id_b)


# ── Step 1: Filter anchor candidates ──────────────────────────────────────

def _filter_anchors(all_kus: list[dict]) -> list[dict]:
    """
    Keep KUs suitable as single-KU MCQ anchors:
      - completeness == "complete"
      - verbatim_evidence >= MIN_EVIDENCE_WORDS words
      (peripheral already dropped in pass2_extractor)
    """
    result = []
    for ku in all_kus:
        if ku.get("completeness") != "complete":
            continue
        if ku.get("prominence") == "peripheral":
            continue
        if len(re.findall(r"\w+", ku.get("verbatim_evidence", ""))) < MIN_EVIDENCE_WORDS:
            continue
        result.append(ku)
    _log(f"[MCQ] anchors after filter: {len(result)}/{len(all_kus)}")
    return result


def _filter_cross_anchors(all_kus: list[dict]) -> set[str]:
    """
    KU ids eligible as cross-MCQ endpoints — more lenient than single-KU filter.
    Cross MCQ tests a relationship, not a specific evidence fragment, so
    evidence length is not required. Only completeness matters.
    """
    return {ku["ku_id"] for ku in all_kus if ku.get("completeness") == "complete"}


# ── Step 4: Bloom level ────────────────────────────────────────────────────

def assign_bloom(ku_type: str) -> str:
    return BLOOM_MAP.get(ku_type, "understand")


# ── Step 7: Priority ───────────────────────────────────────────────────────

def _calc_priority(prominence: str, tier: str) -> int:
    """
    Single-KU MCQs  : 1 (primary+A) → 4 (supporting+B)
    Cross-concept   : 5 (horizontal) | 6 (structural)
    """
    if tier == "C":
        return 5
    p = 0 if prominence == "primary" else 1
    t = 0 if tier == "A"             else 2
    return 1 + p + t


def _cross_priority(relation: str) -> int:
    return 5 if relation in _HORIZONTAL_EDGES else 6


_CROSS_QUALITY_THRESHOLD: dict[str, int] = {
    "CONTRASTS_WITH": 35,
    "ALTERNATIVE_TO": 35,
    "SIMILAR_TO": 50,
    "SIBLING_OF": 50,
    "ENABLES": 35,
    "EXTENDS": 35,
    "APPLIES_TO": 35,
}


def _tokens(text: str) -> set[str]:
    def _depl(t: str) -> str:
        return t.rstrip("s") if len(t) > 3 else t
    return {_depl(t) for t in re.findall(r"\w+", _norm_concept_name(text)) if t}


def _field_match(value: str, target: str) -> bool:
    vt = _tokens(value)
    tt = _tokens(target)
    return bool(vt and tt and (vt <= tt or tt <= vt or len(vt & tt) >= 2))


def _owner_name(ku: dict) -> str:
    return _norm_concept_name(ku.get("owner_concept") or ku.get("parent_l2") or "")


def _is_broad_representative(ku: dict) -> bool:
    """True when the KU concept looks representative of its owner, not a narrow child."""
    concept = ku.get("concept", "")
    local = ku.get("local_concept", "")
    owner = ku.get("owner_concept") or ku.get("parent_l2") or ""
    return (
        _field_match(concept, owner)
        or _field_match(local, owner)
        or ku.get("type") in {"definition", "mechanism", "trade_off"}
    )


def _cross_question_mode(ku_a: dict, ku_b: dict, relation: str) -> tuple[str, str]:
    type_a = ku_a.get("type", "")
    type_b = ku_b.get("type", "")
    pair = {type_a, type_b}

    if relation in {"CONTRASTS_WITH", "ALTERNATIVE_TO"} and "failure_mode" in pair:
        return (
            "limitation",
            "Ask how one concept's mechanism/approach relates to a limitation, risk, or failure mode. Do not force a direct A-vs-B contrast.",
        )
    if relation == "ALTERNATIVE_TO":
        return (
            "alternative",
            "Ask the learner to distinguish when one approach or procedure is more appropriate than the other.",
        )
    if relation == "CONTRASTS_WITH":
        return (
            "compare",
            "Ask for the key conceptual difference or trade-off between the two concepts.",
        )
    if relation in {"SIBLING_OF", "SIMILAR_TO"}:
        return (
            "distinguish",
            "Ask the learner to distinguish two related concepts that are easy to confuse.",
        )
    if relation == "APPLIES_TO":
        return (
            "application",
            "Ask how one concept is used within, or applied to, the other concept.",
        )
    if relation in {"EXTENDS", "ENABLES"}:
        return (
            "builds_on",
            "Ask how one concept builds on, enables, or makes the other concept possible.",
        )
    return (
        "connection",
        "Ask about the meaningful connection between the two concepts.",
    )


def _cross_edge_quality(ku_a: dict, ku_b: dict, relation: str) -> tuple[int, str]:
    """
    Heuristic gate before spending an LLM call on cross-MCQ generation.
    The goal is not to prove correctness, just to reject obviously weak
    representative pairs such as mechanism-vs-random failure-mode.
    """
    type_a = ku_a.get("type", "")
    type_b = ku_b.get("type", "")
    owner_a = _owner_name(ku_a)
    owner_b = _owner_name(ku_b)
    l1_a = _norm_concept_name(ku_a.get("parent_l1", ""))
    l1_b = _norm_concept_name(ku_b.get("parent_l1", ""))

    score = 0
    reasons: list[str] = []

    if owner_a and owner_b and owner_a != owner_b:
        score += 15
        reasons.append("different-owner")
    if l1_a and l1_b and l1_a == l1_b:
        score += 10
        reasons.append("same-l1")
    if ku_a.get("prominence") == "primary":
        score += 6
    if ku_b.get("prominence") == "primary":
        score += 6

    broad_a = _is_broad_representative(ku_a)
    broad_b = _is_broad_representative(ku_b)
    if broad_a:
        score += 10
    if broad_b:
        score += 10

    if relation in {"CONTRASTS_WITH", "ALTERNATIVE_TO"}:
        score += 20
        if type_a in {"definition", "mechanism", "trade_off"} and type_b in {"definition", "mechanism", "trade_off"}:
            score += 15
            reasons.append("comparable-types")
        if "failure_mode" in {type_a, type_b}:
            score += 12
            reasons.append("limitation-mode")
        if not (broad_a and broad_b):
            score -= 4
            reasons.append("weak-representative")
    elif relation == "SIBLING_OF":
        score += 10
        if l1_a and l1_a == l1_b and owner_a and owner_b and owner_a != owner_b:
            score += 25
            reasons.append("true-sibling")
        if type_a == type_b:
            score += 15
            reasons.append("same-type")
    elif relation == "SIMILAR_TO":
        score += 12
        if type_a == type_b:
            score += 15
            reasons.append("same-type")
        if l1_a and l1_a == l1_b:
            score += 15
            reasons.append("same-l1")
    else:
        score += 18
        if owner_a != owner_b:
            score += 10
        if type_a in {"definition", "mechanism", "procedure", "application"}:
            score += 6
        if type_b in {"definition", "mechanism", "procedure", "application"}:
            score += 6

    return score, ",".join(reasons) or "base"


def _tier_for_distractors(n_distractors: int) -> str:
    if n_distractors >= TIER_A_THRESHOLD:
        return "A"
    if n_distractors > 0:
        return "B"
    return "C"


# ── Step 5: Prompt builders ────────────────────────────────────────────────

def _build_tier_a_prompt(anchor: dict, distractors: list[dict], bloom_level: str, language: str = "en") -> str:
    meta   = _BLOOM_META.get(bloom_level, _BLOOM_META["understand"])
    d_lines = "\n".join(
        f"{i}. Concept: {d['concept']} | Type: {d['type']}\n"
        f"   Content: {d['content']}"
        for i, d in enumerate(distractors, 1)
    )
    return f"""## TESTED CONCEPT MATERIAL
Concept: {anchor['concept']}
Type: {anchor['type']}
Content: {anchor['content']}
Quote: {anchor['verbatim_evidence']}

## PLAUSIBLE WRONG-OPTION MATERIAL
{d_lines}

{_language_rules(language)}

{_explanation_rules(language)}

## TASK
Bloom level: {bloom_level}
Cognitive operation: {meta['description']}
Stem verbs to use: {meta['verbs']}

Generate 1 MCQ:
- Correct answer = reformulation of the tested concept material
- Wrong answers = plausible but incorrect alternatives based on the wrong-option material
- Do NOT mention concept names in options
- Options must be mutually exclusive
- Question must be answerable from the tested concept material and quote alone
- If fewer than 3 wrong-option materials are listed above, invent 1 additional plausible wrong answer

## OUTPUT (JSON only)
{{
  "question": "...",
  "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
  "answer": "A|B|C|D",
  "bloom_level": "{bloom_level}",
  "explanation": "learner-facing explanation with no internal wording"
}}"""


def _build_tier_b_prompt(anchor: dict, distractors: list[dict], bloom_level: str, language: str = "en") -> str:
    meta = _BLOOM_META.get(bloom_level, _BLOOM_META["understand"])
    d_lines = "\n".join(
        f"{i}. Concept: {d['concept']} | Type: {d['type']}\n"
        f"   Content: {d['content']}"
        for i, d in enumerate(distractors, 1)
    )
    return f"""## CORRECT ANSWER MATERIAL
Concept: {anchor['concept']}
Type: {anchor['type']}
Correct idea: {anchor['content']}
Grounding quote: {anchor['verbatim_evidence']}

## PROVIDED WRONG-OPTION ANCHORS
{d_lines}

{_language_rules(language)}

{_explanation_rules(language)}

## TASK
Bloom level: {bloom_level}
Cognitive operation: {meta['description']}
Stem verbs to use: {meta['verbs']}

Generate 1 MCQ with 3 plausible but incorrect alternatives.
- The correct option must be a clean reformulation of the Correct idea.
- Use the provided wrong-option anchors as examples of nearby-but-wrong ideas.
- Generate any missing wrong options by contrasting the Correct idea against those anchors.
- Missing wrong options should be plausible misconceptions from the same topic family, not random facts.
- All wrong options must be semantically relevant and clearly distinct from the correct option.
- Do NOT mention: "source", "evidence", "distractor", "wrong-option anchor", "KU", "edge", or "prompt".
- Do NOT reveal that some wrong options were invented.

## OUTPUT (JSON only)
{{
  "question": "...",
  "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
  "answer": "A|B|C|D",
  "bloom_level": "{bloom_level}",
  "explanation": "learner-facing explanation with no internal wording"
}}"""


def _build_tier_c_prompt(anchor: dict, bloom_level: str, language: str = "en") -> str:
    meta = _BLOOM_META.get(bloom_level, _BLOOM_META["understand"])
    return f"""## TESTED CONCEPT MATERIAL
Concept: {anchor['concept']}
Type: {anchor['type']}
Content: {anchor['content']}
Quote: {anchor['verbatim_evidence']}

{_language_rules(language)}

{_explanation_rules(language)}

## TASK
Bloom level: {bloom_level}
Cognitive operation: {meta['description']}
Stem verbs to use: {meta['verbs']}

Generate 1 MCQ with 3 plausible but incorrect alternatives.
- No reliable related concepts are available, so wrong options must be common misconceptions about {anchor['concept']}.
- Keep wrong options close to the tested concept, not random facts from another topic.
- Do NOT mention concept names in options.

## OUTPUT (JSON only)
{{
  "question": "...",
  "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
  "answer": "A|B|C|D",
  "bloom_level": "{bloom_level}",
  "explanation": "learner-facing explanation with no internal wording"
}}"""


def _build_cross_prompt(ku_a: dict, ku_b: dict, relation: str, bloom_level: str, language: str = "en") -> str:
    meta     = _BLOOM_META.get(bloom_level, _BLOOM_META["analyze"])
    rel_desc = _CROSS_RELATION_DESC.get(relation, "two related concepts")
    mode, mode_guidance = _cross_question_mode(ku_a, ku_b, relation)
    return f"""## CONCEPT PAIR MATERIAL
Concept A: {ku_a['concept']} | Type: {ku_a['type']}
Content A: {ku_a['content']}
Quote A: {ku_a['verbatim_evidence']}

Concept B: {ku_b['concept']} | Type: {ku_b['type']}
Content B: {ku_b['content']}
Quote B: {ku_b['verbatim_evidence']}

Relationship: {relation} — {rel_desc}
Question mode: {mode}
Mode guidance: {mode_guidance}

## TASK
Bloom level: {bloom_level}
Cognitive operation: {meta['description']}
Stem verbs to use: {meta['verbs']}

{_language_rules(language)}

{_explanation_rules(language)}

Generate 1 natural MCQ for a learner that tests how the two concepts connect.
- Use concept names that a learner recognizes: "{ku_a['concept']}" and "{ku_b['concept']}".
- Do NOT mention internal labels like "{relation}", "KU", "edge", or "relationship type" in the question.
- The question should sound like a teacher asking about the material, not a graph database.
- Follow the Question mode and Mode guidance above.
- If the mode is limitation, ask about cause, limitation, consequence, or why one concept motivates/relates to the other; do not ask a forced direct contrast.
- If the mode is compare, distinguish, or alternative, ask for the key difference or appropriate use case.
- Correct answer = the best explanation of the connection from the concept descriptions.
- Wrong answers = plausible but wrong descriptions (wrong direction, wrong mechanism, overclaim, unrelated use).
- Options must be mutually exclusive and should not all repeat the concept names.

## OUTPUT (JSON only)
{{
  "question": "...",
  "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
  "answer": "A|B|C|D",
  "bloom_level": "{bloom_level}",
  "explanation": "learner-facing explanation with no internal wording"
}}"""


# ── Step 5+6: Parse + validate ─────────────────────────────────────────────

def _parse_mcq_response(raw: str) -> dict | None:
    text = raw.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _sanitize_explanation(text: str) -> str:
    """Remove test-author/internal wording that sometimes leaks into explanations."""
    if not text:
        return ""
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
    cleaned = str(text)
    for old, new in replacements:
        cleaned = re.sub(re.escape(old), new, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _normalize_mcq_data(data: dict) -> dict:
    """Keep the schema strict while tolerating harmless LLM extras like option E."""
    if not isinstance(data, dict):
        return data
    out = dict(data)
    opts = out.get("options", {})
    if isinstance(opts, dict):
        normalized = {str(k).strip().upper(): str(v).strip() for k, v in opts.items()}
        if {"A", "B", "C", "D"} <= set(normalized):
            out["options"] = {k: normalized[k] for k in ["A", "B", "C", "D"]}
    out["answer"] = str(out.get("answer", "")).strip().upper()
    out["explanation"] = _sanitize_explanation(str(out.get("explanation", "") or ""))
    return out


def _validate_mcq(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    q    = data.get("question", "").strip()
    opts = data.get("options", {})
    ans  = data.get("answer", "").strip().upper()
    if not q:
        return False
    if not isinstance(opts, dict) or set(opts.keys()) != {"A", "B", "C", "D"}:
        return False
    if ans not in {"A", "B", "C", "D"}:
        return False
    values = [v.strip() for v in opts.values()]
    if any(not v for v in values):
        return False
    if len({v.lower() for v in values}) < 4:
        return False
    return True


_ANSWER_LETTERS = ("A", "B", "C", "D")


def _target_answer_letter(index: int) -> str:
    """Deterministic A/B/C/D rotation for the final accepted MCQ list."""
    return _ANSWER_LETTERS[index % len(_ANSWER_LETTERS)]


def _remap_explanation_letters(text: str, letter_map: dict[str, str]) -> str:
    """Remap standalone answer letters in learner-facing explanations."""
    if not text or not letter_map:
        return text
    out = str(text)
    placeholders: dict[str, str] = {}
    for src in letter_map:
        token = f"__MCQ_LETTER_{src}__"
        placeholders[src] = token
        out = re.sub(
            rf"(?<![A-Za-z0-9]){re.escape(src)}(?![A-Za-z0-9])",
            token,
            out,
        )
    for src, token in placeholders.items():
        out = out.replace(token, letter_map[src])
    return out


def _rebalance_answer_slot(data: dict, target_letter: str) -> dict:
    """
    Move the correct option to target_letter by swapping option texts.
    This prevents LLMs from putting almost every correct answer in A while
    preserving exactly four options and the original correct option text.
    """
    out = dict(data)
    opts = dict(out.get("options", {}) or {})
    old_letter = str(out.get("answer", "")).strip().upper()
    target_letter = str(target_letter or "").strip().upper()
    if old_letter not in opts or target_letter not in opts or old_letter == target_letter:
        return out

    opts[old_letter], opts[target_letter] = opts[target_letter], opts[old_letter]
    out["options"] = opts
    out["answer"] = target_letter
    out["explanation"] = _remap_explanation_letters(
        out.get("explanation", ""),
        {old_letter: target_letter, target_letter: old_letter},
    )
    return out


def _build_json_repair_prompt(raw: str) -> str:
    return f"""Repair the following malformed MCQ response into ONLY valid JSON.

Rules:
- Return exactly one JSON object.
- Keep the original question intent and option meanings.
- Use exactly these keys: question, options, answer, explanation.
- options must contain exactly A, B, C, D.
- answer must be one of A, B, C, D.
- No markdown, no commentary.

Malformed response:
{raw[:4000]}
"""


def _invoke_llm(llm_fn, prompt: str, json_mode: bool, temperature: float | None = None) -> str:
    """Call either a 2-arg legacy llm_fn or a 3-arg temperature-aware llm_fn."""
    if temperature is None:
        return llm_fn(prompt, json_mode)
    try:
        return llm_fn(prompt, json_mode, temperature)
    except TypeError:
        return llm_fn(prompt, json_mode)


def _call_llm(prompt: str, context_id: str, llm_fn) -> dict | None:
    """Shared: call LLM → parse JSON → validate. Returns mcq_data or None."""
    try:
        raw = _invoke_llm(llm_fn, prompt, False)
        if not raw or not raw.strip():
            _log(f"  [MCQ] empty response for {context_id}")
            return None
        data = _parse_mcq_response(raw)
        if data is None:
            _log(f"  [MCQ] JSON parse fail for {context_id}")
            repaired_raw = _invoke_llm(
                llm_fn,
                _build_json_repair_prompt(raw),
                False,
                temperature=0.0,
            )
            data = _parse_mcq_response(repaired_raw or "")
            if data is None:
                _log(f"  [MCQ] JSON repair fail for {context_id}")
                return None
            _log(f"  [MCQ] JSON repair ok for {context_id}")
        data = _normalize_mcq_data(data)
        if not _validate_mcq(data):
            _log(
                f"  [MCQ] validation fail for {context_id}: "
                f"q={bool(data.get('question'))} "
                f"opts={set(data.get('options', {}).keys())} "
                f"ans={data.get('answer')!r}"
            )
            return None
        return data
    except Exception as e:
        _log(f"  [MCQ] error {context_id}: {str(e)[:100]}")
        return None


# ── Step 5: Single MCQ ────────────────────────────────────────────────────

def generate_mcq(
    anchor: dict,
    distractors: list[dict],
    bloom_level: str,
    llm_fn,
    language: str = "en",
) -> tuple[dict | None, str]:
    """Returns (mcq_data, tier) where tier = "A" | "B" | "C". (None, "") on failure."""
    tier = _tier_for_distractors(len(distractors))
    if tier == "A":
        prompt = _build_tier_a_prompt(anchor, distractors[:3], bloom_level, language)
    elif tier == "B":
        prompt = _build_tier_b_prompt(anchor, distractors[:2], bloom_level, language)
    else:
        prompt = _build_tier_c_prompt(anchor, bloom_level, language)
    data = _call_llm(prompt, anchor.get("ku_id", "?"), llm_fn)
    return (data, tier) if data else (None, "")


# ── Step 5: Cross-concept MCQ ─────────────────────────────────────────────

def generate_cross_mcq(
    ku_a: dict,
    ku_b: dict,
    relation: str,
    bloom_level: str,
    llm_fn,
    language: str = "en",
) -> dict | None:
    """Returns mcq_data or None."""
    prompt = _build_cross_prompt(ku_a, ku_b, relation, bloom_level, language)
    return _call_llm(prompt, f"{ku_a['ku_id']}↔{ku_b['ku_id']}", llm_fn)


# ── Main: single-KU MCQ generation ────────────────────────────────────────

def run_mcq_generation(
    pass2: Pass2Result,
    llm_fn,
    delay_between: float = 2.0,
    language: str = "en",
) -> list[MCQItem]:
    """
    Generate single-KU MCQs from Pass2Result.
    Returns list sorted by priority ascending (1 = best).
    """
    if not pass2.ok:
        _log("[MCQ] empty Pass2Result — skip")
        return []

    anchors = _filter_anchors(pass2.all_kus)
    if not anchors:
        _log("[MCQ] no valid anchors after filter")
        return []

    mcqs: list[MCQItem] = []

    for i, anchor in enumerate(anchors, 1):
        ku_id      = anchor["ku_id"]
        bloom_lvl  = assign_bloom(anchor["type"])
        prominence = anchor.get("prominence", "supporting")

        _log(f"\n[MCQ] [{i}/{len(anchors)}] {ku_id} — {anchor['concept'][:40]}")
        _log(f"  type={anchor['type']}  bloom={bloom_lvl}  prominence={prominence}")

        distractor_kus = pass2.get_distractors(ku_id, n=3)
        tier_label     = _tier_for_distractors(len(distractor_kus))
        _log(f"  distractors: {len(distractor_kus)} → Tier {tier_label}")

        for j, d in enumerate(distractor_kus, 1):
            source = d.get("_distractor_source", "unknown")
            rel    = d.get("_distractor_relation") or "-"
            score  = d.get("_distractor_score", "?")
            _log(
                f"    [{j}] {str(d.get('concept', ''))[:34]} | "
                f"type={d.get('type', '?')} | source={source} | rel={rel} | score={score}"
            )

        mcq_data, tier = generate_mcq(anchor, distractor_kus, bloom_lvl, llm_fn, language)
        if mcq_data is None:
            _log(f"  [MCQ] skip {ku_id}")
            if i < len(anchors):
                time.sleep(delay_between)
            continue
        target_answer = _target_answer_letter(len(mcqs))
        old_answer = mcq_data.get("answer", "").upper()
        mcq_data = _rebalance_answer_slot(mcq_data, target_answer)
        if old_answer != mcq_data.get("answer"):
            _log(f"  answer slot: {old_answer} → {mcq_data.get('answer')}")

        mcqs.append(MCQItem(
            mcq_id            = f"{ku_id}_{bloom_lvl}",
            anchor_ku_id      = ku_id,
            anchor_concept    = anchor["concept"],
            anchor_type       = anchor["type"],
            distractor_ku_ids = [d["ku_id"] for d in distractor_kus],
            distractor_tier   = tier,
            question          = mcq_data["question"],
            options           = mcq_data["options"],
            answer            = mcq_data["answer"].upper(),
            bloom_level       = mcq_data.get("bloom_level", bloom_lvl),
            explanation       = mcq_data.get("explanation", ""),
            priority          = _calc_priority(prominence, tier),
            source_pages      = anchor.get("source_pages", []),
        ))
        _log(f"  ok — tier={tier}  priority={mcqs[-1].priority}")

        if i < len(anchors):
            time.sleep(delay_between)

    mcqs.sort(key=lambda m: m.priority)
    n_a = sum(1 for m in mcqs if m.distractor_tier == "A")
    n_b = sum(1 for m in mcqs if m.distractor_tier == "B")
    n_c = sum(1 for m in mcqs if m.distractor_tier == "C")
    _log(f"\n[MCQ] done: {len(mcqs)}/{len(anchors)}  (Tier A={n_a}  Tier B={n_b}  Tier C={n_c})")
    return mcqs


# ── Main: cross-concept MCQ generation ────────────────────────────────────

def run_cross_mcq_generation(
    pass2: Pass2Result,
    llm_fn,
    delay_between: float = 2.0,
    horizontal_only: bool = True,
    language: str = "en",
    max_questions: int | None = MAX_CROSS_MCQS,
    max_per_ku: int = MAX_CROSS_PER_KU,
    answer_offset: int = 0,
) -> list[MCQItem]:
    """
    Generate cross-concept MCQs from graph edges in Pass2Result.

    Args:
        horizontal_only : if True (default) only CONTRASTS_WITH / ALTERNATIVE_TO /
                          SIMILAR_TO / SIBLING_OF edges are used. Set False to also
                          include ENABLES / EXTENDS / APPLIES_TO.

    Returns list sorted by priority ascending (5 = horizontal, 6 = structural).
    """
    if not pass2.ok:
        _log("[CrossMCQ] empty Pass2Result — skip")
        return []

    anchor_ids = _filter_cross_anchors(pass2.all_kus)
    ku_by_id   = {ku["ku_id"]: ku for ku in pass2.all_kus}

    _log(f"[CrossMCQ] total edges in graph: {len(pass2.edge_types)}")
    _log(f"[CrossMCQ] anchor_ids (complete KUs): {len(anchor_ids)}")

    # Collect eligible edges (each edge stored once: a < b)
    edges: list[tuple[str, str, str]] = []
    n_skip_structural = 0
    n_skip_anchor     = 0
    n_skip_stale      = 0
    n_skip_same       = 0
    n_skip_duplicate  = 0
    n_skip_per_ku     = 0
    n_skip_quality    = 0
    seen_pairs: set[tuple[str, str, str]] = set()
    per_ku_counts: dict[str, int] = {}
    for (id_a, id_b), relation in pass2.edge_types.items():
        if horizontal_only and relation not in _HORIZONTAL_EDGES:
            n_skip_structural += 1
            continue
        if id_b not in pass2.graph.get(id_a, []) or id_a not in pass2.graph.get(id_b, []):
            n_skip_stale += 1
            continue
        if id_a not in anchor_ids or id_b not in anchor_ids:
            missing = [x for x in (id_a, id_b) if x not in anchor_ids]
            _log(f"  [CrossMCQ] skip edge [{relation}] {id_a}↔{id_b} — "
                 f"non-anchor: {missing}")
            n_skip_anchor += 1
            continue
        ku_a = ku_by_id.get(id_a, {})
        ku_b = ku_by_id.get(id_b, {})
        ca = _norm_concept_name(ku_a.get("concept", ""))
        cb = _norm_concept_name(ku_b.get("concept", ""))
        if not ca or not cb or ca == cb:
            n_skip_same += 1
            continue
        quality, quality_reason = _cross_edge_quality(ku_a, ku_b, relation)
        min_quality = _CROSS_QUALITY_THRESHOLD.get(relation, 50)
        if quality < min_quality:
            _log(
                f"  [CrossMCQ] skip edge [{relation}] {id_a}â†”{id_b} â€” "
                f"quality={quality}<{min_quality} ({quality_reason})"
            )
            n_skip_quality += 1
            continue
        pair_key = (min(ca, cb), max(ca, cb), relation)
        if pair_key in seen_pairs:
            n_skip_duplicate += 1
            continue
        if max_per_ku and (
            per_ku_counts.get(id_a, 0) >= max_per_ku or
            per_ku_counts.get(id_b, 0) >= max_per_ku
        ):
            n_skip_per_ku += 1
            continue
        seen_pairs.add(pair_key)
        per_ku_counts[id_a] = per_ku_counts.get(id_a, 0) + 1
        per_ku_counts[id_b] = per_ku_counts.get(id_b, 0) + 1
        edges.append((id_a, id_b, relation))
        if max_questions and len(edges) >= max_questions:
            break

    _log(f"[CrossMCQ] eligible edges: {len(edges)}  "
         f"(skipped structural={n_skip_structural}  non-anchor={n_skip_anchor})")
    if n_skip_stale or n_skip_same or n_skip_duplicate or n_skip_per_ku or n_skip_quality:
        _log(
            f"[CrossMCQ] cleanup skipped: stale={n_skip_stale}  "
            f"same={n_skip_same}  duplicate={n_skip_duplicate}  "
            f"per_ku={n_skip_per_ku}  quality={n_skip_quality}"
        )
    if not edges:
        return []

    mcqs: list[MCQItem] = []

    for i, (id_a, id_b, relation) in enumerate(edges, 1):
        ku_a      = ku_by_id[id_a]
        ku_b      = ku_by_id[id_b]
        bloom_lvl = _CROSS_BLOOM_MAP.get(relation, "analyze")

        _log(
            f"\n[CrossMCQ] [{i}/{len(edges)}] "
            f"{ku_a['concept'][:25]} ↔ {ku_b['concept'][:25]}  [{relation}]"
        )
        _log(f"  bloom={bloom_lvl}")

        mcq_data = generate_cross_mcq(ku_a, ku_b, relation, bloom_lvl, llm_fn, language)
        if mcq_data is None:
            _log(f"  [CrossMCQ] skip {id_a}↔{id_b}")
            if i < len(edges):
                time.sleep(delay_between)
            continue
        target_answer = _target_answer_letter(answer_offset + len(mcqs))
        old_answer = mcq_data.get("answer", "").upper()
        mcq_data = _rebalance_answer_slot(mcq_data, target_answer)
        if old_answer != mcq_data.get("answer"):
            _log(f"  answer slot: {old_answer} → {mcq_data.get('answer')}")

        pages = sorted(set(ku_a.get("source_pages", []) + ku_b.get("source_pages", [])))
        mcqs.append(MCQItem(
            mcq_id            = f"{id_a}_X_{id_b}_{relation.lower()}",
            anchor_ku_id      = id_a,
            anchor_concept    = ku_a["concept"],
            anchor_type       = ku_a["type"],
            anchor_ku_id_b    = id_b,
            edge_relation     = relation,
            distractor_tier   = "A",
            question          = mcq_data["question"],
            options           = mcq_data["options"],
            answer            = mcq_data["answer"].upper(),
            bloom_level       = mcq_data.get("bloom_level", bloom_lvl),
            explanation       = mcq_data.get("explanation", ""),
            priority          = _cross_priority(relation),
            source_pages      = pages,
        ))
        _log(f"  ok — priority={mcqs[-1].priority}")

        if i < len(edges):
            time.sleep(delay_between)

    mcqs.sort(key=lambda m: m.priority)
    _log(f"\n[CrossMCQ] done: {len(mcqs)}/{len(edges)}")
    return mcqs


# ── Utility ────────────────────────────────────────────────────────────────

def _log(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))
