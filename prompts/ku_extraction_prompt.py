"""
prompts/ku_extraction_prompt.py
--------------------------------
Prompt để LLM extract Knowledge Units từ 1 chunk.
"""

from __future__ import annotations

import json

_KU_TYPES = """
- definition    : what X is, what X consists of              (Bloom: remember, understand)
- mechanism     : how X works, sequence, flow                (Bloom: understand, apply)
- failure_mode  : when X fails, limitations, edge cases      (Bloom: analyze, evaluate)
- trade_off     : X vs Y, when to use which, decisions       (Bloom: evaluate, create)
- procedure     : steps, algorithms, formulas, pseudocode    (Bloom: apply, analyze)
- application   : real-world scenario where X is used        (Bloom: apply)
""".strip()

_FAILURE_MODE_HINTS = """
Failure mode signal patterns — create a failure_mode KU when you detect:
  Vietnamese: "khong hoat dong khi", "han che", "kho toi uu", "de bi gioi han",
              "chi phu hop voi", "yeu cau", "khong the", "khong nen", "van de"
  English   : "fails when", "limited to", "cannot", "expensive for",
              "memory bound", "not suitable for", "requires", "overhead", "drawback"
If you detect such a signal but no failure_mode KU exists for that concept → create one.
""".strip()

_EDGE_RELATIONS = """
HORIZONTAL edges — use these when two concepts are confusable or comparable:
  CONTRASTS_WITH : A and B are opposites or trade-offs (e.g. Supervisor ↔ Parallel)
  ALTERNATIVE_TO : A and B serve the same goal via different approaches
  SIBLING_OF     : A and B are parallel members of the same category
  SIMILAR_TO     : A and B are easily confused / look or sound similar

STRUCTURAL edges — use these for dependencies or hierarchy:
  PART_OF        : A is a component or sub-step of B
  ENABLES        : A is a prerequisite that makes B possible
  EXTENDS        : A builds on B / is a special case of B
  APPLIES_TO     : A is used in the context of B

Prefer HORIZONTAL over STRUCTURAL when both could apply.
""".strip()

_PROMINENCE = """
primary    = this concept is the main topic of this chunk (usually 1–3 per chunk)
supporting = this concept is explained in direct relation to a primary concept
peripheral = this concept is only mentioned by name, not elaborated upon
""".strip()

_COMPLETENESS = """
Mark "incomplete" if ANY of these apply:
  1. The content field would be fewer than 15 words
  2. The concept is mentioned by name only — no definition, mechanism, or evidence given
  3. The concept requires significant external knowledge (not present in source) to be understandable
""".strip()

_FILL_GAP = """
When source content is implicit, use exactly ONE technique:

1. Reformulate — restate with explicit subject + predicate:
   Source:  "Thêm evaluator vào ReAct"
   Content: "Reflexion extends ReAct by adding an Evaluator component"

2. Combine — merge 2–3 scattered fragments into 1 unit:
   Source A: "LSTM có cell state"
   Source B: "cell state mang thông tin dài hạn"
   Content:  "LSTM uses a cell state to carry long-term information across time steps"
   verbatim_evidence: "LSTM có cell state [...] cell state mang thông tin dài hạn"
   → Use "[...]" separator when evidence spans non-adjacent text

3. Convert — turn implicit fact into an explicit, testable statement:
   Source:  "ReAct không detect lỗi"
   Content: "ReAct lacks a self-evaluation mechanism, causing errors to propagate"
""".strip()

_CRITICAL = """
- verbatim_evidence MUST be copied EXACTLY from the source text — no paraphrasing
- For Combine: copy both fragments verbatim, with "[...]" between them
- content must be directly derivable from verbatim_evidence — never add external facts
- Do NOT create application KUs whose content references external knowledge not in the source
""".strip()

_SKIP_PEDAGOGY = """
## DO NOT EXTRACT — Pedagogical content (not domain knowledge)
Skip any passage that describes classroom logistics rather than the subject matter itself.
Signal patterns to SKIP:
  - Group/individual activity instructions: "chia nhóm", "mỗi nhóm nhận", "thảo luận nhóm",
    "card sorting", "case study exercise", "bài tập", "làm việc nhóm"
  - In-class exercises or quizzes: "exercise", "quiz", "practice round", "hands-on"
  - Grading / assignment instructions: "nộp bài", "điểm", "rubric", "submission"
  - Slide meta-content: "learning objectives", "mục tiêu bài học", "agenda", "outline"
  - Pure examples with no transferable rule: "ví dụ: ...<example only, no generalisation>"
If the passage contains BOTH a pedagogy instruction AND real domain content, extract the
domain content only — set concept to the domain concept, not the activity name.
""".strip()


_INTRA_EDGE_RULES = """
## INTRA-SEGMENT EDGE RULES
Sau khi extract xong tất cả KUs, với mỗi KU hỏi:
"Có KU nào khác mà học sinh dễ nhầm với KU này không?"

Chỉ tạo edge nếu CẢ BA điều kiện đúng:
  1. Có verbatim evidence mention CẢ HAI concepts
  2. Relation type phải là horizontal: CONTRASTS_WITH, ALTERNATIVE_TO, SIMILAR_TO
  3. Evidence phải là explicit hoặc discourse_marker — KHÔNG dùng co_occurrence

Cap per relation type (tổng mỗi KU):
  CONTRASTS_WITH : tối đa 2
  ALTERNATIVE_TO : tối đa 2
  SIMILAR_TO     : tối đa 1
""".strip()


def _build_scope_section(concept_name: str, parent_name: str,
                         l1_seen: bool = False) -> str:
    if not concept_name:
        return ""

    if not parent_name:
        return f"""
## SCOPE CONSTRAINT (STRICT)
Segment này về: "{concept_name}"

Chỉ extract KUs về "{concept_name}" và các sub-aspects trực tiếp.
Nếu text mention concept khác dưới 3 câu → đó là context, KHÔNG extract thành KU.
"""

    if l1_seen:
        return f"""
## SCOPE CONSTRAINT
Segment này về: "{concept_name}" (thuộc "{parent_name}")

"{parent_name}" KUs đã được extract ở segment trước — KHÔNG extract lại.
Chỉ extract KUs về "{concept_name}".
Không extract content về các L2 siblings của "{concept_name}" trong "{parent_name}".
"""
    else:
        return f"""
## SCOPE CONSTRAINT
Segment này về: "{concept_name}" (L2 của "{parent_name}")

Extract KUs về:
  1. "{concept_name}" → gán concept = "{concept_name}"
  2. "{parent_name}" nếu text có content mới chưa extract ở segment trước
     → gán concept = "{parent_name}" (KHÔNG gán concept = "{concept_name}")

KHÔNG extract:
  - Content về L2 siblings của "{concept_name}" (các concepts khác cùng cấp trong "{parent_name}")
  - Nếu "{parent_name}" chỉ được mention < 3 câu → đó là context, bỏ qua
"""


def _build_cross_context(cross_rels: list[dict] | None) -> str:
    if not cross_rels:
        return ""
    lines = [
        "\n## CROSS-SEGMENT CONNECTIONS (document-level context)",
        "These concepts from this segment are connected to other parts of the document.",
        "Add their names to related_kus of relevant KUs if they appear in this segment's text.",
    ]
    for r in cross_rels:
        ev = r.get("evidence", "")[:80]
        lines.append(
            f"  [{r.get('relation','?')}] {r.get('from_concept','')} → "
            f"{r.get('to_concept','')}  |  \"{ev}\""
        )
    return "\n".join(lines) + "\n"


def _build_doc_concepts_section(
    doc_concepts: dict | None,
) -> str:
    if not doc_concepts:
        return ""
    seg_concepts = doc_concepts.get("seg") or []
    all_concepts = doc_concepts.get("all") or []

    seg_lines = "\n".join(f"  - {c}" for c in seg_concepts)
    all_lines = "\n".join(f"  - {c}" for c in all_concepts)

    return f"""
## DOCUMENT CONCEPTS — CONSTRAINT
Segment concepts (use for `concept` field — copy EXACTLY, no paraphrasing):
{seg_lines}

Full document concept list (use for related_kus.target_concept):
{all_lines}

CRITICAL NAMING RULE:
- concept field MUST be one of the segment concepts listed above
- If you discover a concept not in any list → output it in "new_concepts" field
- NEVER invent a new concept name variation — use exact names from the list
"""


def _build_recap_section(seen_concepts: list[str] | None) -> str:
    if not seen_concepts:
        return ""
    seen_lines = "\n".join(f"  - {c}" for c in seen_concepts)
    return f"""
## RECAP SEGMENT — EXTRACT NEW INFORMATION ONLY
Concepts already covered in previous segments:
{seen_lines}
Extract a KU ONLY if it contains information not already covered by these concepts.
If this segment only restates existing knowledge, output an empty list [].
"""


def _build_scope_section(concept_name: str, parent_name: str,
                         l1_seen: bool = False) -> str:
    """Ownership-aware scope rules. Defined late to override legacy wording above."""
    if not concept_name:
        return ""

    if not parent_name:
        return f"""
## SCOPE CONSTRAINT (STRICT)
Current segment topic: "{concept_name}"

Extract KUs about "{concept_name}" and direct sub-aspects only.
Ownership:
  - owner_level = "L1"
  - owner_concept = "{concept_name}"
  - parent_l1 = "{concept_name}"
  - parent_l2 = ""
Use `concept` / `local_concept` for the actual local concept being tested.
If another concept is mentioned briefly as context, do NOT extract it as a KU.
"""

    seen_note = (
        f'Parent "{parent_name}" may already have KUs from earlier sibling segments. '
        "Still extract an L1-owned KU if this segment contains NEW broad information "
        "about the parent concept."
        if l1_seen else
        f'Parent "{parent_name}" has not been covered yet in this L2 group.'
    )

    return f"""
## SCOPE CONSTRAINT
Current L2 topic: "{concept_name}"
Parent L1 concept: "{parent_name}"

{seen_note}

Extract KUs with explicit ownership:
  1. L2-specific KU:
     - owner_level = "L2"
     - owner_concept = "{concept_name}"
     - parent_l1 = "{parent_name}"
     - parent_l2 = "{concept_name}"
  2. Parent/L1-level KU:
     - owner_level = "L1"
     - owner_concept = "{parent_name}"
     - parent_l1 = "{parent_name}"
     - parent_l2 = "{concept_name}"

`concept` / `local_concept` is the actual testable concept and may be a sub-concept
inside the text. Example: an L2 segment "{concept_name}" may produce
local_concept = "cell state", while owner_concept is either "{concept_name}" or
"{parent_name}" depending on whether the KU is L2-specific or L1-general.

Do NOT extract content about L2 siblings of "{concept_name}" unless it is explicitly
used in this text to explain "{concept_name}" or the parent "{parent_name}".
"""


def _build_doc_concepts_section(
    doc_concepts: dict | None,
) -> str:
    """Ownership-aware document concept rules. Overrides legacy concept constraint."""
    if not doc_concepts:
        return ""
    seg_concepts = doc_concepts.get("seg") or []
    all_concepts = doc_concepts.get("all") or []

    seg_lines = "\n".join(f"  - {c}" for c in seg_concepts)
    all_lines = "\n".join(f"  - {c}" for c in all_concepts)

    return f"""
## DOCUMENT CONCEPTS - OWNERSHIP CONSTRAINT
Segment owner concepts (use for `owner_concept` - copy EXACTLY):
{seg_lines}

Full document concept list (use for related_kus.target_concept):
{all_lines}

CRITICAL NAMING RULE:
- owner_concept MUST be one of the segment owner concepts listed above
- concept/local_concept may be a precise sub-concept found in the text
- If you discover a local concept not in any list, output it in "new_concepts"
- NEVER invent owner_concept variations; use exact names from the list
"""


def build_extraction_prompt(
    chunk: dict,
    cross_rels: list[dict] | None = None,
    doc_concepts: dict | None = None,
    seen_concepts: list[str] | None = None,
) -> str:
    chunk_id   = chunk["chunk_id"]
    pages_repr = json.dumps(chunk.get("pages", []))

    example = f"""{{
  "knowledge_units": [
    {{
      "ku_id":             "{chunk_id}_ku_01",
      "type":              "definition",
      "concept":           "Reflexion",
      "local_concept":     "Reflexion",
      "owner_level":       "L1",
      "owner_concept":     "Reflexion",
      "parent_l1":         "Reflexion",
      "parent_l2":         "",
      "content":           "Reflexion extends ReAct by adding Evaluator and Reflector components for self-correction",
      "verbatim_evidence": "Reflexion = ReAct + Evaluator + Reflector",
      "related_kus": [
        {{
          "target_concept": "ReAct",
          "relation": "EXTENDS",
          "evidence": "Reflexion = ReAct + Evaluator + Reflector"
        }}
      ],
      "source_pages":      {pages_repr},
      "prominence":        "primary",
      "completeness":      "complete"
    }}
  ],
  "new_concepts": []
}}"""

    doc_concepts_section = _build_doc_concepts_section(doc_concepts)
    recap_section        = _build_recap_section(seen_concepts)

    parent_concept_name = chunk.get("parent_concept_name", "")
    l1_already_seen     = chunk.get("l1_already_seen", False)
    hierarchy_line      = f"Part of  : {parent_concept_name}\n" if parent_concept_name else ""
    scope_section       = _build_scope_section(chunk['topic'], parent_concept_name, l1_already_seen)

    return f"""You are a knowledge engineer analyzing a university lecture slide.

## SOURCE CONTENT
Topic:    {chunk['topic']}
{hierarchy_line}Pages:    {chunk.get('pages', '')}
Chunk ID: {chunk_id}

--- BEGIN CONTENT ---
{chunk['text']}
--- END CONTENT ---

## YOUR TASK
Extract ALL atomic knowledge units (KUs) from the content above.
Each KU must represent EXACTLY 1 idea that can support 1 independent exam question.

## KU TYPES
{_KU_TYPES}

Note:
{_FAILURE_MODE_HINTS}

## EDGE RELATIONS
{_EDGE_RELATIONS}

{_INTRA_EDGE_RULES}

## PROMINENCE CLASSIFICATION
{_PROMINENCE}

## COMPLETENESS CRITERIA
{_COMPLETENESS}

## FILL GAP RULES
{_FILL_GAP}

{_SKIP_PEDAGOGY}
{scope_section}
## CRITICAL RULES
{_CRITICAL}
{doc_concepts_section}{recap_section}{_build_cross_context(cross_rels)}
## OUTPUT FORMAT
Return ONLY a valid JSON object — no markdown, no explanation outside the object.
Number ku_id sequentially: {chunk_id}_ku_01, {chunk_id}_ku_02, ...

Wrap all KUs in a "knowledge_units" key; list any newly discovered concept names in "new_concepts".

Fields for each KU:
  ku_id             : "{chunk_id}_ku_NN"
  type              : one of the 6 types above
  concept           : short concept name (3–6 words max).
                      For supporting/peripheral KUs, use a SPECIFIC name that distinguishes
                      it from the primary concept.
                      BAD : "PCA" for every KU in a PCA chunk
                      GOOD: "PCA", "PCA dimensionality reduction", "PCA noise removal",
                            "Spark MLlib PCA", "PCA vs sklearn"
  local_concept     : same as concept unless a clearer local sub-concept name is needed
  owner_level       : "L1" if KU is broad parent-level, "L2" if KU belongs to current L2,
                      "local" only if no L1/L2 owner is available
  owner_concept     : exact L1/L2 concept node that should contain this KU in the graph UI
  parent_l1         : exact parent L1 concept, or current topic for an L1 segment
  parent_l2         : exact current L2 topic, or "" for an L1 segment
  content           : complete statement (may use Reformulate/Combine/Convert)
  verbatim_evidence : exact quote from source text (or "A [...] B" for Combine)
  related_kus       : list of edges to other concepts in this segment.
                      DEFAULT: empty []. Only add an edge if ALL 3 conditions hold:
                        1. There is an exact quote from the source that mentions BOTH concepts
                        2. The two concepts are genuinely confusable (horizontal) or dependent (structural)
                        3. A student could realistically mix up or confuse the two
                      MAX 3 entries. Each entry:
                        target_concept : EXACT name from document concepts list
                        relation       : one of the 8 types above
                        evidence       : verbatim quote mentioning BOTH concepts
                      Prefer HORIZONTAL relations (CONTRASTS_WITH, ALTERNATIVE_TO, SIBLING_OF, SIMILAR_TO)
                      over STRUCTURAL when both could apply — horizontal edges are more useful.
  source_pages      : list of page numbers where evidence appears
  prominence        : "primary" | "supporting" | "peripheral"
  completeness      : "complete" | "incomplete"

## EXAMPLE (1 KU)
{example}

## EXTRACT NOW:"""


# ---------------------------------------------------------------------------
# Optimized ownership-aware prompt builder.
# This final definition intentionally supersedes the legacy builder above while
# keeping backwards compatibility with tests/imports that use build_extraction_prompt.

_KU_TERMINOLOGY = """
Terminology for this extraction pass:
- KU / Knowledge Unit: one small, grounded, independently testable idea from the source text.
- L1 / broad topic: the broad section/module that owns this segment.
- L2 / taught subtopic: the named subtopic currently being processed under an L1.
- owner_concept: the exact L1/L2 graph node that should contain the KU in the UI.
- concept/local_concept: the precise tested idea; it may be smaller than L2.

These labels are internal schema terms. Do not phrase learner-facing content as
"this KU"; write natural concept statements instead.
""".strip()

def _ownership_scope(
    concept_name: str,
    parent_name: str,
    l1_seen: bool = False,
    owner_concepts: list[str] | None = None,
) -> str:
    if not concept_name:
        return ""
    if not parent_name:
        return f"""
## SCOPE AND OWNERSHIP
Current segment is an L1/root topic: "{concept_name}".

Extract KUs about this topic and its directly explained sub-aspects only.
For every KU:
- owner_level = "L1"
- owner_concept = "{concept_name}"
- parent_l1 = "{concept_name}"
- parent_l2 = ""
- concept/local_concept = the precise tested concept from the source text
"""

    parent_note = (
        "The parent may already have KUs from an earlier sibling segment. "
        "Still extract an L1-owned KU if this segment contains NEW broad information "
        "about the parent concept."
        if l1_seen else
        "The parent has not been covered yet in this L2 group; broad parent-level KUs are allowed."
    )

    l2_owners = [
        str(c or "").strip()
        for c in (owner_concepts or [])
        if str(c or "").strip() and str(c or "").strip() != parent_name
    ]
    if len(l2_owners) > 1:
        owner_lines = "\n".join(f'   - "{name}"' for name in l2_owners)
        return f"""
## SCOPE AND OWNERSHIP
Current segment groups multiple L2 topics under parent L1 "{parent_name}":
{owner_lines}

{parent_note}

Every KU must be assigned to exactly one UI owner:
1. L2-specific KU:
   - owner_level = "L2"
   - owner_concept = the exact L2 topic from the list above that the KU teaches
   - parent_l1 = "{parent_name}"
   - parent_l2 = same as owner_concept
2. Parent/L1-level KU:
   - owner_level = "L1"
   - owner_concept = "{parent_name}"
   - parent_l1 = "{parent_name}"
   - parent_l2 = the most relevant L2 topic from the list above

Do NOT use the combined segment label "{concept_name}" as owner_concept unless it
appears exactly in the allowed owner list. For example, if a KU teaches
"Output gate", owner_concept must be "Output gate", not "{concept_name}".

Use concept/local_concept for the precise tested idea. It may be smaller than L2.
"""

    return f"""
## SCOPE AND OWNERSHIP
Current L2 topic: "{concept_name}"
Parent L1 concept: "{parent_name}"

{parent_note}

Every KU must be assigned to exactly one UI owner:
1. L2-specific KU:
   - owner_level = "L2"
   - owner_concept = "{concept_name}"
   - parent_l1 = "{parent_name}"
   - parent_l2 = "{concept_name}"
2. Parent/L1-level KU:
   - owner_level = "L1"
   - owner_concept = "{parent_name}"
   - parent_l1 = "{parent_name}"
   - parent_l2 = "{concept_name}"

Use concept/local_concept for the precise tested idea. It may be a sub-concept
inside the text, such as "cell state", while owner_concept remains the L1/L2 node
that should contain the KU in the graph UI.

Do NOT extract content about sibling L2 topics unless the text explicitly uses that
sibling to explain the current L2 or the parent L1.
"""


def _ownership_doc_concepts(doc_concepts: dict | None) -> str:
    if not doc_concepts:
        return ""
    seg_concepts = doc_concepts.get("seg") or []
    all_concepts = doc_concepts.get("all") or []
    seg_lines = "\n".join(f"  - {c}" for c in seg_concepts)
    all_lines = "\n".join(f"  - {c}" for c in all_concepts)
    return f"""
## DOCUMENT CONCEPTS
Allowed owner_concept values for this segment:
{seg_lines}

Known document concepts for related_kus.target_concept:
{all_lines}

Rules:
- owner_concept MUST exactly match one of the allowed owner concepts above.
- concept/local_concept may be a precise sub-concept discovered in this segment.
- If you discover a useful local concept not in the list, put it in new_concepts.
- Never invent spelling variants for owner_concept.
"""


def build_extraction_prompt(
    chunk: dict,
    cross_rels: list[dict] | None = None,
    doc_concepts: dict | None = None,
    seen_concepts: list[str] | None = None,
) -> str:
    chunk_id = chunk["chunk_id"]
    pages_repr = json.dumps(chunk.get("pages", []))
    parent_name = chunk.get("parent_concept_name", "")
    l1_seen = chunk.get("l1_already_seen", False)
    owner_concepts = chunk.get("owner_concepts") or []
    hierarchy_line = f"Part of:  {parent_name}\n" if parent_name else ""

    scope_section = _ownership_scope(chunk["topic"], parent_name, l1_seen, owner_concepts)
    doc_concepts_section = _ownership_doc_concepts(doc_concepts)
    recap_section = _build_recap_section(seen_concepts)
    cross_context = _build_cross_context(cross_rels)

    example = f"""{{
  "knowledge_units": [
    {{
      "ku_id": "{chunk_id}_ku_01",
      "type": "definition",
      "concept": "Reflexion",
      "local_concept": "Reflexion",
      "owner_level": "L1",
      "owner_concept": "Reflexion",
      "parent_l1": "Reflexion",
      "parent_l2": "",
      "content": "Reflexion extends ReAct by adding Evaluator and Reflector components for self-correction.",
      "verbatim_evidence": "Reflexion = ReAct + Evaluator + Reflector",
      "related_kus": [
        {{
          "target_concept": "ReAct",
          "relation": "EXTENDS",
          "evidence": "Reflexion = ReAct + Evaluator + Reflector"
        }}
      ],
      "source_pages": {pages_repr},
      "prominence": "primary",
      "completeness": "complete"
    }}
  ],
  "new_concepts": []
}}"""

    return f"""You are a knowledge engineer extracting grounded Knowledge Units from lecture content.

## SOURCE CONTENT
Topic:    {chunk['topic']}
{hierarchy_line}Pages:    {chunk.get('pages', '')}
Chunk ID: {chunk_id}

--- BEGIN CONTENT ---
{chunk['text']}
--- END CONTENT ---

## TERMINOLOGY
{_KU_TERMINOLOGY}

## TASK
Extract ALL atomic Knowledge Units (KUs) from the source content.
Each KU must represent exactly ONE independently testable idea.

## KU TYPES
{_KU_TYPES}

## FAILURE MODE HINTS
{_FAILURE_MODE_HINTS}

## EDGE RELATIONS
{_EDGE_RELATIONS}

{_INTRA_EDGE_RULES}

## PROMINENCE
{_PROMINENCE}

## COMPLETENESS CRITERIA
{_COMPLETENESS}

## FILL GAP RULES
{_FILL_GAP}

{_SKIP_PEDAGOGY}
{scope_section}
{doc_concepts_section}
{recap_section}
{cross_context}
## CRITICAL GROUNDING RULES
{_CRITICAL}
- verbatim_evidence must support the full content claim.
- Do not use outside knowledge to complete a KU.
- Do not create an application KU unless the application is explicitly present in the source.

## OUTPUT FORMAT
Return ONLY a valid JSON object. No markdown. No extra text.
Wrap all KUs in "knowledge_units" and list local discoveries in "new_concepts".
Number ku_id sequentially: {chunk_id}_ku_01, {chunk_id}_ku_02, ...

Fields for each KU:
- ku_id: "{chunk_id}_ku_NN"
- type: definition | mechanism | failure_mode | trade_off | procedure | application
- concept: precise local/tested concept name, usually 3-6 words
- local_concept: same as concept unless a clearer sub-concept name is needed
- owner_level: "L1" | "L2" | "local"
- owner_concept: exact L1/L2 concept node that should contain this KU in the graph UI
- parent_l1: exact parent L1 concept, or current topic for an L1 segment
- parent_l2: exact current L2 topic, or "" for an L1 segment
- content: complete statement using Reformulate/Combine/Convert if needed
- verbatim_evidence: exact quote from source, or "A [...] B" for Combine
- related_kus: list of edges with target_concept, relation, evidence; default []
- source_pages: list of page numbers where evidence appears
- prominence: primary | supporting | peripheral
- completeness: complete | incomplete

## EXAMPLE
{example}

## EXTRACT NOW:"""
