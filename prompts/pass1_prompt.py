from __future__ import annotations

_L2_RULE = """
L2 children MUST be items that the document explicitly enumerates as a set,
AND each item must have its own dedicated page range in the document:
  ✓ Numbered patterns  : "1. Prompt Chaining  2. Routing  3. Parallel ..." (each has slides)
  ✓ Named set          : "5 patterns: A, B, C, D, E" (each has slides)
  ✓ Framework list     : "LangGraph, AutoGen, CrewAI — so sánh công cụ" (each has slides)

L2 children MUST NOT be:
  ✗ Reasons / motivations  : "3 lý do chính: Specialization, Parallelization, Cross-checking"
                             (all reasons share the same slides → no separate page ranges)
  ✗ Section headers        : "Definition", "Why important", "How it works"
  ✗ Properties             : bullet points describing attributes of one concept
  ✗ Sub-steps              : steps within a single algorithm or procedure

Rule of thumb: if items would each get at least 1 dedicated slide → L2.
              If all items fit on the same 1-2 slides → leave children: [].
""".strip()

_L1_RULE = """
L1 nodes MUST be broad module/section topics, not narrow lesson points.

Good L1 names:
  - Stable topic areas: "MCP Fundamentals", "MCP Architecture",
    "MCP Primitives", "MCP Transport"
  - Broad conceptual modules that can contain several teachable subtopics

Bad L1 names:
  - Motivation/problem statements: "Why MCP is needed", "N x M problem",
    "The problem of tool integration"
  - Slide-title questions: "Why do we need X?", "What problem does X solve?"
  - Single claims, examples, benefits, risks, or one-off observations

If the source starts with a motivation/problem section, place it under a broad
umbrella L1 such as "MCP Fundamentals". Do not promote the motivation itself to L1.
""".strip()

_PASS1_TERMINOLOGY = """
Terminology for this structural pass:
- main_concept: the single topic of the whole document.
- L1 / broad topic: a major module or section of the document.
- L2 / taught subtopic: a named subtopic explicitly taught under an L1 topic.
- relationship: an explicit conceptual link between two L1/L2 concepts.

This pass is only a document map. Do NOT extract detailed testable facts here;
those will be extracted later from the segments.
""".strip()

_EVIDENCE_RULE = """
EVIDENCE REQUIREMENT (CRITICAL):
  - The 'evidence' field MUST be a verbatim quote from the document
  - The quote MUST explicitly mention BOTH concepts (or strong synonyms)
  - If you cannot find such a quote → DO NOT create the relationship
  - Inferred / implied relationships without textual support are NOT allowed

evidence_type classification:
  "explicit"         : quote contains both names + a relation marker
                       (e.g. "X uses Y", "X builds on Y", "X vs Y")
  "discourse_marker" : both concepts connected by transition words
                       (e.g. "first X, then Y enables Z")
  "co_occurrence"    : both in same sentence but relation only implied

Prefer explicit > discourse_marker > co_occurrence.
Drop relationship if only co_occurrence evidence exists and value is marginal.
""".strip()

_RELATION_TYPES = """
Horizontal (distractor-relevant — concepts are confusable/comparable):
  CONTRASTS_WITH : A and B are opposites or trade-offs
  ALTERNATIVE_TO : A and B serve the same goal via different approaches
  SIMILAR_TO     : A and B are easily confused

Structural (dependency — for UI/explanation only):
  ENABLES        : A is a prerequisite that makes B possible
  EXTENDS        : A builds on or is a special case of B
  APPLIES_TO     : A is used in the context of B

NOTE: Do NOT use PART_OF — parent-child relationships are captured in
concept_hierarchy, not in relationships.
""".strip()


def build_pass1_prompt(whole_doc_text: str, doc_title: str = "Document",
                       page_list: list[int] | None = None) -> str:
    total_pages = len(page_list) if page_list else "unknown"
    pages_hint = f"\nDocument pages (in order): {page_list}" if page_list else ""

    return f"""You are a knowledge engineer performing structural analysis of a lecture document.

## DOCUMENT
Title: {doc_title}
Total pages: {total_pages}{pages_hint}

--- BEGIN DOCUMENT ---
{whole_doc_text}
--- END DOCUMENT ---

## YOUR TASK
Analyze the ENTIRE document and return a JSON object with exactly 3 keys:
main_concept, concept_hierarchy, relationships.

{_PASS1_TERMINOLOGY}

────────────────────────────────────────────────────────────
## 1. main_concept  (string)
The single primary topic the whole document is about.

────────────────────────────────────────────────────────────
## 2. concept_hierarchy  (array)
A 2-level concept tree. Code will derive segment page boundaries automatically
from last_page — you only need to identify WHERE each concept ends.

Level 1 (L1): Major topic areas.
  - If L1 has no children → L1 becomes one segment.
  - If L1 has children → each L2 child becomes one segment.

{_L1_RULE}

Level 2 (L2): Specific concepts EXPLICITLY enumerated under an L1 concept
  in the source text (e.g. "5 patterns: A, B, C, D, E").

{_L2_RULE}

Each L1 entry:
  concept_id   : snake_case unique id (e.g. "agentic_patterns")
  name         : concept name (consistent with source text)
  level        : 1
  evidence     : verbatim quote from source introducing this topic
  last_page    : sequential page number where this L1's content ends
                 (only used when L1 has no children)
  children     : list of L2 entries (can be empty [])

Each L2 entry:
  concept_id   : snake_case unique id
  name         : concept name (EXACT as source text defines it)
  level        : 2
  evidence     : verbatim quote showing this is listed under the parent
  last_page    : sequential page number where THIS concept's teaching ends
                 (before the next concept in the list starts)
  siblings_auto: true

LAST_PAGE RULES:
  - Use sequential page numbers as shown in the document you received (1..{total_pages})
  - Set last_page to the LAST page where this concept is primarily taught
  - For the last concept overall: stop before trailing Q&A, bibliography,
    hands-on labs, or practice sections at end of file
  - For middle concepts: include pages up to (but not including) the next concept
  - Code derives first_page automatically as prev_last + 1

CRITICAL — only include L2 children the document EXPLICITLY enumerates.
Do NOT invent children. If a concept has no explicit children → empty list [].
If several L2 candidates would have the exact same last_page, they are not
separate teachable segments. Keep them as detailed concepts inside the parent L1,
or create one combined L2 only if the page range teaches them together.

────────────────────────────────────────────────────────────
## 3. relationships  (array)
Relationships between any two concepts in concept_hierarchy — including siblings under the same L1.
Do NOT include PART_OF (hierarchy already captures parent-child).
Sibling concepts (e.g. Prompt Chaining ↔ Routing, LangGraph ↔ CrewAI) are especially valuable
for CONTRASTS_WITH / ALTERNATIVE_TO / SIMILAR_TO.

{_EVIDENCE_RULE}

Each entry:
  from_concept  : concept name (must exist in concept_hierarchy)
  to_concept    : concept name (must exist in concept_hierarchy)
  relation      : one of the types listed below
  evidence      : verbatim quote mentioning BOTH concepts
  evidence_type : "explicit" | "discourse_marker" | "co_occurrence"

Relation types:
{_RELATION_TYPES}

────────────────────────────────────────────────────────────
## NAMING RULES
  - If source defines "Full Name (ABBR)" → use ABBR consistently everywhere
  - If no abbreviation → use full name consistently
  - L2 concept names must match EXACTLY how source text lists them

────────────────────────────────────────────────────────────
## OUTPUT FORMAT
Return ONLY valid JSON — no markdown, no text outside JSON.
Keys in order: main_concept, concept_hierarchy, relationships.

## EXAMPLE
{{
  "main_concept": "Multi-Agent System Design",
  "concept_hierarchy": [
    {{
      "concept_id": "mas_taxonomy",
      "name": "MAS Taxonomy",
      "level": 1,
      "evidence": "3 lý do chính để multi-agent: Specialization, Parallelization, Memory",
      "last_page": 6,
      "children": []
    }},
    {{
      "concept_id": "agentic_patterns",
      "name": "Agentic Workflow Patterns",
      "level": 1,
      "evidence": "5 Agentic Workflow Patterns (Anthropic): Prompt Chaining, Routing, Parallel, Orchestrator, Evaluator",
      "last_page": 21,
      "children": [
        {{"concept_id": "prompt_chaining", "name": "Prompt Chaining", "level": 2,
          "evidence": "Prompt Chaining — sequential, validate each step",
          "last_page": 9,  "siblings_auto": true}},
        {{"concept_id": "routing",         "name": "Routing",         "level": 2,
          "evidence": "Routing — classify input rồi route đến handler",
          "last_page": 12, "siblings_auto": true}},
        {{"concept_id": "parallel",        "name": "Parallel",        "level": 2,
          "evidence": "Parallel — split task, multiple workers simultaneously",
          "last_page": 15, "siblings_auto": true}},
        {{"concept_id": "orchestrator",    "name": "Orchestrator",    "level": 2,
          "evidence": "Orchestrator — hub-spoke delegation",
          "last_page": 18, "siblings_auto": true}},
        {{"concept_id": "evaluator",       "name": "Evaluator",       "level": 2,
          "evidence": "Evaluator — self-correction loop",
          "last_page": 21, "siblings_auto": true}}
      ]
    }},
    {{
      "concept_id": "mas_frameworks",
      "name": "MAS Frameworks",
      "level": 1,
      "evidence": "LangGraph, AutoGen, CrewAI — chọn đúng công cụ",
      "last_page": 25,
      "children": [
        {{"concept_id": "langgraph", "name": "LangGraph", "level": 2,
          "evidence": "LangGraph cho production — full state control",
          "last_page": 23, "siblings_auto": true}},
        {{"concept_id": "autogen",   "name": "AutoGen",   "level": 2,
          "evidence": "AutoGen tốt cho prototype nhanh",
          "last_page": 24, "siblings_auto": true}},
        {{"concept_id": "crewai",    "name": "CrewAI",    "level": 2,
          "evidence": "CrewAI cho role-based multi-agent",
          "last_page": 25, "siblings_auto": true}}
      ]
    }}
  ],
  "relationships": [
    {{
      "from_concept": "Routing",
      "to_concept": "LangGraph",
      "relation": "APPLIES_TO",
      "evidence": "Supervisor Pattern — Hub-spoke delegation với LangGraph",
      "evidence_type": "explicit"
    }}
  ]
}}"""
