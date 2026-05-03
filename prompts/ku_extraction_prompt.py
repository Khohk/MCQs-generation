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


def build_extraction_prompt(chunk: dict) -> str:
    chunk_id   = chunk["chunk_id"]
    pages_repr = json.dumps(chunk.get("pages", []))

    example = f"""[
  {{
    "ku_id":             "{chunk_id}_ku_01",
    "type":              "definition",
    "concept":           "Reflexion",
    "content":           "Reflexion extends ReAct by adding Evaluator and Reflector components for self-correction",
    "verbatim_evidence": "Reflexion = ReAct + Evaluator + Reflector",
    "related_concepts":  ["ReAct", "Evaluator", "Reflector"],
    "source_pages":      {pages_repr},
    "prominence":        "primary",
    "completeness":      "complete"
  }}
]"""

    return f"""You are a knowledge engineer analyzing a university lecture slide.

## SOURCE CONTENT
Topic:    {chunk['topic']}
Pages:    {chunk.get('pages', '')}
Chunk ID: {chunk_id}

--- BEGIN CONTENT ---
{chunk['text']}
--- END CONTENT ---

## YOUR TASK
Extract ALL atomic knowledge units (KUs) from the content above.
Each KU must represent EXACTLY 1 idea that can support 1 independent exam question.

## KU TYPES
{_KU_TYPES}

## PROMINENCE CLASSIFICATION
{_PROMINENCE}

## COMPLETENESS CRITERIA
{_COMPLETENESS}

## FILL GAP RULES
{_FILL_GAP}

## CRITICAL RULES
{_CRITICAL}

## OUTPUT FORMAT
Return ONLY a valid JSON array — no markdown, no explanation outside the array.
Number ku_id sequentially: {chunk_id}_ku_01, {chunk_id}_ku_02, ...

Fields for each KU:
  ku_id             : "{chunk_id}_ku_NN"
  type              : one of the 6 types above
  concept           : short concept name (3–6 words max).
                      For supporting/peripheral KUs, use a SPECIFIC name that distinguishes
                      it from the primary concept.
                      BAD : "PCA" for every KU in a PCA chunk
                      GOOD: "PCA", "PCA dimensionality reduction", "PCA noise removal",
                            "Spark MLlib PCA", "PCA vs sklearn"
  content           : complete statement (may use Reformulate/Combine/Convert)
  verbatim_evidence : exact quote from source text (or "A [...] B" for Combine)
  related_concepts  : list of concept names FROM THIS SOURCE TEXT that this KU depends on
                      or is an aspect of. Rules:
                      - Use the EXACT concept name as it appears (or will appear) in another KU
                      - Supporting KUs MUST reference their parent primary concept
                      - If this KU is a mechanism/failure/trade_off of concept X → add "X"
                      - Do NOT add external domain knowledge not mentioned in the source
                      Example: KU "PCA dimensionality reduction" (mechanism) →
                               related_concepts: ["PCA"]  ← links back to the primary KU
  source_pages      : list of page numbers where evidence appears
  prominence        : "primary" | "supporting" | "peripheral"
  completeness      : "complete" | "incomplete"

## EXAMPLE (1 KU)
{example}

## EXTRACT NOW:"""
