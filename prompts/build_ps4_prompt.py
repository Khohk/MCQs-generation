"""
prompts/build_ps4_prompt.py
---------------------------
PS4 prompt: Chain-of-Thought + Bloom skill description + example question.

One function, one responsibility: build a focused prompt for exactly one
Bloom level. Generator calls this once per (chunk, bloom_level) pair.

PS4 = CoT + skill description + example questions  (from paper taxonomy)
"""

from __future__ import annotations

import re

from prompts.bloom_definitions import BLOOM_SPECS

# ── Reuse the same few-shot example format ────────────────────────
_FEW_SHOT = """EXAMPLE (1 MCQ at "analyze" level):
[
  {
    "question": "What is the key difference between a data lake and a data warehouse that explains why data lakes are preferred for exploratory analytics?",
    "A": "Data lakes store only structured data, while warehouses handle unstructured data",
    "B": "Data lakes preserve raw data in any format, allowing schema-on-read flexibility that warehouses' schema-on-write model cannot provide",
    "C": "Data warehouses are cheaper to maintain than data lakes at the same scale",
    "D": "Data lakes require ETL pipelines before ingestion, unlike data warehouses",
    "answer": "B",
    "explanation": "Schema-on-read (data lake) vs schema-on-write (warehouse) is the structural difference that enables exploratory analytics without upfront schema design.",
    "bloom_level": "analyze",
    "difficulty": "medium",
    "source_chunk": "chunk_004"
  }
]""".strip()


def _clean(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"!\[([^\]]*)\]\([^\)]*\)", r"[Image: \1]", text)
    text = re.sub(r"^#{1,3}\s+.+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_ps4_prompt(
    chunk: dict,
    bloom_level: str,
    n_questions: int = 1,
    language: str = "en",
) -> str:
    """
    Build a PS4-style prompt for one Bloom level.

    Args:
        chunk       : {chunk_id, topic, pages, text, has_image, ...}
        bloom_level : one of the 6 Bloom levels
        n_questions : number of MCQs to generate (1 for regular chunks,
                      2-4 for whole_doc chunks)
        language    : "en" | "vi"

    Returns:
        Prompt string ready to send to the LLM.
    """
    if bloom_level not in BLOOM_SPECS:
        raise ValueError(f"Unknown bloom_level '{bloom_level}'. "
                         f"Valid: {list(BLOOM_SPECS)}")

    spec       = BLOOM_SPECS[bloom_level]
    clean_text = _clean(chunk["text"])
    lang_str   = "Vietnamese (Tiếng Việt)" if language == "vi" else "English"
    has_image  = chunk.get("has_image", False)

    image_note = (
        "\n- This chunk contains visual content marked as [Image: ...]. "
        "If possible, generate a question that tests understanding of that visual."
        if has_image else ""
    )

    plural = "questions" if n_questions > 1 else "question"
    task_line = (
        f"Generate exactly {n_questions} MCQ {plural} "
        f"that test the {bloom_level.upper()} cognitive level."
    )

    skip_note = (
        f"\n## ESCAPE HATCH\n"
        f"If the content genuinely cannot support a {bloom_level.upper()} question "
        f"(e.g. it is a table of contents, an image-only slide, or contains only "
        f"isolated definitions that resist {bloom_level} reasoning), return:\n"
        f'  {{"skip": true, "reason": "..."}}\n'
        f"Do NOT force a fake question. An honest skip is better than a low-quality question."
    )

    return f"""You are an expert educator generating multiple-choice questions (MCQs) for university students.

## SOURCE CONTENT
Topic : {chunk['topic']}
Pages : {chunk['pages']}
Chunk : {chunk['chunk_id']}

--- BEGIN CONTENT ---
{clean_text}
--- END CONTENT ---

## BLOOM'S TAXONOMY — TARGET LEVEL: {bloom_level.upper()}

**Cognitive skill**: {spec['description']}
**Stem verbs to use**: {spec['verbs']}
**Example of a {bloom_level} question**: "{spec['example']}"

> Chain-of-thought before writing:
> 1. Identify the key concept in this content that can be tested at {bloom_level.upper()} level.
> 2. Choose a stem verb from the list above.
> 3. Construct a question that requires {bloom_level} reasoning — NOT just recalling a fact.
> 4. Write 3 plausible distractors that represent believable misconceptions.

## TASK
{task_line}{image_note}
- Write ALL text in {lang_str}. Do NOT mix languages.
- Each question must test a DIFFERENT concept (if n > 1).
- The stem MUST use one of the verbs: {spec['verbs']}.
- Do NOT generate a question that only requires recalling a fact — that is "remember" level.

## DISTRACTOR RULES
1. All 4 options must be the same concept type (all algorithms, all definitions, etc.)
2. Each distractor must represent a specific, believable misconception — not random noise
3. No "All of the above" / "None of the above" / "Both A and B"
4. Options must be similar in length and grammatical structure
5. Distractors must be completely different from the correct answer
6. All 4 options A, B, C, D must have unique text

## OUTPUT FORMAT
- Return ONLY valid JSON, no markdown, no extra text
- Each MCQ must have exactly 10 fields:
  question, A, B, C, D, answer, explanation, bloom_level, difficulty, source_chunk
- "answer": exactly one of A / B / C / D
- "bloom_level": must be "{bloom_level}"
- "difficulty": easy / medium / hard (infer from cognitive load)
- "source_chunk": "{chunk['chunk_id']}"
- Explanation: 1-2 sentences citing the source content
{skip_note}

## EXAMPLE
{_FEW_SHOT}

## GENERATE {n_questions} MCQ(s) AT {bloom_level.upper()} LEVEL NOW:"""
