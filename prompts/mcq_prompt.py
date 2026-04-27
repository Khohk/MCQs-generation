"""
prompts/mcq_prompt.py
---------------------
Prompt engineering cho Gemini MCQ generation.
Techniques: Bloom's Taxonomy, difficulty-aware, distractor quality control.
"""

from __future__ import annotations

import re

# ── Bloom's Taxonomy mapping ───────────────────────────────────────
BLOOM_BY_DIFFICULTY = {
    "easy":   ["remember", "understand"],
    "medium": ["apply", "analyze"],
    "hard":   ["evaluate", "create"],
}

BLOOM_DESCRIPTIONS = {
    "remember":  "recall facts, definitions, basic concepts",
    "understand":"explain ideas, interpret, classify",
    "apply":     "use knowledge in new situations, solve problems",
    "analyze":   "break down info, find patterns, compare",
    "evaluate":  "justify decisions, critique, assess",
    "create":    "produce new ideas, design, construct",
}

# ── Few-shot example ───────────────────────────────────────────────
FEW_SHOT_EXAMPLE = """
EXAMPLE OUTPUT (1 MCQ):
[
  {
    "question": "What is the primary purpose of the Porter Stemmer algorithm?",
    "A": "To translate words between languages",
    "B": "To reduce words to their base/root form by removing affixes",
    "C": "To tokenize sentences into individual words",
    "D": "To identify named entities in text",
    "answer": "B",
    "explanation": "Porter Stemmer uses a series of rewrite rules to chop off suffixes (e.g., 'running' -> 'run'), reducing words to their stems for NLP tasks like IR.",
    "bloom_level": "understand",
    "difficulty": "medium",
    "source_chunk": "chunk_001"
  }
]
""".strip()


# ── Markdown cleaner ───────────────────────────────────────────────

def _clean_chunk_text(text: str) -> str:
    """
    Clean markdown chunk text trước khi đưa vào prompt.
    Giữ lại nội dung học thuật, bỏ metadata cấu trúc.
    """
    # Bỏ HTML comments (<!-- Slide number: N -->, <!-- Notes -->, v.v.)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Đổi image syntax ![alt](src) → [Image: alt] để Gemini biết có hình
    text = re.sub(r"!\[([^\]]*)\]\([^\)]*\)", r"[Image: \1]", text)
    # Bỏ ## heading (đã có topic ở trên, tránh lặp)
    text = re.sub(r"^#{1,3}\s+.+$", "", text, flags=re.MULTILINE)
    # Collapse nhiều dòng trắng liên tiếp thành 1
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Main prompt builder ────────────────────────────────────────────

def build_hint_prompt(chunks: list[dict], slide_summary: str) -> str:
    """
    Build prompt để Gemini lập per-chunk connection hints.
    Gọi 1 lần sau khi có slide_summary — response là JSON.

    Output format:
        {"hints": {"chunk_001": ["MOTIVATES chunk_003"], "chunk_002": []}}
    """
    chunk_list = "\n".join(
        f"  - {c['chunk_id']}: {c['topic']}" for c in chunks
    )
    chunk_ids  = ", ".join(c["chunk_id"] for c in chunks)

    return f"""You are designing exam questions for a university lecture.

## LECTURE KNOWLEDGE MAP
{slide_summary.strip()}

## CHUNKS IN THIS LECTURE
{chunk_list}

## YOUR TASK
For each chunk, identify 0-2 meaningful connections to OTHER chunks that would produce
insightful exam questions. Only include connections where a student genuinely needs to
understand BOTH chunks to answer correctly.

Use exactly one of these relationship verbs:
  MOTIVATES, SOLVES, APPLIES, CONTRASTS, SYNTHESIZES, EXPLAINS, ENABLES

## OUTPUT
Return JSON only, no markdown. Keys must be all chunk IDs: {chunk_ids}

{{
  "hints": {{
    "chunk_001": ["MOTIVATES chunk_003"],
    "chunk_002": ["SOLVES chunk_001", "APPLIES chunk_007"],
    "chunk_003": []
  }}
}}"""


def build_summary_prompt(chunks: list[dict]) -> str:
    """
    Build prompt để Gemini tạo knowledge map cho toàn slide.
    Gọi 1 lần duy nhất — response là plain markdown, không phải JSON.

    Args:
        chunks: list chunk dicts, mỗi chunk cần có chunk_id, topic, text
    """
    topic_lines = []
    for c in chunks:
        snippet = _clean_chunk_text(c["text"])[:200].replace("\n", " ")
        topic_lines.append(
            f"- [{c['chunk_id']}] {c['topic']}\n  Preview: {snippet}..."
        )
    topic_block = "\n".join(topic_lines)

    return f"""You are analyzing a university lecture slide deck.

Below is a list of all topics covered, with a short preview of each section:

{topic_block}

Create a concise knowledge map (under 200 words) describing:
1. The knowledge flow from start to finish (e.g., A → B → C)
2. Key relationships between topics — use verbs like: MOTIVATES, EXPLAINS, APPLIES, SOLVES, SYNTHESIZES
3. Core concepts that appear across multiple topics

Output plain markdown only, no JSON. Be specific to the actual content listed above."""


def build_prompt(
    chunk: dict,
    n_questions: int = 3,
    difficulty: str = "medium",
    slide_context: str = "",
    chunk_hints: list[str] = None,
    language: str = "en",
) -> str:
    """
    Build full prompt string cho 1 chunk.

    Args:
        chunk        : {chunk_id, topic, pages, text, has_image}
        n_questions  : số MCQ cần sinh
        difficulty   : "easy" | "medium" | "hard"
        slide_context: knowledge map của toàn slide
        chunk_hints  : list connection hints cho chunk này,
                       vd. ["SOLVES chunk_002", "APPLIES chunk_009"]

    Returns:
        Prompt string gửi lên Gemini
    """
    if chunk_hints is None:
        chunk_hints = []
    bloom_levels = BLOOM_BY_DIFFICULTY.get(difficulty, ["understand", "apply"])
    bloom_desc = " / ".join(
        f"{lvl} ({BLOOM_DESCRIPTIONS[lvl]})" for lvl in bloom_levels
    )

    clean_text = _clean_chunk_text(chunk["text"])
    has_image  = chunk.get("has_image", False)

    # Instruction bổ sung nếu chunk có ảnh được mô tả
    image_instruction = ""
    if has_image:
        image_instruction = (
            "\n- This slide contains visual content (diagrams/charts) marked as [Image: ...]. "
            "Generate at least 1 question that tests understanding of that visual content."
        )

    # Knowledge map + connection hints — chỉ inject khi có ít nhất 1 trong 2
    context_section = ""
    if slide_context.strip() or chunk_hints:
        map_block  = f"\n{slide_context.strip()}\n" if slide_context.strip() else ""

        if chunk_hints:
            hint_lines = "\n".join(f"  - {h}" for h in chunk_hints)
            hint_block = (
                f"\n### REQUIRED: Cross-chunk connection\n"
                f"You MUST generate at least 1 question that exploits one of these connections.\n"
                f"That question must require knowing BOTH this chunk AND the connected chunk "
                f"to answer correctly — a student who only studied this chunk should not be "
                f"able to answer it from this chunk's content alone:\n"
                f"{hint_lines}\n"
                f"\nCRITICAL: NEVER mention chunk IDs (e.g. 'chunk_005', 'chunk_012') in the "
                f"question stem or any answer option. Use the topic name instead "
                f"(e.g. 'RACI framework', 'ETL process'). Chunk IDs are internal — students never see them.\n"
            )
        else:
            hint_block = ""

        context_section = f"""
## LECTURE KNOWLEDGE MAP
This chunk is part of a larger lecture. Here is the overall knowledge structure:
{map_block}{hint_block}
"""

    prompt = f"""You are an expert educator creating multiple-choice questions (MCQs) for university students.

## SOURCE CONTENT
Topic: {chunk['topic']}
Pages: {chunk['pages']}
Chunk ID: {chunk['chunk_id']}

--- BEGIN CONTENT ---
{clean_text}
--- END CONTENT ---
{context_section}
## YOUR TASK
Generate exactly {n_questions} MCQ(s) based ONLY on the content above.{image_instruction}
- Each question must test a DIFFERENT concept or aspect — no two questions may be paraphrases of each other.
- If the content only supports fewer distinct concepts than {n_questions}, test different cognitive angles (definition vs application vs implication).
- Write ALL questions and answers in {"Vietnamese (Tiếng Việt)" if language == "vi" else "English"}. Do NOT mix languages.

## DIFFICULTY & BLOOM'S TAXONOMY
- Difficulty level: {difficulty.upper()}
- Target Bloom's levels: {bloom_desc}
- Questions must test concepts at the specified cognitive level, not just surface recall.
- Cross-chunk questions are NOT automatically "hard". A cross-chunk question can be "medium" if the connection is direct and the student is expected to know both topics. Match difficulty to cognitive load, not to question type.

## DISTRACTOR QUALITY RULES (CRITICAL)
Each wrong answer (distractor) must:
1. Be the SAME TYPE as the correct answer (e.g., if answer is an algorithm name, all distractors are algorithm names)
2. Be PLAUSIBLE — a student who HAS studied the material but hasn't mastered this specific distinction might choose it. "Hasn't studied" is NOT the bar — set the bar higher.
3. For cross-chunk comparison questions: each distractor must represent a specific, believable misconception about one of the concepts being compared — NOT just a label for another architecture/concept. A student who knows concept A but misunderstands how it differs from concept B should find the distractor tempting.
4. NOT be "All of the above" / "None of the above" / "Both A and B"
5. Be similar in length and grammatical structure to the correct answer
6. Be COMPLETELY DIFFERENT from the correct answer — never copy, paraphrase, or repeat the correct answer text
7. ALL FOUR options A, B, C, D must have unique text — no two options may say the same thing

## OUTPUT FORMAT RULES
- Return ONLY a valid JSON array, no markdown, no extra text
- Each MCQ must have exactly these 10 fields:
  question, A, B, C, D, answer, explanation, bloom_level, difficulty, source_chunk
- "answer" must be exactly one of: A, B, C, D
- "bloom_level" must be one of: remember, understand, apply, analyze, evaluate, create
- "difficulty" must be one of: easy, medium, hard
- "source_chunk" must be: {chunk['chunk_id']}
- Distribute correct answers: A, B, C, D must each appear at least once across all questions. Target ~25% each. NEVER put more than 40% of answers as the same letter.
- Write questions and answers in {"Vietnamese (Tiếng Việt)" if language == "vi" else "English"}
- Keep explanation concise (1-2 sentences), referencing the source content

## EXAMPLE
{FEW_SHOT_EXAMPLE}

## GENERATE {n_questions} MCQ(s) NOW:"""

    return prompt


def build_summary_chunk_prompt(
    slide_summary: str,
    hints_map: dict[str, list[str]],
    n_questions: int = 3,
    difficulty: str = "medium",
    language: str = "en",
) -> str:
    """
    Prompt chuyên biệt cho summary_chunk.
    Yêu cầu model sinh câu hỏi về RELATIONSHIPS giữa các topic,
    không hỏi về nội dung từng topic đơn lẻ.
    """
    bloom_levels = BLOOM_BY_DIFFICULTY.get(difficulty, ["understand", "apply"])
    # Summary chunk nên hướng đến bloom cao hơn
    if difficulty == "medium":
        bloom_levels = ["analyze", "evaluate"]
    bloom_desc = " / ".join(
        f"{lvl} ({BLOOM_DESCRIPTIONS[lvl]})" for lvl in bloom_levels
    )

    # Tổng hợp tất cả hints thành dạng dễ đọc
    all_hints = []
    for chunk_id, hints in hints_map.items():
        for h in hints:
            all_hints.append(f"  - {chunk_id} {h}")
    hints_block = "\n".join(all_hints) if all_hints else "  (no cross-chunk connections identified)"

    return f"""You are an expert educator creating synthesis-level MCQs for a university lecture.

## LECTURE KNOWLEDGE MAP
{slide_summary.strip()}

## KNOWN CROSS-CHUNK CONNECTIONS
{hints_block}

## YOUR TASK
Generate exactly {n_questions} MCQ(s) that test understanding of RELATIONSHIPS between topics.

Each question MUST:
- Require knowledge of at least 2 different topics from the lecture to answer correctly
- Test WHY or HOW topics relate, not WHAT each topic is individually
- Reference a specific connection (e.g., "How does X solve the problem introduced by Y?")

Do NOT generate questions that can be answered from a single topic alone.

## DIFFICULTY & BLOOM'S TAXONOMY
- Difficulty level: {difficulty.upper()}
- Target Bloom's levels: {bloom_desc}

## DISTRACTOR QUALITY RULES (CRITICAL)
Each wrong answer must:
1. Be from the SAME domain as the correct answer (e.g., if answer is an architecture name, all options are architecture names)
2. Be PLAUSIBLE to a student who studied the individual topics but missed how they connect
3. NOT be "All of the above" / "None of the above"
4. Be similar in length and structure to the correct answer

## OUTPUT FORMAT RULES
- Return ONLY a valid JSON array, no markdown, no extra text
- Each MCQ must have exactly these 10 fields:
  question, A, B, C, D, answer, explanation, bloom_level, difficulty, source_chunk
- "answer" must be exactly one of: A, B, C, D
- "bloom_level" must be one of: remember, understand, apply, analyze, evaluate, create
- "difficulty" must be one of: easy, medium, hard
- "source_chunk" must be: __summary__
- Write questions and answers in {"Vietnamese (Tiếng Việt)" if language == "vi" else "English"}
- Explanation must reference at least 2 chunk topics

## EXAMPLE
{FEW_SHOT_EXAMPLE}

## GENERATE {n_questions} MCQ(s) NOW:"""


def build_batch_prompt(
    chunks: list[dict],
    n_per_chunk: int = 3,
    difficulty: str = "medium",
    slide_context: str = "",
    hints_map: dict[str, list[str]] = None,
) -> list[str]:
    """Build danh sách prompts cho nhiều chunks."""
    if hints_map is None:
        hints_map = {}
    return [
        build_prompt(
            chunk,
            n_questions=n_per_chunk,
            difficulty=difficulty,
            slide_context=slide_context,
            chunk_hints=hints_map.get(chunk["chunk_id"], []),
        )
        for chunk in chunks
    ]
