"""
prompts/mcq_prompt.py
---------------------
Prompt engineering cho Gemini MCQ generation.
Techniques: Bloom's Taxonomy, difficulty-aware, distractor quality control.
"""

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


# ── Main prompt builder ────────────────────────────────────────────

def build_prompt(
    chunk: dict,
    n_questions: int = 3,
    difficulty: str = "medium",
) -> str:
    """
    Build full prompt string cho 1 chunk.

    Args:
        chunk      : {chunk_id, topic, pages, text}
        n_questions: số MCQ cần sinh
        difficulty : "easy" | "medium" | "hard"

    Returns:
        Prompt string gửi lên Gemini
    """
    bloom_levels = BLOOM_BY_DIFFICULTY.get(difficulty, ["understand", "apply"])
    bloom_desc = " / ".join(
        f"{lvl} ({BLOOM_DESCRIPTIONS[lvl]})" for lvl in bloom_levels
    )

    prompt = f"""You are an expert educator creating multiple-choice questions (MCQs) for university students.

## SOURCE CONTENT
Topic: {chunk['topic']}
Pages: {chunk['pages']}
Chunk ID: {chunk['chunk_id']}

--- BEGIN CONTENT ---
{chunk['text']}
--- END CONTENT ---

## YOUR TASK
Generate exactly {n_questions} MCQ(s) based ONLY on the content above.

## DIFFICULTY & BLOOM'S TAXONOMY
- Difficulty level: {difficulty.upper()}
- Target Bloom's levels: {bloom_desc}
- Questions must test concepts at the specified cognitive level, not just surface recall.

## DISTRACTOR QUALITY RULES (CRITICAL)
Each wrong answer (distractor) must:
1. Be the SAME TYPE as the correct answer (e.g., if answer is an algorithm name, all distractors are algorithm names)
2. Be PLAUSIBLE — a student who hasn't studied might choose it
3. Be CLEARLY WRONG to someone who understands the material
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
- Write questions and answers in English
- Keep explanation concise (1-2 sentences), referencing the source content

## EXAMPLE
{FEW_SHOT_EXAMPLE}

## GENERATE {n_questions} MCQ(s) NOW:"""

    return prompt


def build_batch_prompt(
    chunks: list[dict],
    n_per_chunk: int = 3,
    difficulty: str = "medium",
) -> list[str]:
    """Build danh sách prompts cho nhiều chunks."""
    return [
        build_prompt(chunk, n_questions=n_per_chunk, difficulty=difficulty)
        for chunk in chunks
    ]