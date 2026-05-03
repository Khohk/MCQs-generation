"""
pipeline/validator.py
---------------------
List[MCQ raw] -> (List[MCQ valid], List[{mcq, reason, layer}])

Two-layer validation:
  Layer 1: schema validity
  Layer 2: formal/content-shape quality checks

The module keeps the old public API, while adding a lightweight quality_score
for experiments and UI statistics.
"""

from __future__ import annotations


VALID_ANSWERS = {"A", "B", "C", "D"}
VALID_BLOOM = {"remember", "understand", "apply", "analyze", "evaluate", "create"}
VALID_DIFFICULTY = {"easy", "medium", "hard"}
REQUIRED_FIELDS = {
    "question", "A", "B", "C", "D", "answer",
    "explanation", "bloom_level", "difficulty", "source_chunk",
}
MIN_QUESTION_LEN = 10

BLOOM_BY_DIFFICULTY = {
    "easy": {"remember", "understand"},
    "medium": {"apply", "analyze"},
    "hard": {"evaluate", "create"},
}


def validate_mcqs(raw_mcqs: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Validate and filter MCQs.

    Returns:
        (valid_mcqs, rejected)
    """
    valid = []
    rejected = []
    seen_questions: set[str] = set()

    for mcq in raw_mcqs:
        schema_reason = _check_schema(mcq)
        if schema_reason:
            rejected.append({"mcq": mcq, "reason": schema_reason, "layer": "schema"})
            continue

        _normalize_fields(mcq)
        quality_reason = _check_quality(mcq, seen_questions)
        mcq["quality_score"] = quality_score(mcq, seen_questions)
        expected = BLOOM_BY_DIFFICULTY.get(mcq["difficulty"], set())
        mcq["bloom_mismatch"] = mcq["bloom_level"] not in expected if expected else False

        if quality_reason:
            rejected.append({"mcq": mcq, "reason": quality_reason, "layer": "quality"})
        else:
            valid.append(mcq)
            seen_questions.add(_normalize_q(mcq["question"]))

    return valid, rejected


def _check_schema(mcq: dict) -> str | None:
    """Layer 1: required fields and controlled vocabularies."""
    if not isinstance(mcq, dict):
        return f"MCQ is not an object: {type(mcq).__name__}"

    missing = REQUIRED_FIELDS - set(mcq.keys())
    if missing:
        return f"Missing fields: {sorted(missing)}"

    answer = str(mcq.get("answer", "")).strip().upper()
    if answer not in VALID_ANSWERS:
        return f"Invalid answer: '{answer}' (must be A/B/C/D)"

    bloom = str(mcq.get("bloom_level", "")).strip().lower()
    if bloom not in VALID_BLOOM:
        return f"Invalid bloom_level: '{bloom}'"

    diff = str(mcq.get("difficulty", "")).strip().lower()
    if diff not in VALID_DIFFICULTY:
        return f"Invalid difficulty: '{diff}'"

    for field in REQUIRED_FIELDS:
        if not isinstance(mcq.get(field), (str, int, float, bool)):
            return f"Invalid type for {field}: {type(mcq.get(field)).__name__}"

    return None


def _check_quality(mcq: dict, seen_questions: set[str]) -> str | None:
    """Layer 2: MCQ formal quality checks."""
    q = str(mcq.get("question", "")).strip()
    if not q or len(q) < MIN_QUESTION_LEN:
        return f"Question too short ({len(q)} chars)"

    options = [str(mcq.get(opt, "")).strip() for opt in ["A", "B", "C", "D"]]
    for opt, val in zip(["A", "B", "C", "D"], options):
        if not val:
            return f"Option {opt} is empty"

    normalized_options = [_normalize_option(o) for o in options]
    if len(set(normalized_options)) < 4:
        return "Duplicate options"

    answer = mcq["answer"]
    correct_text = _normalize_option(str(mcq.get(answer, "")))
    for opt in ["A", "B", "C", "D"]:
        if opt != answer and _normalize_option(str(mcq.get(opt, ""))) == correct_text:
            return f"Option {opt} is identical to correct answer {answer}"

    if not str(mcq.get("explanation", "")).strip():
        return "Explanation is empty"

    if _normalize_q(q) in seen_questions:
        return "Duplicate question"

    bloom = mcq["bloom_level"]
    diff = mcq["difficulty"]
    if diff == "hard" and bloom in {"remember", "understand"}:
        return f"Bloom/Difficulty mismatch: difficulty=hard but bloom={bloom}"

    return None


def quality_score(mcq: dict, seen_questions: set[str] | None = None) -> float:
    """
    Lightweight 0..1 score for experiments.

    Components:
      0.2 schema ok
      0.2 four unique options
      0.2 explanation present
      0.2 bloom/difficulty aligned
      0.2 non-duplicate question
    """
    seen_questions = seen_questions or set()
    score = 0.0

    if _check_schema(mcq) is None:
        score += 0.2

    options = [_normalize_option(str(mcq.get(o, ""))) for o in ["A", "B", "C", "D"]]
    if all(options) and len(set(options)) == 4:
        score += 0.2

    if str(mcq.get("explanation", "")).strip():
        score += 0.2

    diff = str(mcq.get("difficulty", "")).strip().lower()
    bloom = str(mcq.get("bloom_level", "")).strip().lower()
    expected = BLOOM_BY_DIFFICULTY.get(diff)
    if expected and bloom in expected:
        score += 0.2

    q = _normalize_q(str(mcq.get("question", "")))
    if q and q not in seen_questions:
        score += 0.2

    return round(score, 2)


def _normalize_fields(mcq: dict) -> None:
    mcq["answer"] = str(mcq["answer"]).strip().upper()
    mcq["bloom_level"] = str(mcq["bloom_level"]).strip().lower()
    mcq["difficulty"] = str(mcq["difficulty"]).strip().lower()
    for key in REQUIRED_FIELDS:
        mcq[key] = str(mcq[key]).strip()


def _normalize_q(q: str) -> str:
    return " ".join(q.lower().split())


def _normalize_option(option: str) -> str:
    return " ".join(option.lower().split())


def validation_stats(valid: list[dict], rejected: list[dict]) -> dict:
    """Return aggregate stats for UI and experiments."""
    total = len(valid) + len(rejected)
    bloom_dist = {}
    diff_dist = {}
    for mcq in valid:
        b = mcq.get("bloom_level", "unknown")
        d = mcq.get("difficulty", "unknown")
        bloom_dist[b] = bloom_dist.get(b, 0) + 1
        diff_dist[d] = diff_dist.get(d, 0) + 1

    reject_reasons = {}
    reject_layers = {}
    for r in rejected:
        reason_key = r["reason"].split(":")[0]
        reject_reasons[reason_key] = reject_reasons.get(reason_key, 0) + 1
        layer = r.get("layer", "unknown")
        reject_layers[layer] = reject_layers.get(layer, 0) + 1

    bloom_mismatch_count = sum(1 for m in valid if m.get("bloom_mismatch", False))
    avg_quality = (
        sum(float(m.get("quality_score", 0)) for m in valid) / len(valid)
        if valid else 0
    )

    return {
        "total_raw": total,
        "valid": len(valid),
        "rejected": len(rejected),
        "pass_rate": round(len(valid) / total * 100, 1) if total else 0,
        "bloom_dist": bloom_dist,
        "difficulty_dist": diff_dist,
        "reject_reasons": reject_reasons,
        "reject_layers": reject_layers,
        "bloom_mismatch_count": bloom_mismatch_count,
        "bloom_mismatch_rate": round(bloom_mismatch_count / len(valid) * 100, 1) if valid else 0,
        "average_quality_score": round(avg_quality, 3),
    }


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    sample_raw = [
        {
            "question": "What does the pipe symbol | represent in regular expressions?",
            "A": "Concatenation of two patterns",
            "B": "Disjunction, matching either left or right pattern",
            "C": "Negation of a character class",
            "D": "Repetition of the previous character",
            "answer": "B",
            "explanation": "The pipe is used for disjunction in regex.",
            "bloom_level": "understand",
            "difficulty": "easy",
            "source_chunk": "chunk_001",
        },
        {"question": "What is stemming?", "answer": "A"},
    ]
    valid, rejected = validate_mcqs(sample_raw)
    print(validation_stats(valid, rejected))
    print("Rejected:", rejected)
