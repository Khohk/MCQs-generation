"""
pipeline/validator.py
---------------------
List[MCQ raw] → (List[MCQ valid], List[{mcq, reason}])

Kiểm tra schema, field values, và loại duplicate.
"""

# ── Whitelists ─────────────────────────────────────────────────────
VALID_ANSWERS     = {"A", "B", "C", "D"}
VALID_BLOOM       = {"remember", "understand", "apply", "analyze", "evaluate", "create"}
VALID_DIFFICULTY  = {"easy", "medium", "hard"}
REQUIRED_FIELDS   = {"question", "A", "B", "C", "D", "answer",
                     "explanation", "bloom_level", "difficulty", "source_chunk"}
MIN_QUESTION_LEN  = 10

# Bloom levels hợp lệ cho từng difficulty
# Dựa trên Bloom's Taxonomy: easy=lower order, hard=higher order
BLOOM_BY_DIFFICULTY = {
    "easy":   {"remember", "understand"},
    "medium": {"apply", "analyze"},
    "hard":   {"evaluate", "create"},
}


# ── Main function ──────────────────────────────────────────────────

def validate_mcqs(
    raw_mcqs: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Validate và filter MCQ list.

    Returns:
        (valid_mcqs, rejected) where rejected = [{mcq, reason}]
    """
    valid   = []
    rejected = []
    seen_questions = set()

    for mcq in raw_mcqs:
        reason = _check(mcq, seen_questions)
        if reason:
            rejected.append({"mcq": mcq, "reason": reason})
        else:
            valid.append(mcq)
            seen_questions.add(_normalize_q(mcq["question"]))

    return valid, rejected


# ── Validation rules ───────────────────────────────────────────────

def _check(mcq: dict, seen_questions: set) -> str | None:
    """
    Trả về reason string nếu MCQ invalid, None nếu OK.
    Kiểm tra theo thứ tự ưu tiên.
    """
    # 1. Thiếu field
    missing = REQUIRED_FIELDS - set(mcq.keys())
    if missing:
        return f"Missing fields: {missing}"

    # 2. Question rỗng hoặc quá ngắn
    q = str(mcq.get("question", "")).strip()
    if not q or len(q) < MIN_QUESTION_LEN:
        return f"Question too short ({len(q)} chars)"

    # 3. Options rỗng
    for opt in ["A", "B", "C", "D"]:
        val = str(mcq.get(opt, "")).strip()
        if not val:
            return f"Option {opt} is empty"

    # 4. Answer không hợp lệ
    answer = str(mcq.get("answer", "")).strip().upper()
    if answer not in VALID_ANSWERS:
        return f"Invalid answer: '{answer}' (must be A/B/C/D)"

    # 5. bloom_level không hợp lệ
    bloom = str(mcq.get("bloom_level", "")).strip().lower()
    if bloom not in VALID_BLOOM:
        return f"Invalid bloom_level: '{bloom}'"

    # 6. difficulty không hợp lệ
    diff = str(mcq.get("difficulty", "")).strip().lower()
    if diff not in VALID_DIFFICULTY:
        return f"Invalid difficulty: '{diff}'"

    # 7. Explanation rỗng
    exp = str(mcq.get("explanation", "")).strip()
    if not exp:
        return "Explanation is empty"

    # 8. Correct answer trùng với distractor
    correct_text = str(mcq.get(answer, "")).strip().lower()
    for opt in ["A", "B", "C", "D"]:
        if opt != answer:
            dist_text = str(mcq.get(opt, "")).strip().lower()
            if correct_text == dist_text:
                return f"Option {opt} is identical to correct answer {answer}"

    # 9. Duplicate question
    if _normalize_q(q) in seen_questions:
        return "Duplicate question"

    # 10. Bloom/Difficulty consistency
    # Cảnh báo nếu không khớp nhưng KHÔNG reject — vì Gemini hay ignore
    # Chỉ reject trường hợp cực đoan: hard nhưng bloom là remember/understand
    expected_blooms = BLOOM_BY_DIFFICULTY.get(diff, set())
    if expected_blooms and bloom not in expected_blooms:
        # Log warning nhưng chỉ reject nếu hard+lower_order (rõ ràng sai)
        if diff == "hard" and bloom in {"remember", "understand"}:
            return f"Bloom/Difficulty mismatch: difficulty=hard nhung bloom={bloom} (phai la evaluate/create)"

    # Normalize answer field (phòng trường hợp Gemini trả về "b" thay vì "B")
    mcq["answer"] = answer
    mcq["bloom_level"] = bloom
    mcq["difficulty"] = diff
    # Ghi thêm flag nếu bloom không khớp expected (dùng cho scorer.py sau)
    expected = BLOOM_BY_DIFFICULTY.get(diff, set())
    mcq["bloom_mismatch"] = bloom not in expected if expected else False

    return None


def _normalize_q(q: str) -> str:
    """Chuẩn hóa question để so sánh duplicate."""
    return " ".join(q.lower().split())


# ── Stats helper ───────────────────────────────────────────────────

def validation_stats(valid: list[dict], rejected: list[dict]) -> dict:
    """Trả về dict thống kê — dùng trong Streamlit UI."""
    total = len(valid) + len(rejected)
    bloom_dist = {}
    diff_dist  = {}
    for mcq in valid:
        b = mcq.get("bloom_level", "unknown")
        d = mcq.get("difficulty", "unknown")
        bloom_dist[b] = bloom_dist.get(b, 0) + 1
        diff_dist[d]  = diff_dist.get(d, 0) + 1

    reject_reasons = {}
    for r in rejected:
        reason_key = r["reason"].split(":")[0]  # group by reason type
        reject_reasons[reason_key] = reject_reasons.get(reason_key, 0) + 1

    # Dem so MCQ co bloom mismatch (warning, khong phai reject)
    bloom_mismatch_count = sum(1 for m in valid if m.get("bloom_mismatch", False))

    return {
        "total_raw":           total,
        "valid":               len(valid),
        "rejected":            len(rejected),
        "pass_rate":           round(len(valid) / total * 100, 1) if total else 0,
        "bloom_dist":          bloom_dist,
        "difficulty_dist":     diff_dist,
        "reject_reasons":      reject_reasons,
        "bloom_mismatch_count": bloom_mismatch_count,
        "bloom_mismatch_rate": round(bloom_mismatch_count / len(valid) * 100, 1) if valid else 0,
    }


# ── CLI test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Test với data giả: mix valid + invalid
    sample_raw = [
        {   # VALID
            "question": "What does the pipe symbol | represent in regular expressions?",
            "A": "Concatenation of two patterns",
            "B": "Disjunction — matches either left or right pattern",
            "C": "Negation of a character class",
            "D": "Repetition of the previous character",
            "answer": "B",
            "explanation": "The pipe | is used for disjunction in regex, e.g., 'cat|dog' matches either 'cat' or 'dog'.",
            "bloom_level": "understand",
            "difficulty": "easy",
            "source_chunk": "chunk_001",
        },
        {   # INVALID: missing fields
            "question": "What is stemming?",
            "answer": "A",
        },
        {   # INVALID: bad answer
            "question": "Which algorithm reduces words to their root form using rewrite rules?",
            "A": "BPE", "B": "Word2Vec", "C": "Porter Stemmer", "D": "BERT",
            "answer": "X",
            "explanation": "Porter Stemmer uses rule-based approach.",
            "bloom_level": "remember", "difficulty": "easy", "source_chunk": "chunk_002",
        },
        {   # VALID
            "question": "In Heaps Law, what does a beta value between 0.67 and 0.75 indicate?",
            "A": "Vocabulary size grows linearly with corpus size",
            "B": "Vocabulary size grows faster than square root of token count",
            "C": "Vocabulary size remains constant after a threshold",
            "D": "Vocabulary size decreases as corpus grows",
            "answer": "B",
            "explanation": "Heaps Law states |V| = kN^beta; with beta ~0.67-0.75, vocabulary grows sub-linearly but faster than square root.",
            "bloom_level": "analyze",
            "difficulty": "medium",
            "source_chunk": "chunk_003",
        },
        {   # INVALID: duplicate
            "question": "What does the pipe symbol | represent in regular expressions?",
            "A": "A", "B": "Disjunction", "C": "C", "D": "D",
            "answer": "B", "explanation": "dup",
            "bloom_level": "understand", "difficulty": "easy", "source_chunk": "chunk_001",
        },
    ]

    valid, rejected = validate_mcqs(sample_raw)
    stats = validation_stats(valid, rejected)

    print(f"\n{'='*50}")
    print(f"  Total raw : {stats['total_raw']}")
    print(f"  Valid     : {stats['valid']}  ({stats['pass_rate']}%)")
    print(f"  Rejected  : {stats['rejected']}")
    print(f"{'='*50}")

    print(f"\nBloom distribution : {stats['bloom_dist']}")
    print(f"Difficulty dist    : {stats['difficulty_dist']}")
    print(f"Reject reasons     : {stats['reject_reasons']}")

    print(f"\n--- Rejected detail ---")
    for r in rejected:
        print(f"  [{r['reason']}] Q: {str(r['mcq'].get('question',''))[:50]}")

    print(f"\n--- Valid MCQs ---")
    for i, mcq in enumerate(valid, 1):
        print(f"  {i}. [{mcq['bloom_level']}/{mcq['difficulty']}] {mcq['question'][:60]}")