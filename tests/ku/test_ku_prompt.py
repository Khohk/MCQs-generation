"""
tests/ku/test_ku_prompt.py
--------------------------
Test build_extraction_prompt(): kiểm tra output có đủ cấu trúc không.
Không gọi API — chỉ verify prompt string.

Chạy:
    python tests/ku/test_ku_prompt.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from prompts.ku_extraction_prompt import build_extraction_prompt

PASS = "[PASS]"
FAIL = "[FAIL]"

SAMPLE_CHUNK = {
    "chunk_id":   "chunk_001",
    "topic":      "Reflexion Agent Architecture",
    "pages":      "3-5",
    "text":       (
        "Reflexion = ReAct + Evaluator + Reflector. "
        "Agent retry sau khi Evaluator detect failure. "
        "ReAct không có khả năng tự đánh giá."
    ),
    "has_image":  False,
    "chunk_type": "conceptual",
}


def _test(name: str, fn):
    try:
        fn()
        print(f"{PASS} {name}")
        return True
    except AssertionError as e:
        print(f"{FAIL} {name} — {e}")
        return False
    except Exception as e:
        print(f"{FAIL} {name} — unexpected: {e}")
        return False


def run():
    results = []
    prompt = build_extraction_prompt(SAMPLE_CHUNK)

    def test_is_string():
        assert isinstance(prompt, str)
        assert len(prompt) > 200, "Prompt too short"

    results.append(_test("prompt is non-empty string", test_is_string))

    def test_contains_chunk_text():
        assert SAMPLE_CHUNK["text"] in prompt, "Source text not injected into prompt"

    results.append(_test("source text injected", test_contains_chunk_text))

    def test_contains_chunk_id():
        assert SAMPLE_CHUNK["chunk_id"] in prompt, "chunk_id not in prompt"

    results.append(_test("chunk_id present in prompt", test_contains_chunk_id))

    def test_contains_topic():
        assert SAMPLE_CHUNK["topic"] in prompt

    results.append(_test("topic present in prompt", test_contains_topic))

    def test_contains_ku_types():
        for t in ("definition", "mechanism", "failure_mode", "trade_off", "procedure", "application"):
            assert t in prompt, f"KU type '{t}' missing from prompt"

    results.append(_test("all 6 KU types listed in prompt", test_contains_ku_types))

    def test_contains_prominence():
        for p in ("primary", "supporting", "peripheral"):
            assert p in prompt, f"prominence value '{p}' missing"

    results.append(_test("prominence criteria in prompt", test_contains_prominence))

    def test_contains_completeness():
        assert "incomplete" in prompt
        assert "15 words" in prompt  # completeness criterion 1

    results.append(_test("completeness criteria in prompt", test_contains_completeness))

    def test_contains_fill_gap():
        for keyword in ("Reformulate", "Combine", "Convert", "[...]"):
            assert keyword in prompt, f"Fill gap keyword '{keyword}' missing"

    results.append(_test("fill gap rules present (Reformulate/Combine/Convert/[...])", test_contains_fill_gap))

    def test_contains_ku_id_format():
        assert f"{SAMPLE_CHUNK['chunk_id']}_ku_01" in prompt

    results.append(_test("ku_id format example in prompt", test_contains_ku_id_format))

    def test_no_api_leak():
        # Prompt should not contain any API key pattern
        import re
        assert not re.search(r"AIza[A-Za-z0-9_-]{35}", prompt), "API key leaked into prompt"

    results.append(_test("no API key in prompt", test_no_api_leak))

    def test_verbatim_critical_rule():
        assert "verbatim_evidence" in prompt
        assert "EXACT" in prompt or "exactly" in prompt.lower()

    results.append(_test("verbatim EXACT copy rule present", test_verbatim_critical_rule))

    # Print prompt length for reference
    print(f"\n  Prompt length: {len(prompt)} chars / ~{len(prompt)//4} tokens (est.)")

    # ── Summary ────────────────────────────────────────────────────────────
    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*40}")
    print(f"KU Prompt: {passed}/{total} passed")
    print(f"{'='*40}")
    return passed == total


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
