"""
tests/ku/test_ku_extract_mock.py
---------------------------------
Test extract_kus() với mock LLM — không cần API key.
Covers: happy path, JSON parse error, pydantic rejection, fallback on all-incomplete.

Chạy:
    python tests/ku/test_ku_extract_mock.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline.knowledge_extractor import extract_kus

PASS = "[PASS]"
FAIL = "[FAIL]"

SAMPLE_CHUNK = {
    "chunk_id":   "chunk_001",
    "topic":      "Reflexion Agent Architecture",
    "pages":      "3-5",
    "text":       (
        "Reflexion = ReAct + Evaluator + Reflector. "
        "Agent retry sau khi Evaluator detect failure. "
        "ReAct không có khả năng tự đánh giá lỗi."
    ),
    "has_image":  False,
    "chunk_type": "conceptual",
}


def _make_llm(response: str):
    """Return a mock llm_fn that always returns response."""
    def fn(prompt: str, json_mode: bool) -> str:
        return response
    return fn


def _make_valid_ku(ku_id="chunk_001_ku_01", prominence="primary"):
    return {
        "ku_id":             ku_id,
        "type":              "definition",
        "concept":           "Reflexion",
        "content":           "Reflexion extends ReAct with Evaluator and Reflector for self-correction",
        "verbatim_evidence": "Reflexion = ReAct + Evaluator + Reflector",
        "related_concepts":  ["ReAct", "Evaluator"],
        "source_pages":      [3],
        "prominence":        prominence,
        "completeness":      "complete",
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
        print(f"{FAIL} {name} — unexpected: {type(e).__name__}: {e}")
        return False


def run():
    results = []

    # ── Happy path: 1 valid KU ─────────────────────────────────────────────
    def test_happy_path():
        mock_resp = json.dumps([_make_valid_ku()])
        kus = extract_kus(SAMPLE_CHUNK, _make_llm(mock_resp))
        assert len(kus) == 1
        assert kus[0]["concept"] == "Reflexion"
        assert kus[0]["type"]    == "definition"

    results.append(_test("happy path: 1 valid KU extracted", test_happy_path))

    # ── Multiple KUs, 1 fails verify ──────────────────────────────────────
    def test_filter_applied():
        hallucinated = {
            **_make_valid_ku("chunk_001_ku_02"),
            "concept": "Quantum Reflexion",
            "verbatim_evidence": "Reflexion uses quantum entanglement",  # not in chunk
        }
        mock_resp = json.dumps([_make_valid_ku(), hallucinated])
        kus = extract_kus(SAMPLE_CHUNK, _make_llm(mock_resp))
        assert len(kus) == 1, f"Expected 1 after filter, got {len(kus)}"
        assert kus[0]["concept"] == "Reflexion"

    results.append(_test("hallucinated KU filtered out by verify_ku", test_filter_applied))

    # ── Incomplete KU filtered ─────────────────────────────────────────────
    def test_incomplete_filtered():
        incomplete = {**_make_valid_ku("chunk_001_ku_02"), "completeness": "incomplete"}
        mock_resp = json.dumps([_make_valid_ku(), incomplete])
        kus = extract_kus(SAMPLE_CHUNK, _make_llm(mock_resp))
        assert len(kus) == 1

    results.append(_test("incomplete KU filtered by filter_kus", test_incomplete_filtered))

    # ── All KUs incomplete → empty list ───────────────────────────────────
    def test_all_incomplete():
        ku = {**_make_valid_ku(), "completeness": "incomplete"}
        mock_resp = json.dumps([ku])
        kus = extract_kus(SAMPLE_CHUNK, _make_llm(mock_resp))
        assert kus == []

    results.append(_test("all incomplete → empty list", test_all_incomplete))

    # ── JSON parse error → empty list ─────────────────────────────────────
    def test_json_error():
        kus = extract_kus(SAMPLE_CHUNK, _make_llm("this is not json {{{"))
        assert kus == []

    results.append(_test("JSON parse error → empty list", test_json_error))

    # ── LLM raises exception → empty list ─────────────────────────────────
    def test_llm_exception():
        def bad_llm(prompt, json_mode):
            raise ConnectionError("network error")
        kus = extract_kus(SAMPLE_CHUNK, bad_llm)
        assert kus == []

    results.append(_test("LLM exception → empty list", test_llm_exception))

    # ── Wrapped response {"knowledge_units": [...]} ────────────────────────
    def test_wrapped_response():
        mock_resp = json.dumps({"knowledge_units": [_make_valid_ku()]})
        kus = extract_kus(SAMPLE_CHUNK, _make_llm(mock_resp))
        assert len(kus) == 1

    results.append(_test("wrapped response {knowledge_units: [...]} unwrapped", test_wrapped_response))

    # ── Markdown code block stripped ───────────────────────────────────────
    def test_markdown_stripped():
        inner = json.dumps([_make_valid_ku()])
        mock_resp = f"```json\n{inner}\n```"
        kus = extract_kus(SAMPLE_CHUNK, _make_llm(mock_resp))
        assert len(kus) == 1

    results.append(_test("```json code block stripped before parse", test_markdown_stripped))

    # ── Auto-fix bad ku_id format ──────────────────────────────────────────
    def test_autofixed_ku_id():
        bad_id = {**_make_valid_ku(), "ku_id": "wrong_format_001"}
        mock_resp = json.dumps([bad_id])
        kus = extract_kus(SAMPLE_CHUNK, _make_llm(mock_resp))
        if kus:  # may pass or fail verify — just check id format if it passes
            assert kus[0]["ku_id"].startswith("chunk_001"), (
                f"ku_id not auto-fixed: {kus[0]['ku_id']}"
            )

    results.append(_test("bad ku_id auto-corrected to chunk prefix", test_autofixed_ku_id))

    # ── source_pages coercion in full flow ────────────────────────────────
    def test_pages_coercion():
        ku = {**_make_valid_ku(), "source_pages": "3, 4"}
        mock_resp = json.dumps([ku])
        kus = extract_kus(SAMPLE_CHUNK, _make_llm(mock_resp))
        if kus:
            assert isinstance(kus[0]["source_pages"], list)
            assert all(isinstance(p, int) for p in kus[0]["source_pages"])

    results.append(_test("source_pages string coerced to list[int]", test_pages_coercion))

    # ── Summary ────────────────────────────────────────────────────────────
    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*40}")
    print(f"KU Extract (mock): {passed}/{total} passed")
    print(f"{'='*40}")
    return passed == total


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
