"""
tests/ku/test_ku_verify.py
--------------------------
Test verify_ku() và filter_kus().
Covers: grounded, hallucinated, Combine case, incomplete filter.

Chạy:
    python tests/ku/test_ku_verify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline.knowledge_extractor import verify_ku, filter_kus

PASS = "[PASS]"
FAIL = "[FAIL]"


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


CHUNK_TEXT = (
    "Reflexion = ReAct + Evaluator + Reflector. "
    "Agent retry sau khi Evaluator detect failure. "
    "LSTM có cell state. cell state mang thông tin dài hạn."
)


def run():
    results = []

    # ── verify_ku: grounded ────────────────────────────────────────────────
    def test_grounded():
        ku = {"verbatim_evidence": "Reflexion = ReAct + Evaluator + Reflector",
              "completeness": "complete"}
        assert verify_ku(ku, CHUNK_TEXT) is True

    results.append(_test("grounded KU passes", test_grounded))

    # ── verify_ku: hallucinated ────────────────────────────────────────────
    def test_hallucinated():
        ku = {"verbatim_evidence": "Reflexion uses quantum entanglement for self-correction",
              "completeness": "complete"}
        assert verify_ku(ku, CHUNK_TEXT) is False

    results.append(_test("hallucinated evidence rejected", test_hallucinated))

    # ── verify_ku: Combine case with [...] separator ───────────────────────
    def test_combine():
        ku = {"verbatim_evidence": "LSTM có cell state [...] cell state mang thông tin dài hạn",
              "completeness": "complete"}
        assert verify_ku(ku, CHUNK_TEXT) is True

    results.append(_test("Combine case with [...] passes", test_combine))

    # ── verify_ku: Combine — one fragment hallucinated ─────────────────────
    def test_combine_partial_hallucination():
        ku = {"verbatim_evidence": "LSTM có cell state [...] LSTM uses quantum gates",
              "completeness": "complete"}
        assert verify_ku(ku, CHUNK_TEXT) is False

    results.append(_test("Combine with hallucinated fragment rejected", test_combine_partial_hallucination))

    # ── verify_ku: empty evidence ──────────────────────────────────────────
    def test_empty_evidence():
        ku = {"verbatim_evidence": "", "completeness": "complete"}
        assert verify_ku(ku, CHUNK_TEXT) is False

    results.append(_test("empty verbatim_evidence rejected", test_empty_evidence))

    # ── verify_ku: missing evidence key ───────────────────────────────────
    def test_missing_evidence():
        ku = {"completeness": "complete"}
        assert verify_ku(ku, CHUNK_TEXT) is False

    results.append(_test("missing verbatim_evidence key rejected", test_missing_evidence))

    # ── verify_ku: partial match still passes (subset of chunk) ───────────
    def test_partial_match():
        # all words in evidence appear in chunk -> recall = 1.0
        ku = {"verbatim_evidence": "Agent retry sau khi Evaluator detect failure",
              "completeness": "complete"}
        assert verify_ku(ku, CHUNK_TEXT) is True

    results.append(_test("partial sentence grounded", test_partial_match))

    # ── filter_kus: only complete + verified KUs pass ─────────────────────
    def test_filter():
        kus = [
            {"ku_id": "ku_01", "verbatim_evidence": "Reflexion = ReAct + Evaluator + Reflector",
             "completeness": "complete"},
            {"ku_id": "ku_02", "verbatim_evidence": "invented evidence not in chunk",
             "completeness": "complete"},
            {"ku_id": "ku_03", "verbatim_evidence": "Agent retry sau khi Evaluator detect failure",
             "completeness": "incomplete"},  # incomplete → must be filtered
        ]
        kept = filter_kus(kus, CHUNK_TEXT)
        ids  = [k["ku_id"] for k in kept]
        assert "ku_01" in ids,     "grounded+complete KU should pass"
        assert "ku_02" not in ids, "hallucinated KU should be filtered"
        assert "ku_03" not in ids, "incomplete KU should be filtered"
        assert len(kept) == 1

    results.append(_test("filter_kus: grounded+complete only", test_filter))

    # ── filter_kus: all fail → empty list ─────────────────────────────────
    def test_filter_all_fail():
        kus = [
            {"ku_id": "ku_01", "verbatim_evidence": "nonsense fabricated text", "completeness": "complete"},
            {"ku_id": "ku_02", "verbatim_evidence": "another fake sentence",    "completeness": "incomplete"},
        ]
        assert filter_kus(kus, CHUNK_TEXT) == []

    results.append(_test("filter_kus all fail → empty list", test_filter_all_fail))

    # ── Summary ────────────────────────────────────────────────────────────
    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*40}")
    print(f"KU Verify: {passed}/{total} passed")
    print(f"{'='*40}")
    return passed == total


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
