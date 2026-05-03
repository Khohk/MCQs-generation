"""
tests/ku/test_ku_schema.py
--------------------------
Test KUItem Pydantic schema: validation, coercion, rejection.

Chạy:
    python tests/ku/test_ku_schema.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pydantic import ValidationError
from pipeline.schemas import KUItem

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


def run():
    results = []

    # ── Valid KU ────────────────────────────────────────────────────────────
    def test_valid_ku():
        ku = KUItem.model_validate({
            "ku_id":             "chunk_001_ku_01",
            "type":              "definition",
            "concept":           "Reflexion",
            "content":           "Reflexion extends ReAct with Evaluator and Reflector",
            "verbatim_evidence": "Reflexion = ReAct + Evaluator + Reflector",
            "related_concepts":  ["ReAct", "Evaluator"],
            "source_pages":      [3, 4],
            "prominence":        "primary",
            "completeness":      "complete",
        })
        assert ku.type == "definition"
        assert ku.prominence == "primary"

    results.append(_test("valid KU parses correctly", test_valid_ku))

    # ── source_pages coercion: int -> [int] ─────────────────────────────────
    def test_pages_int():
        ku = KUItem.model_validate({
            "ku_id": "c_ku_01", "type": "mechanism", "concept": "X",
            "content": "X works by doing Y", "verbatim_evidence": "X does Y",
            "related_concepts": [], "source_pages": 5,
            "prominence": "supporting", "completeness": "complete",
        })
        assert ku.source_pages == [5]

    results.append(_test("source_pages int -> [int] coercion", test_pages_int))

    # ── source_pages coercion: "3, 4" -> [3, 4] ────────────────────────────
    def test_pages_str():
        ku = KUItem.model_validate({
            "ku_id": "c_ku_01", "type": "mechanism", "concept": "X",
            "content": "X works by doing Y", "verbatim_evidence": "X does Y",
            "related_concepts": [], "source_pages": "3, 4",
            "prominence": "supporting", "completeness": "complete",
        })
        assert ku.source_pages == [3, 4]

    results.append(_test("source_pages '3, 4' -> [3, 4] coercion", test_pages_str))

    # ── related_concepts coercion: str -> [str] ─────────────────────────────
    def test_related_str():
        ku = KUItem.model_validate({
            "ku_id": "c_ku_01", "type": "definition", "concept": "X",
            "content": "X is Y", "verbatim_evidence": "X is Y",
            "related_concepts": "ReAct", "source_pages": [1],
            "prominence": "primary", "completeness": "complete",
        })
        assert ku.related_concepts == ["ReAct"]

    results.append(_test("related_concepts str -> [str] coercion", test_related_str))

    # ── Invalid type -> reject ──────────────────────────────────────────────
    def test_invalid_type():
        raised = False
        try:
            KUItem.model_validate({
                "ku_id": "c_ku_01", "type": "unknown_type", "concept": "X",
                "content": "X is Y", "verbatim_evidence": "X is Y",
                "related_concepts": [], "source_pages": [1],
                "prominence": "primary", "completeness": "complete",
            })
        except ValidationError:
            raised = True
        assert raised, "ValidationError expected for invalid type"

    results.append(_test("invalid type rejected", test_invalid_type))

    # ── Invalid prominence -> reject ────────────────────────────────────────
    def test_invalid_prominence():
        raised = False
        try:
            KUItem.model_validate({
                "ku_id": "c_ku_01", "type": "definition", "concept": "X",
                "content": "X is Y", "verbatim_evidence": "X is Y",
                "related_concepts": [], "source_pages": [1],
                "prominence": "central",  # invalid
                "completeness": "complete",
            })
        except ValidationError:
            raised = True
        assert raised, "ValidationError expected for invalid prominence"

    results.append(_test("invalid prominence rejected", test_invalid_prominence))

    # ── All 6 KU types valid ───────────────────────────────────────────────
    def test_all_types():
        for t in ("definition", "mechanism", "failure_mode",
                  "trade_off", "procedure", "application"):
            KUItem.model_validate({
                "ku_id": f"c_ku_01", "type": t, "concept": "X",
                "content": "X is Y", "verbatim_evidence": "X is Y",
                "related_concepts": [], "source_pages": [1],
                "prominence": "primary", "completeness": "complete",
            })

    results.append(_test("all 6 KU types accepted", test_all_types))

    # ── Summary ────────────────────────────────────────────────────────────
    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*40}")
    print(f"KU Schema: {passed}/{total} passed")
    print(f"{'='*40}")
    return passed == total


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
