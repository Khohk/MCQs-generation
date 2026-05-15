"""
tests/mcq/test_mcq_mock.py
---------------------------
Unit tests for mcq_generator — no API key needed.

Covers:
  _filter_anchors     : peripheral / incomplete / short-evidence filtering
  assign_bloom        : ku_type → bloom level mapping
  _validate_mcq       : schema checks (question, options, answer, distinctness)
  generate_mcq        : Tier A vs B dispatch, parse/validation failures
  generate_cross_mcq  : cross-concept prompt + parse path
  run_mcq_generation  : end-to-end with mock LLM, priority sorting
  run_cross_mcq_generation : edge iteration, anchor filter, MCQItem.is_cross

Run:
    python tests/mcq/test_mcq_mock.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline.mcq_generator import (
    MCQItem,
    assign_bloom,
    generate_cross_mcq,
    generate_mcq,
    run_cross_mcq_generation,
    run_mcq_generation,
    _filter_anchors,
    _validate_mcq,
    BLOOM_MAP,
    TIER_B_THRESHOLD,
)
from pipeline.pass2_extractor import Pass2Result

PASS = "[PASS]"
FAIL = "[FAIL]"


# ── Fixtures ───────────────────────────────────────────────────────────────

def _ku(
    ku_id: str,
    concept: str = "TestConcept",
    ku_type: str = "definition",
    prominence: str = "primary",
    completeness: str = "complete",
    evidence: str = "TestConcept provides key functionality via its main mechanism in the system architecture",
    pages: list[int] | None = None,
) -> dict:
    return {
        "ku_id":             ku_id,
        "type":              ku_type,
        "concept":           concept,
        "content":           f"{concept} is an important concept in this domain",
        "verbatim_evidence": evidence,
        "related_kus":       [],
        "related_concepts":  [],
        "source_pages":      pages or [1, 2],
        "prominence":        prominence,
        "completeness":      completeness,
    }


def _valid_mcq_data(answer: str = "A") -> dict:
    return {
        "question":    "Which statement best describes the forget gate mechanism?",
        "options":     {
            "A": "It uses a sigmoid function to decide what to remove from cell state",
            "B": "It adds new information to the hidden state using tanh",
            "C": "It controls the output of the LSTM to the next layer",
            "D": "It resets all memory after each time step unconditionally",
        },
        "answer":      answer,
        "bloom_level": "understand",
        "explanation": "A is correct because...",
    }


def _make_llm(response: str):
    def fn(prompt: str, json_mode: bool) -> str:
        return response
    return fn


def _make_pass2(kus: list[dict], edges: dict | None = None) -> Pass2Result:
    """Build a minimal Pass2Result from KU list + optional edge_types dict."""
    graph: dict[str, list[str]] = {ku["ku_id"]: [] for ku in kus}
    edge_types: dict[tuple[str, str], str] = edges or {}
    for (a, b), rel in edge_types.items():
        if b not in graph[a]:
            graph[a].append(b)
        if a not in graph.setdefault(b, []):
            graph[b].append(a)
    pool: dict[str, list[dict]] = {}
    for ku in kus:
        pool.setdefault(ku["type"], []).append({"concept": ku["concept"], "content": ku["content"]})
    return Pass2Result(all_kus=kus, graph=graph, pool=pool, edge_types=edge_types)


# ── Test runner ────────────────────────────────────────────────────────────

def _test(name: str, fn) -> bool:
    try:
        fn()
        print(f"{PASS} {name}")
        return True
    except AssertionError as e:
        print(f"{FAIL} {name} — {e}")
        return False
    except Exception as e:
        print(f"{FAIL} {name} — unexpected {type(e).__name__}: {e}")
        return False


# ── _filter_anchors ────────────────────────────────────────────────────────

def test_filter_keeps_valid():
    result = _filter_anchors([_ku("seg_001_ku_01")])
    assert len(result) == 1

def test_filter_removes_peripheral():
    result = _filter_anchors([_ku("seg_001_ku_01", prominence="peripheral")])
    assert result == []

def test_filter_removes_incomplete():
    result = _filter_anchors([_ku("seg_001_ku_01", completeness="incomplete")])
    assert result == []

def test_filter_removes_short_evidence():
    result = _filter_anchors([_ku("seg_001_ku_01", evidence="too short")])
    assert result == []

def test_filter_mixed():
    kus = [
        _ku("seg_001_ku_01"),                           # keep
        _ku("seg_001_ku_02", prominence="peripheral"),  # drop
        _ku("seg_001_ku_03", completeness="incomplete"),# drop
        _ku("seg_001_ku_04", evidence="short"),         # drop
        _ku("seg_001_ku_05", prominence="supporting"),  # keep
    ]
    result = _filter_anchors(kus)
    assert len(result) == 2
    assert {r["ku_id"] for r in result} == {"seg_001_ku_01", "seg_001_ku_05"}


# ── assign_bloom ───────────────────────────────────────────────────────────

def test_bloom_all_types():
    assert assign_bloom("definition")   == "understand"
    assert assign_bloom("mechanism")    == "understand"
    assert assign_bloom("procedure")    == "apply"
    assert assign_bloom("trade_off")    == "evaluate"
    assert assign_bloom("failure_mode") == "analyze"
    assert assign_bloom("application")  == "apply"

def test_bloom_unknown_defaults():
    assert assign_bloom("unknown_type") == "understand"

def test_bloom_map_complete():
    ku_types = {"definition", "mechanism", "procedure", "trade_off",
                "failure_mode", "application"}
    assert set(BLOOM_MAP.keys()) == ku_types


# ── _validate_mcq ──────────────────────────────────────────────────────────

def test_validate_valid():
    assert _validate_mcq(_valid_mcq_data()) is True

def test_validate_missing_question():
    d = _valid_mcq_data()
    d["question"] = ""
    assert _validate_mcq(d) is False

def test_validate_missing_option_key():
    d = _valid_mcq_data()
    del d["options"]["D"]
    assert _validate_mcq(d) is False

def test_validate_answer_not_in_abcd():
    d = _valid_mcq_data()
    d["answer"] = "E"
    assert _validate_mcq(d) is False

def test_validate_duplicate_options():
    d = _valid_mcq_data()
    d["options"]["B"] = d["options"]["A"]
    assert _validate_mcq(d) is False

def test_validate_empty_option():
    d = _valid_mcq_data()
    d["options"]["C"] = ""
    assert _validate_mcq(d) is False

def test_validate_answer_case_insensitive():
    # _validate_mcq does .upper() internally, so "b" is treated as "B" → passes
    d = _valid_mcq_data(answer="b")
    assert _validate_mcq(d) is True

def test_validate_not_dict():
    assert _validate_mcq([]) is False
    assert _validate_mcq("string") is False


# ── generate_mcq ───────────────────────────────────────────────────────────

def test_generate_mcq_tier_a_three_distractors():
    distractors = [
        _ku("seg_002_ku_01", "Input Gate"),
        _ku("seg_002_ku_02", "Output Gate"),
        _ku("seg_002_ku_03", "Cell State"),
    ]
    llm = _make_llm(json.dumps(_valid_mcq_data("B")))
    data, tier = generate_mcq(_ku("seg_001_ku_01", "Forget Gate"), distractors, "understand", llm)
    assert tier == "A"
    assert data is not None
    assert data["answer"] == "B"

def test_generate_mcq_tier_b_one_distractor():
    distractors = [_ku("seg_002_ku_01", "Input Gate")]
    assert len(distractors) < TIER_B_THRESHOLD
    llm = _make_llm(json.dumps(_valid_mcq_data("C")))
    data, tier = generate_mcq(_ku("seg_001_ku_01", "Forget Gate"), distractors, "understand", llm)
    assert tier == "B"
    assert data is not None

def test_generate_mcq_tier_b_zero_distractors():
    llm = _make_llm(json.dumps(_valid_mcq_data()))
    data, tier = generate_mcq(_ku("seg_001_ku_01"), [], "understand", llm)
    assert tier == "B"

def test_generate_mcq_bad_json_returns_none():
    llm = _make_llm("not valid json {{{")
    data, tier = generate_mcq(_ku("seg_001_ku_01"), [], "understand", llm)
    assert data is None
    assert tier == ""

def test_generate_mcq_validation_fail_returns_none():
    bad = _valid_mcq_data()
    bad["answer"] = "Z"  # invalid
    llm = _make_llm(json.dumps(bad))
    data, tier = generate_mcq(_ku("seg_001_ku_01"), [], "understand", llm)
    assert data is None

def test_generate_mcq_empty_response():
    llm = _make_llm("")
    data, tier = generate_mcq(_ku("seg_001_ku_01"), [], "understand", llm)
    assert data is None

def test_generate_mcq_llm_exception():
    def bad_llm(p, j):
        raise ConnectionError("network down")
    data, tier = generate_mcq(_ku("seg_001_ku_01"), [], "understand", bad_llm)
    assert data is None

def test_generate_mcq_markdown_stripped():
    inner = json.dumps(_valid_mcq_data())
    llm = _make_llm(f"```json\n{inner}\n```")
    data, tier = generate_mcq(_ku("seg_001_ku_01"), [], "understand", llm)
    assert data is not None


# ── generate_cross_mcq ─────────────────────────────────────────────────────

def test_generate_cross_mcq_ok():
    llm = _make_llm(json.dumps(_valid_mcq_data("D")))
    result = generate_cross_mcq(
        _ku("seg_001_ku_01", "LSTM"),
        _ku("seg_002_ku_01", "GRU"),
        "CONTRASTS_WITH",
        "evaluate",
        llm,
    )
    assert result is not None
    assert result["answer"] == "D"

def test_generate_cross_mcq_bad_json():
    llm = _make_llm("{broken")
    result = generate_cross_mcq(
        _ku("seg_001_ku_01", "LSTM"),
        _ku("seg_002_ku_01", "GRU"),
        "CONTRASTS_WITH",
        "evaluate",
        llm,
    )
    assert result is None


# ── run_mcq_generation ─────────────────────────────────────────────────────

def test_run_mcq_empty_pass2():
    p2 = Pass2Result()
    result = run_mcq_generation(p2, _make_llm("{}"))
    assert result == []

def test_run_mcq_generates_items():
    kus = [
        _ku("seg_001_ku_01", "Forget Gate", prominence="primary"),
        _ku("seg_001_ku_02", "Input Gate",  prominence="supporting"),
    ]
    p2  = _make_pass2(kus)
    llm = _make_llm(json.dumps(_valid_mcq_data()))
    mcqs = run_mcq_generation(p2, llm, delay_between=0)
    assert len(mcqs) == 2
    for m in mcqs:
        assert m.ok
        assert m.anchor_ku_id in {ku["ku_id"] for ku in kus}
        assert m.bloom_level in {"remember", "understand", "apply", "analyze", "evaluate", "create"}

def test_run_mcq_sorted_by_priority():
    kus = [
        _ku("seg_001_ku_01", prominence="supporting"),
        _ku("seg_001_ku_02", prominence="primary"),
    ]
    p2  = _make_pass2(kus)
    llm = _make_llm(json.dumps(_valid_mcq_data()))
    mcqs = run_mcq_generation(p2, llm, delay_between=0)
    priorities = [m.priority for m in mcqs]
    assert priorities == sorted(priorities)

def test_run_mcq_skips_on_llm_failure():
    kus = [_ku("seg_001_ku_01"), _ku("seg_001_ku_02")]
    p2  = _make_pass2(kus)
    # LLM always returns invalid JSON
    mcqs = run_mcq_generation(p2, _make_llm("bad"), delay_between=0)
    assert mcqs == []

def test_run_mcq_mcq_id_format():
    kus = [_ku("seg_001_ku_01", "Forget Gate")]
    p2  = _make_pass2(kus)
    llm = _make_llm(json.dumps(_valid_mcq_data()))
    mcqs = run_mcq_generation(p2, llm, delay_between=0)
    assert len(mcqs) == 1
    # mcq_id = {ku_id}_{bloom_level}
    assert mcqs[0].mcq_id.startswith("seg_001_ku_01_")

def test_run_mcq_is_not_cross():
    kus = [_ku("seg_001_ku_01")]
    p2  = _make_pass2(kus)
    llm = _make_llm(json.dumps(_valid_mcq_data()))
    mcqs = run_mcq_generation(p2, llm, delay_between=0)
    assert len(mcqs) == 1
    assert not mcqs[0].is_cross
    assert mcqs[0].anchor_ku_id_b == ""
    assert mcqs[0].edge_relation  == ""


# ── run_cross_mcq_generation ───────────────────────────────────────────────

def test_cross_run_empty_pass2():
    p2 = Pass2Result()
    result = run_cross_mcq_generation(p2, _make_llm("{}"))
    assert result == []

def test_cross_run_no_edges():
    kus = [_ku("seg_001_ku_01"), _ku("seg_001_ku_02")]
    p2  = _make_pass2(kus)  # no edges
    result = run_cross_mcq_generation(p2, _make_llm(json.dumps(_valid_mcq_data())), delay_between=0)
    assert result == []

def test_cross_run_with_horizontal_edge():
    id_a, id_b = "seg_001_ku_01", "seg_002_ku_01"
    kus = [
        _ku(id_a, "LSTM"),
        _ku(id_b, "GRU"),
    ]
    edges = {(id_a, id_b): "CONTRASTS_WITH"}
    p2  = _make_pass2(kus, edges)
    llm = _make_llm(json.dumps(_valid_mcq_data()))
    mcqs = run_cross_mcq_generation(p2, llm, delay_between=0)
    assert len(mcqs) == 1
    m = mcqs[0]
    assert m.is_cross
    assert m.anchor_ku_id   == id_a
    assert m.anchor_ku_id_b == id_b
    assert m.edge_relation  == "CONTRASTS_WITH"
    assert m.priority       == 5

def test_cross_run_structural_skipped_when_horizontal_only():
    id_a, id_b = "seg_001_ku_01", "seg_002_ku_01"
    kus   = [_ku(id_a, "A"), _ku(id_b, "B")]
    edges = {(id_a, id_b): "ENABLES"}
    p2    = _make_pass2(kus, edges)
    mcqs  = run_cross_mcq_generation(p2, _make_llm(json.dumps(_valid_mcq_data())),
                                     horizontal_only=True, delay_between=0)
    assert mcqs == []

def test_cross_run_structural_included_when_flag_false():
    id_a, id_b = "seg_001_ku_01", "seg_002_ku_01"
    kus   = [_ku(id_a, "A"), _ku(id_b, "B")]
    edges = {(id_a, id_b): "ENABLES"}
    p2    = _make_pass2(kus, edges)
    mcqs  = run_cross_mcq_generation(p2, _make_llm(json.dumps(_valid_mcq_data())),
                                     horizontal_only=False, delay_between=0)
    assert len(mcqs) == 1
    assert mcqs[0].priority == 6

def test_cross_run_skips_non_anchor_ku():
    id_a, id_b = "seg_001_ku_01", "seg_002_ku_01"
    kus = [
        _ku(id_a, "LSTM"),
        _ku(id_b, "GRU", completeness="incomplete"),  # will be filtered from anchors
    ]
    edges = {(id_a, id_b): "CONTRASTS_WITH"}
    p2    = _make_pass2(kus, edges)
    mcqs  = run_cross_mcq_generation(p2, _make_llm(json.dumps(_valid_mcq_data())), delay_between=0)
    assert mcqs == []

def test_cross_run_source_pages_merged():
    id_a, id_b = "seg_001_ku_01", "seg_002_ku_01"
    kus = [
        _ku(id_a, "LSTM", pages=[1, 2]),
        _ku(id_b, "GRU",  pages=[3, 4]),
    ]
    edges = {(id_a, id_b): "ALTERNATIVE_TO"}
    p2    = _make_pass2(kus, edges)
    llm   = _make_llm(json.dumps(_valid_mcq_data()))
    mcqs  = run_cross_mcq_generation(p2, llm, delay_between=0)
    assert len(mcqs) == 1
    assert set(mcqs[0].source_pages) == {1, 2, 3, 4}


# ── MCQItem.ok ─────────────────────────────────────────────────────────────

def test_mcq_item_ok_true():
    m = MCQItem(
        mcq_id="seg_001_ku_01_understand",
        anchor_ku_id="seg_001_ku_01",
        anchor_concept="Forget Gate",
        anchor_type="definition",
        question="Which describes the forget gate?",
        options={"A": "opt1", "B": "opt2", "C": "opt3", "D": "opt4"},
        answer="A",
        bloom_level="understand",
    )
    assert m.ok is True

def test_mcq_item_ok_false_missing_question():
    m = MCQItem(
        mcq_id="x", anchor_ku_id="x", anchor_concept="X", anchor_type="definition",
        options={"A": "a", "B": "b", "C": "c", "D": "d"}, answer="A",
    )
    assert m.ok is False


# ── Main ───────────────────────────────────────────────────────────────────

def run():
    tests = [
        # _filter_anchors
        ("filter: keeps valid KU",                   test_filter_keeps_valid),
        ("filter: drops peripheral",                 test_filter_removes_peripheral),
        ("filter: drops incomplete",                 test_filter_removes_incomplete),
        ("filter: drops short evidence",             test_filter_removes_short_evidence),
        ("filter: mixed batch",                      test_filter_mixed),
        # assign_bloom
        ("bloom: all 6 types mapped correctly",      test_bloom_all_types),
        ("bloom: unknown type → understand",         test_bloom_unknown_defaults),
        ("bloom: BLOOM_MAP covers all KU types",     test_bloom_map_complete),
        # _validate_mcq
        ("validate: valid MCQ passes",               test_validate_valid),
        ("validate: empty question → False",         test_validate_missing_question),
        ("validate: missing option key → False",     test_validate_missing_option_key),
        ("validate: answer not A-D → False",         test_validate_answer_not_in_abcd),
        ("validate: duplicate options → False",      test_validate_duplicate_options),
        ("validate: empty option value → False",     test_validate_empty_option),
        ("validate: lowercase answer uppercased → True", test_validate_answer_case_insensitive),
        ("validate: non-dict input → False",         test_validate_not_dict),
        # generate_mcq
        ("generate_mcq: 3 distractors → Tier A",     test_generate_mcq_tier_a_three_distractors),
        ("generate_mcq: 1 distractor → Tier B",      test_generate_mcq_tier_b_one_distractor),
        ("generate_mcq: 0 distractors → Tier B",     test_generate_mcq_tier_b_zero_distractors),
        ("generate_mcq: bad JSON → (None, '')",      test_generate_mcq_bad_json_returns_none),
        ("generate_mcq: invalid MCQ → (None, '')",   test_generate_mcq_validation_fail_returns_none),
        ("generate_mcq: empty response → None",      test_generate_mcq_empty_response),
        ("generate_mcq: LLM exception → None",       test_generate_mcq_llm_exception),
        ("generate_mcq: ```json block stripped",     test_generate_mcq_markdown_stripped),
        # generate_cross_mcq
        ("generate_cross_mcq: valid → data dict",    test_generate_cross_mcq_ok),
        ("generate_cross_mcq: bad JSON → None",      test_generate_cross_mcq_bad_json),
        # run_mcq_generation
        ("run_mcq: empty Pass2Result → []",          test_run_mcq_empty_pass2),
        ("run_mcq: generates MCQItems",              test_run_mcq_generates_items),
        ("run_mcq: sorted by priority",              test_run_mcq_sorted_by_priority),
        ("run_mcq: all LLM failures → []",           test_run_mcq_skips_on_llm_failure),
        ("run_mcq: mcq_id = {ku_id}_{bloom}",        test_run_mcq_mcq_id_format),
        ("run_mcq: is_cross=False, no b/relation",   test_run_mcq_is_not_cross),
        # run_cross_mcq_generation
        ("cross_run: empty Pass2Result → []",        test_cross_run_empty_pass2),
        ("cross_run: no edges → []",                 test_cross_run_no_edges),
        ("cross_run: horizontal edge → 1 CrossMCQ", test_cross_run_with_horizontal_edge),
        ("cross_run: ENABLES skipped (h-only=True)", test_cross_run_structural_skipped_when_horizontal_only),
        ("cross_run: ENABLES included (h-only=F)",   test_cross_run_structural_included_when_flag_false),
        ("cross_run: non-anchor KU → edge skipped",  test_cross_run_skips_non_anchor_ku),
        ("cross_run: source_pages merged from both", test_cross_run_source_pages_merged),
        # MCQItem
        ("MCQItem.ok: valid item → True",            test_mcq_item_ok_true),
        ("MCQItem.ok: missing question → False",     test_mcq_item_ok_false_missing_question),
    ]

    results = [_test(name, fn) for name, fn in tests]
    passed  = sum(results)
    total   = len(results)
    print(f"\n{'='*50}")
    print(f"MCQ Mock tests: {passed}/{total} passed")
    print(f"{'='*50}")
    return passed == total


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
