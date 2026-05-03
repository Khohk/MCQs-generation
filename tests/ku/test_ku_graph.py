"""
tests/ku/test_ku_graph.py
-------------------------
Test dedup_kus(), build_ku_graph(), build_distractor_pool(), compute_priority().

Chạy:
    python tests/ku/test_ku_graph.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline.knowledge_extractor import (
    dedup_kus, build_ku_graph, build_distractor_pool, compute_priority
)

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


# ── Sample KU set ──────────────────────────────────────────────────────────
SAMPLE_KUS = [
    {
        "ku_id": "c1_ku_01", "type": "definition",
        "concept": "Reflexion",
        "content": "Reflexion extends ReAct with Evaluator and Reflector",
        "related_concepts": ["ReAct", "Evaluator"],
        "prominence": "primary", "completeness": "complete",
    },
    {
        "ku_id": "c1_ku_02", "type": "mechanism",
        "concept": "ReAct",
        "content": "ReAct generates action then observes result",
        "related_concepts": ["Reflexion"],
        "prominence": "supporting", "completeness": "complete",
    },
    {
        "ku_id": "c1_ku_03", "type": "failure_mode",
        "concept": "ReAct limitation",
        "content": "ReAct cannot self-evaluate, errors propagate",
        "related_concepts": ["ReAct", "Reflexion"],
        "prominence": "supporting", "completeness": "complete",
    },
    {
        "ku_id": "c2_ku_01", "type": "definition",
        "concept": "Evaluator",
        "content": "Evaluator component judges agent action quality",
        "related_concepts": ["Reflexion"],
        "prominence": "supporting", "completeness": "complete",
    },
    {
        "ku_id": "c2_ku_02", "type": "procedure",
        "concept": "Isolated concept",
        "content": "A standalone concept with no connections",
        "related_concepts": [],
        "prominence": "peripheral", "completeness": "complete",
    },
]


def run():
    results = []

    # ── dedup_kus ──────────────────────────────────────────────────────────
    def test_dedup_exact():
        kus = [
            {"type": "definition", "concept": "Reflexion", "ku_id": "c1_ku_01"},
            {"type": "definition", "concept": "Reflexion", "ku_id": "c1_ku_02"},  # exact dup
            {"type": "mechanism",  "concept": "Reflexion", "ku_id": "c1_ku_03"},  # different type → keep
        ]
        deduped = dedup_kus(kus)
        assert len(deduped) == 2, f"Expected 2, got {len(deduped)}"

    results.append(_test("dedup: exact duplicate removed", test_dedup_exact))

    def test_dedup_case_insensitive():
        kus = [
            {"type": "definition", "concept": "reflexion",  "ku_id": "ku_01"},
            {"type": "definition", "concept": "Reflexion",  "ku_id": "ku_02"},  # same after normalize
            {"type": "definition", "concept": "REFLEXION",  "ku_id": "ku_03"},  # same after normalize
        ]
        assert len(dedup_kus(kus)) == 1

    results.append(_test("dedup: case-insensitive normalization", test_dedup_case_insensitive))

    def test_dedup_hyphen():
        kus = [
            {"type": "mechanism", "concept": "self-attention",  "ku_id": "ku_01"},
            {"type": "mechanism", "concept": "self attention",  "ku_id": "ku_02"},  # hyphen → space
        ]
        assert len(dedup_kus(kus)) == 1

    results.append(_test("dedup: hyphen normalized to space", test_dedup_hyphen))

    def test_dedup_preserves_order():
        kus = [
            {"type": "definition", "concept": "A", "ku_id": "ku_01"},
            {"type": "definition", "concept": "B", "ku_id": "ku_02"},
            {"type": "definition", "concept": "A", "ku_id": "ku_03"},  # dup of ku_01
        ]
        deduped = dedup_kus(kus)
        assert deduped[0]["ku_id"] == "ku_01"  # first occurrence kept
        assert len(deduped) == 2

    results.append(_test("dedup: first occurrence kept", test_dedup_preserves_order))

    # ── build_ku_graph ─────────────────────────────────────────────────────
    def test_graph_edge_exists():
        graph = build_ku_graph(SAMPLE_KUS)
        # c1_ku_01 (Reflexion) references "ReAct" → c1_ku_02 (ReAct)
        assert "c1_ku_02" in graph["c1_ku_01"], "Edge Reflexion → ReAct missing"

    results.append(_test("graph: edge from related_concepts created", test_graph_edge_exists))

    def test_graph_bidirectional():
        graph = build_ku_graph(SAMPLE_KUS)
        # c1_ku_02 references Reflexion → c1_ku_01 should also have c1_ku_02
        assert "c1_ku_01" in graph["c1_ku_02"], "Bidirectional edge missing"

    results.append(_test("graph: edges are bidirectional", test_graph_bidirectional))

    def test_graph_no_self_loop():
        graph = build_ku_graph(SAMPLE_KUS)
        for ku_id, neighbors in graph.items():
            assert ku_id not in neighbors, f"Self-loop found at {ku_id}"

    results.append(_test("graph: no self-loops", test_graph_no_self_loop))

    def test_graph_isolated_node():
        graph = build_ku_graph(SAMPLE_KUS)
        assert graph["c2_ku_02"] == [], "Isolated node should have no edges"

    results.append(_test("graph: isolated node has empty edge list", test_graph_isolated_node))

    def test_graph_all_nodes_present():
        graph = build_ku_graph(SAMPLE_KUS)
        for ku in SAMPLE_KUS:
            assert ku["ku_id"] in graph, f"{ku['ku_id']} missing from graph"

    results.append(_test("graph: all KUs are nodes", test_graph_all_nodes_present))

    # ── build_distractor_pool ──────────────────────────────────────────────
    def test_pool_grouped_by_type():
        pool = build_distractor_pool(SAMPLE_KUS)
        assert "definition" in pool
        assert "mechanism"  in pool
        assert "procedure"  in pool
        # definition has 2 entries: Reflexion + Evaluator
        assert len(pool["definition"]) == 2

    results.append(_test("pool: grouped by type correctly", test_pool_grouped_by_type))

    def test_pool_entry_fields():
        pool = build_distractor_pool(SAMPLE_KUS)
        entry = pool["definition"][0]
        assert "concept" in entry
        assert "content" in entry

    results.append(_test("pool: entries have concept + content", test_pool_entry_fields))

    # ── compute_priority ───────────────────────────────────────────────────
    def test_priority_primary_higher_than_peripheral():
        graph = build_ku_graph(SAMPLE_KUS)
        p_primary    = compute_priority(SAMPLE_KUS[0], graph, SAMPLE_KUS)  # Reflexion, primary
        p_peripheral = compute_priority(SAMPLE_KUS[4], graph, SAMPLE_KUS)  # Isolated, peripheral
        assert p_primary > p_peripheral, (
            f"primary ({p_primary}) should score higher than peripheral ({p_peripheral})"
        )

    results.append(_test("priority: primary > peripheral", test_priority_primary_higher_than_peripheral))

    def test_priority_hub_higher_than_leaf():
        graph = build_ku_graph(SAMPLE_KUS)
        # Reflexion (many connections) vs Evaluator (fewer)
        p_hub  = compute_priority(SAMPLE_KUS[0], graph, SAMPLE_KUS)
        p_leaf = compute_priority(SAMPLE_KUS[3], graph, SAMPLE_KUS)
        assert p_hub >= p_leaf, f"Hub ({p_hub}) should score >= leaf ({p_leaf})"

    results.append(_test("priority: hub >= leaf node", test_priority_hub_higher_than_leaf))

    def test_priority_range():
        graph = build_ku_graph(SAMPLE_KUS)
        for ku in SAMPLE_KUS:
            p = compute_priority(ku, graph, SAMPLE_KUS)
            assert 0.0 <= p <= 3.0, f"Priority {p} out of expected range for {ku['ku_id']}"

    results.append(_test("priority: all scores in range [0, 3]", test_priority_range))

    # ── Summary ────────────────────────────────────────────────────────────
    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*40}")
    print(f"KU Graph: {passed}/{total} passed")
    print(f"{'='*40}")
    return passed == total


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
