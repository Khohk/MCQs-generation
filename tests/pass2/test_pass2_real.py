"""
tests/pass2/test_pass2_real.py
-------------------------------
Test full 2-pass pipeline voi API that.

Chay:
    python tests/pass2/test_pass2_real.py
    python tests/pass2/test_pass2_real.py data/W02-VersionControl.pptx
    python tests/pass2/test_pass2_real.py data/PCA.docx
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from pipeline.file_router import parse_file
from pipeline.pass1_extractor import run_pass1
from pipeline.pass2_extractor import run_pass2
from pipeline.knowledge_extractor import compute_priority
import pipeline.generator as gen
from pipeline.generator import PROVIDERS


# ── LLM wrapper ───────────────────────────────────────────────────────────

def _llm_fn(prompt: str, json_mode: bool) -> str:
    attempts = len(PROVIDERS) * gen.RETRY_LIMIT
    for _ in range(attempts):
        idx = gen._next_available_provider()
        if idx is None:
            time.sleep(5)
            continue
        gen._current_provider_idx = idx
        try:
            return gen._call_provider(prompt, idx, json_mode)
        except Exception as e:
            err = str(e)
            is_rate  = any(k in err for k in ("429", "RESOURCE_EXHAUSTED", "rate_limit"))
            is_404   = any(k in err for k in ("404", "NOT_FOUND", "model_not_found"))
            is_nokey = "not found in .env" in err
            if is_404 or is_nokey:
                gen._mark_disabled(idx, "404/nokey")
            elif is_rate:
                gen._mark_cooldown(idx)
            else:
                time.sleep(3)
            time.sleep(gen.SWITCH_DELAY)
    raise RuntimeError("All providers failed")


# ── Pretty print ──────────────────────────────────────────────────────────

def _print_ku(ku: dict, graph: dict, all_kus: list):
    priority  = compute_priority(ku, graph, all_kus)
    neighbors = graph.get(ku["ku_id"], [])
    print(f"\n  [{ku['ku_id']}]")
    print(f"    type      : {ku['type']}  |  prominence: {ku['prominence']}"
          f"  |  priority: {priority}")
    print(f"    concept   : {ku['concept']}")
    print(f"    content   : {ku['content'][:100]}{'...' if len(ku['content'])>100 else ''}")
    print(f"    evidence  : {ku['verbatim_evidence'][:100]}")
    print(f"    related   : {ku['related_concepts']}")
    print(f"    edges     : {neighbors if neighbors else '(none)'}")


def _print_pass1_summary(pass1):
    print(f"\n  main_concept : {pass1.main_concept}")
    print(f"  segments     : {len(pass1.segments)}")
    for seg in pass1.segments:
        print(f"    {seg.segment_id}  pages={seg.source_pages}  "
              f"chars={len(seg.text)}  {seg.label[:45]}")
    print(f"  cross-rels   : {len(pass1.relationships)}")
    for r in pass1.relationships:
        print(f"    [{r['relation']}] {r['from_concept']} -> {r['to_concept']}")


# ── Assertions ────────────────────────────────────────────────────────────

def _assert_pass2(result2) -> list[str]:
    errors = []
    if not result2.ok:
        return ["No KUs extracted"]

    for ku in result2.all_kus:
        if not ku.get("verbatim_evidence"):
            errors.append(f"{ku['ku_id']}: empty verbatim_evidence")
        if not ku.get("concept"):
            errors.append(f"{ku['ku_id']}: empty concept")
        if ku.get("completeness") != "complete":
            errors.append(f"{ku['ku_id']}: completeness != complete")

    # Graph: no self-loops
    for ku_id, neighbors in result2.graph.items():
        if ku_id in neighbors:
            errors.append(f"graph: self-loop at {ku_id}")

    return errors


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 \
                else "data/Chapter3.2-PerceptronLearningAlgorithm.pdf"

    fpath = ROOT / file_path
    if not fpath.exists():
        fpath = Path(file_path)
    if not fpath.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  File : {fpath.name}")
    print(f"{'='*60}")

    pages = parse_file(str(fpath))
    print(f"  Parsed: {len(pages)} pages, "
          f"{sum(p['char_count'] for p in pages):,} chars")

    # ── Pass 1 ────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  PASS 1 — segmentation + cross-relationships")
    print(f"{'─'*60}")
    t0    = time.time()
    pass1 = run_pass1(pages, _llm_fn)
    t1    = round(time.time() - t0, 1)
    print(f"  Pass 1 done in {t1}s")
    _print_pass1_summary(pass1)

    if not pass1.ok:
        print("\n[FAIL] Pass 1 returned no segments")
        sys.exit(1)

    # ── Pass 2 ────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  PASS 2 — KU extraction per segment")
    print(f"{'─'*60}")
    t2      = time.time()
    result2 = run_pass2(pass1, _llm_fn)
    t3      = round(time.time() - t2, 1)
    print(f"\n  Pass 2 done in {t3}s")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Total KUs     : {len(result2.all_kus)}")
    n_edges = sum(len(v) for v in result2.graph.values()) // 2
    print(f"  Graph edges   : {n_edges}")
    print(f"  Distractor pool: { {k: len(v) for k, v in result2.pool.items()} }")
    print(f"  Total time    : {round(t1+t3, 1)}s")
    print(f"{'='*60}")

    # ── KU details ────────────────────────────────────────────────────────
    print("\n  KU Details:")
    for ku in result2.all_kus:
        _print_ku(ku, result2.graph, result2.all_kus)

    # ── Assertions ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    errors = _assert_pass2(result2)
    if errors:
        print(f"[FAIL] {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"[PASS] all {len(result2.all_kus)} KUs pass structural assertions")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
