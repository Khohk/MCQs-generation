"""
tests/ku/test_ku_extract_real.py
---------------------------------
Test extract_kus() với API thật — cần GEMINI_API_KEY trong .env.
Chạy trên 1 chunk thực từ file slide, in KU kết quả để human review.

Chạy:
    python tests/ku/test_ku_extract_real.py
    python tests/ku/test_ku_extract_real.py data/NLP_Week_6_7.pptx
    python tests/ku/test_ku_extract_real.py data/NLP_Week_6_7.pptx 2
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
from pipeline.chunker import chunk_pages
from pipeline.knowledge_extractor import (
    extract_kus, build_ku_graph, build_distractor_pool, compute_priority
)
from pipeline.generator import _call_provider, _next_available_provider, PROVIDERS


# ── llm_fn wrapper using generator's provider fallback ────────────────────

def _llm_fn(prompt: str, json_mode: bool) -> str:
    """Single-call wrapper: tries providers in order, raises on all fail."""
    import pipeline.generator as gen

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
            is_rate = any(k in err for k in ("429", "RESOURCE_EXHAUSTED", "rate_limit"))
            is_404  = any(k in err for k in ("404", "NOT_FOUND", "model_not_found"))
            is_nokey = "not found in .env" in err
            if is_404 or is_nokey:
                gen._mark_disabled(idx, "404/nokey")
            elif is_rate:
                gen._mark_cooldown(idx)
            else:
                time.sleep(3)
            time.sleep(gen.SWITCH_DELAY)
    raise RuntimeError("All providers failed")


# ── Pretty print ───────────────────────────────────────────────────────────

def _print_ku(ku: dict, graph: dict, all_kus: list):
    priority = compute_priority(ku, graph, all_kus)
    neighbors = graph.get(ku["ku_id"], [])
    print(f"\n  ┌─ {ku['ku_id']}")
    print(f"  │  type:       {ku['type']}")
    print(f"  │  prominence: {ku['prominence']}  |  priority: {priority}")
    print(f"  │  concept:    {ku['concept']}")
    print(f"  │  content:    {ku['content'][:100]}{'...' if len(ku['content']) > 100 else ''}")
    print(f"  │  evidence:   {ku['verbatim_evidence'][:100]}{'...' if len(ku['verbatim_evidence']) > 100 else ''}")
    print(f"  │  related:    {ku['related_concepts']}")
    print(f"  │  edges:      {neighbors if neighbors else '(none)'}")
    print(f"  └─ completeness: {ku['completeness']}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    # Default test file
    file_path  = sys.argv[1] if len(sys.argv) > 1 else "data/NLP_Week_6_7.pptx"
    n_chunks   = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    fpath = ROOT / file_path
    if not fpath.exists():
        # Try relative
        fpath = Path(file_path)
    if not fpath.exists():
        print(f"File not found: {file_path}")
        print("Usage: python tests/ku/test_ku_extract_real.py <file> [n_chunks]")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  File   : {fpath.name}")
    print(f"  Chunks : {n_chunks} conceptual chunk(s)")
    print(f"{'='*55}")

    pages  = parse_file(str(fpath))
    chunks = chunk_pages(pages)
    conceptual = [c for c in chunks if c.get("chunk_type") == "conceptual"]

    if not conceptual:
        print("No conceptual chunks found.")
        sys.exit(1)

    test_chunks = conceptual[:n_chunks]
    all_kus = []

    for i, chunk in enumerate(test_chunks, 1):
        print(f"\n[{i}/{len(test_chunks)}] Extracting: {chunk['chunk_id']} — {chunk['topic'][:50]}")
        print(f"  Text preview: {chunk['text'][:120].replace(chr(10), ' ')}...")

        t0   = time.time()
        kus  = extract_kus(chunk, _llm_fn)
        elapsed = round(time.time() - t0, 1)

        print(f"  → {len(kus)} KUs in {elapsed}s")
        all_kus.extend(kus)

        if i < len(test_chunks):
            time.sleep(4.5)

    if not all_kus:
        print("\nNo KUs extracted. Check API key and chunk content.")
        sys.exit(1)

    # ── Build graph + pool ────────────────────────────────────────────────
    graph = build_ku_graph(all_kus)
    pool  = build_distractor_pool(all_kus)

    print(f"\n{'='*55}")
    print(f"  Total KUs extracted : {len(all_kus)}")
    print(f"  Distractor pool     : { {k: len(v) for k, v in pool.items()} }")
    print(f"{'='*55}")

    print("\n── KU Details ──────────────────────────────────────")
    for ku in all_kus:
        _print_ku(ku, graph, all_kus)

    # ── Assertions for CI ─────────────────────────────────────────────────
    print("\n── Assertions ──────────────────────────────────────")

    errors = []

    for ku in all_kus:
        if not ku.get("verbatim_evidence"):
            errors.append(f"{ku['ku_id']}: empty verbatim_evidence")
        if not ku.get("concept"):
            errors.append(f"{ku['ku_id']}: empty concept")
        if ku.get("completeness") != "complete":
            errors.append(f"{ku['ku_id']}: completeness != complete (should have been filtered)")
        if not ku["ku_id"].startswith(ku["ku_id"].split("_ku_")[0]):
            errors.append(f"{ku['ku_id']}: ku_id format wrong")

    if errors:
        print(f"[FAIL] {len(errors)} assertion(s) failed:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"[PASS] All {len(all_kus)} KUs pass structural assertions")
        print("\n  Manual review: check that verbatim_evidence appears in slide text above.")

    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    main()
