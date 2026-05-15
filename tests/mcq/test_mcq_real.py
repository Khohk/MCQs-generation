"""
tests/mcq/test_mcq_real.py
---------------------------
Integration test: full pipeline Pass1 → Pass2 → MCQ generation với real API.

Run:
    python tests/mcq/test_mcq_real.py
    python tests/mcq/test_mcq_real.py data/NLP_Week_6_7.pdf
    python tests/mcq/test_mcq_real.py data/NLP_Week_2.pdf
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
from pipeline.mcq_generator import (
    MCQItem,
    run_mcq_generation,
    run_cross_mcq_generation,
)
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

def _print_mcq(m: MCQItem, idx: int):
    kind = "CROSS" if m.is_cross else "SINGLE"
    print(f"\n  [{idx}] {kind}  priority={m.priority}  tier={m.distractor_tier}")
    if m.is_cross:
        print(f"    A: {m.anchor_concept}  ↔  B: ...")
        print(f"    edge: {m.edge_relation}")
    else:
        print(f"    concept: {m.anchor_concept}  type={m.anchor_type}  bloom={m.bloom_level}")
    print(f"    Q: {m.question[:100]}{'...' if len(m.question) > 100 else ''}")
    for k, v in m.options.items():
        marker = " ←" if k == m.answer else ""
        print(f"    {k}: {v[:80]}{marker}")
    if m.explanation:
        print(f"    Exp: {m.explanation[:80]}...")


# ── Assertions ────────────────────────────────────────────────────────────

def _assert_mcqs(mcqs: list[MCQItem], label: str) -> list[str]:
    errors = []
    if not mcqs:
        return [f"{label}: no MCQs generated"]

    for m in mcqs:
        prefix = f"{label} {m.mcq_id}"

        if not m.question.strip():
            errors.append(f"{prefix}: empty question")
        if set(m.options.keys()) != {"A", "B", "C", "D"}:
            errors.append(f"{prefix}: options keys != A/B/C/D — {set(m.options.keys())}")
        if m.answer not in {"A", "B", "C", "D"}:
            errors.append(f"{prefix}: answer not in A-D — {m.answer!r}")
        if m.bloom_level not in {"remember", "understand", "apply", "analyze", "evaluate", "create"}:
            errors.append(f"{prefix}: unknown bloom_level {m.bloom_level!r}")
        if not isinstance(m.priority, int) or m.priority < 1:
            errors.append(f"{prefix}: bad priority {m.priority}")

        # Option uniqueness
        vals = [v.strip().lower() for v in m.options.values()]
        if len(set(vals)) < 4:
            errors.append(f"{prefix}: duplicate options")

        # Cross-concept extras
        if m.is_cross:
            if not m.anchor_ku_id_b:
                errors.append(f"{prefix}: cross MCQ missing anchor_ku_id_b")
            if not m.edge_relation:
                errors.append(f"{prefix}: cross MCQ missing edge_relation")
            if m.priority not in {5, 6}:
                errors.append(f"{prefix}: cross MCQ priority should be 5 or 6, got {m.priority}")
        else:
            if m.priority not in {1, 2, 3, 4}:
                errors.append(f"{prefix}: single MCQ priority should be 1-4, got {m.priority}")

    # Sorted by priority
    priorities = [m.priority for m in mcqs]
    if priorities != sorted(priorities):
        errors.append(f"{label}: MCQs not sorted by priority")

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
    print(f"\n{'─'*60}\n  PASS 1\n{'─'*60}")
    t0    = time.time()
    pass1 = run_pass1(pages, _llm_fn)
    print(f"  done in {round(time.time()-t0,1)}s  "
          f"— {len(pass1.segments)} segs  "
          f"{len(pass1.relationships)} cross-rels")

    if not pass1.ok:
        print("[FAIL] Pass 1 returned no segments")
        sys.exit(1)

    # ── Pass 2 ────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}\n  PASS 2\n{'─'*60}")
    t1      = time.time()
    pass2   = run_pass2(pass1, _llm_fn)
    t2      = round(time.time()-t1, 1)
    n_edges = sum(len(v) for v in pass2.graph.values()) // 2
    print(f"  done in {t2}s  "
          f"— {len(pass2.all_kus)} KUs  "
          f"{n_edges} graph edges")

    if not pass2.ok:
        print("[FAIL] Pass 2 returned no KUs")
        sys.exit(1)

    # ── MCQ generation (single-KU) ────────────────────────────────────────
    print(f"\n{'─'*60}\n  MCQ GENERATION — single-KU\n{'─'*60}")
    t3   = time.time()
    mcqs = run_mcq_generation(pass2, _llm_fn, delay_between=2.0)
    t4   = round(time.time()-t3, 1)

    n_a = sum(1 for m in mcqs if m.distractor_tier == "A")
    n_b = sum(1 for m in mcqs if m.distractor_tier == "B")
    print(f"\n  done in {t4}s  — {len(mcqs)} MCQs  (Tier A={n_a}  Tier B={n_b})")

    for i, m in enumerate(mcqs, 1):
        _print_mcq(m, i)

    errors_single = _assert_mcqs(mcqs, "single")

    # ── MCQ generation (cross-concept) ────────────────────────────────────
    print(f"\n{'─'*60}\n  MCQ GENERATION — cross-concept\n{'─'*60}")
    t5        = time.time()
    cross_mcqs = run_cross_mcq_generation(pass2, _llm_fn, delay_between=2.0)
    t6        = round(time.time()-t5, 1)

    print(f"\n  done in {t6}s  — {len(cross_mcqs)} cross MCQs")

    for i, m in enumerate(cross_mcqs, 1):
        _print_mcq(m, i)

    errors_cross = _assert_mcqs(cross_mcqs, "cross") if cross_mcqs else []

    # ── Summary ───────────────────────────────────────────────────────────
    total_time = round(time.time()-t0, 1)
    print(f"\n{'='*60}")
    print(f"  Pass1+2+MCQ time : {total_time}s")
    print(f"  KUs              : {len(pass2.all_kus)}")
    print(f"  Single MCQs      : {len(mcqs)}  (A={n_a} B={n_b})")
    print(f"  Cross MCQs       : {len(cross_mcqs)}")
    print(f"{'='*60}")

    all_errors = errors_single + errors_cross
    if all_errors:
        print(f"\n[FAIL] {len(all_errors)} assertion error(s):")
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        total_mcqs = len(mcqs) + len(cross_mcqs)
        print(f"\n[PASS] all {total_mcqs} MCQs pass structural assertions")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
