"""
tests/pass1/test_pass1_real.py
-------------------------------
Test run_pass1() voi API that -- can GEMINI_API_KEY trong .env.
In Pass1Result de human review.

Chay:
    python tests/pass1/test_pass1_real.py
    python tests/pass1/test_pass1_real.py data/Chapter3.2-PerceptronLearningAlgorithm.pdf
    python tests/pass1/test_pass1_real.py data/NLP_Week_6_7.pdf
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
import pipeline.generator as gen
from pipeline.generator import PROVIDERS


# ── LLM wrapper (same as test_ku_extract_real) ────────────────────────────

def _llm_fn(prompt: str, json_mode: bool) -> str:
    attempts = len(PROVIDERS) * gen.RETRY_LIMIT
    for _ in range(attempts):
        idx = gen._next_available_provider()
        if idx is None:
            time.sleep(5)
            continue
        gen._current_provider_idx = idx
        try:
            return gen._call_provider(prompt, idx, json_mode, max_tokens=8192)
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

def _print_result(result, pages):
    print(f"\n  main_concept : {result.main_concept}")

    print(f"\n  concept_hierarchy (raw from LLM):")
    for l1 in result.raw_hierarchy:
        print(f"    [L1] {l1.get('name','?')}")
        for child in l1.get("children", []):
            print(f"         └─ [L2] {child.get('name','?')}")

    print(f"\n  sub_concepts ({len(result.sub_concepts)}):")
    for sc in result.sub_concepts:
        ev = sc.get("evidence", "")[:80]
        print(f"    - {sc['name']:<35} pages={sc.get('pages',[])}  ev={ev!r}")

    print(f"\n  relationships ({len(result.relationships)}):")
    for r in result.relationships:
        print(f"    [{r.get('relation','?')}] {r['from_concept']} -> {r['to_concept']}")
        print(f"       ev_type={r.get('evidence_type','?')}  "
              f"ev={r.get('evidence','')[:70]!r}")

    print(f"\n  segments ({len(result.segments)}):")
    for seg in result.segments:
        print(f"\n    [{seg.segment_id}] {seg.label}")
        print(f"       pages   : {seg.source_pages}")
        print(f"       concepts: {seg.concepts}")
        print(f"       text len: {len(seg.text)} chars")
        print(f"       preview : {seg.text[:100].replace(chr(10),' ')!r}")

    # Coverage check: pages covered vs total
    covered = set()
    for seg in result.segments:
        covered.update(seg.source_pages)
    total_pages = {p["page_num"] for p in pages}
    missing = total_pages - covered
    extra   = covered - total_pages

    print(f"\n  page coverage : {len(covered)}/{len(total_pages)} pages covered")
    if missing:
        print(f"  [WARN] missing pages : {sorted(missing)}")
    if extra:
        print(f"  [WARN] extra pages   : {sorted(extra)}")


# ── Assertions ────────────────────────────────────────────────────────────

def _assert_result(result, pages):
    errors = []

    if not result.ok:
        errors.append("No segments returned")
        return errors

    if result.main_concept in ("Unknown", "", None):
        errors.append("main_concept is empty/Unknown")

    for seg in result.segments:
        if not seg.text.strip():
            errors.append(f"{seg.segment_id}: empty text")
        if not seg.source_pages:
            errors.append(f"{seg.segment_id}: empty source_pages")
        # Pages must be sorted and contiguous (first_page..last_page range)
        sp = seg.source_pages
        for i in range(1, len(sp)):
            if sp[i] != sp[i-1] + 1:
                errors.append(f"{seg.segment_id}: non-contiguous pages {sp}")
                break

    valid_relations = {
        "CONTRASTS_WITH", "ALTERNATIVE_TO", "SIMILAR_TO",
        "ENABLES", "EXTENDS", "PART_OF", "APPLIES_TO",
    }
    for r in result.relationships:
        if not r.get("from_concept") or not r.get("to_concept"):
            errors.append(f"relationship missing from/to: {r}")
        if r.get("relation") not in valid_relations:
            errors.append(f"unknown relation type: {r.get('relation')}")

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
    print(f"  Parsed : {len(pages)} pages")
    print(f"  Total chars : {sum(p['char_count'] for p in pages):,}")

    t0 = time.time()
    result = run_pass1(pages, _llm_fn)
    elapsed = round(time.time() - t0, 1)

    print(f"\n  Pass 1 done in {elapsed}s")
    _print_result(result, pages)

    print(f"\n{'='*60}")
    print("  Assertions")
    print(f"{'='*60}")
    errors = _assert_result(result, pages)
    if errors:
        print(f"[FAIL] {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"[PASS] all assertions passed")
        print(f"\n  Manual review: check segments cover logical topics,")
        print(f"  relationships span different segments,")
        print(f"  evidence is verbatim from source text.")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
