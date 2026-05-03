"""
evaluation/benchmark.py
-----------------------
Thesis benchmark: Axis 1 (Pipeline Robustness) + Axis 2 (MCQ Quality).

Usage:
  # Full benchmark on multiple files (generate + judge):
  python evaluation/benchmark.py data/file1.pdf data/file2.pptx --judge

  # Generate only (Axis 1, no API cost for judge):
  python evaluation/benchmark.py data/*.pdf --density Vua --difficulty medium

  # Evaluate from pre-generated MCQ files:
  python evaluation/benchmark.py --mcq data/mcqs.json --chunks data/chunks.json --judge

Output:
  evaluation/results/benchmark_<timestamp>.json  — full data
  stdout                                          — summary tables
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BLOOM_ORDER = ["remember", "understand", "apply", "analyze", "evaluate", "create"]


# ── Axis 1: pipeline robustness ────────────────────────────────────

def run_axis1(file_path: str, bloom_levels: list, difficulty: str) -> dict:
    """
    Run full pipeline on one file and collect robustness metrics.
    Returns dict with all Axis 1 metrics + saves mcqs/chunks as side-effect.
    """
    from pipeline.file_router import parse_file, get_metadata
    from pipeline.document_analyzer import analyze_document
    from pipeline.chunker import chunk_pages
    from pipeline.generator import generate_mcqs, get_provider_stats, reset_provider_stats
    from pipeline.validator import validate_mcqs, validation_stats

    stem = Path(file_path).stem
    print(f"\n{'─'*60}")
    print(f"  FILE: {Path(file_path).name}")
    print(f"{'─'*60}")

    # ── Parse ──────────────────────────────────────────────────────
    meta  = get_metadata(file_path)
    t0    = time.time()
    pages = parse_file(file_path)
    parse_time = round(time.time() - t0, 1)

    analysis = analyze_document(pages)
    total_pages_in_file = meta.get("total_pages", len(pages))
    parse_success_rate  = round(len(pages) / total_pages_in_file, 3) if total_pages_in_file else 1.0

    print(f"  Parse: {len(pages)}/{total_pages_in_file} pages "
          f"({parse_success_rate*100:.1f}%)  [{parse_time}s]")
    print(f"  Strategy: {analysis['chunk_strategy']}  "
          f"scan={analysis['scan_suspected']}  "
          f"avg_chars={analysis['avg_chars']}")

    # ── Chunk ──────────────────────────────────────────────────────
    strategy = analysis["chunk_strategy"]
    chunks   = chunk_pages(pages, strategy=strategy)

    type_counts = Counter(c.get("chunk_type", "?") for c in chunks)
    conceptual  = type_counts.get("conceptual", 0)
    conceptual_ratio = round(conceptual / len(chunks), 3) if chunks else 0.0

    print(f"  Chunks: {len(chunks)} total  {dict(type_counts)}  "
          f"conceptual={conceptual_ratio*100:.0f}%")

    # Save chunks for judge step
    out_dir = Path("data"); out_dir.mkdir(exist_ok=True)
    chunks_path = out_dir / f"{stem}__chunks.json"
    chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── Generate ───────────────────────────────────────────────────
    reset_provider_stats()
    pdf_name   = Path(file_path).name
    t0         = time.time()
    raw_mcqs   = generate_mcqs(
        chunks,
        bloom_levels=bloom_levels,
        difficulty=difficulty,
        pdf_name=file_path,
    )
    gen_time   = round(time.time() - t0, 1)
    prov_stats = get_provider_stats()

    # Latency from chunk_logs (successful calls only)
    success_logs = [
        lg for lg in prov_stats["chunk_logs"]
        if lg["status"] == "success" and lg["latency_seconds"] > 0
    ]
    latency_avg  = round(
        sum(lg["latency_seconds"] for lg in success_logs) / len(success_logs), 2
    ) if success_logs else 0.0
    provider_calls = Counter(
        f"{lg['provider']}/{lg['model'].split('/')[-1]}"
        for lg in success_logs
    )
    skipped_calls = len(prov_stats["chunks_skipped"])

    print(f"  Generate: {len(raw_mcqs)} raw MCQs  "
          f"latency_avg={latency_avg}s  total={gen_time}s  "
          f"skipped={skipped_calls}")

    # ── Validate ───────────────────────────────────────────────────
    valid, rejected = validate_mcqs(raw_mcqs)
    vstats = validation_stats(valid, rejected)

    print(f"  Validate: {vstats['valid']}/{vstats['total_raw']} valid  "
          f"schema_fail={vstats['reject_layers'].get('schema',0)}  "
          f"quality_fail={vstats['reject_layers'].get('quality',0)}")

    # Bloom dist in valid MCQs
    bloom_dist = {lvl: vstats["bloom_dist"].get(lvl, 0) for lvl in BLOOM_ORDER}

    # Save MCQs for judge step
    mcqs_path = out_dir / f"{stem}__mcqs.json"
    mcqs_path.write_text(
        json.dumps(valid, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "file"               : Path(file_path).name,
        "total_pages_in_file": total_pages_in_file,
        "pages_extracted"    : len(pages),
        "parse_success_rate" : parse_success_rate,
        "avg_chars_per_page" : analysis["avg_chars"],
        "scan_suspected"     : analysis["scan_suspected"],
        "chunk_strategy"     : strategy,
        "total_chunks"       : len(chunks),
        "chunk_type_dist"    : dict(type_counts),
        "conceptual_ratio"   : conceptual_ratio,
        "bloom_levels_used"  : bloom_levels,
        "total_raw_mcqs"     : len(raw_mcqs),
        "valid_mcqs"         : vstats["valid"],
        "schema_rejected"    : vstats["reject_layers"].get("schema", 0),
        "quality_rejected"   : vstats["reject_layers"].get("quality", 0),
        "schema_pass_rate"   : round(
            1 - vstats["reject_layers"].get("schema", 0) / max(vstats["total_raw"], 1), 3
        ),
        "content_pass_rate"  : round(vstats["valid"] / max(vstats["total_raw"], 1), 3),
        "bloom_dist"         : bloom_dist,
        "skipped_bloom_calls": skipped_calls,
        "latency_total_s"    : gen_time,
        "latency_per_call_avg_s": latency_avg,
        "provider_calls"     : dict(provider_calls),
        "reject_reasons"     : vstats["reject_reasons"],
        # saved file paths for judge / parse quality (reuse parsed pages)
        "_mcqs_path"         : str(mcqs_path),
        "_chunks_path"       : str(chunks_path),
        "_pages"             : pages,
    }


# ── Axis 2: MCQ quality (LLM judge) ───────────────────────────────

def run_axis2(mcqs_path: str, chunks_path: str) -> dict:
    """Run LLM judge on saved MCQ + chunks files. Returns Axis 2 metrics."""
    from evaluation.llm_judge import evaluate_all, compute_stats

    with open(mcqs_path, encoding="utf-8") as f:
        mcqs = json.load(f)
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"\n  [judge] Evaluating {len(mcqs)} MCQs...")
    results = evaluate_all(mcqs, chunks)
    stats   = compute_stats(results)

    # Bloom alignment per level
    by_level: dict[str, list[float]] = {}
    for r in results:
        lvl = r.get("bloom_level", "unknown")
        by_level.setdefault(lvl, []).append(r["scores"].get("bloom_alignment", 0))
    bloom_align_by_level = {
        lvl: round(sum(v) / len(v), 2) for lvl, v in by_level.items()
    }

    return {
        "judge_model"         : _get_judge_model(),
        "total_evaluated"     : stats.get("total_mcqs", 0),
        "judge_pass_rate"     : stats.get("pass_rate", 0),
        "avg_relevance"       : stats.get("avg_relevance", 0),
        "avg_answerability"   : stats.get("avg_answerability", 0),
        "avg_bloom_alignment" : stats.get("avg_bloom_alignment", 0),
        "avg_answer_correctness": stats.get("avg_answer_correctness", 0),
        "avg_semantic"        : stats.get("avg_semantic", 0),
        "bloom_alignment_by_level": bloom_align_by_level,
        "by_chunk"            : stats.get("by_chunk", {}),
    }


def _get_judge_model() -> str:
    from dotenv import load_dotenv
    load_dotenv()
    from evaluation.llm_judge import JUDGE_PROVIDERS, ENV_KEYS
    for p in JUDGE_PROVIDERS:
        if os.getenv(ENV_KEYS[p["provider"]], "").strip():
            return f"{p['provider']}/{p['model']}"
    return "unknown"


# ── Print tables ────────────────────────────────────────────────────

def print_axis1_table(results: list[dict]):
    print(f"\n{'═'*100}")
    print("  AXIS 1 — PIPELINE ROBUSTNESS")
    print(f"{'═'*100}")

    # Header
    h = (f"{'File':<38} {'Pages':>6} {'Parse%':>7} {'Strategy':<12} "
         f"{'Chunks':>6} {'Cncpt%':>7} {'MCQs':>5} {'Schema%':>8} "
         f"{'Valid%':>7} {'Lat/call':>9}")
    print(f"  {h}")
    print(f"  {'─'*96}")

    for r in results:
        fname   = r["file"][:37]
        pages   = f"{r['pages_extracted']}/{r['total_pages_in_file']}"
        parsep  = f"{r['parse_success_rate']*100:.0f}%"
        strat   = r["chunk_strategy"][:11]
        chunks  = r["total_chunks"]
        concptp = f"{r['conceptual_ratio']*100:.0f}%"
        mcqs    = r["valid_mcqs"]
        schemap = f"{r['schema_pass_rate']*100:.0f}%"
        validp  = f"{r['content_pass_rate']*100:.0f}%"
        lat     = f"{r['latency_per_call_avg_s']:.1f}s"
        print(f"  {fname:<38} {pages:>6} {parsep:>7} {strat:<12} "
              f"{chunks:>6} {concptp:>7} {mcqs:>5} {schemap:>8} "
              f"{validp:>7} {lat:>9}")

    print(f"\n  BLOOM DISTRIBUTION (valid MCQs per level):")
    bloom_h = f"  {'File':<38} " + "".join(f"{l[:3]:>5}" for l in BLOOM_ORDER) + f"  {'skip':>5}"
    print(bloom_h)
    print(f"  {'─'*70}")
    for r in results:
        row = f"  {r['file'][:37]:<38} "
        row += "".join(f"{r['bloom_dist'].get(l,0):>5}" for l in BLOOM_ORDER)
        row += f"  {r['skipped_bloom_calls']:>5}"
        print(row)


def print_axis2_table(results: list[dict]):
    has_axis2 = any("axis2" in r for r in results)
    if not has_axis2:
        print("\n  (Axis 2 not run — use --judge to include)")
        return

    print(f"\n{'═'*90}")
    print("  AXIS 2 — MCQ QUALITY (LLM Judge)")
    print(f"{'═'*90}")

    h = (f"{'File':<38} {'n':>4} {'Pass%':>6} "
         f"{'Relev':>6} {'Answr':>6} {'Bloom':>6} {'Corr':>6} {'Avg':>6}")
    print(f"  {h}")
    print(f"  {'─'*86}")

    for r in results:
        if "axis2" not in r:
            print(f"  {r['file'][:37]:<38}  (skipped)")
            continue
        a2 = r["axis2"]
        print(
            f"  {r['file'][:37]:<38} {a2['total_evaluated']:>4} "
            f"{a2['judge_pass_rate']:>5.1f}% "
            f"{a2['avg_relevance']:>6.2f} "
            f"{a2['avg_answerability']:>6.2f} "
            f"{a2['avg_bloom_alignment']:>6.2f} "
            f"{a2['avg_answer_correctness']:>6.2f} "
            f"{a2['avg_semantic']:>6.2f}"
        )

    # Bloom alignment per level breakdown
    print(f"\n  BLOOM ALIGNMENT by level (judge score /5):")
    bloom_h = f"  {'File':<38} " + "".join(f"{l[:3]:>5}" for l in BLOOM_ORDER)
    print(bloom_h)
    print(f"  {'─'*70}")
    for r in results:
        if "axis2" not in r:
            continue
        ba = r["axis2"].get("bloom_alignment_by_level", {})
        row = f"  {r['file'][:37]:<38} "
        row += "".join(
            f"{ba[l]:>5.2f}" if l in ba else f"{'—':>5}"
            for l in BLOOM_ORDER
        )
        print(row)


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Thesis benchmark: Axis 1 (Pipeline) + Axis 2 (MCQ Quality)"
    )
    parser.add_argument("files",      nargs="*", help="Input files (PDF/PPTX/DOCX/...)")
    parser.add_argument("--mcq",      help="Pre-generated MCQ JSON (skip generation)")
    parser.add_argument("--chunks",   help="Chunks JSON (dùng kèm --mcq)")
    parser.add_argument("--difficulty", default="medium",
                        help="easy | medium | hard (default: medium)")
    parser.add_argument("--parse",    action="store_true",
                        help="Run parse quality analysis for Axis 0")
    parser.add_argument("--judge",    action="store_true",
                        help="Run LLM judge for Axis 2")
    parser.add_argument("--out",      default=None,
                        help="Output JSON path (default: auto)")
    args = parser.parse_args()

    from pipeline.generator import DIFFICULTY_TO_BLOOM

    bloom_levels = DIFFICULTY_TO_BLOOM.get(args.difficulty, DIFFICULTY_TO_BLOOM["medium"])

    print(f"\n{'═'*60}")
    print(f"  BENCHMARK — {datetime.now():%Y-%m-%d %H:%M}")
    print(f"  Difficulty: {args.difficulty} → {bloom_levels}")
    print(f"  Judge   : {'yes (Axis 2)' if args.judge else 'no (Axis 1 only)'}")
    print(f"{'═'*60}")

    all_results = []

    # Mode A: pre-generated MCQ files
    if args.mcq and args.chunks:
        print(f"\n[mode] Load from {args.mcq}")
        ax2 = run_axis2(args.mcq, args.chunks) if args.judge else None
        entry: dict = {
            "file"  : Path(args.mcq).name,
            "axis1" : None,
        }
        if ax2:
            entry["axis2"] = ax2
        all_results.append(entry)

    # Mode B: run pipeline on files
    elif args.files:
        for fp in args.files:
            if not Path(fp).exists():
                print(f"  [SKIP] {fp} — file not found")
                continue
            try:
                ax1 = run_axis1(fp, bloom_levels, args.difficulty)
                entry = {"file": ax1["file"], "axis1": ax1}
                if args.parse:
                    try:
                        from evaluation.parse_quality import run_parse_quality
                        ax0 = run_parse_quality(fp, pages=ax1.get("_pages"))
                        entry["axis0"] = ax0
                    except Exception as e:
                        print(f"  [parse quality ERROR] {e}")
                if args.judge:
                    try:
                        ax2 = run_axis2(ax1["_mcqs_path"], ax1["_chunks_path"])
                        entry["axis2"] = ax2
                    except Exception as e:
                        print(f"  [judge ERROR] {e}")
                all_results.append(entry)
            except Exception as e:
                print(f"  [ERROR] {fp}: {e}")
                import traceback; traceback.print_exc()

    else:
        parser.print_help()
        sys.exit(1)

    # ── Print tables ────────────────────────────────────────────────
    ax0_entries = [r["axis0"] for r in all_results if r.get("axis0")]
    if ax0_entries:
        from evaluation.parse_quality import print_parse_quality_table
        print_parse_quality_table(ax0_entries)

    ax1_entries = [r["axis1"] for r in all_results if r.get("axis1")]
    if ax1_entries:
        print_axis1_table(ax1_entries)
    print_axis2_table(all_results)

    # ── Save full results ────────────────────────────────────────────
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"evaluation/results/benchmark__{ts}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    output = {
        "meta": {
            "timestamp"  : ts,
            "bloom_levels": bloom_levels,
            "difficulty" : args.difficulty,
            "judge"      : args.judge,
        },
        "results": all_results,
    }
    Path(out_path).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  Saved → {out_path}\n")


if __name__ == "__main__":
    main()
