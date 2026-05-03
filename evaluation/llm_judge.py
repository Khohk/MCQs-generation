"""
evaluation/llm_judge.py
-----------------------
Đánh giá MCQ bằng LLM-as-a-Judge (OpenAI-compatible API).

Tiêu chí đánh giá (Likert 1–5):
  1. Relevance         — câu hỏi bám sát source không
  2. Answerability     — trả lời được chỉ từ source không
  3. Bloom Alignment   — bloom_level có khớp với câu hỏi thực tế không
  4. Answer Correctness— đáp án gán có đúng theo source không

Input:
  - List[MCQ dict] + List[chunk dict] (output từ chunker)
  - Judge tự map source_chunk → chunk text, không đọc lại file gốc

Usage:
  python evaluation/llm_judge.py data/W02-VersionControl.pptx
  python evaluation/llm_judge.py --mcq results.json --chunks chunks.json

.env:
  OPENAI_API_KEY=...       # GPT-4o-mini (default judge)
  GROQ_API_KEY=...         # fallback nếu không có OpenAI key
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Judge config ───────────────────────────────────────────────────

JUDGE_PROVIDERS = [
    {"provider": "openai",    "model": "gpt-4o-mini",             "base_url": None},
    {"provider": "groq",      "model": "llama-3.3-70b-versatile", "base_url": "https://api.groq.com/openai/v1"},
    {"provider": "cerebras",  "model": "llama3.3-70b",            "base_url": "https://api.cerebras.ai/v1"},
]

ENV_KEYS = {
    "openai"   : "OPENAI_API_KEY",
    "groq"     : "GROQ_API_KEY",
    "cerebras" : "CEREBRAS_API_KEY",
}

PASS_THRESHOLD   = 3.5   # semantic_avg >= threshold → passed
REQUEST_DELAY    = 2.0   # giây giữa các request

# ── System prompt với rubric ───────────────────────────────────────

SYSTEM_PROMPT = """You are an expert educational assessment evaluator.
You will evaluate a multiple-choice question (MCQ) against its source text.

## SCORING RUBRIC

### 1. Relevance (1–5): Does the question focus on content from the source?
- 5: Directly asks about the main concept/information in the source
- 4: Closely related, asks about an important detail
- 3: Related but asks about a minor detail or small example
- 2: Weakly related; most content not in source
- 1: Irrelevant or hallucinated — asks about something not in source

### 2. Answerability (1–5): Can the question be answered using ONLY the source?
- 5: Source provides a clear, unambiguous answer
- 4: Answerable with minor inference from source
- 3: Requires significant inference or light background knowledge
- 2: Source hints at the answer but is insufficient
- 1: Source does not contain enough information to answer

### 3. Bloom Alignment (1–5): Does the assigned bloom_level match the actual cognitive demand?
Bloom levels (low→high): remember → understand → apply → analyze → evaluate → create
- 5: Assigned level matches the actual cognitive demand perfectly
- 4: Slight mismatch (e.g., assigned "understand" but actually "remember")
- 3: Off by 1 clear level
- 2: Off by 2 levels
- 1: Completely wrong (e.g., assigned "evaluate" but actually just "remember")

### 4. Answer Correctness (1–5): Is the assigned answer correct according to the source?
- 5: Assigned answer is completely correct per source
- 4: Correct but explanation is incomplete
- 3: Correct but arguable
- 2: Answer may be wrong or misleading
- 1: Answer is incorrect per source

## OUTPUT FORMAT
Return ONLY valid JSON, no markdown, no extra text:
{
  "scores": {
    "relevance": <int 1-5>,
    "answerability": <int 1-5>,
    "bloom_alignment": <int 1-5>,
    "answer_correctness": <int 1-5>
  },
  "reasoning": {
    "relevance": "<1 sentence>",
    "answerability": "<1 sentence>",
    "bloom_alignment": "<1 sentence>",
    "answer_correctness": "<1 sentence>"
  }
}"""


# ── Client factory ─────────────────────────────────────────────────

def _get_judge_client() -> tuple[OpenAI, str]:
    """Trả về (client, model) cho provider đầu tiên có API key."""
    for p in JUDGE_PROVIDERS:
        key = os.getenv(ENV_KEYS[p["provider"]], "").strip()
        if key:
            kwargs = {"api_key": key}
            if p["base_url"]:
                kwargs["base_url"] = p["base_url"]
            _log(f"  [judge] Using {p['provider']}/{p['model']}")
            return OpenAI(**kwargs), p["model"]
    raise ValueError(
        "Không tìm thấy API key nào. Cần ít nhất 1 trong: "
        + ", ".join(ENV_KEYS.values())
    )


# ── Per-MCQ evaluation ─────────────────────────────────────────────

def evaluate_mcq(mcq: dict, source_text: str, client: OpenAI, model: str) -> dict:
    """
    Đánh giá 1 MCQ bằng LLM judge.

    Returns:
        {chunk_id, question, scores, reasoning, semantic_avg, passed}
    """
    user_msg = f"""## SOURCE TEXT
{source_text}

## MCQ TO EVALUATE
Question    : {mcq.get('question', '')}
A           : {mcq.get('A', '')}
B           : {mcq.get('B', '')}
C           : {mcq.get('C', '')}
D           : {mcq.get('D', '')}
Assigned answer     : {mcq.get('answer', '')}
Assigned bloom_level: {mcq.get('bloom_level', '')}
Explanation : {mcq.get('explanation', '')}

Evaluate this MCQ against the source text using the rubric."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=512,
    )

    raw  = response.choices[0].message.content.strip()
    data = json.loads(raw)

    scores = data.get("scores", {})
    semantic_scores = [
        scores.get("relevance", 0),
        scores.get("answerability", 0),
        scores.get("bloom_alignment", 0),
    ]
    semantic_avg = round(sum(semantic_scores) / len(semantic_scores), 2)

    return {
        "chunk_id"   : mcq.get("source_chunk", ""),
        "bloom_level": mcq.get("bloom_level", ""),
        "difficulty" : mcq.get("difficulty", ""),
        "question"   : mcq.get("question", "")[:80],
        "scores"     : scores,
        "reasoning"  : data.get("reasoning", {}),
        "semantic_avg": semantic_avg,
        "passed"     : semantic_avg >= PASS_THRESHOLD,
    }


# ── Batch evaluation ───────────────────────────────────────────────

def evaluate_all(
    mcqs: list[dict],
    chunks: list[dict],
    on_progress=None,
) -> list[dict]:
    """
    Đánh giá toàn bộ MCQ list.

    Args:
        mcqs  : output từ generator/validator
        chunks: output từ chunker (dùng để map source_chunk → text)

    Returns:
        List[result dict] — 1 result per MCQ
    """
    # Map chunk_id → text
    chunk_map = {c["chunk_id"]: c["text"] for c in chunks}

    client, model = _get_judge_client()
    results = []
    total   = len(mcqs)

    for idx, mcq in enumerate(mcqs):
        chunk_id    = mcq.get("source_chunk", "")
        source_text = chunk_map.get(chunk_id, "")

        if not source_text:
            _log(f"  [{idx+1}/{total}] SKIP — chunk '{chunk_id}' không tìm thấy")
            continue

        _log(f"  [{idx+1}/{total}] {chunk_id} | Q: {mcq.get('question','')[:50]}...")

        try:
            result = evaluate_mcq(mcq, source_text, client, model)
            results.append(result)
            s = result["scores"]
            _log(
                f"    R={s.get('relevance')} A={s.get('answerability')} "
                f"B={s.get('bloom_alignment')} C={s.get('answer_correctness')} "
                f"| avg={result['semantic_avg']} "
                f"| {'PASS' if result['passed'] else 'FAIL'}"
            )
        except Exception as e:
            _log(f"    ERROR: {str(e)[:100]}")

        if on_progress:
            on_progress(idx, total, chunk_id)

        if idx < total - 1:
            time.sleep(REQUEST_DELAY)

    return results


# ── Aggregate stats ────────────────────────────────────────────────

def compute_stats(results: list[dict]) -> dict:
    """Tính thống kê tổng hợp từ list results."""
    if not results:
        return {}

    n = len(results)

    def avg_score(key):
        vals = [r["scores"].get(key, 0) for r in results]
        return round(sum(vals) / len(vals), 2)

    passed = sum(1 for r in results if r["passed"])

    # Group by chunk
    by_chunk: dict[str, list] = {}
    for r in results:
        by_chunk.setdefault(r["chunk_id"], []).append(r["semantic_avg"])
    chunk_avgs = {cid: round(sum(v)/len(v), 2) for cid, v in by_chunk.items()}

    return {
        "total_mcqs"        : n,
        "passed"            : passed,
        "pass_rate"         : round(passed / n * 100, 1),
        "avg_relevance"     : avg_score("relevance"),
        "avg_answerability" : avg_score("answerability"),
        "avg_bloom_alignment"   : avg_score("bloom_alignment"),
        "avg_answer_correctness": avg_score("answer_correctness"),
        "avg_semantic"      : round(sum(r["semantic_avg"] for r in results) / n, 2),
        "by_chunk"          : chunk_avgs,
    }


# ── Utility ────────────────────────────────────────────────────────

def _log(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


def _print_stats(stats: dict):
    print(f"\n{'='*60}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total MCQs     : {stats['total_mcqs']}")
    print(f"  Passed         : {stats['passed']} / {stats['total_mcqs']} ({stats['pass_rate']}%)")
    print(f"{'─'*60}")
    print(f"  Avg Relevance         : {stats['avg_relevance']:.2f} / 5")
    print(f"  Avg Answerability     : {stats['avg_answerability']:.2f} / 5")
    print(f"  Avg Bloom Alignment   : {stats['avg_bloom_alignment']:.2f} / 5")
    print(f"  Avg Answer Correctness: {stats['avg_answer_correctness']:.2f} / 5")
    print(f"  Avg Semantic (R+A+B)  : {stats['avg_semantic']:.2f} / 5")
    print(f"{'─'*60}")
    print(f"  By chunk:")
    for cid, avg in stats["by_chunk"].items():
        bar = "█" * int(avg) + "░" * (5 - int(avg))
        flag = "PASS" if avg >= PASS_THRESHOLD else "FAIL"
        print(f"    {cid:<12} {bar} {avg:.2f}  [{flag}]")
    print(f"{'='*60}\n")


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import argparse
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge MCQ evaluator")
    parser.add_argument("file",        nargs="?", help="File slide (PDF/PPTX/DOCX/...)")
    parser.add_argument("--mcq",       help="Path tới MCQ JSON (bỏ qua generate)")
    parser.add_argument("--chunks",    help="Path tới chunks JSON (dùng kèm --mcq)")
    parser.add_argument("--density",   default="Vừa",
                        help="Bloom density: Ít|Vừa|Nhiều (default: Vừa)")
    parser.add_argument("--difficulty",default="medium",          help="easy|medium|hard")
    parser.add_argument("--out", default=None,
                        help="Path output (default: evaluation/results/<file>_<model>_<timestamp>.json)")
    args = parser.parse_args()

    # ── Load MCQs + chunks ──────────────────────────────────────────
    if args.mcq and args.chunks:
        with open(args.mcq, encoding="utf-8") as f:
            mcqs = json.load(f)
        with open(args.chunks, encoding="utf-8") as f:
            chunks = json.load(f)
        _log(f"Loaded {len(mcqs)} MCQs, {len(chunks)} chunks từ file")

    elif args.file:
        from pipeline.file_router import parse_file
        from pipeline.chunker import chunk_pages
        from pipeline.document_analyzer import analyze_document, select_chunk_strategy
        from pipeline.generator import generate_mcqs
        from pipeline.validator import validate_mcqs
        from prompts.bloom_definitions import DENSITY_TO_LEVELS

        _log(f"\nParsing {args.file}...")
        pages    = parse_file(args.file)
        analysis = analyze_document(pages)
        strategy = analysis["chunk_strategy"]
        chunks   = chunk_pages(pages, strategy=strategy)
        _log(f"  {len(pages)} pages → {len(chunks)} chunks  [strategy={strategy}]")

        bloom_levels = DENSITY_TO_LEVELS.get(args.density, DENSITY_TO_LEVELS["Vừa"])
        _log(f"\nGenerating MCQs (PS4 | levels={bloom_levels} | {args.difficulty})...")
        raw_mcqs = generate_mcqs(
            chunks,
            bloom_levels=bloom_levels,
            difficulty=args.difficulty,
            pdf_name=Path(args.file).name,
        )
        mcqs, _ = validate_mcqs(raw_mcqs)
        _log(f"  {len(mcqs)} valid MCQs")

    else:
        parser.print_help()
        sys.exit(1)

    # ── Evaluate ────────────────────────────────────────────────────
    _log(f"\nEvaluating {len(mcqs)} MCQs with LLM judge...")
    _log(f"Pass threshold: semantic_avg >= {PASS_THRESHOLD}\n")

    results = evaluate_all(mcqs, chunks)
    stats   = compute_stats(results)

    # ── Print summary ───────────────────────────────────────────────
    _print_stats(stats)

    # ── Detail per MCQ ──────────────────────────────────────────────
    print("DETAIL PER MCQ:")
    print(f"  {'#':<4} {'CHUNK':<12} {'R':<3} {'A':<3} {'B':<3} {'C':<3} {'AVG':<5} {'PASS':<5} QUESTION")
    print(f"  {'-'*80}")
    for i, r in enumerate(results, 1):
        s    = r["scores"]
        flag = "YES" if r["passed"] else "no"
        print(
            f"  {i:<4} {r['chunk_id']:<12} "
            f"{s.get('relevance',0):<3} {s.get('answerability',0):<3} "
            f"{s.get('bloom_alignment',0):<3} {s.get('answer_correctness',0):<3} "
            f"{r['semantic_avg']:<5} {flag:<5} {r['question'][:40]}"
        )

    # ── Build output path ────────────────────────────────────────────
    from datetime import datetime
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    _, judge_model = _get_judge_client()
    model_slug = judge_model.replace("/", "-").replace(":", "-")

    if args.file:
        file_slug = Path(args.file).stem
    elif args.mcq:
        file_slug = Path(args.mcq).stem
    else:
        file_slug = "unknown"

    out_path = args.out or f"evaluation/results/{file_slug}__{model_slug}__{timestamp}.json"

    # ── Save results ────────────────────────────────────────────────
    output = {
        "meta": {
            "source_file" : args.file or args.mcq or "",
            "judge_model" : judge_model,
            "timestamp"   : timestamp,
            "pass_threshold": PASS_THRESHOLD,
        },
        "stats"  : stats,
        "results": results,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"\nSaved → {out_path}")
