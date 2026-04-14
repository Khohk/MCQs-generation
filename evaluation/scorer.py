"""
evaluation/scorer.py
--------------------
Automatic evaluation metrics cho MCQ quality.

Metrics:
  1. BERTScore F1  — relevance giữa câu hỏi và source chunk
  2. Cosine similarity — distractor plausibility (dùng sentence-transformers)
  3. Bloom mismatch rate — tổng hợp từ validator flag
  4. Answer distribution — kiểm tra bias A/B/C/D

Cài dependencies trước:
  pip install bert-score sentence-transformers torch

Usage:
  python -m evaluation.scorer data/mcqs_NLP_Week_2.json data/chunks_NLP_Week_2.json
  python -m evaluation.scorer data/mcqs_NLP_Week_2.json  # nếu không có chunk file
"""

import json
import sys
import os
import time
from pathlib import Path
from collections import Counter


# ── Constants ──────────────────────────────────────────────────────
SBERT_MODEL   = "all-MiniLM-L6-v2"   # nhẹ ~80MB, đủ tốt cho thesis
BERT_SCORE_LANG = "en"

# Ngưỡng đánh giá distractor cosine similarity
DISTRACTOR_MIN = 0.20   # < min → distractor quá khác biệt (trivial)
DISTRACTOR_MAX = 0.85   # > max → distractor quá giống đáp án (ambiguous)

BLOOM_BY_DIFFICULTY = {
    "easy":   {"remember", "understand"},
    "medium": {"apply", "analyze"},
    "hard":   {"evaluate", "create"},
}


# ── Main scorer function ───────────────────────────────────────────

def score_mcqs(
    mcqs: list[dict],
    chunks: dict | None = None,   # {chunk_id: text} — None nếu không có
    verbose: bool = True,
) -> dict:
    """
    Tính toàn bộ automatic metrics cho list MCQ.

    Args:
        mcqs   : List MCQ valid dict (output từ validator)
        chunks : Dict {chunk_id: chunk_text} để tính BERTScore
                 Nếu None → bỏ qua BERTScore
        verbose: In tiến trình ra terminal

    Returns:
        Dict kết quả đầy đủ
    """
    results = {
        "n_mcqs":          len(mcqs),
        "bert_score":      None,
        "cosine_sim":      None,
        "bloom_mismatch":  None,
        "answer_dist":     None,
        "per_mcq":         [],
    }

    if not mcqs:
        _log("Khong co MCQ nao de score.", verbose)
        return results

    _log(f"\n{'='*55}", verbose)
    _log(f"  Scoring {len(mcqs)} MCQs", verbose)
    _log(f"{'='*55}", verbose)

    # ── 1. BERTScore ─────────────────────────────────────────────
    # Chay duoc ca khi khong co chunks — fallback dung explanation
    if chunks:
        _log("\n[1/3] Tinh BERTScore (question vs chunk text)...", verbose)
    else:
        _log("\n[1/3] Tinh BERTScore (question vs explanation — fallback)...", verbose)
    bert_results = _compute_bert_score(mcqs, chunks, verbose)
    results["bert_score"] = bert_results

    # ── 2. Cosine similarity (distractor plausibility) ───────────
    _log("\n[2/3] Tinh cosine similarity (distractor plausibility)...", verbose)
    cosine_results = _compute_cosine_similarity(mcqs, verbose)
    results["cosine_sim"] = cosine_results

    # ── 3. Bloom mismatch + Answer distribution ──────────────────
    _log("\n[3/3] Tinh Bloom mismatch rate + Answer distribution...", verbose)
    bloom_results = _compute_bloom_stats(mcqs)
    dist_results  = _compute_answer_distribution(mcqs)
    results["bloom_mismatch"] = bloom_results
    results["answer_dist"]    = dist_results

    # ── Per-MCQ summary ──────────────────────────────────────────
    results["per_mcq"] = _build_per_mcq(mcqs, results)

    # ── Print summary ─────────────────────────────────────────────
    _print_summary(results, verbose)

    return results


# ── Metric 1: BERTScore ────────────────────────────────────────────

def _compute_bert_score(mcqs: list[dict], chunks: dict, verbose: bool) -> dict:
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        _log("  [WARN] bert-score chua duoc cai. Chay: pip install bert-score", verbose)
        return {"error": "bert-score not installed", "avg_f1": None}

    candidates = []   # câu hỏi
    references = []   # source chunk text (hoac explanation neu khong co chunk)

    for mcq in mcqs:
        chunk_id   = mcq.get("source_chunk", "")
        chunk_text = chunks.get(chunk_id, "") if chunks else ""
        # Fallback: dung explanation neu khong co chunk text
        ref = chunk_text if chunk_text else mcq.get("explanation", mcq.get("question", ""))
        candidates.append(mcq.get("question", ""))
        references.append(ref)

    try:
        _log(f"  Computing BERTScore cho {len(candidates)} pairs...", verbose)
        P, R, F1 = bert_score_fn(
            candidates, references,
            lang=BERT_SCORE_LANG,
            verbose=False,
            batch_size=16,
        )
        f1_list  = F1.tolist()
        avg_f1   = round(sum(f1_list) / len(f1_list), 4)
        avg_p    = round(sum(P.tolist()) / len(P.tolist()), 4)
        avg_r    = round(sum(R.tolist()) / len(R.tolist()), 4)

        # Phân tầng chất lượng
        good    = sum(1 for f in f1_list if f >= 0.60)
        ok      = sum(1 for f in f1_list if 0.45 <= f < 0.60)
        poor    = sum(1 for f in f1_list if f < 0.45)

        return {
            "avg_f1":   avg_f1,
            "avg_p":    avg_p,
            "avg_r":    avg_r,
            "min_f1":   round(min(f1_list), 4),
            "max_f1":   round(max(f1_list), 4),
            "quality":  {"good (>=0.60)": good, "ok (0.45-0.60)": ok, "poor (<0.45)": poor},
            "per_mcq":  [round(f, 4) for f in f1_list],
        }
    except Exception as e:
        _log(f"  [ERROR] BERTScore failed: {e}", verbose)
        return {"error": str(e), "avg_f1": None}


# ── Metric 2: Cosine similarity (distractor plausibility) ─────────

def _compute_cosine_similarity(mcqs: list[dict], verbose: bool) -> dict:
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        _log("  [WARN] sentence-transformers chua duoc cai.", verbose)
        _log("  Chay: pip install sentence-transformers", verbose)
        return {"error": "sentence-transformers not installed"}

    _log(f"  Loading model {SBERT_MODEL}...", verbose)
    model = SentenceTransformer(SBERT_MODEL)

    per_mcq_cosine = []

    for mcq in mcqs:
        answer = mcq.get("answer", "A")
        correct_text = mcq.get(answer, "")
        distractors  = [mcq.get(o, "") for o in ["A","B","C","D"] if o != answer]

        texts = [correct_text] + distractors
        try:
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms

            correct_emb = embeddings[0]
            dist_embs   = embeddings[1:]

            sims = [float(np.dot(correct_emb, d)) for d in dist_embs]
            per_mcq_cosine.append({
                "chunk_id":    mcq.get("source_chunk", ""),
                "sims":        [round(s, 4) for s in sims],
                "mean_sim":    round(sum(sims)/len(sims), 4),
                "min_sim":     round(min(sims), 4),
                "max_sim":     round(max(sims), 4),
                "trivial":     sum(1 for s in sims if s < DISTRACTOR_MIN),    # quá khác biệt
                "ambiguous":   sum(1 for s in sims if s > DISTRACTOR_MAX),    # quá giống
                "good":        sum(1 for s in sims if DISTRACTOR_MIN <= s <= DISTRACTOR_MAX),
            })
        except Exception as e:
            per_mcq_cosine.append({"error": str(e)})

    # Aggregate
    valid_results = [r for r in per_mcq_cosine if "error" not in r]
    if not valid_results:
        return {"error": "All failed"}

    all_means  = [r["mean_sim"] for r in valid_results]
    all_trivial   = sum(r["trivial"]   for r in valid_results)
    all_ambiguous = sum(r["ambiguous"] for r in valid_results)
    all_good      = sum(r["good"]      for r in valid_results)
    total_distractors = all_trivial + all_ambiguous + all_good

    return {
        "avg_cosine_sim":  round(sum(all_means)/len(all_means), 4),
        "min_cosine_sim":  round(min(all_means), 4),
        "max_cosine_sim":  round(max(all_means), 4),
        "distractor_quality": {
            "good (0.20-0.85)":    all_good,
            "trivial (<0.20)":     all_trivial,
            "ambiguous (>0.85)":   all_ambiguous,
            "total_distractors":   total_distractors,
            "good_rate":           round(all_good / total_distractors * 100, 1) if total_distractors else 0,
        },
        "per_mcq": per_mcq_cosine,
    }


# ── Metric 3: Bloom mismatch ───────────────────────────────────────

def _compute_bloom_stats(mcqs: list[dict]) -> dict:
    mismatch = 0
    by_difficulty = {}

    for mcq in mcqs:
        diff  = mcq.get("difficulty", "unknown")
        bloom = mcq.get("bloom_level", "unknown")
        expected = BLOOM_BY_DIFFICULTY.get(diff, set())
        is_mismatch = bool(expected) and bloom not in expected

        if is_mismatch:
            mismatch += 1

        if diff not in by_difficulty:
            by_difficulty[diff] = {"total": 0, "mismatch": 0, "bloom_counts": {}}
        by_difficulty[diff]["total"] += 1
        if is_mismatch:
            by_difficulty[diff]["mismatch"] += 1
        bc = by_difficulty[diff]["bloom_counts"]
        bc[bloom] = bc.get(bloom, 0) + 1

    # Tính mismatch rate per difficulty
    for d, v in by_difficulty.items():
        v["mismatch_rate"] = round(v["mismatch"] / v["total"] * 100, 1) if v["total"] else 0

    return {
        "total_mismatch":   mismatch,
        "mismatch_rate":    round(mismatch / len(mcqs) * 100, 1) if mcqs else 0,
        "by_difficulty":    by_difficulty,
    }


# ── Metric 4: Answer distribution ─────────────────────────────────

def _compute_answer_distribution(mcqs: list[dict]) -> dict:
    counts = Counter(mcq.get("answer", "?") for mcq in mcqs)
    total  = len(mcqs)
    dist   = {k: {"count": counts.get(k,0),
                  "pct": round(counts.get(k,0)/total*100, 1)} for k in ["A","B","C","D"]}

    # Kiểm tra bias: lý tưởng mỗi option ~25%
    max_pct = max(d["pct"] for d in dist.values())
    bias    = max_pct > 40  # nếu 1 option > 40% → có bias

    return {
        "distribution": dist,
        "is_biased":    bias,
        "note": "Balanced nếu mỗi option 20-30%" if not bias else f"WARNING: bias phát hiện, 1 option > 40%"
    }


# ── Per-MCQ summary ────────────────────────────────────────────────

def _build_per_mcq(mcqs: list[dict], results: dict) -> list[dict]:
    bert_per  = results["bert_score"]["per_mcq"] if results["bert_score"] and "per_mcq" in results["bert_score"] else []
    cos_per   = results["cosine_sim"]["per_mcq"]  if results["cosine_sim"]  and "per_mcq" in results["cosine_sim"]  else []

    per_mcq = []
    for i, mcq in enumerate(mcqs):
        entry = {
            "idx":          i,
            "chunk_id":     mcq.get("source_chunk", ""),
            "bloom_level":  mcq.get("bloom_level", ""),
            "difficulty":   mcq.get("difficulty", ""),
            "question":     mcq.get("question", "")[:80],
        }
        if i < len(bert_per):
            entry["bert_f1"] = bert_per[i]
        if i < len(cos_per) and "error" not in cos_per[i]:
            entry["cosine_mean"] = cos_per[i]["mean_sim"]
            entry["distractor_trivial"]   = cos_per[i]["trivial"]
            entry["distractor_ambiguous"] = cos_per[i]["ambiguous"]

        # Bloom mismatch flag
        diff  = mcq.get("difficulty", "")
        bloom = mcq.get("bloom_level", "")
        exp   = BLOOM_BY_DIFFICULTY.get(diff, set())
        entry["bloom_mismatch"] = bool(exp) and bloom not in exp

        per_mcq.append(entry)

    return per_mcq


# ── Print summary ─────────────────────────────────────────────────

def _print_summary(results: dict, verbose: bool):
    if not verbose:
        return

    print(f"\n{'='*55}")
    print(f"  SCORING SUMMARY — {results['n_mcqs']} MCQs")
    print(f"{'='*55}")

    # BERTScore
    bs = results.get("bert_score")
    if bs and bs.get("avg_f1") is not None:
        print(f"\n  BERTScore (question relevance to source):")
        print(f"    Avg F1  : {bs['avg_f1']:.4f}  (good >= 0.60)")
        print(f"    Range   : {bs['min_f1']:.4f} - {bs['max_f1']:.4f}")
        q = bs.get("quality", {})
        print(f"    Quality : good={q.get('good (>=0.60)',0)} | ok={q.get('ok (0.45-0.60)',0)} | poor={q.get('poor (<0.45)',0)}")

    # Cosine sim
    cs = results.get("cosine_sim")
    if cs and "avg_cosine_sim" in cs:
        print(f"\n  Distractor Plausibility (cosine sim):")
        print(f"    Avg sim : {cs['avg_cosine_sim']:.4f}  (ideal: 0.20-0.85)")
        dq = cs.get("distractor_quality", {})
        print(f"    Good    : {dq.get('good (0.20-0.85)',0)} distractors ({dq.get('good_rate',0)}%)")
        print(f"    Trivial : {dq.get('trivial (<0.20)',0)}  (qua khac biet, de loai)")
        print(f"    Ambig   : {dq.get('ambiguous (>0.85)',0)}  (qua giong dap an dung)")

    # Bloom mismatch
    bm = results.get("bloom_mismatch")
    if bm:
        print(f"\n  Bloom/Difficulty Consistency:")
        print(f"    Mismatch: {bm['total_mismatch']}/{results['n_mcqs']} ({bm['mismatch_rate']}%)")
        for diff, v in bm.get("by_difficulty", {}).items():
            print(f"    [{diff}] mismatch={v['mismatch_rate']}% | blooms={v['bloom_counts']}")

    # Answer distribution
    ad = results.get("answer_dist")
    if ad:
        print(f"\n  Answer Distribution:")
        for opt, v in ad.get("distribution", {}).items():
            bar = "#" * int(v["pct"] / 5)
            print(f"    {opt}: {bar:<20} {v['count']:2d} ({v['pct']}%)")
        print(f"    {ad.get('note','')}")

    print(f"\n{'='*55}\n")


def _log(msg: str, verbose: bool = True):
    if verbose:
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode("ascii", errors="replace").decode("ascii"))


# ── Save results ──────────────────────────────────────────────────

def save_results(results: dict, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m evaluation.scorer <mcqs.json> [chunks.json]")
        print()
        print("Chuan bi file:")
        print("  mcqs.json   : export tu Tab Export > Download MCQs.json")
        print("  chunks.json : tu chunker (optional, can cho BERTScore)")
        print()
        print("Vi du:")
        print("  python -m evaluation.scorer data/mcqs_NLP_Week_2.json")
        sys.exit(1)

    mcq_path   = sys.argv[1]
    chunk_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Load MCQs
    print(f"\nLoading MCQs tu: {mcq_path}")
    with open(mcq_path, encoding="utf-8") as f:
        mcqs = json.load(f)
    print(f"Loaded {len(mcqs)} MCQs")

    # Load chunks (optional — neu co thi BERTScore chinh xac hon)
    chunks = None
    if chunk_path and Path(chunk_path).exists():
        print(f"Loading chunks tu: {chunk_path}")
        with open(chunk_path, encoding="utf-8") as f:
            chunk_list = json.load(f)
        chunks = {c["chunk_id"]: c["text"] for c in chunk_list}
        print(f"Loaded {len(chunks)} chunks")
    else:
        print("Khong co chunk file → BERTScore dung explanation lam reference (fallback)")

    # Run scorer
    results = score_mcqs(mcqs, chunks=chunks, verbose=True)

    # Save output
    stem = Path(mcq_path).stem.replace("mcqs_", "")
    out_path = f"data/scores_{stem}.json"
    save_results(results, out_path)

    print(f"\nDone. Ket qua luu tai: {out_path}")
    print("Dung file nay de bao cao trong luan an.")