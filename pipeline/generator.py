"""
pipeline/generator.py
---------------------
List[chunk] → List[MCQ raw JSON]

Features:
- Model rotation khi bị 429 rate limit
- Checkpoint: lưu sau mỗi chunk, resume nếu bị ngắt giữa chừng
- Estimated time trước khi chạy
"""

import json
import time
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

from prompts.mcq_prompt import build_prompt

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────
# Free tier limits (2026):
#   gemini-2.5-flash       : Latest, best for MCQ generation
#   gemini-2.5-flash-lite  : Lighter version, faster
#   gemini-2.0-flash-lite  : Fallback if newer models rate limited
# LUU Y: tat ca model trong cung 1 project dung chung daily quota
# => rotate chi giup khi bi RPM (per-minute), KHONG giup khi het RPD (per-day)
# => Neu het RPD: phai doi reset luc midnight Pacific (14h-15h gio VN)
# Actual available models (confirmed 2026-04-09)
MODEL_FALLBACKS = [
    "models/gemini-2.5-flash",       # PRIMARY: Latest, best quality
    "models/gemini-2.5-flash-lite",  # SECONDARY: Lighter, might have better quota
    "models/gemini-2.0-flash-lite",  # TERTIARY: Older but reliable fallback
]
RETRY_LIMIT   = 3
RETRY_DELAY   = 10.0
REQUEST_DELAY = 4.5   # ~13 RPM, an toàn với limit 15 RPM

CHECKPOINT_DIR = Path("data/checkpoints")


# ── Client & model state ───────────────────────────────────────────
_client = None
_current_model_idx = 0

def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY khong tim thay trong .env")
        _client = genai.Client(api_key=api_key)
    return _client

def _get_model() -> str:
    return MODEL_FALLBACKS[_current_model_idx]

def _next_model() -> str:
    global _current_model_idx
    _current_model_idx = (_current_model_idx + 1) % len(MODEL_FALLBACKS)
    model = MODEL_FALLBACKS[_current_model_idx]
    _log(f"    Switch model -> {model.split('/')[-1]}")
    return model


# ── Checkpoint helpers ─────────────────────────────────────────────

def _checkpoint_path(pdf_name: str) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    safe = pdf_name.replace(".pdf", "").replace(" ", "_")
    return CHECKPOINT_DIR / f"checkpoint_{safe}.json"

def _load_checkpoint(pdf_name: str) -> dict:
    """
    Load checkpoint nếu tồn tại.
    Returns: {chunk_id: [mcq, ...], ...}
    """
    path = _checkpoint_path(pdf_name)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            _log(f"  [checkpoint] Loaded {len(data)} chunks da lam tu {path.name}")
            return data
        except Exception:
            _log("  [checkpoint] Doc loi, bo qua checkpoint")
    return {}

def _save_checkpoint(pdf_name: str, done: dict):
    """Lưu checkpoint dict {chunk_id: [mcqs]}."""
    path = _checkpoint_path(pdf_name)
    path.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")

def clear_checkpoint(pdf_name: str):
    """Xóa checkpoint sau khi hoàn tất — gọi từ app.py khi export."""
    path = _checkpoint_path(pdf_name)
    if path.exists():
        path.unlink()
        _log(f"  [checkpoint] Cleared {path.name}")

def get_checkpoint_info(pdf_name: str) -> dict | None:
    """Trả về info checkpoint để hiển thị trong UI."""
    path = _checkpoint_path(pdf_name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        total_mcqs = sum(len(v) for v in data.values())
        return {
            "chunks_done": len(data),
            "mcqs_so_far": total_mcqs,
            "path": str(path),
        }
    except Exception:
        return None


# ── Estimate time ──────────────────────────────────────────────────

def estimate_seconds(n_chunks: int, n_done: int = 0) -> int:
    """Ước tính thời gian chạy còn lại (seconds)."""
    remaining = n_chunks - n_done
    return int(remaining * REQUEST_DELAY)


# ── Main function ──────────────────────────────────────────────────

def generate_mcqs(
    chunks: list[dict],
    n_per_chunk: int = 2,
    difficulty: str = "medium",
    pdf_name: str = "unknown",
    on_progress=None,   # callback(idx, total, chunk_id, eta_seconds)
) -> list[dict]:
    """
    Sinh MCQ từ list chunks với checkpoint và model rotation.

    Args:
        chunks      : output từ chunker.chunk_pages()
        n_per_chunk : số MCQ mỗi chunk
        difficulty  : "easy" | "medium" | "hard"
        pdf_name    : tên file PDF — dùng để đặt tên checkpoint
        on_progress : callback(idx, total, chunk_id, eta_sec)

    Returns:
        List[MCQ dict] — tất cả MCQ từ mọi chunk, chưa validate
    """
    # Load checkpoint — skip chunks đã làm
    done_map = _load_checkpoint(pdf_name)  # {chunk_id: [mcqs]}
    total    = len(chunks)
    n_done   = len(done_map)

    if n_done > 0:
        _log(f"  Resume: {n_done}/{total} chunks da co trong checkpoint")

    for idx, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]

        # Skip nếu đã có trong checkpoint
        if chunk_id in done_map:
            if on_progress:
                eta = estimate_seconds(total, idx + 1)
                on_progress(idx, total, chunk_id, eta, skipped=True)
            continue

        if on_progress:
            eta = estimate_seconds(total, idx)
            on_progress(idx, total, chunk_id, eta, skipped=False)

        _log(f"  [{idx+1}/{total}] {chunk_id} — {chunk['topic'][:40]}")

        mcqs = _generate_for_chunk(chunk, n_per_chunk, difficulty)

        # Lưu vào checkpoint dù có MCQ hay không (tránh retry vô tận)
        done_map[chunk_id] = mcqs
        _save_checkpoint(pdf_name, done_map)

        if idx < total - 1:
            time.sleep(REQUEST_DELAY)

    # Flatten kết quả theo thứ tự chunk
    all_mcqs = []
    for chunk in chunks:
        all_mcqs.extend(done_map.get(chunk["chunk_id"], []))

    _log(f"\nTotal MCQs generated: {len(all_mcqs)}")
    return all_mcqs


# ── Per-chunk generation ───────────────────────────────────────────

def _generate_for_chunk(chunk: dict, n_questions: int, difficulty: str) -> list[dict]:
    """Gọi Gemini cho 1 chunk, retry + rotate model nếu bị rate limit."""
    prompt   = build_prompt(chunk, n_questions=n_questions, difficulty=difficulty)
    client   = _get_client()
    raw_text = ""

    rate_limit_count = 0

    for attempt in range(1, RETRY_LIMIT + 1):
        model = _get_model()
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.7,
                    max_output_tokens=2048,
                ),
            )
            raw_text = response.text.strip()
            mcqs = _parse_response(raw_text, chunk["chunk_id"])
            _log(f"    -> {len(mcqs)} MCQs OK [{model.split('/')[-1]}]")
            # Reset ve model chinh sau khi thanh cong
            global _current_model_idx
            _current_model_idx = 0
            return mcqs

        except json.JSONDecodeError as e:
            _log(f"    [attempt {attempt}] JSON error: {e}")
            _log(f"    Raw preview: {raw_text[:200]}")

        except Exception as e:
            err = str(e)
            _log(f"    [attempt {attempt}] {err[:120]}")
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                rate_limit_count += 1
                _next_model()
                if rate_limit_count >= len(MODEL_FALLBACKS):
                    # Tat ca model deu bi rate limit — cho 60s roi thu lai
                    _log(f"    Tat ca model bi rate limit, cho 90s (het RPD thi phai doi den midnight Pacific)...")
                    time.sleep(90)
                    rate_limit_count = 0
                    _current_model_idx = 0
                else:
                    time.sleep(3)
                continue
            elif "404" in err or "NOT_FOUND" in err:
                # Model khong ton tai — bo qua, thu model tiep theo
                _log(f"    Model {model} khong kha dung, thu model tiep theo")
                _next_model()
                continue

        if attempt < RETRY_LIMIT:
            time.sleep(RETRY_DELAY)

    _log(f"    SKIP {chunk['chunk_id']} after {RETRY_LIMIT} attempts")
    return []


def _parse_response(raw: str, chunk_id: str) -> list[dict]:
    text = raw
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()

    data = json.loads(text)
    if isinstance(data, dict):
        data = [data]

    for mcq in data:
        mcq["source_chunk"] = chunk_id
    return data


# ── Utility ────────────────────────────────────────────────────────

def _log(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


# ── CLI test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.generator <pdf> [n_chunks] [difficulty]")
        sys.exit(1)

    pdf_path   = sys.argv[1]
    n_chunks   = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    difficulty = sys.argv[3] if len(sys.argv) > 3 else "medium"

    from pipeline.pdf_parser import parse_pdf
    from pipeline.chunker import chunk_pages

    pages  = parse_pdf(pdf_path)
    chunks = chunk_pages(pages)[:n_chunks]

    pdf_name = Path(pdf_path).name
    eta = estimate_seconds(len(chunks))
    print(f"\nETA: ~{eta}s for {len(chunks)} chunks\n")

    mcqs = generate_mcqs(chunks, n_per_chunk=2, difficulty=difficulty, pdf_name=pdf_name)

    print(f"\n{'='*55}")
    print(f"Total: {len(mcqs)} MCQs")
    print(f"{'='*55}\n")
    for i, m in enumerate(mcqs, 1):
        print(f"--- MCQ {i} [{m.get('bloom_level')}/{m.get('difficulty')}] ---")
        print(f"Q: {m.get('question')}")
        print(f"A: {m.get('A')}  B: {m.get('B')}")
        print(f"C: {m.get('C')}  D: {m.get('D')}")
        print(f"Ans: {m.get('answer')} | {m.get('explanation','')[:80]}")
        print()