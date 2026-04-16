"""
pipeline/generator.py
---------------------
List[chunk] → List[MCQ raw JSON]

Features:
- Multi-provider fallback: Gemini → Cerebras → Groq → OpenRouter
- Checkpoint: lưu sau mỗi chunk, resume nếu bị ngắt giữa chừng
- Estimated time trước khi chạy

.env cần:
  GEMINI_API_KEY=...
  CEREBRAS_API_KEY=...
  GROQ_API_KEY=...
  OPENROUTER_API_KEY=...   (optional)
"""

import json
import sys
import time
import os
import re
from pathlib import Path
from dotenv import load_dotenv

# Đảm bảo project root trong sys.path khi chạy trực tiếp
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompts.mcq_prompt import build_prompt

load_dotenv()

# ── Provider config ────────────────────────────────────────────────
# tpm  : token/phút — dùng để tính REQUEST_DELAY
# Thứ tự = thứ tự ưu tiên (A → B → C khi bị 429)

PROVIDERS = [
    # Primary — Gemini (chất lượng tốt nhất, JSON mode native)
    {"provider": "gemini",     "model": "gemini-2.5-flash",         "tpm": 1_000_000},
    {"provider": "gemini",     "model": "gemini-2.0-flash",         "tpm": 1_000_000},
    {"provider": "gemini",     "model": "gemini-2.0-flash-lite",    "tpm": 1_000_000},

    # Fallback 1 — Cerebras (nhanh, 1M token/ngày, TPM cao)
    {"provider": "cerebras",   "model": "llama3.3-70b",             "tpm": 60_000},

    # Fallback 2 — Groq (free nhưng TPM thấp)
    {"provider": "groq",       "model": "llama-3.3-70b-versatile",  "tpm": 6_000},
    {"provider": "groq",       "model": "llama4-scout-17b-16e",     "tpm": 6_000},

    # Fallback 3 — OpenRouter (last resort)
    {"provider": "openrouter", "model": "meta-llama/llama-4-scout:free", "tpm": 20_000},
]

RETRY_LIMIT    = 2      # số lần retry trên cùng 1 provider trước khi chuyển
SWITCH_DELAY   = 2.0    # giây chờ khi switch provider
REQUEST_DELAY  = 4.5    # giây giữa các request (safe với Gemini 15 RPM)

CHECKPOINT_DIR = Path("data/checkpoints")


# ── Provider state ─────────────────────────────────────────────────
_current_provider_idx = 0
_provider_cooldown: dict[int, float] = {}   # idx → thời điểm có thể dùng lại
COOLDOWN_SECONDS = 60.0


# ── Client factory ─────────────────────────────────────────────────

def _get_gemini_client():
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")
    return genai.Client(api_key=api_key)


def _get_openai_compat_client(provider: str):
    from openai import OpenAI
    base_urls = {
        "groq"      : "https://api.groq.com/openai/v1",
        "cerebras"  : "https://api.cerebras.ai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }
    env_keys = {
        "groq"      : "GROQ_API_KEY",
        "cerebras"  : "CEREBRAS_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    api_key = os.getenv(env_keys[provider], "").strip()
    if not api_key:
        raise ValueError(f"{env_keys[provider]} not found in .env")
    return OpenAI(base_url=base_urls[provider], api_key=api_key)


# ── Provider selection ─────────────────────────────────────────────

def _next_available_provider() -> int | None:
    """
    Tìm provider tiếp theo không bị cooldown.
    Trả về index hoặc None nếu tất cả đang cooling down.
    """
    now = time.time()
    for i in range(len(PROVIDERS)):
        idx = (_current_provider_idx + i) % len(PROVIDERS)
        if now >= _provider_cooldown.get(idx, 0):
            return idx
    return None


def _mark_cooldown(idx: int):
    """Đánh dấu provider đang bị rate limit, chờ COOLDOWN_SECONDS."""
    _provider_cooldown[idx] = time.time() + COOLDOWN_SECONDS
    p = PROVIDERS[idx]
    _log(f"    [cooldown] {p['provider']}/{p['model'].split('/')[-1]} "
         f"cooling down {COOLDOWN_SECONDS}s")


# ── Unified API call ───────────────────────────────────────────────

def _call_provider(prompt: str, idx: int) -> str:
    """
    Gọi 1 provider và trả về raw text response.
    Raise exception nếu lỗi để caller xử lý.
    """
    p = PROVIDERS[idx]
    provider = p["provider"]
    model    = p["model"]

    if provider == "gemini":
        from google.genai import types
        client   = _get_gemini_client()
        response = client.models.generate_content(
            model=f"models/{model}",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.7,
                max_output_tokens=2048,
            ),
        )
        return response.text.strip()

    else:
        # Groq / Cerebras / OpenRouter — tất cả OpenAI-compatible
        client   = _get_openai_compat_client(provider)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=2048,
        )
        return response.choices[0].message.content.strip()


# ── Per-chunk generation ───────────────────────────────────────────

def _generate_for_chunk(chunk: dict, n_questions: int, difficulty: str) -> list[dict]:
    """
    Gọi API cho 1 chunk.
    Nếu provider hiện tại bị 429 → switch sang provider tiếp theo.
    """
    global _current_provider_idx

    prompt   = build_prompt(chunk, n_questions=n_questions, difficulty=difficulty)
    raw_text = ""

    # Thử lần lượt từng provider cho đến khi có kết quả
    attempts_total = len(PROVIDERS) * RETRY_LIMIT

    for attempt in range(attempts_total):
        idx = _next_available_provider()

        if idx is None:
            # Tất cả đang cooldown — chờ provider nào hết sớm nhất
            wait = min(_provider_cooldown.values()) - time.time()
            wait = max(wait, 1.0)
            _log(f"    Tat ca provider dang cooldown, cho {wait:.0f}s...")
            time.sleep(wait)
            idx = _next_available_provider()
            if idx is None:
                break

        _current_provider_idx = idx
        p = PROVIDERS[idx]
        label = f"{p['provider']}/{p['model'].split('/')[-1]}"

        try:
            raw_text = _call_provider(prompt, idx)
            mcqs     = _parse_response(raw_text, chunk["chunk_id"])
            _log(f"    -> {len(mcqs)} MCQs OK [{label}]")
            return mcqs

        except json.JSONDecodeError as e:
            _log(f"    [attempt {attempt+1}] JSON parse error [{label}]: {e}")
            _log(f"    Raw: {raw_text[:150]}")
            # JSON error không phải lỗi provider → retry cùng provider 1 lần
            time.sleep(2)

        except Exception as e:
            err = str(e)
            _log(f"    [attempt {attempt+1}] {label}: {err[:100]}")

            is_rate_limit = any(k in err for k in ("429", "RESOURCE_EXHAUSTED", "rate_limit", "RateLimitError"))
            is_not_found  = any(k in err for k in ("404", "NOT_FOUND", "model_not_found"))
            is_no_key     = "not found in .env" in err

            if is_rate_limit or is_not_found or is_no_key:
                _mark_cooldown(idx)
                # Chuyển ngay sang provider tiếp theo
                next_idx = _next_available_provider()
                if next_idx is not None and next_idx != idx:
                    _current_provider_idx = next_idx
                    next_p = PROVIDERS[next_idx]
                    _log(f"    Switch -> {next_p['provider']}/{next_p['model'].split('/')[-1]}")
                time.sleep(SWITCH_DELAY)
            else:
                # Lỗi khác (network, timeout) — retry cùng provider
                time.sleep(5)

    _log(f"    SKIP {chunk['chunk_id']} sau {attempts_total} attempts")
    return []


def _parse_response(raw: str, chunk_id: str) -> list[dict]:
    text = raw
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()

    data = json.loads(text)
    if isinstance(data, dict):
        # OpenAI json_object đôi khi wrap trong {"mcqs": [...]}
        for key in ("mcqs", "questions", "items", "data"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            data = [data]

    for mcq in data:
        mcq["source_chunk"] = chunk_id
    return data


# ── Checkpoint helpers ─────────────────────────────────────────────

def _checkpoint_path(file_name: str) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-]", "_", Path(file_name).stem)
    return CHECKPOINT_DIR / f"checkpoint_{safe}.json"

def _load_checkpoint(file_name: str) -> dict:
    path = _checkpoint_path(file_name)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            _log(f"  [checkpoint] Loaded {len(data)} chunks da lam tu {path.name}")
            return data
        except Exception:
            _log("  [checkpoint] Doc loi, bo qua checkpoint")
    return {}

def _save_checkpoint(file_name: str, done: dict):
    path = _checkpoint_path(file_name)
    path.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")

def clear_checkpoint(file_name: str):
    path = _checkpoint_path(file_name)
    if path.exists():
        path.unlink()
        _log(f"  [checkpoint] Cleared {path.name}")

def get_checkpoint_info(file_name: str) -> dict | None:
    path = _checkpoint_path(file_name)
    if not path.exists():
        return None
    try:
        data  = json.loads(path.read_text(encoding="utf-8"))
        total = sum(len(v) for v in data.values())
        return {"chunks_done": len(data), "mcqs_so_far": total, "path": str(path)}
    except Exception:
        return None


# ── Estimate time ──────────────────────────────────────────────────

def estimate_seconds(n_chunks: int, n_done: int = 0) -> int:
    remaining = n_chunks - n_done
    return int(remaining * REQUEST_DELAY)


# ── Main function ──────────────────────────────────────────────────

def generate_mcqs(
    chunks: list[dict],
    n_per_chunk: int = 2,
    difficulty: str = "medium",
    pdf_name: str = "unknown",
    on_progress=None,   # callback(idx, total, chunk_id, eta_seconds, skipped)
) -> list[dict]:
    """
    Sinh MCQ từ list chunks với multi-provider fallback + checkpoint.

    Args:
        chunks      : output từ chunker.chunk_pages()
        n_per_chunk : số MCQ mỗi chunk
        difficulty  : "easy" | "medium" | "hard"
        pdf_name    : tên file — dùng để đặt tên checkpoint
        on_progress : callback(idx, total, chunk_id, eta_sec, skipped)

    Returns:
        List[MCQ dict] — tất cả MCQ từ mọi chunk, chưa validate
    """
    done_map = _load_checkpoint(pdf_name)
    total    = len(chunks)
    n_done   = len(done_map)

    if n_done > 0:
        _log(f"  Resume: {n_done}/{total} chunks da co trong checkpoint")

    # In provider list đang active
    available = [p for p in PROVIDERS if os.getenv(
        {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY",
         "cerebras": "CEREBRAS_API_KEY", "openrouter": "OPENROUTER_API_KEY"}[p["provider"]], ""
    ).strip()]
    labels = [p["provider"] + "/" + p["model"].split("/")[-1] for p in available]
    _log(f"  Active providers: {labels}")

    for idx, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]

        if chunk_id in done_map:
            if on_progress:
                on_progress(idx, total, chunk_id, estimate_seconds(total, idx+1), skipped=True)
            continue

        if on_progress:
            on_progress(idx, total, chunk_id, estimate_seconds(total, idx), skipped=False)

        _log(f"\n  [{idx+1}/{total}] {chunk_id} — {chunk['topic'][:45]}")

        mcqs = _generate_for_chunk(chunk, n_per_chunk, difficulty)

        done_map[chunk_id] = mcqs
        _save_checkpoint(pdf_name, done_map)

        if idx < total - 1:
            time.sleep(REQUEST_DELAY)

    all_mcqs = []
    for chunk in chunks:
        all_mcqs.extend(done_map.get(chunk["chunk_id"], []))

    _log(f"\nTotal MCQs generated: {len(all_mcqs)}")
    return all_mcqs


# ── Utility ────────────────────────────────────────────────────────

def _log(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


# ── CLI test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("Usage: python pipeline/generator.py <file> [n_chunks] [difficulty]")
        print("       file      : .pdf | .pptx | .docx | ...")
        print("       n_chunks  : số chunk test (default: 2)")
        print("       difficulty: easy | medium | hard (default: medium)")
        sys.exit(1)

    file_path  = sys.argv[1]
    n_chunks   = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    difficulty = sys.argv[3] if len(sys.argv) > 3 else "medium"

    from pipeline.file_router import parse_file
    from pipeline.chunker import chunk_pages

    pages  = parse_file(file_path)
    chunks = chunk_pages(pages)[:n_chunks]

    file_name = Path(file_path).name
    print(f"\n{'='*55}")
    print(f"  File      : {file_name}")
    print(f"  Chunks    : {len(chunks)}")
    print(f"  Difficulty: {difficulty}")
    print(f"  ETA       : ~{estimate_seconds(len(chunks))}s")
    print(f"{'='*55}\n")

    mcqs = generate_mcqs(chunks, n_per_chunk=2, difficulty=difficulty, pdf_name=file_name)

    print(f"\n{'='*55}")
    print(f"Total: {len(mcqs)} MCQs")
    print(f"{'='*55}\n")
    for i, m in enumerate(mcqs, 1):
        print(f"--- MCQ {i} [{m.get('bloom_level','?')}/{m.get('difficulty','?')}] ---")
        print(f"Q: {m.get('question','')}")
        print(f"A: {m.get('A','')}  |  B: {m.get('B','')}")
        print(f"C: {m.get('C','')}  |  D: {m.get('D','')}")
        print(f"Ans: {m.get('answer','?')} — {m.get('explanation','')[:80]}")
        print()
