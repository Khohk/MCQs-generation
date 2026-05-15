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

from __future__ import annotations

import json
import sys
import time
import os
import re
import hashlib
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Đảm bảo project root trong sys.path khi chạy trực tiếp
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompts.build_ps4_prompt import build_ps4_prompt
from pipeline.schemas import MCQItem
from pydantic import ValidationError

load_dotenv()

DIFFICULTY_TO_BLOOM: dict[str, list[str]] = {
    "easy":   ["remember", "understand"],
    "medium": ["apply", "analyze"],
    "hard":   ["evaluate", "create"],
}

# ── Provider config ────────────────────────────────────────────────
# tpm  : token/phút — dùng để tính REQUEST_DELAY
# Thứ tự = thứ tự ưu tiên (A → B → C khi bị 429)

PROVIDERS = [
    # Primary — Gemini (chất lượng tốt nhất, JSON mode native)
    {"provider": "gemini",     "model": "gemini-3.1-flash-lite-preview", "tpm": 1_000_000},
    {"provider": "gemini",     "model": "gemini-3-flash-preview",        "tpm": 1_000_000},
    {"provider": "gemini",     "model": "gemini-2.5-flash",              "tpm": 1_000_000},

    # Fallback 1 — Cerebras (nhanh, 1M token/ngày, TPM cao)
    {"provider": "cerebras",   "model": "llama3.1-8b",                   "tpm": 60_000},

    # Fallback 2 — Groq (free, nhiều model)
    {"provider": "groq",       "model": "llama-3.3-70b-versatile",       "tpm": 6_000},
    {"provider": "groq",       "model": "llama-3.1-70b-versatile",       "tpm": 6_000},
    {"provider": "groq",       "model": "meta-llama/llama-4-scout-17b-16e-instruct", "tpm": 6_000},
    {"provider": "groq",       "model": "meta-llama/llama-4-maverick-17b-128e-instruct", "tpm": 6_000},
    {"provider": "groq",       "model": "gemma2-9b-it",                  "tpm": 15_000},
    {"provider": "groq",       "model": "mixtral-8x7b-32768",            "tpm": 5_000},

    # Fallback 3 — OpenRouter (free tier)
    {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free",  "tpm": 20_000},
    {"provider": "openrouter", "model": "google/gemma-3-27b-it:free",              "tpm": 15_000},
    {"provider": "openrouter", "model": "mistralai/mistral-7b-instruct:free",      "tpm": 10_000},
    {"provider": "openrouter", "model": "deepseek/deepseek-r1-distill-llama-70b:free", "tpm": 10_000},
]

RETRY_LIMIT    = 2      # số lần retry trên cùng 1 provider trước khi chuyển
SWITCH_DELAY   = 2.0    # giây chờ khi switch provider
REQUEST_DELAY  = 4.5    # giây giữa các request (safe với Gemini 15 RPM)

CHECKPOINT_DIR = Path("data/checkpoints")


# ── Provider state ─────────────────────────────────────────────────
_current_provider_idx = 0
_provider_cooldown: dict[int, float] = {}   # idx → thời điểm có thể dùng lại
_provider_disabled: set[int] = set()        # idx → disabled vĩnh viễn (404/no key)
COOLDOWN_SECONDS = 60.0

# ── Provider usage tracking ────────────────────────────────────────
_provider_call_counts: dict[int, int] = {}      # idx → số lần gọi thành công
_provider_tokens: dict[int, int] = {}           # idx → tổng token (prompt + completion)
_provider_response_times: dict[int, list] = {}  # idx → list[float] latency mỗi call (giây)
_session_chunks_skipped: list = []              # chunk_id bị skip (0 MCQ sau hết attempts)


# ── Client factory ─────────────────────────────────────────────────

_provider_chunk_logs: list[dict] = []


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
    Tìm provider tiếp theo không bị cooldown và không bị disabled.
    Trả về index hoặc None nếu tất cả đang cooling down.
    """
    now = time.time()
    for i in range(len(PROVIDERS)):
        idx = (_current_provider_idx + i) % len(PROVIDERS)
        if idx in _provider_disabled:
            continue
        if now >= _provider_cooldown.get(idx, 0):
            return idx
    return None


def _mark_cooldown(idx: int):
    """Đánh dấu provider bị rate limit, chờ COOLDOWN_SECONDS."""
    _provider_cooldown[idx] = time.time() + COOLDOWN_SECONDS
    p = PROVIDERS[idx]
    _log(f"    [cooldown] {p['provider']}/{p['model'].split('/')[-1]} "
         f"cooling down {COOLDOWN_SECONDS}s")


def _mark_disabled(idx: int, reason: str):
    """Disable vĩnh viễn trong session — dùng cho 404 và missing key."""
    _provider_disabled.add(idx)
    p = PROVIDERS[idx]
    _log(f"    [disabled] {p['provider']}/{p['model'].split('/')[-1]}: {reason}")


# ── Unified API call ───────────────────────────────────────────────

def _call_provider(
    prompt: str,
    idx: int,
    json_mode: bool = True,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> str:
    """
    Gọi 1 provider và trả về raw text response.
    Raise exception nếu lỗi để caller xử lý.

    Args:
        json_mode  : True → ép response format JSON (dùng cho MCQ)
                     False → plain text (dùng cho slide summary)
        max_tokens : output token limit — tăng lên cho whole_doc (nhiều MCQs)
    """
    p = PROVIDERS[idx]
    provider = p["provider"]
    model    = p["model"]

    t0 = time.time()

    if provider == "gemini":
        from google.genai import types
        client     = _get_gemini_client()
        cfg_kwargs = {"temperature": temperature, "max_output_tokens": max_tokens}
        if json_mode:
            cfg_kwargs["response_mime_type"] = "application/json"
            cfg_kwargs["response_schema"]    = list[MCQItem]  # enforce schema ở phía model
        response = client.models.generate_content(
            model=f"models/{model}",
            contents=prompt,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
        text   = response.text.strip()
        tokens = getattr(response.usage_metadata, "total_token_count", 0) or 0

    else:
        # Groq / Cerebras / OpenRouter — tất cả OpenAI-compatible
        client  = _get_openai_compat_client(provider)
        kwargs  = {
            "model"      : model,
            "messages"   : [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens" : max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        text   = response.choices[0].message.content.strip()
        usage  = getattr(response, "usage", None)
        tokens = getattr(usage, "total_tokens", 0) or 0

    elapsed = round(time.time() - t0, 2)
    _provider_call_counts[idx] = _provider_call_counts.get(idx, 0) + 1
    _provider_tokens[idx]      = _provider_tokens.get(idx, 0) + tokens
    _provider_response_times.setdefault(idx, []).append(elapsed)
    return text


def get_provider_stats() -> dict:
    """Trả về stats đầy đủ của session: per-provider + chunks skipped."""
    providers = []
    for idx, count in _provider_call_counts.items():
        p      = PROVIDERS[idx]
        times  = _provider_response_times.get(idx, [])
        avg_ms = round(sum(times) / len(times) * 1000) if times else 0
        providers.append({
            "provider"    : p["provider"],
            "model"       : p["model"],
            "calls"       : count,
            "tokens"      : _provider_tokens.get(idx, 0),
            "avg_latency_ms": avg_ms,
            "disabled"    : idx in _provider_disabled,
        })
    return {
        "providers"     : sorted(providers, key=lambda x: -x["calls"]),
        "chunks_skipped": list(_session_chunks_skipped),
        "chunk_logs"    : list(_provider_chunk_logs),
    }


def reset_provider_stats():
    _provider_call_counts.clear()
    _provider_tokens.clear()
    _provider_response_times.clear()
    _session_chunks_skipped.clear()
    _provider_chunk_logs.clear()


# ── Per-chunk generation ───────────────────────────────────────────

def _generate_for_chunk(
    chunk: dict,
    prompt: str,
    max_tokens: int = 2048,
    call_label: str = "",
) -> list[dict]:
    """
    Gọi API với prompt đã build sẵn. Nếu provider bị 429 → switch.
    Trả về [] nếu model trả skip hoặc sau khi hết attempts.

    Args:
        call_label: key để log (mặc định chunk_id, PS4 dùng "chunk_001__remember")
    """
    global _current_provider_idx

    log_key  = call_label or chunk["chunk_id"]
    raw_text = ""

    # Thử lần lượt từng provider cho đến khi có kết quả
    attempts_total = len(PROVIDERS) * RETRY_LIMIT

    for attempt in range(attempts_total):
        idx = _next_available_provider()

        if idx is None:
            # Kiểm tra còn provider nào không bị disabled không
            active = [i for i in range(len(PROVIDERS)) if i not in _provider_disabled]
            if not active:
                _log("    Tat ca provider deu bi disabled (404/no key). ABORT.")
                break
            # Chờ provider cooldown sớm nhất trong số active
            active_cooldowns = {i: _provider_cooldown.get(i, 0) for i in active}
            wait = min(active_cooldowns.values()) - time.time()
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
            t0 = time.time()
            raw_text = _call_provider(prompt, idx, max_tokens=max_tokens)
            mcqs     = _parse_response(raw_text, chunk["chunk_id"])
            latency = round(time.time() - t0, 2)
            _record_provider_log(
                chunk_id=log_key,
                provider=p["provider"],
                model=p["model"],
                status="success",
                retry_count=attempt,
                latency_seconds=latency,
                generated_count=len(mcqs),
            )
            _log(f"    -> {len(mcqs)} MCQs OK [{label}]")
            return mcqs

        except json.JSONDecodeError as e:
            _record_provider_log(
                chunk_id=log_key,
                provider=p["provider"],
                model=p["model"],
                status="json_error",
                retry_count=attempt,
                latency_seconds=0,
                generated_count=0,
                error=str(e)[:160],
            )
            _log(f"    [attempt {attempt+1}] JSON parse error [{label}]: {e}")
            _log(f"    Raw: {raw_text[:150]}")
            # JSON error không phải lỗi provider → retry cùng provider 1 lần
            time.sleep(2)

        except Exception as e:
            err = str(e)
            _record_provider_log(
                chunk_id=log_key,
                provider=p["provider"],
                model=p["model"],
                status="error",
                retry_count=attempt,
                latency_seconds=0,
                generated_count=0,
                error=err[:160],
            )
            _log(f"    [attempt {attempt+1}] {label}: {err[:100]}")

            is_rate_limit = any(k in err for k in ("429", "RESOURCE_EXHAUSTED", "rate_limit", "RateLimitError"))
            is_not_found  = any(k in err for k in ("404", "NOT_FOUND", "model_not_found", "does not exist"))
            is_no_key     = "not found in .env" in err

            if is_not_found or is_no_key:
                # Model không tồn tại / không có key → disable vĩnh viễn, không retry
                reason = "model not found" if is_not_found else "no API key"
                _mark_disabled(idx, reason)
            elif is_rate_limit:
                # Rate limit → cooldown tạm thời
                _mark_cooldown(idx)
            else:
                # Lỗi khác (network, timeout) → retry cùng provider
                time.sleep(5)
                continue

            # Chuyển sang provider tiếp theo
            next_idx = _next_available_provider()
            if next_idx is not None and next_idx != idx:
                _current_provider_idx = next_idx
                next_p = PROVIDERS[next_idx]
                _log(f"    Switch -> {next_p['provider']}/{next_p['model'].split('/')[-1]}")
            time.sleep(SWITCH_DELAY)

    _log(f"    SKIP {log_key} sau {attempts_total} attempts")
    _session_chunks_skipped.append(log_key)
    _record_provider_log(
        chunk_id=log_key,
        provider="",
        model="",
        status="skipped",
        retry_count=attempts_total,
        latency_seconds=0,
        generated_count=0,
    )
    return []


def _record_provider_log(
    chunk_id: str,
    provider: str,
    model: str,
    status: str,
    retry_count: int,
    latency_seconds: float,
    generated_count: int,
    error: str = "",
) -> None:
    _provider_chunk_logs.append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "chunk_id": chunk_id,
        "provider": provider,
        "model": model,
        "status": status,
        "retry_count": retry_count,
        "latency_seconds": latency_seconds,
        "generated_count": generated_count,
        "error": error,
    })


def _parse_response(raw: str, chunk_id: str) -> list[dict]:
    text = raw
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()

    data = json.loads(text)

    # PS4 escape hatch: {"skip": true, "reason": "..."}
    if isinstance(data, dict) and data.get("skip"):
        _log(f"    [skip] {chunk_id}: {data.get('reason', 'bloom level not supported')}")
        return []

    if isinstance(data, dict):
        # OpenAI json_object đôi khi wrap trong {"mcqs": [...]}
        for key in ("mcqs", "questions", "items", "data"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            data = [data]

    validated = []
    for item in data:
        item["source_chunk"] = chunk_id
        try:
            validated.append(MCQItem.model_validate(item).model_dump())
        except ValidationError as e:
            _log(f"    [pydantic] skip 1 MCQ — {e.error_count()} lỗi: "
                 f"{[err['loc'] for err in e.errors()]}")
    return validated


# ── Checkpoint helpers ─────────────────────────────────────────────

def _checkpoint_path(file_name: str) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    file_hash = _file_hash(file_name)
    if file_hash:
        return CHECKPOINT_DIR / f"{file_hash}_checkpoint.json"
    safe = re.sub(r"[^\w\-]", "_", Path(file_name).stem)
    return CHECKPOINT_DIR / f"checkpoint_{safe}.json"


def _file_hash(file_name: str) -> str:
    path = Path(file_name)
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()[:12]


def _checkpoint_meta(file_name: str) -> dict:
    path = Path(file_name)
    return {
        "file_name": path.name,
        "file_hash": _file_hash(file_name),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }

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
    meta = done.get("__meta__", {})
    meta.update(_checkpoint_meta(file_name))
    if "created_at" not in meta:
        meta["created_at"] = meta["updated_at"]
    done["__meta__"] = meta
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
        data = json.loads(path.read_text(encoding="utf-8"))
        # Keys: "{chunk_id}__{bloom_level}" for PS4 items
        work_data = {k: v for k, v in data.items()
                     if not k.startswith("__") and isinstance(v, list)}
        total_mcqs    = sum(len(v) for v in work_data.values())
        unique_chunks = len(set(k.split("__")[0] for k in work_data))
        return {
            "chunks_done": unique_chunks,
            "items_done" : len(work_data),
            "mcqs_so_far": total_mcqs,
            "file_hash"  : data.get("__meta__", {}).get("file_hash", ""),
            "path"       : str(path),
        }
    except Exception:
        return None


# ── Estimate time ──────────────────────────────────────────────────

def estimate_seconds(n_chunks: int, n_done: int = 0) -> int:
    remaining = n_chunks - n_done
    return int(remaining * REQUEST_DELAY)


# ── Main function ──────────────────────────────────────────────────

def generate_mcqs(
    chunks: list[dict],
    bloom_levels: list[str] | None = None,
    difficulty: str = "medium",
    pdf_name: str = "unknown",
    language: str = "en",
    on_progress=None,   # callback(idx, total, key, eta_seconds, skipped)
) -> list[dict]:
    """
    Sinh MCQ bằng PS4 approach: 1 API call per (chunk × bloom_level).

    Args:
        chunks      : output từ chunker.chunk_pages()
        bloom_levels: list Bloom levels cần test, vd ["understand","apply","analyze"]
                      Nếu None → dùng DENSITY_TO_LEVELS["Vừa"] (4 levels)
        difficulty  : "easy" | "medium" | "hard" — hint cho model, không ép buộc
        pdf_name    : tên file — dùng để đặt tên checkpoint
        on_progress : callback(idx, total, key, eta_sec, skipped)

    Returns:
        List[MCQ dict] chưa validate — theo thứ tự chunk rồi bloom level
    """
    if bloom_levels is None:
        bloom_levels = DIFFICULTY_TO_BLOOM.get(difficulty, DIFFICULTY_TO_BLOOM["medium"])

    done_map = _load_checkpoint(pdf_name)

    # Filter to conceptual chunks; log skipped
    conceptual_chunks = [c for c in chunks if c.get("chunk_type", "conceptual") == "conceptual"]
    skipped_types = len(chunks) - len(conceptual_chunks)
    if skipped_types:
        type_summary: dict[str, int] = {}
        for c in chunks:
            ct = c.get("chunk_type", "?")
            if ct != "conceptual":
                type_summary[ct] = type_summary.get(ct, 0) + 1
        _log(f"  [filter] skip {skipped_types} non-conceptual: {type_summary}")

    # Build work list: (chunk, bloom_level) pairs
    work_items: list[tuple[dict, str]] = [
        (chunk, level)
        for chunk in conceptual_chunks
        for level in bloom_levels
    ]
    total  = len(work_items)
    n_done = sum(
        1 for chunk, level in work_items
        if f"{chunk['chunk_id']}__{level}" in done_map
    )

    _log(f"  [PS4] {len(conceptual_chunks)} chunks × {len(bloom_levels)} levels "
         f"= {total} API calls  (resume: {n_done} done)")

    available = [p for p in PROVIDERS if os.getenv(
        {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY",
         "cerebras": "CEREBRAS_API_KEY", "openrouter": "OPENROUTER_API_KEY"}[p["provider"]], ""
    ).strip()]
    _log(f"  Active providers: {[p['provider']+'/'+p['model'].split('/')[-1] for p in available]}")

    # ── Generate ───────────────────────────────────────────────────
    for idx, (chunk, bloom_level) in enumerate(work_items):
        ck_key   = f"{chunk['chunk_id']}__{bloom_level}"
        chunk_id = chunk["chunk_id"]

        if ck_key in done_map:
            if on_progress:
                on_progress(idx, total, ck_key, estimate_seconds(total, idx + 1), skipped=True)
            continue

        if on_progress:
            on_progress(idx, total, ck_key, estimate_seconds(total, idx), skipped=False)

        _log(f"\n  [{idx+1}/{total}] {chunk_id}/{bloom_level} — {chunk['topic'][:40]}")

        # whole_doc: ask for multiple MCQs per level to cover the whole document
        if chunk.get("is_whole_doc"):
            n_pages    = len(chunk.get("pages", []))
            n_per_level = max(2, n_pages // (3 * len(bloom_levels)))
            n_per_level = min(n_per_level, 5)
            max_tokens  = min(4096, n_per_level * 600 + 500)
            _log(f"    whole_doc/{bloom_level} → {n_per_level} MCQs | max_tokens={max_tokens}")
        else:
            n_per_level = 1
            max_tokens  = 1536

        prompt = build_ps4_prompt(chunk, bloom_level, n_per_level, language)
        mcqs   = _generate_for_chunk(chunk, prompt, max_tokens=max_tokens, call_label=ck_key)

        done_map[ck_key] = mcqs
        _save_checkpoint(pdf_name, done_map)

        if idx < total - 1:
            time.sleep(REQUEST_DELAY)

    # ── Assemble in order: chunk order → bloom level order ─────────
    all_mcqs: list[dict] = []
    for chunk in conceptual_chunks:
        for level in bloom_levels:
            all_mcqs.extend(done_map.get(f"{chunk['chunk_id']}__{level}", []))

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
    bloom_levels = DENSITY_TO_LEVELS["Vừa"]
    n_calls      = len(chunks) * len(bloom_levels)
    print(f"  Chunks    : {len(chunks)}")
    print(f"  Difficulty: {difficulty}")
    print(f"  Levels    : {bloom_levels}")
    print(f"  API calls : {n_calls}  ETA: ~{estimate_seconds(n_calls)}s")
    print(f"{'='*55}\n")

    mcqs = generate_mcqs(chunks, bloom_levels=bloom_levels, difficulty=difficulty, pdf_name=file_name)

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
