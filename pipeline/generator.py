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
from pathlib import Path
from dotenv import load_dotenv

# Đảm bảo project root trong sys.path khi chạy trực tiếp
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompts.mcq_prompt import (
    build_prompt, build_summary_prompt,
    build_hint_prompt, build_summary_chunk_prompt,
)
from pipeline.schemas import MCQItem
from pydantic import ValidationError

load_dotenv()

# ── Provider config ────────────────────────────────────────────────
# tpm  : token/phút — dùng để tính REQUEST_DELAY
# Thứ tự = thứ tự ưu tiên (A → B → C khi bị 429)

PROVIDERS = [
    # Primary — Gemini (chất lượng tốt nhất, JSON mode native)
    {"provider": "gemini",     "model": "gemini-2.5-flash",              "tpm": 1_000_000},
    {"provider": "gemini",     "model": "gemini-2.0-flash",              "tpm": 1_000_000},
    {"provider": "gemini",     "model": "gemini-2.0-flash-lite",         "tpm": 1_000_000},

    # Fallback 1 — Cerebras (nhanh, 1M token/ngày, TPM cao)
    {"provider": "cerebras",   "model": "llama3.1-70b",                  "tpm": 60_000},

    # Fallback 2 — Groq (free nhưng TPM thấp)
    {"provider": "groq",       "model": "llama-3.3-70b-versatile",       "tpm": 6_000},
    {"provider": "groq",       "model": "meta-llama/llama-4-scout-17b-16e-instruct", "tpm": 6_000},

    # Fallback 3 — OpenRouter (last resort)
    {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free", "tpm": 20_000},
]

RETRY_LIMIT    = 2      # số lần retry trên cùng 1 provider trước khi chuyển
SWITCH_DELAY   = 2.0    # giây chờ khi switch provider
REQUEST_DELAY  = 4.5    # giây giữa các request (safe với Gemini 15 RPM)

CHECKPOINT_DIR  = Path("data/checkpoints")
SUMMARY_CHUNK_ID = "__summary__"   # chunk_id dành riêng cho knowledge map


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

def _call_provider(prompt: str, idx: int, json_mode: bool = True) -> str:
    """
    Gọi 1 provider và trả về raw text response.
    Raise exception nếu lỗi để caller xử lý.

    Args:
        json_mode: True  → ép response format JSON (dùng cho MCQ)
                   False → plain text (dùng cho slide summary)
    """
    p = PROVIDERS[idx]
    provider = p["provider"]
    model    = p["model"]

    t0 = time.time()

    if provider == "gemini":
        from google.genai import types
        client     = _get_gemini_client()
        cfg_kwargs = {"temperature": 0.7, "max_output_tokens": 2048}
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
            "temperature": 0.7,
            "max_tokens" : 2048,
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
    }


def reset_provider_stats():
    _provider_call_counts.clear()
    _provider_tokens.clear()
    _provider_response_times.clear()
    _session_chunks_skipped.clear()


# ── Slide summary (knowledge map) ─────────────────────────────────

def build_slide_summary(chunks: list[dict]) -> str:
    """
    Đọc tất cả chunks → sinh knowledge map dạng markdown.
    Dùng provider fallback system giống MCQ generation.
    Trả về "" nếu thất bại (graceful degradation).
    """
    global _current_provider_idx

    prompt         = build_summary_prompt(chunks)
    attempts_total = len(PROVIDERS) * RETRY_LIMIT

    for attempt in range(attempts_total):
        idx = _next_available_provider()

        if idx is None:
            active = [i for i in range(len(PROVIDERS)) if i not in _provider_disabled]
            if not active:
                break
            wait = min(_provider_cooldown.get(i, 0) for i in active) - time.time()
            time.sleep(max(wait, 1.0))
            idx = _next_available_provider()
            if idx is None:
                break

        _current_provider_idx = idx
        p     = PROVIDERS[idx]
        label = f"{p['provider']}/{p['model'].split('/')[-1]}"

        try:
            raw = _call_provider(prompt, idx, json_mode=False)
            _log(f"  [slide_summary] OK [{label}] ({len(raw)} chars)")
            return raw.strip()
        except Exception as e:
            err = str(e)
            _log(f"  [slide_summary] attempt {attempt+1} [{label}]: {err[:80]}")
            is_rate_limit = any(k in err for k in ("429", "RESOURCE_EXHAUSTED", "rate_limit", "RateLimitError"))
            is_not_found  = any(k in err for k in ("404", "NOT_FOUND", "model_not_found", "does not exist"))
            is_no_key     = "not found in .env" in err
            if is_not_found or is_no_key:
                _mark_disabled(idx, "model not found" if is_not_found else "no API key")
            elif is_rate_limit:
                _mark_cooldown(idx)
            else:
                time.sleep(5)
                continue
            next_idx = _next_available_provider()
            if next_idx is not None and next_idx != idx:
                _current_provider_idx = next_idx
            time.sleep(SWITCH_DELAY)

    _log("  [slide_summary] Failed — continuing without cross-chunk context")
    return ""


def build_chunk_hints(chunks: list[dict], slide_summary: str) -> dict[str, list[str]]:
    """
    1 API call → per-chunk connection hints dạng {chunk_id: [hint, ...]}.
    Dùng provider fallback system. Trả về {} nếu thất bại.
    """
    global _current_provider_idx

    prompt         = build_hint_prompt(chunks, slide_summary)
    attempts_total = len(PROVIDERS) * RETRY_LIMIT

    for attempt in range(attempts_total):
        idx = _next_available_provider()

        if idx is None:
            active = [i for i in range(len(PROVIDERS)) if i not in _provider_disabled]
            if not active:
                break
            wait = min(_provider_cooldown.get(i, 0) for i in active) - time.time()
            time.sleep(max(wait, 1.0))
            idx = _next_available_provider()
            if idx is None:
                break

        _current_provider_idx = idx
        p     = PROVIDERS[idx]
        label = f"{p['provider']}/{p['model'].split('/')[-1]}"

        try:
            raw  = _call_provider(prompt, idx, json_mode=True)
            data = json.loads(raw)
            # Unwrap {"hints": {...}} hoặc dùng thẳng nếu là flat dict
            hints = data.get("hints", data) if isinstance(data, dict) else {}
            # Validate: mỗi value phải là list[str]
            hints = {
                k: [str(h) for h in v] if isinstance(v, list) else []
                for k, v in hints.items()
            }
            total_hints = sum(len(v) for v in hints.values())
            _log(f"  [chunk_hints] OK [{label}] — {total_hints} hints across {len(hints)} chunks")
            return hints
        except Exception as e:
            err = str(e)
            _log(f"  [chunk_hints] attempt {attempt+1} [{label}]: {err[:80]}")
            is_rate_limit = any(k in err for k in ("429", "RESOURCE_EXHAUSTED", "rate_limit", "RateLimitError"))
            is_not_found  = any(k in err for k in ("404", "NOT_FOUND", "model_not_found", "does not exist"))
            is_no_key     = "not found in .env" in err
            if is_not_found or is_no_key:
                _mark_disabled(idx, "model not found" if is_not_found else "no API key")
            elif is_rate_limit:
                _mark_cooldown(idx)
            else:
                time.sleep(5)
                continue
            next_idx = _next_available_provider()
            if next_idx is not None and next_idx != idx:
                _current_provider_idx = next_idx
            time.sleep(SWITCH_DELAY)

    _log("  [chunk_hints] Failed — continuing without connection hints")
    return {}


def _make_summary_chunk(slide_summary: str) -> dict:
    """Tạo synthetic chunk từ knowledge map để generate MCQ riêng."""
    return {
        "chunk_id" : SUMMARY_CHUNK_ID,
        "topic"    : "Lecture Overview & Knowledge Connections",
        "pages"    : "all",
        "text"     : slide_summary,
        "has_image": False,
    }


# ── Per-chunk generation ───────────────────────────────────────────

def _generate_for_chunk(
    chunk: dict,
    n_questions: int,
    difficulty: str,
    slide_context: str = "",
    chunk_hints: list[str] = None,
    summary_chunk_prompt: str = "",
    language: str = "en",
) -> list[dict]:
    """
    Gọi API cho 1 chunk.
    Nếu provider hiện tại bị 429 → switch sang provider tiếp theo.

    Args:
        summary_chunk_prompt: nếu có, dùng thẳng prompt này thay vì build_prompt()
                              (dành riêng cho summary chunk)
    """
    global _current_provider_idx

    prompt = summary_chunk_prompt or build_prompt(
        chunk,
        n_questions=n_questions,
        difficulty=difficulty,
        slide_context=slide_context,
        chunk_hints=chunk_hints or [],
        language=language,
    )
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

    _log(f"    SKIP {chunk['chunk_id']} sau {attempts_total} attempts")
    _session_chunks_skipped.append(chunk["chunk_id"])
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
        data        = json.loads(path.read_text(encoding="utf-8"))
        chunk_data  = {k: v for k, v in data.items()
                       if not k.startswith("__") and isinstance(v, list)}
        total       = sum(len(v) for v in chunk_data.values())
        has_summary = "__slide_summary__" in data
        return {
            "chunks_done" : len(chunk_data),
            "mcqs_so_far" : total,
            "has_summary" : has_summary,
            "path"        : str(path),
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
    n_per_chunk: int = 2,
    difficulty: str = "medium",
    pdf_name: str = "unknown",
    language: str = "en",
    on_progress=None,   # callback(idx, total, chunk_id, eta_seconds, skipped)
) -> list[dict]:
    """
    Sinh MCQ từ list chunks với multi-provider fallback + checkpoint.

    Flow:
      1. Build slide knowledge map (1 API call) → slide_summary
      2. Tạo summary_chunk từ knowledge map → sinh MCQ tổng hợp riêng
      3. Sinh MCQ từng chunk, inject slide_summary làm context

    Args:
        chunks      : output từ chunker.chunk_pages()
        n_per_chunk : số MCQ mỗi chunk
        difficulty  : "easy" | "medium" | "hard"
        pdf_name    : tên file — dùng để đặt tên checkpoint
        on_progress : callback(idx, total, chunk_id, eta_sec, skipped)

    Returns:
        List[MCQ dict] — summary MCQs trước, rồi MCQ từng chunk, chưa validate
    """
    done_map = _load_checkpoint(pdf_name)

    # ── Step 1: build or load slide summary ───────────────────────
    if "__slide_summary__" in done_map:
        slide_summary = done_map["__slide_summary__"]
        _log(f"  [slide_summary] Loaded from checkpoint ({len(slide_summary)} chars)")
    else:
        _log(f"  [slide_summary] Building knowledge map ({len(chunks)} chunks)...")
        slide_summary = build_slide_summary(chunks)
        if slide_summary:
            done_map["__slide_summary__"] = slide_summary
            _save_checkpoint(pdf_name, done_map)

    # ── Step 2: build or load per-chunk hints ─────────────────────
    if "__chunk_hints__" in done_map:
        hints_map = done_map["__chunk_hints__"]
        _log(f"  [chunk_hints] Loaded from checkpoint")
    elif slide_summary:
        _log(f"  [chunk_hints] Building connection hints...")
        hints_map = build_chunk_hints(chunks, slide_summary)
        if hints_map:
            done_map["__chunk_hints__"] = hints_map
            _save_checkpoint(pdf_name, done_map)
    else:
        hints_map = {}

    # ── Step 3: build processing list ─────────────────────────────
    # Chỉ sinh MCQ từ conceptual chunks; bỏ stub/structural/instructional
    conceptual_chunks = [c for c in chunks if c.get("chunk_type", "conceptual") == "conceptual"]
    skipped_types = len(chunks) - len(conceptual_chunks)
    if skipped_types:
        type_summary = {}
        for c in chunks:
            ct = c.get("chunk_type", "?")
            if ct != "conceptual":
                type_summary[ct] = type_summary.get(ct, 0) + 1
        _log(f"  [filter] skip {skipped_types} non-conceptual chunks: {type_summary}")

    # summary_chunk đứng đầu (nếu có summary), rồi mới đến conceptual chunks
    processing_chunks = list(conceptual_chunks)
    if slide_summary:
        processing_chunks = [_make_summary_chunk(slide_summary)] + processing_chunks

    total  = len(processing_chunks)
    # n_done đếm conceptual chunks đã có trong checkpoint
    batch_ids = {c["chunk_id"] for c in processing_chunks}
    n_done    = sum(1 for k in done_map if k in batch_ids)

    if n_done > 0:
        _log(f"  Resume: {n_done}/{len(chunks)} content chunks da co trong checkpoint")

    # In provider list đang active
    available = [p for p in PROVIDERS if os.getenv(
        {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY",
         "cerebras": "CEREBRAS_API_KEY", "openrouter": "OPENROUTER_API_KEY"}[p["provider"]], ""
    ).strip()]
    labels = [p["provider"] + "/" + p["model"].split("/")[-1] for p in available]
    _log(f"  Active providers: {labels}")

    # ── Step 4: generate MCQ từng chunk ───────────────────────────
    for idx, chunk in enumerate(processing_chunks):
        chunk_id = chunk["chunk_id"]

        if chunk_id in done_map:
            if on_progress:
                on_progress(idx, total, chunk_id, estimate_seconds(total, idx+1), skipped=True)
            continue

        if on_progress:
            on_progress(idx, total, chunk_id, estimate_seconds(total, idx), skipped=False)

        _log(f"\n  [{idx+1}/{total}] {chunk_id} — {chunk['topic'][:45]}")

        if chunk_id == SUMMARY_CHUNK_ID:
            mcqs = _generate_for_chunk(
                chunk, n_per_chunk, difficulty,
                summary_chunk_prompt=build_summary_chunk_prompt(
                    slide_summary, hints_map, n_per_chunk, difficulty, language
                ),
                language=language,
            )
        else:
            mcqs = _generate_for_chunk(
                chunk, n_per_chunk, difficulty,
                slide_context=slide_summary,
                chunk_hints=hints_map.get(chunk_id, []),
                language=language,
            )

        done_map[chunk_id] = mcqs
        _save_checkpoint(pdf_name, done_map)

        if idx < total - 1:
            time.sleep(REQUEST_DELAY)

    # ── Assemble: summary MCQs trước, rồi conceptual chunks ───────
    all_mcqs = []
    if SUMMARY_CHUNK_ID in done_map:
        all_mcqs.extend(done_map[SUMMARY_CHUNK_ID])
    for chunk in conceptual_chunks:
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
