"""
LLM transcript cleanup — separate from summarization.

Operates on speaker turns (not tiny ASR fragments). Rules:
  - Do not invent words or facts
  - Do not translate
  - Preserve code-switching
  - Merge/punctuate for readability only

Falls back to rule-based cleaned_text when Ollama is unavailable or the
model output fails verification (words must come from the raw turn).
"""

from __future__ import annotations

import logging
import re
import urllib.error
import urllib.request

from models.speaker_turns import rule_based_cleanup

logger = logging.getLogger("transcript_cleanup")

try:
    from config import (
        OLLAMA_KEEP_ALIVE,
        OLLAMA_URL,
        SUMMARY_MODEL,
        SUMMARY_NUM_CTX,
        TRANSCRIPT_CLEANUP_ENABLED,
        TRANSCRIPT_CLEANUP_TIMEOUT_SECONDS,
    )
except Exception:
    TRANSCRIPT_CLEANUP_ENABLED = True
    TRANSCRIPT_CLEANUP_TIMEOUT_SECONDS = 120
    OLLAMA_URL = "http://localhost:11434"
    SUMMARY_MODEL = "qwen2.5:7b"
    SUMMARY_NUM_CTX = 8192
    OLLAMA_KEEP_ALIVE = "0"

_CLEANUP_PROMPT = """You are a multilingual transcript cleanup engine.
Supported languages: English, Hindi, Marathi.

Clean the ASR transcript while preserving the exact meaning.

Rules:
- Do not invent information.
- Do not hallucinate words.
- Do not add facts.
- Do not translate.
- Preserve the original language and natural code-switching.
- Merge fragmented speech when appropriate.
- Restore punctuation.
- Improve readability.
- Correct only obvious ASR errors when strongly supported by context.
- Preserve names and technical terms unless clearly incorrect.
- If uncertain, keep the original wording.
- Return ONLY the cleaned speech text, no labels or commentary.

RAW TRANSCRIPT:
"""


def _token_set(text: str) -> set[str]:
    """Word tokens for subset verification (Latin + Devanagari runs)."""
    return set(re.findall(r"[\u0900-\u097f]+|[A-Za-z0-9']+", (text or "").lower()))


def _output_is_faithful(raw: str, cleaned: str) -> bool:
    """
    Reject LLM output that introduces many tokens absent from raw ASR.
    Allows punctuation-only changes and minor reordering.
    """
    raw_tokens = _token_set(raw)
    clean_tokens = _token_set(cleaned)
    if not clean_tokens:
        return False
    if not raw_tokens:
        return bool(cleaned.strip())
    novel = clean_tokens - raw_tokens
    # Allow a few short function-word fixes; block large invention.
    novel = {t for t in novel if len(t) > 2}
    ratio = len(novel) / max(len(clean_tokens), 1)
    return ratio <= 0.08


def _call_ollama(prompt: str) -> str | None:
    payload = {
        "model": SUMMARY_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": SUMMARY_NUM_CTX, "temperature": 0.1},
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    import json

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TRANSCRIPT_CLEANUP_TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read().decode())
            return (body.get("response") or "").strip()
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.warning(f"transcript cleanup LLM unavailable: {exc}")
        return None


def cleanup_turn_text(raw_text: str, *, use_llm: bool | None = None) -> tuple[str, str]:
    """
    Return (cleaned_text, engine) where engine is 'llm', 'rules', or 'rules_fallback'.
    """
    if not raw_text or not raw_text.strip():
        return raw_text or "", "rules"

    llm_on = TRANSCRIPT_CLEANUP_ENABLED if use_llm is None else use_llm
    if llm_on:
        prompt = _CLEANUP_PROMPT + raw_text.strip() + "\n\nCLEANED TRANSCRIPT:\n"
        llm_out = _call_ollama(prompt)
        if llm_out and _output_is_faithful(raw_text, llm_out):
            return llm_out.strip(), "llm"
        if llm_out:
            logger.info("transcript cleanup LLM rejected — output not faithful to raw ASR")

    return rule_based_cleanup(raw_text), "rules"


def cleanup_turns(
    turns: list[dict],
    *,
    use_llm: bool | None = None,
    batch_size: int = 5,
) -> list[dict]:
    """
    Apply cleanup to speaker turns in small batches (LLM) or rule-based fallback.
    Preserves raw_text; updates cleaned_text and text.
    """
    if not turns:
        return turns

    llm_on = TRANSCRIPT_CLEANUP_ENABLED if use_llm is None else use_llm
    updated: list[dict] = []

    if llm_on and len(turns) > batch_size:
        # Batch adjacent turns for fewer LLM calls on long meetings.
        for i in range(0, len(turns), batch_size):
            chunk = turns[i : i + batch_size]
            combined_raw = "\n".join(t.get("raw_text") or t.get("text") or "" for t in chunk)
            cleaned, engine = cleanup_turn_text(combined_raw, use_llm=True)
            if engine == "llm" and len(chunk) > 1:
                # Split batch output proportionally by raw length (conservative).
                parts = _split_batch_output(combined_raw, cleaned, chunk)
                for turn, part in zip(chunk, parts):
                    row = dict(turn)
                    row["cleaned_text"] = part
                    row["text"] = part
                    row["cleanup_engine"] = engine
                    updated.append(row)
            else:
                for turn in chunk:
                    row = dict(turn)
                    c, eng = cleanup_turn_text(
                        row.get("raw_text") or row.get("text") or "",
                        use_llm=llm_on,
                    )
                    row["cleaned_text"] = c
                    row["text"] = c
                    row["cleanup_engine"] = eng
                    updated.append(row)
    else:
        for turn in turns:
            row = dict(turn)
            raw = row.get("raw_text") or row.get("text") or ""
            cleaned, engine = cleanup_turn_text(raw, use_llm=llm_on)
            row["cleaned_text"] = cleaned
            row["text"] = cleaned
            row["cleanup_engine"] = engine
            updated.append(row)

    logger.info(
        f"transcript cleanup: {len(updated)} turn(s), "
        f"engines={ {t.get('cleanup_engine') for t in updated} }"
    )
    return updated


def _split_batch_output(combined_raw: str, cleaned: str, chunk: list[dict]) -> list[str]:
    """When batch LLM returns one block, assign slices by raw token counts."""
    if len(chunk) == 1:
        return [cleaned]
    raw_lens = [
        len(re.findall(r"[\u0900-\u097f]+|[A-Za-z0-9']+", t.get("raw_text") or t.get("text") or ""))
        for t in chunk
    ]
    total = sum(raw_lens) or len(chunk)
    words = cleaned.split()
    if not words:
        return [rule_based_cleanup(t.get("raw_text") or t.get("text") or "") for t in chunk]
    out: list[str] = []
    pos = 0
    for i, rl in enumerate(raw_lens):
        if i == len(chunk) - 1:
            part = " ".join(words[pos:])
        else:
            take = max(1, round(len(words) * rl / total))
            part = " ".join(words[pos : pos + take])
            pos += take
        out.append(part.strip() or rule_based_cleanup(chunk[i].get("raw_text") or ""))
    return out
