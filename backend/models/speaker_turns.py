"""
Speaker-turn construction — merge fragmented ASR segments into readable turns.

Whisper emits short segments; the UI should show coherent speaker turns.
Language change alone never splits a turn — speaker identity is independent
of language (trilingual code-switching within one person is normal).

Each turn preserves:
  raw_text     — verbatim concatenation of ASR fragments (never invented)
  cleaned_text — rule-based readability pass (punctuation, spacing)
  text         — alias of cleaned_text for backward-compatible consumers
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("speaker_turns")

try:
    from config import TURN_MERGE_MAX_GAP_SEC, TURN_MAX_DURATION_SEC
except Exception:
    TURN_MERGE_MAX_GAP_SEC = 2.5
    TURN_MAX_DURATION_SEC = 120.0


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def rule_based_cleanup(raw_text: str) -> str:
    """
    Light formatting only — no word invention. Adds spacing and terminal
    punctuation when clearly missing; preserves Devanagari and Latin mix.
    """
    text = _normalize_whitespace(raw_text)
    if not text:
        return text
    # Avoid stacking punctuation.
    if text[-1] not in ".!?…।":
        # Devanagari-heavy lines often end with danda in speech but ASR omits it.
        devanagari_chars = sum(1 for c in text if "\u0900" <= c <= "\u097f")
        if devanagari_chars > len(text) * 0.3:
            text = text + "।"
        elif re.search(r"[A-Za-z]", text):
            text = text + "."
    return text


def _segment_key(seg: dict) -> str:
    return str(seg.get("speaker_label") or seg.get("speaker") or "")


def _start_turn(seg: dict) -> dict[str, Any]:
    raw = seg.get("raw_text") or seg.get("text") or ""
    return {
        "start_sec": float(seg.get("start_sec") or 0),
        "end_sec": float(seg.get("end_sec") or 0),
        "time": seg.get("time"),
        "speaker": seg.get("speaker"),
        "speaker_label": seg.get("speaker_label"),
        "identified_as": seg.get("identified_as"),
        "confidence": seg.get("confidence", 0.0),
        "color": seg.get("color"),
        "language": seg.get("language"),
        "language_name": seg.get("language_name"),
        "language_prob": seg.get("language_prob", 0.0),
        "language_detected": seg.get("language_detected"),
        "language_fallback": seg.get("language_fallback", False),
        "languages": [seg.get("language")] if seg.get("language") else [],
        "raw_parts": [raw] if raw else [],
        "segment_count": 1,
        "is_turn": True,
    }


def _extend_turn(turn: dict, seg: dict) -> None:
    raw = seg.get("raw_text") or seg.get("text") or ""
    if raw:
        turn["raw_parts"].append(raw)
    turn["end_sec"] = float(seg.get("end_sec") or turn["end_sec"])
    turn["segment_count"] = turn.get("segment_count", 1) + 1
    lang = seg.get("language")
    if lang and lang not in turn.get("languages", []):
        turn.setdefault("languages", []).append(lang)
    # Dominant language by segment count in turn (simple majority proxy).
    turn["language"] = seg.get("language") or turn.get("language")
    turn["language_name"] = seg.get("language_name") or turn.get("language_name")
    turn["language_prob"] = max(
        float(turn.get("language_prob") or 0),
        float(seg.get("language_prob") or 0),
    )


def _finalize_turn(turn: dict) -> dict:
    raw_text = _normalize_whitespace(" ".join(turn.pop("raw_parts", [])))
    cleaned = rule_based_cleanup(raw_text)
    turn["raw_text"] = raw_text
    turn["cleaned_text"] = cleaned
    turn["text"] = cleaned
    langs = turn.pop("languages", [])
    if len(langs) > 1:
        turn["language_mix"] = langs
    return turn


def should_merge_turn(
    current: dict,
    seg: dict,
    *,
    max_gap_sec: float = TURN_MERGE_MAX_GAP_SEC,
    max_turn_sec: float = TURN_MAX_DURATION_SEC,
) -> bool:
    """True when seg continues the same speaker turn (language change OK)."""
    if _segment_key(current) != _segment_key(seg):
        return False
    gap = float(seg.get("start_sec") or 0) - float(current.get("end_sec") or 0)
    if gap > max_gap_sec:
        return False
    turn_duration = float(seg.get("end_sec") or 0) - float(current.get("start_sec") or 0)
    if turn_duration > max_turn_sec:
        return False
    return True


def build_speaker_turns(
    segments: list[dict],
    *,
    max_gap_sec: float | None = None,
    max_turn_sec: float | None = None,
) -> list[dict]:
    """
    Merge consecutive same-speaker segments into speaker turns.

    Input segments should already be sorted by start_sec. Each segment may
    carry raw_text (preferred) or text. Returns a new list of turn dicts.
    """
    if not segments:
        return []

    gap = max_gap_sec if max_gap_sec is not None else TURN_MERGE_MAX_GAP_SEC
    max_dur = max_turn_sec if max_turn_sec is not None else TURN_MAX_DURATION_SEC

    ordered = sorted(segments, key=lambda s: float(s.get("start_sec") or 0))
    turns: list[dict] = []
    current: dict | None = None

    for seg in ordered:
        if current is None:
            current = _start_turn(seg)
            continue
        if should_merge_turn(current, seg, max_gap_sec=gap, max_turn_sec=max_dur):
            _extend_turn(current, seg)
        else:
            turns.append(_finalize_turn(current))
            current = _start_turn(seg)

    if current is not None:
        turns.append(_finalize_turn(current))

    logger.info(
        f"speaker turns: {len(ordered)} segment(s) -> {len(turns)} turn(s)"
    )
    return turns
