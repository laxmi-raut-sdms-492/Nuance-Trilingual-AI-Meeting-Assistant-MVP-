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


_ROMAN_INDIC_MAP = {
    r"\baahe\b": "आहे",
    r"\bahet\b": "आहेत",
    r"\bnahi\b": "नाही",
    r"\bpan\b": "पण",
    r"\bhaye\b": "हाय",
    r"\bshikshak\b": "शिक्षक",
}


def rule_based_cleanup(raw_text: str) -> str:
    """
    Light formatting only — no word invention. Adds spacing and terminal
    punctuation when clearly missing; preserves Devanagari and Latin mix.
    Converts common Romanized Indic words in Devanagari sentences into Devanagari.
    """
    text = _normalize_whitespace(raw_text)
    if not text:
        return text

    # Harmonize Romanized Indic words when Devanagari script is present in the line
    has_devanagari = any("\u0900" <= c <= "\u097f" for c in text)
    if has_devanagari:
        for pattern, dev_repl in _ROMAN_INDIC_MAP.items():
            text = re.sub(pattern, dev_repl, text, flags=re.IGNORECASE)

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


def _seg_seconds(seg: dict) -> float:
    return max(float(seg.get("end_sec") or 0) - float(seg.get("start_sec") or 0), 0.0)


def _record_language(turn: dict, seg: dict) -> None:
    """
    Accumulate talk time per language inside this turn.

    Kept per language rather than as a running "current language" so the turn
    can be labelled by how much of it was actually spoken in each — see
    _apply_dominant_language.
    """
    lang = seg.get("language")
    if not lang:
        return

    stats = turn.setdefault("_lang_stats", {})
    entry = stats.get(lang)
    if entry is None:
        entry = {
            "seconds": 0.0,
            "segments": 0,
            "name": seg.get("language_name"),
            "prob": 0.0,
            "detected": seg.get("language_detected"),
            "fallback": False,
        }
        stats[lang] = entry
        # First-seen order, which is the order language_mix is reported in.
        turn.setdefault("languages", []).append(lang)

    entry["seconds"] += _seg_seconds(seg)
    entry["segments"] += 1
    entry["prob"] = max(entry["prob"], float(seg.get("language_prob") or 0.0))
    entry["fallback"] = entry["fallback"] or bool(seg.get("language_fallback", False))
    if not entry["name"]:
        entry["name"] = seg.get("language_name")


def _apply_dominant_language(turn: dict) -> None:
    """
    Label the turn with the language most of it was spoken in.

    Previously the last segment merged into a turn simply overwrote
    turn["language"], so a turn that ran 20s English then 5s Marathi was
    labelled Marathi — and language_breakdown, which reads that field,
    charged all 25s to Marathi. Ranking by seconds fixes both.

    Nothing is discarded: the minority language is still in raw_text and
    still listed in language_mix.
    """
    stats = turn.pop("_lang_stats", {}) or {}
    if not stats:
        return

    order = turn.get("languages") or list(stats)

    # Seconds decide it. Segment count and first-seen order only break ties,
    # which happens when segments carry no usable duration.
    def rank(code: str):
        return (
            stats[code]["seconds"],
            stats[code]["segments"],
            -order.index(code) if code in order else 0,
        )

    winner = max(stats, key=rank)
    best = stats[winner]

    turn["language"] = winner
    turn["language_name"] = best["name"] or turn.get("language_name")
    turn["language_prob"] = round(best["prob"], 3)
    turn["language_detected"] = best["detected"]
    turn["language_fallback"] = best["fallback"]


def _start_turn(seg: dict) -> dict[str, Any]:
    raw = seg.get("raw_text") or seg.get("text") or ""
    turn = {
        "start_sec": float(seg.get("start_sec") or 0),
        "end_sec": float(seg.get("end_sec") or 0),
        "time": seg.get("time"),
        "speaker": seg.get("speaker"),
        "speaker_label": seg.get("speaker_label"),
        "identified_as": seg.get("identified_as"),
        "confidence": seg.get("confidence", 0.0),
        "color": seg.get("color"),
        "is_overlap": bool(seg.get("is_overlap", False)),
        "candidate_speakers": list(seg.get("candidate_speakers") or []),
        "candidate_labels": list(seg.get("candidate_labels") or []),
        "language": seg.get("language"),
        "language_name": seg.get("language_name"),
        "language_prob": seg.get("language_prob", 0.0),
        "language_detected": seg.get("language_detected"),
        "language_fallback": seg.get("language_fallback", False),
        "language_margin": float(seg.get("language_margin", 1.0) or 0.0),
        "language_mixed_suspected": bool(seg.get("language_mixed_suspected", False)),
        "languages": [],
        "_lang_stats": {},
        "raw_parts": [raw] if raw else [],
        "segment_count": 1,
        "is_turn": True,
    }
    _record_language(turn, seg)
    return turn


def _extend_turn(turn: dict, seg: dict) -> None:
    raw = seg.get("raw_text") or seg.get("text") or ""
    if raw:
        turn["raw_parts"].append(raw)
    turn["end_sec"] = float(seg.get("end_sec") or turn["end_sec"])
    turn["segment_count"] = turn.get("segment_count", 1) + 1
    if seg.get("is_overlap"):
        turn["is_overlap"] = True
        for cs in seg.get("candidate_speakers") or []:
            if cs not in turn["candidate_speakers"]:
                turn["candidate_speakers"].append(cs)
        for cl in seg.get("candidate_labels") or []:
            if cl not in turn["candidate_labels"]:
                turn["candidate_labels"].append(cl)
        if len(turn["candidate_speakers"]) > 1:
            turn["speaker"] = " + ".join(turn["candidate_speakers"])
            turn["speaker_label"] = " + ".join(turn["candidate_labels"])
    turn["language_margin"] = min(
        float(turn.get("language_margin", 1.0) or 0.0),
        float(seg.get("language_margin", 1.0) or 0.0),
    )
    turn["language_mixed_suspected"] = bool(
        turn.get("language_mixed_suspected", False)
    ) or bool(seg.get("language_mixed_suspected", False))
    _record_language(turn, seg)


def _deduplicate_phrase_parts(parts: list[str]) -> list[str]:
    """Remove exact and overlapping duplicate phrases between consecutive segments."""
    if not parts:
        return []

    clean_parts: list[str] = []
    for part in parts:
        p = _normalize_whitespace(part)
        if not p:
            continue
        if not clean_parts:
            clean_parts.append(p)
            continue

        prev = clean_parts[-1]
        if p == prev or p in prev:
            continue
        if prev in p:
            clean_parts[-1] = p
            continue

        # Check word-level overlap (3+ words)
        prev_words = prev.split()
        curr_words = p.split()
        max_k = min(len(prev_words), len(curr_words))
        overlap_k = 0
        for k in range(max_k, 2, -1):
            if [w.lower() for w in prev_words[-k:]] == [w.lower() for w in curr_words[:k]]:
                overlap_k = k
                break

        if overlap_k > 0:
            p = " ".join(curr_words[overlap_k:]).strip()

        if p:
            clean_parts.append(p)

    return clean_parts


def _finalize_turn(turn: dict) -> dict:
    parts = _deduplicate_phrase_parts(turn.pop("raw_parts", []))
    raw_text = _normalize_whitespace(" ".join(parts))
    cleaned = rule_based_cleanup(raw_text)
    turn["raw_text"] = raw_text
    turn["cleaned_text"] = cleaned
    turn["text"] = cleaned

    if turn.get("is_overlap") and turn.get("candidate_speakers") and len(turn["candidate_speakers"]) > 1:
        turn["speaker"] = " + ".join(turn["candidate_speakers"])
        if turn.get("candidate_labels") and len(turn["candidate_labels"]) > 1:
            turn["speaker_label"] = " + ".join(turn["candidate_labels"])

    _apply_dominant_language(turn)
    langs = turn.pop("languages", [])

    dev_count = sum(1 for ch in raw_text if "\u0900" <= ch <= "\u097f")
    lat_count = sum(1 for ch in raw_text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))

    if dev_count > 0 and lat_count > 0:
        indic_code = turn.get("language") if turn.get("language") in ("hi", "mr") else "mr"
        if "en" not in langs:
            langs.append("en")
        if indic_code not in langs:
            langs.append(indic_code)

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
