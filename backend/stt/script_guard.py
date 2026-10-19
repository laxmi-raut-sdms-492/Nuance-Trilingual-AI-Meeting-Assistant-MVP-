"""
Keep transcripts to the scripts this product supports: Latin (English) and
Devanagari (Hindi/Marathi).

A cloud engine that mishears a clip can answer in Urdu, Bengali, Tamil, Arabic,
Cyrillic or CJK. Relabelling the language is not enough — the garbled text is
still in the transcript — so the text itself is checked by script.
"""

from __future__ import annotations

import re
import unicodedata

_KEEP_PREFIXES = ("LATIN", "DEVANAGARI")
# Zero-width joiners are part of correct Devanagari conjuncts.
_KEEP_CHARS = {"‌", "‍"}
# More than this share of letters in other scripts means the whole clip is
# garbage, not a few stray characters worth cleaning.
GARBLED_LETTER_RATIO = 0.4


def _script_of(ch: str) -> str:
    try:
        return unicodedata.name(ch).split(" ", 1)[0]
    except ValueError:
        return "UNKNOWN"


# Marks and digits that belong to no particular script ("COMBINING ACUTE
# ACCENT", "DIGIT ZERO") are not foreign.
_NEUTRAL_PREFIXES = _KEEP_PREFIXES + ("COMBINING", "DIGIT", "VEDIC")


def _is_script_char(ch: str) -> bool:
    """Letters, plus the vowel signs and digits a script writes with them."""
    category = unicodedata.category(ch)
    return category[0] in "LM" or category == "Nd"


def _is_foreign_letter(ch: str) -> bool:
    # Vowel signs count too: Gujarati "ૂ" is not isalpha() and used to
    # survive stripping, leaving a stray foreign mark in a Marathi line.
    if ch in _KEEP_CHARS or not _is_script_char(ch):
        return False
    keep = _KEEP_PREFIXES if ch.isalpha() else _NEUTRAL_PREFIXES
    return _script_of(ch) not in keep


def foreign_letter_ratio(text: str) -> float:
    """Share of letters that are neither Latin nor Devanagari (0.0 - 1.0)."""
    letters = [c for c in (text or "") if _is_script_char(c)]
    if not letters:
        return 0.0
    return sum(_is_foreign_letter(c) for c in letters) / len(letters)


def has_foreign_script(text: str) -> bool:
    return any(_is_foreign_letter(c) for c in (text or ""))


def is_garbled(text: str) -> bool:
    return foreign_letter_ratio(text) > GARBLED_LETTER_RATIO


def strip_foreign_script(text: str) -> str:
    """Remove letters outside Latin/Devanagari and tidy the leftover spacing."""
    cleaned = "".join(c for c in (text or "") if not _is_foreign_letter(c))
    return re.sub(r"\s+", " ", cleaned).strip()


def clean_text(text: str | None) -> str:
    """
    Text with every unsupported-script letter gone.

    Mostly-foreign text is dropped whole: stripping it would leave a few
    stray words that read as a sentence nobody said. What is left after
    stripping must still hold a letter or digit — "હા." is not "." worth
    keeping.
    """
    text = text or ""
    if not has_foreign_script(text):
        return text
    if is_garbled(text):
        return ""
    cleaned = strip_foreign_script(text)
    return cleaned if any(c.isalnum() for c in cleaned) else ""


_LINE_TEXT_FIELDS = ("text", "raw_text", "cleaned_text")


def clean_transcript(lines: list[dict] | None) -> list[dict]:
    """
    The last gate before a transcript is stored or shown.

    Every STT path already cleans its own output, but a line can reach the
    database or the screen by other routes (overlap separation, an older
    meeting stored before those checks existed). Lines whose text was only
    foreign script are removed; the rest come back as cleaned copies.
    """
    out = []
    for line in lines or []:
        if not isinstance(line, dict):
            out.append(line)
            continue
        if not any(has_foreign_script(line.get(f) or "") for f in _LINE_TEXT_FIELDS) and not _spans_need_cleaning(line):
            out.append(line)
            continue

        cleaned = dict(line)
        for field in _LINE_TEXT_FIELDS:
            if cleaned.get(field):
                cleaned[field] = clean_text(cleaned[field])
        spans = cleaned.get("attributed_spans")
        if isinstance(spans, list):
            cleaned["attributed_spans"] = [
                {**s, "text": clean_text(s.get("text"))} if isinstance(s, dict) else s
                for s in spans
            ]
        if not (cleaned.get("cleaned_text") or cleaned.get("text") or "").strip():
            continue
        out.append(cleaned)
    return out


def _spans_need_cleaning(line: dict) -> bool:
    spans = line.get("attributed_spans")
    return isinstance(spans, list) and any(
        isinstance(s, dict) and has_foreign_script(s.get("text") or "") for s in spans
    )
