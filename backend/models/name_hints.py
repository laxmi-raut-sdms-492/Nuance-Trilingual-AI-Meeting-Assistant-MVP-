"""
Light-weight name hints from transcript text (not voice biometrics).

Used when voice identification has no enrolled match yet — e.g. someone says
"My name is Anushka", or two people greet each other by name — so display
labels become real names and those voices are enrolled permanently for every
future meeting.
"""

from __future__ import annotations

import re

# Prefer explicit self-introductions. Avoid bare "I am …" — too many false
# positives ("I am going", "I am ready").
_INTRO_PATTERNS = (
    re.compile(r"\bmy\s+name\s+is\s+([A-Za-z][A-Za-z'-]{1,40})\b", re.IGNORECASE),
    re.compile(r"\bthis\s+is\s+([A-Za-z][A-Za-z'-]{1,40})\s+speaking\b", re.IGNORECASE),
    re.compile(r"मेरा\s+नाम\s+([^\s,।.!?]+)"),
    re.compile(r"माझ[ें]\s+नाव\s+([^\s,।.!?]+)"),
)

# "Good morning Lakshmi" — Lakshmi is the addressee, NOT the speaker.
_ADDRESSEE_PATTERNS = (
    re.compile(
        r"\b(?:good\s+morning|good\s+afternoon|good\s+evening|hello|hi|hey)"
        r"\s+([A-Za-z][A-Za-z'-]{1,40})\b",
        re.IGNORECASE,
    ),
)


def _normalize_person_name(raw: str) -> str | None:
    raw = (raw or "").strip(" .,!?;:'\"")
    if not raw or len(raw) < 2:
        return None
    if re.search(r"[A-Za-z]", raw):
        return raw.title()
    return raw


def extract_self_introduction_name(text: str) -> str | None:
    """Return a person name if the utterance introduces the speaker, else None."""
    if not text or not text.strip():
        return None
    for pattern in _INTRO_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        name = _normalize_person_name(match.group(1))
        if name:
            return name
    return None


def extract_greeted_name(text: str) -> str | None:
    """Name being greeted (addressee), e.g. Lakshmi in 'Good morning Lakshmi'."""
    if not text or not text.strip():
        return None
    for pattern in _ADDRESSEE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        name = _normalize_person_name(match.group(1))
        if name:
            return name
    return None


def resolve_names_from_greetings(transcript: list[dict]) -> dict[str, str]:
    """
    Infer speaker identities from mutual greetings.

    Example:
      Speaker_00: "Good morning Lakshmi…"
      Speaker_01: "Good morning Anushka…"
    → Speaker_00 = Anushka, Speaker_01 = Lakshmi

    Only assigns when two diarization labels greet distinct names (swap).
    """
    # label -> first greeted name we saw from that label
    greeted: dict[str, str] = {}
    for line in transcript or []:
        label = line.get("speaker_label") or line.get("speaker")
        if not label or label in greeted:
            continue
        name = extract_greeted_name(line.get("text") or "")
        if name:
            greeted[label] = name

    if len(greeted) < 2:
        return {}

    labels = list(greeted.keys())
    assignment: dict[str, str] = {}

    for i, label_a in enumerate(labels):
        for label_b in labels[i + 1 :]:
            name_b = greeted[label_a]  # A greets B's name
            name_a = greeted[label_b]  # B greets A's name
            if not name_a or not name_b or name_a == name_b:
                continue
            # Consistent swap only — don't overwrite a prior assignment.
            if label_a not in assignment and label_b not in assignment:
                assignment[label_a] = name_a
                assignment[label_b] = name_b
            elif label_a in assignment and assignment[label_a] == name_a and label_b not in assignment:
                assignment[label_b] = name_b
            elif label_b in assignment and assignment[label_b] == name_b and label_a not in assignment:
                assignment[label_a] = name_a

    return assignment
