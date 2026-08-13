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
    re.compile(r"माझ[ेंं]\s+नाव\s+([^\s,।.!?]+)"),
    re.compile(r"माझं\s+नाव\s+([^\s,।.!?]+)"),
)

# Greeting addressees — English and Devanagari (Hindi/Marathi).
_GREETING_PREFIX = (
    r"(?:"
    r"good\s+morning|good\s+afternoon|good\s+evening|hello|hi|hey"
    r"|नमस्ते|नमस्कार|सुप्रभात|शुभ\s+प्रभात|शुभ\s+सकाळ|शुभ\s+संध्याकाळ"
    r"|हॅलो|हैलो"
    r")"
)
_LATIN_NAME = r"([A-Za-z][A-Za-z'-]{1,40})"
_DEVANAGARI_NAME = r"([\u0900-\u097F][\u0900-\u097F\-]{1,30})"

_ADDRESSEE_PATTERNS = (
    re.compile(rf"\b{_GREETING_PREFIX}\s+{_LATIN_NAME}\b", re.IGNORECASE),
    re.compile(rf"{_GREETING_PREFIX}\s+{_DEVANAGARI_NAME}"),
)


FORBIDDEN_NAME_WORDS = {
    "everyone",
    "everybody",
    "guys",
    "all",
    "there",
    "team",
    "folks",
    "sir",
    "maam",
    "ma'am",
    "friend",
    "friends",
    "anyone",
    "anybody",
    "people",
    "students",
    "here",
    "again",
    "back",
    "today",
    "well",
    "so",
    "now",
    "too",
    "also",
    "welcome",
}


def _normalize_person_name(raw: str) -> str | None:
    raw = (raw or "").strip(" .,!?;:'\"")
    if not raw or len(raw) < 2:
        return None
    if raw.lower() in FORBIDDEN_NAME_WORDS:
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

    Only assigns when two (or more) diarization labels greet distinct names
    via pairwise name-swaps — works for dyads and larger groups that greet
    each other.
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


def repair_collapsed_greeting_turns(transcript: list[dict]) -> list[dict]:
    """
    Fix under-diarization after greetings for any number of people.

    Example (all lines wrongly labeled Speaker_00 / Vaishnavi):
      Vaishnavi: "Good morning Lakshmi…"
      Vaishnavi: "I am fine…"          → Lakshmi
      Vaishnavi: "Good morning Priya…" → (greeter stays; pending = Priya)
      Vaishnavi: "Hello everyone"      → Priya

    Each greeting on a collapsed label can claim the next same-label reply
    for its addressee. Works for 2, 3, or more speakers in one meeting.

    Returns line reassignments:
      [{start_sec, end_sec, old_speaker, new_speaker, speaker_label, source}]
    """
    lines = sorted(
        [dict(t) for t in (transcript or [])],
        key=lambda t: float(t.get("start_sec") or 0),
    )
    changes: list[dict] = []
    # Track the latest intended display name per start_sec so chained
    # greetings on the same collapsed label see updated speaker names.
    renamed: dict[float, str] = {}

    pending_addressee: str | None = None
    pending_label: str | None = None
    pending_end = -1.0

    def _display(line: dict) -> str:
        key = round(float(line.get("start_sec") or 0), 2)
        if key in renamed:
            return renamed[key]
        return line.get("speaker") or line.get("speaker_label") or ""

    for line in lines:
        start = float(line.get("start_sec") or 0)
        end = float(line.get("end_sec") or start)
        label = line.get("speaker_label") or line.get("speaker")
        speaker = _display(line)
        text = line.get("text") or ""
        greeted = extract_greeted_name(text)

        if greeted:
            # New greeting opens a turn: speaker is NOT the addressee.
            pending_addressee = greeted
            pending_label = label
            pending_end = end
            continue

        if not pending_addressee or not pending_label:
            continue
        if label != pending_label:
            pending_addressee = None
            pending_label = None
            continue
        if start - pending_end > 20.0:
            pending_addressee = None
            pending_label = None
            continue

        # Same collapsed label after greeting → addressee answering.
        if speaker and speaker.lower() != pending_addressee.lower():
            changes.append(
                {
                    "start_sec": start,
                    "end_sec": end,
                    "old_speaker": speaker,
                    "new_speaker": pending_addressee,
                    "speaker_label": label,
                    "source": "greeting_turn",
                }
            )
            renamed[round(start, 2)] = pending_addressee
            # Clear so a later greeting on this label can name the next person.
            pending_addressee = None
            pending_label = None
            continue
        pending_end = end

    return changes
