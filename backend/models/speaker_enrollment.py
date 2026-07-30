"""
Enroll a reusable voice profile from meeting audio (or a raw embedding).

Distinct from cosmetic rename: rename only changes labels inside one meeting.
Enrollment stores an ECAPA embedding under a person name so future meetings
can auto-label that voice.

Reuses models.embedding.get_embedding and SpeakerIdentifier.enroll — does not
introduce a second profile store.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from audio_utils import load_audio_file
from config import SAMPLE_RATE
from models.embedding import get_embedding

logger = logging.getLogger("speaker_enrollment")

# Cap how much speech we pull from a meeting when building a profile — enough
# for a stable ECAPA vector without re-encoding an hour of talk-time.
MAX_ENROLLMENT_SECONDS = 30.0
MIN_ENROLLMENT_SECONDS = 0.5


def embedding_from_segments(
    audio: np.ndarray,
    segments: list[tuple[float, float]],
    *,
    max_seconds: float = MAX_ENROLLMENT_SECONDS,
) -> np.ndarray | None:
    """
    Build one L2-normalized embedding by averaging per-segment ECAPA vectors
    over the given (start_sec, end_sec) ranges. Returns None if there is not
    enough usable audio.
    """
    if audio is None or len(audio) == 0 or not segments:
        return None

    collected: list[np.ndarray] = []
    used_seconds = 0.0

    for start, end in segments:
        if used_seconds >= max_seconds:
            break
        start_i = max(0, int(start * SAMPLE_RATE))
        end_i = min(len(audio), int(end * SAMPLE_RATE))
        if end_i - start_i < int(MIN_ENROLLMENT_SECONDS * SAMPLE_RATE):
            continue
        remaining = max_seconds - used_seconds
        max_samples = int(remaining * SAMPLE_RATE)
        if end_i - start_i > max_samples:
            end_i = start_i + max_samples
        clip = audio[start_i:end_i]
        if len(clip) < int(MIN_ENROLLMENT_SECONDS * SAMPLE_RATE):
            continue
        collected.append(get_embedding(clip))
        used_seconds += (end_i - start_i) / SAMPLE_RATE

    if not collected:
        return None

    blended = np.mean(np.stack(collected, axis=0), axis=0)
    norm = np.linalg.norm(blended)
    return blended / norm if norm > 0 else blended


def enroll_from_embedding(identifier, name: str, embedding: np.ndarray, *, overwrite: bool = False) -> dict:
    """
    Persist a profile via the existing SpeakerIdentifier.

    overwrite=False (default): create or blend into an existing profile —
    never replace the centroid outright.
    overwrite=True: replace the stored centroid (explicit user request only).
    """
    name = name.strip()
    if not name:
        raise ValueError("Speaker name is required.")
    if embedding is None or len(embedding) == 0:
        raise ValueError("Embedding is empty.")

    identifier.refresh(force=False)
    existed = name in identifier.enrolled
    identifier.enroll(name, embedding, overwrite=overwrite)
    profile = identifier.enrolled.get(name, {})
    return {
        "status": "enrolled",
        "name": name,
        "created": not existed,
        "overwritten": bool(overwrite and existed),
        "sample_count": int(profile.get("sample_count", 1)),
    }


def enroll_from_meeting_audio(
    identifier,
    *,
    audio_path: str,
    segments: list[tuple[float, float]],
    name: str,
    overwrite: bool = False,
) -> dict:
    """
    Load a meeting recording, extract an embedding from the speaker's
    segments, and enroll under `name`.
    """
    audio = load_audio_file(audio_path)
    embedding = embedding_from_segments(audio, segments)
    if embedding is None:
        raise ValueError(
            "Not enough speech from this speaker to build a voice profile. "
            "Try a longer stretch of their talk-time."
        )
    result = enroll_from_embedding(identifier, name, embedding, overwrite=overwrite)
    logger.info(
        f"enrolled '{result['name']}' from meeting audio "
        f"({len(segments)} segment(s), sample_count={result['sample_count']})"
    )
    return result


def identify_speakers_in_meeting(
    identifier,
    *,
    audio_path: str,
    transcript: list[dict],
) -> list[dict]:
    """
    Match each diarized speaker in a finished meeting against enrolled voices.

    Returns a list of
      {speaker_label, old_name, identified_as, confidence, matched: bool}
    without mutating storage — the API applies renames for matches.
    """
    if not transcript:
        return []

    identifier.refresh(force=False)
    if not identifier.enrolled:
        return []

    by_label: dict[str, list[tuple[float, float]]] = {}
    display_by_label: dict[str, str] = {}
    for line in transcript:
        label = line.get("speaker_label") or line.get("speaker")
        if not label:
            continue
        start = float(line.get("start_sec") or 0)
        end = float(line.get("end_sec") or start)
        by_label.setdefault(label, []).append((start, end))
        display_by_label.setdefault(label, line.get("speaker") or label)

    audio = load_audio_file(audio_path)
    results: list[dict] = []
    for label, segments in by_label.items():
        embedding = embedding_from_segments(audio, segments)
        old_name = display_by_label.get(label, label)
        if embedding is None:
            results.append(
                {
                    "speaker_label": label,
                    "old_name": old_name,
                    "identified_as": "Unknown",
                    "confidence": 0.0,
                    "matched": False,
                }
            )
            continue
        name, confidence = identifier.identify(embedding)
        matched = name != "Unknown"
        results.append(
            {
                "speaker_label": label,
                "old_name": old_name,
                "identified_as": name if matched else old_name,
                "confidence": confidence,
                "matched": matched,
            }
        )
        logger.info(
            f"identify-in-meeting: {label} -> "
            f"{'match ' + name if matched else 'no match'} ({confidence:.3f})"
        )
    return results


def introduction_labels_from_transcript(transcript: list[dict]) -> list[dict]:
    """
    Find names for generic Speaker_XX lines from self-intros and greetings.

    Returns [{speaker_label, old_name, identified_as, confidence, matched, source}].
    """
    from models.name_hints import (
        extract_self_introduction_name,
        resolve_names_from_greetings,
    )

    def _is_generic(name: str) -> bool:
        return bool(re.match(r"(?i)^speaker[_\s]?\d+$", str(name or "").strip()))

    display_by_label: dict[str, str] = {}
    for line in transcript or []:
        label = line.get("speaker_label") or line.get("speaker")
        display = line.get("speaker") or label
        if label:
            display_by_label.setdefault(label, display)

    found: dict[str, dict] = {}

    for line in transcript or []:
        label = line.get("speaker_label") or line.get("speaker")
        if not label or label in found:
            continue
        display = display_by_label.get(label, label)
        if not _is_generic(display):
            continue
        hint = extract_self_introduction_name(line.get("text") or "")
        if hint:
            found[label] = {
                "speaker_label": label,
                "old_name": display,
                "identified_as": hint,
                "confidence": 1.0,
                "matched": True,
                "source": "introduction",
            }

    for label, name in resolve_names_from_greetings(transcript).items():
        display = display_by_label.get(label, label)
        if not _is_generic(display) or label in found:
            continue
        found[label] = {
            "speaker_label": label,
            "old_name": display,
            "identified_as": name,
            "confidence": 1.0,
            "matched": True,
            "source": "greeting",
        }

    return list(found.values())


def same_meeting_fragment_merges(
    *,
    audio_path: str | None,
    transcript: list[dict],
    named_labels: dict[str, str],
) -> list[dict]:
    """
    Attach leftover Speaker_XX labels to a human name already used in this
    meeting.

    1) Embedding similarity to a named cluster (when audio is available)
    2) Turn-taking fallback for short inconclusive leftovers: after Lakshmi
       speaks, a brief Speaker_02 is treated as the other known person
       (Anushka) in a two-person meeting — embeddings on 2–3s clips are often
       useless (~0.27 to everyone).
    """
    from config import (
        WITHIN_MEETING_MERGE_SIMILARITY,
        SHORT_LEFTOVER_SECONDS,
        INCONCLUSIVE_MERGE_SIMILARITY,
    )
    from models.speaker_matcher import cosine_similarity

    def _is_generic(name: str) -> bool:
        return bool(re.match(r"(?i)^speaker[_\s]?\d+$", str(name or "").strip()))

    if not transcript or not named_labels:
        return []

    by_label: dict[str, list[tuple[float, float]]] = {}
    display_by_label: dict[str, str] = {}
    # Chronological previous human display name before each label's first line.
    previous_human: dict[str, str] = {}
    last_human: str | None = None
    for line in sorted(transcript, key=lambda t: float(t.get("start_sec") or 0)):
        label = line.get("speaker_label") or line.get("speaker")
        display = line.get("speaker") or label
        if not label:
            continue
        if label not in by_label and last_human and _is_generic(str(display)):
            previous_human[label] = last_human
        by_label.setdefault(label, []).append(
            (float(line.get("start_sec") or 0), float(line.get("end_sec") or 0))
        )
        display_by_label.setdefault(label, display)
        if display and not _is_generic(str(display)):
            last_human = display

    durations = {
        label: sum(max(0.0, e - s) for s, e in segs) for label, segs in by_label.items()
    }

    centroids: dict[str, np.ndarray] = {}
    if audio_path:
        try:
            audio = load_audio_file(audio_path)
            for label, segments in by_label.items():
                emb = embedding_from_segments(audio, segments)
                if emb is not None:
                    centroids[label] = emb
        except Exception:
            logger.exception("fragment merge: failed to embed meeting audio")

    named_centroids = {
        label: centroids[label] for label in named_labels if label in centroids
    }
    human_names = list(dict.fromkeys(named_labels.values()))

    results: list[dict] = []
    for label, display in display_by_label.items():
        if label in named_labels or not _is_generic(display):
            continue

        best_name, best_sim = None, -1.0
        second_sim = -1.0
        if label in centroids and named_centroids:
            for other, centroid in named_centroids.items():
                sim = cosine_similarity(centroids[label], centroid)
                if sim > best_sim:
                    second_sim = best_sim
                    best_name, best_sim = named_labels[other], sim
                elif sim > second_sim:
                    second_sim = sim

        chosen = None
        confidence = 0.0
        source = "fragment_merge"

        if best_name and best_sim >= WITHIN_MEETING_MERGE_SIMILARITY:
            chosen, confidence = best_name, best_sim
        elif durations.get(label, 0) <= SHORT_LEFTOVER_SECONDS:
            # Embedding inconclusive on a short leftover — use dialogue structure.
            prev = previous_human.get(label)
            if (
                best_name
                and best_sim >= 0.20
                and best_sim >= second_sim + 0.05
            ):
                chosen, confidence = best_name, best_sim
                source = "fragment_merge_weak"
            elif len(human_names) == 2 and prev:
                # Two known people: a brief new label after A is usually B.
                other = next((n for n in human_names if n.lower() != prev.lower()), None)
                if other:
                    chosen, confidence = other, max(best_sim, 0.0)
                    source = "turn_taking"
                    logger.info(
                        f"turn-taking merge: {label} -> {other} "
                        f"(after {prev}, short {durations.get(label, 0):.1f}s, "
                        f"best_emb={best_sim:.3f})"
                    )
            elif prev and (best_sim < INCONCLUSIVE_MERGE_SIMILARITY or best_name is None):
                chosen, confidence = prev, max(best_sim, 0.0)
                source = "continuity"
                logger.info(
                    f"continuity merge: {label} -> {prev} "
                    f"(short {durations.get(label, 0):.1f}s leftover)"
                )

        if chosen:
            results.append(
                {
                    "speaker_label": label,
                    "old_name": display,
                    "identified_as": chosen,
                    "confidence": round(float(confidence), 3),
                    "matched": True,
                    "source": source,
                }
            )
    return results
