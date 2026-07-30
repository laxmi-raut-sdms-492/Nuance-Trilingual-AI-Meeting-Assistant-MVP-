"""
One MeetingSession instance = one meeting, live or uploaded.

Flow per audio block:

  raw PCM bytes
      -> feed into SpeechSegmenter (Silero VAD) — buffers internally,
         emits a completed segment only when real speech starts and ends
      -> Speaker Change Detection (models/scd.py) — a VAD segment can still
         contain MORE THAN ONE SPEAKER if they talk back-to-back with no
         pause; this splits it into homogeneous single-speaker sub-segments
         BEFORE anything else touches the audio
      -> for each sub-segment:
           - silence/too-short check (cheap safety net)
           - voice embedding (models/embedding.py)
           - diarization: assign a stable Speaker_XX label (models/diarizer.py)
           - identification: match the cluster's stable centroid against
             enrolled voices
           - transcription of just this sub-segment, with its own language
             detected independently (models/asr.py)
           - one transcript entry, sent back to the frontend immediately

The same class drives both paths:
  live   — main.py's WebSocket calls process_chunk() as audio arrives
  upload — api.py calls process_audio() once with the whole decoded file
"""

import logging
import re
import time

import numpy as np

from config import (
    SAMPLE_RATE,
    SILENCE_RMS_THRESHOLD,
    MIN_SPEECH_SECONDS,
    SPEAKER_COLORS,
    DEFAULT_LANGUAGE,
    LANGUAGE_NAMES,
)
from audio_utils import pcm16_bytes_to_float32, rms
from models.embedding import get_embedding
from models.diarizer import SessionDiarizer
from models.vad import SpeechSegmenter
from models.scd import split_on_speaker_change
from models.asr import transcribe
from models.identifier import SpeakerIdentifier
from models.name_hints import (
    extract_self_introduction_name,
    resolve_names_from_greetings,
)
from models.speaker_matcher import UNKNOWN

logger = logging.getLogger("pipeline")
MIN_SPEECH_SAMPLES = int(MIN_SPEECH_SECONDS * SAMPLE_RATE)

# How much audio to hand the segmenter at a time when processing an uploaded
# file. Feeding a 60-minute array in one call would work, but processing in
# blocks keeps memory flat and lets progress be reported as it goes.
FILE_BLOCK_SAMPLES = 30 * SAMPLE_RATE


def format_timestamp(seconds: float) -> str:
    """12.4 -> '00:12'. Matches what the transcript UI renders per line."""
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_duration(seconds: float) -> str:
    """195.0 -> '3m 15s'. Used for the per-speaker talk-time labels."""
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


class MeetingSession:
    def __init__(self, session_id: str, identifier: SpeakerIdentifier):
        self.session_id = session_id
        self.identifier = identifier
        self.diarizer = SessionDiarizer()
        self.segmenter = SpeechSegmenter()
        self.transcript: list[dict] = []
        self.created_at = time.time()
        # Segments lost to an exception rather than to a silence/length check.
        # An empty transcript because nobody spoke and an empty transcript
        # because every embedding call threw look identical from the outside;
        # this is what tells them apart.
        self.failed_segments = 0
        self.last_error: str | None = None
        # Colour is assigned on a speaker's first appearance and then reused,
        # so the transcript, the talk-time bars, and the pie chart all agree.
        self._speaker_colors: dict[str, str] = {}
        # Running tally of which language this meeting is mostly in — used as
        # the fallback when one short segment's own detection is inconclusive.
        self._language_counts: dict[str, int] = {}

    # -- live path --

    def process_chunk(self, pcm_bytes: bytes) -> list[dict]:
        """Feed in raw PCM16 mono 16kHz bytes. Returns any new transcript entries."""
        audio = pcm16_bytes_to_float32(pcm_bytes)
        return self._consume(self.segmenter.process(audio))

    def finish(self) -> list[dict]:
        """End of stream — drain the segment still buffered in the VAD."""
        entries = self._consume(self.segmenter.flush())
        self.finalize_labels()
        return entries

    def finalize_labels(self) -> None:
        """
        Resolve display names for every diarization label, then permanently
        enroll those voices.

        Order matters:
          1) match against already-enrolled profiles
          2) apply text hints (self-intro / mutual greetings)
          3) enroll resolved voices
          4) re-match leftover Speaker_XX against enrollments (just updated)
          5) merge same-meeting fragments (Speaker_02 ≈ Anushka's Speaker_00)

        Diarization ids stay on speaker_label; only display fields change.
        """
        if not self.transcript:
            return

        from config import WITHIN_MEETING_MERGE_SIMILARITY
        from models.speaker_matcher import cosine_similarity

        def _is_generic(name: str) -> bool:
            import re

            return bool(re.match(r"(?i)^speaker[_\s]?\d+$", str(name or "").strip()))

        # label -> (name, confidence)
        resolved: dict[str, tuple[str, float]] = {}

        if self.diarizer.clusters:
            for label in self.diarizer.clusters:
                name, conf = self.identifier.identify(self.diarizer.get_centroid(label))
                if name != UNKNOWN:
                    resolved[label] = (name, conf)

        # Text hints for labels still unknown.
        for entry in self.transcript:
            label = entry.get("speaker_label")
            if not label or label in resolved:
                continue
            hint = extract_self_introduction_name(entry.get("text") or "")
            if hint:
                resolved[label] = (hint, 1.0)

        for label, name in resolve_names_from_greetings(self.transcript).items():
            resolved.setdefault(label, (name, 1.0))

        # Permanently enroll every resolved voice (needed before re-identify).
        for label, (name, _conf) in list(resolved.items()):
            if label not in self.diarizer.clusters:
                continue
            try:
                self.identifier.enroll(
                    name, self.diarizer.get_centroid(label), overwrite=False
                )
                logger.info(f"permanently enrolled voice profile for '{name}'")
            except Exception as exc:
                logger.warning(f"finalize enroll '{name}' failed: {exc}")

        # Re-identify leftover clusters against profiles we just enrolled.
        if self.diarizer.clusters:
            self.identifier.refresh(force=True)
            for label in self.diarizer.clusters:
                if label in resolved:
                    continue
                name, conf = self.identifier.identify(self.diarizer.get_centroid(label))
                if name != UNKNOWN:
                    resolved[label] = (name, conf)
                    logger.info(
                        f"re-identify: {label} -> {name} ({conf:.3f}) after enrollment"
                    )

        # Same-meeting fragment repair via embeddings.
        if self.diarizer.clusters:
            named = {
                label: (name, conf)
                for label, (name, conf) in resolved.items()
                if label in self.diarizer.clusters
            }
            for label in self.diarizer.clusters:
                if label in resolved:
                    continue
                best_name, best_sim = None, -1.0
                centroid = self.diarizer.get_centroid(label)
                for other, (name, conf) in named.items():
                    sim = cosine_similarity(centroid, self.diarizer.get_centroid(other))
                    if sim > best_sim:
                        best_name, best_sim = name, sim
                if best_name and best_sim >= WITHIN_MEETING_MERGE_SIMILARITY:
                    resolved[label] = (best_name, round(float(best_sim), 3))
                    logger.info(
                        f"within-meeting merge: {label} -> {best_name} "
                        f"(sim={best_sim:.3f} vs named cluster)"
                    )
                    try:
                        self.identifier.enroll(
                            best_name, centroid, overwrite=False
                        )
                    except Exception as exc:
                        logger.warning(f"merge enroll '{best_name}' failed: {exc}")

        # Turn-taking / continuity for short leftovers embeddings couldn't resolve.
        from models.speaker_enrollment import same_meeting_fragment_merges

        named_for_merge = {label: name for label, (name, _) in resolved.items()}
        # Also include humans already on the transcript.
        for entry in self.transcript:
            label = entry.get("speaker_label")
            speaker = entry.get("speaker")
            if label and speaker and label in resolved:
                named_for_merge[label] = resolved[label][0]
            elif label and speaker and not re.match(r"(?i)^speaker[_\s]?\d+$", str(speaker)):
                named_for_merge.setdefault(label, speaker)

        for frag in same_meeting_fragment_merges(
            audio_path=None,
            transcript=self.transcript,
            named_labels=named_for_merge,
        ):
            label = frag["speaker_label"]
            if label not in resolved:
                resolved[label] = (frag["identified_as"], frag["confidence"])
                logger.info(
                    f"{frag['source']}: {label} -> {frag['identified_as']}"
                )

        for entry in self.transcript:
            label = entry.get("speaker_label")
            if not label:
                continue
            if label in resolved:
                name, conf = resolved[label]
                entry["identified_as"] = name
                entry["confidence"] = conf
                entry["speaker"] = name
            else:
                # Keep prior human names; only fall back to diarization id when
                # still generic / unknown.
                current = entry.get("speaker") or label
                identified = entry.get("identified_as") or UNKNOWN
                if identified != UNKNOWN and not _is_generic(str(identified)):
                    entry["speaker"] = identified
                elif not _is_generic(str(current)):
                    entry["speaker"] = current
                else:
                    entry["identified_as"] = UNKNOWN
                    entry["speaker"] = label

    # -- upload path --

    def process_audio(self, audio: np.ndarray, on_progress=None) -> list[dict]:
        """
        Run a complete, already-decoded recording through the same pipeline.

        audio: 1-D float32, 16kHz mono (see audio_utils.load_audio_file).
        on_progress: optional callback(fraction_0_to_1) for status reporting.
        """
        total = max(len(audio), 1)
        entries = []

        for offset in range(0, len(audio), FILE_BLOCK_SAMPLES):
            block = audio[offset : offset + FILE_BLOCK_SAMPLES]
            entries.extend(self._consume(self.segmenter.process(block)))
            if on_progress:
                on_progress(min((offset + len(block)) / total, 0.99))

        entries.extend(self.finish())
        if on_progress:
            on_progress(1.0)
        return entries

    # -- shared core --

    def _consume(self, vad_segments: list[dict]) -> list[dict]:
        new_entries = []

        for vad_seg in vad_segments:
            seg_audio = vad_seg["audio"]
            seg_start = vad_seg["start"]

            if len(seg_audio) == 0:
                continue

            try:
                # A VAD segment can still contain a speaker change (back-to-back
                # talkers, no pause) — split into homogeneous pieces first.
                sub_bounds = split_on_speaker_change(seg_audio)
            except Exception as e:
                logger.error(f"[{seg_start:.1f}s] SCD failed, falling back to whole segment: {e}")
                self.last_error = str(e)
                sub_bounds = [(0, len(seg_audio))]

            for sub_start_sample, sub_end_sample in sub_bounds:
                sub_audio = seg_audio[sub_start_sample:sub_end_sample]
                sub_start = seg_start + sub_start_sample / SAMPLE_RATE
                sub_end = seg_start + sub_end_sample / SAMPLE_RATE

                try:
                    entry = self._process_subsegment(sub_start, sub_end, sub_audio)
                except Exception as e:
                    # One bad segment (degenerate/NaN embedding, empty audio,
                    # unexpected model error) must not kill the rest of the
                    # session — log it and keep going.
                    logger.error(f"[{sub_start:.1f}-{sub_end:.1f}s] segment processing failed, skipping: {e}")
                    self.failed_segments += 1
                    self.last_error = str(e)
                    entry = None

                if entry:
                    new_entries.append(entry)

        return new_entries

    def _process_subsegment(self, start: float, end: float, audio) -> dict | None:
        if len(audio) < MIN_SPEECH_SAMPLES or rms(audio) < SILENCE_RMS_THRESHOLD:
            return None  # too short or too quiet to be meaningful speech

        embedding = get_embedding(audio)
        speaker_label = self.diarizer.add_segment(start, end, embedding)

        stable_embedding = self.diarizer.get_centroid(speaker_label)
        identified_as, confidence = self.identifier.identify(stable_embedding)

        asr = transcribe(audio, hint_language=self.dominant_language)
        if not asr["text"]:
            return None

        self._language_counts[asr["language"]] = self._language_counts.get(asr["language"], 0) + 1

        # Voice match first. If nobody is enrolled yet, fall back to an
        # explicit self-introduction in the words ("My name is Anushka") and
        # enroll that voice automatically so later meetings label themselves.
        if identified_as == UNKNOWN:
            hint = extract_self_introduction_name(asr["text"])
            if hint:
                identified_as = hint
                confidence = 1.0
                try:
                    self.identifier.enroll(hint, stable_embedding, overwrite=False)
                    logger.info(
                        f"[{start:.1f}s] auto-enrolled '{hint}' from self-introduction"
                    )
                except Exception as exc:
                    logger.warning(f"[{start:.1f}s] auto-enroll '{hint}' failed: {exc}")

        display_name = identified_as if identified_as != UNKNOWN else speaker_label
        entry = {
            "start_sec": round(start, 2),
            "end_sec": round(end, 2),
            "time": format_timestamp(start),
            "speaker": display_name,
            "speaker_label": speaker_label,
            "identified_as": identified_as,
            "confidence": confidence,
            "color": self._color_for(speaker_label),
            "language": asr["language"],
            "language_name": asr["language_name"],
            "language_prob": asr["language_prob"],
            "language_detected": asr["language_detected"],
            "language_fallback": asr["language_fallback"],
            "text": asr["text"],
        }
        self.transcript.append(entry)
        return entry

    def _color_for(self, speaker_label: str) -> str:
        if speaker_label not in self._speaker_colors:
            index = len(self._speaker_colors) % len(SPEAKER_COLORS)
            self._speaker_colors[speaker_label] = SPEAKER_COLORS[index]
        return self._speaker_colors[speaker_label]

    @property
    def dominant_language(self) -> str:
        if not self._language_counts:
            return DEFAULT_LANGUAGE
        return max(self._language_counts, key=self._language_counts.get)

    # -- results --

    def speaker_stats(self) -> list[dict]:
        """
        Talk time per speaker, shaped for the details page's bars and the
        dashboard pie: {name, seconds, time, pct, color}.
        """
        totals: dict[str, float] = {}
        for entry in self.transcript:
            key = entry["speaker"]
            totals[key] = totals.get(key, 0.0) + (entry["end_sec"] - entry["start_sec"])

        grand_total = sum(totals.values())
        if grand_total <= 0:
            return []

        colors = {e["speaker"]: e["color"] for e in self.transcript}
        return [
            {
                "name": name,
                "seconds": round(seconds, 1),
                "time": format_duration(seconds),
                "pct": round(seconds / grand_total * 100, 1),
                "color": colors.get(name, SPEAKER_COLORS[0]),
            }
            for name, seconds in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
        ]

    def language_breakdown(self) -> list[dict]:
        """Which of the three languages this meeting was actually spoken in."""
        totals: dict[str, float] = {}
        for entry in self.transcript:
            code = entry["language"]
            totals[code] = totals.get(code, 0.0) + (entry["end_sec"] - entry["start_sec"])

        grand_total = sum(totals.values())
        if grand_total <= 0:
            return []

        return [
            {
                "code": code,
                "name": LANGUAGE_NAMES.get(code, code),
                "seconds": round(seconds, 1),
                "pct": round(seconds / grand_total * 100, 1),
            }
            for code, seconds in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
        ]

    def spoken_duration(self) -> float:
        """Total seconds of actual speech (not wall-clock file length)."""
        return sum(e["end_sec"] - e["start_sec"] for e in self.transcript)

    def full_transcript(self) -> dict:
        return {
            "session_id": self.session_id,
            "transcript": self.transcript,
            "speaker_stats": self.speaker_stats(),
            "languages": self.language_breakdown(),
            "speaking_time_seconds": {s["name"]: s["seconds"] for s in self.speaker_stats()},
        }
