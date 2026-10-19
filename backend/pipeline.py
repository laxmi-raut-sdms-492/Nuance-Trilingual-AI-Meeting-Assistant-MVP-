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
           - Language Change Detection (models/lcd.py) — one speaker can
             CODE-SWITCH mid-breath with no pause, so a single-speaker
             sub-segment may still hold two languages; this cuts it into
             language-homogeneous pieces. Everything above is computed once
             and shared by every piece: the speaker did not change.
           - for each piece: transcription with its own language detected
             independently (models/asr.py)
           - one transcript entry per piece, sent back to the frontend
             immediately

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
from audio_utils import pcm16_bytes_to_float32, rms, prepare_upload_audio
from models.embedding import get_embedding
from models.diarizer import SessionDiarizer
from models.vad import SpeechSegmenter, fallback_segments
from models.scd import split_on_speaker_change
from models.lcd import split_on_language_change
from models.identifier import SpeakerIdentifier
from stt.script_guard import clean_text, has_foreign_script
from models.name_hints import (
    extract_self_introduction_name,
    line_can_name_a_speaker,
    resolve_names_from_greetings,
)
from models.speaker_matcher import UNKNOWN
from stt.base import STTAdapter
from stt.local_adapter import LocalSTTAdapter

logger = logging.getLogger("pipeline")


def _get_min_speech_samples() -> int:
    from config import MIN_SPEECH_SECONDS, SAMPLE_RATE
    return int(MIN_SPEECH_SECONDS * SAMPLE_RATE)


MIN_SPEECH_SAMPLES = _get_min_speech_samples()

# How much audio to hand the segmenter at a time when processing an uploaded
# file. Feeding a 60-minute array in one call would work, but processing in
# blocks keeps memory flat and lets progress be reported as it goes.
FILE_BLOCK_SAMPLES = 30 * SAMPLE_RATE


def _is_generic(name: str) -> bool:
    import re
    return bool(re.match(r"(?i)^speaker[_\s]?\d+$", str(name or "").strip()))


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


from stt.base import STTAdapter
from stt.factory import get_stt_adapter


class MeetingSession:
    def __init__(
        self,
        session_id: str,
        identifier: SpeakerIdentifier,
        stt_adapter: STTAdapter | None = None,
    ):
        self.session_id = session_id
        self.identifier = identifier
        self.stt_adapter = stt_adapter or get_stt_adapter()
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
        # Per-transcript-line ECAPA embeddings (aligned with self.transcript).
        # Used for an offline auto-k recluster pass after the full upload.
        self._embeddings: list = []
        self.raw_segments: list[dict] = []  # pre-turn ASR fragments (debug)
        self._full_audio: np.ndarray | None = None
        self.audio_quality_warning: str | None = None
        self._upload_profile = None
        self._silence_threshold = SILENCE_RMS_THRESHOLD
        self._on_transcript_update = None

    # -- live path --

    def process_chunk(self, pcm_bytes: bytes) -> list[dict]:
        """Feed in raw PCM16 mono 16kHz bytes. Returns any new transcript entries."""
        audio = pcm16_bytes_to_float32(pcm_bytes)
        return self._consume(self.segmenter.process(audio))

    def finish(self) -> list[dict]:
        """End of stream — drain the segment still buffered in the VAD."""
        entries = self._consume(self.segmenter.flush())
        self._finalize_session()
        return entries

    def _finalize_session(self) -> None:
        """Offline recluster + name resolution + speaker turns — shared by live and upload."""
        self.apply_pending_relabels()
        self.apply_offline_recluster()
        self.apply_overlap_detection()
        # After overlap detection, so a line already attributed to two people
        # on purpose is left alone rather than re-cut to one speaker each.
        self.apply_resegmentation()
        self.finalize_labels()
        self.build_and_apply_turns()

    def _maybe_vad_fallback(self, audio: np.ndarray, vad_emitted: int) -> list[dict]:
        """
        When streaming VAD produces no usable transcript, try batch VAD and
        fixed windows. Also runs when VAD emitted tiny false positives on a
        flat-tone upload — a common failure mode on corrupted WAV exports.
        """
        if self.transcript:
            return []
        if len(audio) < _get_min_speech_samples() or rms(audio) < self._silence_threshold:
            return []

        profile = self._upload_profile
        fb_kwargs: dict = {}
        if profile:
            fb_kwargs["batch_vad_threshold"] = profile.batch_vad_threshold
            fb_kwargs["batch_vad_min_speech_ratio"] = profile.batch_vad_min_speech_ratio
        segments = fallback_segments(audio, **fb_kwargs)
        if not segments:
            return []

        logger.warning(
            f"[{self.session_id}] streaming VAD emitted {vad_emitted} segment(s) "
            f"but produced no transcript — falling back to {len(segments)} segment(s)"
        )
        return self._consume(segments)

    def apply_pending_relabels(self) -> int:
        """
        Apply the diarizer's retroactive corrections to lines already written.

        A segment whose voice was too far from every cluster to trust is given
        the nearest label and buffered. If a later segment then establishes the
        cluster it actually belonged to, the diarizer works out the correction —
        but by then this line has already been emitted (live, it has been sent
        to the browser), so the diarizer records the correction rather than
        applying it, and this is where it lands.

        Matched on the segment's own time span, which is unique per line and
        survives the label changes happening around it.
        """
        relabels = getattr(self.diarizer, "pending_relabels", None)
        if not relabels or not self.transcript:
            return 0

        applied = 0
        for change in relabels:
            for entry in self.transcript:
                if (
                    abs(float(entry.get("start_sec") or 0) - change["start"]) > 0.01
                    or abs(float(entry.get("end_sec") or 0) - change["end"]) > 0.01
                ):
                    continue
                if entry.get("speaker_label") != change["from_label"]:
                    continue
                entry["speaker_label"] = change["to_label"]
                # `speaker` is the display name; it only tracks the label while
                # it still *is* the label. A name resolved from an enrollment or
                # a text hint must not be overwritten here — finalize_labels
                # runs next and owns that field.
                if entry.get("speaker") == change["from_label"]:
                    entry["speaker"] = change["to_label"]
                entry["color"] = self._color_for(change["to_label"])
                applied += 1
                break

        if applied:
            logger.info(
                f"[{self.session_id}] applied {applied} retroactive speaker "
                f"correction(s) deferred during streaming"
            )
        relabels.clear()
        return applied

    def apply_offline_recluster(self) -> dict:
        """
        Second-pass diarization once the full meeting is in memory.

        Streaming labels are good enough live; file uploads over-split. Offline
        agglomerative + silhouette picks the real speaker count dynamically.
        """
        from models.offline_diarizer import recluster_from_embeddings

        if not self.transcript or not self._embeddings:
            return {"applied": False, "reason": "no_data"}
        if len(self._embeddings) != len(self.transcript):
            logger.warning(
                f"[{self.session_id}] embedding/transcript length mismatch "
                f"{len(self._embeddings)}/{len(self.transcript)} — skip recluster"
            )
            return {"applied": False, "reason": "length_mismatch"}

        k_max = self._upload_profile.k_max if self._upload_profile else 12
        new_transcript, info = recluster_from_embeddings(
            self.transcript, self._embeddings, k_max=k_max
        )
        if info.get("applied"):
            self.transcript = new_transcript
            self._rebuild_diarizer_from_embeddings()
            # Colours must follow the new labels.
            self._speaker_colors = {}
            for entry in self.transcript:
                entry["color"] = self._color_for(entry["speaker_label"])
            logger.info(
                f"[{self.session_id}] offline recluster: "
                f"{info.get('streaming_k')} -> {info.get('k')} speakers "
                f"(silhouette={info.get('silhouette')})"
            )
        return info

    def apply_overlap_detection(self) -> dict:
        """
        Re-decide overlap with the final speakers (models/overlap_detection.py).

        Uploads only: it needs the whole recording, and the live path never
        keeps it. Runs after the recluster so overlap names the same speakers
        every other line does.
        """
        from config import OVERLAP_DETECTION_ENABLED
        from models.overlap_detection import detect_overlaps

        if not OVERLAP_DETECTION_ENABLED or self._full_audio is None:
            return {"applied": False, "reason": "disabled_or_live"}
        if len(self._embeddings) != len(self.transcript):
            return {"applied": False, "reason": "length_mismatch"}

        self.transcript, info = detect_overlaps(self.transcript, self._embeddings, self._full_audio)
        if info.get("applied"):
            for entry in self.transcript:
                main = (entry.get("candidate_labels") or [entry.get("speaker_label")])[0]
                entry["color"] = self._color_for(main)
            logger.info(f"[{self.session_id}] overlap detection: {info.get('overlaps', 0)} line(s)")
        return info

    def _rebuild_diarizer_from_embeddings(self) -> None:
        """Replace streaming clusters with centroids of the offline labels."""
        self.diarizer = SessionDiarizer()
        by_label: dict[str, list] = {}
        for entry, emb in zip(self.transcript, self._embeddings):
            label = entry.get("speaker_label") or entry.get("speaker")
            if not label:
                continue
            by_label.setdefault(label, []).append(np.asarray(emb, dtype=np.float32))

        max_id = -1
        for label, embs in by_label.items():
            stacked = np.stack(embs)
            centroid = stacked.mean(axis=0)
            norm = float(np.linalg.norm(centroid)) + 1e-9
            centroid = centroid / norm
            self.diarizer.clusters[label] = {
                "centroid": centroid.astype(np.float32),
                "count": len(embs),
                "embeddings": [e.copy() for e in embs[-24:]],
            }
            m = re.match(r"(?i)^speaker[_\s]?(\d+)$", str(label).strip())
            if m:
                max_id = max(max_id, int(m.group(1)))
        self.diarizer._next_label_id = max_id + 1

    def apply_resegmentation(self) -> int:
        """
        Re-cut lines that hold more than one speaker, against known voices.

        SCD decides where a speaker changes by comparing each short window to
        the window before it — noisy against noisy — so it either misses a
        short turn or invents cuts. Once the meeting is complete there is a
        better reference available than the neighbouring window: the speaker
        centroids themselves, averaged over every confident second each
        speaker owns. models/resegment.py scores windows against those.

        Only lines this pass finds to be mixed are touched, and those are
        re-transcribed per piece, because the text of a line that turned out
        to hold two speakers belongs to neither of them as a whole. Everything
        else keeps the text it already has, so the cost is a window scan plus
        ASR on the handful of lines that were actually wrong.

        Uploads only: it needs the full audio and the finished cluster set.
        """
        if self._full_audio is None or not self.transcript:
            return 0
        if len(self._embeddings) != len(self.transcript):
            logger.warning(
                f"[{self.session_id}] embedding/transcript mismatch "
                f"{len(self._embeddings)}/{len(self.transcript)} — skip resegmentation"
            )
            return 0

        from models.embedding import get_embedding
        from models.resegment import build_references, refine_span

        references = build_references(self._embeddings, self.transcript)
        if len(references) < 2:
            return 0

        rebuilt: list[dict] = []
        rebuilt_embeddings: list = []
        changed = 0

        # _transcribe_piece appends to self.transcript; swap it out so the
        # rewrite builds a new list instead of growing the one being walked.
        original_transcript = self.transcript
        original_embeddings = self._embeddings
        original_callback = self._on_transcript_update
        self._on_transcript_update = None

        try:
            for entry, embedding in zip(original_transcript, original_embeddings):
                start = float(entry.get("start_sec") or 0)
                end = float(entry.get("end_sec") or 0)
                label = entry.get("speaker_label")

                # An overlap line is already attributed to two people on
                # purpose; cutting it into one speaker per piece would throw
                # that away.
                if entry.get("is_overlap") or end <= start or not label:
                    rebuilt.append(entry)
                    rebuilt_embeddings.append(embedding)
                    continue

                pieces = refine_span(
                    self._full_audio, start, end, references,
                    sample_rate=SAMPLE_RATE, embed=get_embedding,
                )
                if len(pieces) < 2:
                    # Unchanged, including when the single piece disagrees with
                    # the current label — reassigning a whole line is the
                    # offline recluster's job, decided over the meeting rather
                    # than from one span.
                    rebuilt.append(entry)
                    rebuilt_embeddings.append(embedding)
                    continue

                logger.info(
                    f"[{self.session_id}] resegmenting {start:.1f}-{end:.1f}s "
                    f"({entry.get('speaker_label')}) into "
                    f"{[(round(a, 1), round(b, 1), l) for a, b, l in pieces]}"
                )

                self.transcript = []
                self._embeddings = []
                produced = self._retranscribe_pieces(pieces, embedding)
                self.transcript = original_transcript
                self._embeddings = original_embeddings

                if produced:
                    rebuilt.extend(produced)
                    rebuilt_embeddings.extend(self._resegment_embeddings)
                    changed += 1
                else:
                    # Re-transcription produced nothing usable (too short, or
                    # the guards rejected every piece). Keeping the original
                    # line is strictly better than dropping speech.
                    rebuilt.append(entry)
                    rebuilt_embeddings.append(embedding)
        finally:
            self.transcript = original_transcript
            self._embeddings = original_embeddings
            self._on_transcript_update = original_callback

        if not changed:
            return 0

        self.transcript = rebuilt
        self._embeddings = rebuilt_embeddings
        self.transcript.sort(key=lambda e: float(e.get("start_sec") or 0))
        logger.info(
            f"[{self.session_id}] resegmentation rewrote {changed} mixed-speaker line(s); "
            f"{len(original_transcript)} -> {len(self.transcript)} lines"
        )
        return changed

    def _retranscribe_pieces(self, pieces, fallback_embedding) -> list[dict]:
        """Embed, label and transcribe each piece of a line being re-cut."""
        from models.embedding import get_embedding

        self._resegment_embeddings = []
        produced: list[dict] = []

        for piece_start, piece_end, label in pieces:
            segment = self._full_audio[
                int(piece_start * SAMPLE_RATE) : int(piece_end * SAMPLE_RATE)
            ]
            if len(segment) < _get_min_speech_samples():
                continue
            try:
                embedding = get_embedding(segment)
            except Exception as exc:
                logger.warning(f"[{piece_start:.1f}s] resegment embed failed: {exc}")
                embedding = np.asarray(fallback_embedding, dtype=np.float32)

            speaker_label = label or ""
            if not speaker_label:
                continue

            centroid = self.diarizer.get_centroid(speaker_label)
            stable = centroid if centroid is not None else embedding
            # Voice profiles do not name anyone here either — same rule as
            # _process_subsegment. A spoken name is applied afterwards by
            # finalize_labels; until then the label stays Speaker_XX.
            identified_as, confidence = UNKNOWN, 0.0

            before = len(self.transcript)
            self._transcribe_piece(
                piece_start, piece_end, segment,
                speaker_label=speaker_label,
                embedding=embedding,
                stable_embedding=stable,
                identified_as=identified_as,
                confidence=confidence,
                was_split=True,
            )
            if len(self.transcript) > before:
                produced.append(self.transcript[-1])
                self._resegment_embeddings.append(self._embeddings[-1])

        return produced

    def finalize_labels(self) -> None:
        """
        Fill a display name only when that name is spoken in the transcript.

        A self-introduction ("my name is Anushka") names that speaker.
        A mutual greeting ("Good morning Lakshmi" / "Good morning Anushka")
        names both speakers from the names that were said. Saved voice
        profiles are not used, so an unspoken speaker stays Speaker_00,
        Speaker_01, and so on.
        """
        if not self.transcript:
            return

        # label -> (name, confidence)
        resolved: dict[str, tuple[str, float]] = {}

        for entry in self.transcript:
            label = entry.get("speaker_label")
            if not label or label in resolved or not line_can_name_a_speaker(entry):
                continue
            hint = extract_self_introduction_name(entry.get("text") or "")
            if hint:
                resolved[label] = (hint, 1.0)

        reliable = [entry for entry in self.transcript if line_can_name_a_speaker(entry)]
        for label, name in resolve_names_from_greetings(reliable).items():
            resolved.setdefault(label, (name, 1.0))

        for entry in self.transcript:
            if entry.get("is_overlap"):
                c_labels = entry.get("candidate_labels") or []
                c_speakers = entry.get("candidate_speakers") or []
                resolved_candidates = []
                for lab in c_labels:
                    name = resolved[lab][0] if lab in resolved else lab
                    # Two speakers who were given the same spoken name would
                    # read as "Asha + Asha"; keep the second one's label.
                    resolved_candidates.append(name if name not in resolved_candidates else lab)
                if resolved_candidates:
                    entry["candidate_speakers"] = resolved_candidates
                    disp = " + ".join(resolved_candidates)
                    entry["speaker"] = disp
                    entry["identified_as"] = disp
                continue
            label = entry.get("speaker_label")
            if not label:
                continue
            if label in resolved:
                name, conf = resolved[label]
                entry["identified_as"] = name
                entry["confidence"] = conf
                entry["speaker"] = name
            else:
                # A name that was not spoken stays off the transcript.
                entry["identified_as"] = UNKNOWN
                entry["speaker"] = label

        # Under-split repair: greeting reply glued onto the greeter
        # (e.g. both lines labeled Vaishnavi; second is Lakshmi answering).
        from models.name_hints import repair_collapsed_greeting_turns

        greeting_fixes = repair_collapsed_greeting_turns(self.transcript)
        if greeting_fixes:
            used_ids: set[int] = set()
            for entry in self.transcript:
                m = re.match(
                    r"(?i)^speaker[_\s]?(\d+)$",
                    str(entry.get("speaker_label") or "").strip(),
                )
                if m:
                    used_ids.add(int(m.group(1)))
            next_id = (max(used_ids) + 1) if used_ids else 0
            by_start = {
                round(float(c["start_sec"]), 2): c for c in greeting_fixes
            }
            for entry in self.transcript:
                key = round(float(entry.get("start_sec") or 0), 2)
                change = by_start.get(key)
                if not change:
                    continue
                new_name = change["new_speaker"]
                new_label = f"Speaker_{next_id:02d}"
                next_id += 1
                entry["speaker"] = new_name
                entry["identified_as"] = new_name
                entry["speaker_label"] = new_label
                entry["confidence"] = 1.0
                logger.info(
                    f"greeting_turn: line @{key:.2f}s -> {new_name} ({new_label})"
                )

    def build_and_apply_turns(self) -> None:
        """
        Merge fragmented ASR segments into readable speaker turns.
        Preserves raw_segments for debugging; transcript becomes turns for UI.
        """
        if not self.transcript:
            return

        from config import TRANSCRIPT_CLEANUP_ENABLED
        from models.speaker_turns import build_speaker_turns
        from models.transcript_cleanup import cleanup_turns

        # Snapshot raw ASR fragments before merging.
        self.raw_segments = [
            {**entry, "raw_text": entry.get("raw_text") or entry.get("text", "")}
            for entry in self.transcript
        ]

        turns = build_speaker_turns(self.raw_segments)
        if TRANSCRIPT_CLEANUP_ENABLED:
            turns = cleanup_turns(turns, use_llm=True)
        else:
            from models.transcript_cleanup import cleanup_turns as _ct
            turns = _ct(turns, use_llm=False)

        logger.info(
            f"[{self.session_id}] speaker turns: "
            f"{len(self.raw_segments)} segment(s) -> {len(turns)} turn(s)"
        )
        self.transcript = turns

    # -- upload path --

    def process_audio(
        self,
        audio: np.ndarray,
        on_progress=None,
        on_transcript_update=None,
    ) -> list[dict]:
        """
        Run a complete, already-decoded recording through the same pipeline.

        audio: 1-D float32, 16kHz mono (see audio_utils.load_audio_file).
        on_progress: optional callback(fraction_0_to_1) for status reporting.
        on_transcript_update: optional callback() after each new transcript line.
        """
        audio, profile = prepare_upload_audio(audio)
        self._full_audio = audio
        self._upload_profile = profile
        self._silence_threshold = profile.silence_rms_threshold
        self.audio_quality_warning = profile.warning
        self._on_transcript_update = on_transcript_update
        if profile.warning:
            logger.warning(f"[{self.session_id}] {profile.warning}")
        self.segmenter = SpeechSegmenter(
            min_silence_ms=profile.min_silence_ms,
            threshold=profile.vad_threshold,
        )
        logger.info(
            f"[{self.session_id}] upload profile: vad={profile.vad_threshold:.2f}, "
            f"silence_rms={profile.silence_rms_threshold:.4f}, k_max={profile.k_max}"
        )

        total = max(len(audio), 1)
        entries = []

        # Whole-file VAD segmentation prevents sentence truncation across 30s block boundaries
        # and guarantees that trailing speech at the end of the meeting is never lost.
        from models.vad import batch_vad_segments
        segs = batch_vad_segments(audio, threshold=profile.vad_threshold)
        if segs:
            total_segs = max(len(segs), 1)
            for idx, seg in enumerate(segs, start=1):
                entries.extend(self._consume([seg]))
                if on_progress:
                    on_progress(idx / total_segs)
                if on_transcript_update:
                    on_transcript_update()
        else:
            entries.extend(self._maybe_vad_fallback(audio, 0))
            if on_progress:
                on_progress(0.9)
            if on_transcript_update:
                on_transcript_update()

        self._finalize_session()
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
                    entries = self._process_subsegment(sub_start, sub_end, sub_audio)
                except (AttributeError, TypeError, NameError, KeyError) as e:
                    logger.exception(f"[{sub_start:.1f}-{sub_end:.1f}s] Critical code error in segment processing: {e}")
                    self.failed_segments += 1
                    self.last_error = str(e)
                    raise
                except Exception as e:
                    # One bad segment (degenerate/NaN embedding, empty audio,
                    # unexpected model error) must not kill the rest of the
                    # session — log it with full traceback and keep going.
                    logger.error(f"[{sub_start:.1f}-{sub_end:.1f}s] segment processing failed, skipping: {e}", exc_info=True)
                    self.failed_segments += 1
                    self.last_error = str(e)
                    entries = []

                new_entries.extend(entries)

        return new_entries

    def _process_subsegment(self, start: float, end: float, audio) -> list[dict]:
        """
        One single-speaker sub-segment -> zero or more transcript entries.

        More than one when the speaker code-switched inside it with no pause:
        models/lcd.py cuts the audio where the language changes, and each
        piece is transcribed by the engine its own language calls for. Before
        that split existed the whole sub-segment went to one engine and the
        minority-language half came back mangled.

        Everything about WHO is computed once for the whole sub-segment and
        shared by every piece. The speaker did not change — only the language
        did — and re-embedding a 1.5s fragment would risk a false speaker
        split, which is the over-split problem offline_diarizer.py exists to
        undo. Sharing the embedding also keeps self._embeddings aligned with
        self.transcript, which the offline recluster pass requires.
        """
        if len(audio) < _get_min_speech_samples() or rms(audio) < self._silence_threshold:
            return []  # too short or too quiet to be meaningful speech

        embedding = get_embedding(audio)
        speaker_label = self.diarizer.add_segment(start, end, embedding)
        overlap_info = getattr(self.diarizer, "last_segment_info", {}) or {}
        is_overlap = bool(overlap_info.get("is_overlap", False))
        candidate_labels = overlap_info.get("candidate_labels", [speaker_label])

        candidate_speakers = []
        valid_candidate_labels = []
        if is_overlap:
            for lab in candidate_labels:
                if lab in self.diarizer.clusters:
                    # Ignore single-sample / noise fragment clusters when multiple clusters exist
                    cluster_embs = self.diarizer.get_embeddings(lab)
                    if len(cluster_embs) < 2 and len(self.diarizer.clusters) > 1:
                        continue
                    name = lab
                else:
                    name = lab

                if name not in candidate_speakers:
                    candidate_speakers.append(name)
                if lab not in valid_candidate_labels:
                    valid_candidate_labels.append(lab)

            if not valid_candidate_labels:
                valid_candidate_labels = [speaker_label]
                candidate_speakers = [speaker_label]

            candidate_labels = valid_candidate_labels

            if len(candidate_speakers) < 2:
                is_overlap = False
                speaker_label = candidate_labels[0]
                stable_embedding = self.diarizer.get_centroid(speaker_label) if speaker_label in self.diarizer.clusters else embedding
                identified_as, confidence = UNKNOWN, 0.0
                display_name = candidate_speakers[0]
                candidate_speakers = [display_name]
            else:
                display_name = " + ".join(candidate_speakers)
                identified_as = display_name
                speaker_label = " + ".join(candidate_labels)
                stable_embedding = embedding
                confidence = 0.5
        else:
            stable_embedding = self.diarizer.get_centroid(speaker_label)
            if stable_embedding is None:
                stable_embedding = embedding
            # Voice profiles do not name anyone. A spoken name is applied
            # after transcription; until then the label stays Speaker_XX.
            identified_as, confidence = UNKNOWN, 0.0
            candidate_speakers = [speaker_label]

        if is_overlap and len(candidate_labels) >= 2:
            attributed_entry = self._attribute_overlap(
                start,
                end,
                audio,
                embedding=embedding,
                candidate_labels=candidate_labels,
                candidate_speakers=candidate_speakers,
                speaker_label=speaker_label,
                display_name=display_name,
                confidence=confidence,
            )
            if attributed_entry:
                return [attributed_entry]

        try:
            pieces = split_on_language_change(audio)
        except Exception as exc:
            # A boundary search that fails must cost nothing but the split —
            # the segment is still perfectly transcribable as one piece.
            logger.warning(f"[{start:.1f}s] language change detection failed: {exc}")
            pieces = [(0, len(audio), None)]

        was_split = len(pieces) > 1

        entries: list[dict] = []
        for piece_start_sample, piece_end_sample, detected_by_lcd in pieces:
            piece_start = start + piece_start_sample / SAMPLE_RATE
            piece_end = start + piece_end_sample / SAMPLE_RATE
            piece_audio = audio[piece_start_sample:piece_end_sample]

            entry = self._transcribe_piece(
                piece_start,
                piece_end,
                piece_audio,
                speaker_label=speaker_label,
                embedding=embedding,
                stable_embedding=stable_embedding,
                identified_as=identified_as,
                confidence=confidence,
                was_split=was_split,
                lcd_language=detected_by_lcd if was_split else None,
                is_overlap=is_overlap,
                candidate_speakers=candidate_speakers,
                candidate_labels=candidate_labels,
            )
            if entry:
                entries.append(entry)

        return entries

    def _attribute_overlap(
        self,
        start: float,
        end: float,
        audio,
        *,
        embedding,
        candidate_labels: list[str],
        candidate_speakers: list[str],
        speaker_label: str,
        display_name: str,
        confidence: float,
    ) -> dict | None:
        """
        Try to resolve WHO said WHAT inside an overlap window via source
        separation, and fold the result into one transcript entry.
        """
        from models.overlap_attribution import resolve_overlap_attribution

        try:
            attributed = resolve_overlap_attribution(
                audio,
                start=start,
                end=end,
                candidate_labels=candidate_labels,
                diarizer=self.diarizer,
                identifier=self.identifier,
                stt_adapter=self.stt_adapter,
                dominant_language=self.dominant_language,
            )
        except Exception as exc:
            logger.warning(f"[{start:.1f}-{end:.1f}s] overlap attribution failed, falling back: {exc}")
            return None

        if not attributed:
            return None

        spans = [
            {**span, "text": clean_text(span.get("text"))}
            for span in attributed[0].get("attributed_spans") or []
        ]
        spans = [span for span in spans if span["text"]]
        if len(spans) < 2:
            return None

        for span in spans:
            span["color"] = self._color_for(span.get("speaker_label") or speaker_label)

        separation_confidence = attributed[0].get("separation_confidence")
        combined_text = " / ".join(f"{s['speaker']}: {s['text']}" for s in spans if s.get("text"))
        language = self.dominant_language

        entry = {
            "start_sec": round(start, 2),
            "end_sec": round(end, 2),
            "time": format_timestamp(start),
            "speaker": display_name,
            "speaker_label": speaker_label,
            "identified_as": display_name,
            "confidence": confidence,
            "color": self._color_for(speaker_label),
            "is_overlap": True,
            "is_separated_overlap": True,
            "separation_confidence": separation_confidence,
            "attributed_spans": spans,
            "candidate_speakers": candidate_speakers,
            "candidate_labels": candidate_labels,
            "language": language,
            "language_name": LANGUAGE_NAMES.get(language, language),
            "language_prob": 1.0,
            "language_detected": language,
            "language_fallback": False,
            "adapter": self.stt_adapter.adapter_type,
            "provider": self.stt_adapter.provider_name,
            "model": self.stt_adapter.model_name,
            "language_margin": 1.0,
            "language_mixed_suspected": False,
            "raw_text": combined_text,
            "text": combined_text,
            "is_turn": False,
        }
        self.transcript.append(entry)
        self._embeddings.append(np.asarray(embedding, dtype=np.float32).copy())
        self._language_counts[language] = self._language_counts.get(language, 0) + 1
        if self._on_transcript_update:
            try:
                self._on_transcript_update()
            except Exception as exc:
                logger.warning(
                    f"[{self.session_id}] transcript update callback failed: {exc}"
                )
        return entry

    def _transcribe_piece(
        self,
        start: float,
        end: float,
        audio,
        *,
        speaker_label: str,
        embedding,
        stable_embedding,
        identified_as: str,
        confidence: float,
        was_split: bool = False,
        lcd_language: str | None = None,
        is_overlap: bool = False,
        candidate_speakers: list[str] | None = None,
        candidate_labels: list[str] | None = None,
    ) -> dict | None:
        """
        Transcribe one language-homogeneous piece and append its entry.
        """
        if len(audio) < _get_min_speech_samples():
            return None

        hint = lcd_language or self.dominant_language

        if self._full_audio is not None and not was_split:
            asr = self.stt_adapter.transcribe_with_context(
                self._full_audio, start, end, hint_language=hint
            )
        else:
            asr = self.stt_adapter.transcribe(audio, hint_language=hint)
        if asr["text"] and has_foreign_script(asr["text"]):
            original = asr["text"]
            asr["text"] = clean_text(original)
            if not asr["text"]:
                logger.info(f"[{start:.1f}-{end:.1f}s] dropping garbled foreign-script text: {original[:60]!r}")
                return None
        if not asr["text"]:
            return None

        text_clean = (asr["text"] or "").strip().lower().rstrip(".!?,")
        duration = end - start
        FILLER_HALLUCINATIONS = {
            "hello", "hi", "thank you", "thanks", "bye", "goodbye",
            "subtitles by", "subtitles", "amara.org", "blank_audio",
            "thank you for watching", "see you next time"
        }
        if duration < 1.8 and text_clean in FILLER_HALLUCINATIONS:
            logger.info(f"[{start:.1f}-{end:.1f}s] dropping STT noise hallucination: {asr['text']!r}")
            return None
        # Guard against hallucinated non-supported languages (e.g. Kannada 'kn') on short/noisy clips
        from config import SUPPORTED_LANGUAGES
        if asr.get("language") not in SUPPORTED_LANGUAGES:
            logger.warning(
                f"[{start:.1f}s] unsupported language '{asr.get('language')}' detected -> "
                f"falling back to dominant language '{self.dominant_language}'"
            )
            asr["language"] = self.dominant_language
            asr["language_name"] = LANGUAGE_NAMES.get(self.dominant_language, self.dominant_language)
            asr["language_fallback"] = True

        self._language_counts[asr["language"]] = self._language_counts.get(asr["language"], 0) + 1

        if identified_as == UNKNOWN and line_can_name_a_speaker(asr):
            hint = extract_self_introduction_name(asr["text"])
            if hint:
                identified_as = hint
                confidence = 1.0
                logger.info(f"[{start:.1f}s] named '{hint}' from self-introduction")

        display_name = identified_as if identified_as != UNKNOWN else speaker_label
        raw_text = asr["text"]
        entry = {
            "start_sec": round(start, 2),
            "end_sec": round(end, 2),
            "time": format_timestamp(start),
            "speaker": display_name,
            "speaker_label": speaker_label,
            "identified_as": identified_as,
            "confidence": confidence,
            "color": self._color_for(speaker_label),
            "is_overlap": is_overlap,
            "candidate_speakers": candidate_speakers or [display_name],
            "candidate_labels": candidate_labels or [speaker_label],
            "language": asr["language"],
            "language_name": asr["language_name"],
            "language_prob": asr["language_prob"],
            "language_detected": asr["language_detected"],
            "language_fallback": asr["language_fallback"],
            "adapter": asr.get("adapter", self.stt_adapter.adapter_type),
            "provider": asr.get("provider", self.stt_adapter.provider_name),
            "model": asr.get("model", self.stt_adapter.model_name),
            "language_margin": asr.get("language_margin", 1.0),
            "language_mixed_suspected": asr.get("language_mixed_suspected", False),
            "raw_text": raw_text,
            "text": raw_text,
            "is_turn": False,
        }
        self.transcript.append(entry)
        # Keep the raw segment embedding (not the cluster centroid) so the
        # offline recluster pass sees each line's own voice fingerprint. Every
        # piece of one sub-segment shares it — same voice, and the recluster
        # pass requires one embedding per transcript line.
        self._embeddings.append(np.asarray(embedding, dtype=np.float32).copy())
        if self._on_transcript_update:
            try:
                self._on_transcript_update()
            except Exception as exc:
                logger.warning(
                    f"[{self.session_id}] transcript update callback failed: {exc}"
                )
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

        Grouped by display name so each speaker name appears exactly once,
        preventing uq_speaker_meeting_name unique constraint violations in DB.
        """
        totals: dict[str, float] = {}
        colors: dict[str, str] = {}
        speaker_labels: dict[str, str] = {}

        for entry in self.transcript:
            dur = max(float(entry.get("end_sec", 0)) - float(entry.get("start_sec", 0)), 0.0)
            if dur <= 0:
                continue

            if entry.get("is_overlap"):
                c_spks = entry.get("candidate_speakers") or []
                clean_spks = [s for s in c_spks if s and "+" not in s and " & " not in s]
                if not clean_spks:
                    clean_spks = [entry.get("speaker") or "Unknown"]
                share_dur = dur / len(clean_spks)
                for spk in clean_spks:
                    colors.setdefault(spk, self._color_for(spk))
                    speaker_labels.setdefault(spk, spk)
                    totals[spk] = totals.get(spk, 0.0) + share_dur
            else:
                display = entry.get("speaker") or entry.get("speaker_label") or "Unknown"
                if "+" in display or " & " in display:
                    parts = [p.strip() for p in re.split(r"\s*(?:\+|\&)\s*", display) if p.strip()]
                    share_dur = dur / max(len(parts), 1)
                    for spk in parts:
                        colors.setdefault(spk, self._color_for(spk))
                        speaker_labels.setdefault(spk, spk)
                        totals[spk] = totals.get(spk, 0.0) + share_dur
                else:
                    colors.setdefault(display, entry.get("color") or self._color_for(display))
                    speaker_labels.setdefault(display, entry.get("speaker_label") or display)
                    totals[display] = totals.get(display, 0.0) + dur

        named_speakers = [spk for spk in totals if not _is_generic(spk)]
        if named_speakers:
            filtered_totals = {}
            for spk, secs in totals.items():
                if _is_generic(spk) and secs < 5.0:
                    continue
                filtered_totals[spk] = secs
            if filtered_totals:
                totals = filtered_totals

        grand_total = sum(totals.values())
        if grand_total <= 0:
            return []

        return [
            {
                "name": display_name,
                "speaker_label": speaker_labels.get(display_name, display_name),
                "seconds": round(seconds, 1),
                "time": format_duration(seconds),
                "pct": round(seconds / grand_total * 100, 1),
                "color": colors.get(display_name, SPEAKER_COLORS[0]),
            }
            for display_name, seconds in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
        ]

    def participant_count(self) -> int:
        """
        Dynamic speaker count — ignore tiny fragment clusters (< 3s or < 2%
        talk time) that are usually false splits from one voice.
        """
        stats = self.speaker_stats()
        if not stats:
            return 0
        total = sum(s["seconds"] for s in stats)
        significant = [
            s for s in stats
            if s["seconds"] >= 3.0 or (total > 0 and s["seconds"] / total >= 0.02)
        ]
        return max(len(significant), 1) if stats else 0

    def language_breakdown(self) -> list[dict]:
        """
        Which of the three languages this meeting was actually spoken in.

        Counted over the pre-merge ASR segments, NOT the merged turns. A turn
        can hold more than one language — someone answers in English and
        finishes in Marathi — and a turn carries a single dominant label.
        Counting turns therefore charges the whole turn's duration to that one
        label and the minority language vanishes from the breakdown entirely,
        even though its text is right there in the transcript.

        `raw_segments` is only populated once build_and_apply_turns has run;
        before that `transcript` still holds the unmerged segments, so the
        fallback is the same data either way and mid-processing calls (the
        upload progress sync) stay correct.
        """
        source = self.raw_segments or self.transcript

        totals: dict[str, float] = {}
        for entry in source:
            dur = max(float(entry.get("end_sec", 0)) - float(entry.get("start_sec", 0)), 0.0)
            if dur <= 0:
                continue

            primary_code = entry.get("language")
            if not primary_code:
                continue

            text = entry.get("raw_text") or entry.get("text") or ""
            dev_count = sum(1 for ch in text if "\u0900" <= ch <= "\u097f")
            lat_count = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))

            if dev_count > 0 and lat_count > 0:
                total_letters = dev_count + lat_count
                dev_ratio = dev_count / total_letters
                lat_ratio = lat_count / total_letters
                # A Marathi sentence that names "Pune" or "SI" is still
                # Marathi. Split only when both scripts are a real share
                # of the line.
                if dev_ratio >= 0.2 and lat_ratio >= 0.2:
                    indic_code = primary_code if primary_code in ("hi", "mr") else "mr"
                    totals[indic_code] = totals.get(indic_code, 0.0) + (dur * dev_ratio)
                    totals["en"] = totals.get("en", 0.0) + (dur * lat_ratio)
                elif dev_ratio >= lat_ratio:
                    indic_code = primary_code if primary_code in ("hi", "mr") else "mr"
                    totals[indic_code] = totals.get(indic_code, 0.0) + dur
                else:
                    totals["en"] = totals.get("en", 0.0) + dur
            else:
                totals[primary_code] = totals.get(primary_code, 0.0) + dur

        grand_total = sum(totals.values())
        if grand_total <= 0:
            return []

        rows = [
            {
                "code": code,
                "name": LANGUAGE_NAMES.get(code, code),
                "seconds": round(seconds, 1),
                "pct": round(seconds / grand_total * 100, 1),
            }
            for code, seconds in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
            if seconds > 0.05
        ]
        # Drop a language that is only a brief mislabel. Keep it when it is
        # at least 10% of the speech, and always keep the main language.
        spoken = [row for row in rows if row["pct"] >= 10]
        return spoken or rows[:1]

    def spoken_duration(self) -> float:
        """Total seconds of actual speech (not wall-clock file length)."""
        return sum(e["end_sec"] - e["start_sec"] for e in self.transcript)

    def full_transcript(self) -> dict:
        return {
            "session_id": self.session_id,
            "transcript": self.transcript,
            "raw_segments": self.raw_segments,
            "speaker_stats": self.speaker_stats(),
            "languages": self.language_breakdown(),
            "speaking_time_seconds": {s["name"]: s["seconds"] for s in self.speaker_stats()},
        }
