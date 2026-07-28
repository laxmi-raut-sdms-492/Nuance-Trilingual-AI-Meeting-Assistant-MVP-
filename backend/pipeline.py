"""
One MeetingSession instance = one live meeting.

Flow per audio block received from the frontend:

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
           - transcription of just this sub-segment (models/asr.py)
           - one transcript entry, sent back to the frontend immediately
"""

import logging
import time

from config import SAMPLE_RATE, SILENCE_RMS_THRESHOLD, MIN_SPEECH_SECONDS
from audio_utils import pcm16_bytes_to_float32, rms
from models.embedding import get_embedding
from models.diarizer import SessionDiarizer
from models.vad import SpeechSegmenter
from models.scd import split_on_speaker_change
from models.asr import transcribe
from models.identifier import SpeakerIdentifier

logger = logging.getLogger("pipeline")
MIN_SPEECH_SAMPLES = int(MIN_SPEECH_SECONDS * SAMPLE_RATE)


class MeetingSession:
    def __init__(self, session_id: str, identifier: SpeakerIdentifier):
        self.session_id = session_id
        self.identifier = identifier
        self.diarizer = SessionDiarizer()
        self.segmenter = SpeechSegmenter()
        self.transcript: list[dict] = []
        self.created_at = time.time()

    def process_chunk(self, pcm_bytes: bytes) -> list[dict]:
        """Feed in raw PCM16 mono 16kHz bytes. Returns any new transcript entries."""
        audio = pcm16_bytes_to_float32(pcm_bytes)
        vad_segments = self.segmenter.process(audio)
        new_entries = []

        for vad_seg in vad_segments:
            seg_audio = vad_seg["audio"]
            seg_start = vad_seg["start"]

            try:
                # A VAD segment can still contain a speaker change (back-to-back
                # talkers, no pause) — split into homogeneous pieces first.
                sub_bounds = split_on_speaker_change(seg_audio)
            except Exception as e:
                logger.error(f"[{seg_start:.1f}s] SCD failed, falling back to whole segment: {e}")
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

        text = transcribe(audio)
        if not text:
            return None

        entry = {
            "start_sec": round(start, 2),
            "end_sec": round(end, 2),
            "speaker_label": speaker_label,
            "identified_as": identified_as,
            "confidence": confidence,
            "text": text,
        }
        self.transcript.append(entry)
        return entry

    def full_transcript(self) -> dict:
        speaker_totals: dict[str, float] = {}
        for entry in self.transcript:
            key = entry["identified_as"] if entry["identified_as"] != "Unknown" else entry["speaker_label"]
            duration = entry["end_sec"] - entry["start_sec"]
            speaker_totals[key] = speaker_totals.get(key, 0.0) + duration

        return {
            "session_id": self.session_id,
            "transcript": self.transcript,
            "speaking_time_seconds": {k: round(v, 1) for k, v in speaker_totals.items()},
        }

