"""
Streaming Voice Activity Detection using Silero VAD.

Replaces fixed 3-second windows with real speech-boundary detection:
a segment starts when someone actually begins talking and ends when they
pause — instead of chopping audio at an arbitrary fixed interval regardless
of what's being said. This directly fixes the root cause behind the
diarization fragmentation and identification misses seen earlier: fixed
windows were cutting sentences mid-word, producing noisy embeddings.
"""

import numpy as np
import torch

from config import SAMPLE_RATE, MAX_SEGMENT_SECONDS, MIN_SILENCE_MS

SILERO_FRAME_SAMPLES = 512  # required frame size for 16kHz audio, per Silero VAD

_model = None
_VADIterator = None


def _load():
    global _model, _VADIterator
    if _model is None:
        _model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        # utils = (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
        _VADIterator = utils[3]
    return _model, _VADIterator


class SpeechSegmenter:
    """
    Feed it raw PCM as it arrives (any chunk size); it emits complete speech
    segments bounded by real speech start/pause events, rather than fixed
    windows. One instance per live meeting session.
    """

    def __init__(self, max_segment_seconds: float = MAX_SEGMENT_SECONDS, min_silence_ms: int = MIN_SILENCE_MS):
        model, vad_iterator_cls = _load()
        self.iterator = vad_iterator_cls(
            model,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=min_silence_ms,
        )
        self.max_segment_samples = int(max_segment_seconds * SAMPLE_RATE)
        self._frame_buffer = np.array([], dtype=np.float32)
        self._segment_audio: list[np.ndarray] = []
        self._in_speech = False
        self._elapsed_samples = 0
        self._segment_start_sample = 0

    def process(self, audio: np.ndarray) -> list[dict]:
        """
        audio: newly arrived float32 PCM, 16kHz mono, any length.
        Returns completed segments: [{"start": sec, "end": sec, "audio": np.ndarray}, ...]
        """
        completed = []
        self._frame_buffer = np.concatenate([self._frame_buffer, audio])

        while len(self._frame_buffer) >= SILERO_FRAME_SAMPLES:
            frame = self._frame_buffer[:SILERO_FRAME_SAMPLES]
            self._frame_buffer = self._frame_buffer[SILERO_FRAME_SAMPLES:]

            event = self.iterator(torch.from_numpy(frame), return_seconds=False)

            if self._in_speech:
                self._segment_audio.append(frame)

            if event and "start" in event and not self._in_speech:
                self._in_speech = True
                self._segment_start_sample = self._elapsed_samples
                self._segment_audio = [frame]

            elif event and "end" in event and self._in_speech:
                completed.append(self._flush_segment())

            elif (
                self._in_speech
                and sum(len(f) for f in self._segment_audio) >= self.max_segment_samples
            ):
                # Force-cut very long uninterrupted speech so latency stays
                # bounded — otherwise one long monologue would never produce
                # a transcript entry until the speaker finally paused.
                completed.append(self._flush_segment())
                self._in_speech = True
                self._segment_start_sample = self._elapsed_samples
                self._segment_audio = []

            self._elapsed_samples += SILERO_FRAME_SAMPLES

        return completed

    def flush(self) -> list[dict]:
        """
        End of stream: emit whatever speech is still buffered.

        A live meeting always ends mid-something — the last speaker stops and
        the socket closes before Silero ever fires its "end" event, because
        that event needs MIN_SILENCE_MS of trailing silence that never
        arrives. Same for the final segment of an uploaded file. Without
        this, the last thing said in every meeting is silently dropped.
        """
        completed = []
        if self._in_speech and self._segment_audio:
            completed.append(self._flush_segment())
        self._frame_buffer = np.array([], dtype=np.float32)
        self.iterator.reset_states()
        return completed

    def _flush_segment(self) -> dict:
        audio = (
            np.concatenate(self._segment_audio)
            if self._segment_audio
            else np.array([], dtype=np.float32)
        )
        start_sec = self._segment_start_sample / SAMPLE_RATE
        end_sec = self._elapsed_samples / SAMPLE_RATE
        self._in_speech = False
        self._segment_audio = []
        return {"start": start_sec, "end": end_sec, "audio": audio}
