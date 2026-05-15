"""Voice Activity Detection using Silero VAD.

Silero VAD takes 512-sample chunks at 16 kHz (32 ms each) and returns a
speech-probability scalar. We wrap it with a simple state machine that
emits speech segments when silence persists after speech.

Uses the official `silero-vad` PyPI package, which bundles the model and
the correct inference logic. Earlier versions of this file tried to invoke
a raw ONNX file directly, which silently produced constant probabilities
on a non-matching model layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

VAD_SAMPLE_RATE = 16000
VAD_FRAME_SAMPLES = 512  # required by Silero v5
VAD_FRAME_MS = VAD_FRAME_SAMPLES * 1000 // VAD_SAMPLE_RATE  # 32 ms


@dataclass
class SpeechSegment:
    pcm: np.ndarray  # float32 mono at 16 kHz
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / VAD_SAMPLE_RATE


class SileroVAD:
    """Stateful Silero VAD wrapper. Feed 512-sample frames; get speech probabilities."""

    def __init__(self, threshold: float = 0.5):
        try:
            from silero_vad import load_silero_vad
        except ImportError as exc:
            import platform
            system = platform.system()
            if system == "Darwin":
                setup_cmd = "bash scripts/setup_mac.sh"
            elif system == "Windows":
                setup_cmd = "powershell -ExecutionPolicy Bypass -File scripts\\setup_windows.ps1"
            else:
                setup_cmd = "bash scripts/setup_rpi.sh"
            raise ImportError(
                "silero-vad package is required. Install with: pip install silero-vad\n"
                f"(The setup script does this for you — re-run: {setup_cmd}.)"
            ) from exc
        # onnx=True avoids loading the full PyTorch model graph; CPU-only, ~2MB.
        self._model = load_silero_vad(onnx=True)
        self.threshold = threshold

    def reset(self) -> None:
        self._model.reset_states()

    def speech_prob(self, frame: np.ndarray) -> float:
        if frame.shape != (VAD_FRAME_SAMPLES,):
            raise ValueError(
                f"Silero needs frames of shape ({VAD_FRAME_SAMPLES},), got {frame.shape}"
            )
        tensor = torch.from_numpy(frame.astype(np.float32))
        return float(self._model(tensor, VAD_SAMPLE_RATE).item())


class SegmentBuilder:
    """Turns a stream of 512-sample frames into discrete speech segments.

    State machine:
      idle ─speech─> active ─silence_ms_hangover─> idle (emit segment)

    A small pre-roll of frames before the first detected speech is included so
    the first phoneme isn't clipped.
    """

    def __init__(
        self,
        vad: SileroVAD | None = None,
        *,
        min_speech_ms: int = 250,
        silence_ms_to_end: int = 700,
        pre_roll_ms: int = 200,
        max_segment_ms: int = 15000,
    ):
        self.vad = vad or SileroVAD()
        self.min_speech_frames = max(1, min_speech_ms // VAD_FRAME_MS)
        self.silence_frames_to_end = max(1, silence_ms_to_end // VAD_FRAME_MS)
        self.pre_roll_frames = max(0, pre_roll_ms // VAD_FRAME_MS)
        self.max_segment_frames = max(1, max_segment_ms // VAD_FRAME_MS)

        self._pre_roll: list[np.ndarray] = []
        self._buffer: list[np.ndarray] = []
        self._is_speaking = False
        self._speech_frames = 0
        self._silence_run = 0
        self._frame_idx = 0
        self._segment_start_frame = 0

        # Observable state (read by UIs/dashboards)
        self.last_speech_prob: float = 0.0

    @property
    def is_speaking(self) -> bool:
        """True while inside an active speech segment."""
        return self._is_speaking

    def push(self, frame: np.ndarray) -> SpeechSegment | None:
        """Feed one 512-sample frame. Returns a SpeechSegment when one ends, else None."""
        prob = self.vad.speech_prob(frame)
        self.last_speech_prob = prob
        is_speech = prob >= self.vad.threshold

        if not self._is_speaking:
            # idle: keep a rolling pre-roll of recent frames
            self._pre_roll.append(frame)
            if len(self._pre_roll) > self.pre_roll_frames:
                self._pre_roll.pop(0)

            if is_speech:
                self._is_speaking = True
                self._segment_start_frame = self._frame_idx - len(self._pre_roll)
                self._buffer = list(self._pre_roll)
                self._buffer.append(frame)
                self._pre_roll.clear()
                self._speech_frames = 1
                self._silence_run = 0
        else:
            # active: accumulate, watch for trailing silence or max length
            self._buffer.append(frame)
            if is_speech:
                self._speech_frames += 1
                self._silence_run = 0
            else:
                self._silence_run += 1

            should_end = (
                self._silence_run >= self.silence_frames_to_end
                or len(self._buffer) >= self.max_segment_frames
            )
            if should_end:
                segment = self._flush()
                self._frame_idx += 1
                return segment

        self._frame_idx += 1
        return None

    def _flush(self) -> SpeechSegment | None:
        if self._speech_frames < self.min_speech_frames:
            self._reset_active()
            return None

        pcm = np.concatenate(self._buffer)
        start_s = self._segment_start_frame * VAD_FRAME_SAMPLES / VAD_SAMPLE_RATE
        end_s = start_s + len(pcm) / VAD_SAMPLE_RATE
        self._reset_active()
        return SpeechSegment(pcm=pcm, start_s=start_s, end_s=end_s)

    def _reset_active(self) -> None:
        self._buffer = []
        self._is_speaking = False
        self._speech_frames = 0
        self._silence_run = 0
        self._pre_roll.clear()
