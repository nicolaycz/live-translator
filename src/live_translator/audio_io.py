"""Microphone capture, speaker playback and device discovery via sounddevice."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from live_translator.vad import VAD_FRAME_SAMPLES, VAD_SAMPLE_RATE


@dataclass
class DeviceInfo:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    is_default_input: bool = False
    is_default_output: bool = False

    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        return self.max_output_channels > 0


def list_devices() -> list[DeviceInfo]:
    """Return all audio devices with default-input/output flags set."""
    raw = sd.query_devices()
    default_in, default_out = sd.default.device  # tuple (in_idx, out_idx)
    out: list[DeviceInfo] = []
    for idx, d in enumerate(raw):
        out.append(
            DeviceInfo(
                index=idx,
                name=d["name"],
                max_input_channels=d["max_input_channels"],
                max_output_channels=d["max_output_channels"],
                default_samplerate=d["default_samplerate"],
                is_default_input=(idx == default_in),
                is_default_output=(idx == default_out),
            )
        )
    return out


def resolve_device(spec: str | int | None, *, kind: str) -> int | None:
    """Resolve a device spec to an integer index.

    spec may be:
      - None  → returns None (system default)
      - int   → used as-is
      - str   → if numeric, parsed as int; else case-insensitive substring match
                against device names. Must match exactly one device of the given
                kind ("input" or "output").
    """
    if spec is None:
        return None
    if isinstance(spec, int):
        return spec
    spec = spec.strip()
    if spec.isdigit() or (spec.startswith("-") and spec[1:].isdigit()):
        return int(spec)

    needle = spec.lower()
    candidates = [
        d for d in list_devices()
        if (kind == "input" and d.is_input) or (kind == "output" and d.is_output)
    ]
    matches = [d for d in candidates if needle in d.name.lower()]
    if not matches:
        names = ", ".join(f"[{d.index}] {d.name}" for d in candidates)
        raise ValueError(f"No {kind} device matches '{spec}'. Available: {names}")
    if len(matches) > 1:
        names = ", ".join(f"[{d.index}] {d.name}" for d in matches)
        raise ValueError(f"Ambiguous {kind} device '{spec}', matches: {names}")
    return matches[0].index


def device_name(index: int | None, *, kind: str) -> str:
    """Pretty name for display: '[idx] Name' or '(system default)'."""
    if index is None:
        for d in list_devices():
            if (kind == "input" and d.is_default_input) or (kind == "output" and d.is_default_output):
                return f"[{d.index}] {d.name} (default)"
        return "(system default)"
    for d in list_devices():
        if d.index == index:
            return f"[{d.index}] {d.name}"
    return f"[{index}] (unknown)"


def _resample_linear(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Cheap linear-interpolation resampler. Good enough for VAD/Whisper input.

    For high-quality audio you'd use scipy.signal.resample_poly, but linear
    interpolation is fine here since Whisper's mel-spectrogram smooths things
    out and we only ever go from 24/44.1/48 kHz down to 16 kHz.
    """
    if src_sr == dst_sr or pcm.size == 0:
        return pcm
    n_src = pcm.shape[0]
    n_dst = max(1, int(round(n_src * dst_sr / src_sr)))
    src_idx = np.linspace(0, n_src - 1, n_dst, dtype=np.float64)
    left = np.floor(src_idx).astype(np.int64)
    right = np.minimum(left + 1, n_src - 1)
    frac = (src_idx - left).astype(np.float32)
    return (pcm[left] * (1.0 - frac) + pcm[right] * frac).astype(np.float32)


@contextmanager
def microphone_frames(
    sample_rate: int = VAD_SAMPLE_RATE,
    frame_samples: int = VAD_FRAME_SAMPLES,
    device: int | str | None = None,
    on_status: callable | None = None,
) -> Iterator[Iterator[np.ndarray]]:
    """Open a mic stream and yield an iterator of mono float32 frames at `sample_rate`.

    Strategy: try to open the device at the requested sample_rate first. If the
    device rejects it (common with Bluetooth headsets like AirPods), fall back
    to the device's native default sample rate and resample in software. This
    makes the pipeline robust across mic types.

    Parameters
    ----------
    sample_rate
        Target sample rate produced to the consumer (16 kHz for our VAD/Whisper).
    frame_samples
        Target frame size in samples *at the target sample rate* (512 = 32ms @ 16kHz).
    device
        Integer device index, substring of the device name (e.g. "AirPods"),
        or None for the system default input.
    on_status
        Optional callback invoked when sounddevice reports a stream status
        (e.g. input overflow). Receives the status object.

    Usage:
        with microphone_frames(device="AirPods") as frames:
            for frame in frames:
                vad.push(frame)
    """
    device_idx = resolve_device(device, kind="input")

    # Decide native vs. requested sample rate.
    devices = list_devices()
    dev_info = next((d for d in devices if d.index == (device_idx if device_idx is not None
                                                       else next(x.index for x in devices if x.is_default_input))),
                    None)
    native_sr = int(dev_info.default_samplerate) if dev_info else sample_rate

    # Try the requested rate first; if portaudio rejects, fall back to native.
    try:
        sd.check_input_settings(device=device_idx, samplerate=sample_rate, channels=1, dtype="float32")
        stream_sr = sample_rate
    except Exception:
        stream_sr = native_sr

    needs_resample = stream_sr != sample_rate
    blocksize = int(round(frame_samples * stream_sr / sample_rate)) if needs_resample else frame_samples

    q: queue.Queue[np.ndarray] = queue.Queue()
    pending = np.zeros(0, dtype=np.float32)  # carry-over for resample alignment
    pending_lock = threading.Lock()

    def _callback(indata, frames_count, time_info, status):  # noqa: ARG001
        nonlocal pending
        if status and on_status is not None:
            on_status(status)
        mono = indata[:, 0].copy()
        if needs_resample:
            mono = _resample_linear(mono, stream_sr, sample_rate)
        with pending_lock:
            pending = np.concatenate([pending, mono])
            while pending.shape[0] >= frame_samples:
                q.put(pending[:frame_samples].copy())
                pending = pending[frame_samples:]

    try:
        stream = sd.InputStream(
            samplerate=stream_sr,
            blocksize=blocksize,
            channels=1,
            dtype="float32",
            callback=_callback,
            device=device_idx,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not open input device [{device_idx}] at {stream_sr} Hz: {exc}\n"
            f"Try a different device with --input-device, or run "
            f"`live-translator devices` to see what's available."
        ) from exc

    def _frame_iter() -> Iterator[np.ndarray]:
        while True:
            yield q.get()

    with stream:
        yield _frame_iter()


def play_pcm(
    pcm: np.ndarray,
    sample_rate: int,
    blocking: bool = True,
    device: int | str | None = None,
) -> None:
    """Play a mono float32 PCM array through the chosen output device."""
    if pcm.size == 0:
        return
    device_idx = resolve_device(device, kind="output")
    sd.play(pcm, samplerate=sample_rate, blocking=blocking, device=device_idx)


def rms_dbfs(pcm: np.ndarray) -> float:
    """Root-mean-square of mono PCM in dBFS. Returns -120 for silence."""
    if pcm.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(pcm.astype(np.float64))) + 1e-12))
    if rms <= 1e-6:
        return -120.0
    return 20.0 * float(np.log10(rms))
