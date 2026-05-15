"""Text-to-speech wrapper around Piper.

Synthesizes English text into PCM audio using a downloaded Piper voice
(default: en_US-lessac-medium). Returns float32 mono PCM at the voice's
native sample rate (typically 22050 Hz for medium voices).
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import typer

from live_translator.paths import DEFAULT_PIPER_VOICE, piper_voice_paths


@dataclass
class SynthesisResult:
    pcm: np.ndarray  # float32 mono in [-1, 1]
    sample_rate: int
    elapsed_s: float

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / self.sample_rate if self.sample_rate else 0.0

    @property
    def real_time_factor(self) -> float:
        return self.duration_s / self.elapsed_s if self.elapsed_s > 0 else 0.0


@lru_cache(maxsize=2)
def _load_voice(voice: str):
    """Lazy-load and cache a Piper voice."""
    from piper import PiperVoice  # imported here to keep CLI startup fast

    import platform
    system = platform.system()
    if system == "Darwin":
        setup_cmd = "bash scripts/setup_mac.sh"
    elif system == "Windows":
        setup_cmd = "powershell -ExecutionPolicy Bypass -File scripts\\setup_windows.ps1"
    else:
        setup_cmd = "bash scripts/setup_rpi.sh"
    onnx, config = piper_voice_paths(voice)
    if not onnx.exists() or not config.exists():
        raise FileNotFoundError(
            f"Piper voice files missing for '{voice}':\n"
            f"  {onnx}\n  {config}\n"
            f"Run: {setup_cmd}"
        )
    return PiperVoice.load(str(onnx), config_path=str(config))


def synthesize(text: str, voice: str = DEFAULT_PIPER_VOICE) -> SynthesisResult:
    """Synthesize English text to mono float32 PCM."""
    text = text.strip()
    if not text:
        return SynthesisResult(pcm=np.zeros(0, dtype=np.float32), sample_rate=22050, elapsed_s=0.0)

    voice_obj = _load_voice(voice)

    import time
    start = time.perf_counter()

    # Piper exposes a streaming API: synthesize() yields AudioChunk objects with
    # .audio_int16_bytes (raw int16 PCM) and .sample_rate. We accumulate to one buffer.
    int16_chunks: list[bytes] = []
    sample_rate: int | None = None
    for chunk in voice_obj.synthesize(text):
        if sample_rate is None:
            sample_rate = chunk.sample_rate
        int16_chunks.append(chunk.audio_int16_bytes)

    elapsed = time.perf_counter() - start

    if not int16_chunks or sample_rate is None:
        return SynthesisResult(pcm=np.zeros(0, dtype=np.float32), sample_rate=22050, elapsed_s=elapsed)

    raw = b"".join(int16_chunks)
    pcm_int16 = np.frombuffer(raw, dtype=np.int16)
    pcm_f32 = pcm_int16.astype(np.float32) / 32768.0

    return SynthesisResult(pcm=pcm_f32, sample_rate=sample_rate, elapsed_s=elapsed)


def synthesize_to_wav(text: str, out_path: Path | str, voice: str = DEFAULT_PIPER_VOICE) -> Path:
    """Synthesize and write a 16-bit PCM WAV file."""
    out_path = Path(out_path)
    result = synthesize(text, voice=voice)
    pcm_int16 = (np.clip(result.pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(result.sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return out_path


# --- CLI ---

cli = typer.Typer(add_completion=False, help="Piper TTS for English output")


@cli.command()
def speak(
    text: str = typer.Argument(..., help="English text to synthesize."),
    out: Path | None = typer.Option(None, "--out", "-o", help="If set, write WAV here instead of playing."),
    voice: str = typer.Option(DEFAULT_PIPER_VOICE, "--voice", "-v"),
    play: bool = typer.Option(True, "--play/--no-play", help="Play through default speaker."),
):
    """Synthesize English text and either play it or save to WAV."""
    result = synthesize(text, voice=voice)
    typer.echo(
        f"[Synthesized {result.duration_s:.2f}s of audio in {result.elapsed_s:.2f}s "
        f"(RTF {result.real_time_factor:.1f}x, sr={result.sample_rate})]",
        err=True,
    )

    if out is not None:
        synthesize_to_wav(text, out, voice=voice)
        typer.echo(f"Wrote {out}", err=True)
        return

    if play:
        import sounddevice as sd
        sd.play(result.pcm, samplerate=result.sample_rate, blocking=True)


if __name__ == "__main__":
    cli()
