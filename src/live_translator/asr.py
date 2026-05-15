"""ASR + translation wrapper around whisper.cpp's `whisper-cli` binary.

Translates Spanish audio to English text. Whisper natively supports translation
to English from any of its 99 source languages via the --translate flag.

Audio input must be 16 kHz mono 16-bit PCM WAV. Use prepare_wav() to convert
arbitrary WAVs (or pass numpy arrays via transcribe_pcm()).
"""

from __future__ import annotations

import json
import struct
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import typer

from live_translator.paths import (
    DEFAULT_WHISPER_MODEL,
    whisper_cli_path,
    whisper_model_path,
)

SAMPLE_RATE = 16000


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration_s: float
    elapsed_s: float

    @property
    def real_time_factor(self) -> float:
        return self.duration_s / self.elapsed_s if self.elapsed_s > 0 else 0.0


def _setup_command(arg: str = "") -> str:
    """Return the platform-appropriate setup command (mac/linux/windows)."""
    import platform
    system = platform.system()
    suffix = f" {arg}" if arg else ""
    if system == "Darwin":
        return f"bash scripts/setup_mac.sh{suffix}"
    if system == "Windows":
        return f"powershell -ExecutionPolicy Bypass -File scripts\\setup_windows.ps1{suffix}"
    return f"bash scripts/setup_rpi.sh{suffix}"


def _ensure_binaries(model: str) -> Path:
    cli = whisper_cli_path()
    if not cli.exists():
        raise FileNotFoundError(
            f"whisper-cli not found at {cli}. Run: {_setup_command()}"
        )
    model_path = whisper_model_path(model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Whisper model not found at {model_path}. Run: {_setup_command(model)}"
        )
    return model_path


def transcribe_file(
    wav_path: Path | str,
    model: str = DEFAULT_WHISPER_MODEL,
    source_language: str = "es",
    threads: int | None = None,
) -> TranscriptionResult:
    """Transcribe + translate a 16kHz mono WAV file to English text."""
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)

    model_path = _ensure_binaries(model)

    with tempfile.TemporaryDirectory() as tmp:
        out_prefix = Path(tmp) / "out"
        cmd = [
            str(whisper_cli_path()),
            "-m", str(model_path),
            "-f", str(wav_path),
            "-l", source_language,
            "--translate",
            "--output-json",
            "--no-prints",
            "-of", str(out_prefix),
        ]
        if threads is not None:
            cmd.extend(["-t", str(threads)])

        import time
        start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.perf_counter() - start

        if proc.returncode != 0:
            raise RuntimeError(
                f"whisper-cli failed (code {proc.returncode}):\n"
                f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
            )

        json_file = out_prefix.with_suffix(".json")
        if not json_file.exists():
            raise RuntimeError(f"whisper-cli produced no JSON output. STDERR: {proc.stderr}")

        data = json.loads(json_file.read_text())

    segments = data.get("transcription", [])
    text = " ".join(seg.get("text", "").strip() for seg in segments).strip()
    duration = _wav_duration_seconds(wav_path)
    detected_lang = data.get("result", {}).get("language", source_language)

    return TranscriptionResult(
        text=text,
        language=detected_lang,
        duration_s=duration,
        elapsed_s=elapsed,
    )


def transcribe_pcm(
    pcm: np.ndarray,
    model: str = DEFAULT_WHISPER_MODEL,
    source_language: str = "es",
    threads: int | None = None,
) -> TranscriptionResult:
    """Transcribe + translate raw float32 PCM at 16 kHz mono.

    Used by the live pipeline once VAD has emitted a speech segment.
    """
    if pcm.ndim != 1:
        raise ValueError(f"Expected mono 1-D PCM, got shape {pcm.shape}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _write_wav(tmp_path, pcm, SAMPLE_RATE)
        return transcribe_file(tmp_path, model=model, source_language=source_language, threads=threads)
    finally:
        tmp_path.unlink(missing_ok=True)


def _write_wav(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    pcm_clipped = np.clip(pcm, -1.0, 1.0)
    pcm_int16 = (pcm_clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / rate if rate else 0.0


# --- CLI ---

cli = typer.Typer(add_completion=False, help="ASR + ES→EN translation via whisper.cpp")


@cli.command()
def translate(
    wav: Path = typer.Argument(..., help="Path to a WAV file (any sample rate, will be converted)."),
    model: str = typer.Option(DEFAULT_WHISPER_MODEL, "--model", "-m", help="tiny|base|small|medium"),
    language: str = typer.Option("es", "--language", "-l", help="Source language code"),
):
    """Translate a Spanish (or other-language) WAV to English text."""
    result = transcribe_file(wav, model=model, source_language=language)
    typer.echo(result.text)
    typer.echo(
        f"\n[{result.elapsed_s:.2f}s elapsed, {result.duration_s:.2f}s audio, "
        f"RTF {result.real_time_factor:.2f}x, lang={result.language}]",
        err=True,
    )


if __name__ == "__main__":
    cli()
