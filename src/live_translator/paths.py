"""Centralized paths for binaries and model files."""

import platform
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
VENDOR_DIR = REPO_ROOT / "vendor"

WHISPER_CPP_DIR = VENDOR_DIR / "whisper.cpp"


def _resolve_whisper_cli() -> Path:
    """Locate the whisper-cli binary across build layouts.

    - macOS / Linux:    build/bin/whisper-cli
    - Windows (MSVC):   build/bin/Release/whisper-cli.exe    (multi-config generator)
    - Windows (Ninja):  build/bin/whisper-cli.exe            (single-config generator)
    """
    bin_dir = WHISPER_CPP_DIR / "build" / "bin"
    if platform.system() == "Windows":
        candidates = [
            bin_dir / "Release" / "whisper-cli.exe",
            bin_dir / "whisper-cli.exe",
        ]
    else:
        candidates = [bin_dir / "whisper-cli"]
    for c in candidates:
        if c.exists():
            return c
    # Fall back to the first candidate so callers get a readable "missing" path.
    return candidates[0]


def whisper_cli_path() -> Path:
    """Re-resolve at call-time so a build that happens after import is still picked up."""
    return _resolve_whisper_cli()


# Eagerly resolved value, kept for backward compatibility with existing imports.
WHISPER_CLI = _resolve_whisper_cli()

WHISPER_MODELS_DIR = MODELS_DIR / "whisper"
VAD_MODELS_DIR = MODELS_DIR / "vad"
PIPER_MODELS_DIR = MODELS_DIR / "piper"

WHISPER_MODEL_FILES = {
    "tiny": "ggml-tiny.bin",
    "base": "ggml-base.bin",
    "small": "ggml-small.bin",
    "medium": "ggml-medium.bin",
}

DEFAULT_WHISPER_MODEL = "small"
DEFAULT_PIPER_VOICE = "en_US-lessac-medium"


def whisper_model_path(name: str) -> Path:
    if name not in WHISPER_MODEL_FILES:
        raise ValueError(f"Unknown whisper model '{name}'. Choose from {list(WHISPER_MODEL_FILES)}")
    return WHISPER_MODELS_DIR / WHISPER_MODEL_FILES[name]


def piper_voice_paths(voice: str = DEFAULT_PIPER_VOICE) -> tuple[Path, Path]:
    onnx = PIPER_MODELS_DIR / f"{voice}.onnx"
    config = PIPER_MODELS_DIR / f"{voice}.onnx.json"
    return onnx, config


def silero_vad_path() -> Path:
    return VAD_MODELS_DIR / "silero_vad.onnx"
