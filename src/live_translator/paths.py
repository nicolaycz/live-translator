"""Centralized paths for binaries and model files."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
VENDOR_DIR = REPO_ROOT / "vendor"

WHISPER_CPP_DIR = VENDOR_DIR / "whisper.cpp"
WHISPER_CLI = WHISPER_CPP_DIR / "build" / "bin" / "whisper-cli"

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
