#!/usr/bin/env bash
# Setup script for macOS (Apple Silicon).
#
# Everything Python lives inside .venv/ at the repo root — your system Python
# is NOT touched. To use the installed package later, run:
#     source .venv/bin/activate
#
# Steps performed:
#   1. Create an isolated Python virtualenv at .venv/
#   2. Install Python deps into that .venv/ via pip install -e .
#   3. Compile whisper.cpp with Metal support
#   4. Download Whisper, Silero VAD and Piper models into models/
#
# Usage: bash scripts/setup_mac.sh [whisper_model]
#   whisper_model: tiny | base | small | medium  (default: small)

set -euo pipefail

WHISPER_MODEL="${1:-small}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/vendor"
MODELS_DIR="$REPO_ROOT/models"
VENV_DIR="$REPO_ROOT/.venv"
WHISPER_DIR="$VENDOR_DIR/whisper.cpp"
PIPER_VOICE="en_US-lessac-medium"

# Pick a Python interpreter: prefer python3.11 / python3.12, fall back to python3
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done

echo "==> live-translator setup (model: $WHISPER_MODEL)"

# --- Sanity ---
if [[ "$(uname)" != "Darwin" ]]; then
  echo "This script targets macOS."
  echo "  Raspberry Pi: bash scripts/setup_rpi.sh"
  echo "  Windows:      powershell -ExecutionPolicy Bypass -File scripts\\setup_windows.ps1"
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "ERROR: cmake not found. Install with: brew install cmake"
  exit 1
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: no python3 found. Install with: brew install python@3.11"
  exit 1
fi

# Verify Python version is in supported range (3.11 or 3.12)
PY_OK=$("$PYTHON_BIN" -c 'import sys; print(1 if (3,11) <= sys.version_info[:2] < (3,13) else 0)')
if [[ "$PY_OK" != "1" ]]; then
  echo "ERROR: $PYTHON_BIN is $($PYTHON_BIN --version), need Python 3.11 or 3.12."
  echo "       Install with: brew install python@3.11"
  exit 1
fi
echo "==> Using $PYTHON_BIN ($($PYTHON_BIN --version))"

mkdir -p "$VENDOR_DIR" "$MODELS_DIR/whisper" "$MODELS_DIR/vad" "$MODELS_DIR/piper"

# --- whisper.cpp build ---
if [[ ! -d "$WHISPER_DIR" ]]; then
  echo "==> Cloning whisper.cpp"
  git clone --depth=1 https://github.com/ggerganov/whisper.cpp "$WHISPER_DIR"
fi

if [[ ! -x "$WHISPER_DIR/build/bin/whisper-cli" ]]; then
  echo "==> Building whisper.cpp with Metal"
  cmake -S "$WHISPER_DIR" -B "$WHISPER_DIR/build" \
    -DGGML_METAL=ON \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_EXAMPLES=ON
  cmake --build "$WHISPER_DIR/build" -j --config Release
else
  echo "==> whisper.cpp already built"
fi

# --- Whisper model ---
WHISPER_MODEL_FILE="$MODELS_DIR/whisper/ggml-${WHISPER_MODEL}.bin"
if [[ ! -f "$WHISPER_MODEL_FILE" ]]; then
  echo "==> Downloading Whisper model: $WHISPER_MODEL"
  curl -L --fail \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${WHISPER_MODEL}.bin" \
    -o "$WHISPER_MODEL_FILE"
else
  echo "==> Whisper model already present: ggml-${WHISPER_MODEL}.bin"
fi

# Silero VAD ships inside the silero-vad PyPI package — no separate download needed.

# --- Piper voice ---
PIPER_ONNX="$MODELS_DIR/piper/${PIPER_VOICE}.onnx"
PIPER_JSON="$MODELS_DIR/piper/${PIPER_VOICE}.onnx.json"
if [[ ! -f "$PIPER_ONNX" ]]; then
  echo "==> Downloading Piper voice: $PIPER_VOICE"
  curl -L --fail \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/${PIPER_VOICE}.onnx" \
    -o "$PIPER_ONNX"
fi
if [[ ! -f "$PIPER_JSON" ]]; then
  curl -L --fail \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/${PIPER_VOICE}.onnx.json" \
    -o "$PIPER_JSON"
fi
echo "==> Piper voice ready"

# --- Python env (isolated in .venv/, never touches your system Python) ---
cd "$REPO_ROOT"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "==> Creating isolated Python virtualenv at: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "==> Reusing existing Python virtualenv at: $VENV_DIR"
fi

echo "==> Installing Python dependencies into .venv/ (not your system Python)"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -e .
echo "==> Dependencies installed in $VENV_DIR"

echo ""
echo "✓ Setup complete."
echo ""
echo "Activate the virtualenv in your shell:"
echo "  source .venv/bin/activate"
echo ""
echo "Quick test (offline, no mic):"
echo "  python -m live_translator.asr --help"
echo ""
echo "Live mode (mic → speaker):"
echo "  live-translator run"
