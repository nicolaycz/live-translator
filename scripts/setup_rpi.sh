#!/usr/bin/env bash
# Setup script for Raspberry Pi 5 (64-bit Raspberry Pi OS / Debian Bookworm).
#
# Everything Python lives inside .venv/ at the repo root — your system Python
# is NOT touched. To use the installed package later, run:
#     source .venv/bin/activate
#
# Steps performed:
#   1. Install system packages via apt (cmake, build tools, portaudio, python3-venv)
#   2. Create an isolated Python virtualenv at .venv/
#   3. Install Python deps into that .venv/ via pip install -e .
#   4. Compile whisper.cpp for ARM64 (CPU, NEON-optimized — no GPU on Pi)
#   5. Download Whisper, Silero VAD and Piper models into models/
#
# Usage: bash scripts/setup_rpi.sh [whisper_model]
#   whisper_model: tiny | base | small | medium  (default: base — small/medium
#                  are usable on Pi 5 but slower; tiny/base are recommended)

set -euo pipefail

WHISPER_MODEL="${1:-base}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/vendor"
MODELS_DIR="$REPO_ROOT/models"
VENV_DIR="$REPO_ROOT/.venv"
WHISPER_DIR="$VENDOR_DIR/whisper.cpp"
PIPER_VOICE="en_US-lessac-medium"

echo "==> live-translator setup for Raspberry Pi (model: $WHISPER_MODEL)"

# --- Sanity ---
if [[ "$(uname)" != "Linux" ]]; then
  echo "This script targets Linux (Raspberry Pi OS). For macOS, see scripts/setup_mac.sh."
  exit 1
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
  echo "WARN: detected architecture '$ARCH'. This script is tuned for 64-bit ARM (Pi 5)."
  echo "      It will still try to proceed, but you must be on 64-bit Raspberry Pi OS."
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "ERROR: apt-get not found. This script assumes Debian/Raspberry Pi OS."
  exit 1
fi

# --- System packages ---
# Required:
#   build-essential, cmake, git → to build whisper.cpp
#   portaudio19-dev             → sounddevice / PyAudio runtime
#   libsndfile1                 → soundfile (used by piper)
#   python3-venv, python3-pip   → Python virtualenv
#   python3-dev                 → headers for any C-extension wheels that fall back to source
#   curl                        → model downloads
#   ffmpeg                      → optional, but used by the README's "record a test WAV" recipe
SYSTEM_PKGS=(
  build-essential
  cmake
  git
  curl
  portaudio19-dev
  libsndfile1
  python3-venv
  python3-pip
  python3-dev
  ffmpeg
)

# Check which packages are missing before sudo'ing so we don't prompt unnecessarily.
MISSING=()
for pkg in "${SYSTEM_PKGS[@]}"; do
  if ! dpkg -s "$pkg" >/dev/null 2>&1; then
    MISSING+=("$pkg")
  fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "==> Installing system packages: ${MISSING[*]}"
  echo "    (you may be prompted for your sudo password)"
  sudo apt-get update
  sudo apt-get install -y "${MISSING[@]}"
else
  echo "==> All system packages already installed"
fi

# --- Python interpreter ---
# Raspberry Pi OS Bookworm ships Python 3.11 as python3. Trixie ships 3.12.
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: no python3 found. apt-get install python3 python3-venv"
  exit 1
fi

PY_OK=$("$PYTHON_BIN" -c 'import sys; print(1 if (3,11) <= sys.version_info[:2] < (3,13) else 0)')
if [[ "$PY_OK" != "1" ]]; then
  echo "ERROR: $PYTHON_BIN is $($PYTHON_BIN --version), need Python 3.11 or 3.12."
  echo "       On Raspberry Pi OS Bookworm: 'sudo apt-get install python3.11 python3.11-venv'"
  exit 1
fi
echo "==> Using $PYTHON_BIN ($($PYTHON_BIN --version))"

mkdir -p "$VENDOR_DIR" "$MODELS_DIR/whisper" "$MODELS_DIR/vad" "$MODELS_DIR/piper"

# --- whisper.cpp build (CPU, ARM NEON) ---
if [[ ! -d "$WHISPER_DIR" ]]; then
  echo "==> Cloning whisper.cpp"
  git clone --depth=1 https://github.com/ggerganov/whisper.cpp "$WHISPER_DIR"
fi

if [[ ! -x "$WHISPER_DIR/build/bin/whisper-cli" ]]; then
  echo "==> Building whisper.cpp (CPU, NEON)"
  # No Metal/CUDA on Pi. NEON is auto-detected by ggml on aarch64.
  # Pi 5 has 4 Cortex-A76 cores @ 2.4 GHz — use them all for the build.
  cmake -S "$WHISPER_DIR" -B "$WHISPER_DIR/build" \
    -DGGML_METAL=OFF \
    -DGGML_CUDA=OFF \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_EXAMPLES=ON \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build "$WHISPER_DIR/build" -j"$(nproc)" --config Release
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
# torch's default index serves CPU-only aarch64 wheels — no extra flags needed.
"$VENV_DIR/bin/pip" install -e .
echo "==> Dependencies installed in $VENV_DIR"

echo ""
echo "✓ Setup complete."
echo ""
echo "Activate the virtualenv in your shell:"
echo "  source .venv/bin/activate"
echo ""
echo "List audio devices (USB mic, HAT, etc.):"
echo "  live-translator devices"
echo ""
echo "Verify your mic before going live:"
echo "  live-translator mic-test"
echo ""
echo "Live mode (mic → speaker):"
echo "  live-translator run"
