# live-translator

Real-time **Spanish → English** voice-to-voice translator. Offline, open source, edge-friendly.

You speak Spanish into your microphone, you hear the English translation through your speakers. No cloud, no API keys, no telemetry. Same code path is intended to run on your Mac and on a Raspberry Pi 5.

## Stack

```
[Mic] → [Silero VAD] → [whisper.cpp --translate] → [Piper TTS] → [Speakers]
```

- **whisper.cpp** — ASR + translation (ES audio → EN text), Metal on Mac, ARM NEON on Pi
- **Silero VAD** — voice activity detection, segments speech turns
- **Piper** — neural TTS for English output, optimized for edge

---

## Installation (macOS, Apple Silicon)

The first run downloads ~565 MB of models and compiles `whisper.cpp` from source. Plan for ~10 minutes the first time, depending on your connection. Subsequent runs reuse everything.

### Step 1 — Install Homebrew (skip if you already have it)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After install, follow the on-screen instructions to add Homebrew to your `PATH` (the installer prints two `echo ... >> ~/.zprofile` commands — copy and run them, then restart your terminal).

Verify:
```bash
brew --version
```

### Step 2 — Install system dependencies

You need `cmake` (to compile `whisper.cpp`), `git`, and Python 3.11 or 3.12.

```bash
brew install cmake
```

```bash
brew install git
```

```bash
brew install python@3.11
```

Verify:
```bash
cmake --version
python3.11 --version
```

### Step 3 — Clone (or open) this repo

If you're reading this from inside the repo already, skip this step.

```bash
git clone <repo-url> live-translator
cd live-translator
```

### Step 4 — Run the setup script

This single command:
1. Creates a Python virtualenv at `.venv/`
2. Installs Python dependencies via `pip install -e .`
3. Clones and builds `whisper.cpp` with Metal acceleration into `vendor/whisper.cpp/`
4. Downloads the Whisper model (`small` by default — ~488 MB)
5. Downloads Silero VAD (~2 MB)
6. Downloads the Piper voice `en_US-lessac-medium` (~75 MB)

```bash
bash scripts/setup_mac.sh
```

Or pick a different Whisper model size:
```bash
bash scripts/setup_mac.sh tiny     # ~75 MB,  fastest, lower quality
bash scripts/setup_mac.sh base     # ~142 MB, balanced
bash scripts/setup_mac.sh small    # ~488 MB, recommended for M-series  (default)
bash scripts/setup_mac.sh medium   # ~1.5 GB, highest quality, slower
```

You can run the script multiple times to add more model sizes — it skips anything already downloaded.

### Step 5 — Activate the virtualenv

Every new terminal session that wants to use `live-translator` needs the virtualenv active:

```bash
source .venv/bin/activate
```

You'll know it's active when your shell prompt is prefixed with `(.venv)`. To leave the env, run `deactivate`.

### Step 6 — Grant microphone permission

The first time you run live mode, macOS will pop up a permission dialog asking to allow your terminal app (Terminal.app, iTerm, Ghostty, VSCode, etc.) to access the microphone. Allow it. If you accidentally deny, fix it in:

> System Settings → Privacy & Security → Microphone → enable your terminal app

---

## Usage

All commands assume you're in the repo root **with the virtualenv activated** (`source .venv/bin/activate`). Your prompt should show `(.venv)`.

### Recommended first run order

```bash
# 1. See what audio devices the system exposes
live-translator devices

# 2. Verify your mic actually works (live VU meter + speech detector for 10s)
live-translator mic-test
# ...or pick a specific mic by name or index:
live-translator mic-test --input-device "AirPods"
live-translator mic-test -i 1

# 3. Go live
live-translator run
```

### `devices` — list all mics and speakers

Shows two tables (inputs and outputs) with each device's index, name, channels, default sample rate, and which one the system uses by default.

```bash
live-translator devices
```

### `mic-test` — verify mic + VAD before going live

Opens the mic for 10 seconds and shows a live dashboard with:

- **level** bar (dBFS) — proves audio is reaching the app
- **speech** bar (Silero VAD probability) — proves the VAD is firing on your voice
- a final summary of how many frames were detected as speech

```bash
live-translator mic-test                       # default mic, 10s
live-translator mic-test --duration 30
live-translator mic-test --input-device 1
live-translator mic-test -i "MacBook"          # substring match
live-translator mic-test --vad-threshold 0.3   # more sensitive
```

If you see the level bar moving but the speech bar stays low, your mic works but Silero isn't recognizing your voice — try `--vad-threshold 0.3` or speak louder/closer.

### Live translation (mic → speaker)

```bash
live-translator run
```

Once running, you'll see a live TUI dashboard with:

- 🎙 **input** and 🔈 **output** device names
- A **phase badge**: `🎧 listening` → `🗣 capturing speech` → `⚙ translating` → `🔊 speaking`
- **level** and **speech** meters updating in real time
- The last 8 translations with timestamps and per-segment RTF
- Any audio xruns or errors surfaced inline

Press **Ctrl+C** to stop.

Useful flags:

```bash
# Pick specific input/output devices (index or name substring)
live-translator run --input-device "AirPods" --output-device "MacBook"
live-translator run -i 1 -o 2

# Use a different Whisper model (must have been downloaded)
live-translator run --model medium

# Don't speak the output, just print it
live-translator run --no-speak

# Tune VAD: lower threshold = more sensitive, higher = ignores soft speech
live-translator run --vad-threshold 0.4

# How much trailing silence (ms) ends a turn (default 700)
live-translator run --silence-ms 500

# Hard cap per segment so very long monologues still get processed (default 15s)
live-translator run --max-segment-ms 10000

# Different source language (Whisper supports 99 languages → English)
live-translator run --language fr
```

> ⚠ If you pick the same physical device for input and output (e.g. AirPods both ways), the TTS output may feed back into the mic and trigger more translations. The TUI warns you when it detects this.

### Translate a pre-recorded WAV file

Useful for sanity checks before going live, or batch-translating recordings.

```bash
# Just print the English translation
live-translator translate-file path/to/spanish.wav

# Print AND speak the translation
live-translator translate-file path/to/spanish.wav --speak
```

### Test ASR or TTS in isolation

```bash
# ASR only — translate a WAV to English text
python -m live_translator.asr translate path/to/spanish.wav

# TTS only — speak an English sentence
python -m live_translator.tts speak "Hello, this is the translator."

# Save TTS to a WAV file instead of playing
python -m live_translator.tts speak "Hello world" --out hello.wav
```

### Benchmark all available models on the same audio

```bash
python -m live_translator.bench path/to/spanish.wav
```

Prints a table comparing elapsed time, real-time factor (RTF), and the produced English text for every Whisper model you've downloaded.

---

## How to record a quick test WAV

If you don't have a Spanish audio sample handy, the easiest way on macOS:

1. Open **QuickTime Player**
2. **File → New Audio Recording**
3. Click record, say something in Spanish, click stop
4. **File → Export As → Audio Only**, save as `.m4a`
5. Convert to WAV with ffmpeg (`brew install ffmpeg` if needed):

```bash
ffmpeg -i recording.m4a -ar 16000 -ac 1 samples/test.wav
```

Then:
```bash
live-translator translate-file samples/test.wav --speak
```

---

## Expected performance (Apple Silicon)

| Model    | Disk    | RTF on M-series | Quality       | Latency end-to-end |
|----------|---------|-----------------|---------------|--------------------|
| `tiny`   | ~75 MB  | ~15-25x         | Rough         | ~0.8 s             |
| `base`   | ~142 MB | ~10-15x         | OK            | ~0.9 s             |
| `small`  | ~488 MB | ~5-10x          | Good          | ~1.0-1.5 s         |
| `medium` | ~1.5 GB | ~2-4x           | Very good     | ~1.5-2.5 s         |

"Latency end-to-end" = from when you stop speaking to when you hear the English translation. Includes ~700ms of silence detection. RTF = real-time factor (e.g., `5x` means 1 second of audio is processed in 200ms). Numbers are rough estimates; your mileage will vary by chip and segment length.

---

## Troubleshooting

**"command not found: cmake" or "python3.11"**
You skipped or failed step 2. Re-run `brew install cmake python@3.11` and restart your terminal.

**"command not found: live-translator"**
The virtualenv isn't activated. Run `source .venv/bin/activate` from the repo root. Your prompt should then show `(.venv)`.

**"whisper-cli not found"**
The setup script didn't finish. Re-run `bash scripts/setup_mac.sh`. Look for cmake errors — usually a missing Xcode Command Line Tools dependency: `xcode-select --install`.

**"Whisper model not found at ..."**
You're trying to use a model you haven't downloaded. Run `bash scripts/setup_mac.sh medium` (or whichever size).

**No mic input / silence forever**
First, check microphone permission for your terminal (see Step 6). Then run `live-translator mic-test` — if the level bar stays flat at -120 dBFS, the system isn't sending you any audio. List devices with `live-translator devices` and try a specific one with `live-translator mic-test -i 1`.

**Mic level moves but no speech is detected**
Run `live-translator mic-test`. If the level bar moves but the speech bar stays low, lower the VAD threshold: `live-translator mic-test --vad-threshold 0.3`.

**Wrong mic or speaker is being used**
Use `live-translator devices` to see all devices, then pick one explicitly with `--input-device "AirPods"` (substring match) or `--input-device 1` (index). Same for `--output-device`.

**Hearing yourself / feedback loop**
You're using the same physical device for input and output (common with AirPods). Either set `--no-speak` to disable TTS, or use a different output device with `--output-device "MacBook"`.

**Setup downloads are very slow**
HuggingFace can be slow from some regions. The script uses `curl -L --fail`, so just re-run if it stalls — partial files will be re-downloaded; complete ones are skipped.

---

## Project layout

```
live-translator/
├── scripts/setup_mac.sh         # one-shot bootstrap
├── vendor/whisper.cpp/          # cloned + built (gitignored)
├── models/                      # downloaded weights (gitignored)
│   ├── whisper/ggml-*.bin
│   ├── vad/silero_vad.onnx
│   └── piper/en_US-lessac-medium.{onnx,onnx.json}
├── src/live_translator/
│   ├── paths.py                 # central paths
│   ├── asr.py                   # whisper.cpp wrapper
│   ├── tts.py                   # Piper wrapper
│   ├── vad.py                   # Silero VAD + segment state machine
│   ├── audio_io.py              # mic + speaker
│   ├── pipeline.py              # orchestration
│   ├── cli.py                   # entry point: `live-translator`
│   └── bench.py                 # compare model sizes
└── samples/                     # your test WAVs (gitignored)
```

---

## Roadmap

- [x] Phase 0 — Repo bootstrap, setup script
- [x] Phase 1 — ASR wrapper (whisper.cpp via subprocess)
- [x] Phase 2 — TTS wrapper (Piper)
- [x] Phase 3 — Mic capture + Silero VAD
- [x] Phase 4 — Full live pipeline
- [x] Phase 5 — Metrics, model size flag, benchmark
- [ ] Phase 6 — Raspberry Pi 5 port (`scripts/setup_rpi.sh`)

## License

MIT
