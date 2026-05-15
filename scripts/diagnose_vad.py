"""Quick diagnostic: capture 5s from MacBook mic while user speaks,
then probe Silero with both raw and resampled audio."""

import time
import numpy as np
import sounddevice as sd

from live_translator.audio_io import _resample_linear, list_devices
from live_translator.vad import SileroVAD, VAD_FRAME_SAMPLES

mb = next((d for d in list_devices() if "macbook" in d.name.lower() and d.is_input), None)
assert mb, "MacBook Pro Microphone not found"
print(f"Device: [{mb.index}] {mb.name} (native sr={int(mb.default_samplerate)})\n")

DURATION = 5
NATIVE_SR = int(mb.default_samplerate)

print(f"Speak now for {DURATION} seconds…")
audio_native = sd.rec(NATIVE_SR * DURATION, samplerate=NATIVE_SR, channels=1,
                      dtype="float32", device=mb.index)
sd.wait()
audio_native = audio_native[:, 0]
print(f"Captured at {NATIVE_SR} Hz.")
print(f"  range: [{audio_native.min():+.3f}, {audio_native.max():+.3f}]  RMS: {np.sqrt(np.mean(audio_native**2)):.4f}\n")

# Resample to 16k using OUR resampler
audio_16k_ours = _resample_linear(audio_native, NATIVE_SR, 16000)
print(f"Resampled (our linear): {audio_16k_ours.shape[0]} samples")
print(f"  range: [{audio_16k_ours.min():+.3f}, {audio_16k_ours.max():+.3f}]  RMS: {np.sqrt(np.mean(audio_16k_ours**2)):.4f}")

# Also let sounddevice capture directly at 16k for comparison
print("\nNow capturing the SAME duration directly at 16k (sounddevice resampling)…")
print(f"Speak now for {DURATION} seconds…")
audio_16k_sd = sd.rec(16000 * DURATION, samplerate=16000, channels=1,
                       dtype="float32", device=mb.index)
sd.wait()
audio_16k_sd = audio_16k_sd[:, 0]
print(f"  range: [{audio_16k_sd.min():+.3f}, {audio_16k_sd.max():+.3f}]  RMS: {np.sqrt(np.mean(audio_16k_sd**2)):.4f}\n")

# Probe Silero with both
def probe(label, audio):
    vad = SileroVAD()
    n = audio.shape[0] // VAD_FRAME_SAMPLES
    probs = []
    for i in range(n):
        f = audio[i*VAD_FRAME_SAMPLES:(i+1)*VAD_FRAME_SAMPLES].astype(np.float32)
        probs.append(vad.speech_prob(f))
    if not probs:
        print(f"{label}: no frames")
        return
    arr = np.array(probs)
    print(f"{label}:")
    print(f"  frames: {len(probs)}, prob min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.3f}")
    print(f"  frames >=0.5: {(arr >= 0.5).sum()}/{len(probs)} ({(arr >= 0.5).mean()*100:.1f}%)")
    print(f"  prob distribution: ", end="")
    for thr in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f">={thr}: {(arr >= thr).sum():3d}  ", end="")
    print()

probe("Silero on OUR resampled 16k", audio_16k_ours)
probe("Silero on sounddevice-native 16k", audio_16k_sd)
