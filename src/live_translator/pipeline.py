"""End-to-end live pipeline with a live TUI dashboard.

Threading model:
  - Capture thread: pulls frames from the mic generator, feeds VAD, pushes
    completed segments to a queue. Updates shared state (level meter, vad prob).
  - Worker thread: consumes segments, runs ASR + TTS, plays output. Updates
    pipeline phase ('translating' / 'speaking') in shared state.
  - Main thread: rich.Live renderer at ~10 Hz reading shared state.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from live_translator.asr import transcribe_pcm
from live_translator.audio_io import (
    device_name,
    microphone_frames,
    play_pcm,
    rms_dbfs,
)
from live_translator.paths import DEFAULT_PIPER_VOICE, DEFAULT_WHISPER_MODEL
from live_translator.tts import synthesize
from live_translator.vad import (
    VAD_FRAME_SAMPLES,
    VAD_SAMPLE_RATE,
    SegmentBuilder,
    SileroVAD,
    SpeechSegment,
)

console = Console()


class Phase(str, Enum):
    LISTENING = "listening"
    CAPTURING = "capturing speech"
    TRANSLATING = "translating"
    SPEAKING = "speaking"


@dataclass
class PipelineConfig:
    whisper_model: str = DEFAULT_WHISPER_MODEL
    piper_voice: str = DEFAULT_PIPER_VOICE
    source_language: str = "es"
    vad_threshold: float = 0.5
    silence_ms_to_end: int = 700
    min_speech_ms: int = 250
    pre_roll_ms: int = 200
    max_segment_ms: int = 15000
    speak: bool = True
    input_device: str | int | None = None
    output_device: str | int | None = None


@dataclass
class TranslationEntry:
    text: str
    duration_s: float
    asr_elapsed_s: float
    rtf: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class LiveState:
    phase: Phase = Phase.LISTENING
    level_dbfs: float = -120.0
    speech_prob: float = 0.0
    segments_done: int = 0
    started_at: float = field(default_factory=time.time)
    last_status: str | None = None
    history: deque[TranslationEntry] = field(default_factory=lambda: deque(maxlen=8))
    error: str | None = None
    # Persistent log of recent errors. Always shown — never overwritten.
    error_log: deque[tuple[float, str]] = field(default_factory=lambda: deque(maxlen=4))
    last_tts_info: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def _vu_bar(dbfs: float, width: int = 24) -> Text:
    """Render a colored VU meter from dBFS. -60 dBFS → empty, 0 dBFS → full."""
    floor = -60.0
    norm = max(0.0, min(1.0, (dbfs - floor) / -floor))
    filled = int(round(norm * width))
    bar = "█" * filled + "░" * (width - filled)
    if dbfs > -6:
        color = "red"
    elif dbfs > -18:
        color = "green"
    elif dbfs > -40:
        color = "cyan"
    else:
        color = "grey50"
    return Text(bar, style=color)


def _phase_badge(phase: Phase) -> Text:
    icon, color = {
        Phase.LISTENING: ("🎧", "grey70"),
        Phase.CAPTURING: ("🗣 ", "yellow"),
        Phase.TRANSLATING: ("⚙ ", "cyan"),
        Phase.SPEAKING: ("🔊", "magenta"),
    }[phase]
    return Text(f"{icon} {phase.value}", style=f"bold {color}")


def _build_dashboard(state: LiveState, cfg: PipelineConfig) -> Panel:
    # Pre-resolved display strings are attached to state in run_live().
    in_name = state.__dict__.get("_in_display", "(unknown input)")
    out_name = state.__dict__.get("_out_display", "(unknown output)")

    with state.lock:
        phase = state.phase
        level = state.level_dbfs
        prob = state.speech_prob
        done = state.segments_done
        history = list(state.history)
        last_status = state.last_status
        error = state.error
        error_log = list(state.error_log)
        last_tts_info = state.last_tts_info
        elapsed = time.time() - state.started_at

    # Header table: device + phase
    header = Table.grid(expand=True)
    header.add_column(ratio=2)
    header.add_column(ratio=1, justify="right")
    header.add_row(
        Text.assemble(("🎙 input  ", "dim"), (in_name, "white")),
        _phase_badge(phase),
    )
    header.add_row(
        Text.assemble(("🔈 output ", "dim"), (out_name, "white")),
        Text(f"segments: {done}  uptime: {elapsed:.0f}s", style="dim"),
    )

    # Meter rows
    meter = Table.grid(expand=True)
    meter.add_column(width=10)
    meter.add_column(ratio=1)
    meter.add_column(width=14, justify="right")
    meter.add_row(Text("level", style="dim"), _vu_bar(level), Text(f"{level:6.1f} dBFS", style="dim"))

    speech_bar = _vu_bar(-60 + prob * 60)  # remap 0..1 → -60..0 so the bar moves
    speech_label = Text(f"prob {prob:0.2f}", style="bold yellow" if prob >= cfg.vad_threshold else "dim")
    meter.add_row(Text("speech", style="dim"), speech_bar, speech_label)

    # History panel
    if history:
        hist = Table.grid(expand=True)
        hist.add_column(width=8, style="dim")
        hist.add_column(ratio=1)
        hist.add_column(width=18, justify="right", style="dim")
        for entry in reversed(history):
            ts = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))
            hist.add_row(
                ts,
                Text(entry.text, overflow="fold"),
                Text(f"{entry.duration_s:.1f}s · RTF {entry.rtf:.1f}x"),
            )
        body = hist
    else:
        body = Text("waiting for speech…", style="dim italic")

    # Status / error footer
    footer_lines = []
    if last_tts_info:
        footer_lines.append(Text(f"🔊 last tts: {last_tts_info}", style="dim cyan"))
    if last_status:
        footer_lines.append(Text(f"⚠ audio status: {last_status}", style="yellow"))
    # Persistent error log: never overwritten, last 4 errors with timestamps
    for ts, msg in error_log:
        ts_str = time.strftime("%H:%M:%S", time.localtime(ts))
        footer_lines.append(Text(f"✗ [{ts_str}] {msg}", style="bold red"))
    if error and not any(error in m for _, m in error_log):
        footer_lines.append(Text(f"✗ {error}", style="bold red"))
    footer_lines.append(Text("Ctrl+C to stop", style="dim"))

    return Panel(
        Group(header, Text(""), meter, Text(""), body, Text(""), *footer_lines),
        title=Text(f"live-translator · model={cfg.whisper_model} · voice={cfg.piper_voice}", style="bold"),
        border_style="cyan",
    )


def _capture_loop(
    cfg: PipelineConfig,
    state: LiveState,
    seg_queue: queue.Queue[SpeechSegment | None],
    stop_event: threading.Event,
) -> None:
    vad = SileroVAD(threshold=cfg.vad_threshold)
    builder = SegmentBuilder(
        vad=vad,
        min_speech_ms=cfg.min_speech_ms,
        silence_ms_to_end=cfg.silence_ms_to_end,
        pre_roll_ms=cfg.pre_roll_ms,
        max_segment_ms=cfg.max_segment_ms,
    )

    def _on_status(status):
        with state.lock:
            state.last_status = str(status)

    try:
        with microphone_frames(device=cfg.input_device, on_status=_on_status) as frames:
            for frame in frames:
                if stop_event.is_set():
                    break
                segment = builder.push(frame)
                # Update meters for every frame
                with state.lock:
                    state.level_dbfs = rms_dbfs(frame)
                    state.speech_prob = builder.last_speech_prob
                    if state.phase == Phase.LISTENING and builder.is_speaking:
                        state.phase = Phase.CAPTURING
                    elif state.phase == Phase.CAPTURING and not builder.is_speaking and segment is None:
                        # Capture ended without enough speech (false trigger)
                        state.phase = Phase.LISTENING
                if segment is not None:
                    with state.lock:
                        if state.phase == Phase.CAPTURING:
                            state.phase = Phase.LISTENING
                    try:
                        seg_queue.put_nowait(segment)
                    except queue.Full:
                        with state.lock:
                            state.last_status = "queue full, dropped a segment"
    except Exception as exc:
        with state.lock:
            state.error = f"capture: {exc}"
        stop_event.set()


def _worker_loop(
    cfg: PipelineConfig,
    state: LiveState,
    seg_queue: queue.Queue[SpeechSegment | None],
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            segment = seg_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if segment is None:
            return

        with state.lock:
            state.phase = Phase.TRANSLATING
        try:
            asr = transcribe_pcm(
                segment.pcm,
                model=cfg.whisper_model,
                source_language=cfg.source_language,
            )
        except Exception as exc:
            with state.lock:
                state.error = f"ASR: {exc}"
                state.phase = Phase.LISTENING
            continue

        text = asr.text.strip()
        if not text:
            with state.lock:
                state.phase = Phase.LISTENING
            continue

        with state.lock:
            state.history.append(
                TranslationEntry(
                    text=text,
                    duration_s=segment.duration_s,
                    asr_elapsed_s=asr.elapsed_s,
                    rtf=asr.real_time_factor,
                )
            )
            state.segments_done += 1

        if cfg.speak:
            tts_start = time.time()
            try:
                tts = synthesize(text, voice=cfg.piper_voice)
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                with state.lock:
                    state.error = f"TTS synth failed: {type(exc).__name__}: {exc}"
                    state.error_log.append((time.time(), f"SYNTH: {tb.splitlines()[-1]}"))
                with state.lock:
                    state.phase = Phase.LISTENING
                continue

            with state.lock:
                state.phase = Phase.SPEAKING
                state.last_tts_info = (
                    f"synth {time.time() - tts_start:.2f}s, "
                    f"{tts.duration_s:.2f}s audio @ {tts.sample_rate} Hz"
                )

            try:
                # Stop any prior playback that's still hanging around. This
                # protects against the output stream getting stuck after the
                # first segment (common with Bluetooth devices that renegotiate
                # codec mid-session).
                import sounddevice as sd
                sd.stop()
                play_pcm(tts.pcm, tts.sample_rate, blocking=True, device=cfg.output_device)
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                with state.lock:
                    state.error = f"playback failed: {type(exc).__name__}: {exc}"
                    state.error_log.append((time.time(), f"PLAY: {tb.splitlines()[-1]}"))
                # Try fallback: system default output
                if cfg.output_device is not None:
                    try:
                        sd.stop()
                        play_pcm(tts.pcm, tts.sample_rate, blocking=True, device=None)
                        with state.lock:
                            state.error = (
                                f"playback on {cfg.output_device} failed, "
                                f"fell back to system default"
                            )
                    except Exception as exc2:
                        with state.lock:
                            state.error_log.append(
                                (time.time(), f"FALLBACK: {type(exc2).__name__}: {exc2}")
                            )

        with state.lock:
            state.phase = Phase.LISTENING


def _preflight_warnings(in_idx: int | None, out_idx: int | None,
                        in_display: str, out_display: str) -> list[str]:
    """Sanity checks that should be shown to the user before starting the TUI."""
    warnings: list[str] = []
    bt_keywords = ("airpods", "bluetooth", "hands-free", "buds", "headset")
    in_is_bt = any(kw in in_display.lower() for kw in bt_keywords)
    out_is_bt = any(kw in out_display.lower() for kw in bt_keywords)

    if in_idx is not None and out_idx is not None and in_idx == out_idx:
        if in_is_bt:
            warnings.append(
                f"Input and output are the SAME Bluetooth device ({in_display}). "
                f"macOS will switch the device to HFP mode (phone-call codec), "
                f"which (a) drops mic quality to 8 kHz mono and (b) cuts off "
                f"any audio playing through the device. Use a different mic, e.g.:\n"
                f"      live-translator run -i 'MacBook' -o 'AirPods'"
            )
        else:
            warnings.append(
                f"Input and output are the same device ({in_display}). "
                f"TTS output will leak into the mic and trigger more translations. "
                f"Pass --output-device to use a different speaker."
            )
    elif in_is_bt:
        warnings.append(
            f"Input is a Bluetooth device ({in_display}). Mic quality drops "
            f"to phone-call grade (8 kHz mono) and macOS may cut off other "
            f"audio when capturing. Prefer the MacBook mic for input:\n"
            f"      live-translator run -i 'MacBook'"
        )
    return warnings


def run_live(cfg: PipelineConfig) -> None:
    """Run the full live pipeline with a rich TUI until KeyboardInterrupt."""
    from live_translator.audio_io import resolve_device

    in_idx = resolve_device(cfg.input_device, kind="input")
    out_idx = resolve_device(cfg.output_device, kind="output")
    in_display = device_name(in_idx, kind="input")
    out_display = device_name(out_idx, kind="output")

    # Print preflight warnings BEFORE entering Live (which would erase them).
    for w in _preflight_warnings(in_idx, out_idx, in_display, out_display):
        console.print(f"[yellow]⚠ {w}[/yellow]")

    state = LiveState()
    setattr(state, "_in_display", in_display)
    setattr(state, "_out_display", out_display)

    seg_queue: queue.Queue[SpeechSegment | None] = queue.Queue(maxsize=8)
    stop_event = threading.Event()

    capture = threading.Thread(
        target=_capture_loop, args=(cfg, state, seg_queue, stop_event), daemon=True
    )
    worker = threading.Thread(
        target=_worker_loop, args=(cfg, state, seg_queue, stop_event), daemon=True
    )
    capture.start()
    # Give the capture thread a beat to either fail fast or open the stream
    # successfully — this turns a silent crash into a visible error before the TUI starts.
    time.sleep(0.5)
    if stop_event.is_set():
        with state.lock:
            err = state.error
        console.print(f"[bold red]✗ Could not start capture:[/bold red] {err}")
        console.print("[dim]Run `live-translator devices` to see available inputs, "
                      "or `live-translator mic-test -i <name>` to test one.[/dim]")
        return

    worker.start()

    interrupted_by_user = False
    try:
        with Live(_build_dashboard(state, cfg), console=console,
                  refresh_per_second=12, transient=False) as live:
            while not stop_event.is_set():
                live.update(_build_dashboard(state, cfg))
                time.sleep(1 / 12)
    except KeyboardInterrupt:
        interrupted_by_user = True
    finally:
        stop_event.set()
        try:
            seg_queue.put_nowait(None)
        except queue.Full:
            pass
        capture.join(timeout=1.0)
        worker.join(timeout=2.0)

        with state.lock:
            err = state.error
            n = state.segments_done
        if interrupted_by_user:
            console.print(f"\n[dim]Stopped by user · {n} segments translated.[/dim]")
        elif err:
            console.print(f"\n[bold red]✗ Pipeline stopped due to error:[/bold red] {err}")
        else:
            console.print(f"\n[dim]Stopped · {n} segments translated.[/dim]")


# --- Standalone helpers used by the CLI ---

def run_mic_test(cfg: PipelineConfig, duration_s: float = 10.0) -> None:
    """Open the mic for `duration_s` and show a live VU meter + VAD probability."""
    from live_translator.audio_io import resolve_device
    in_idx = resolve_device(cfg.input_device, kind="input")
    in_display = device_name(in_idx, kind="input")

    vad = SileroVAD(threshold=cfg.vad_threshold)

    state_lock = threading.Lock()
    shared = {"level": -120.0, "prob": 0.0, "status": None, "speech_frames": 0, "total_frames": 0}
    stop = threading.Event()

    def _on_status(status):
        with state_lock:
            shared["status"] = str(status)

    def _capture():
        try:
            with microphone_frames(device=cfg.input_device, on_status=_on_status) as frames:
                for frame in frames:
                    if stop.is_set():
                        break
                    prob = vad.speech_prob(frame)
                    level = rms_dbfs(frame)
                    with state_lock:
                        shared["level"] = level
                        shared["prob"] = prob
                        shared["total_frames"] += 1
                        if prob >= cfg.vad_threshold:
                            shared["speech_frames"] += 1
        except Exception as exc:
            with state_lock:
                shared["status"] = f"error: {exc}"
            stop.set()

    def _render() -> Panel:
        with state_lock:
            level = shared["level"]
            prob = shared["prob"]
            status = shared["status"]
            sp = shared["speech_frames"]
            tot = shared["total_frames"]
        ratio = (sp / tot) if tot else 0.0
        body = Table.grid(expand=True)
        body.add_column(width=10)
        body.add_column(ratio=1)
        body.add_column(width=18, justify="right")
        body.add_row("level", _vu_bar(level), Text(f"{level:6.1f} dBFS"))
        body.add_row(
            "speech",
            _vu_bar(-60 + prob * 60),
            Text(f"prob {prob:0.2f}",
                 style="bold yellow" if prob >= cfg.vad_threshold else "dim"),
        )
        footer = Text(
            f"speech ratio: {ratio:0.0%}  ({sp}/{tot} frames)  ·  {VAD_FRAME_SAMPLES} samples @ {VAD_SAMPLE_RATE} Hz",
            style="dim",
        )
        if status:
            footer = Group(Text(f"⚠ {status}", style="yellow"), footer)
        return Panel(
            Group(Text(f"🎙 {in_display}", style="white"), Text(""), body, Text(""), footer),
            title=Text(f"mic-test · {duration_s:.0f}s", style="bold"),
            border_style="cyan",
        )

    cap_thread = threading.Thread(target=_capture, daemon=True)
    cap_thread.start()
    # Let the capture thread fail-fast if the device can't be opened.
    time.sleep(0.5)
    with state_lock:
        early_status = shared["status"]
    if stop.is_set():
        console.print(f"[bold red]✗ Could not start capture:[/bold red] {early_status}")
        return

    deadline = time.time() + duration_s
    try:
        with Live(_render(), console=console, refresh_per_second=15) as live:
            while time.time() < deadline and not stop.is_set():
                live.update(_render())
                time.sleep(1 / 15)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        cap_thread.join(timeout=1.0)

    with state_lock:
        sp, tot = shared["speech_frames"], shared["total_frames"]
        status = shared["status"]
    if status and tot == 0:
        console.print(f"[bold red]✗ Capture failed:[/bold red] {status}")
    elif tot == 0:
        console.print("[red]✗ No audio frames captured. Check mic permission and device selection.[/red]")
    elif sp == 0:
        console.print("[yellow]⚠ Audio captured but no speech detected. Try speaking louder or lower --vad-threshold.[/yellow]")
    else:
        console.print(f"[green]✓ Captured {tot} frames, {sp} of them ({sp / tot:0.0%}) flagged as speech.[/green]")
