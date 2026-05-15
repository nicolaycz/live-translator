"""Top-level CLI: `live-translator <command>`."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from live_translator.audio_io import list_devices, resolve_device
from live_translator.paths import DEFAULT_PIPER_VOICE, DEFAULT_WHISPER_MODEL
from live_translator.pipeline import PipelineConfig, run_live, run_mic_test

app = typer.Typer(add_completion=False, help="Real-time ES→EN voice-to-voice translator.")
console = Console()


@app.command()
def run(
    model: str = typer.Option(DEFAULT_WHISPER_MODEL, "--model", "-m",
                              help="Whisper model: tiny|base|small|medium"),
    voice: str = typer.Option(DEFAULT_PIPER_VOICE, "--voice", "-v",
                              help="Piper voice id"),
    language: str = typer.Option("es", "--language", "-l", help="Source language code"),
    no_speak: bool = typer.Option(False, "--no-speak", help="Print text only; skip TTS playback."),
    vad_threshold: float = typer.Option(0.5, "--vad-threshold", min=0.0, max=1.0),
    silence_ms: int = typer.Option(700, "--silence-ms",
                                   help="Trailing silence (ms) that ends a turn."),
    max_segment_ms: int = typer.Option(15000, "--max-segment-ms",
                                       help="Hard cap per segment so we don't wait forever."),
    input_device: str | None = typer.Option(
        None, "--input-device", "-i",
        help="Input device: index, exact name, or substring (e.g. 'AirPods'). Default: system default.",
    ),
    output_device: str | None = typer.Option(
        None, "--output-device", "-o",
        help="Output device: index, exact name, or substring. Default: system default.",
    ),
):
    """Capture mic, translate Spanish to English, speak it back."""
    cfg = PipelineConfig(
        whisper_model=model,
        piper_voice=voice,
        source_language=language,
        vad_threshold=vad_threshold,
        silence_ms_to_end=silence_ms,
        max_segment_ms=max_segment_ms,
        speak=not no_speak,
        input_device=input_device,
        output_device=output_device,
    )
    # Validate device specs early with a helpful error
    try:
        resolve_device(input_device, kind="input")
        resolve_device(output_device, kind="output")
    except ValueError as exc:
        console.print(f"[red]✗ {exc}[/red]")
        raise typer.Exit(1)
    run_live(cfg)


@app.command()
def devices():
    """List all audio input and output devices with their IDs."""
    devs = list_devices()
    inputs = [d for d in devs if d.is_input]
    outputs = [d for d in devs if d.is_output]

    in_table = Table(title="Input devices (microphones)", title_style="bold cyan")
    in_table.add_column("idx", justify="right", style="dim")
    in_table.add_column("name")
    in_table.add_column("ch", justify="right", style="dim")
    in_table.add_column("default sr", justify="right", style="dim")
    in_table.add_column("default", style="green")
    for d in inputs:
        in_table.add_row(
            str(d.index), d.name, str(d.max_input_channels),
            f"{int(d.default_samplerate)} Hz",
            "✓" if d.is_default_input else "",
        )
    console.print(in_table)

    out_table = Table(title="Output devices (speakers)", title_style="bold cyan")
    out_table.add_column("idx", justify="right", style="dim")
    out_table.add_column("name")
    out_table.add_column("ch", justify="right", style="dim")
    out_table.add_column("default sr", justify="right", style="dim")
    out_table.add_column("default", style="green")
    for d in outputs:
        out_table.add_row(
            str(d.index), d.name, str(d.max_output_channels),
            f"{int(d.default_samplerate)} Hz",
            "✓" if d.is_default_output else "",
        )
    console.print(out_table)

    console.print(
        "\n[dim]Use the index or a substring of the name with "
        "--input-device / --output-device.[/dim]\n"
        "[dim]Example: live-translator run --input-device AirPods[/dim]"
    )


@app.command("mic-test")
def mic_test(
    duration: float = typer.Option(10.0, "--duration", "-d", min=2.0, max=120.0,
                                   help="How long to listen, in seconds."),
    input_device: str | None = typer.Option(
        None, "--input-device", "-i",
        help="Input device: index, exact name, or substring. Default: system default.",
    ),
    vad_threshold: float = typer.Option(0.5, "--vad-threshold", min=0.0, max=1.0),
):
    """Open the mic for N seconds and show a live VU meter + speech probability.

    Use this BEFORE the live mode to verify the right device is picked up and
    your voice triggers the VAD.
    """
    try:
        resolve_device(input_device, kind="input")
    except ValueError as exc:
        console.print(f"[red]✗ {exc}[/red]")
        raise typer.Exit(1)

    cfg = PipelineConfig(input_device=input_device, vad_threshold=vad_threshold)
    run_mic_test(cfg, duration_s=duration)


@app.command()
def translate_file(
    wav: Path = typer.Argument(..., help="Path to a WAV file."),
    model: str = typer.Option(DEFAULT_WHISPER_MODEL, "--model", "-m"),
    language: str = typer.Option("es", "--language", "-l"),
    speak: bool = typer.Option(False, "--speak", help="Also play the English translation."),
    output_device: str | None = typer.Option(
        None, "--output-device", "-o",
        help="Output device for --speak. Default: system default.",
    ),
):
    """One-shot: translate a WAV file to English text (and optionally speak it)."""
    from live_translator.asr import transcribe_file
    from live_translator.audio_io import play_pcm
    from live_translator.tts import synthesize

    asr = transcribe_file(wav, model=model, source_language=language)
    typer.echo(asr.text)
    typer.echo(
        f"[asr {asr.elapsed_s:.2f}s, audio {asr.duration_s:.2f}s, "
        f"RTF {asr.real_time_factor:.2f}x]",
        err=True,
    )
    if speak and asr.text.strip():
        tts = synthesize(asr.text)
        play_pcm(tts.pcm, tts.sample_rate, blocking=True, device=output_device)


if __name__ == "__main__":
    app()
