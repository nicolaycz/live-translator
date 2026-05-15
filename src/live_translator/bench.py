"""Benchmark: run the same audio through every available Whisper model."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from live_translator.asr import transcribe_file
from live_translator.paths import WHISPER_MODEL_FILES, whisper_model_path

cli = typer.Typer(add_completion=False)
console = Console()


@cli.command()
def main(
    wav: Path = typer.Argument(..., help="Spanish WAV file."),
    language: str = typer.Option("es", "--language", "-l"),
):
    """Translate the same WAV with every locally-available Whisper model."""
    table = Table(title=f"Translation benchmark — {wav.name}")
    table.add_column("model")
    table.add_column("elapsed (s)", justify="right")
    table.add_column("RTF", justify="right")
    table.add_column("English text", overflow="fold")

    for name in WHISPER_MODEL_FILES:
        if not whisper_model_path(name).exists():
            table.add_row(name, "—", "—", "[dim](not downloaded)[/dim]")
            continue
        try:
            r = transcribe_file(wav, model=name, source_language=language)
            table.add_row(name, f"{r.elapsed_s:.2f}", f"{r.real_time_factor:.2f}x", r.text)
        except Exception as exc:
            table.add_row(name, "—", "—", f"[red]{exc}[/red]")

    console.print(table)


if __name__ == "__main__":
    cli()
