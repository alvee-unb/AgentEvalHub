"""
generate_report.py — Report generation entry point.

Usage:
    python generate_report.py
    python generate_report.py --format html
    python generate_report.py --format markdown
    python generate_report.py --format all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import get_report_data
from src.html_report import generate_html
from src.markdown_report import generate_markdown

console = Console()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Safety Evaluation Report Generator")
    parser.add_argument("--format", choices=["html", "markdown", "all"], default="all")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    setup_logging()

    console.print(Panel.fit(
        "[bold cyan]AI Safety Evaluation Report Generator[/bold cyan]",
        border_style="cyan",
    ))

    data = get_report_data()
    out = Path(args.output_dir)
    generated = []

    if args.format in ("html", "all"):
        p = generate_html(data, out / "html" / "evaluation_report.html")
        generated.append(f"HTML     -> {p}")
        console.print(f"  [green]HTML[/green]     -> {p}")

    if args.format in ("markdown", "all"):
        p = generate_markdown(data, out / "markdown" / "evaluation_report.md")
        generated.append(f"Markdown -> {p}")
        console.print(f"  [green]Markdown[/green] -> {p}")

    console.print(Panel.fit(
        f"[green]{len(generated)} report(s) generated[/green]",
        border_style="green",
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
