"""
generate_dataset.py — Main entry point for dataset generation.

Usage:
    python generate_dataset.py
    python generate_dataset.py --seed 123
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure src is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.generator import generate_dataset
from src.validator import validate_dataset, compute_statistics
from src.exporter import export_json, export_parquet, export_dataset_card

console = Console()


def setup_logging(log_level: str, log_file: str | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
    )


def load_config(config_path: str = "configs/config.yaml") -> dict:
    # Resolve relative to this script's directory so it works from any cwd
    p = Path(config_path)
    if not p.is_absolute():
        p = Path(__file__).parent / config_path
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(args: argparse.Namespace) -> int:
    config = load_config()
    setup_logging(
        config["logging"]["level"],
        config["logging"].get("file"),
    )
    log = logging.getLogger(__name__)

    seed = args.seed if args.seed is not None else config["dataset"]["seed"]

    console.print(Panel.fit(
        "[bold cyan]AI Safety Benchmark Dataset Generator[/bold cyan]\n"
        f"Version {config['dataset']['version']} · Seed {seed}",
        border_style="cyan",
    ))

    # ── 1. Generate ────────────────────────────────────────────────────────────
    console.print("\n[bold]Step 1/4:[/bold] Generating records…")
    t0 = time.perf_counter()
    records = generate_dataset(seed=seed)
    console.print(f"  [green]OK[/green] Generated [green]{len(records)}[/green] records in "
                  f"{time.perf_counter()-t0:.2f}s")

    # ── 2. Validate ────────────────────────────────────────────────────────────
    console.print("\n[bold]Step 2/4:[/bold] Validating dataset…")
    result = validate_dataset(records)
    if result.is_valid:
        console.print(f"  [OK] All checks passed ({len(result.passed)} checks)")
    else:
        console.print("[red]  [FAIL] Validation FAILED:[/red]")
        for failure in result.failed:
            console.print(f"     - {failure}")
        return 1

    if result.warnings:
        for w in result.warnings:
            console.print(f"  [yellow][WARN]  {w}[/yellow]")

    # ── 3. Statistics ──────────────────────────────────────────────────────────
    console.print("\n[bold]Step 3/4:[/bold] Computing statistics…")
    stats = compute_statistics(records)

    table = Table(title="Category Distribution", show_header=True, header_style="bold cyan")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("% of Total", justify="right")
    for cat, count in sorted(stats["category_distribution"].items(), key=lambda x: -x[1]):
        pct = count / stats["total_records"] * 100
        table.add_row(cat.replace("_", " ").title(), str(count), f"{pct:.1f}%")
    console.print(table)

    sev_table = Table(title="Severity Distribution", show_header=True, header_style="bold magenta")
    sev_table.add_column("Severity", style="magenta")
    sev_table.add_column("Count", justify="right")
    for sev, count in sorted(stats["severity_distribution"].items(), key=lambda x: -x[1]):
        sev_table.add_row(sev.title(), str(count))
    console.print(sev_table)

    console.print(
        f"  Risk Score — min: [red]{stats['risk_score']['min']}[/red]  "
        f"mean: [yellow]{stats['risk_score']['mean']}[/yellow]  "
        f"max: [red]{stats['risk_score']['max']}[/red]"
    )

    # ── 4. Export ──────────────────────────────────────────────────────────────
    console.print("\n[bold]Step 4/4:[/bold] Exporting dataset…")

    json_path = export_json(records, config["output"]["json_path"])
    console.print(f"  [OK] JSON    → [green]{json_path}[/green]")

    parquet_path = export_parquet(records, config["output"]["parquet_path"])
    console.print(f"  [OK] Parquet → [green]{parquet_path}[/green]")

    card_path = export_dataset_card(records, Path(config["output"]["json_path"]).parent)
    console.print(f"  [OK] README  → [green]{card_path}[/green]")

    raw_path = export_json(records, config["output"]["raw_path"])
    console.print(f"  [OK] Raw     → [green]{raw_path}[/green]")

    console.print(Panel.fit(
        "[bold green][OK] Dataset generation complete![/bold green]\n"
        f"  Records  : {stats['total_records']}\n"
        f"  JSON     : {json_path}\n"
        f"  Parquet  : {parquet_path}\n"
        f"  Run tests: [cyan]pytest tests/ -v[/cyan]",
        border_style="green",
    ))
    log.info("Pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AI Safety Benchmark Dataset")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    sys.exit(main(parser.parse_args()))
