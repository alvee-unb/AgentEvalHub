"""
run_evaluation.py — Programmatic Inspect AI evaluation runner.

Usage:
    python run_evaluation.py --task safety_benchmark --model phi4:latest
    python run_evaluation.py --task jailbreak_benchmark --model gpt-oss:20b
    python run_evaluation.py --task all --model phi4:latest
    python run_evaluation.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# SSL fix before any inspect_ai / httpx import
import certifi
cert_env = os.environ.get("SSL_CERT_FILE", "")
if cert_env and not Path(cert_env).exists():
    os.environ["SSL_CERT_FILE"] = certifi.where()

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))

console = Console()

TASK_MAP = {
    "safety_benchmark":      "tasks/safety_tasks.py@safety_benchmark",
    "refusal_benchmark":     "tasks/safety_tasks.py@refusal_benchmark",
    "jailbreak_benchmark":   "tasks/safety_tasks.py@jailbreak_benchmark",
    "hallucination_benchmark": "tasks/safety_tasks.py@hallucination_benchmark",
}


def setup_logging(level: str = "INFO") -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/evaluation.log", encoding="utf-8"),
        ],
    )


def load_config() -> dict:
    p = Path(__file__).parent / "configs/config.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_task(task_spec: str, model: str, results_dir: Path) -> dict:
    """Run a single Inspect AI task and return result summary."""
    from inspect_ai import eval as inspect_eval

    model_spec = f"ollama/{model}"
    log = logging.getLogger(__name__)
    log.info("Running: %s | model=%s", task_spec, model_spec)

    t0 = time.perf_counter()
    results = inspect_eval(task_spec, model=model_spec, log_dir=str(results_dir))
    elapsed = time.perf_counter() - t0

    # Collect metrics from results
    summary = {
        "task": task_spec.split("@")[-1],
        "model": model,
        "elapsed_s": round(elapsed, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scores": {},
        "status": "completed",
    }

    for result in results:
        if result.scores:
            for scorer_name, metric_values in result.scores.items():
                if hasattr(metric_values, "__iter__") and not isinstance(metric_values, str):
                    for metric_name, metric_val in (
                        metric_values.items() if hasattr(metric_values, "items") else []
                    ):
                        key = f"{scorer_name}/{metric_name}"
                        summary["scores"][key] = (
                            round(float(metric_val.value), 4)
                            if hasattr(metric_val, "value") else float(metric_val)
                        )
                else:
                    summary["scores"][scorer_name] = float(metric_values) if metric_values else 0.0

    return summary


def dry_run(config: dict) -> None:
    """Show what would be evaluated without calling the LLM."""
    from src.dataset_loader import load_raw_records
    from collections import Counter

    dataset_path = Path(config["evaluation"]["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).parent / dataset_path
    records = load_raw_records(dataset_path)
    cat_counts = Counter(r["category"] for r in records)

    table = Table(title="[DRY RUN] Dataset Preview", show_header=True, header_style="bold cyan")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        table.add_row(cat.replace("_", " ").title(), str(count))
    console.print(table)

    console.print("\n[bold]Tasks that would run:[/bold]")
    for name in TASK_MAP:
        console.print(f"  - {name}")

    console.print("\n[bold]Models:[/bold]")
    for m in config["models"]:
        console.print(f"  - {m['ollama_name']}")

    console.print("\n[yellow]Add --task and --model flags to run for real.[/yellow]")


def display_summary(summaries: list[dict]) -> None:
    table = Table(title="Evaluation Results", show_header=True, header_style="bold green")
    table.add_column("Task",    style="cyan",   no_wrap=True)
    table.add_column("Model",   style="magenta")
    table.add_column("Status",  style="green")
    table.add_column("Time(s)", justify="right")
    table.add_column("Key Score", justify="right")

    for s in summaries:
        # pick the composite or first available score
        key_score = next(
            (f"{v:.3f}" for k, v in s["scores"].items() if "composite" in k),
            next(iter(f"{v:.3f}" for v in s["scores"].values()), "N/A"),
        )
        table.add_row(
            s["task"], s["model"],
            "[green]OK[/green]" if s["status"] == "completed" else "[red]ERR[/red]",
            str(s["elapsed_s"]),
            key_score,
        )
    console.print(table)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect AI Safety Evaluation Runner")
    parser.add_argument("--task",    choices=list(TASK_MAP) + ["all"], default=None)
    parser.add_argument("--model",   default="phi4:latest")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output",  default="results/eval_results.json")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config["logging"]["level"])

    console.print(Panel.fit(
        f"[bold cyan]Inspect AI Safety Evaluation[/bold cyan]\n"
        f"Version {config['evaluation']['version']}",
        border_style="cyan",
    ))

    if args.dry_run:
        dry_run(config)
        return 0

    if not args.task:
        parser.print_help()
        return 1

    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(exist_ok=True)

    tasks_to_run = list(TASK_MAP.values()) if args.task == "all" else [TASK_MAP[args.task]]

    summaries = []
    for task_spec in tasks_to_run:
        try:
            summary = run_task(task_spec, args.model, results_dir)
            summaries.append(summary)
        except Exception as exc:
            logging.getLogger(__name__).error("Task failed: %s", exc)
            summaries.append({
                "task": task_spec.split("@")[-1],
                "model": args.model,
                "status": "error",
                "error": str(exc),
                "scores": {},
                "elapsed_s": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    console.print(f"\n[green]Results saved:[/green] {output_path}")

    display_summary(summaries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
