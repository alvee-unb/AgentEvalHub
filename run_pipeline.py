"""
run_pipeline.py — Single-command orchestrator for the full AgentEvalHub pipeline.

Architecture:
    Load Dataset  ->  Run Evaluation  ->  Log MLflow  ->  Generate Report

Usage:
    python run_pipeline.py
    python run_pipeline.py --config pipeline_config.yaml
    python run_pipeline.py --steps dataset track report
    python run_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ── Data classes ──────────────────────────────────────────────────────────────

class StepResult:
    def __init__(self, step_id: str, name: str):
        self.step_id  = step_id
        self.name     = name
        self.status   = "pending"   # pending | running | success | failed | skipped
        self.returncode: int | None = None
        self.duration_s: float = 0.0
        self.error: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "success"


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(config: dict) -> None:
    log_cfg = config.get("logging", {})
    log_file = log_cfg.get("file", "logs/pipeline.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO"), logging.INFO),
        format=log_cfg.get("format", "%(asctime)s | %(levelname)-8s | %(message)s"),
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


# ── Step runner ───────────────────────────────────────────────────────────────

def _python_in_venv(venv: str) -> str:
    """Return the Python executable path inside a venv."""
    base = Path(venv)
    for candidate in [base / "Scripts" / "python.exe", base / "bin" / "python"]:
        if candidate.exists():
            return str(candidate)
    return sys.executable  # fallback: current interpreter


def run_step(step: dict, dry_run: bool = False) -> StepResult:
    result = StepResult(step["id"], step["name"])
    log = logging.getLogger("pipeline")

    script   = step["script"]
    venv     = step.get("venv", "")
    args     = step.get("args", [])
    timeout  = step.get("timeout", 300)
    retries  = step.get("retry", 1)
    python   = _python_in_venv(venv) if venv else sys.executable

    cmd = [python, "-u", script, *[str(a) for a in args]]

    log.info("[%s] Starting: %s", step["id"], step["name"])

    if dry_run:
        console.print(f"  [cyan][DRY RUN][/cyan] Would run: {' '.join(cmd)}")
        result.status = "success"
        return result

    result.status = "running"
    t0 = time.perf_counter()

    for attempt in range(1, retries + 1):
        try:
            proc = subprocess.run(
                cmd,
                capture_output=False,
                timeout=timeout,
                env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
            )
            result.returncode = proc.returncode
            result.duration_s = time.perf_counter() - t0

            if proc.returncode == 0:
                result.status = "success"
                log.info("[%s] OK (%.1fs)", step["id"], result.duration_s)
                break
            else:
                log.warning("[%s] Exit %d (attempt %d/%d)", step["id"], proc.returncode, attempt, retries)
                if attempt < retries:
                    time.sleep(2)
        except subprocess.TimeoutExpired:
            result.status = "failed"
            result.error  = f"Timeout after {timeout}s"
            result.duration_s = time.perf_counter() - t0
            log.error("[%s] Timeout", step["id"])
            break
        except Exception as exc:
            result.status = "failed"
            result.error  = str(exc)
            result.duration_s = time.perf_counter() - t0
            log.error("[%s] Exception: %s", step["id"], exc)
            break
    else:
        result.status = "failed"

    if result.status != "success":
        result.status = "failed"

    return result


# ── Display ───────────────────────────────────────────────────────────────────

def display_plan(steps: list[dict]) -> None:
    table = Table(title="Pipeline Steps", show_header=True, header_style="bold cyan")
    table.add_column("#",      width=3)
    table.add_column("ID",     style="cyan",    width=12)
    table.add_column("Name",   style="white",   width=28)
    table.add_column("Script", style="dim",     width=45)
    table.add_column("Retry",  justify="right", width=6)

    for i, s in enumerate(steps, 1):
        table.add_row(str(i), s["id"], s["name"], s["script"], str(s.get("retry", 1)))
    console.print(table)


def display_summary(results: list[StepResult], total_s: float) -> None:
    table = Table(title="Pipeline Results", show_header=True, header_style="bold green")
    table.add_column("Step",     style="cyan",  width=14)
    table.add_column("Name",               width=28)
    table.add_column("Status",  justify="center", width=10)
    table.add_column("Time(s)", justify="right",  width=8)

    for r in results:
        if r.status == "success":
            status = "[green]OK[/green]"
        elif r.status == "skipped":
            status = "[yellow]SKIP[/yellow]"
        else:
            status = "[red]FAIL[/red]"
        table.add_row(r.step_id, r.name, status, f"{r.duration_s:.1f}")

    console.print(table)
    passed = sum(1 for r in results if r.passed)
    console.print(
        f"\n  [bold]{'[green]' if passed == len(results) else '[yellow]'}"
        f"{passed}/{len(results)} steps passed in {total_s:.1f}s[/bold]"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="AgentEvalHub Pipeline Orchestrator")
    parser.add_argument("--config",  default="pipeline_config.yaml")
    parser.add_argument("--steps",   nargs="*", help="Run only these step IDs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list",    action="store_true", help="List steps and exit")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    setup_logging(config)
    log = logging.getLogger("pipeline")

    console.print(Panel.fit(
        f"[bold cyan]AgentEvalHub Pipeline v{config['pipeline']['version']}[/bold cyan]\n"
        f"{config['pipeline']['description']}",
        border_style="cyan",
    ))

    all_steps = [s for s in config["pipeline"]["steps"] if s.get("enabled", True)]

    if args.steps:
        all_steps = [s for s in all_steps if s["id"] in args.steps]

    if args.list or args.dry_run:
        display_plan(all_steps)
        if args.list:
            return 0

    on_failure = config["pipeline"].get("on_failure", "stop")
    results: list[StepResult] = []
    pipeline_start = time.perf_counter()

    for step in all_steps:
        console.print(f"\n[bold yellow]Step [{step['id']}]:[/bold yellow] {step['name']}")
        r = run_step(step, dry_run=args.dry_run)
        results.append(r)

        if not r.passed and on_failure == "stop":
            log.error("Pipeline halted at step [%s]: %s", r.step_id, r.error)
            console.print("  [red]FAILED — pipeline stopped.[/red]")
            break
        elif not r.passed:
            console.print("  [yellow]Step failed — continuing (on_failure=continue)[/yellow]")

    total_s = time.perf_counter() - pipeline_start
    display_summary(results, total_s)

    all_passed = all(r.passed for r in results)
    log.info("Pipeline %s in %.1fs", "PASSED" if all_passed else "FAILED", total_s)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
