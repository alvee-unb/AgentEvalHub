"""
safety_tasks.py — Inspect AI @task definitions for the safety benchmark.

Tasks:
  - safety_benchmark:     Full 50-record evaluation
  - refusal_benchmark:    Harmful-prompt-only refusal test
  - jailbreak_benchmark:  Jailbreak + prompt injection only
  - hallucination_benchmark: Hallucination category only
"""

from __future__ import annotations

import sys
from pathlib import Path

# SSL fix must apply before inspect_ai imports httpx at module level
import os, certifi as _certifi
cert = os.environ.get("SSL_CERT_FILE", "")
if cert and not Path(cert).exists():
    os.environ["SSL_CERT_FILE"] = _certifi.where()

sys.path.insert(0, str(Path(__file__).parent.parent))

from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.scorer import accuracy, mean

from src.dataset_loader import load_inspect_dataset
from scorers.safety_scorers import (
    refusal_score,
    jailbreak_score,
    hallucination_score,
    composite_safety_score,
)

_DATASET_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"


def _get_dataset_path() -> str:
    import yaml
    cfg_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["evaluation"]["dataset_path"]


@task
def safety_benchmark() -> Task:
    """
    Full safety benchmark: all 50 records, all four scorers.
    Run with: inspect eval tasks/safety_tasks.py::safety_benchmark --model ollama/phi4:latest
    """
    dataset = load_inspect_dataset(_get_dataset_path())
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=[
            refusal_score(),
            jailbreak_score(),
            hallucination_score(),
            composite_safety_score(),
        ],
    )


@task
def refusal_benchmark() -> Task:
    """
    Refusal-only benchmark: harmful prompts (prompt_injection, jailbreak,
    tool_abuse, data_leakage). Tests whether the model correctly refuses.
    """
    dataset = load_inspect_dataset(
        _get_dataset_path(),
        category_filter=["prompt_injection", "jailbreak", "tool_abuse", "data_leakage"],
    )
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=[refusal_score(), composite_safety_score()],
    )


@task
def jailbreak_benchmark() -> Task:
    """
    Jailbreak resistance: only prompt_injection and jailbreak categories.
    """
    dataset = load_inspect_dataset(
        _get_dataset_path(),
        category_filter=["prompt_injection", "jailbreak"],
    )
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=[jailbreak_score(), refusal_score()],
    )


@task
def hallucination_benchmark() -> Task:
    """
    Hallucination resistance: only hallucination category.
    """
    dataset = load_inspect_dataset(
        _get_dataset_path(),
        category_filter=["hallucination"],
    )
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=[hallucination_score(), composite_safety_score()],
    )
