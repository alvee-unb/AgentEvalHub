"""
dataset_loader.py — Loads the Project 1 safety benchmark dataset and converts
it to Inspect AI Dataset format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample

logger = logging.getLogger(__name__)

# Harmful categories for refusal-rate calculation
HARMFUL_CATEGORIES = {
    "prompt_injection",
    "jailbreak",
    "tool_abuse",
    "data_leakage",
}

HALLUCINATION_CATEGORIES = {"hallucination"}


def load_raw_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    logger.info("Loaded %d records from %s", len(records), path)
    return records


def records_to_inspect_dataset(
    records: list[dict[str, Any]],
    category_filter: list[str] | None = None,
    max_records: int | None = None,
) -> Dataset:
    """
    Convert raw JSON records to an Inspect AI Dataset.

    Each Sample carries:
      - input:    the user prompt
      - target:   expected_behaviour string
      - metadata: full record dict for use in scorers
    """
    if category_filter:
        records = [r for r in records if r["category"] in category_filter]

    if max_records:
        records = records[:max_records]

    samples = []
    for rec in records:
        sample = Sample(
            id=rec["id"],
            input=rec["prompt"],
            target=rec["expected_behaviour"],
            metadata={
                "category":          rec["category"],
                "severity":          rec["severity"],
                "harmful_intent":    rec["harmful_intent"],
                "risk_score":        rec["risk_score"],
                "attack_vector":     rec.get("attack_vector"),
                "system_prompt":     rec.get("system_prompt"),
                "tags":              rec.get("tags", ""),
            },
        )
        samples.append(sample)

    logger.info("Built Inspect AI dataset: %d samples", len(samples))
    return MemoryDataset(samples=samples, name="ai-safety-benchmark")


def load_inspect_dataset(
    path: str | Path,
    category_filter: list[str] | None = None,
    max_records: int | None = None,
) -> Dataset:
    records = load_raw_records(path)
    return records_to_inspect_dataset(records, category_filter, max_records)
