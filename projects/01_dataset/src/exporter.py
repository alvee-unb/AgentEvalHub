"""
exporter.py — Exports the dataset to JSON, Parquet, and HuggingFace-ready format.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.schema import DatasetMetadata, SafetyRecord
from src.validator import compute_statistics

logger = logging.getLogger(__name__)


def _records_to_dicts(records: list[SafetyRecord]) -> list[dict]:
    """Serialise records to plain dicts (enums → string values)."""
    return [
        {
            **r.model_dump(),
            "category": r.category.value,
            "severity": r.severity.value,
            "expected_behaviour": r.expected_behaviour.value,
            "tags": [t.value for t in r.tags],
        }
        for r in records
    ]


def export_json(records: list[SafetyRecord], path: str | Path) -> Path:
    """Export records to a JSON file (HuggingFace data-files compatible)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dicts = _records_to_dicts(records)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dicts, f, indent=2, ensure_ascii=False)

    size_kb = path.stat().st_size / 1024
    logger.info("Exported JSON → %s (%.1f KB, %d records)", path, size_kb, len(records))
    return path


def export_parquet(records: list[SafetyRecord], path: str | Path) -> Path:
    """Export records to a Parquet file (columnar, HuggingFace preferred format)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dicts = _records_to_dicts(records)
    df = pd.DataFrame(dicts)

    # Parquet doesn't support list-of-strings natively without schema hints
    # Convert tags list to pipe-separated string for broad compatibility
    df["tags"] = df["tags"].apply(lambda x: "|".join(x) if x else "")

    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, compression="snappy")

    size_kb = path.stat().st_size / 1024
    logger.info("Exported Parquet → %s (%.1f KB, %d records)", path, size_kb, len(records))
    return path


def export_dataset_card(
    records: list[SafetyRecord],
    output_dir: str | Path,
) -> Path:
    """
    Generate a HuggingFace-style dataset card (README.md) in the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = compute_statistics(records)
    cat_dist = stats["category_distribution"]
    sev_dist = stats["severity_distribution"]

    cat_rows = "\n".join(
        f"| {k.replace('_', ' ').title()} | {v} |"
        for k, v in sorted(cat_dist.items(), key=lambda x: -x[1])
    )
    sev_rows = "\n".join(
        f"| {k.title()} | {v} |"
        for k, v in sorted(sev_dist.items(), key=lambda x: -x[1])
    )

    card = f"""---
language:
- en
license: apache-2.0
task_categories:
- text-classification
tags:
- ai-safety
- red-teaming
- evaluation
- benchmark
pretty_name: AI Safety Benchmark Dataset
size_categories:
- n<1K
---

# AI Safety Benchmark Dataset

A curated benchmark dataset for evaluating AI model safety across six risk categories.
Designed for use with Inspect AI, MLflow, and LangGraph evaluation pipelines.

## Dataset Summary

| Field | Value |
|---|---|
| Records | {stats['total_records']} |
| Categories | 6 |
| Harmful Intent | {stats['harmful_intent_count']} |
| Benign | {stats['benign_count']} |
| Mean Risk Score | {stats['risk_score']['mean']} / 10 |
| License | Apache-2.0 |
| Language | English |

## Category Distribution

| Category | Count |
|---|---|
{cat_rows}

## Severity Distribution

| Severity | Count |
|---|---|
{sev_rows}

## Schema

```python
{{
  "id":                        str,   # UUIDv4
  "version":                   str,
  "category":                  str,   # See categories below
  "severity":                  str,   # low | medium | high | critical
  "tags":                      str,   # pipe-separated evaluation tags
  "prompt":                    str,   # Input prompt for model evaluation
  "system_prompt":             str | None,
  "expected_behaviour":        str,   # refuse | answer_safely | detect_inject | stay_grounded
  "harmful_intent":            bool,
  "ground_truth_safe_response":str | None,
  "risk_score":                float, # 0.0 – 10.0
  "attack_vector":             str | None,
  "source":                    str,
  "created_at":                str,   # ISO-8601 UTC
  "notes":                     str | None
}}
```

## Usage

```python
import json

with open("data/processed/ai_safety_benchmark.json") as f:
    records = json.load(f)

# Filter by category
injections = [r for r in records if r["category"] == "prompt_injection"]
```

## Categories

| ID | Description |
|---|---|
| `prompt_injection` | Attempts to override model instructions via crafted inputs |
| `jailbreak` | Techniques designed to bypass safety training |
| `hallucination` | Queries designed to elicit fabricated or unverifiable information |
| `tool_abuse` | Misuse of tool-calling capabilities for harmful ends |
| `data_leakage` | Attempts to extract system prompts, PII, or configuration |
| `unsafe_advice` | Requests for advice that could cause real-world harm |

## Citation

```bibtex
@dataset{{ai_safety_benchmark_2025,
  title  = {{AI Safety Benchmark Dataset}},
  author = {{AgentEvalHub}},
  year   = {{2025}},
  note   = {{Synthetic dataset for AI evaluation research}}
}}
```
"""

    card_path = output_dir / "README.md"
    card_path.write_text(card, encoding="utf-8")
    logger.info("Dataset card written → %s", card_path)
    return card_path
