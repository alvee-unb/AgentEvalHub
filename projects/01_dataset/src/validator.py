"""
validator.py — Validates the generated dataset for integrity and consistency.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from pydantic import ValidationError
from src.schema import SafetyRecord, Category, Severity

logger = logging.getLogger(__name__)


class ValidationResult:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.warnings: list[str] = []

    @property
    def is_valid(self) -> bool:
        return len(self.failed) == 0

    def summary(self) -> str:
        lines = [
            f"  ✅ Passed : {len(self.passed)}",
            f"  ❌ Failed : {len(self.failed)}",
            f"  ⚠️  Warnings: {len(self.warnings)}",
        ]
        if self.failed:
            lines.append("\nFailures:")
            lines.extend(f"  - {f}" for f in self.failed)
        if self.warnings:
            lines.append("\nWarnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


def validate_dataset(records: list[SafetyRecord]) -> ValidationResult:
    """Run all validation checks and return a ValidationResult."""
    result = ValidationResult()

    # 1. Total record count
    if len(records) == 50:
        result.passed.append(f"Record count: {len(records)} == 50")
    else:
        result.failed.append(f"Record count: expected 50, got {len(records)}")

    # 2. No duplicate IDs
    ids = [r.id for r in records]
    if len(ids) == len(set(ids)):
        result.passed.append("All IDs are unique")
    else:
        dupes = [id_ for id_, count in Counter(ids).items() if count > 1]
        result.failed.append(f"Duplicate IDs found: {dupes}")

    # 3. Category distribution
    cat_counts = Counter(r.category for r in records)
    expected = {
        Category.PROMPT_INJECTION: 9,
        Category.JAILBREAK:        9,
        Category.HALLUCINATION:    8,
        Category.TOOL_ABUSE:       8,
        Category.DATA_LEAKAGE:     8,
        Category.UNSAFE_ADVICE:    8,
    }
    for cat, exp_count in expected.items():
        actual = cat_counts.get(cat, 0)
        if actual == exp_count:
            result.passed.append(f"Category '{cat.value}': {actual} records")
        else:
            result.failed.append(
                f"Category '{cat.value}': expected {exp_count}, got {actual}"
            )

    # 4. Risk score range
    out_of_range = [
        r.id for r in records if not (0.0 <= r.risk_score <= 10.0)
    ]
    if not out_of_range:
        result.passed.append("All risk scores in [0.0, 10.0]")
    else:
        result.failed.append(f"Risk scores out of range for IDs: {out_of_range}")

    # 5. Critical records must be harmful
    bad_critical = [
        r.id for r in records
        if r.severity == Severity.CRITICAL and not r.harmful_intent
    ]
    if not bad_critical:
        result.passed.append("All CRITICAL records have harmful_intent=True")
    else:
        result.failed.append(
            f"CRITICAL records without harmful_intent: {bad_critical}"
        )

    # 6. No blank prompts
    blank_prompts = [r.id for r in records if not r.prompt.strip()]
    if not blank_prompts:
        result.passed.append("All prompts are non-empty")
    else:
        result.failed.append(f"Blank prompts found in records: {blank_prompts}")

    # 7. Prompt length sanity check (10–2000 chars)
    long_prompts = [r.id for r in records if len(r.prompt) > 2000]
    short_prompts = [r.id for r in records if len(r.prompt) < 10]
    if not long_prompts:
        result.passed.append("All prompts within max length (2000 chars)")
    else:
        result.warnings.append(f"Prompts exceeding 2000 chars: {long_prompts}")
    if not short_prompts:
        result.passed.append("All prompts meet minimum length (10 chars)")
    else:
        result.failed.append(f"Prompts below minimum length: {short_prompts}")

    # 8. Timestamps present and not empty
    missing_ts = [r.id for r in records if not r.created_at]
    if not missing_ts:
        result.passed.append("All records have timestamps")
    else:
        result.failed.append(f"Missing timestamps: {missing_ts}")

    # 9. Severity distribution — at least one of each severity
    sev_counts = Counter(r.severity for r in records)
    for sev in Severity:
        if sev_counts.get(sev, 0) > 0:
            result.passed.append(f"Severity '{sev.value}' present ({sev_counts[sev]} records)")
        else:
            result.warnings.append(f"Severity '{sev.value}' has zero records")

    # 10. Pydantic re-validation round-trip
    failures = []
    for r in records:
        try:
            SafetyRecord.model_validate(r.model_dump())
        except ValidationError as e:
            failures.append(f"{r.id}: {e.error_count()} errors")
    if not failures:
        result.passed.append("All records pass Pydantic round-trip validation")
    else:
        result.failed.extend(failures)

    return result


def compute_statistics(records: list[SafetyRecord]) -> dict[str, Any]:
    """Return summary statistics for the dataset."""
    cat_counts = Counter(r.category.value for r in records)
    sev_counts = Counter(r.severity.value for r in records)
    risk_scores = [r.risk_score for r in records]
    harmful_count = sum(1 for r in records if r.harmful_intent)

    return {
        "total_records": len(records),
        "category_distribution": dict(cat_counts),
        "severity_distribution": dict(sev_counts),
        "harmful_intent_count": harmful_count,
        "benign_count": len(records) - harmful_count,
        "risk_score": {
            "min": min(risk_scores),
            "max": max(risk_scores),
            "mean": round(sum(risk_scores) / len(risk_scores), 2),
        },
        "tags_used": list({tag.value for r in records for tag in r.tags}),
    }
