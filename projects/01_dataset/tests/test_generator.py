"""
test_generator.py — Unit tests for the dataset generator.
"""

import pytest
from collections import Counter

from src.generator import generate_dataset
from src.schema import Category, Severity


class TestGenerateDataset:

    @pytest.fixture(scope="class")
    def records(self):
        return generate_dataset(seed=42)

    def test_returns_50_records(self, records):
        assert len(records) == 50

    def test_deterministic_with_same_seed(self):
        r1 = generate_dataset(seed=42)
        r2 = generate_dataset(seed=42)
        assert [r.id for r in r1] == [r.id for r in r2]

    def test_different_seeds_produce_different_ids(self):
        r1 = generate_dataset(seed=42)
        r2 = generate_dataset(seed=99)
        assert [r.id for r in r1] != [r.id for r in r2]

    def test_all_ids_unique(self, records):
        ids = [r.id for r in records]
        assert len(ids) == len(set(ids))

    def test_category_distribution(self, records):
        counts = Counter(r.category for r in records)
        assert counts[Category.PROMPT_INJECTION] == 9
        assert counts[Category.JAILBREAK]        == 9
        assert counts[Category.HALLUCINATION]    == 8
        assert counts[Category.TOOL_ABUSE]       == 8
        assert counts[Category.DATA_LEAKAGE]     == 8
        assert counts[Category.UNSAFE_ADVICE]    == 8

    def test_all_severities_represented(self, records):
        severities = {r.severity for r in records}
        assert Severity.LOW      in severities
        assert Severity.MEDIUM   in severities
        assert Severity.HIGH     in severities
        assert Severity.CRITICAL in severities

    def test_risk_scores_in_range(self, records):
        assert all(0.0 <= r.risk_score <= 10.0 for r in records)

    def test_critical_records_are_harmful(self, records):
        critical = [r for r in records if r.severity == Severity.CRITICAL]
        assert all(r.harmful_intent for r in critical)
        assert len(critical) > 0

    def test_all_prompts_non_empty(self, records):
        assert all(len(r.prompt.strip()) >= 10 for r in records)

    def test_all_records_have_timestamps(self, records):
        assert all(r.created_at for r in records)

    def test_source_is_synthetic(self, records):
        assert all(r.source == "synthetic" for r in records)

    def test_version_set(self, records):
        assert all(r.version == "1.0.0" for r in records)
