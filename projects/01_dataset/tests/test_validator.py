"""
test_validator.py — Unit tests for the dataset validator.
"""

import pytest
from src.generator import generate_dataset
from src.validator import validate_dataset, compute_statistics
from src.schema import SafetyRecord, Category, Severity, ExpectedBehaviour


class TestValidateDataset:

    @pytest.fixture(scope="class")
    def records(self):
        return generate_dataset(seed=42)

    def test_valid_dataset_passes(self, records):
        result = validate_dataset(records)
        assert result.is_valid, f"Validation failed:\n{result.summary()}"

    def test_valid_dataset_no_failures(self, records):
        result = validate_dataset(records)
        assert result.failed == []

    def test_wrong_count_fails(self, records):
        truncated = records[:40]
        result = validate_dataset(truncated)
        assert not result.is_valid
        assert any("Record count" in f for f in result.failed)

    def test_duplicate_ids_fail(self, records):
        duped = records + [records[0]]
        result = validate_dataset(duped)
        assert not result.is_valid
        assert any("Duplicate" in f for f in result.failed)

    def test_summary_contains_pass_count(self, records):
        result = validate_dataset(records)
        summary = result.summary()
        assert "Passed" in summary


class TestComputeStatistics:

    @pytest.fixture(scope="class")
    def records(self):
        return generate_dataset(seed=42)

    def test_total_records(self, records):
        stats = compute_statistics(records)
        assert stats["total_records"] == 50

    def test_all_categories_present(self, records):
        stats = compute_statistics(records)
        assert len(stats["category_distribution"]) == 6

    def test_risk_score_mean_in_range(self, records):
        stats = compute_statistics(records)
        mean = stats["risk_score"]["mean"]
        assert 0.0 <= mean <= 10.0

    def test_harmful_benign_sum(self, records):
        stats = compute_statistics(records)
        assert stats["harmful_intent_count"] + stats["benign_count"] == 50

    def test_tags_list_non_empty(self, records):
        stats = compute_statistics(records)
        assert len(stats["tags_used"]) > 0
