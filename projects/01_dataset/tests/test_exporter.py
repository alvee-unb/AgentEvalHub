"""
test_exporter.py — Unit tests for the dataset exporter.
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import pyarrow.parquet as pq

from src.generator import generate_dataset
from src.exporter import export_json, export_parquet, export_dataset_card


@pytest.fixture(scope="module")
def records():
    return generate_dataset(seed=42)


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


class TestExportJson:
    def test_creates_file(self, records, tmp):
        path = export_json(records, tmp / "out.json")
        assert path.exists()

    def test_valid_json(self, records, tmp):
        path = export_json(records, tmp / "out.json")
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_correct_record_count(self, records, tmp):
        path = export_json(records, tmp / "out.json")
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 50

    def test_fields_present(self, records, tmp):
        path = export_json(records, tmp / "out.json")
        with open(path) as f:
            data = json.load(f)
        required = {"id", "category", "severity", "prompt", "risk_score"}
        assert required.issubset(data[0].keys())

    def test_enums_serialised_as_strings(self, records, tmp):
        path = export_json(records, tmp / "out.json")
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data[0]["category"], str)
        assert isinstance(data[0]["severity"], str)


class TestExportParquet:
    def test_creates_file(self, records, tmp):
        path = export_parquet(records, tmp / "out.parquet")
        assert path.exists()

    def test_correct_record_count(self, records, tmp):
        path = export_parquet(records, tmp / "out.parquet")
        table = pq.read_table(path)
        assert table.num_rows == 50

    def test_columns_present(self, records, tmp):
        path = export_parquet(records, tmp / "out.parquet")
        df = pd.read_parquet(path)
        assert "category" in df.columns
        assert "risk_score" in df.columns
        assert "prompt" in df.columns


class TestExportDatasetCard:
    def test_creates_readme(self, records, tmp):
        path = export_dataset_card(records, tmp)
        assert (tmp / "README.md").exists()

    def test_readme_contains_schema(self, records, tmp):
        export_dataset_card(records, tmp)
        content = (tmp / "README.md").read_text()
        assert "risk_score" in content
        assert "category" in content

    def test_readme_contains_stats(self, records, tmp):
        export_dataset_card(records, tmp)
        content = (tmp / "README.md").read_text()
        assert "50" in content
