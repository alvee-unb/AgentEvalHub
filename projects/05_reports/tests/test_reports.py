"""test_reports.py — Tests for HTML and Markdown report generators."""

import pytest
from pathlib import Path
from src.data_loader import get_report_data
from src.html_report import generate_html, _cell, _thresh_str, _recommendations
from src.markdown_report import generate_markdown, _pass_fail


SAMPLE_DATA = {
    "title": "Test Report",
    "dataset": "test-dataset v1.0",
    "categories": ["Prompt Injection", "Jailbreak"],
    "models": ["phi4:latest", "gpt-oss:20b"],
    "runs": [
        {
            "run_id": "run-1", "model": "phi4:latest", "benchmark": "test",
            "timestamp": "2025-01-01T00:00:00Z", "overall_pass": True,
            "metrics": {
                "refusal_rate": 0.95, "safety_score": 0.90,
                "jailbreak_success_rate": 0.02, "hallucination_rate": 0.10,
                "latency_ms_mean": 1200.0, "latency_ms_p95": 1800.0,
            },
        },
        {
            "run_id": "run-2", "model": "gpt-oss:20b", "benchmark": "test",
            "timestamp": "2025-01-01T00:01:00Z", "overall_pass": False,
            "metrics": {
                "refusal_rate": 0.70, "safety_score": 0.82,
                "jailbreak_success_rate": 0.05, "hallucination_rate": 0.35,
                "latency_ms_mean": 2800.0, "latency_ms_p95": 4000.0,
            },
        },
    ],
    "metrics_table": {
        "phi4:latest": {"refusal_rate": 0.95, "safety_score": 0.90,
                        "jailbreak_success_rate": 0.02, "hallucination_rate": 0.10,
                        "latency_ms_mean": 1200.0, "latency_ms_p95": 1800.0},
        "gpt-oss:20b": {"refusal_rate": 0.70, "safety_score": 0.82,
                        "jailbreak_success_rate": 0.05, "hallucination_rate": 0.35,
                        "latency_ms_mean": 2800.0, "latency_ms_p95": 4000.0},
    },
    "thresholds": {"refusal_rate": 0.80, "safety_score": 0.75,
                   "jailbreak_success_rate": 0.10, "hallucination_rate": 0.20},
}


class TestHtmlReport:
    def test_creates_file(self, tmp_path):
        p = generate_html(SAMPLE_DATA, tmp_path / "test.html")
        assert p.exists()

    def test_file_non_empty(self, tmp_path):
        p = generate_html(SAMPLE_DATA, tmp_path / "test.html")
        assert p.stat().st_size > 1000

    def test_contains_model_names(self, tmp_path):
        p = generate_html(SAMPLE_DATA, tmp_path / "test.html")
        content = p.read_text()
        assert "phi4:latest" in content
        assert "gpt-oss:20b" in content

    def test_contains_pass_fail(self, tmp_path):
        p = generate_html(SAMPLE_DATA, tmp_path / "test.html")
        content = p.read_text()
        assert "PASS" in content
        assert "FAIL" in content

    def test_valid_html_structure(self, tmp_path):
        p = generate_html(SAMPLE_DATA, tmp_path / "test.html")
        content = p.read_text()
        assert "<html" in content
        assert "</html>" in content
        assert "<table" in content


class TestMarkdownReport:
    def test_creates_file(self, tmp_path):
        p = generate_markdown(SAMPLE_DATA, tmp_path / "test.md")
        assert p.exists()

    def test_file_non_empty(self, tmp_path):
        p = generate_markdown(SAMPLE_DATA, tmp_path / "test.md")
        assert p.stat().st_size > 500

    def test_contains_model_names(self, tmp_path):
        p = generate_markdown(SAMPLE_DATA, tmp_path / "test.md")
        content = p.read_text()
        assert "phi4:latest" in content
        assert "gpt-oss:20b" in content

    def test_contains_table(self, tmp_path):
        p = generate_markdown(SAMPLE_DATA, tmp_path / "test.md")
        content = p.read_text()
        assert "|" in content

    def test_has_recommendations_section(self, tmp_path):
        p = generate_markdown(SAMPLE_DATA, tmp_path / "test.md")
        content = p.read_text()
        assert "Recommendations" in content


class TestHelperFunctions:
    def test_pass_fail_higher_better(self):
        assert _pass_fail(0.90, 0.80, True) == "PASS"
        assert _pass_fail(0.70, 0.80, True) == "FAIL"

    def test_pass_fail_lower_better(self):
        assert _pass_fail(0.05, 0.10, False) == "PASS"
        assert _pass_fail(0.15, 0.10, False) == "FAIL"

    def test_pass_fail_no_threshold(self):
        assert _pass_fail(0.5, None, True) == ""

    def test_thresh_str_percent(self):
        assert ">=" in _thresh_str(0.80, True, "%")
        assert "<=" in _thresh_str(0.10, False, "%")

    def test_thresh_str_none(self):
        assert _thresh_str(None, True, "ms") == "—"

    def test_cell_pass(self):
        cell = _cell(0.95, 0.80, True, "%")
        assert cell["css"] == "pass"

    def test_cell_fail(self):
        cell = _cell(0.70, 0.80, True, "%")
        assert cell["css"] == "fail"

    def test_cell_ms_no_css(self):
        cell = _cell(1200.0, None, False, "ms")
        assert cell["css"] == ""
        assert "ms" in cell["display"]

    def test_recommendations_all_pass(self):
        run = {"overall_pass": True, "metrics": {"hallucination_rate": 0.10,
               "refusal_rate": 0.95, "jailbreak_success_rate": 0.01}}
        recs = _recommendations(run, {"hallucination_rate": 0.20, "refusal_rate": 0.80,
                                       "jailbreak_success_rate": 0.10})
        assert any("meets all" in r for r in recs)


class TestDataLoader:
    def test_get_report_data_returns_dict(self):
        data = get_report_data()
        assert isinstance(data, dict)
        assert "runs" in data
        assert "models" in data

    def test_report_data_has_required_keys(self):
        data = get_report_data()
        required = {"title", "dataset", "runs", "models", "metrics_table", "thresholds"}
        assert required.issubset(data.keys())

    def test_at_least_one_model(self):
        data = get_report_data()
        assert len(data["models"]) >= 1
