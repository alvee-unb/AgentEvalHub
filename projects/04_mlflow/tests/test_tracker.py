"""test_tracker.py — Tests for the MLflow tracker and compute_run_result."""

import pytest
import tempfile
from pathlib import Path

from src.schemas import SampleResult
from src.tracker import compute_run_result, SafetyEvalTracker


def make_samples(n: int = 10, harmful_frac: float = 0.6, refused_frac: float = 0.9) -> list[SampleResult]:
    samples = []
    for i in range(n):
        harmful = i < int(n * harmful_frac)
        refused = i < int(n * refused_frac)
        cat = "prompt_injection" if harmful else "hallucination"
        samples.append(SampleResult(
            sample_id=f"s-{i:03d}",
            category=cat,
            severity="high",
            harmful_intent=harmful,
            risk_score=8.0 if harmful else 3.0,
            prompt="Test prompt.",
            response="I cannot do that." if refused else "Sure, here you go.",
            refused=refused,
            jailbroken=False,
            hallucinated=False,
            refusal_score=1.0 if refused else 0.0,
            jailbreak_score=1.0,
            hallucination_score=1.0,
            composite_score=0.9 if refused else 0.3,
            latency_ms=float(300 + i * 10),
        ))
    return samples


class TestComputeRunResult:
    def test_basic_computation(self):
        samples = make_samples(10)
        result = compute_run_result("phi4:latest", "test_bench", samples)
        assert result.total_samples == 10
        assert 0.0 <= result.refusal_rate <= 1.0
        assert 0.0 <= result.safety_score <= 1.0

    def test_empty_samples_raises(self):
        with pytest.raises(ValueError):
            compute_run_result("phi4:latest", "bench", [])

    def test_latency_computed(self):
        samples = make_samples(5)
        result = compute_run_result("phi4:latest", "bench", samples)
        assert result.latency_ms_mean > 0
        assert result.latency_ms_p95 >= result.latency_ms_mean

    def test_high_refusal_passes_threshold(self):
        samples = make_samples(20, harmful_frac=0.8, refused_frac=1.0)
        result = compute_run_result("phi4:latest", "bench", samples)
        assert result.passed_refusal_threshold is True

    def test_low_refusal_fails_threshold(self):
        samples = make_samples(20, harmful_frac=0.8, refused_frac=0.0)
        result = compute_run_result("phi4:latest", "bench", samples)
        assert result.passed_refusal_threshold is False

    def test_jailbreak_rate_zero(self):
        samples = make_samples(10)
        result = compute_run_result("phi4:latest", "bench", samples)
        assert result.jailbreak_success_rate == 0.0
        assert result.passed_jailbreak_threshold is True

    def test_run_id_contains_model(self):
        result = compute_run_result("phi4:latest", "bench", make_samples())
        assert "phi4" in result.run_id


class TestSafetyEvalTracker:
    @pytest.fixture
    def config(self, tmp_path):
        # Use SQLite backend — file store was removed in MLflow 3.x
        uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        return {
            "mlflow": {
                "tracking_uri": uri,
                "experiment_name": "test-experiment",
                "artifact_location": str(tmp_path / "artifacts"),
                "tags": {"project": "test", "version": "1.0.0", "dataset": "test"},
            },
            "models": [],
        }

    def test_tracker_initialises(self, config):
        tracker = SafetyEvalTracker(config)
        assert tracker is not None

    def test_log_eval_run_returns_run_id(self, config):
        tracker = SafetyEvalTracker(config)
        samples = make_samples(5)
        result = compute_run_result("phi4:latest", "test_bench", samples)
        run_id = tracker.log_eval_run(result)
        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_multiple_runs_different_ids(self, config):
        tracker = SafetyEvalTracker(config)
        samples = make_samples(5)
        r1 = compute_run_result("phi4:latest",  "bench", samples)
        r2 = compute_run_result("gpt-oss:20b", "bench", samples)
        id1 = tracker.log_eval_run(r1)
        id2 = tracker.log_eval_run(r2)
        assert id1 != id2
