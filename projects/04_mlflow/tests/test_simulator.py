"""test_simulator.py — Tests for the evaluation simulator."""

import pytest
import json
import tempfile
from pathlib import Path

from src.simulator import simulate_eval_run, MODEL_PROFILES
from src.schemas import SampleResult

DATASET_PATH = Path(__file__).parent.parent.parent / "01_dataset/data/processed/ai_safety_benchmark.json"


@pytest.fixture(scope="module")
def phi4_results():
    if not DATASET_PATH.exists():
        pytest.skip("Project 1 dataset not found")
    return simulate_eval_run("phi4:latest", DATASET_PATH, seed=42)


class TestSimulateEvalRun:
    def test_returns_50_results(self, phi4_results):
        assert len(phi4_results) == 50

    def test_all_are_sample_results(self, phi4_results):
        assert all(isinstance(r, SampleResult) for r in phi4_results)

    def test_scores_in_range(self, phi4_results):
        for r in phi4_results:
            assert 0.0 <= r.refusal_score <= 1.0
            assert 0.0 <= r.composite_score <= 1.0

    def test_latency_positive(self, phi4_results):
        assert all(r.latency_ms > 0 for r in phi4_results)

    def test_deterministic_with_seed(self):
        if not DATASET_PATH.exists():
            pytest.skip("Dataset not found")
        r1 = simulate_eval_run("phi4:latest", DATASET_PATH, seed=42)
        r2 = simulate_eval_run("phi4:latest", DATASET_PATH, seed=42)
        assert [s.refused for s in r1] == [s.refused for s in r2]

    def test_different_seeds_differ(self):
        if not DATASET_PATH.exists():
            pytest.skip("Dataset not found")
        r1 = simulate_eval_run("phi4:latest", DATASET_PATH, seed=1)
        r2 = simulate_eval_run("phi4:latest", DATASET_PATH, seed=99)
        assert [s.refused for s in r1] != [s.refused for s in r2]

    def test_phi4_better_than_gpt_oss(self):
        if not DATASET_PATH.exists():
            pytest.skip("Dataset not found")
        phi4    = simulate_eval_run("phi4:latest",  DATASET_PATH, seed=42)
        gpt_oss = simulate_eval_run("gpt-oss:20b", DATASET_PATH, seed=42)
        phi4_mean    = sum(r.composite_score for r in phi4)    / len(phi4)
        gpt_oss_mean = sum(r.composite_score for r in gpt_oss) / len(gpt_oss)
        assert phi4_mean > gpt_oss_mean  # phi4 has better profile

    def test_phi4_faster_than_gpt_oss(self):
        if not DATASET_PATH.exists():
            pytest.skip("Dataset not found")
        phi4    = simulate_eval_run("phi4:latest",  DATASET_PATH, seed=42)
        gpt_oss = simulate_eval_run("gpt-oss:20b", DATASET_PATH, seed=42)
        phi4_lat    = sum(r.latency_ms for r in phi4)    / len(phi4)
        gpt_oss_lat = sum(r.latency_ms for r in gpt_oss) / len(gpt_oss)
        assert phi4_lat < gpt_oss_lat   # phi4 smaller model = faster


class TestModelProfiles:
    def test_both_models_defined(self):
        assert "phi4:latest" in MODEL_PROFILES
        assert "gpt-oss:20b" in MODEL_PROFILES

    def test_profiles_have_required_keys(self):
        required = {"refusal_probability", "jailbreak_resistance",
                    "hallucination_caution", "base_latency_ms", "latency_jitter"}
        for profile in MODEL_PROFILES.values():
            assert required.issubset(profile.keys())

    def test_probabilities_in_range(self):
        for profile in MODEL_PROFILES.values():
            assert 0 < profile["refusal_probability"] <= 1
            assert 0 < profile["jailbreak_resistance"] <= 1
            assert 0 < profile["hallucination_caution"] <= 1
