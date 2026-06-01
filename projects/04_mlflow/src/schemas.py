"""
schemas.py — Pydantic schemas for MLflow evaluation run data.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class SampleResult(BaseModel):
    """Result of evaluating a single dataset sample."""
    sample_id: str
    category: str
    severity: str
    harmful_intent: bool
    risk_score: float
    prompt: str
    response: str
    refused: bool
    jailbroken: bool
    hallucinated: bool
    refusal_score: float        # 0.0 – 1.0
    jailbreak_score: float      # 0.0 – 1.0
    hallucination_score: float  # 0.0 – 1.0
    composite_score: float      # 0.0 – 1.0
    latency_ms: float


class EvalRunResult(BaseModel):
    """Aggregated results for a full evaluation run."""
    run_id: str
    model_name: str
    benchmark: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Aggregate metrics
    total_samples: int
    refusal_rate: float        # fraction of harmful prompts refused
    safety_score: float        # mean composite safety score
    jailbreak_success_rate: float  # fraction of jailbreaks that succeeded
    hallucination_rate: float  # fraction of hallucination prompts where model hallucinated

    # Latency
    latency_ms_mean: float
    latency_ms_p95: float
    latency_ms_total: float

    # Per-sample breakdown
    samples: list[SampleResult] = Field(default_factory=list)

    # Pass/fail against thresholds
    passed_refusal_threshold: bool = False
    passed_jailbreak_threshold: bool = False
    passed_safety_threshold: bool = False

    @property
    def overall_pass(self) -> bool:
        return (
            self.passed_refusal_threshold
            and self.passed_jailbreak_threshold
            and self.passed_safety_threshold
        )
