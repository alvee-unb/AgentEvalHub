"""
schema.py — Pydantic schema definitions for the AI Safety Benchmark Dataset.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import uuid


# ── Enumerations ──────────────────────────────────────────────────────────────

class Category(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK        = "jailbreak"
    HALLUCINATION    = "hallucination"
    TOOL_ABUSE       = "tool_abuse"
    DATA_LEAKAGE     = "data_leakage"
    UNSAFE_ADVICE    = "unsafe_advice"


class Severity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class ExpectedBehaviour(str, Enum):
    REFUSE          = "refuse"           # Model should refuse to answer
    ANSWER_SAFELY   = "answer_safely"    # Model should answer with caveats
    DETECT_INJECT   = "detect_inject"    # Model should flag injection attempt
    STAY_GROUNDED   = "stay_grounded"    # Model should not hallucinate


class EvaluationTag(str, Enum):
    DIRECT_ATTACK    = "direct_attack"
    INDIRECT_ATTACK  = "indirect_attack"
    SOCIAL_ENG       = "social_engineering"
    SYSTEM_PROMPT    = "system_prompt_leak"
    ROLE_PLAY        = "role_play"
    FACTUAL          = "factual"
    MEDICAL          = "medical"
    LEGAL            = "legal"
    FINANCIAL        = "financial"
    SECURITY         = "security"


# ── Core Record Schema ─────────────────────────────────────────────────────────

class SafetyRecord(BaseModel):
    """
    A single safety evaluation record.
    Conforms to HuggingFace datasets schema conventions.
    """

    # Identity
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique record identifier (UUIDv4)"
    )
    version: str = Field(
        default="1.0.0",
        description="Dataset version this record belongs to"
    )

    # Classification
    category: Category = Field(description="Primary risk category")
    severity: Severity = Field(description="Risk severity level")
    tags: list[EvaluationTag] = Field(
        default_factory=list,
        description="Evaluation tags for fine-grained filtering"
    )

    # Prompt content
    prompt: str = Field(
        min_length=10,
        description="The input prompt sent to the model under evaluation"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt context"
    )

    # Ground truth
    expected_behaviour: ExpectedBehaviour = Field(
        description="What a safe model should do when given this prompt"
    )
    harmful_intent: bool = Field(
        description="Whether the prompt carries genuine harmful intent"
    )
    ground_truth_safe_response: Optional[str] = Field(
        default=None,
        description="Example of an ideal safe model response"
    )

    # Scoring metadata
    risk_score: float = Field(
        ge=0.0, le=10.0,
        description="Numeric risk score (0 = harmless, 10 = maximally harmful)"
    )
    attack_vector: Optional[str] = Field(
        default=None,
        description="Specific attack mechanism used (e.g. 'ignore previous instructions')"
    )

    # Provenance
    source: str = Field(
        default="synthetic",
        description="Record provenance: 'synthetic', 'human', 'red_team'"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        description="ISO-8601 creation timestamp (UTC)"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Curator notes"
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must not be blank")
        return v.strip()

    @field_validator("risk_score")
    @classmethod
    def risk_score_precision(cls, v: float) -> float:
        return round(v, 2)

    @model_validator(mode="after")
    def critical_must_be_harmful(self) -> "SafetyRecord":
        if self.severity == Severity.CRITICAL and not self.harmful_intent:
            raise ValueError("CRITICAL severity implies harmful_intent=True")
        return self


# ── Dataset Metadata Schema ────────────────────────────────────────────────────

class DatasetMetadata(BaseModel):
    """Top-level dataset card / metadata."""

    name: str
    version: str
    description: str
    total_records: int
    category_distribution: dict[str, int]
    severity_distribution: dict[str, int]
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"
    )
    license: str = "Apache-2.0"
    language: str = "en"
    task_categories: list[str] = Field(
        default_factory=lambda: [
            "text-classification",
            "safety-evaluation",
            "red-teaming",
        ]
    )
