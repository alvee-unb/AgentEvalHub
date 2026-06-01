"""test_prompts.py — Tests for prompt templates."""

from src.prompts.templates import (
    PLANNER_SYSTEM, PLANNER_USER,
    REASONING_SYSTEM, REASONING_USER,
    RESPONSE_SYSTEM, RESPONSE_USER,
)


class TestPromptTemplates:
    def test_planner_system_contains_format_instructions(self):
        assert "RISK_FLAGS" in PLANNER_SYSTEM
        assert "REQUIRES_SAFETY_CHECK" in PLANNER_SYSTEM
        assert "PLAN" in PLANNER_SYSTEM

    def test_planner_user_has_placeholder(self):
        assert "{prompt}" in PLANNER_USER
        formatted = PLANNER_USER.format(prompt="test input")
        assert "test input" in formatted

    def test_reasoning_system_contains_assessment_levels(self):
        assert "SAFE" in REASONING_SYSTEM
        assert "BORDERLINE" in REASONING_SYSTEM
        assert "UNSAFE" in REASONING_SYSTEM

    def test_reasoning_user_placeholders(self):
        formatted = REASONING_USER.format(
            prompt="test", plan="some plan", risk_flags="injection"
        )
        assert "test" in formatted
        assert "some plan" in formatted

    def test_response_system_has_refusal_rule(self):
        assert "UNSAFE" in RESPONSE_SYSTEM
        assert "refuse" in RESPONSE_SYSTEM.lower()

    def test_response_user_placeholders(self):
        formatted = RESPONSE_USER.format(
            prompt="test", safety_assessment="SAFE", unsafe_reason="N/A"
        )
        assert "SAFE" in formatted
