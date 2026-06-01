"""test_state.py — Tests for AgentState schema."""

import pytest
from src.state import AgentState


class TestAgentState:
    def test_defaults(self):
        s = AgentState(user_prompt="hello world test prompt")
        assert s.plan is None
        assert s.risk_flags == []
        assert s.requires_safety_check is False
        assert s.response_refused is False
        assert s.node_trace == []
        assert s.latency_ms == {}

    def test_model_default(self):
        s = AgentState(user_prompt="hello world test prompt")
        assert s.model_name == "phi4:latest"

    def test_node_trace_accumulates(self):
        s = AgentState(user_prompt="test", node_trace=["planner", "reasoning"])
        assert len(s.node_trace) == 2

    def test_risk_flags_list(self):
        s = AgentState(user_prompt="test prompt here", risk_flags=["injection", "jailbreak"])
        assert "injection" in s.risk_flags

    def test_latency_ms_dict(self):
        s = AgentState(user_prompt="test prompt here", latency_ms={"planner": 123.4})
        assert s.latency_ms["planner"] == 123.4

    def test_refused_state(self):
        s = AgentState(
            user_prompt="test prompt here",
            response_refused=True,
            refusal_reason="unsafe content",
        )
        assert s.response_refused is True
        assert s.refusal_reason == "unsafe content"
